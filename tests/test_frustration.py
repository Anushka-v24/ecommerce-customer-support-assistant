"""Offline tests for frustration scoring; the sentiment model is always mocked."""

import os
import unittest
from unittest.mock import MagicMock, patch

from app.services import frustration


NEUTRAL = {"negative": 0.05, "neutral": 0.85, "positive": 0.10}
ANGRY = {"negative": 0.92, "neutral": 0.06, "positive": 0.02}
MILD = {"negative": 0.55, "neutral": 0.40, "positive": 0.05}


class FrustrationTestCase(unittest.TestCase):
    def setUp(self):
        frustration.clear_sentiment_cache()
        self.env = patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("FRUSTRATION_THRESHOLD", None)
        os.environ.pop("SENTIMENT_ALLOW_DOWNLOAD", None)
        self.probs = patch.object(frustration, "sentiment_probabilities", return_value=NEUTRAL)
        self.mock_probs = self.probs.start()

    def tearDown(self):
        self.probs.stop()
        self.env.stop()
        frustration.clear_sentiment_cache()

    def use(self, probs):
        self.mock_probs.return_value = probs


class ScoreMessageTests(FrustrationTestCase):
    def test_calm_message(self):
        result = frustration.score_message("Can you tell me the delivery date for ORD-1001?")
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["signals"], [])
        self.assertLess(result["score"], 0.2)
        self.assertEqual(set(result), {"sentiment", "negative_prob", "signals", "score"})

    def test_angry_english(self):
        self.use(ANGRY)
        result = frustration.score_message(
            "This is the WORST SERVICE ever, I am fed up!!! I will go to consumer court."
        )
        self.assertEqual(result["sentiment"], "negative")
        self.assertAlmostEqual(result["negative_prob"], 0.92)
        for signal in ("escalation_phrase:worst service", "escalation_phrase:fed up",
                       "legal_threat:consumer court", "shouting", "repeated_punctuation"):
            self.assertIn(signal, result["signals"])
        self.assertGreater(result["score"], 0.9)

    def test_angry_hinglish(self):
        self.use(MILD)
        result = frustration.score_message(
            "Kya bakwas service hai, teesri baar bol raha hoon, pagal bana rahe ho kya??"
        )
        self.assertIn("escalation_phrase:bakwas", result["signals"])
        self.assertIn("escalation_phrase:pagal bana rahe", result["signals"])
        self.assertIn("repetition_word:teesri baar", result["signals"])
        self.assertIn("repeated_punctuation", result["signals"])
        self.assertGreater(result["score"], 0.65)

    def test_angry_hindi(self):
        self.use(ANGRY)
        result = frustration.score_message("मेरा ऑर्डर अभी तक नहीं आया, बहुत बेकार सर्विस है, मैं शिकायत करूंगा")
        self.assertIn("legal_threat:शिकायत करूंगा", result["signals"])
        self.assertIn("escalation_phrase:बेकार", result["signals"])
        self.assertIn("repetition_word:अभी तक", result["signals"])
        self.assertGreater(result["score"], 0.9)

    def test_word_boundaries_avoid_false_positives(self):
        result = frustration.score_message("Hello, I want to buy a stillness lamp. Is it available?")
        self.assertEqual(result["signals"], [])

    def test_order_ids_are_not_shouting(self):
        result = frustration.score_message("Status of ORD-1001 and ORD-1002 please")
        self.assertNotIn("shouting", result["signals"])

    def test_lexicon_fallback_without_model(self):
        self.use(None)
        angry = frustration.score_message("Worst service, I am fed up, talk to your manager!!")
        self.assertEqual(angry["sentiment"], "negative")
        self.assertGreater(angry["score"], 0.65)
        self.assertEqual(frustration.score_message("Thank you so much!")["sentiment"], "positive")
        self.assertEqual(frustration.score_message("Where is my parcel")["score"], 0.0)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        frustration.clear_sentiment_cache()

    def tearDown(self):
        frustration.clear_sentiment_cache()

    def test_missing_model_returns_none_and_is_cached(self):
        fake_transformers = MagicMock()
        fake_transformers.AutoTokenizer.from_pretrained.side_effect = OSError("not cached")
        real_import = frustration.importlib.import_module

        def fake_import(name):
            return fake_transformers if name == "transformers" else real_import(name)

        with patch.dict(os.environ, {"SENTIMENT_ALLOW_DOWNLOAD": "false"}), \
                patch.object(frustration.importlib, "import_module", side_effect=fake_import):
            self.assertIsNone(frustration.load_sentiment_model())
            self.assertIsNone(frustration.load_sentiment_model())
            self.assertIsNone(frustration.sentiment_probabilities("hello"))
            result = frustration.score_message("bakwas service!!")
        kwargs = fake_transformers.AutoTokenizer.from_pretrained.call_args.kwargs
        self.assertTrue(kwargs["local_files_only"])
        self.assertEqual(fake_transformers.AutoTokenizer.from_pretrained.call_count, 1)
        self.assertEqual(result["sentiment"], "negative")


class ConversationTests(FrustrationTestCase):
    def test_calm_conversation_does_not_escalate(self):
        result = frustration.conversation_frustration([
            {"text": "Hi, where is my order ORD-1001?", "intent": "order_tracking"},
            {"role": "assistant", "text": "It ships tomorrow."},
            {"text": "Can I also return a shirt?", "intent": "return_request"},
        ])
        self.assertFalse(result["escalate"])
        self.assertLess(result["score"], 0.3)
        self.assertEqual(set(result), {"score", "escalate", "reasons"})

    def test_empty_conversation(self):
        self.assertEqual(
            frustration.conversation_frustration([]),
            {"score": 0.0, "escalate": False, "reasons": []},
        )

    def test_repetition_escalates_mildly_annoyed_customer(self):
        turns = [
            {"text": "Where is my order ORD-1001?", "intent": "order_tracking"},
            {"text": "Still no update on ORD-1001", "intent": "order_tracking"},
            {"text": "Asking again about ORD-1001, kab tak?", "intent": "order_tracking"},
        ]
        self.use(MILD)
        result = frustration.conversation_frustration(turns)
        self.assertTrue(result["escalate"])
        self.assertTrue(any("order_tracking" in r for r in result["reasons"]))
        self.assertTrue(any("ORD-1001" in r for r in result["reasons"]))

        # Same messages without the repeat context stay below the threshold.
        single = frustration.conversation_frustration(turns[-1:])
        self.assertLess(single["score"], result["score"])
        self.assertFalse(single["escalate"])

    def test_resolved_order_is_not_a_repeat_signal(self):
        result = frustration.conversation_frustration([
            {"text": "Where is ORD-1001?"},
            {"text": "Thanks, ORD-1001 arrived", "resolved": True},
        ])
        self.assertFalse(any("ORD-1001" in r for r in result["reasons"]))

    def test_only_last_three_user_turns_count(self):
        self.use(ANGRY)
        old_angry = [{"text": "Worst service, fed up!!"}] * 3
        self.mock_probs.side_effect = lambda text: ANGRY if "Worst" in text else NEUTRAL
        result = frustration.conversation_frustration(
            old_angry + [{"text": "ok"}, {"text": "fine"}, {"text": "thanks"}]
        )
        self.assertFalse(result["escalate"])

    def test_angry_latest_turn_escalates_alone(self):
        self.mock_probs.side_effect = lambda text: ANGRY if "court" in text else NEUTRAL
        result = frustration.conversation_frustration([
            {"text": "hello"},
            {"text": "Worst service, I will go to consumer court!!"},
        ])
        self.assertTrue(result["escalate"])

    def test_threshold_from_env_and_argument(self):
        turns = [{"text": "This is ridiculous"}]
        self.use(MILD)
        score = frustration.conversation_frustration(turns)["score"]
        self.assertGreater(score, 0.5)
        self.assertLess(score, 0.65)
        self.assertFalse(frustration.conversation_frustration(turns)["escalate"])
        with patch.dict(os.environ, {"FRUSTRATION_THRESHOLD": "0.5"}):
            self.assertTrue(frustration.conversation_frustration(turns)["escalate"])
        with patch.dict(os.environ, {"FRUSTRATION_THRESHOLD": "not-a-number"}):
            self.assertEqual(frustration.frustration_threshold(), 0.65)
        self.assertTrue(frustration.conversation_frustration(turns, threshold=score)["escalate"])


class HandoffSummaryTests(FrustrationTestCase):
    def test_summary_contents(self):
        turns = [
            {"text": "Where is ORD-1001?", "intent": "order_tracking"},
            {"role": "assistant", "text": "Checking."},
            {"text": "Still   nothing, call me at 9876543210!!", "intent": "order_tracking"},
        ]
        score = {"score": 0.82, "escalate": True, "reasons": ["intent 'order_tracking' repeated"]}
        with patch.object(frustration, "_get_masker",
                          return_value=lambda s: s.replace("9876543210", "[PHONE]")):
            summary = frustration.handoff_summary(
                turns, ["order_tracking", "refund_request", "order_tracking"], ["ord-1001"], score
            )
        self.assertIn("Mood: frustrated (frustration 0.82, escalate: yes)", summary)
        self.assertIn("Intents seen: order_tracking, refund_request", summary)
        self.assertIn("Order IDs: ORD-1001", summary)
        self.assertIn("Customer messages: 2", summary)
        self.assertIn('Last customer message: "Still nothing, call me at [PHONE]!!"', summary)
        self.assertNotIn("9876543210", summary)
        self.assertIn("Why: intent 'order_tracking' repeated", summary)

    def test_summary_defaults_and_truncation(self):
        summary = frustration.handoff_summary([{"text": "x" * 500}], [], None, 0.1)
        self.assertIn("Mood: calm", summary)
        self.assertIn("Intents seen: none", summary)
        self.assertIn("Order IDs: none", summary)
        self.assertIn("...", summary)
        self.assertLess(len(summary), 600)

    def test_masker_falls_back_to_identity(self):
        with patch.dict("sys.modules", {"app.services.pii": None}):
            masker = frustration._get_masker()
        self.assertIs(masker, frustration._identity)
        self.assertEqual(masker("call 9876543210"), "call 9876543210")


if __name__ == "__main__":
    unittest.main()
