import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.config import Config
from app.services.reply_translation import translate_reply


def groq_returning(payload):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))]
    )
    return MagicMock(return_value=client), client


class ReplyTranslationTests(unittest.TestCase):
    def setUp(self):
        for name, value in {"REPLY_TRANSLATION_ENABLED": True}.items():
            patcher = patch.object(Config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)

    def run_with(self, payload, query, response, replies=None):
        factory, client = groq_returning(payload)
        with patch.dict("sys.modules", {"groq": SimpleNamespace(Groq=factory)}):
            return translate_reply(query, response, replies), client

    def test_english_query_is_not_translated(self):
        result, client = self.run_with({"reply": "x", "button_labels": []}, "where is my order", "Please share your order ID.")
        self.assertIsNone(result)
        client.chat.completions.create.assert_not_called()

    def test_hinglish_query_gets_hinglish_reply_and_labels_keep_values(self):
        replies = [{"label": "Yes, cancel it", "value": "yes"}, {"label": "No, keep it", "value": "no"}]
        result, client = self.run_with(
            {"reply": "Kya aap ORD-1002 cancel karna chahte hain?", "button_labels": ["Haan, cancel karo", "Nahi, rehne do"]},
            "mera order cancel karna hai", "Do you want to cancel ORD-1002?", replies,
        )
        self.assertEqual(result["lang"], "hinglish")
        self.assertIn("ORD-1002", result["response"])
        self.assertEqual([r["value"] for r in result["quick_replies"]], ["yes", "no"])
        self.assertEqual(result["quick_replies"][0]["label"], "Haan, cancel karo")

    def test_hindi_reply_already_in_devanagari_is_left_alone(self):
        result, client = self.run_with({"reply": "x", "button_labels": []}, "मेरा ऑर्डर कहाँ है", "आपका ऑर्डर ORD-1001 भेज दिया गया है।")
        self.assertEqual(result["lang"], "hi")
        self.assertEqual(result["response"], "आपका ऑर्डर ORD-1001 भेज दिया गया है।")
        client.chat.completions.create.assert_not_called()

    def test_prompt_is_pii_masked(self):
        _, client = self.run_with({"reply": "ठीक है", "button_labels": []}, "मुझे कॉल करें", "We will call you on 9876543210.")
        sent = json.dumps(client.chat.completions.create.call_args.kwargs["messages"], ensure_ascii=False)
        self.assertNotIn("9876543210", sent)
        self.assertIn("[PHONE]", sent)

    def test_provider_failure_falls_back_to_english(self):
        factory = MagicMock(side_effect=RuntimeError("network down"))
        with patch.dict("sys.modules", {"groq": SimpleNamespace(Groq=factory)}):
            self.assertIsNone(translate_reply("order kab aayega", "It will arrive tomorrow."))

    def test_disabled_flag_skips_translation(self):
        with patch.object(Config, "REPLY_TRANSLATION_ENABLED", False):
            result, client = self.run_with({"reply": "x", "button_labels": []}, "order kab aayega", "Tomorrow.")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
