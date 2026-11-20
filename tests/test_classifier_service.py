"""Unit tests for the XLM-R adapter; model weights and Groq are mocked."""

import contextlib
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.classifier import classifier_service as service
from app.classifier import model_loader


LABELS = [
    "account_management",
    "add_to_cart",
    "billing_and_payment",
    "customer_care_support",
    "delivery_options",
    "issue_support",
    "order_management",
    "order_tracking",
    "out_of_scope",
    "place_order",
    "product_inquiry",
    "remove_from_cart",
    "shipping_cost",
    "store_information",
    "submit_feedback",
]


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeProbabilities:
    def __init__(self, index: int, confidence: float):
        self.index = index
        self.confidence = confidence

    def max(self, dim: int):
        return FakeScalar(self.confidence), FakeScalar(self.index)


class FakeTensor:
    def to(self, device: str):
        return self


class FakeTokenizer:
    def __init__(self):
        self.queries = []

    def __call__(self, query: str, **kwargs):
        self.queries.append(query)
        return {"input_ids": FakeTensor()}


class FakeTorch:
    cuda = SimpleNamespace(is_available=lambda: False)
    inference_mode = staticmethod(contextlib.nullcontext)
    softmax = staticmethod(lambda logits, dim: logits)


class FakeModel:
    def __init__(self, index: int, confidence: float):
        self.index = index
        self.confidence = confidence

    def __call__(self, **inputs):
        return SimpleNamespace(
            logits=FakeProbabilities(self.index, self.confidence)
        )

# Added in the 20-label model; ids 15-19 follow the original 15 in this order.
NEW_LABELS = ["exchange_request", "goodbye_thanks", "greeting", "product_recommendation", "warranty_inquiry"]


def make_bundle(index: int, confidence: float):
    return model_loader.ModelBundle(
        tokenizer=FakeTokenizer(),
        model=FakeModel(index, confidence),
        id2label=dict(enumerate(LABELS)),
        device="cpu",
        torch=FakeTorch(),
    )


def verifier_result(status: str, predicted: str, verified: str | None):
    return {
        "status": status,
        "predicted_intent": predicted,
        "verified_intent": verified,
        "reason": "Mocked verification for a unit test.",
    }


class ModelLoaderTests(unittest.TestCase):
    def setUp(self):
        model_loader.clear_model_cache()

    def tearDown(self):
        model_loader.clear_model_cache()

    def test_loads_tokenizer_and_model_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model"
            model_path.mkdir()
            for name in (
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ):
                (model_path / name).write_text("test artifact", encoding="utf-8")
            mapping_path = Path(temp_dir) / "label_mapping.json"
            mapping_path.write_text(
                json.dumps({"id2label": {str(i): label for i, label in enumerate(LABELS)}}),
                encoding="utf-8",
            )

            tokenizer = MagicMock()
            model = MagicMock()
            model.to.return_value = model
            transformers = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer)),
                AutoModelForSequenceClassification=SimpleNamespace(
                    from_pretrained=MagicMock(return_value=model)
                ),
            )
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

            with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
                first = model_loader.load_model(model_path, mapping_path)
                second = model_loader.load_model(model_path, mapping_path)

            self.assertIs(first, second)
            self.assertEqual(len(first.id2label), 15)
            transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
                str(model_path.resolve()), local_files_only=True
            )
            transformers.AutoModelForSequenceClassification.from_pretrained.assert_called_once_with(
                str(model_path.resolve()), local_files_only=True
            )
            model.eval.assert_called_once()

    def test_default_model_path_is_inside_this_project(self):
        with patch.dict(os.environ, {"XLMR_PROJECT_PATH": r"Z:\missing-original-classifier"}, clear=True):
            expected = model_loader.PROJECT_ROOT / "models" / "xlmr_intent_classifier"
            self.assertEqual(model_loader.resolve_model_path(), expected.resolve())

    def test_default_intent_descriptions_are_bundled(self):
        with patch.dict(
            os.environ,
            {"XLMR_INTENT_DESCRIPTIONS_PATH": "app/classifier/intent_descriptions.json"},
            clear=True,
        ):
            self.assertEqual(set(service._load_descriptions()), set(LABELS) | set(NEW_LABELS))

    def test_issue_support_description_covers_resolver_subtypes(self):
        # Groq picks labels from these descriptions, so issue_support must name
        # every problem its resolver handles or low-confidence queries drift away.
        with patch.dict(
            os.environ,
            {"XLMR_INTENT_DESCRIPTIONS_PATH": "app/classifier/intent_descriptions.json"},
            clear=True,
        ):
            description = service._load_descriptions()["issue_support"].lower()
        for phrase in ("refund status", "return an item", "damaged", "has not arrived", "dissatisfaction"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, description)

    def test_missing_model_files_report_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            mapping_path = model_path / "label_mapping.json"
            mapping_path.write_text(
                json.dumps({"id2label": {str(i): label for i, label in enumerate(LABELS)}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "model.safetensors.*tokenizer.json"):
                model_loader.load_model(model_path, mapping_path)

    def test_bundled_label_map_preserves_original_fifteen_labels(self):
        labels = model_loader.load_label_mapping(model_loader.resolve_mapping_path())
        self.assertEqual([labels[index] for index in range(15)], LABELS)


class ClassifierServiceTests(unittest.TestCase):
    def test_default_confidence_threshold_remains_95_percent(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(service._confidence_threshold(), 95.0)

    def test_service_import_does_not_require_groq(self):
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": ""}),
            patch.dict(sys.modules, {"groq": None}),
        ):
            reloaded_service = importlib.reload(service)
            self.assertTrue(callable(reloaded_service.classify_and_verify))

    def test_english_prediction_uses_xlmr(self):
        query = "Where is my order?"
        bundle = make_bundle(7, 0.97)
        with patch.object(service, "load_model", return_value=bundle):
            result = service.classify_and_verify(query)

        self.assertEqual(result["predicted_intent"], "order_tracking")
        self.assertEqual(result["final_intent"], "order_tracking")
        self.assertEqual(bundle.tokenizer.queries, [query])

    def test_hindi_prediction_uses_xlmr(self):
        query = "मेरा ऑर्डर कहाँ पहुँचा है?"
        bundle = make_bundle(7, 0.97)
        with patch.object(service, "load_model", return_value=bundle):
            result = service.classify_and_verify(query)

        self.assertEqual(result["predicted_intent"], "order_tracking")
        self.assertEqual(bundle.tokenizer.queries, [query])

    def test_high_confidence_bypasses_groq(self):
        bundle = make_bundle(7, 0.9501)
        with (
            patch.object(service, "load_model", return_value=bundle),
            patch.object(service, "_verify_with_groq") as verify,
        ):
            result = service.classify_and_verify("track my parcel")

        self.assertEqual(result["routing"], "DIRECT_CLASSIFIER_OUTPUT")
        self.assertEqual(result["verifier_status"], "ACCEPTED_BY_CLASSIFIER")
        self.assertEqual(result["final_intent"], "order_tracking")
        verify.assert_not_called()

    def test_low_confidence_calls_groq_at_threshold(self):
        bundle = make_bundle(7, 0.95)
        with (
            patch.object(service, "load_model", return_value=bundle),
            patch.object(
                service,
                "_verify_with_groq",
                return_value=verifier_result("CORRECT_INTENT", "order_tracking", "order_tracking"),
            ) as verify,
        ):
            result = service.classify_and_verify("track my parcel")

        self.assertEqual(result["routing"], "LLM_VERIFICATION")
        self.assertEqual(result["verifier_status"], "CORRECT_INTENT")
        verify.assert_called_once()

    def test_previous_turns_reach_groq_trimmed_to_the_last_four(self):
        history = [{"role": "user", "content": f"turn {n}"} for n in range(6)]
        history.append({"role": "assistant", "content": "x" * 500})
        completion = MagicMock()
        completion.choices[0].message.content = json.dumps(
            verifier_result("CORRECT_INTENT", "order_tracking", "order_tracking")
        )
        client = MagicMock()
        client.chat.completions.create.return_value = completion
        with (
            patch.object(service, "load_model", return_value=make_bundle(7, 0.80)),
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key", "XLMR_INTENT_DESCRIPTIONS_PATH": "app/classifier/intent_descriptions.json"}),
            patch("groq.Groq", return_value=client),
        ):
            result = service.classify_and_verify("and that one?", history=history)

        self.assertEqual(result["verifier_status"], "CORRECT_INTENT")
        sent = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        payload = json.loads(sent.split("\n", 1)[1])
        self.assertEqual([t["content"] for t in payload["previous_turns"][:3]], ["turn 3", "turn 4", "turn 5"])
        self.assertEqual(len(payload["previous_turns"][3]["content"]), 300)

    def test_corrected_intent_is_used(self):
        bundle = make_bundle(7, 0.80)
        with (
            patch.object(service, "load_model", return_value=bundle),
            patch.object(
                service,
                "_verify_with_groq",
                return_value=verifier_result("CORRECTED_INTENT", "order_tracking", "issue_support"),
            ),
        ):
            result = service.classify_and_verify("I was charged twice")

        self.assertEqual(result["final_intent"], "issue_support")
        self.assertEqual(result["verifier_status"], "CORRECTED_INTENT")

    def test_unknown_ecommerce_intent_has_no_final_label(self):
        bundle = make_bundle(7, 0.80)
        with (
            patch.object(service, "load_model", return_value=bundle),
            patch.object(
                service,
                "_verify_with_groq",
                return_value=verifier_result("UNKNOWN_ECOMMERCE_INTENT", "order_tracking", None),
            ),
        ):
            result = service.classify_and_verify("Can you change my delivery address?")

        self.assertIsNone(result["final_intent"])
        self.assertEqual(result["verifier_status"], "UNKNOWN_ECOMMERCE_INTENT")

    def test_out_of_scope_intent_has_no_final_label(self):
        bundle = make_bundle(7, 0.80)
        with (
            patch.object(service, "load_model", return_value=bundle),
            patch.object(
                service,
                "_verify_with_groq",
                return_value=verifier_result("OUT_OF_SCOPE", "order_tracking", None),
            ),
        ):
            result = service.classify_and_verify("Who won the match yesterday?")

        self.assertIsNone(result["final_intent"])
        self.assertEqual(result["verifier_status"], "OUT_OF_SCOPE")

    def test_groq_failure_fails_closed_below_fallback_confidence(self):
        bundle = make_bundle(7, 0.70)
        with (
            patch.object(service, "load_model", return_value=bundle),
            patch.object(service, "_verify_with_groq", side_effect=RuntimeError("network unavailable")),
        ):
            result = service.classify_and_verify("track my parcel")

        self.assertEqual(result["verifier_status"], "UNVERIFIED")
        self.assertIsNone(result["final_intent"])
        self.assertIn("network unavailable", result["error"])

    def test_groq_configuration_is_checked_only_when_needed(self):
        bundle = make_bundle(7, 0.70)
        with (
            patch.object(service, "load_model", return_value=bundle),
            patch.dict("os.environ", {"GROQ_API_KEY": ""}),
        ):
            result = service.classify_and_verify("track my parcel")

        self.assertEqual(result["verifier_status"], "UNVERIFIED")
        self.assertIsNone(result["final_intent"])
        self.assertIn("GROQ_API_KEY is not configured", result["error"])

    def test_verifier_describes_only_likely_labels_but_allows_all(self):
        completion = MagicMock()
        completion.choices[0].message.content = json.dumps(
            verifier_result("CORRECT_INTENT", "order_tracking", "order_tracking")
        )
        client = MagicMock()
        client.chat.completions.create.return_value = completion
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key", "XLMR_INTENT_DESCRIPTIONS_PATH": "app/classifier/intent_descriptions.json"}),
            patch("groq.Groq", return_value=client),
        ):
            service._verify_with_groq("where is it", "order_tracking", 70.0, LABELS, candidates=["issue_support", "delivery_options"])

        kwargs = client.chat.completions.create.call_args.kwargs
        payload = json.loads(kwargs["messages"][1]["content"].split("\n", 1)[1])
        self.assertEqual(set(payload["intent_descriptions"]), {"order_tracking", "issue_support", "delivery_options"})
        self.assertEqual(payload["known_intent_labels"], LABELS)
        schema_labels = kwargs["response_format"]["json_schema"]["schema"]["properties"]["predicted_intent"]["enum"]
        self.assertEqual(schema_labels, LABELS)

    def test_groq_failure_accepts_confident_prediction(self):
        bundle = make_bundle(7, 0.85)
        with (
            patch.object(service, "load_model", return_value=bundle),
            patch.object(service, "_verify_with_groq", side_effect=RuntimeError("rate limit")),
        ):
            result = service.classify_and_verify("track my parcel")

        self.assertEqual(result["verifier_status"], "ACCEPTED_BY_CLASSIFIER")
        self.assertEqual(result["routing"], "CLASSIFIER_FALLBACK_VERIFIER_UNAVAILABLE")
        self.assertEqual(result["final_intent"], "order_tracking")
        self.assertIn("rate limit", result["error"])

    def test_fallback_can_be_disabled(self):
        bundle = make_bundle(7, 0.90)
        with (
            patch.object(service, "load_model", return_value=bundle),
            patch.object(service, "_verify_with_groq", side_effect=RuntimeError("rate limit")),
            patch.object(service.Config, "VERIFIER_FALLBACK_MIN_CONFIDENCE", 101.0),
        ):
            result = service.classify_and_verify("track my parcel")

        self.assertEqual(result["verifier_status"], "UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
