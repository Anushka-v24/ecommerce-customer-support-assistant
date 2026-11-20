"""Structured extraction fallback: validation, fail-closed behaviour, and wiring."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.config import Config
from app.services import llm_extraction
from app.services.issue_resolver import resolve_issue_type
from app.services.product_service import search_products


patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()


def groq_returning(payload):
    """Patch the Groq client so every completion returns ``payload`` as JSON."""
    client = MagicMock()
    message = SimpleNamespace(content=json.dumps(payload) if payload is not None else None)
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    return patch("groq.Groq", return_value=client)


class ExtractionTestCase(unittest.TestCase):
    def setUp(self):
        llm_extraction.extract_issue_details.cache_clear()
        llm_extraction.match_catalog_product.cache_clear()
        patcher = patch.dict(os.environ, {"GROQ_API_KEY": "test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)
        enabled = patch.object(Config, "LLM_EXTRACTION_ENABLED", True)
        enabled.start()
        self.addCleanup(enabled.stop)


class IssueExtractionTests(ExtractionTestCase):
    def test_valid_issue_type_and_present_order_id_are_returned(self):
        with groq_returning({"issue_type": "refund_status", "order_id": "ord-1002"}):
            result = llm_extraction.extract_issue_details("status of my refund for ORD-1002")
        self.assertEqual(result, {"issue_type": "refund_status", "order_id": "ORD-1002"})

    def test_order_id_not_in_message_is_discarded(self):
        with groq_returning({"issue_type": "missing_order", "order_id": "ORD-1001"}):
            result = llm_extraction.extract_issue_details("parcel abhi tak nahi aaya")
        self.assertEqual(result, {"issue_type": "missing_order", "order_id": None})

    def test_unknown_issue_type_becomes_none(self):
        with groq_returning({"issue_type": "wrong_item", "order_id": None}):
            result = llm_extraction.extract_issue_details("I got the wrong item")
        self.assertIsNone(result["issue_type"])

    def test_provider_failure_fails_closed(self):
        with patch("groq.Groq", side_effect=RuntimeError("network down")):
            self.assertIsNone(llm_extraction.extract_issue_details("mera refund kab aayega"))

    def test_disabled_or_missing_key_makes_no_call(self):
        with patch.object(Config, "LLM_EXTRACTION_ENABLED", False), patch("groq.Groq") as groq:
            self.assertIsNone(llm_extraction.extract_issue_details("mera refund kab aayega"))
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}), patch("groq.Groq") as groq_no_key:
            self.assertIsNone(llm_extraction.extract_issue_details("mera refund kab aayega"))
        groq.assert_not_called()
        groq_no_key.assert_not_called()


class IssueResolverFallbackTests(ExtractionTestCase):
    def test_rules_win_without_calling_the_llm(self):
        with patch("groq.Groq") as groq:
            result = resolve_issue_type("my order arrived damaged")
        self.assertEqual((result["issue_type"], result["source"]), ("damaged_order", "phrase_rules"))
        groq.assert_not_called()

    def test_hindi_message_resolves_through_extraction(self):
        with groq_returning({"issue_type": "refund_status", "order_id": None}):
            result = resolve_issue_type("मेरा रिफंड कहाँ है?")
        self.assertEqual((result["outcome"], result["issue_type"], result["source"]), ("resolved", "refund_status", "llm_extraction"))

    def test_null_extraction_keeps_clarification(self):
        with groq_returning({"issue_type": None, "order_id": None}):
            result = resolve_issue_type("i do not like my order")
        self.assertEqual(result["outcome"], "clarification_required")


class ProductMatchingTests(ExtractionTestCase):
    def test_plural_and_typo_match_without_llm(self):
        with patch("groq.Groq") as groq:
            self.assertEqual(search_products("do you sell laptops?")["matches"][0]["name"], "Laptop Pro")
            self.assertEqual(search_products("is the wireless headphone in stock?")["matches"][0]["name"], "Noise-Cancelling Headphones")
            self.assertEqual(search_products("headphnes available?")["matches"][0]["name"], "Noise-Cancelling Headphones")
        groq.assert_not_called()

    def test_category_word_uses_llm_catalogue_match(self):
        with groq_returning({"product_name": "Galaxy S21"}):
            result = search_products("koi accha phone hai kya?")
        self.assertEqual((result["outcome"], result["matches"][0]["name"]), ("found", "Galaxy S21"))

    def test_llm_name_outside_catalogue_is_rejected(self):
        with groq_returning({"product_name": "iPhone 15"}):
            self.assertEqual(search_products("do you have an iphone?")["outcome"], "not_found")


if __name__ == "__main__":
    unittest.main()
