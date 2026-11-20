"""Tests for the read-only demonstration product service."""

import unittest
from unittest.mock import patch

from app.config import Config
from app.services.product_service import get_product_by_id, get_product_stock, search_products


patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()
# Unit tests stay offline; the Groq extraction fallback has its own mocked tests.
patch.object(Config, "LLM_EXTRACTION_ENABLED", False).start()


class ProductServiceTests(unittest.TestCase):
    def test_existing_product_lookup_by_canonical_id(self):
        result = get_product_by_id("prod-001")
        self.assertEqual(result["outcome"], "found")
        self.assertEqual(result["product"]["name"], "Laptop Pro")
        self.assertEqual(result["product"]["product_id"], "PROD-001")

    def test_product_stock_lookup(self):
        result = get_product_stock("PROD-001")
        self.assertEqual(result["outcome"], "found")
        self.assertEqual(result["stock"], 8)
        self.assertEqual(result["restock_date"], "2025-11-15")

    def test_zero_stock_is_a_supported_value(self):
        result = get_product_stock("PROD-002")
        self.assertEqual(result["outcome"], "found")
        self.assertEqual(result["stock"], 0)

    def test_unknown_product(self):
        self.assertEqual(get_product_by_id("PROD-999")["outcome"], "not_found")
        self.assertEqual(search_products("No Such Product")["outcome"], "not_found")

    def test_search_by_name_and_natural_query(self):
        result = search_products("Do you have Laptop Pro in stock?")
        self.assertEqual(result["outcome"], "found")
        self.assertEqual(result["matches"][0]["product_id"], "PROD-001")

    def test_ambiguous_query_returns_candidates_without_selecting_one(self):
        result = search_products("Laptop Pro or Galaxy S21")
        self.assertEqual(result["outcome"], "ambiguous")
        self.assertEqual(
            {item["product_id"] for item in result["matches"]},
            {"PROD-001", "PROD-002"},
        )
        self.assertTrue(all("stock" not in item for item in result["matches"]))

    def test_missing_product_query(self):
        self.assertEqual(search_products(None)["outcome"], "missing_query")

    def test_stock_is_not_invented_when_record_does_not_supply_it(self):
        product = {"PROD-010": {"product_id": "PROD-010", "name": "Test Product", "stock": None}}
        with patch("app.services.product_service.load_products", return_value=product):
            result = get_product_stock("PROD-010")
        self.assertEqual(result["outcome"], "stock_unavailable")
        self.assertNotIn("stock", result)


if __name__ == "__main__":
    unittest.main()
