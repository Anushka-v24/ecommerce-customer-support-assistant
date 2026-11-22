"""The fictional demo fixtures are large enough to answer from, and consistent with each other."""

import json
import re
import unittest

from app.services.data_repository import DEFAULT_DATA_DIR
from app.services.order_service import ORDER_ID_PATTERN
from app.services.product_service import PRODUCT_ID_PATTERN, search_products
from pathlib import Path


def records(name):
    return json.loads((Path(DEFAULT_DATA_DIR) / f"{name}.json").read_text(encoding="utf-8-sig"))[name]


class FixtureSizeAndConsistencyTests(unittest.TestCase):
    def test_every_table_has_at_least_100_rows(self):
        for name in ("products", "orders", "refunds", "customers"):
            self.assertGreaterEqual(len(records(name)), 100, name)

    def test_identifiers_are_unique_and_well_formed(self):
        products, orders = records("products"), records("orders")
        self.assertEqual(len({item["product_id"] for item in products}), len(products))
        self.assertEqual(len({item["order_id"] for item in orders}), len(orders))
        self.assertTrue(all(PRODUCT_ID_PATTERN.fullmatch(item["product_id"]) for item in products))
        self.assertTrue(all(ORDER_ID_PATTERN.fullmatch(item["order_id"]) for item in orders))
        self.assertEqual(len({item["name"] for item in products}), len(products))

    def test_references_point_at_real_rows(self):
        product_ids = {item["product_id"] for item in records("products")}
        order_ids = {item["order_id"] for item in records("orders")}
        self.assertTrue(all(item["product_id"] in product_ids for item in records("orders")))
        refunds = records("refunds")
        self.assertTrue(all(item["order_id"] in order_ids for item in refunds))
        self.assertEqual(len({item["order_id"] for item in refunds}), len(refunds))
        self.assertEqual(len({item["refund_id"] for item in refunds}), len(refunds))

    def test_every_order_belongs_to_exactly_one_customer(self):
        owned = [order_id for customer in records("customers") for order_id in customer["order_ids"]]
        self.assertEqual(sorted(owned), sorted(item["order_id"] for item in records("orders")))

    def test_values_use_the_vocabulary_the_app_understands(self):
        self.assertTrue({item["status"] for item in records("orders")} <= {"Delivered", "Shipped", "Processing", "Delayed"})
        self.assertTrue({item["status"] for item in records("refunds")} <= {"not_initiated", "processing", "completed"})
        for item in records("products"):
            self.assertGreaterEqual(item["stock"], 0)
            if item["stock"] == 0:
                self.assertRegex(item["restock_date"] or "", r"^\d{4}-\d{2}-\d{2}$")
        for item in records("orders"):
            if item["eta"]:
                self.assertRegex(item["eta"], r"^\d{4}-\d{2}-\d{2}$")

    def test_the_mix_covers_the_cases_the_chat_handles(self):
        statuses = {item["status"] for item in records("orders")}
        self.assertEqual(statuses, {"Delivered", "Shipped", "Processing", "Delayed"})
        self.assertTrue(any(item["stock"] == 0 for item in records("products")))
        refund_orders = {item["order_id"] for item in records("refunds")}
        self.assertTrue(any(item["order_id"] not in refund_orders for item in records("orders")))
        self.assertEqual({item["status"] for item in records("refunds")}, {"not_initiated", "processing", "completed"})

    def test_the_original_records_are_unchanged(self):
        self.assertEqual(records("products")[0], {"product_id": "PROD-001", "name": "Laptop Pro", "stock": 8, "restock_date": "2025-11-15"})
        self.assertEqual(records("orders")[0]["order_id"], "ORD-1001")
        self.assertEqual(records("customers")[0]["name"], "Aarav Sharma")


class LargeCatalogSearchTests(unittest.TestCase):
    def test_a_full_product_name_resolves_to_that_product_only(self):
        for item in records("products"):
            result = search_products(f"is {item['name']} in stock?")
            self.assertEqual(result["outcome"], "found", item["name"])
            self.assertEqual(result["matches"][0]["product_id"], item["product_id"], item["name"])

    def test_a_product_id_resolves_exactly(self):
        self.assertEqual(search_products("PROD-050")["matches"][0]["product_id"], "PROD-050")

    def test_a_category_word_is_ambiguous_and_the_list_is_short(self):
        result = search_products("speaker")
        self.assertEqual(result["outcome"], "ambiguous")
        self.assertLessEqual(len(result["matches"]), 8)
        broad = search_products("Plus")
        self.assertEqual(broad["outcome"], "ambiguous")
        self.assertLessEqual(len(broad["matches"]), 8)

    def test_original_names_stay_unambiguous(self):
        self.assertEqual(search_products("Laptop Pro")["matches"][0]["product_id"], "PROD-001")
        self.assertEqual(search_products("galaxy s21")["matches"][0]["product_id"], "PROD-002")


if __name__ == "__main__":
    unittest.main()
