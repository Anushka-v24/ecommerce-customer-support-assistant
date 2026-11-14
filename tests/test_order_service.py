"""Tests for the read-only demonstration order service."""

import unittest
from datetime import date
from unittest.mock import patch

from app.config import Config
from app.services.order_service import get_order_by_id, get_order_status


patch.object(Config, "BUSINESS_DATA_BACKEND", "json").start()


class OrderServiceTests(unittest.TestCase):
    def test_existing_order_lookup_returns_only_required_fields(self):
        result = get_order_by_id("ORD-1001")
        self.assertEqual(result["outcome"], "found")
        self.assertEqual(result["order"], {
            "order_id": "ORD-1001",
            "status": "Shipped",
            "eta": "2025-11-08",
            "product_id": "PROD-001",
        })
        self.assertNotIn("user_email", result["order"])

    def test_order_status_lookup(self):
        result = get_order_status("ord-1002")
        self.assertEqual(result["outcome"], "found")
        self.assertEqual(result["order_status"], "Processing")
        self.assertIsNone(result["eta"])

    def test_unknown_order_id(self):
        result = get_order_status("ORD-9999")
        self.assertEqual(result["outcome"], "not_found")
        self.assertIsNone(result.get("order_status"))

    def test_missing_order_id(self):
        for order_id in (None, "", "   "):
            with self.subTest(order_id=order_id):
                self.assertEqual(get_order_status(order_id)["outcome"], "missing_order_id")

    def test_delayed_delivery_is_derived_from_past_eta_without_inventing_events(self):
        result = get_order_status("ORD-1001", as_of=date(2026, 1, 1))
        self.assertEqual(result["delivery_progress"]["stage"], "delayed")
        self.assertTrue(result["delivery_progress"]["is_delayed"])
        self.assertIn("no carrier scans", result["delivery_progress"]["explanation"])

    def test_delivered_status_is_preserved(self):
        result = get_order_status("ORD-1003", as_of=date(2026, 1, 1))
        self.assertEqual(result["delivery_progress"]["stage"], "delivered")
        self.assertFalse(result["delivery_progress"]["is_delayed"])
    def test_unknown_delivery_status_is_not_interpreted_as_a_tracking_event(self):
        from app.services.order_service import assess_delivery_progress
        result = assess_delivery_progress("Carrier update pending", None, as_of=date(2026, 1, 1))
        self.assertEqual(result["stage"], "unknown")
        self.assertFalse(result["is_delayed"])
        self.assertIn("does not contain a recognized", result["explanation"])
    def test_invalid_order_id_is_not_looked_up(self):
        self.assertEqual(get_order_status("order 1001")["outcome"], "invalid_order_id")


if __name__ == "__main__":
    unittest.main()


