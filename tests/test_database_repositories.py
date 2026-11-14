"""Repository contract tests using session doubles, not a substitute SQL dialect."""

import unittest
from contextlib import contextmanager
from datetime import date
from unittest.mock import patch

from app.db.models import Order, Product, RefundRecord, SupportCase
from app.db.repositories import (
    create_support_case,
    get_order_by_id,
    get_product_by_id,
    get_refund_record,
    get_support_case,
    search_products,
)


class Result:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeSession:
    def __init__(self):
        self.rows = {}
        self.added = []
        self.query = None
        self.committed = False
        self.refreshed = False

    def get(self, model, key):
        return self.rows.get((model, key))

    def add(self, row):
        self.added.append(row)

    def commit(self):
        self.committed = True

    def refresh(self, row):
        self.refreshed = True
        if not getattr(row, "created_at", None):
            from datetime import datetime, timezone
            row.created_at = datetime.now(timezone.utc)

    def scalars(self, statement):
        self.query = statement
        return Result([Product(product_id="PROD-001", name="Laptop Pro", stock=8,
                               restock_date=date(2025, 11, 15))])


class DatabaseRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()

        @contextmanager
        def scope(*args, **kwargs):
            yield self.session

        self.patcher = patch("app.db.repositories.session_scope", scope)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_order_and_product_retrieval_preserve_service_data_shape(self):
        self.session.rows[(Order, "ORD-1001")] = Order(
            order_id="ORD-1001", status="Shipped", eta=date(2025, 11, 8), product_id="PROD-001"
        )
        self.session.rows[(Product, "PROD-001")] = Product(
            product_id="PROD-001", name="Laptop Pro", stock=8,
            restock_date=date(2025, 11, 15),
        )
        self.assertEqual(get_order_by_id("ord-1001"), {
            "order_id": "ORD-1001", "status": "Shipped", "eta": "2025-11-08",
            "product_id": "PROD-001",
        })
        self.assertEqual(get_product_by_id("prod-001")["restock_date"], "2025-11-15")

    def test_product_search_uses_bound_parameter_and_escapes_wildcards(self):
        result = search_products("Laptop%_")
        self.assertEqual(result[0]["product_id"], "PROD-001")
        compiled = self.session.query.compile()
        self.assertNotIn("Laptop%_", str(compiled))
        self.assertTrue(any("Laptop" in str(value) for value in compiled.params.values()))

    def test_create_and_retrieve_support_case_commits_and_returns_record_shape(self):
        result = create_support_case(reference="DEMO-CASE-1", record_type="damage",
                                    session_id="session-1", status="demo", summary="Demo",
                                    order_id="ORD-1001", details={"note": "sample"})
        row = self.session.added[0]
        self.session.rows[(SupportCase, "DEMO-CASE-1")] = row
        self.assertTrue(self.session.committed)
        self.assertTrue(self.session.refreshed)
        self.assertTrue(row.is_demo)
        self.assertEqual(result["details"], {"note": "sample"})
        self.assertEqual(get_support_case("DEMO-CASE-1")["record_type"], "damage")

    def test_refund_lookup_reads_missing_and_stored_status_records(self):
        self.assertIsNone(get_refund_record("ORD-1001"))
        self.session.rows[(RefundRecord, "ORD-1001")] = RefundRecord(
            order_id="ORD-1001", status="demo_pending", refund_id="R-DEMO-1", details={}
        )
        self.assertEqual(get_refund_record("ORD-1001")["status"], "demo_pending")

    def test_demo_refund_import_skips_missing_or_non_demo_parent_orders(self):
        from app.db.repositories import import_refund

        refund = {"order_id": "ORD-REAL-1", "status": "completed", "refund_id": "DEMO-REF-X"}
        self.assertFalse(import_refund(refund))
        real_order = Order(order_id="ORD-REAL-1", status="Shipped", is_demo=False)
        self.session.rows[(Order, "ORD-REAL-1")] = real_order
        self.assertFalse(import_refund(refund))
        self.assertFalse(any(isinstance(row, RefundRecord) for row in self.session.added))


if __name__ == "__main__":
    unittest.main()
