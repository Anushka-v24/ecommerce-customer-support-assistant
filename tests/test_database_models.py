"""PostgreSQL ORM metadata and fixture-import unit coverage (no server required)."""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

from sqlalchemy import CheckConstraint, ForeignKeyConstraint

from app.config import Config
from app.db.base import Base
from app.db.import_demo_data import import_fixtures
from app.db.models import Order, Product, RefundRecord, ReturnRequestDraft, SupportCase
from app.db.repositories import import_order, import_product
from app.services.data_repository import DEFAULT_DATA_DIR


class MemorySession:
    """Tiny session double for idempotency logic, not a SQL dialect substitute."""
    def __init__(self):
        self.rows = {}

    def get(self, model, key):
        return self.rows.get((model, key))

    def add(self, row):
        key_name = "product_id" if isinstance(row, Product) else "order_id"
        self.rows[(type(row), getattr(row, key_name))] = row

    def commit(self):
        pass

    def refresh(self, row):
        pass

    def close(self):
        pass


class DatabaseModelTests(unittest.TestCase):
    def test_models_register_exact_expected_business_tables(self):
        self.assertEqual(set(Base.metadata.tables), {
            "products", "orders", "support_cases", "return_request_drafts", "refund_records", "support_tickets",
        })

    def test_order_product_and_business_record_foreign_keys_exist(self):
        self.assertTrue(any(isinstance(item, ForeignKeyConstraint) for item in Order.__table__.constraints))
        self.assertTrue(any(isinstance(item, ForeignKeyConstraint) for item in SupportCase.__table__.constraints))
        self.assertTrue(any(isinstance(item, ForeignKeyConstraint) for item in ReturnRequestDraft.__table__.constraints))
        self.assertTrue(any(isinstance(item, ForeignKeyConstraint) for item in RefundRecord.__table__.constraints))

    def test_product_stock_has_nonnegative_database_constraint(self):
        constraints = [item for item in Product.__table__.constraints if isinstance(item, CheckConstraint)]
        self.assertTrue(any("stock >= 0" in str(item.sqltext) for item in constraints))

    def test_no_customer_or_order_item_tables_without_fixture_support(self):
        self.assertNotIn("customers", Base.metadata.tables)
        self.assertNotIn("order_items", Base.metadata.tables)

    def test_imported_fixture_records_are_demo_only_and_duplicate_safe(self):
        memory = MemorySession()

        @contextmanager
        def session_scope():
            yield memory

        with patch("app.db.repositories.session_scope", session_scope), \
             patch.object(Config, "DATABASE_URL", "postgresql+psycopg://unit:unit@localhost/unit"), \
             patch.object(Config, "BUSINESS_DATA_BACKEND", "postgres"):
            first = import_fixtures(DEFAULT_DATA_DIR)
            second = import_fixtures(DEFAULT_DATA_DIR)
        self.assertEqual(first, {"imported": 10, "skipped": 0, "failed": 0})
        self.assertEqual(second, {"imported": 0, "skipped": 10, "failed": 0})
        self.assertTrue(all(row.is_demo for (model, _), row in memory.rows.items()))

    def test_dry_run_validates_json_without_database_configuration(self):
        with patch.object(Config, "DATABASE_URL", None):
            counts = import_fixtures(DEFAULT_DATA_DIR, dry_run=True)
        self.assertEqual(counts, {"products": 3, "orders": 4, "refunds": 3, "failed": 0})

    def test_importer_requires_explicit_postgres_backend_and_url(self):
        with patch.object(Config, "DATABASE_URL", None):
            with self.assertRaisesRegex(RuntimeError, "Set DATABASE_URL"):
                import_fixtures(DEFAULT_DATA_DIR)


if __name__ == "__main__":
    unittest.main()
