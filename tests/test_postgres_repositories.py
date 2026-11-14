"""Opt-in repository integration coverage against a real migrated PostgreSQL DB."""

import unittest
from uuid import uuid4

from sqlalchemy import delete

from app.config import Config
from app.db.models import Order, Product, ReturnRequestDraft, SupportCase
from app.db.repositories import (
    create_return_request_draft,
    create_support_case,
    get_order_by_id,
    get_product_by_id,
    get_refund_record,
    get_support_case,
    search_products,
)
from app.db.session import dispose_engines, get_engine, session_scope


DATABASE_URL = (Config.DATABASE_URL or "").strip()


@unittest.skipUnless(DATABASE_URL, "Set DATABASE_URL to run PostgreSQL repository integration tests.")
class PostgreSQLRepositoryTests(unittest.TestCase):
    def test_repository_reads_and_writes_against_migrated_tables(self):
        engine = get_engine(DATABASE_URL)
        suffix = uuid4().hex[:8].upper()
        product_id = f"PROD-T{suffix}"
        order_id = f"ORD-T{suffix}"
        case_ref = f"TEST-CASE-{suffix}"
        return_ref = f"TEST-RETURN-{suffix}"
        try:
            with session_scope(engine) as session:
                session.add(Product(product_id=product_id, name=f"Test Widget {suffix}", stock=4, is_demo=True))
                session.commit()
            with session_scope(engine) as session:
                session.add(Order(order_id=order_id, status="Processing", product_id=product_id, is_demo=True))
                session.commit()

            self.assertEqual(get_product_by_id(product_id)["product_id"], product_id)
            self.assertEqual(get_order_by_id(order_id)["product_id"], product_id)
            self.assertEqual(search_products(f"Widget {suffix}")[0]["product_id"], product_id)
            self.assertIsNone(get_refund_record(order_id))

            case = create_support_case(reference=case_ref, record_type="test_demo", session_id="test-session",
                                       status="demo", summary="Test support record", order_id=order_id)
            self.assertEqual(get_support_case(case_ref)["reference"], case_ref)
            draft = create_return_request_draft(reference=return_ref, session_id="test-session",
                                                status="demo_pending", summary="Test return draft",
                                                order_id=order_id, reason="test reason")
            self.assertEqual(draft["record_type"], "return_request")
        finally:
            with session_scope(engine) as session:
                session.execute(delete(SupportCase).where(SupportCase.reference == case_ref))
                session.execute(delete(ReturnRequestDraft).where(ReturnRequestDraft.reference == return_ref))
                session.execute(delete(Order).where(Order.order_id == order_id))
                session.execute(delete(Product).where(Product.product_id == product_id))
                session.commit()
            dispose_engines(engine)


if __name__ == "__main__":
    unittest.main()
