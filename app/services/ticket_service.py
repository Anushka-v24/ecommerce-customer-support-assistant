"""Customer-facing support tickets raised at the end of a confirmed conversation.

Tickets live beside the demo support records: the local SQLite file for the JSON
backend, or the ``support_tickets`` table for the PostgreSQL backend. A ticket
is a tracking record for the customer; it does not contact a retailer or courier.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import Config
from app.services import case_repository
from app.services.pii import mask_obj, mask_pii


CATEGORY_SUBJECTS = {
    "damaged_order": "Damaged item",
    "missing_order": "Missing delivery",
    "return_order": "Return request",
    "customer_care": "Speak to a support agent",
    "delivery_speedup": "Delivery speed-up requested",
    "exchange_request": "Exchange request",
    "escalation": "Escalated: customer needs a person",
}
# Undelivered and damaged orders block the customer, so they are triaged first.
CATEGORY_PRIORITY = {
    "damaged_order": "high",
    "missing_order": "high",
    "return_order": "normal",
    "customer_care": "normal",
    "delivery_speedup": "high",
    "exchange_request": "normal",
    "escalation": "high",
}
PUBLIC_FIELDS = (
    "ticket_id", "category", "subject", "description", "order_id",
    "case_reference", "status", "priority", "created_at", "updated_at",
)


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS support_tickets (
            ticket_id TEXT PRIMARY KEY,
            customer_id TEXT,
            session_id TEXT NOT NULL,
            category TEXT NOT NULL,
            subject TEXT NOT NULL,
            description TEXT NOT NULL,
            order_id TEXT,
            case_reference TEXT,
            status TEXT NOT NULL,
            priority TEXT NOT NULL,
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )"""
    )


def public_ticket(ticket: dict[str, Any]) -> dict[str, Any]:
    """Return only the fields a customer should see."""
    return {field: ticket.get(field) for field in PUBLIC_FIELDS}


def create_ticket(
    *,
    session_id: str,
    category: str,
    description: str,
    customer_id: str | None = None,
    order_id: str | None = None,
    case_reference: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one open ticket and return it."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("A session ID is required to raise a ticket.")
    if category not in CATEGORY_SUBJECTS:
        raise ValueError(f"Unknown ticket category: {category}")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("A ticket needs a non-empty description.")

    subject = CATEGORY_SUBJECTS[category] + (f" - {order_id}" if order_id else "")
    fields = {
        "ticket_id": f"TCK-{uuid4().hex[:8].upper()}",
        "customer_id": customer_id,
        "session_id": session_id.strip(),
        "category": category,
        "subject": subject,
        "description": mask_pii(description.strip())[:2000],
        "order_id": order_id,
        "case_reference": case_reference,
        "status": "open",
        "priority": CATEGORY_PRIORITY[category],
        "details": mask_obj(details or {}),
    }
    if Config.BUSINESS_DATA_BACKEND == "postgres":
        from app.db.repositories import create_support_ticket

        return create_support_ticket(**fields)
    if Config.BUSINESS_DATA_BACKEND != "json":
        raise ValueError("BUSINESS_DATA_BACKEND must be 'json' or 'postgres'.")

    now = datetime.now(timezone.utc).isoformat()
    database_path = Path(case_repository.DEFAULT_DB_PATH)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=10)
    try:
        with case_repository._DB_LOCK, connection:
            _ensure_schema(connection)
            connection.execute(
                """INSERT INTO support_tickets
                   (ticket_id, customer_id, session_id, category, subject, description,
                    order_id, case_reference, status, priority, details_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fields["ticket_id"], customer_id, fields["session_id"], category, subject,
                    fields["description"], order_id, case_reference, "open", fields["priority"],
                    json.dumps(details or {}, ensure_ascii=False), now, now,
                ),
            )
    finally:
        connection.close()
    return {**fields, "created_at": now, "updated_at": now}


def list_tickets(customer_id: str | None = None, session_id: str | None = None) -> list[dict[str, Any]]:
    """List tickets for one customer or one conversation, newest first."""
    if not customer_id and not session_id:
        raise ValueError("A customer ID or session ID is required to list tickets.")
    if Config.BUSINESS_DATA_BACKEND == "postgres":
        from app.db.repositories import list_support_tickets

        return list_support_tickets(customer_id=customer_id, session_id=session_id)
    if Config.BUSINESS_DATA_BACKEND != "json":
        raise ValueError("BUSINESS_DATA_BACKEND must be 'json' or 'postgres'.")

    database_path = Path(case_repository.DEFAULT_DB_PATH)
    if not database_path.is_file():
        return []
    clauses, values = [], []
    if customer_id:
        clauses.append("customer_id = ?")
        values.append(customer_id)
    if session_id:
        clauses.append("session_id = ?")
        values.append(session_id)
    connection = sqlite3.connect(database_path, timeout=10)
    try:
        with case_repository._DB_LOCK:
            _ensure_schema(connection)
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM support_tickets WHERE " + " AND ".join(clauses)
                + " ORDER BY created_at DESC, ticket_id",
                values,
            ).fetchall()
    finally:
        connection.close()
    return [
        {
            "ticket_id": row["ticket_id"], "customer_id": row["customer_id"],
            "session_id": row["session_id"], "category": row["category"],
            "subject": row["subject"], "description": row["description"],
            "order_id": row["order_id"], "case_reference": row["case_reference"],
            "status": row["status"], "priority": row["priority"],
            "details": json.loads(row["details_json"]),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }
        for row in rows
    ]
