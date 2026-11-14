"""Small local SQLite store for explicitly simulated support records."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "runtime" / "support_cases.sqlite3"
_DB_LOCK = threading.Lock()


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS demo_support_records (
            reference TEXT PRIMARY KEY,
            record_type TEXT NOT NULL,
            session_id TEXT NOT NULL,
            order_id TEXT,
            reason TEXT,
            summary TEXT NOT NULL,
            status TEXT NOT NULL,
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )


def create_demo_record(
    record_type: str,
    session_id: str,
    status: str,
    summary: str,
    order_id: str | None = None,
    reason: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one local simulation record and return its explicit demo reference."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("A session ID is required for a demonstration support record.")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("A non-empty summary is required.")
    if not isinstance(record_type, str) or not record_type.strip():
        raise ValueError("A record type is required.")

    reference = f"DEMO-{record_type.upper().replace('_', '-')}-{uuid4().hex[:10].upper()}"
    created_at = datetime.now(timezone.utc).isoformat()
    if Config.BUSINESS_DATA_BACKEND == "postgres":
        from app.db.repositories import create_return_request_draft, create_support_case
        common = dict(reference=reference, session_id=session_id.strip(), status=status,
                      summary=summary.strip(), order_id=order_id, reason=reason,
                      details=details or {})
        if record_type == "return_request":
            return create_return_request_draft(
                reference=reference, session_id=session_id.strip(), status=status,
                summary=summary.strip(), order_id=order_id or "", reason=reason or "",
                details=details or {},
            )
        return create_support_case(record_type=record_type, **common)
    if Config.BUSINESS_DATA_BACKEND != "json":
        raise ValueError("BUSINESS_DATA_BACKEND must be 'json' or 'postgres'.")
    database_path = Path(DEFAULT_DB_PATH)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=10)
    try:
        with _DB_LOCK, connection:
            _ensure_schema(connection)
            connection.execute(
                """INSERT INTO demo_support_records
                   (reference, record_type, session_id, order_id, reason, summary,
                    status, details_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    reference, record_type, session_id.strip(), order_id,
                    reason, summary.strip(), status,
                    json.dumps(details or {}, ensure_ascii=False), created_at,
                ),
            )
    finally:
        connection.close()

    return {
        "reference": reference, "record_type": record_type,
        "session_id": session_id.strip(), "order_id": order_id,
        "reason": reason, "summary": summary.strip(), "status": status,
        "details": details or {}, "created_at": created_at,
    }


def list_demo_records(
    record_type: str | None = None, session_id: str | None = None
) -> list[dict[str, Any]]:
    """Read local simulation records; intended for tests and local inspection."""
    if Config.BUSINESS_DATA_BACKEND == "postgres":
        from app.db.repositories import list_return_request_drafts, list_support_cases
        cases = [] if record_type == "return_request" else list_support_cases(record_type, session_id)
        drafts = [] if record_type and record_type != "return_request" else list_return_request_drafts(session_id)
        return sorted(cases + drafts, key=lambda item: (item.get("created_at") or "", item["reference"]))
    if Config.BUSINESS_DATA_BACKEND != "json":
        raise ValueError("BUSINESS_DATA_BACKEND must be 'json' or 'postgres'.")
    database_path = Path(DEFAULT_DB_PATH)
    if not database_path.is_file():
        return []
    clauses: list[str] = []
    values: list[str] = []
    if record_type:
        clauses.append("record_type = ?")
        values.append(record_type)
    if session_id:
        clauses.append("session_id = ?")
        values.append(session_id)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    connection = sqlite3.connect(database_path, timeout=10)
    try:
        with _DB_LOCK:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM demo_support_records" + where + " ORDER BY created_at, reference",
                values,
            ).fetchall()
    finally:
        connection.close()
    return [
        {
            "reference": row["reference"], "record_type": row["record_type"],
            "session_id": row["session_id"], "order_id": row["order_id"],
            "reason": row["reason"], "summary": row["summary"],
            "status": row["status"], "details": json.loads(row["details_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]
