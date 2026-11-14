"""Tool wrapper for recording a simulated human-support handoff."""

from __future__ import annotations

from typing import Any

from app.services.customer_care_service import request_customer_handoff


def customer_handoff_tool(
    session_id: str | None, reason: str | None, summary: str | None
) -> dict[str, Any]:
    return request_customer_handoff(session_id, reason, summary)
