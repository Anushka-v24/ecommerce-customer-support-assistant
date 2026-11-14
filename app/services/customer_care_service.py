"""Prepare a local simulated handoff; no agent platform is contacted."""

from __future__ import annotations

from typing import Any

from app.services.case_repository import create_demo_record


def request_customer_handoff(
    session_id: str | None, reason: str | None, summary: str | None
) -> dict[str, Any]:
    """Persist a handoff request with an explicit simulation status."""
    if not isinstance(session_id, str) or not session_id.strip():
        return {"outcome": "missing_session_id", "handoff": None, "error": "A conversation session is required."}
    if not isinstance(reason, str) or not reason.strip():
        return {"outcome": "missing_reason", "handoff": None, "error": "Please share why you need a human representative."}
    if not isinstance(summary, str) or not summary.strip():
        return {"outcome": "missing_summary", "handoff": None, "error": "A conversation summary is required."}

    record = create_demo_record(
        record_type="customer_handoff",
        session_id=session_id,
        reason=reason.strip()[:1000],
        summary=summary.strip()[:2000],
        status="simulated_handoff_ready",
        details={"live_agent_contacted": False},
    )
    return {
        "outcome": "simulated_handoff_ready", "handoff": record,
        "live_agent_contacted": False, "error": None,
    }
