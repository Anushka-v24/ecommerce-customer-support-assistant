"""Rule-based second-level resolver for clear support-problem phrases.

Patterns describe semantic phrases, not isolated keyword presence. Conflicting or
unmatched cues are returned for clarification; this resolver never changes the
trained top-level classifier labels.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.llm_extraction import extract_issue_details


ISSUE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "refund_status": (
        re.compile(r"\bwhere\s+(?:is|are)\s+(?:my\s+)?refund\b"),
        re.compile(r"\b(?:refund\s+(?:status|hasn't|has not|isn't|is not|still)|still\s+waiting\s+for\s+(?:my\s+)?refund)\b"),
        re.compile(r"\b(?:when\s+will\s+i\s+get|still\s+waiting\s+for)\s+(?:my\s+)?money\s+back\b"),
        re.compile(r"\bhas\s+my\s+refund\s+(?:been\s+)?(?:processed|issued)\b"),
    ),
    "return_order": (
        re.compile(r"\b(?:i\s+want|i'd\s+like|i\s+need|can\s+i|how\s+can\s+i)\s+(?:to\s+)?return\s+(?:my\s+|this\s+|the\s+)?(?:order|item|product)?\b"),
        re.compile(r"\b(?:start|initiate)\s+(?:a\s+)?return\b"),
        re.compile(r"\b(?:send|ship)\s+(?:my\s+|this\s+)?(?:order|item|product)\s+back\b"),
    ),
    "damaged_order": (
        re.compile(r"\b(?:arrived|came|was\s+delivered)\s+(?:broken|damaged|cracked)\b"),
        re.compile(r"\b(?:broken|damaged|cracked|defective)\s+(?:item|product|screen|on\s+arrival)\b"),
        re.compile(r"\b(?:screen|item|product)\s+(?:is|was|arrived)\s+(?:broken|damaged|cracked|defective)\b"),
        re.compile(r"\b(?:doesn't|does\s+not|isn't|is\s+not)\s+work\s+(?:after\s+delivery|out\s+of\s+the\s+box)\b"),
    ),
    "missing_order": (
        re.compile(r"\b(?:order|parcel|package|delivery)\s+(?:hasn't\s+arrived|has\s+not\s+arrived|never\s+arrived|not\s+arrived|is\s+missing)\b"),
        re.compile(r"\b(?:haven't|have\s+not|never)\s+received\s+(?:my\s+)?(?:order|package|parcel)\b"),
        re.compile(r"\bshows\s+delivered\s+but\s+(?:i|we)\s+(?:didn't|did\s+not)\s+receive\b"),
        re.compile(r"\bmy\s+(?:order|package|parcel)\s+is\s+missing\b"),
    ),
}


def resolve_issue_type(
    query: str | None, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve one clear issue, reuse explicit context, or ask for clarification."""
    if not isinstance(query, str) or not query.strip():
        return {"outcome": "clarification_required", "issue_type": None, "candidates": [], "error": "Please describe the issue."}

    normalized = " ".join(query.casefold().split())
    matches = [
        issue_type for issue_type, patterns in ISSUE_PATTERNS.items()
        if any(pattern.search(normalized) for pattern in patterns)
    ]
    if len(matches) == 1:
        return {"outcome": "resolved", "issue_type": matches[0], "candidates": matches, "source": "phrase_rules", "error": None}
    if len(matches) > 1:
        return {"outcome": "clarification_required", "issue_type": None, "candidates": matches, "error": "The message describes more than one support issue."}

    # Phrase rules are English-only; Hindi, Hinglish, and unusual phrasing go to
    # the structured extractor, which can only answer with a known issue type.
    extracted = extract_issue_details(query) or {}
    if extracted.get("issue_type") in ISSUE_PATTERNS:
        return {
            "outcome": "resolved", "issue_type": extracted["issue_type"], "candidates": [extracted["issue_type"]],
            "order_id": extracted.get("order_id"), "source": "llm_extraction", "error": None,
        }

    prior_issue = (context or {}).get("issue_type")
    if prior_issue in ISSUE_PATTERNS:
        return {"outcome": "resolved", "issue_type": prior_issue, "candidates": [prior_issue], "source": "conversation_context", "error": None}
    return {
        "outcome": "clarification_required", "issue_type": None,
        "candidates": list(ISSUE_PATTERNS),
        "error": "Please clarify whether this is about a refund status, a return, a damaged item, or an order that has not arrived.",
    }

