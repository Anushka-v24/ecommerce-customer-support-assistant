"""Read demo store sections without treating sample offers as current/live."""

from __future__ import annotations

import json
from typing import Any

from app.services.store_information_service import get_store_information


def format_store_information(result: dict[str, Any]) -> str:
    if result.get("outcome") != "found":
        return "Demonstration store information is unavailable right now."
    sections = result.get("sections", {})
    lines = ["Demonstration store information only — these details and offers are not live or redeemable."]
    for name, value in sections.items():
        lines.append(f"{name.replace('_', ' ').title()}: {json.dumps(value, ensure_ascii=False)}")
    return "\n".join(lines)

