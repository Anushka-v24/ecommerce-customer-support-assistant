"""Tool wrapper for local demonstration store information."""

from __future__ import annotations

from typing import Any

from app.services.store_information_service import get_store_information


def store_information_tool(query: str | None) -> dict[str, Any]:
    return get_store_information(query)
