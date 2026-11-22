"""Fictional demo customers used to attribute tickets until real sign-in exists."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CUSTOMERS_PATH = Path(__file__).resolve().parents[2] / "data" / "mock" / "customers.json"


@lru_cache(maxsize=1)
def _customers() -> dict[str, dict[str, Any]]:
    with CUSTOMERS_PATH.open("r", encoding="utf-8-sig") as customers_file:
        records = json.load(customers_file)["customers"]
    return {record["customer_id"]: record for record in records}


def list_customers() -> list[dict[str, Any]]:
    return [{"customer_id": item["customer_id"], "name": item["name"], "order_ids": list(item["order_ids"])}
            for item in _customers().values()]


def get_customer(customer_id: str | None) -> dict[str, Any] | None:
    return _customers().get(customer_id) if isinstance(customer_id, str) else None
