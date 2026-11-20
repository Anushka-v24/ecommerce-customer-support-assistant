"""Read demonstration JSON without coupling storage access to graph nodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "mock"


class DataRepositoryError(RuntimeError):
    """Raised when a demonstration data file is missing or malformed."""


def _uses_postgres() -> bool:
    from app.config import Config
    backend = Config.BUSINESS_DATA_BACKEND
    if backend not in {"json", "postgres"}:
        raise DataRepositoryError("BUSINESS_DATA_BACKEND must be 'json' or 'postgres'.")
    return backend == "postgres"


def _load_document(filename: str, data_dir: str | Path | None = None) -> dict[str, Any]:
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    path = directory / filename
    try:
        with path.open("r", encoding="utf-8-sig") as data_file:
            payload = json.load(data_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataRepositoryError(f"Unable to read demonstration data file: {filename}") from exc
    if not isinstance(payload, dict):
        raise DataRepositoryError(f"Invalid document structure in {filename}.")
    return payload


def _load_index(
    filename: str,
    collection_name: str,
    id_field: str,
    data_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    payload = _load_document(filename, data_dir)
    records = payload.get(collection_name)
    if not isinstance(records, list):
        raise DataRepositoryError(f"Invalid {collection_name} collection in {filename}.")

    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise DataRepositoryError(f"Invalid record in {filename}.")
        record_id = record.get(id_field)
        if not isinstance(record_id, str) or not record_id.strip():
            raise DataRepositoryError(f"Record has no valid {id_field} in {filename}.")
        record_id = record_id.strip().upper()
        if record_id in indexed:
            raise DataRepositoryError(f"Duplicate {id_field} in {filename}.")
        indexed[record_id] = dict(record)
    return indexed


def load_orders(data_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return a fresh order index keyed by normalized order ID."""
    if _uses_postgres():
        from app.db.repositories import list_orders
        return {row["order_id"]: row for row in list_orders()}
    return _load_index("orders.json", "orders", "order_id", data_dir)


def load_products(data_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return a fresh product index keyed by normalized product ID."""
    if _uses_postgres():
        from app.db.repositories import list_products
        return {row["product_id"]: row for row in list_products()}
    return _load_index("products.json", "products", "product_id", data_dir)


def load_refunds(data_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return refund status records keyed by order ID; the supplied fixture is empty."""
    if _uses_postgres():
        from app.db.repositories import list_refunds
        return {row["order_id"]: row for row in list_refunds()}
    return _load_index("refunds.json", "refunds", "order_id", data_dir)


def load_policies(data_dir: str | Path | None = None) -> dict[str, Any]:
    """Return the demonstration return/refund policy document."""
    return _load_document("policies.json", data_dir)


def load_store_information(data_dir: str | Path | None = None) -> dict[str, Any]:
    """Return structured demonstration sales, offers, discounts, and store details."""
    return _load_document("store_information.json", data_dir)


def load_shipping_payment(data_dir: str | Path | None = None) -> dict[str, Any]:
    """Return demonstration delivery methods, serviceable cities, fees, and payment terms."""
    return _load_document("shipping_payment.json", data_dir)
