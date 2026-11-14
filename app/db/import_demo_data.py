"""Explicit, repeatable import of fictional JSON fixture records into PostgreSQL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.services.data_repository import DEFAULT_DATA_DIR


def _read_records(data_dir: Path, filename: str, collection: str) -> list[dict[str, Any]]:
    with (data_dir / filename).open("r", encoding="utf-8-sig") as source:
        document = json.load(source)
    if document.get("data_type") != "demonstration_only":
        raise ValueError(f"{filename} is not marked demonstration_only")
    records = document.get(collection)
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise ValueError(f"{filename} has an invalid {collection} collection")
    return records


def import_fixtures(data_dir: str | Path = DEFAULT_DATA_DIR, *, dry_run: bool = False) -> dict[str, int]:
    """Import products before dependent orders; existing identifiers are skipped.

    Each record has its own transaction, so a bad record is reported and does not
    roll back earlier successful records. The source JSON files are read-only.
    """
    directory = Path(data_dir)
    result = {"imported": 0, "skipped": 0, "failed": 0}
    fixtures = (
        ("products.json", "products", "product", None),
        ("orders.json", "orders", "order", None),
        ("refunds.json", "refunds", "refund", None),
    )
    if dry_run:
        counts = {"products": 0, "orders": 0, "refunds": 0}
        for filename, collection, _, _ in fixtures:
            try:
                counts[collection] = len(_read_records(directory, filename, collection))
            except Exception as exc:
                print(f"FAILED {filename}: {exc}")
                result["failed"] += 1
        print(f"Dry run (no database access): {counts}; validation failures={result['failed']}")
        return {**counts, "failed": result["failed"]}

    from app.config import Config
    if not Config.DATABASE_URL:
        raise RuntimeError("Set DATABASE_URL before importing demo fixtures.")
    if Config.BUSINESS_DATA_BACKEND != "postgres":
        raise RuntimeError("Set BUSINESS_DATA_BACKEND=postgres explicitly before importing.")
    from app.db.repositories import import_order, import_product, import_refund

    importers = {"product": import_product, "order": import_order, "refund": import_refund}
    for filename, collection, kind, _ in fixtures:
        try:
            records = _read_records(directory, filename, collection)
        except Exception as exc:
            print(f"FAILED {filename}: {exc}")
            result["failed"] += 1
            continue
        for index, record in enumerate(records, start=1):
            try:
                imported = importers[kind](record)
                result["imported" if imported else "skipped"] += 1
                print(f"{'IMPORTED' if imported else 'SKIPPED'} {filename}[{index}]")
            except Exception as exc:
                result["failed"] += 1
                print(f"FAILED {filename}[{index}]: {type(exc).__name__}: {exc}")
    print(f"Import complete (demo data only): {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Validate/count fixtures without PostgreSQL access")
    arguments = parser.parse_args()
    result = import_fixtures(arguments.data_dir, dry_run=arguments.dry_run)
    raise SystemExit(1 if result.get("failed", 0) else 0)


if __name__ == "__main__":
    main()
