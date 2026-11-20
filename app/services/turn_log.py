"""Append-only JSONL records of chat turns and customer feedback.

Turn records hold what is needed to monitor accuracy (intent, confidence,
route, latency); feedback records pair a rated answer with its message so
thumbs-down turns can become new labelled training or evaluation data.
Files live under ``data/runtime/`` by default, which Git ignores.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def append_record(directory: str | Path, filename: str, record: dict[str, Any]) -> None:
    """Write one JSON line, stamped with UTC time; safe across Flask worker threads."""
    path = Path(directory) / filename
    line = json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}, ensure_ascii=False)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
