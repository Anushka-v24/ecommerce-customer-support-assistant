"""Describe a customer's damage photo with a vision model, as evidence only.

The photo is checked (type, size), stored under ``data/runtime/evidence/`` and
described by a Groq vision model in strict JSON. The result is attached to the
conversation as evidence for a damaged-item report; it never approves a refund
or replacement by itself.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

from app.config import Config
from app.services.pii import mask_pii
from app.services.turn_log import append_record

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
_SIGNATURES = {"image/jpeg": (b"\xff\xd8\xff",), "image/png": (b"\x89PNG\r\n\x1a\n",), "image/webp": (b"RIFF",)}
DAMAGE_TYPES = {"none", "cracked", "broken", "dented", "scratched", "torn", "crushed_packaging", "water_damage", "missing_parts", "other"}
SEVERITIES = {"none", "minor", "moderate", "severe"}

SYSTEM_PROMPT = (
    "You inspect photos sent by e-commerce customers who report a damaged delivery. Describe only what is visible. "
    "Never decide refunds. Reply with a single JSON object with exactly these keys: "
    '"is_product_photo" (boolean: the photo shows a product or its packaging), '
    '"damage_visible" (boolean), '
    f'"damage_type" (one of: {", ".join(sorted(DAMAGE_TYPES))}), '
    f'"severity" (one of: {", ".join(sorted(SEVERITIES))}), '
    '"description" (one or two plain sentences, no personal data), '
    '"confidence" (number from 0 to 1).'
)


class EvidenceError(ValueError):
    """The upload is not a usable photo; the message is safe to show."""


def _check_image(data: bytes, mime: str) -> None:
    if mime not in ALLOWED_TYPES:
        raise EvidenceError("Please upload a JPEG, PNG or WebP photo.")
    if not data:
        raise EvidenceError("The photo is empty.")
    if len(data) > Config.EVIDENCE_MAX_BYTES:
        raise EvidenceError(f"The photo is too large; the limit is {Config.EVIDENCE_MAX_BYTES // (1024 * 1024)} MB.")
    if not data.startswith(_SIGNATURES[mime]) or (mime == "image/webp" and data[8:12] != b"WEBP"):
        raise EvidenceError("That file doesn't look like the photo type it claims to be.")


def _validate(parsed: object) -> dict:
    """Accept only the exact schema; anything else is treated as a failed analysis."""
    keys = {"is_product_photo", "damage_visible", "damage_type", "severity", "description", "confidence"}
    if not isinstance(parsed, dict) or set(parsed) != keys:
        raise ValueError("unexpected keys")
    if not isinstance(parsed["is_product_photo"], bool) or not isinstance(parsed["damage_visible"], bool):
        raise ValueError("flags must be booleans")
    if parsed["damage_type"] not in DAMAGE_TYPES or parsed["severity"] not in SEVERITIES:
        raise ValueError("unknown category")
    if not isinstance(parsed["description"], str) or not parsed["description"].strip():
        raise ValueError("empty description")
    confidence = parsed["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence out of range")
    return {**parsed, "description": mask_pii(parsed["description"].strip())[:400], "confidence": round(float(confidence), 2)}


def _describe(data: bytes, mime: str) -> dict:
    from groq import Groq

    url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    response = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=Config.EVIDENCE_TIMEOUT_SECONDS, max_retries=1).chat.completions.create(
        model=Config.VISION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Inspect this delivery photo and return the JSON object."},
                {"type": "image_url", "image_url": {"url": url}},
            ]},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return _validate(json.loads(response.choices[0].message.content or "{}"))


def analyze_damage_photo(data: bytes, mime: str, session_id: str, order_id: str | None = None) -> dict:
    """Store the photo and return {evidence_id, analysis | None, summary} for the chat."""
    _check_image(data, mime)
    evidence_id = f"EV-{uuid4().hex[:8].upper()}"
    folder = Path(Config.TURN_LOG_DIR) / "evidence"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{evidence_id}{ALLOWED_TYPES[mime]}").write_bytes(data)

    analysis = None
    if os.getenv("GROQ_API_KEY"):
        try:
            analysis = _describe(data, mime)
        except Exception as exc:
            # The photo is still kept as evidence for a person to look at.
            logger.warning("Damage photo analysis failed (%s)", type(exc).__name__)
    if analysis is None:
        summary = f"Photo evidence {evidence_id} attached (not analysed automatically; a support agent will review it)."
    elif not analysis["is_product_photo"]:
        summary = f"Photo evidence {evidence_id} attached, but it doesn't appear to show a product or its packaging."
    elif analysis["damage_visible"]:
        summary = f"Photo evidence {evidence_id}: {analysis['description']} (looks {analysis['severity']}, {analysis['damage_type'].replace('_', ' ')})."
    else:
        summary = f"Photo evidence {evidence_id}: no damage is clearly visible. {analysis['description']}"
    append_record(Config.TURN_LOG_DIR, "evidence.jsonl", {
        "evidence_id": evidence_id, "session_id": session_id, "order_id": order_id,
        "mime": mime, "bytes": len(data), "analysis": analysis, "model": Config.VISION_MODEL if analysis else None,
    })
    return {"evidence_id": evidence_id, "analysis": analysis, "summary": summary}
