"""Structured LLM extraction for support details the phrase rules cannot resolve.

The deterministic rules in ``issue_resolver`` and ``product_service`` stay the
first step. This module is only consulted when they find nothing, so clear
English messages never pay for a network call. Every answer is validated
against a closed set of values; anything else, or any provider failure, returns
``None`` so callers keep their existing clarification behaviour.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

from app.config import Config
from app.services.pii import mask_obj

logger = logging.getLogger(__name__)

ISSUE_TYPES = ("refund_status", "return_order", "damaged_order", "missing_order")
_ORDER_ID = re.compile(r"^ORD-\d{4}$")

ISSUE_PROMPT = """
You extract structured details from one e-commerce customer support message.
The message may be English, Hindi (Devanagari), or Hinglish (Hindi in Latin letters).

issue_type must be exactly one of:
- refund_status: asking where a refund is, whether it was processed, or when money comes back
- return_order: wanting to return or send back an item, or asking to start a return
- damaged_order: an item arrived broken, cracked, defective, or not working
- missing_order: an order or parcel has not arrived, is missing, or shows delivered but was not received
- null: none of these clearly applies, or more than one applies equally (for example a wrong item,
  a vague complaint, or a payment failure)

order_id: the order ID written in the message normalised to the form ORD-1234, or null if none is written.
Never guess an order ID.

Examples:
"the item I received is broken" -> damaged_order
"what is the status of my refund for ORD-1002" -> refund_status, ORD-1002
"मेरा रिफंड कहाँ है?" -> refund_status
"parcel abhi tak nahi aaya" -> missing_order
"product kharab nikla, return karna hai" -> return_order
"i do not like my order" -> null
""".strip()

PRODUCT_PROMPT = """
You match a customer's product mention to one product in a small store catalogue.
The message may be English, Hindi (Devanagari), or Hinglish (Hindi in Latin letters).
Return the exact catalogue name the customer most plausibly means, including category words
(for example "phone" or "mobile" for a smartphone, "earphones" for headphones).
Return null if the message names no product, names a product the catalogue does not have,
or could equally mean more than one catalogue product.
""".strip()


def _enabled() -> bool:
    return Config.LLM_EXTRACTION_ENABLED and bool(os.getenv("GROQ_API_KEY"))


def _structured_call(system_prompt: str, payload: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from groq import Groq

        client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=Config.LLM_EXTRACTION_TIMEOUT_SECONDS,
            max_retries=1,
        )
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", Config.GROQ_MODEL),
            messages=mask_obj([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]),
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "extraction", "strict": True, "schema": schema},
            },
        )
        content = response.choices[0].message.content
        parsed = json.loads(content) if content else None
    except Exception as exc:
        # Provider errors fail closed: the caller asks the customer instead.
        logger.warning("LLM extraction failed (%s)", type(exc).__name__)
        return None
    return parsed if isinstance(parsed, dict) else None


@lru_cache(maxsize=512)
def extract_issue_details(query: str) -> dict[str, Any] | None:
    """Return ``{"issue_type", "order_id"}`` for a support message, or ``None``."""
    if not _enabled() or not isinstance(query, str) or not query.strip():
        return None
    schema = {
        "type": "object",
        "properties": {
            "issue_type": {"anyOf": [{"type": "string", "enum": list(ISSUE_TYPES)}, {"type": "null"}]},
            "order_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["issue_type", "order_id"],
        "additionalProperties": False,
    }
    parsed = _structured_call(ISSUE_PROMPT, {"message": query}, schema)
    if parsed is None:
        return None
    issue_type = parsed.get("issue_type")
    order_id = parsed.get("order_id")
    order_id = order_id.strip().upper() if isinstance(order_id, str) else None
    return {
        "issue_type": issue_type if issue_type in ISSUE_TYPES else None,
        # Only keep an ID that is literally present, so the model cannot invent one.
        "order_id": order_id if order_id and _ORDER_ID.fullmatch(order_id) and order_id in query.upper() else None,
    }


@lru_cache(maxsize=512)
def match_catalog_product(query: str, catalog_names: tuple[str, ...]) -> str | None:
    """Return one exact catalogue name the message refers to, or ``None``."""
    if not _enabled() or not isinstance(query, str) or not query.strip() or not catalog_names:
        return None
    schema = {
        "type": "object",
        "properties": {
            "product_name": {"anyOf": [{"type": "string", "enum": list(catalog_names)}, {"type": "null"}]},
        },
        "required": ["product_name"],
        "additionalProperties": False,
    }
    parsed = _structured_call(PRODUCT_PROMPT, {"message": query, "catalogue": list(catalog_names)}, schema)
    name = (parsed or {}).get("product_name")
    return name if name in catalog_names else None
