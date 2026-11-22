"""Translate an English reply into the customer's script (Hindi or Hinglish).

Workflows write their replies in English (some also in Hindi). After the turn,
the reply and its quick-reply labels are rewritten with Groq to match how the
customer wrote. IDs, amounts and dates are kept verbatim, personal data is
masked before the call, and any failure falls back to the original reply.
"""

from __future__ import annotations

import json
import logging
import os
import re

from app.config import Config
from app.services.language import detect_reply_language, language_name
from app.services.pii import mask_obj

logger = logging.getLogger(__name__)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")

SYSTEM_PROMPT = (
    "You translate customer-support replies for an Indian e-commerce store. Rewrite each text in the target "
    "language, keeping the meaning, tone and politeness exactly. Hindi means Devanagari script. Hinglish means "
    "romanised Hindi mixed with common English words, written in Latin script, the way customers type on WhatsApp. "
    "Keep verbatim: order IDs (ORD-1234), ticket IDs (TCK-...), references (FB-..., DEMO-...), product names, "
    "amounts, dates, URLs and anything in square brackets. Do not add or remove information."
)


def _already_in_script(text: str, target: str) -> bool:
    devanagari = len(_DEVANAGARI.findall(text))
    latin = len(_LATIN.findall(text))
    if target == "hi":
        return devanagari > latin
    return False


def translate_reply(query: str, response: str, quick_replies: list[dict[str, str]] | None = None) -> dict | None:
    """Return {"lang", "response", "quick_replies"} or None when no translation is needed or possible."""
    if not Config.REPLY_TRANSLATION_ENABLED or not response:
        return None
    target = detect_reply_language(query)
    if target == "en":
        return None
    if _already_in_script(response, target):
        return {"lang": target, "response": response, "quick_replies": quick_replies or []}
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    labels = [item["label"] for item in quick_replies or []]
    payload = {"target_language": language_name(target), "reply": response, "button_labels": labels}
    schema = {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "button_labels": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["reply", "button_labels"],
        "additionalProperties": False,
    }
    try:
        from groq import Groq

        result = Groq(api_key=api_key, timeout=Config.REPLY_TRANSLATION_TIMEOUT_SECONDS, max_retries=1).chat.completions.create(
            model=os.getenv("GROQ_MODEL", Config.GROQ_MODEL),
            messages=mask_obj([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]),
            temperature=0,
            response_format={"type": "json_schema", "json_schema": {"name": "translation", "strict": True, "schema": schema}},
        )
        parsed = json.loads(result.choices[0].message.content or "{}")
    except Exception as exc:
        # The English reply is always a safe answer; never fail the turn over translation.
        logger.warning("Reply translation failed (%s)", type(exc).__name__)
        return None
    text = parsed.get("reply")
    new_labels = parsed.get("button_labels")
    if not isinstance(text, str) or not text.strip():
        return None
    translated_replies = quick_replies or []
    if isinstance(new_labels, list) and len(new_labels) == len(labels) and all(isinstance(x, str) and x.strip() for x in new_labels):
        # Button values stay in English: they are what the workflow matches on.
        translated_replies = [{**item, "label": label} for item, label in zip(quick_replies or [], new_labels)]
    return {"lang": target, "response": text.strip(), "quick_replies": translated_replies}
