"""Mask personal and payment identifiers in free text before it is logged or stored.

Each kind of identifier is replaced with a stable tag (``[PHONE]``, ``[EMAIL]``,
``[CARD]``, ``[UPI]``, ``[AADHAAR]``, ``[PAN]``). Order IDs such as ``ORD-1001``,
amounts, dates, pincodes and product names are left untouched. Patterns run in
a fixed order so broader numeric shapes (cards) are claimed before narrower
ones (phones, Aadhaar) can match fragments of them.
"""

from __future__ import annotations

import re
from typing import Any

# Digit guards: a match must not start or end inside a longer digit run.
_NO_DIGIT_BEFORE = r"(?<![\w+])(?<!\d[ -])"
_NO_DIGIT_AFTER = r"(?![ -]?\d)"

_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
_UPI_RE = re.compile(r"(?<![\w.@-])[\w.-]{2,256}@[A-Za-z][A-Za-z0-9]{1,63}(?![\w@-])(?!\.[A-Za-z])")
_CARD_RE = re.compile(r"(?<![\d+])(?:\d[ -]?){12,18}\d(?!\d)")
_INDIAN_PHONE_RE = re.compile(_NO_DIGIT_BEFORE + r"(?:\+?91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}" + _NO_DIGIT_AFTER)
_INTL_PHONE_RE = re.compile(r"(?<![\w+])\+\d{1,3}(?:[ -]?\(?\d{1,5}\)?){2,5}" + _NO_DIGIT_AFTER)
_AADHAAR_RE = re.compile(_NO_DIGIT_BEFORE + r"[2-9]\d{3}([ -]?)\d{4}\1\d{4}" + _NO_DIGIT_AFTER)
_PAN_RE = re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b")


def _luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _mask_card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if 13 <= len(digits) <= 19 and _luhn_valid(digits):
        return "[CARD]"
    return match.group(0)


def _mask_intl_phone(match: re.Match[str]) -> str:
    digit_count = len(re.sub(r"\D", "", match.group(0)))
    return "[PHONE]" if 8 <= digit_count <= 15 else match.group(0)


def mask_pii(text: str) -> str:
    """Return ``text`` with phones, emails, cards, UPI IDs, Aadhaar and PAN numbers tagged."""
    if not text:
        return text
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _UPI_RE.sub("[UPI]", text)
    text = _CARD_RE.sub(_mask_card, text)
    text = _INDIAN_PHONE_RE.sub("[PHONE]", text)
    text = _INTL_PHONE_RE.sub(_mask_intl_phone, text)
    text = _AADHAAR_RE.sub("[AADHAAR]", text)
    text = _PAN_RE.sub("[PAN]", text)
    return text


def mask_obj(obj: Any) -> Any:
    """Recursively mask every string inside dicts, lists and tuples; other values pass through."""
    if isinstance(obj, str):
        return mask_pii(obj)
    if isinstance(obj, dict):
        return {key: mask_obj(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [mask_obj(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(mask_obj(item) for item in obj)
    return obj
