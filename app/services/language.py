"""Pick the language a reply should be written in: Hindi, Hinglish or English.

Detection is a cheap heuristic: Devanagari script share decides Hindi, and a
curated set of romanised Hindi words decides Hinglish. Words that are also
common English (``main``, ``me``, ``to``, ``do``, ``hi``) are deliberately
excluded so ordinary English messages are not misread.
"""

from __future__ import annotations

import re

LANGUAGE_NAMES = {"hi": "Hindi", "hinglish": "Hinglish", "en": "English"}

HINGLISH_MARKERS = frozenset({
    "hai", "hain", "nahi", "nahin", "nhi", "kya", "kyaa", "mera", "meri", "mere", "mujhe", "mujhko",
    "kab", "kaise", "kaisa", "karna", "karo", "kariye", "karenge", "kar", "chahiye", "bhai", "yaar",
    "abhi", "kyun", "kyon", "kahan", "kidhar", "tha", "thi", "hoga", "hogi", "hoge", "ho",
    "gaya", "gayi", "aaya", "aayi", "aayega", "aayegi", "milega", "milegi", "mila", "mili", "wala",
    "wali", "bhi", "aur", "lekin", "par", "ka", "ki", "ke", "se", "ko", "ne", "hum", "humara",
    "hamara", "aap", "aapka", "aapki", "tum", "tera", "teri", "kuch", "koi", "bahut", "bohot",
    "accha", "acha", "theek", "thik", "jaldi", "kal", "aaj", "paisa", "paise", "wapas", "batao",
    "bataiye", "dijiye", "chahta", "chahti", "raha", "rahi", "rahe", "sakta", "sakti", "kitna",
    "kitne", "matlab", "haan", "ji",
})

_DEVANAGARI_RE = re.compile("[\u0900-\u097F]")
_WORD_RE = re.compile(r"[A-Za-z]+")


def detect_reply_language(text: str) -> str:
    """Return ``"hi"``, ``"hinglish"`` or ``"en"`` for the customer's message."""
    if not text or not text.strip():
        return "en"
    letters = [char for char in text if char.isalpha() or _DEVANAGARI_RE.match(char)]
    if letters:
        devanagari = sum(1 for char in letters if _DEVANAGARI_RE.match(char))
        if devanagari / len(letters) > 0.3:
            return "hi"
    words = [word.lower() for word in _WORD_RE.findall(text)]
    markers = sum(1 for word in words if word in HINGLISH_MARKERS)
    required = 1 if len(words) <= 4 else 2
    return "hinglish" if markers >= required else "en"


def language_name(code: str) -> str:
    """Human-readable name for a language code; unknown codes fall back to English."""
    return LANGUAGE_NAMES.get(code, "English")
