"""Customer frustration scoring and human-handoff summaries.

Combines a multilingual sentiment model
(``cardiffnlp/twitter-xlm-roberta-base-sentiment``) with rule-based signals
for English, Hindi (Devanagari) and Hinglish: escalation and legal-threat
phrases, ALL CAPS shouting, repeated ``?``/``!``, mild swearing and
"still/again/third time" repetition words. The model is loaded lazily, once
per process, and only from the local Hugging Face cache unless
``SENTIMENT_ALLOW_DOWNLOAD`` is true. When the model is unavailable the
scorer falls back to the lexicon alone and never raises.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
DEFAULT_FRUSTRATION_THRESHOLD = 0.65
SENTIMENT_LABELS = ("negative", "neutral", "positive")
MAX_SUMMARY_MESSAGE_CHARS = 280

# Weight of the model's negative probability when blended with rule signals.
_MODEL_WEIGHT = 0.6
# Recency weights for the last three customer turns, oldest to newest.
_RECENCY_WEIGHTS = (0.2, 0.3, 0.5)
_REPEATED_INTENT_BONUS = 0.2
_REPEATED_ORDER_BONUS = 0.15
# Intents too generic to count as "asking the same thing again".
_GENERIC_INTENTS = frozenset({"greeting", "goodbye", "thanks", "out_of_scope", "unknown"})
_ASSISTANT_ROLES = frozenset({"assistant", "bot", "agent", "system"})
_ORDER_ID_PATTERN = re.compile(r"\bORD-\d{4}\b", re.IGNORECASE)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# --- Rule lexicon ---------------------------------------------------------
# (category, weight, phrases). Latin phrases match case-insensitively on word
# boundaries; Devanagari phrases match as substrings after NFC normalisation.
_LEXICON: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    (
        "legal_threat",
        0.75,
        (
            "consumer court", "consumer forum", "legal action", "lawyer", "police complaint",
            "file a complaint", "report you", "complaint karunga", "complaint karungi",
            "case karunga", "case karungi", "court jaunga", "court jaungi",
            "शिकायत करूंगा", "शिकायत करूँगा", "शिकायत करुंगा", "शिकायत करूंगी",
            "शिकायत करूँगी", "उपभोक्ता फोरम", "उपभोक्ता अदालत", "कोर्ट", "कानूनी कार्रवाई",
        ),
    ),
    (
        "manager_request",
        0.55,
        (
            "talk to your manager", "speak to your manager", "speak to a manager",
            "talk to a manager", "your supervisor", "your senior", "real person",
            "human agent", "manager se baat", "senior se baat", "kisi insaan se baat",
            "मैनेजर से बात", "सीनियर से बात",
        ),
    ),
    (
        "escalation_phrase",
        0.45,
        (
            "worst service", "worst experience", "worst company", "fed up", "sick of",
            "pathetic", "ridiculous", "disgusting", "unacceptable", "fraud", "cheated",
            "scam", "waste of time", "never ordering again", "bakwas", "bakwaas",
            "bekar", "bekaar", "pagal bana rahe", "pagal bana diya", "pagal samjha",
            "thak gaya", "thak gayi", "tang aa gaya", "tang aa gayi", "pareshan",
            "dhokha", "बेकार", "बकवास", "परेशान", "धोखा", "तंग आ गया", "तंग आ गई",
            "पागल बना", "थक गया", "थक गई",
        ),
    ),
    (
        "swear_lite",
        0.35,
        (
            "damn", "crap", "bloody", "stupid", "idiot", "idiots", "useless", "nonsense",
            "rubbish", "wtf", "hell", "bewakoof", "ghatiya", "faltu", "bekaar log",
            "घटिया", "बेवकूफ", "फालतू",
        ),
    ),
    (
        "repetition_word",
        0.25,
        (
            "still", "again", "third time", "3rd time", "second time", "fourth time",
            "multiple times", "so many times", "how many times", "phir se", "fir se",
            "abhi tak", "ab tak", "kitni baar", "kitni bar", "kab tak", "baar baar",
            "teesri baar", "tisri baar", "dusri baar", "doosri baar",
            "फिर से", "अभी तक", "अब तक", "कितनी बार", "तीसरी बार", "कब तक",
        ),
    ),
)
_SHOUTING_WEIGHT = 0.3
_REPEATED_PUNCTUATION_WEIGHT = 0.2
_POSITIVE_WORDS = re.compile(
    r"\b(thanks|thank you|thx|great|awesome|perfect|love it|shukriya|dhanyavaad|dhanyawad|badhiya)\b"
    r"|धन्यवाद|शुक्रिया|बढ़िया",
    re.IGNORECASE,
)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_REPEATED_PUNCTUATION = re.compile(r"[!?]{2,}")
_LATIN_WORD = re.compile(r"[A-Za-z]+")


def _compile_lexicon() -> list[tuple[str, float, str, Callable[[str], bool]]]:
    compiled = []
    for category, weight, phrases in _LEXICON:
        for phrase in phrases:
            normalized = unicodedata.normalize("NFC", phrase)
            if _DEVANAGARI.search(normalized):
                compiled.append((category, weight, phrase, lambda text, p=normalized: p in text))
            else:
                pattern = re.compile(r"(?<![A-Za-z])" + re.escape(normalized) + r"(?![A-Za-z])", re.IGNORECASE)
                compiled.append((category, weight, phrase, lambda text, rx=pattern: bool(rx.search(text))))
    return compiled


_COMPILED_LEXICON = _compile_lexicon()


def _is_shouting(text: str) -> bool:
    words = [word for word in _LATIN_WORD.findall(_ORDER_ID_PATTERN.sub(" ", text)) if len(word) >= 3]
    caps = [word for word in words if word.isupper()]
    if len(caps) < 2:
        return False
    # Mostly-caps text, or two capitalised words in a row ("WORST SERVICE");
    # a lone acronym such as COD or UPI does not count.
    consecutive = any(a.isupper() and b.isupper() for a, b in zip(words, words[1:]))
    return consecutive or len(caps) / len(words) >= 0.5


def rule_signals(text: str) -> tuple[list[str], float]:
    """Return matched rule signals and their combined 0..1 rule score."""
    normalized = unicodedata.normalize("NFC", text or "")
    signals: list[str] = []
    weights: list[float] = []
    seen_categories: set[str] = set()
    for category, weight, phrase, matches in _COMPILED_LEXICON:
        if matches(normalized):
            signals.append(f"{category}:{phrase}")
            # One weight per category so a single rant does not saturate the score.
            if category not in seen_categories:
                seen_categories.add(category)
                weights.append(weight)
    if _is_shouting(normalized):
        signals.append("shouting")
        weights.append(_SHOUTING_WEIGHT)
    if _REPEATED_PUNCTUATION.search(normalized):
        signals.append("repeated_punctuation")
        weights.append(_REPEATED_PUNCTUATION_WEIGHT)

    remaining = 1.0
    for weight in weights:
        remaining *= 1.0 - weight
    return signals, round(1.0 - remaining, 4)


# --- Sentiment model --------------------------------------------------------
@dataclass(frozen=True)
class SentimentBundle:
    tokenizer: Any
    model: Any
    device: str
    torch: Any
    labels: tuple[str, ...]


# Cache successes and failures so a missing model is only probed once.
_SENTIMENT_CACHE: dict[tuple[str, bool], SentimentBundle | None] = {}
_SENTIMENT_LOCK = threading.Lock()


def resolve_sentiment_model_name() -> str:
    return os.getenv("SENTIMENT_MODEL_NAME", DEFAULT_SENTIMENT_MODEL)


def _pick_device(torch: Any) -> str:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_sentiment_model(allow_download: bool | None = None) -> SentimentBundle | None:
    """Load the sentiment model once; return ``None`` if it cannot be loaded.

    ``allow_download`` defaults to the ``SENTIMENT_ALLOW_DOWNLOAD`` env flag
    (false), in which case only the local Hugging Face cache is used.
    """
    if allow_download is None:
        allow_download = _as_bool(os.getenv("SENTIMENT_ALLOW_DOWNLOAD"), default=False)
    model_name = resolve_sentiment_model_name()
    cache_key = (model_name, allow_download)

    with _SENTIMENT_LOCK:
        if cache_key in _SENTIMENT_CACHE:
            return _SENTIMENT_CACHE[cache_key]
        bundle: SentimentBundle | None = None
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name, local_files_only=not allow_download
                )
            except (ImportError, ValueError):
                # The repo ships only a sentencepiece model; building the fast
                # tokenizer needs protobuf, so fall back to the slow one.
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name, local_files_only=not allow_download, use_fast=False
                )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_name, local_files_only=not allow_download
            )
            device = _pick_device(torch)
            model.to(device)
            model.eval()
            id2label = getattr(model.config, "id2label", None) or {}
            labels = tuple(
                str(id2label.get(index, SENTIMENT_LABELS[index] if index < 3 else index)).lower()
                for index in range(len(id2label) or 3)
            )
            bundle = SentimentBundle(tokenizer, model, device, torch, labels)
        except Exception as exc:  # noqa: BLE001 - lexicon fallback must never raise
            logger.warning("Sentiment model %s unavailable, using lexicon only: %s", model_name, exc)
        _SENTIMENT_CACHE[cache_key] = bundle
        return bundle


def clear_sentiment_cache() -> None:
    """Clear the loader cache; primarily useful for isolated tests."""
    with _SENTIMENT_LOCK:
        _SENTIMENT_CACHE.clear()


def sentiment_probabilities(text: str) -> dict[str, float] | None:
    """Return ``{negative, neutral, positive}`` probabilities, or ``None``."""
    bundle = load_sentiment_model()
    if bundle is None or not (text or "").strip():
        return None
    try:
        torch = bundle.torch
        inputs = bundle.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = bundle.model(**inputs).logits[0]
        probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sentiment inference failed, using lexicon only: %s", exc)
        return None
    by_label = dict(zip(bundle.labels, probs))
    return {label: float(by_label.get(label, 0.0)) for label in SENTIMENT_LABELS}


# --- Public scoring API ------------------------------------------------------
def score_message(text: str) -> dict[str, Any]:
    """Score one customer message for frustration.

    Returns ``{"sentiment", "negative_prob", "signals", "score"}`` where
    ``score`` is in 0..1 and blends the model's negative probability with the
    rule score (noisy-OR), or is the rule score alone without a model.
    """
    signals, rule_score = rule_signals(text)
    probs = sentiment_probabilities(text)

    if probs is not None:
        sentiment = max(SENTIMENT_LABELS, key=lambda label: probs[label])
        negative_prob = probs["negative"]
        score = 1.0 - (1.0 - _MODEL_WEIGHT * negative_prob) * (1.0 - rule_score)
    else:
        negative_prob = rule_score
        score = rule_score
        if rule_score >= 0.3:
            sentiment = "negative"
        elif _POSITIVE_WORDS.search(unicodedata.normalize("NFC", text or "")):
            sentiment = "positive"
        else:
            sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "negative_prob": round(float(negative_prob), 4),
        "signals": signals,
        "score": round(min(max(score, 0.0), 1.0), 4),
    }


def frustration_threshold() -> float:
    raw = os.getenv("FRUSTRATION_THRESHOLD")
    try:
        value = float(raw) if raw not in (None, "") else DEFAULT_FRUSTRATION_THRESHOLD
    except ValueError:
        return DEFAULT_FRUSTRATION_THRESHOLD
    return value if 0.0 <= value <= 1.0 else DEFAULT_FRUSTRATION_THRESHOLD


def _user_turns(turns: list[dict]) -> list[dict]:
    return [
        turn for turn in turns or []
        if str(turn.get("role", "user")).lower() not in _ASSISTANT_ROLES and turn.get("text")
    ]


def _turn_order_ids(turn: dict) -> set[str]:
    ids = {match.upper() for match in _ORDER_ID_PATTERN.findall(turn.get("text") or "")}
    if turn.get("order_id"):
        ids.add(str(turn["order_id"]).upper())
    return ids


def conversation_frustration(turns: list[dict], threshold: float | None = None) -> dict[str, Any]:
    """Score frustration over the last three customer turns.

    Each turn needs ``text`` and may carry ``intent``, ``order_id``,
    ``resolved`` and ``role`` (assistant turns are ignored). Per-turn scores
    are recency-weighted (the latest turn alone can also trigger escalation),
    then boosted when the same intent recurs in 2+ of the last 3 turns or an
    order ID is repeated without being resolved.
    """
    threshold = frustration_threshold() if threshold is None else threshold
    recent = _user_turns(turns)[-3:]
    if not recent:
        return {"score": 0.0, "escalate": False, "reasons": []}

    scored = [score_message(turn["text"]) for turn in recent]
    weights = _RECENCY_WEIGHTS[-len(recent):]
    weighted = sum(w * s["score"] for w, s in zip(weights, scored)) / sum(weights)
    latest = scored[-1]
    base = max(weighted, latest["score"])

    reasons: list[str] = []
    signal_kinds = Counter(signal.split(":", 1)[0] for s in scored for signal in s["signals"])
    for kind, count in signal_kinds.most_common():
        reasons.append(f"{kind.replace('_', ' ')} detected ({count}x)")
    if latest["sentiment"] == "negative":
        reasons.append(f"latest message negative (p={latest['negative_prob']:.2f})")

    bonuses: list[float] = []
    intents = [
        str(turn["intent"]) for turn in recent
        if turn.get("intent") and str(turn["intent"]) not in _GENERIC_INTENTS
    ]
    if intents:
        intent, count = Counter(intents).most_common(1)[0]
        if count >= 2:
            bonuses.append(_REPEATED_INTENT_BONUS)
            reasons.append(f"intent '{intent}' repeated in {count} of last {len(recent)} turns")

    if not recent[-1].get("resolved"):
        order_counts = Counter(oid for turn in recent if not turn.get("resolved") for oid in _turn_order_ids(turn))
        repeated_orders = sorted(oid for oid, count in order_counts.items() if count >= 2)
        if repeated_orders:
            bonuses.append(_REPEATED_ORDER_BONUS)
            reasons.append(f"order {', '.join(repeated_orders)} repeated without resolution")

    remaining = 1.0 - base
    for bonus in bonuses:
        remaining *= 1.0 - bonus
    score = round(min(max(1.0 - remaining, 0.0), 1.0), 4)
    escalate = score >= threshold
    if escalate:
        reasons.append(f"score {score:.2f} >= threshold {threshold:.2f}")
    return {"score": score, "escalate": escalate, "reasons": reasons}


# --- Handoff summary ---------------------------------------------------------
def _identity(text: str) -> str:
    return text


def _get_masker() -> Callable[[str], str]:
    try:
        from app.services.pii import mask_pii  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - optional module
        return _identity
    return mask_pii


def _mood(score: float, escalate: bool) -> str:
    if escalate or score >= 0.65:
        return "frustrated"
    if score >= 0.4:
        return "annoyed"
    return "calm"


def handoff_summary(
    turns: list[dict],
    intents: list[str] | None,
    order_ids: list[str] | None,
    frustration: dict[str, Any] | float | None,
) -> str:
    """Build a short plain-text summary for a human agent, with PII masked."""
    mask = _get_masker()
    if isinstance(frustration, dict):
        score = float(frustration.get("score", 0.0))
        escalate = bool(frustration.get("escalate", False))
        reasons = list(frustration.get("reasons") or [])
    else:
        score, escalate, reasons = float(frustration or 0.0), False, []

    user_turns = _user_turns(turns)
    unique_intents = list(dict.fromkeys(str(i) for i in intents or [] if i))
    unique_orders = list(dict.fromkeys(str(o).upper() for o in order_ids or [] if o))
    last_message = " ".join((user_turns[-1]["text"] if user_turns else "").split())
    last_message = mask(last_message)
    if len(last_message) > MAX_SUMMARY_MESSAGE_CHARS:
        last_message = last_message[: MAX_SUMMARY_MESSAGE_CHARS - 3].rstrip() + "..."

    lines = [
        "Customer handoff summary",
        f"Mood: {_mood(score, escalate)} (frustration {score:.2f}, escalate: {'yes' if escalate else 'no'})",
    ]
    if reasons:
        lines.append("Why: " + "; ".join(mask(reason) for reason in reasons))
    lines.append("Intents seen: " + (", ".join(unique_intents) or "none"))
    lines.append("Order IDs: " + (", ".join(unique_orders) or "none"))
    lines.append(f"Customer messages: {len(user_turns)}")
    lines.append(f'Last customer message: "{last_message}"' if last_message else "Last customer message: none")
    return "\n".join(lines)
