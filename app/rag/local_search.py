"""Offline keyword retrieval over the knowledge base, used when RAG providers are not configured.

Documents are split into their ``##`` sections and ranked with BM25. The
answer is the best section quoted as written (extractive, so always grounded)
with its document as the source. Approved documents are used first; the
fictional demo set only in development, and the answer says so. A weak match
returns nothing, so the assistant declines instead of guessing.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from app.config import Config
from app.rag.documents import load_chunks

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "the", "is", "are", "was", "be", "to", "of", "and", "or", "in", "on", "for", "with", "my", "me", "i",
    "you", "your", "it", "this", "that", "do", "does", "can", "could", "would", "should", "how", "what", "when",
    "where", "which", "who", "why", "please", "pls", "there", "any", "will", "if", "at", "by", "from", "about", "as",
    "we", "our", "us", "get", "have", "has", "so", "just", "want", "need", "like", "know", "tell",
    "hai", "hain", "ka", "ki", "ke", "ko", "se", "me", "mein", "kya", "kaise", "kab", "kahan", "mujhe", "mera", "meri",
    "karte", "karna", "karni", "karu", "karo", "going", "currently", "now", "right", "today",
    "item", "items", "many", "much", "thing", "things", "stuff", "send", "find", "getting", "got", "am", "not",
}
# Common romanised Hindi words mapped to the English terms the documents use.
_HINGLISH = {
    "wapas": "return", "wapsi": "return", "paisa": "refund", "paise": "refund", "shikayat": "complaint",
    "complain": "complaint", "kharidna": "order", "kharidu": "order", "khareed": "order", "kharid": "order",
    "khata": "account", "password": "password", "badalna": "change", "badlo": "change", "radd": "cancel",
    "bhejo": "delivery", "delivery": "delivery", "pahunchega": "delivery", "cart": "cart", "hatao": "remove",
    "daalo": "add", "dalo": "add", "review": "review", "rai": "feedback", "sujhav": "suggestion",
}
# Question words mapped to the words the documents use for the same idea.
_SYNONYMS = {
    "window": "day", "period": "day", "long": "day", "leave": "write", "post": "write", "rate": "review",
    "complain": "issue", "complaint": "issue", "grievance": "issue", "place": "checkout", "buy": "order", "purchase": "order",
    "deliver": "delivery", "ship": "shipping", "shipped": "shipping", "money": "refund", "login": "sign",
    "signin": "sign", "otp": "otp", "delete": "delete", "basket": "cart", "coupon": "promotion",
    "discount": "promotion", "offer": "promotion", "sale": "promotion",
}
MIN_SCORE = 2.5
MIN_COVERAGE = 0.6


def _stem(word: str) -> str:
    if word.endswith(("ses", "xes", "ches", "shes")) and len(word) > 5:
        return word[:-2]
    for suffix in ("ing", "ed", "s"):
        if word.endswith(suffix) and not word.endswith("ss") and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _terms(text: str, expand: bool = False) -> list[str]:
    words = [_HINGLISH.get(w, w) for w in _WORD.findall(text.casefold())]
    terms = [_stem(w) for w in words if w not in _STOP and len(w) > 1]
    if expand:
        for w in words:
            synonym = _SYNONYMS.get(w) or _SYNONYMS.get(_stem(w))
            if synonym:
                terms.append(_stem(synonym))
    return terms


def _knowledge_dir() -> tuple[Path, bool] | None:
    approved = Path(Config.RAG_KNOWLEDGE_DIR)
    if load_chunks(approved):
        return approved, False
    if Config.APP_ENV.lower() == "development" or Config.RAG_USE_DEMO_KNOWLEDGE:
        return Path(Config.RAG_DEMO_KNOWLEDGE_DIR), True
    return None


@lru_cache(maxsize=4)
def _index(directory: str) -> tuple[list[dict], dict[str, float], float]:
    sections: dict[tuple[str, str], dict] = {}
    titles: dict[str, str] = {}
    for chunk in load_chunks(directory):
        doc_id = str(chunk["document_id"])
        text = str(chunk.get("text") or chunk.get("content") or "")
        heading = re.search(r"^# (.+)$", text, re.MULTILINE)
        if heading and doc_id not in titles:
            titles[doc_id] = re.sub(r"^DEMO ONLY\s*[—-]\s*(?:fictional\s+(?:guide|test\s+\w+)?:?\s*)?", "", heading.group(1), flags=re.I).strip()
        parts = re.split(r"\n(?=## )", text)
        for part in parts:
            match = re.match(r"## ([^\n]+)\n+(.*)", part.strip(), re.DOTALL)
            if match:
                key, body = (doc_id, match.group(1).strip()), match.group(2)
            elif len(parts) == 1 and heading:
                # Short documents without sections are one section under their title.
                key, body = (doc_id, ""), text[heading.end():]
            else:
                continue
            body = " ".join(body.split())
            # Overlapping chunks repeat sections; keep the most complete copy.
            if key not in sections or len(body) > len(sections[key]["body"]):
                sections[key] = {"document_id": doc_id, "heading": key[1], "body": body,
                                 "terms": Counter(_terms(f"{key[1]} {key[1]} {body}"))}
    docs = list(sections.values())
    for doc in docs:
        doc["title"] = titles.get(doc["document_id"], doc["document_id"])
        if not doc["heading"]:
            doc["heading"] = doc["title"][:1].upper() + doc["title"][1:]
        # The document title describes every section in it.
        doc["terms"].update(_terms(doc["title"]))
        doc["length"] = sum(doc["terms"].values())
    df = Counter(term for doc in docs for term in doc["terms"])
    n = len(docs) or 1
    idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}
    avg = sum(doc["length"] for doc in docs) / n
    return docs, idf, avg


def search(query: str, k: int = 4) -> list[dict]:
    """Top sections by BM25 with their scores (empty when nothing matches well)."""
    located = _knowledge_dir()
    if located is None:
        return []
    docs, idf, avg = _index(str(located[0]))
    terms = _terms(query, expand=True)
    base_terms = set(_terms(query))
    scored = []
    for doc in docs:
        score, matched = 0.0, 0
        for term in set(terms):
            tf = doc["terms"].get(term, 0)
            if tf:
                matched += 1
                score += idf[term] * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * doc["length"] / avg))
        if score > 0:
            # Coverage counts the customer's own words; a synonym hit stands in for its word.
            hits_base = sum(1 for t in base_terms if doc["terms"].get(t) or doc["terms"].get(_stem(_SYNONYMS.get(t, t))))
            coverage = hits_base / len(base_terms) if base_terms else 0.0
            scored.append({**doc, "score": round(score, 2), "coverage": round(coverage, 2), "demo": located[1]})
    return sorted(scored, key=lambda d: d["score"], reverse=True)[:k]


def answer(query: str) -> dict | None:
    """An extractive, cited answer, or None when the documents don't cover the question."""
    hits = search(query)
    # Both a strong match and most of the question's words: a stray shared word isn't enough.
    if not hits or hits[0]["score"] < MIN_SCORE or hits[0]["coverage"] < MIN_COVERAGE:
        return None
    best = hits[0]
    body = best["body"] if len(best["body"]) <= 700 else best["body"][:700].rsplit(". ", 1)[0] + "."
    note = " (from our demo help documents)" if best["demo"] else ""
    return {
        "answer": f"Here's what our help guide says{note}:\n\n{best['heading']}: {body}",
        "sources": [{"title": best["title"], "document_id": best["document_id"], "section": best["heading"]}],
        "documents": [{"document_id": h["document_id"], "section": h["heading"], "score": h["score"]} for h in hits],
    }
