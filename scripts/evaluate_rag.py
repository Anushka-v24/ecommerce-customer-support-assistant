"""Score document retrieval on evals/rag_cases.jsonl (30 questions, 6 that must be declined).

Metrics:
- hit@4: an expected document is among the top 4 retrieved sections (answerable questions).
- answered: the assistant gives an answer (answerable) / correct refusal: it declines (unanswerable).
- groundedness: every sentence of the answer appears in, or is mostly made of words from, the retrieved text.

``--mode local`` (default) scores the offline keyword search used when RAG providers are not configured;
``--mode provider`` runs the full Pinecone + DeepSeek RAG pipeline (needs its keys).

Usage (from the project root)::

    PYTHONPATH=. .venv/bin/python scripts/evaluate_rag.py --mode local
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_WORD = re.compile(r"[a-z0-9]+")


def grounded(answer: str, context: str) -> bool:
    words_in_context = set(_WORD.findall(context.casefold()))
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        words = _WORD.findall(sentence.casefold())
        if words and sum(w in words_in_context for w in words) / len(words) < 0.8:
            return False
    return True


def run_local(question: str) -> tuple[list[str], str | None, str]:
    from app.rag import local_search

    hits = local_search.search(question)
    result = local_search.answer(question)
    # Grounded against what was retrieved plus every section of the cited document.
    cited = {source["document_id"] for source in result["sources"]} if result else set()
    docs, _, _ = local_search._index(str(local_search._knowledge_dir()[0]))
    sections = hits + [d for d in docs if d["document_id"] in cited]
    context = " ".join(f"{h['heading']} {h['body']}" for h in sections)
    body = result["answer"].split(":\n\n", 1)[-1] if result else None
    return [h["document_id"] for h in hits], body, context


def run_provider(question: str) -> tuple[list[str], str | None, str]:
    from app.rag.service import get_rag_service
    from app.rag.workflow import run_agentic_rag

    state = run_agentic_rag(question, [], service=get_rag_service())
    docs = state.get("rag_documents") or []
    context = " ".join(str(d.get("text") or d.get("content") or "") for d in docs)
    answer = state.get("rag_answer") if state.get("rag_status") == "answered" else None
    return [str(d.get("document_id")) for d in docs][:4], answer, context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["local", "provider"], default="local")
    parser.add_argument("--cases", type=Path, default=ROOT / "evals" / "rag_cases.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "evals" / "reports" / "rag_eval.json")
    args = parser.parse_args()
    run = run_local if args.mode == "local" else run_provider

    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for case in cases:
        retrieved, answer, context = run(case["question"])
        answerable = bool(case["expected"])
        rows.append({
            "id": case["id"], "question": case["question"], "answerable": answerable,
            "hit_at_4": answerable and bool(set(case["expected"]) & set(retrieved[:4])),
            "answered": answer is not None,
            "grounded": answer is None or grounded(answer, context),
            "retrieved": retrieved[:4],
        })
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if r["answered"]]
    summary = {
        "mode": args.mode, "cases": len(rows),
        "hit_at_4": round(100 * sum(r["hit_at_4"] for r in answerable) / len(answerable), 1),
        "answer_rate": round(100 * sum(r["answered"] for r in answerable) / len(answerable), 1),
        "correct_refusals": round(100 * sum(not r["answered"] for r in unanswerable) / len(unanswerable), 1),
        "groundedness": round(100 * sum(r["grounded"] for r in answered) / len(answered), 1) if answered else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for row in rows:
        if (row["answerable"] and not (row["hit_at_4"] and row["answered"])) or (not row["answerable"] and row["answered"]):
            print("MISS", row["id"], row["question"], row["retrieved"][:2], "answered" if row["answered"] else "declined")


if __name__ == "__main__":
    main()
