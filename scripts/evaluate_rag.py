"""Score document retrieval on evals/rag_cases.jsonl (30 questions, 6 that must be declined).

Metrics:
- hit@4: an expected document is among the top 4 retrieved sections (answerable questions).
- answered: the assistant gives an answer (answerable) / correct refusal: it declines (unanswerable).
- groundedness: a heuristic for paraphrased answers. Every number/amount/ID in the answer must appear in the
  retrieved text, and at least 60% of the answer's content words must too. (A word-for-word check would fail
  every reworded answer.)

``--mode retrieval`` scores only the vector search (local embeddings, no LLM calls, so it costs no Groq quota);
``--mode local`` (default) scores the offline keyword search used when RAG providers are not configured;
``--mode provider`` runs the full Pinecone + DeepSeek RAG pipeline (needs its keys).

Usage (from the project root)::

    PYTHONPATH=. .venv/bin/python scripts/evaluate_rag.py --mode local
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from app.config import Config  # noqa: E402  (run with PYTHONPATH=.)
_WORD = re.compile(r"[a-z0-9]+")


REFUSAL = "i could not find that information"
ERROR = "<provider error>"


def grounded(answer: str, context: str) -> bool:
    from app.rag.local_search import _STOP, _stem

    in_context = {_stem(w) for w in _WORD.findall(context.casefold())}
    words = [_stem(w) for w in _WORD.findall(answer.casefold())]
    if any(any(c.isdigit() for c in w) and w not in in_context for w in words):
        return False
    content = [w for w in words if len(w) > 3 and w not in _STOP and not any(c.isdigit() for c in w)]
    return not content or sum(w in in_context for w in content) / len(content) >= 0.6


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


def run_retrieval(question: str) -> tuple[list[str], str | None, str]:
    import app.graph  # noqa: F401
    from app.rag.service import get_rag_service

    docs = get_rag_service().retrieve(question)
    return [str(d["document_id"]) for d in docs], None, " ".join(str(d["text"]) for d in docs)


def top_score(question: str) -> float:
    from app.rag.service import get_rag_service

    service = get_rag_service()
    vector = service._embed([question])[0]
    matches = service._ensure_index().query(vector=vector, top_k=1, namespace=service._namespace())["matches"]
    return matches[0]["score"] if matches else 0.0


def run_provider(question: str) -> tuple[list[str], str | None, str]:
    import app.graph  # noqa: F401  (import the graph package first; app.rag.workflow and app.graph.nodes import each other)
    from app.rag.service import get_rag_service
    from app.rag.workflow import run_agentic_rag

    state = run_agentic_rag(question, [], service=get_rag_service())
    docs = state.get("rag_documents") or []
    context = " ".join(str(d.get("text") or d.get("content") or "") for d in docs)
    if state.get("rag_error"):
        # A provider failure (e.g. a 429 on the free tier) says nothing about the documents: report it separately.
        return [str(d.get("document_id")) for d in docs][:4], ERROR, context
    answer = state.get("rag_answer") if state.get("rag_status") == "answered" else None
    if answer and answer.strip().casefold().startswith(REFUSAL):
        answer = None  # the pipeline's fixed "not found" reply is a refusal, not an answer
    return [str(d.get("document_id")) for d in docs][:4], answer, context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["local", "retrieval", "provider"], default="local")
    parser.add_argument("--cases", type=Path, default=ROOT / "evals" / "rag_cases.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "evals" / "reports" / "rag_eval.json")
    parser.add_argument("--delay", type=float, default=None, help="Seconds between questions (default 5 in provider mode, to respect free-tier rate limits).")
    args = parser.parse_args()
    delay = args.delay if args.delay is not None else (5.0 if args.mode == "provider" else 0.0)
    run = {"local": run_local, "retrieval": run_retrieval, "provider": run_provider}[args.mode]
    if args.mode == "retrieval":
        import statistics

        cases_ = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
        hits, scores = [], {True: [], False: []}
        for case in cases_:
            ids, _, _ = run_retrieval(case["question"])
            scores[bool(case["expected"])].append(top_score(case["question"]))
            if case["expected"]:
                hits.append(bool(set(case["expected"]) & set(ids[:4])))
        summary = {
            "mode": "retrieval", "answerable": len(hits), "hit_at_4": round(100 * sum(hits) / len(hits), 1),
            "top_score_answerable": {"min": round(min(scores[True]), 2), "median": round(statistics.median(scores[True]), 2)},
            "top_score_unanswerable": {"max": round(max(scores[False]), 2), "median": round(statistics.median(scores[False]), 2)},
            "min_score_threshold": Config.RAG_MIN_SCORE,
        }
        print(json.dumps(summary, indent=2))
        return

    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for case in cases:
        if rows and delay:
            time.sleep(delay)
        retrieved, answer, context = run(case["question"])
        answerable = bool(case["expected"])
        if answer == ERROR:
            rows.append({"id": case["id"], "question": case["question"], "answerable": answerable, "error": True,
                         "hit_at_4": answerable and bool(set(case["expected"]) & set(retrieved[:4])),
                         "answered": False, "grounded": True, "retrieved": retrieved[:4]})
            continue
        rows.append({
            "id": case["id"], "question": case["question"], "answerable": answerable,
            "hit_at_4": answerable and bool(set(case["expected"]) & set(retrieved[:4])),
            "answered": answer is not None,
            "grounded": answer is None or grounded(answer, context),
            "retrieved": retrieved[:4],
        })
    errors = [r for r in rows if r.get("error")]
    rows_ok = [r for r in rows if not r.get("error")]
    answerable = [r for r in rows_ok if r["answerable"]]
    unanswerable = [r for r in rows_ok if not r["answerable"]]
    answered = [r for r in rows_ok if r["answered"]]
    hit_pool = [r for r in rows if r["answerable"]]
    summary = {
        "mode": args.mode, "cases": len(rows),
        "provider_errors": len(errors),
        "hit_at_4": round(100 * sum(r["hit_at_4"] for r in hit_pool) / len(hit_pool), 1),
        "answer_rate": round(100 * sum(r["answered"] for r in answerable) / len(answerable), 1),
        "correct_refusals": round(100 * sum(not r["answered"] for r in unanswerable) / len(unanswerable), 1),
        "groundedness": round(100 * sum(r["grounded"] for r in answered) / len(answered), 1) if answered else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for row in rows:
        if row.get("error"):
            print("ERROR", row["id"], row["question"])
        elif (row["answerable"] and not (row["hit_at_4"] and row["answered"])) or (not row["answerable"] and row["answered"]):
            print("MISS", row["id"], row["question"], row["retrieved"][:2], "answered" if row["answered"] else "declined")


if __name__ == "__main__":
    main()
