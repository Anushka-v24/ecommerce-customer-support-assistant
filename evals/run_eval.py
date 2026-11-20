"""Run the labelled cases in ``evals/cases.jsonl`` through the real graph and score them.

Usage (from the project root)::

    .venv/bin/python -m evals.run_eval
    .venv/bin/python -m evals.run_eval --lang hinglish --tag dissatisfaction
    .venv/bin/python -m evals.run_eval --compare evals/reports/<earlier>.json

Each case runs in its own in-memory conversation thread with the real XLM-R model
and, below the confidence threshold, the real Groq verifier. Business data uses the
JSON fixtures by default so runs never write demo records into PostgreSQL.

Case fields:
    id, lang, turns   one or more customer messages sent in the same session
    intent            expected final intent of the first turn (string or list of acceptable labels)
    route             expected ``selected_route`` after the last turn (string or list);
                      a trailing ``*`` matches by prefix, e.g. ``issue_support:*``
    contains          optional substrings that must all appear in the final response
    tags              optional labels for filtering and per-tag reporting
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = EVAL_DIR / "cases.jsonl"
REPORTS_DIR = EVAL_DIR / "reports"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            for field in ("id", "lang", "turns", "intent", "route"):
                if field not in case:
                    raise ValueError(f"{path.name}:{line_number} is missing '{field}'")
            cases.append(case)
    ids = [case["id"] for case in cases]
    duplicates = [case_id for case_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate case ids: {duplicates}")
    return cases


def _as_list(value: Any) -> list[str]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def intent_matches(expected: Any, final_intent: str | None, verifier_status: str | None) -> bool:
    accepted = _as_list(expected)
    if "out_of_scope" in accepted and verifier_status == "OUT_OF_SCOPE":
        return True
    return final_intent in accepted


def route_matches(expected: Any, route: str | None) -> bool:
    if not route:
        return False
    for pattern in _as_list(expected):
        if pattern.endswith("*") and route.startswith(pattern[:-1]):
            return True
        if route == pattern:
            return True
    return False


def run_case(graph: Any, case: dict[str, Any]) -> dict[str, Any]:
    from app.graph.workflow import invoke_graph

    session_id = f"eval-{case['id']}-{uuid4().hex[:8]}"
    first: dict[str, Any] | None = None
    last: dict[str, Any] = {}
    started = time.perf_counter()
    for turn in case["turns"]:
        last = invoke_graph(turn, session_id, graph=graph)
        if first is None:
            first = last
    elapsed_ms = (time.perf_counter() - started) * 1000
    first = first or {}

    response = last.get("response") or ""
    contains = case.get("contains") or []
    result = {
        "id": case["id"],
        "lang": case["lang"],
        "tags": case.get("tags", []),
        "turns": case["turns"],
        "expected_intent": case["intent"],
        "expected_route": case["route"],
        "predicted_intent": first.get("predicted_intent"),
        "confidence": first.get("confidence_score"),
        "final_intent": first.get("final_intent"),
        "verifier_status": first.get("verifier_status"),
        "used_groq": first.get("classifier_routing") == "LLM_VERIFICATION",
        "route": last.get("selected_route"),
        "response": response,
        "latency_ms": round(elapsed_ms, 1),
    }
    result["xlmr_ok"] = intent_matches(case["intent"], result["predicted_intent"], None)
    result["intent_ok"] = intent_matches(case["intent"], result["final_intent"], result["verifier_status"])
    result["route_ok"] = route_matches(case["route"], result["route"])
    result["response_ok"] = all(text in response for text in contains) if contains else None
    result["error"] = result["verifier_status"] in {"UNVERIFIED", "CLASSIFICATION_ERROR"}
    result["passed"] = result["intent_ok"] and result["route_ok"] and result["response_ok"] is not False
    return result


def _pct(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 1) if denominator else None


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    with_contains = [r for r in results if r["response_ok"] is not None]
    groq = [r for r in results if r["used_groq"]]
    changed = [r for r in groq if r["final_intent"] != r["predicted_intent"]]
    latencies = sorted(r["latency_ms"] for r in results)

    def breakdown(key) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in results:
            for group in key(r):
                groups[group].append(r)
        return {
            name: {
                "n": len(items),
                "intent_acc": _pct(sum(r["intent_ok"] for r in items), len(items)),
                "route_acc": _pct(sum(r["route_ok"] for r in items), len(items)),
                "pass_rate": _pct(sum(r["passed"] for r in items), len(items)),
            }
            for name, items in sorted(groups.items())
        }

    confusions = Counter(
        (str(_as_list(r["expected_intent"])[0]), str(r["final_intent"] or r["verifier_status"]))
        for r in results if not r["intent_ok"]
    )
    return {
        "cases": total,
        "pass_rate": _pct(sum(r["passed"] for r in results), total),
        "intent_acc": _pct(sum(r["intent_ok"] for r in results), total),
        "xlmr_only_acc": _pct(sum(r["xlmr_ok"] for r in results), total),
        "route_acc": _pct(sum(r["route_ok"] for r in results), total),
        "response_acc": _pct(sum(r["response_ok"] for r in with_contains), len(with_contains)),
        "groq_rate": _pct(len(groq), total),
        "groq_changed": len(changed),
        "groq_fixed": sum(1 for r in changed if r["intent_ok"] and not r["xlmr_ok"]),
        "groq_broke": sum(1 for r in changed if r["xlmr_ok"] and not r["intent_ok"]),
        "verification_errors": sum(r["error"] for r in results),
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None,
        "by_intent": breakdown(lambda r: [str(_as_list(r["expected_intent"])[0])]),
        "by_lang": breakdown(lambda r: [r["lang"]]),
        "by_tag": breakdown(lambda r: r["tags"]),
        "top_confusions": [
            {"expected": expected, "got": got, "count": count}
            for (expected, got), count in confusions.most_common(10)
        ],
    }


def _fmt(value: Any) -> str:
    return "  n/a" if value is None else f"{value:5.1f}"


def _delta(current: Any, previous: Any) -> str:
    if current is None or previous is None:
        return ""
    diff = current - previous
    return f"  ({'+' if diff >= 0 else ''}{diff:.1f})" if abs(diff) >= 0.05 else "  (=)"


def print_report(summary: dict[str, Any], results: list[dict[str, Any]], previous: dict[str, Any] | None) -> None:
    prev = previous or {}
    print(f"\n=== Evaluation: {summary['cases']} cases ===")
    for key, label in (
        ("pass_rate", "Overall pass rate"),
        ("intent_acc", "Intent accuracy (final)"),
        ("xlmr_only_acc", "Intent accuracy (XLM-R only)"),
        ("route_acc", "Route accuracy"),
        ("response_acc", "Response checks"),
        ("groq_rate", "Sent to Groq"),
    ):
        print(f"  {label:<30}{_fmt(summary[key])}%{_delta(summary[key], prev.get(key))}")
    print(
        f"  Groq changed {summary['groq_changed']} labels: fixed {summary['groq_fixed']}, "
        f"broke {summary['groq_broke']}; verification errors: {summary['verification_errors']}"
    )
    print(f"  Latency p50 {summary['latency_p50_ms']} ms, p95 {summary['latency_p95_ms']} ms")

    for title, key in (("By intent", "by_intent"), ("By language", "by_lang")):
        print(f"\n  {title:<24}   n  intent  route   pass")
        for name, row in summary[key].items():
            prev_pass = (prev.get(key) or {}).get(name, {}).get("pass_rate")
            print(
                f"  {name:<24}{row['n']:4d}  {_fmt(row['intent_acc'])}  {_fmt(row['route_acc'])}  "
                f"{_fmt(row['pass_rate'])}{_delta(row['pass_rate'], prev_pass)}"
            )

    if summary["top_confusions"]:
        print("\n  Top intent confusions (expected -> got)")
        for item in summary["top_confusions"]:
            print(f"    {item['count']:3d}  {item['expected']} -> {item['got']}")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n  Failures ({len(failures)})")
        for r in failures:
            problems = []
            if not r["intent_ok"]:
                problems.append(f"intent {r['final_intent'] or r['verifier_status']} (xlmr {r['predicted_intent']} {r['confidence']}%)")
            if not r["route_ok"]:
                problems.append(f"route {r['route']} != {r['expected_route']}")
            if r["response_ok"] is False:
                problems.append("response check failed")
            print(f"    [{r['id']}] {' | '.join(r['turns'])}\n        {'; '.join(problems)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--lang", action="append", help="Only run cases in this language (repeatable).")
    parser.add_argument("--tag", action="append", help="Only run cases with this tag (repeatable).")
    parser.add_argument("--id", action="append", dest="ids", help="Only run this case id (repeatable).")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between cases (Groq rate limits).")
    parser.add_argument("--backend", choices=("json", "postgres"), default="json", help="Business data backend.")
    parser.add_argument("--compare", type=Path, help="Earlier report to show deltas against.")
    parser.add_argument("--fail-under", type=float, help="Exit non-zero if the overall pass rate is below this percentage.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.ERROR)
    cases = load_cases(args.cases)
    if args.lang:
        cases = [c for c in cases if c["lang"] in args.lang]
    if args.tag:
        cases = [c for c in cases if set(args.tag) & set(c.get("tags", []))]
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No cases selected.")
        return 1

    from langgraph.checkpoint.memory import InMemorySaver

    from app.config import Config
    from app.graph.workflow import build_graph

    Config.BUSINESS_DATA_BACKEND = args.backend
    graph = build_graph(InMemorySaver())

    results = []
    for index, case in enumerate(cases, start=1):
        result = run_case(graph, case)
        results.append(result)
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"[{index:3d}/{len(cases)}] {mark} {case['id']:<10} {result['latency_ms']:7.0f} ms", file=sys.stderr)
        if args.delay and index < len(cases):
            time.sleep(args.delay)

    summary = summarise(results)
    previous = None
    if args.compare:
        previous = json.loads(args.compare.read_text(encoding="utf-8"))["summary"]
    print_report(summary, results, previous)

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cases_file": str(args.cases),
        "filters": {"lang": args.lang, "tag": args.tag, "ids": args.ids, "limit": args.limit},
        "summary": summary,
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to {report_path.relative_to(EVAL_DIR.parent)}")

    if args.fail_under is not None and (summary["pass_rate"] or 0) < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
