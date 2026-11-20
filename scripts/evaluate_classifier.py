"""Compare intent classifiers offline (no Groq) on held-out and hand-labelled data.

Reports, per model and per test set: accuracy, macro-F1, accuracy by language,
calibration error, how often the model is confidently wrong (above the Groq
threshold, so Groq never gets a chance to fix it), and how often it would defer
to Groq. A 15-label model cannot predict the new labels, so results are also
given on the subset of rows whose label it knows.

Usage (from the project root)::

    .venv/bin/python scripts/evaluate_classifier.py \\
        --model models/xlmr_intent_classifier --model models/xlmr_intent_classifier_v2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def load_sets(data_dir: Path, cases: Path) -> dict[str, list[dict]]:
    sets = {}
    for name in ("test", "test_original"):
        path = data_dir / f"{name}.jsonl"
        if path.is_file():
            sets[name] = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    hand = []
    for line in cases.read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = json.loads(line)
            gold = case["intent"] if isinstance(case["intent"], list) else [case["intent"]]
            hand.append({"text": case["turns"][0], "gold": gold, "lang": case["lang"]})
    sets["hand_eval_326"] = hand
    for rows in sets.values():
        for row in rows:
            row.setdefault("gold", [row.get("label")])
    return sets


@torch.no_grad()
def score(model_dir: Path, rows: list[dict], threshold: float) -> dict:
    mapping = json.loads((model_dir / "label_mapping.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    known = set(id2label.values())
    calibration = model_dir / "calibration.json"
    temperature = json.loads(calibration.read_text())["temperature"] if calibration.is_file() else 1.0
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()

    preds, confs = [], []
    for start in range(0, len(rows), 128):
        texts = [r["text"] for r in rows[start:start + 128]]
        enc = tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors="pt").to(device)
        probs = (model(**enc).logits.float().cpu() / temperature).softmax(-1)
        c, p = probs.max(-1)
        preds += [id2label[int(i)] for i in p]
        confs += [float(x) * 100 for x in c]

    def summary(subset_idx: list[int]) -> dict:
        n = len(subset_idx)
        if not n:
            return {"n": 0}
        correct = [preds[i] in rows[i]["gold"] for i in subset_idx]
        by_lang = defaultdict(list)
        for i, ok in zip(subset_idx, correct):
            by_lang[rows[i].get("lang", "?")].append(ok)
        labels = sorted({g for i in subset_idx for g in rows[i]["gold"][:1]})
        f1s = []
        for label in labels:
            tp = sum(1 for i in subset_idx if preds[i] == label and rows[i]["gold"][0] == label)
            fp = sum(1 for i in subset_idx if preds[i] == label and rows[i]["gold"][0] != label)
            fn = sum(1 for i in subset_idx if preds[i] != label and rows[i]["gold"][0] == label)
            if tp + fp + fn:
                f1s.append(2 * tp / (2 * tp + fp + fn))
        confident_wrong = sum(1 for i, ok in zip(subset_idx, correct) if not ok and confs[i] > threshold)
        bins = defaultdict(list)
        for i, ok in zip(subset_idx, correct):
            bins[min(int(confs[i] // (100 / 15)), 14)].append((confs[i] / 100, ok))
        ece = sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in bins.values())
        return {
            "n": n,
            "accuracy": round(100 * sum(correct) / n, 1),
            "macro_f1": round(100 * sum(f1s) / len(f1s), 1) if f1s else None,
            "by_lang": {k: round(100 * sum(v) / len(v), 1) for k, v in sorted(by_lang.items())},
            "confidently_wrong_pct": round(100 * confident_wrong / n, 1),
            "sent_to_groq_pct": round(100 * sum(1 for i in subset_idx if confs[i] <= threshold) / n, 1),
            "ece": round(ece, 4),
        }

    all_idx = list(range(len(rows)))
    known_idx = [i for i in all_idx if set(rows[i]["gold"]) & known]
    errors = [{"text": rows[i]["text"], "gold": rows[i]["gold"], "pred": preds[i], "conf": round(confs[i], 1)}
              for i in all_idx if preds[i] not in rows[i]["gold"]]
    return {"all": summary(all_idx), "known_labels_only": summary(known_idx), "temperature": temperature, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "training")
    parser.add_argument("--cases", type=Path, default=ROOT / "evals" / "cases.jsonl")
    parser.add_argument("--threshold", type=float, default=95.0)
    parser.add_argument("--report", type=Path, default=ROOT / "evals" / "reports" / "classifier_comparison.json")
    args = parser.parse_args()

    sets = load_sets(args.data, args.cases)
    report = {}
    for model_dir in args.model:
        name = model_dir.name
        report[name] = {}
        for set_name, rows in sets.items():
            result = score(model_dir, rows, args.threshold)
            report[name][set_name] = result
            a, k = result["all"], result["known_labels_only"]
            print(f"{name:<28} {set_name:<14} n={a['n']:<5} acc={a['accuracy']:5.1f} macroF1={a['macro_f1']} "
                  f"known-label acc={k.get('accuracy')} confident-wrong={a['confidently_wrong_pct']}% "
                  f"to-Groq={a['sent_to_groq_pct']}% ECE={a['ece']} by_lang={a['by_lang']}", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report: {args.report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
