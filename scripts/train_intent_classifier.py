"""Fine-tune the XLM-R intent classifier on the expanded label set and calibrate it.

Warm-starts from the existing 15-label model: the encoder and the output rows of
existing labels are copied, so only the new labels start from scratch. After
training, a single temperature is fitted on the validation split so confidence
scores (and the Groq-verification threshold) are meaningful. Only the final
model is written, to keep disk use to one copy.

Usage (from the project root)::

    .venv/bin/python scripts/train_intent_classifier.py --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def label_order(base: Path, rows: list[dict]) -> list[str]:
    """Keep the base model's ids for existing labels and append new labels."""
    mapping = json.loads((base / "label_mapping.json").read_text(encoding="utf-8"))
    base_labels = [mapping["id2label"][str(i)] for i in range(len(mapping["id2label"]))]
    new_labels = sorted({r["label"] for r in rows} - set(base_labels))
    return base_labels + new_labels


def batches(rows: list[dict], size: int, shuffle: bool, rng: random.Random):
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), size):
        yield [rows[i] for i in order[start:start + size]]


@torch.no_grad()
def predict_logits(model, tokenizer, rows, device, max_len, batch_size=128) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, len(rows), batch_size):
        texts = [r["text"] for r in rows[start:start + batch_size]]
        enc = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(device)
        chunks.append(model(**enc).logits.float().cpu())
    return torch.cat(chunks)


def macro_f1(pred: torch.Tensor, gold: torch.Tensor, n_labels: int) -> float:
    scores = []
    for k in range(n_labels):
        tp = int(((pred == k) & (gold == k)).sum())
        fp = int(((pred == k) & (gold != k)).sum())
        fn = int(((pred != k) & (gold == k)).sum())
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def expected_calibration_error(probs: torch.Tensor, gold: torch.Tensor, bins: int = 15) -> float:
    confidence, pred = probs.max(dim=-1)
    correct = (pred == gold).float()
    ece = 0.0
    edges = torch.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.float().mean().item() * abs(confidence[mask].mean().item() - correct[mask].mean().item())
    return ece


def fit_temperature(logits: torch.Tensor, gold: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits / log_t.exp(), gold)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, default=ROOT / "models" / "xlmr_intent_classifier")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "xlmr_intent_classifier_v2")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train = read_jsonl(args.data / "train.jsonl")
    val = read_jsonl(args.data / "val.jsonl")
    labels = label_order(args.base, train)
    label2id = {label: i for i, label in enumerate(labels)}
    print(f"device={device} train={len(train)} val={len(val)} labels={len(labels)}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.base), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.base), local_files_only=True, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=label2id, ignore_mismatched_sizes=True,
    )
    # Copy the trained output rows for labels the base model already knew.
    with safe_open(str(args.base / "model.safetensors"), framework="pt") as weights:
        old_w = weights.get_tensor("classifier.out_proj.weight")
        old_b = weights.get_tensor("classifier.out_proj.bias")
    with torch.no_grad():
        model.classifier.out_proj.weight[: old_w.shape[0]] = old_w
        model.classifier.out_proj.bias[: old_b.shape[0]] = old_b
    model.to(device)

    counts = Counter(r["label"] for r in train)
    weights = torch.tensor([len(train) / (len(labels) * counts[label]) for label in labels], dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    val_gold = torch.tensor([label2id[r["label"]] for r in val])

    rng = random.Random(args.seed)
    best = {"macro_f1": -1.0}
    best_state = None
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(batches(train, args.batch_size, True, rng), start=1):
            enc = tokenizer([r["text"] for r in batch], truncation=True, padding=True,
                            max_length=args.max_len, return_tensors="pt").to(device)
            gold = torch.tensor([label2id[r["label"]] for r in batch], device=device)
            loss = loss_fn(model(**enc).logits, gold)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {running / 50:.4f} elapsed {time.time() - started:.0f}s", flush=True)
                running = 0.0
        logits = predict_logits(model, tokenizer, val, device, args.max_len)
        pred = logits.argmax(-1)
        metrics = {"epoch": epoch, "val_accuracy": float((pred == val_gold).float().mean()), "macro_f1": macro_f1(pred, val_gold, len(labels))}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["macro_f1"] > best["macro_f1"]:
            best = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    val_logits = predict_logits(model, tokenizer, val, device, args.max_len)
    temperature = fit_temperature(val_logits, val_gold)
    ece_before = expected_calibration_error(val_logits.softmax(-1), val_gold)
    ece_after = expected_calibration_error((val_logits / temperature).softmax(-1), val_gold)
    print(f"temperature={temperature:.3f} ECE {ece_before:.4f} -> {ece_after:.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.to("cpu").save_pretrained(str(args.out), safe_serialization=True)
    tokenizer.save_pretrained(str(args.out))
    (args.out / "label_mapping.json").write_text(json.dumps(
        {"id2label": {str(i): l for i, l in enumerate(labels)}, "label2id": label2id}, indent=2), encoding="utf-8")
    (args.out / "calibration.json").write_text(json.dumps({"temperature": temperature}, indent=2), encoding="utf-8")
    (args.out / "training_report.json").write_text(json.dumps({
        "base_model": str(args.base.relative_to(ROOT)), "labels": labels, "train_size": len(train), "val_size": len(val),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "history": history, "best": best,
        "temperature": temperature, "val_ece_before": ece_before, "val_ece_after": ece_after,
        "train_seconds": round(time.time() - started),
    }, indent=2), encoding="utf-8")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
