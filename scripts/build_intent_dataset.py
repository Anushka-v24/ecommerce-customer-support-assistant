"""Build a balanced English/Hindi/Hinglish intent dataset for the 20-label classifier.

Two sources:
1. Hand-written templates in ``scripts/intent_templates/{en,hi,hinglish}.json``,
   filled with slot values and lightly augmented. Every template is assigned to
   exactly one split (train/val/test) by a stable hash, so test sentences never
   share a template with training sentences and accuracy is not inflated.
2. The original English training CSV (optional), sampled per intent so the
   retrained model keeps what it already knew. It is used for train/val and a
   separate ``test_original.jsonl`` retention check.

Usage (from the project root)::

    .venv/bin/python scripts/build_intent_dataset.py \\
        --original data/training/source/original_153k.csv --per-cell 200
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "scripts" / "intent_templates"
OUTPUT_DIR = ROOT / "data" / "training"
LANGS = ("en", "hi", "hinglish")
SPLIT_WEIGHTS = (("train", 70), ("val", 15), ("test", 15))
SLOT = re.compile(r"\{([a-z_0-9]+)\}")

PRODUCTS = ["Laptop Pro", "Galaxy S21", "Noise-Cancelling Headphones", "iPhone 15", "running shoes",
            "smartwatch", "air fryer", "Bluetooth speaker", "office chair", "cotton kurta", "gaming mouse", "4K TV"]
SLOT_VALUES = {
    "en": {
        "product": PRODUCTS,
        "city": ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune", "Hyderabad", "Kolkata", "Jaipur", "Lucknow", "Patna"],
        "payment_method": ["UPI", "credit card", "debit card", "PayPal", "net banking", "cash on delivery", "EMI"],
        "amount": ["₹499", "₹1,200", "₹15,000", "50k", "20000 rupees", "Rs 999", "$100"],
        "size": ["M", "L", "XL", "size 8", "size 10", "32", "small"],
        "color": ["black", "blue", "red", "white", "grey", "green"],
        "category": ["laptops", "phones", "headphones", "shoes", "kitchen appliances", "watches", "TVs"],
    },
    "hi": {
        "product": PRODUCTS + ["जूते", "कुर्ता", "मिक्सर"],
        "city": ["दिल्ली", "मुंबई", "बैंगलोर", "चेन्नई", "पुणे", "हैदराबाद", "कोलकाता", "जयपुर", "लखनऊ", "पटना"],
        "payment_method": ["UPI", "क्रेडिट कार्ड", "डेबिट कार्ड", "नेट बैंकिंग", "कैश ऑन डिलीवरी", "EMI"],
        "amount": ["₹499", "₹1,200", "₹15,000", "50 हज़ार", "20000 रुपये", "999 रुपये"],
        "size": ["M", "L", "XL", "8 नंबर", "10 नंबर", "छोटा साइज़", "बड़ा साइज़"],
        "color": ["काला", "नीला", "लाल", "सफ़ेद", "हरा", "ग्रे"],
        "category": ["लैपटॉप", "फ़ोन", "हेडफ़ोन", "जूते", "घड़ियाँ", "टीवी"],
    },
    "hinglish": {
        "product": PRODUCTS + ["joote", "kurta", "mixer grinder"],
        "city": ["Delhi", "Dilli", "Mumbai", "Bangalore", "Chennai", "Pune", "Hyderabad", "Kolkata", "Lucknow", "Patna"],
        "payment_method": ["UPI", "credit card", "debit card", "GPay", "PhonePe", "COD", "cash on delivery", "EMI"],
        "amount": ["₹499", "1200 rs", "15k", "50 hazaar", "20000 rupaye", "999"],
        "size": ["M", "L", "XL", "8 number", "10 number", "chhota size", "bada size"],
        "color": ["black", "kaala", "blue", "neela", "red", "laal", "white"],
        "category": ["laptop", "phone", "mobile", "headphone", "joote", "watch", "TV"],
    },
}
FILLERS = {
    "en": ["pls", "please", "hey", "hi,", "urgent", "asap", "thanks"],
    "hi": ["कृपया", "जी", "भाई", "जल्दी"],
    "hinglish": ["pls", "yaar", "bhai", "ji", "plz", "jaldi"],
}


def stable_bucket(key: str) -> str:
    value = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 100
    upper = 0
    for name, weight in SPLIT_WEIGHTS:
        upper += weight
        if value < upper:
            return name
    return SPLIT_WEIGHTS[-1][0]


def order_id(rng: random.Random) -> str:
    return f"ORD-{rng.randint(1000, 9999)}"


def fill(template: str, lang: str, rng: random.Random) -> str:
    values = SLOT_VALUES[lang]

    def replace(match: re.Match[str]) -> str:
        slot = match.group(1)
        if slot == "order_id":
            return order_id(rng)
        if slot == "product2":
            return rng.choice(values["product"])
        return rng.choice(values[slot]) if slot in values else match.group(0)

    return SLOT.sub(replace, template)


def typo(text: str, rng: random.Random) -> str:
    """Swap two adjacent letters inside one longer Latin word."""
    words = text.split()
    candidates = [i for i, w in enumerate(words) if len(w) > 4 and w.isascii() and w.isalpha()]
    if not candidates:
        return text
    i = rng.choice(candidates)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def augment(text: str, lang: str, rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.25:
        text = text.lower().rstrip("?.!")
    elif roll < 0.40 and lang != "hi":
        text = typo(text, rng)
    elif roll < 0.50:
        filler = rng.choice(FILLERS[lang])
        text = f"{filler} {text}" if rng.random() < 0.5 else f"{text} {filler}"
    return " ".join(text.split())


def generate_from_templates(per_cell: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for lang in LANGS:
        path = TEMPLATE_DIR / f"{lang}.json"
        templates_by_intent = json.loads(path.read_text(encoding="utf-8"))
        for intent, templates in sorted(templates_by_intent.items()):
            unique = list(dict.fromkeys(t.strip() for t in templates if t.strip()))
            by_split: dict[str, list[str]] = defaultdict(list)
            for template in unique:
                by_split[stable_bucket(f"{lang}|{intent}|{template}")].append(template)
            for split, weight in SPLIT_WEIGHTS:
                pool = by_split.get(split) or []
                if not pool:
                    continue
                target = max(1, per_cell * weight // 100)
                seen: set[str] = set()
                attempts = 0
                while len(seen) < target and attempts < target * 20:
                    attempts += 1
                    template = rng.choice(pool)
                    text = augment(fill(template, lang, rng), lang, rng)
                    if text in seen:
                        continue
                    seen.add(text)
                    rows.append({
                        "text": text, "label": intent, "lang": lang, "split": split,
                        "source": "template", "template_id": hashlib.sha1(f"{lang}|{intent}|{template}".encode()).hexdigest()[:12],
                    })
    return rows


def sample_original(path: Path, per_label: dict[str, int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_label: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                text = re.sub(r"\{\{\s*order number\s*\}\}", lambda _: order_id(rng), row[0].strip(), flags=re.I)
                text = re.sub(r"\{\{[^}]*\}\}", "", text)
                by_label[row[1].strip()].append(" ".join(text.split()))
    rows = []
    for label, texts in sorted(by_label.items()):
        texts = list(dict.fromkeys(texts))
        rng.shuffle(texts)
        start = 0
        for split, count in per_label.items():
            for text in texts[start:start + count]:
                rows.append({"text": text, "label": label, "lang": "en", "split": split, "source": "original", "template_id": None})
            start += count
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-cell", type=int, default=200, help="Generated examples per (intent, language).")
    parser.add_argument("--original", type=Path, help="Original English CSV (instructions,intents).")
    parser.add_argument("--original-train", type=int, default=300)
    parser.add_argument("--original-val", type=int, default=40)
    parser.add_argument("--original-test", type=int, default=60)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    generated = generate_from_templates(args.per_cell, args.seed)
    original = []
    if args.original:
        original = sample_original(
            args.original,
            {"train": args.original_train, "val": args.original_val, "test_original": args.original_test},
            args.seed,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict]] = defaultdict(list)
    for row in generated + original:
        splits[row["split"]].append(row)
    rng = random.Random(args.seed)
    for split, rows in splits.items():
        rng.shuffle(rows)
        with (OUTPUT_DIR / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_templates = {r["template_id"] for r in splits["train"] if r["template_id"]}
    leaked = sum(1 for r in splits["test"] if r["template_id"] in train_templates)
    stats = {
        "generated_total": len(generated),
        "original_total": len(original),
        "split_sizes": {k: len(v) for k, v in sorted(splits.items())},
        "generated_by_label": dict(sorted(Counter(r["label"] for r in generated).items())),
        "generated_by_lang": dict(Counter(r["lang"] for r in generated)),
        "test_rows_sharing_a_train_template": leaked,
    }
    (OUTPUT_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
