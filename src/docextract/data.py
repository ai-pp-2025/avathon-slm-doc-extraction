"""Prepare SROIE receipts as (OCR text -> 4 fields) records with leakage-aware splits.

Source: HF `jsdnrs/ICDAR2019-SROIE` (CC-BY-4.0). We use the provided OCR lines (`words`) as the document text
and 3 of the 4 gold entities (company, date, total). Images/boxes are dropped: the task is text -> JSON.

`address` is excluded on purpose. In this HF conversion the OCR lines are cut at the first comma, so most gold
addresses are not recoverable from the text; the data card quantifies this.
"""
import argparse
import json
import random
import statistics
from pathlib import Path

from .metrics import FIELDS, grounded, norm_text

DATASET_ID = "jsdnrs/ICDAR2019-SROIE"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _convert(split):
    return [
        {
            "key": ex["key"],
            "text": "\n".join(ex["words"]),
            "entities": {f: (ex["entities"][f] or "").strip() for f in (*FIELDS, "address")},
        }
        for ex in split
    ]


def split_by_merchant(rows, val_frac, seed):
    """Hold out whole merchants for validation: SROIE repeats merchants/templates, a random split would leak."""
    groups = {}
    for r in rows:
        groups.setdefault(norm_text(r["entities"]["company"]), []).append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target_n = int(len(rows) * val_frac)
    train, val = [], []
    for k in keys:
        (val if len(val) < target_n else train).extend(groups[k])
    return train, val


def prepare(out_dir, val_frac=0.1, seed=42):
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID).remove_columns(["image", "bboxes"])
    train_all, test = _convert(ds["train"]), _convert(ds["test"])
    train, val = split_by_merchant(train_all, val_frac, seed)

    seen = {norm_text(r["entities"]["company"]) for r in train}
    for r in test:
        r["seen_merchant"] = norm_text(r["entities"]["company"]) in seen

    address_missing = sum(
        norm_text(r["entities"]["address"]).replace(" ", "") not in norm_text(r["text"]).replace(" ", "")
        for r in train
    )
    for r in (*train, *val, *test):
        r["entities"] = {f: r["entities"][f] for f in FIELDS}

    out = Path(out_dir)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        write_jsonl(out / f"{name}.jsonl", rows)

    lens = sorted(len(r["text"]) for r in train)
    card = {
        "source": DATASET_ID,
        "license": "CC-BY-4.0 (ICDAR2019 SROIE)",
        "seed": seed,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "test_unseen_merchant": sum(not r["seen_merchant"] for r in test),
        "val_split": "grouped by normalised merchant name (no merchant appears in both train and val)",
        "chars_per_receipt": {"median": statistics.median(lens), "p95": lens[int(0.95 * len(lens))]},
        "empty_gold_in_train": {f: sum(not r["entities"][f] for r in train) for f in FIELDS},
        "gold_not_found_in_text_train": {
            f: sum(not grounded(r["entities"], r["text"])[f] for r in train) for f in FIELDS
        },
        "address_not_found_in_text_train": address_missing,
        "note": "address dropped from the task: OCR lines are truncated at commas in this dataset conversion",
    }
    (out / "data_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    prepare(a.out, a.val_frac, a.seed)


if __name__ == "__main__":
    main()
