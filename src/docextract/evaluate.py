"""Score prediction files and print a comparison table (markdown).

Metrics: JSON validity, per-field exact match after normalisation (company also token F1), `all3` = company+date+total
all correct, the same on merchants never seen in training, and the share of predictions with a value that does not
occur in the receipt text (hallucination proxy; `gold ungrounded` is the same check on the gold labels = noise floor).
"""
import argparse
from pathlib import Path
from statistics import mean

from .data import load_jsonl
from .metrics import FIELDS, score_example


def summarize(rows):
    scores = [score_example(r["pred"], r["gold"]) for r in rows]
    all3 = [all(s[f]["em"] for f in FIELDS) for s in scores]
    unseen = [c for c, r in zip(all3, rows) if not r["seen_merchant"]]
    with_pred = [r for r in rows if r["grounded"]]
    return {
        "n": len(rows),
        "json_ok": mean(r["pred"] is not None for r in rows),
        "company_em": mean(s["company"]["em"] for s in scores),
        "company_f1": mean(s["company"]["f1"] for s in scores),
        "date_em": mean(s["date"]["em"] for s in scores),
        "total_em": mean(s["total"]["em"] for s in scores),
        "all3_em": mean(all3),
        "all3_em_unseen": mean(unseen) if unseen else float("nan"),
        "n_unseen": len(unseen),
        "ungrounded": mean(not all(r["grounded"].values()) for r in with_pred) if with_pred else float("nan"),
        "gold_ungrounded": mean(not all(r["gold_grounded"].values()) for r in rows),
        "ms_per_receipt": 1000 * mean(r["latency_s"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--out", default=None, help="also write the table to this markdown file")
    args = ap.parse_args()

    cols = [
        ("n", "n", "{:d}"),
        ("json_ok", "JSON ok", "{:.1%}"),
        ("company_em", "company EM", "{:.1%}"),
        ("company_f1", "company F1", "{:.1%}"),
        ("date_em", "date EM", "{:.1%}"),
        ("total_em", "total EM", "{:.1%}"),
        ("all3_em", "all3 EM", "{:.1%}"),
        ("all3_em_unseen", "all3 EM (unseen merchants)", "{:.1%}"),
        ("ungrounded", "ungrounded", "{:.1%}"),
        ("gold_ungrounded", "gold ungrounded", "{:.1%}"),
        ("ms_per_receipt", "ms/receipt", "{:.0f}"),
    ]
    lines = ["| run | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for p in args.preds:
        s = summarize(load_jsonl(p))
        lines.append(f"| {Path(p).stem} | " + " | ".join(fmt.format(s[k]) for k, _, fmt in cols) + " |")
    table = "\n".join(lines)
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n")


if __name__ == "__main__":
    main()
