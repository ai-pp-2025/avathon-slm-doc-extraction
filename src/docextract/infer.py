"""Run extraction on a split with a regex baseline, a prompted base model, or a LoRA-tuned model.

Writes one JSON line per receipt: key, raw output, parsed prediction, gold, grounding flags, latency.
"""
import argparse
import json
import time
from pathlib import Path

from .data import load_jsonl, write_jsonl
from .metrics import FIELDS, grounded, parse_json
from .prompts import build_prompt, select_shots
from .regex_baseline import predict as regex_predict


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["llm", "regex"], default="llm")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter dir (omit for the base model)")
    ap.add_argument("--load-4bit", action="store_true", help="load the base model in 4-bit (CUDA only)")
    ap.add_argument("--shots", type=int, default=0, help="few-shot examples taken from the train split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--text-budget", type=int, default=800, help="must match training")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def run_llm(args, rows, shots):
    import torch
    from peft import PeftModel

    from .modeling import get_device, infer_dtype, load_model, load_tokenizer

    tok = load_tokenizer(args.model)
    tok.padding_side = "left"
    model = load_model(args.model, infer_dtype(), load_4bit=args.load_4bit)
    if not args.load_4bit:
        model.to(get_device())
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        if not args.load_4bit:
            model = model.merge_and_unload()
    model.eval()

    prompts = [build_prompt(r["text"], tok, args.text_budget, shots) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: -len(prompts[i]))
    results = [None] * len(rows)
    for s in range(0, len(order), args.batch_size):
        idx = order[s : s + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
                stop_strings=["}"],
                tokenizer=tok,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per_item = (time.time() - t0) / len(idx)
        raws = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for i, raw in zip(idx, raws):
            results[i] = (raw, per_item)
        print(f"{min(s + args.batch_size, len(order))}/{len(order)}", flush=True)
    return results


def main():
    args = parse_args()
    rows = load_jsonl(Path(args.data_dir) / f"{args.split}.jsonl")[: args.limit]

    if args.method == "regex":
        results = []
        for r in rows:
            t0 = time.time()
            pred = regex_predict(r["text"])
            results.append((json.dumps(pred), time.time() - t0))
    else:
        shots = select_shots(load_jsonl(Path(args.data_dir) / "train.jsonl"), args.shots)
        results = run_llm(args, rows, shots)

    records = []
    for r, (raw, latency) in zip(rows, results):
        pred = parse_json(raw)
        records.append(
            {
                "key": r["key"],
                "raw": raw,
                "pred": pred,
                "gold": r["entities"],
                "seen_merchant": r.get("seen_merchant", True),
                "grounded": grounded(pred, r["text"]) if pred else None,
                "gold_grounded": grounded(r["entities"], r["text"]),
                "latency_s": latency,
                "batch_size": 1 if args.method == "regex" else args.batch_size,
            }
        )
    write_jsonl(args.out, records)
    print(f"wrote {len(records)} predictions to {args.out}; fields={FIELDS}")


if __name__ == "__main__":
    main()
