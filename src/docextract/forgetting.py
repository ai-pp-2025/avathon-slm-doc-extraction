"""Catastrophic-forgetting probe: perplexity on general text (WikiText-2 test) with the adapter off vs on."""
import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel

from .modeling import get_device, infer_dtype, load_model, load_tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n-docs", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_dataset

    texts = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t.split()) > 50]
    texts = texts[: args.n_docs]

    tok = load_tokenizer(args.model)
    model = load_model(args.model, infer_dtype()).to(get_device())
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    @torch.no_grad()
    def perplexity():
        nll, n_tok = 0.0, 0
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=args.max_len).input_ids.to(model.device)
            n = ids.shape[1] - 1
            nll += model(ids, labels=ids).loss.item() * n
            n_tok += n
        return math.exp(nll / n_tok)

    with model.disable_adapter():
        base = perplexity()
    tuned = perplexity()
    res = {"model": args.model, "adapter": args.adapter, "n_docs": len(texts), "ppl_base": base, "ppl_tuned": tuned,
           "ppl_increase_pct": 100 * (tuned / base - 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
