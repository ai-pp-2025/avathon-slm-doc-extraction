"""LoRA / QLoRA supervised fine-tuning for receipt -> JSON extraction. Prompt tokens are masked from the loss."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments, set_seed

from .data import load_jsonl
from .modeling import load_model, load_tokenizer, train_dtype
from .prompts import build_prompt, target

ATTN_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", required=True, help="run directory, e.g. runs/qwen3-0.6b_lora_r16")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=None, help="default: 2 * rank")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", choices=["all", "attn"], default="all", help="all linear layers vs q/k/v/o only")
    ap.add_argument("--qlora", action="store_true", help="4-bit NF4 base model (CUDA + bitsandbytes only)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--text-budget", type=int, default=800, help="max receipt tokens in the prompt")
    ap.add_argument("--max-train-samples", type=int, default=None, help="learning-curve / smoke-test subset")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def encode(row, tok, budget):
    prompt_ids = tok(build_prompt(row["text"], tok, budget), add_special_tokens=False)["input_ids"]
    comp_ids = tok(target(row["entities"]), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return {"input_ids": prompt_ids + comp_ids, "labels": [-100] * len(prompt_ids) + comp_ids}


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), m), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), m), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), m), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        attn[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


def main():
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2))

    tok = load_tokenizer(args.model)
    train_rows = load_jsonl(Path(args.data_dir) / "train.jsonl")
    val_rows = load_jsonl(Path(args.data_dir) / "val.jsonl")
    if args.max_train_samples:
        random.Random(args.seed).shuffle(train_rows)
        train_rows = train_rows[: args.max_train_samples]
    train_ds = [encode(r, tok, args.text_budget) for r in train_rows]
    val_ds = [encode(r, tok, args.text_budget) for r in val_rows]
    print(f"train={len(train_ds)} val={len(val_ds)} max_len={max(len(x['input_ids']) for x in train_ds)}")

    dtype = train_dtype()
    model = load_model(args.model, dtype, load_4bit=args.qlora)
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_ckpt)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha or 2 * args.rank,
        lora_dropout=args.dropout,
        target_modules="all-linear" if args.targets == "all" else ATTN_ONLY,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    cuda = torch.cuda.is_available()
    bf16 = cuda and dtype == torch.bfloat16
    total_steps = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum) * args.epochs)
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * total_steps)),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        fp16=cuda and not bf16,
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.grad_ckpt else None,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )

    t0 = time.time()
    result = trainer.train()
    final_eval = trainer.evaluate()

    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info = {
        "model": args.model,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "adapter_mb": round(sum(f.stat().st_size for f in adapter_dir.glob("*.safetensors")) / 2**20, 1),
        "train_seconds": round(time.time() - t0, 1),
        "train_loss": result.metrics.get("train_loss"),
        "best_val_loss": final_eval.get("eval_loss"),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
