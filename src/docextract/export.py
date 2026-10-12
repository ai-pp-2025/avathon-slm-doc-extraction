"""Merge a LoRA adapter into its base model and save a standalone HF checkpoint (input for GGUF / LiteRT / MLX export)."""
import argparse

from peft import PeftModel

from .modeling import infer_dtype, load_model, load_tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = PeftModel.from_pretrained(load_model(args.model, infer_dtype()), args.adapter).merge_and_unload()
    model.save_pretrained(args.out)
    load_tokenizer(args.model).save_pretrained(args.out)
    print(f"saved merged model to {args.out}")


if __name__ == "__main__":
    main()
