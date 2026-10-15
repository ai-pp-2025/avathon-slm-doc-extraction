# Receipt field extraction with a LoRA-tuned small language model

Avathon AI/ML Hiring Challenge: **Track B (SLM fine-tuning)** applied to **Scenario S2 (Gen AI for enterprise documents)**.

## Problem

Accounts-payable and expense teams receive receipts and invoices as scanned or photographed documents. OCR turns them into
noisy text lines; downstream systems need structured fields (`company`, `date`, `total`) to book the expense. Rules break on
layout variety (the total is on a different line per merchant, "TOTAL" appears several times, date formats differ), and
sending financial documents to a hosted LLM API is often not allowed or too costly per document. A small model that runs
in-house (or on-device) and is adapted to the task is the target.

**Task:** OCR text of one receipt in, JSON `{"company", "date", "total"}` out.

**Why fine-tune rather than prompt or retrieve** (numbers in `results/summary.md`):
- The failure modes are not missing knowledge but output discipline and field selection (which of several "TOTAL" lines,
  copy the date string exactly, stop after the JSON). That is a behaviour change, which weights learn better than a prompt.
- RAG adds nothing: the answer is in the one document already in context.
- Larger prompted models (for example a 7B instruct model in 4-bit) were not evaluated; the comparison is against the same
  0.6B model prompted zero-shot and 3-shot, and a regex baseline.

## Data

[`jsdnrs/ICDAR2019-SROIE`](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) (CC-BY-4.0): 626 train / 361 test
receipts with OCR lines and gold entities. The 626 train receipts are split into 551 train / 75 validation. Built by `python -m docextract.data` into `data/processed/` plus `data_card.json`.

| Decision | Why |
|---|---|
| `address` is **not** a task field | In this HF conversion the OCR lines are cut at the first comma, so 504 of 551 gold addresses are not recoverable from the text. Training on them would teach the model to guess from merchant identity. Quantified in `data_card.json`. |
| Validation split is grouped by merchant | SROIE repeats merchants and templates; a random split leaks. Val has no merchant that is in train. |
| Test is the official held-out split, with a `seen_merchant` flag | Results are reported overall and on **unseen merchants**, the honest measure of "learned the task vs memorised the template". |
| Long receipts are truncated in the middle (40% head / 60% tail) | Company is at the top, total at the bottom. p95 receipt is ~940 chars, so truncation rarely triggers. |

Known data limits: ~2% of gold labels do not appear verbatim in the OCR text (OCR typos), and `total` is written with or
without currency symbols. Scoring normalises both; the same grounding check on the gold labels is reported as `gold ungrounded`.

## Method

- Base model: **Qwen3-0.6B**, chosen for the free Colab T4 budget and on-device deployment. Other base models
  (Qwen3-1.7B, Qwen2.5-0.5B-Instruct, SmolLM2-1.7B-Instruct) were considered but not trained.
- Adaptation: **LoRA** r=16, alpha=32, dropout 0.05, all linear layers, lr 2e-4 cosine, 3 epochs, loss on the JSON only. Ablations (run with 100 training samples): rank 4/64, attention-only targets, QLoRA; data-size ablation: 100/300/551 samples.
- Prompt and target are built in one place (`prompts.py`), identical for training and inference. Greedy decoding, stop at `}`.
- Baselines: regex extractor, the same base model zero-shot and 3-shot.
- Guardrail: every predicted value is checked against the source text (`grounded`); ungrounded outputs are the ones to route to human review.

## Metrics

JSON validity, exact match per field after normalisation (`total` compared as a number), company token-F1, `all3 EM`
(company, date and total all correct), `all3 EM` on unseen merchants, ungrounded rate, ms per receipt, and training cost
(`runs/*/run_info.json`). A catastrophic-forgetting probe (`forgetting.py`, WikiText-2 perplexity with the adapter off vs on)
is implemented but was not run for the submitted results.

## Reproduce

Colab (recommended, T4): open `notebooks/fine_tune_on_colab.ipynb`, set `REPO_URL`, run all cells.

Local (Apple Silicon, CPU/MPS, no QLoRA):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps
pytest                                                     # unit tests
python -m docextract.data --out data/processed             # downloads ~500 MB
python -m docextract.infer --method regex --out preds/regex.jsonl
python -m docextract.evaluate preds/*.jsonl

# smoke test of the whole training path (a few minutes on an M1)
python -m docextract.train --out runs/smoke --max-train-samples 16 --epochs 1 --batch-size 1 --grad-accum 2 --grad-ckpt
python -m docextract.infer --adapter runs/smoke/adapter --limit 8 --out preds/smoke.jsonl
```

Dependencies are lower-bounded in `requirements.txt` (Python >= 3.10); the exact versions of the reported run are in
`results/env_freeze.txt`, written by the last notebook cell. All seeds are fixed (`--seed 42`), hyper-parameters are saved to `runs/<name>/train_args.json`.

## Layout

```
src/docextract/   data.py  prompts.py  metrics.py  regex_baseline.py  modeling.py
                  train.py  infer.py  evaluate.py  forgetting.py  export.py
notebooks/        fine_tune_on_colab.ipynb # the whole experiment, top to bottom
tests/            test_metrics.py
preds/            per-receipt predictions of every run (committed)
results/          summary.md, data_card.json, env_freeze.txt, latency/
runs/<name>/      train_args.json, run_info.json, log_history.json (weights are git-ignored)
WRITEUP.md        source of the 1-2 page PDF
```

## Results

Held-out test set (361 receipts). `all3 EM` = company, date and total all correct. Ablation rows ending `_n100` / `_n300` were trained on 100 / 300 examples (gradient accumulation 8); the final run `qwen3-0.6b_lora_r16` uses all 551 (accumulation 4). One seed per run.

| run | n | JSON ok | company EM | company F1 | date EM | total EM | all3 EM | all3 EM (unseen merchants) | ungrounded | gold ungrounded | ms/receipt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b_zeroshot | 361 | 100.0% | 66.8% | 87.2% | 63.7% | 80.9% | 37.1% | 32.0% | 8.0% | 2.8% | 902 |
| regex | 361 | 100.0% | 64.0% | 90.7% | 95.0% | 71.7% | 46.5% | 35.2% | 0.0% | 2.8% | 0 |
| qwen3-0.6b_3shot | 361 | 100.0% | 69.3% | 87.9% | 91.1% | 73.7% | 54.0% | 40.0% | 2.8% | 2.8% | 2350 |
| **qwen3-0.6b_lora_r16** | 361 | 100.0% | 93.4% | 96.4% | 98.3% | 95.8% | 88.1% | 73.6% | 0.0% | 2.8% | 789 |
| qwen3-0.6b_lora_r16_n300 | 361 | 100.0% | 92.2% | 96.1% | 97.0% | 94.5% | 85.0% | 68.8% | 0.3% | 2.8% | 789 |
| qwen3-0.6b_lora_r16_n100 | 361 | 100.0% | 83.9% | 91.0% | 96.4% | 91.4% | 74.5% | 52.0% | 1.1% | 2.8% | 808 |
| qwen3-0.6b_lora_r64_n100 | 361 | 100.0% | 84.2% | 91.0% | 95.6% | 91.4% | 74.2% | 58.4% | 0.8% | 2.8% | 795 |
| qwen3-0.6b_qlora_r16_n100 | 361 | 100.0% | 85.9% | 92.8% | 96.1% | 87.8% | 74.2% | 52.0% | 3.6% | 2.8% | 797 |
| qwen3-0.6b_lora_attn_r16_n100 | 361 | 100.0% | 81.7% | 91.5% | 94.7% | 90.9% | 71.5% | 50.4% | 3.9% | 2.8% | 794 |
| qwen3-0.6b_lora_r4_n100 | 361 | 100.0% | 81.4% | 90.8% | 93.9% | 87.5% | 70.1% | 46.4% | 3.0% | 2.8% | 792 |

ms/receipt is not comparable across rows (batch size 8, or 4 for the 3-shot run). Batch size 1 on 50 receipts: about 3.0 s per receipt (`results/latency/`).
Final run cost on a Colab T4: 35 min, 7.4 GB peak GPU memory, 10.1M trainable parameters (1.7%), 38.5 MB adapter.
Not run: catastrophic-forgetting probe, larger prompted model, other base models, full fine-tuning.
Interpretation and error analysis are in `WRITEUP.md`.