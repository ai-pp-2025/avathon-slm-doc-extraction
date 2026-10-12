# Receipt field extraction with a LoRA-tuned small language model

Avathon AI/ML Hiring Challenge: **Track B (SLM fine-tuning)** applied to **Scenario S2 (Gen AI for enterprise documents)**.

## Problem

Accounts-payable and expense teams receive receipts and invoices as scanned or photographed documents. OCR turns them into
noisy text lines; downstream systems need structured fields (`company`, `date`, `total`) to book the expense. Rules break on
layout variety (the total is on a different line per merchant, "TOTAL" appears several times, date formats differ), and
sending financial documents to a hosted LLM API is often not allowed or too costly per document. A small model that runs
in-house (or on-device) and is adapted to the task is the target.

**Task:** OCR text of one receipt in, JSON `{"company", "date", "total"}` out.

**Why fine-tune rather than prompt or retrieve** (to be backed by the numbers in `results/summary.md`):
- The failure modes are not missing knowledge but output discipline and field selection (which of several "TOTAL" lines,
  copy the date string exactly, stop after the JSON). That is a behaviour change, which weights learn better than a prompt.
- RAG adds nothing: the answer is in the one document already in context.
- A prompted 7B model is the strongest alternative; it is evaluated (`qwen2.5-7b-instruct_3shot_4bit`) and compared on
  accuracy, latency and size.

## Data

[`jsdnrs/ICDAR2019-SROIE`](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) (CC-BY-4.0): 626 train / 361 test
receipts with OCR lines and gold entities. Built by `python -m docextract.data` into `data/processed/` plus `data_card.json`.

| Decision | Why |
|---|---|
| `address` is **not** a task field | In this HF conversion the OCR lines are cut at the first comma, so 504 of 551 gold addresses are not recoverable from the text. Training on them would teach the model to guess from merchant identity. Quantified in `data_card.json`. |
| Validation split is grouped by merchant | SROIE repeats merchants and templates; a random split leaks. Val has no merchant that is in train. |
| Test is the official held-out split, with a `seen_merchant` flag | Results are reported overall and on **unseen merchants**, the honest measure of "learned the task vs memorised the template". |
| Long receipts are truncated in the middle (40% head / 60% tail) | Company is at the top, total at the bottom. p95 receipt is ~940 chars, so truncation rarely triggers. |

Known data limits: ~2% of gold labels do not appear verbatim in the OCR text (OCR typos), and `total` is written with or
without currency symbols. Scoring normalises both; the same grounding check on the gold labels is reported as `gold ungrounded`.

## Method

- Base model: **Qwen3-0.6B** (ablate: Qwen3-1.7B, Qwen2.5-0.5B-Instruct, SmolLM2-1.7B-Instruct). Small enough to fine-tune in
  minutes on a free Colab T4 and to deploy on-device.
- Adaptation: **LoRA** r=16, alpha=32, dropout 0.05, all linear layers, lr 2e-4 cosine, 3 epochs, loss on the JSON only. Ablations: rank 4/64, attention-only targets, QLoRA, 100/300/551 training samples.
- Prompt and target are built in one place (`prompts.py`), identical for training and inference. Greedy decoding, stop at `}`.
- Baselines: regex extractor, base model zero-shot and 3-shot, 7B instruct 3-shot in 4-bit.
- Guardrail: every predicted value is checked against the source text (`grounded`); ungrounded outputs are the ones to route to human review.

## Metrics

JSON validity, exact match per field after normalisation (`total` compared as a number), company token-F1, `all3 EM`
(company, date and total all correct), `all3 EM` on unseen merchants, ungrounded rate, ms per receipt, plus forgetting
(WikiText-2 perplexity adapter off vs on) and training cost (`runs/*/run_info.json`).

## Reproduce

Colab (recommended, T4): open `notebooks/train_colab.ipynb`, set `REPO_URL`, run all cells.

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
notebooks/        train_colab.ipynb        # the whole experiment, top to bottom
tests/            test_metrics.py
preds/            per-receipt predictions of every run (committed)
results/          summary.md, data_card.json, forgetting_*.json, env_freeze.txt
runs/<name>/      train_args.json, run_info.json, log_history.json (weights are git-ignored)
WRITEUP.md        source of the 1-2 page PDF
```

## Results

_Filled in after the Colab run: paste `results/summary.md` here._

## Submission checklist

- [ ] Results table above filled in; `WRITEUP.md` exported to PDF (name + track B at the top, 11 pt)
- [ ] Video link (5 min max): problem and approach 1 min, decisions and what was ruled out 2 min, results and failures 1.5 min, next steps 0.5 min
- [ ] Iterative commit history (data, baselines, training, ablations, write-up as separate commits)
- [ ] `results/env_freeze.txt` committed
