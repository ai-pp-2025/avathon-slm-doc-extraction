# Receipt field extraction with a LoRA-tuned small LM

**Ganesh Patil, Track B (SLM fine-tuning), Scenario S2 (Gen AI for enterprise documents)**

_Target 1-2 pages, 11 pt. Every number below comes from `results/summary.md` or `runs/*/run_info.json`._

## 1. Problem and domain

- Business problem: AP/expense automation. Receipts and invoices arrive as noisy OCR text, and systems need `company`, `date`, `total`. Data may not leave the company, and per-document API cost matters.
- Why an SLM and why Track B: _one paragraph. Behaviour (field choice, format discipline) rather than knowledge; the answer is in context so RAG adds nothing; on-device / in-house inference._
- Why this is not already solved by the base model: _zero-shot and 3-shot numbers vs fine-tuned._

## 2. Approach and algorithm decisions

For each choice: what I chose, the alternative(s), why rejected, and the evidence.

| Decision | Chosen | Alternatives considered | Why rejected / evidence |
|---|---|---|---|
| Fine-tune vs prompt | LoRA fine-tune | zero/few-shot Qwen3-0.6B; 3-shot Qwen2.5-7B (4-bit); regex | _numbers; latency; size_ |
| Base model | Qwen3-0.6B | Qwen3-1.7B, Qwen2.5-0.5B, SmolLM2-1.7B | _accuracy vs training time vs size_ |
| PEFT method | LoRA | QLoRA, full fine-tune | _QLoRA: accuracy delta vs memory/time. Full FT: not needed at 0.6B with 551 examples, risks forgetting_ |
| Rank / alpha / targets | r=16, alpha=32, all linear | r=4, r=64, attention-only | _ablation table_ |
| Data size | 551 train | 100, 300 | _learning curve: still improving or saturated_ |
| Validation split | grouped by merchant | random split | _SROIE repeats merchants; random split leaks_ |
| Task fields | company, date, total | + address | _504/551 gold addresses not in OCR text (comma truncation); would teach guessing_ |
| Decoding | greedy, stop at `}` | sampling, constrained JSON decoding | _JSON validity rate; why constrained decoding was not needed_ |

Evidence of learning vs memorising: _unseen-merchant accuracy vs seen, learning curve._
Catastrophic forgetting: _WikiText-2 perplexity adapter off/on._

## 3. Results and error analysis

- Table: regex, zero-shot, 3-shot, 7B 3-shot, LoRA r=16 (overall and unseen merchants; JSON validity; ms/receipt).
- Qualitative: one receipt, base vs fine-tuned output.
- Failure analysis: _which field fails most, on what kind of receipts (multiple TOTAL lines, rounding adjustments, OCR typos, unseen merchants); gold-label noise floor (`gold ungrounded`)._
- Hallucination: _ungrounded rate; what the grounding check catches and misses._

## 4. Production and limitations

- Production: _in-house or on-device serving of the merged (and quantised) model, grounding check plus confidence threshold routing to human review, drift monitoring on JSON validity and ungrounded rate, adapter swap without redeploying the base model._
- Limitation: _single dataset, Malaysian receipts, English, clean OCR text as input (no layout, no images); address excluded; one seed per configuration._
