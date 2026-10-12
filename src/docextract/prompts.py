"""Prompt / target construction shared by training and inference (they must match exactly)."""
import json

from .metrics import FIELDS, norm_text

INSTRUCTION = (
    "Extract the company, date and total from the receipt below. "
    "Answer with a single JSON object with keys company, date, total."
)


def target(entities: dict) -> str:
    return json.dumps({f: entities[f] for f in FIELDS}, ensure_ascii=False)


def truncate_middle(text: str, tok, budget: int) -> str:
    """Keep the head (company, address) and the tail (total) of long receipts."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return text
    head = int(budget * 0.4)
    tail = budget - head
    return tok.decode(ids[:head]) + "\n...\n" + tok.decode(ids[-tail:])


def build_prompt(text: str, tok, budget: int = 800, shots=()) -> str:
    parts = [INSTRUCTION, ""]
    for s in shots:
        parts += ["Receipt:", truncate_middle(s["text"], tok, 300), "JSON:", target(s["entities"]), ""]
    parts += ["Receipt:", truncate_middle(text, tok, budget), "JSON:", ""]
    return "\n".join(parts)


def select_shots(train_rows, k: int):
    """k few-shot examples: distinct merchants, complete labels, length near the 25th percentile."""
    if k <= 0:
        return []
    anchor = sorted(len(r["text"]) for r in train_rows)[len(train_rows) // 4]
    shots, seen = [], set()
    for r in sorted(train_rows, key=lambda r: abs(len(r["text"]) - anchor)):
        c = norm_text(r["entities"]["company"])
        if c in seen or not all(r["entities"].values()):
            continue
        seen.add(c)
        shots.append(r)
        if len(shots) == k:
            break
    return shots
