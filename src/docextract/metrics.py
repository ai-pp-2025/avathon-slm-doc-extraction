"""Field-level scoring, JSON parsing and grounding checks."""
import json
import re
from collections import Counter

# `address` is deliberately not a task field: the dataset's OCR lines are truncated at commas, so it is not recoverable
# from the text (see data_card.json -> address_not_found_in_text_train).
FIELDS = ("company", "date", "total")

_NUM = re.compile(r"\d*\.?\d+")


def norm_text(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", str(s).upper())).strip()


def norm_date(s) -> str:
    return re.sub(r"\s+", "", str(s).upper())


def norm_total(s) -> str:
    m = _NUM.search(str(s).replace(",", ""))
    return f"{float(m.group()):.2f}" if m else ""


def token_f1(pred, gold) -> float:
    p, g = norm_text(pred).split(), norm_text(gold).split()
    if not p and not g:
        return 1.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def field_em(field, pred, gold) -> bool:
    if field == "total":
        return norm_total(pred) == norm_total(gold)
    if field == "date":
        return norm_date(pred) == norm_date(gold)
    return norm_text(pred) == norm_text(gold)


def score_example(pred, gold) -> dict:
    """pred is a dict (or None if the model output was not valid JSON)."""
    out = {}
    for f in FIELDS:
        if pred is None:
            out[f] = {"em": False, "f1": 0.0}
            continue
        em = field_em(f, pred.get(f, ""), gold[f])
        f1 = token_f1(pred.get(f, ""), gold[f]) if f == "company" else float(em)
        out[f] = {"em": em, "f1": f1}
    return out


def parse_json(raw: str):
    """First {...} object in the model output as {field: str}, or None."""
    m = re.search(r"\{.*?\}", raw, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return {f: str(obj.get(f, "") or "") for f in FIELDS}


def grounded(values: dict, text: str) -> dict:
    """Per field: does the value appear in the source text? False flags a hallucinated / reformatted value."""
    compact = norm_text(text).replace(" ", "")
    amounts = {f"{float(m):.2f}" for m in _NUM.findall(text.replace(",", ""))}
    out = {}
    for f in FIELDS:
        v = values.get(f, "")
        if f == "total":
            out[f] = norm_total(v) == "" or norm_total(v) in amounts
        else:
            out[f] = norm_text(v).replace(" ", "") in compact
    return out
