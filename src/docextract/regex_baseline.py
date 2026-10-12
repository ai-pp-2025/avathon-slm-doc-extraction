"""Rule-based baseline: what a team would build first, and the bar the SLM has to clear."""
import re

_DATE = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{2,4}\b",
    r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
]
_AMOUNT = re.compile(r"(?<![\d.])\d+(?:,\d{3})*\.\d{2}(?!\d)")
_TOTAL_KEY = re.compile(r"TOTAL", re.I)
_TOTAL_SKIP = re.compile(r"SUB|EXCL|QTY|ITEM|SAVING|DISC|CHANGE|CASH", re.I)
_COMPANY_HINT = re.compile(
    r"\b(SDN\.?\s*BHD|S/B|BHD|ENTERPRISE|TRADING|RESTAURANT|STORE|MART|CAFE|BISTRO|HARDWARE|STATIONERY)\b", re.I
)


def _date(text):
    best = None
    for p in _DATE:
        m = re.search(p, text, re.I)
        if m and (best is None or m.start() < best.start()):
            best = m
    return best.group() if best else ""


def _total(lines):
    cands = []
    for i, line in enumerate(lines):
        if not _TOTAL_KEY.search(line) or _TOTAL_SKIP.search(line):
            continue
        for j in range(i, min(i + 3, len(lines))):  # the amount is often on one of the next OCR lines
            if j > i and (_TOTAL_SKIP.search(lines[j]) or _TOTAL_KEY.search(lines[j])):
                break
            m = _AMOUNT.search(lines[j])
            if m:
                cands.append(m.group())
                break
    return max(cands, key=lambda s: float(s.replace(",", ""))) if cands else ""


def _company_idx(lines):
    for i, line in enumerate(lines[:8]):
        if _COMPANY_HINT.search(line):
            return i
    for i, line in enumerate(lines):
        if len(re.findall(r"[A-Za-z]", line)) >= 3:
            return i
    return 0


def predict(text: str) -> dict:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return {"company": "", "date": "", "total": ""}
    return {"company": lines[_company_idx(lines)], "date": _date(text), "total": _total(lines)}
