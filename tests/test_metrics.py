from docextract.metrics import grounded, norm_total, parse_json, score_example, token_f1
from docextract.regex_baseline import predict

TEXT = "\n".join(
    ["ACME TRADING SDN BHD", "(123456-K)", "12 JALAN ABC", "50000 KUALA LUMPUR", "TEL: 03-1234567",
     "DATE: 25/12/2018 8:13 PM", "SUB TOTAL", "9.00", "TOTAL", "RM 9.50", "CASH", "10.00"]
)
GOLD = {"company": "ACME TRADING SDN BHD", "date": "25/12/2018", "total": "9.50"}


def test_norm_total():
    assert norm_total("RM 1,234.5") == "1234.50"
    assert norm_total("$8.20") == "8.20"
    assert norm_total("") == ""


def test_token_f1():
    assert token_f1("A B C", "A B C") == 1.0
    assert token_f1("", "") == 1.0
    assert 0 < token_f1("A B", "A B C") < 1


def test_parse_json():
    assert parse_json('{"company": "X", "date": "d", "total": "1"}')["company"] == "X"
    assert parse_json("no json here") is None
    assert parse_json('{"company": ') is None


def test_score_example_none_pred():
    assert not any(v["em"] for v in score_example(None, GOLD).values())


def test_grounding_flags_hallucinated_values():
    assert all(grounded(GOLD, TEXT).values())
    assert not grounded({**GOLD, "company": "OTHER SHOP"}, TEXT)["company"]
    assert not grounded({**GOLD, "total": "7.77"}, TEXT)["total"]


def test_regex_baseline_finds_fields():
    pred = predict(TEXT)
    assert pred["company"] == "ACME TRADING SDN BHD"
    assert pred["date"] == "25/12/2018"
    assert pred["total"] == "9.50"
