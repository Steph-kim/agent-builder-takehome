import re
from pathlib import Path

from avis_agent.reasons import CUSTOMER_COPY, MODEL_RAISED, OFFERED, ReasonCode

AOP = Path(__file__).resolve().parents[1] / "docs" / "aop-extend.md"
ROW = re.compile(
    r"^\|\s*`(?P<code>[a-z_]+)`\s*\|\s*(?P<raiser>code|model)\s*\|\s*(?P<then>offer|transfer)\s*\|"
    r"[^|]*\|(?P<hears>[^|]*)\|\s*$",
    re.MULTILINE,
)


def table_text() -> str:
    return AOP.read_text().split("<!-- reason-codes:start -->")[1].split("<!-- reason-codes:end -->")[0]


def aop_rows() -> dict[str, dict[str, str]]:
    rows = [m.groupdict() for m in ROW.finditer(table_text())]
    by_code = {r["code"]: r for r in rows}
    assert len(rows) == len(by_code), "duplicate reason code in AOP table"
    return by_code


def test_aop_table_matches_code_on_codes_raiser_and_offer_vs_transfer():
    in_code = {
        (c.value, "model" if c in MODEL_RAISED else "code", "offer" if c in OFFERED else "transfer")
        for c in ReasonCode
    }
    assert {(c, r["raiser"], r["then"]) for c, r in aop_rows().items()} == in_code


def test_customer_copy_matches_aop_word_for_word():
    for code, row in aop_rows().items():
        quoted = " ".join(re.findall(r'"([^"]+)"', row["hears"]))
        assert quoted == CUSTOMER_COPY[ReasonCode(code)], code
