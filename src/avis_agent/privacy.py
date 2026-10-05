"""Scrub payment data and emails from customer text before it reaches the model or a log.

The agent never needs a card number, CVV, ZIP or email in chat — the CLI collects
those off-model after the customer approves a charge. Anything volunteered is replaced
here. Reservation ids, dates, times, money amounts and confirmation numbers must pass
through untouched (the negatives in tests/test_privacy.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 13–19 digits, optionally grouped by spaces or dashes, not glued to letters/digits/dashes
# (so the digits inside "AVS-29471835" are never a candidate).
_CARD = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")
_SECRET_AFTER_KEYWORD = re.compile(
    r"\b(?P<kw>cvv2?|cvc|csc|security\s+code|(?:billing\s+)?zip(?:\s*code)?|postal\s+code)\b"
    r"(?P<gap>[^\d\n]{0,20}?)(?P<val>\d{5}(?:-\d{4})?|\d{3,4})\b",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

REDACTED = "[redacted]"


@dataclass
class ScrubResult:
    text: str
    kinds: list[str] = field(default_factory=list)

    @property
    def redacted(self) -> bool:
        return bool(self.kinds)


def scrub(text: str) -> ScrubResult:
    kinds: list[str] = []

    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        if not _luhn_ok(digits):
            return m.group()
        kinds.append("card_number")
        return REDACTED

    def secret(m: re.Match) -> str:
        kw = m.group("kw").lower()
        kinds.append("zip" if ("zip" in kw or "postal" in kw) else "cvv")
        return f"{m.group('kw')}{m.group('gap')}{REDACTED}"

    def email(m: re.Match) -> str:
        kinds.append("email")
        return REDACTED

    text = _CARD.sub(card, text)
    text = _SECRET_AFTER_KEYWORD.sub(secret, text)
    text = _EMAIL.sub(email, text)
    return ScrubResult(text, kinds)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0
