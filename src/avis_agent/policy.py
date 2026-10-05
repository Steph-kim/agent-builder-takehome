"""Eligibility gates for an extension (D6, AOP §4). Pure functions — no I/O — so the recorded fixtures and a
frozen clock are the test oracle. The order lives in `extend.evaluate`; the first failing gate wins.

Threshold values come from `config.Thresholds` and never reach the model (it only sees reason codes).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from .config import Thresholds
from .reasons import ReasonCode


def reservation_gate(
    record: dict, now: datetime, thresholds: Thresholds, pilot_locations: frozenset[str]
) -> ReasonCode | None:
    """Gates that need only the reservation: status → market → overdue."""
    if record.get("status") != "active":
        return ReasonCode.NOT_ACTIVE
    if pilot_locations and record["return_location"]["code"].upper() not in pilot_locations:
        return ReasonCode.OUT_OF_MARKET
    current_return = datetime.fromisoformat(record["dates"]["current_return_datetime"])
    if now > current_return + timedelta(hours=thresholds.overdue_hours):
        return ReasonCode.OVERDUE_BEYOND_POLICY
    return None


def value_gate(charges: dict, thresholds: Thresholds) -> ReasonCode | None:
    """Quote-based gate: a large total or a long extension needs a person (kb_sup_01)."""
    if money(charges["total_charged"]) > money(thresholds.max_total_usd):
        return ReasonCode.HIGH_VALUE
    if int(charges["extension_days"]) > thresholds.max_added_days:
        return ReasonCode.HIGH_VALUE
    return None


def money(value: float | str | Decimal) -> Decimal:
    """JSON floats → cents, so 100.49 == 100.49 however it was parsed."""
    return Decimal(str(value)).quantize(Decimal("0.01"))
