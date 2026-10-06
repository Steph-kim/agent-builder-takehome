"""Eligibility gates (D6) — worked examples from the six recorded reservations under a frozen clock."""

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from avis_agent.config import Thresholds
from avis_agent.policy import money, reservation_gate, value_gate
from avis_agent.reasons import ReasonCode

FIXTURES = Path(__file__).parent / "fixtures"
REC = {p.stem: json.loads(p.read_text()) for p in FIXTURES.glob("AVS-*.json")}
NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)  # the day the fixtures were recorded
T = Thresholds()


def gate(rid, now=NOW, pilot=frozenset(), record=None):
    return reservation_gate(record or REC[rid], now, T, pilot)


@pytest.mark.parametrize("rid", ["AVS-29471835", "AVS-66002030", "AVS-50000001", "AVS-99004050"])
def test_future_returns_pass(rid):
    assert gate(rid) is None


def test_marcus_103_days_overdue():
    assert gate("AVS-48372915") is ReasonCode.OVERDUE_BEYOND_POLICY


def test_priya_running_late_passes_until_the_overdue_window_ends():
    ret = datetime.fromisoformat(REC["AVS-77001020"]["dates"]["current_return_datetime"])
    assert gate("AVS-77001020", now=ret - timedelta(minutes=40)) is None
    assert gate("AVS-77001020", now=ret + timedelta(hours=T.overdue_hours)) is None  # boundary passes
    late = ret + timedelta(hours=T.overdue_hours, minutes=1)
    assert gate("AVS-77001020", now=late) is ReasonCode.OVERDUE_BEYOND_POLICY


def test_status_before_market_before_overdue():
    marcus = copy.deepcopy(REC["AVS-48372915"])  # ORD, overdue
    assert gate(None, pilot=frozenset({"LAX"}), record=marcus) is ReasonCode.OUT_OF_MARKET
    marcus["status"] = "cancelled"
    assert gate(None, pilot=frozenset({"LAX"}), record=marcus) is ReasonCode.NOT_ACTIVE


def test_pilot_set_matches_return_location():
    assert gate("AVS-29471835", pilot=frozenset({"LAX"})) is None
    assert gate("AVS-29471835", pilot=frozenset({"SFO"})) is ReasonCode.OUT_OF_MARKET


@pytest.mark.parametrize(
    "total, days, expected",
    [
        (100.49, 2, None),
        (500.00, 3, None),
        (500.01, 3, ReasonCode.HIGH_VALUE),
        (120.0, 14, None),
        (120.0, 15, ReasonCode.HIGH_VALUE),
        (4419.13, 103, ReasonCode.HIGH_VALUE),  # Marcus's live quote
    ],
)
def test_value_gate_boundaries(total, days, expected):
    assert value_gate({"total_charged": total, "extension_days": days}, T) is expected


def test_money_compares_cents_not_floats():
    assert money(0.1 + 0.2) == money("0.30")
    assert money(100.49) == money(100.490000001)
