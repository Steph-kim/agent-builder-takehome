"""The sim fault wrapper fails exactly the call it names and passes everything else through. Offline."""

from __future__ import annotations

import pytest

from avis_agent.client import AvisAPIError, OutcomeUnknown
from evals.faults import DRIFT, FAULTS, FaultyClient


class Stub:
    def __init__(self):
        self.extends = 0

    def get_reservation(self, rid):
        return {"reservation_id": rid}

    def get_availability(self, *a):
        return {"availability": {"requested_type": {"available": True}}}

    def quote_extension(self, *a):
        return {"quote": {"charges": {"taxes_and_fees": 4.25, "total_charged": 50.24}}}

    def extend(self, rid, when, **kw):
        self.extends += 1
        return {"charges": {"taxes_and_fees": 4.25, "total_charged": 50.24}}


@pytest.mark.parametrize(
    "fault,exc,status",
    [
        ("extend_402", AvisAPIError, 402),
        ("extend_409", AvisAPIError, 409),
        ("extend_unknown", OutcomeUnknown, None),
    ],
)
def test_faulted_extend_raises_and_never_reaches_the_api(fault, exc, status):
    inner = Stub()
    with pytest.raises(exc) as e:
        FaultyClient(inner, fault).extend(
            "AVS-1", "t", email="e", cvv="c", billing_zip="z", idempotency_key="k"
        )
    assert inner.extends == 0
    if status:
        assert e.value.status == status
    else:
        assert e.value.idempotency_key == "k"


def test_vehicle_unavailable_flips_only_availability():
    c = FaultyClient(Stub(), "vehicle_unavailable")
    assert c.get_availability()["availability"]["requested_type"]["available"] is False
    assert c.quote_extension()["quote"]["charges"]["total_charged"] == 50.24


def test_price_drift_moves_every_quote_after_the_first_and_the_charge():
    c = FaultyClient(Stub(), "price_drift")
    totals = [c.quote_extension()["quote"]["charges"]["total_charged"] for _ in range(3)]
    assert totals == [50.24, 50.24 + DRIFT, 50.24 + DRIFT]
    assert c.extend("AVS-1", "t")["charges"]["total_charged"] == 50.24 + DRIFT


def test_untouched_calls_pass_through():
    for fault in FAULTS:
        assert FaultyClient(Stub(), fault).get_reservation("AVS-1") == {"reservation_id": "AVS-1"}
    with pytest.raises(ValueError):
        FaultyClient(Stub(), "nope")
