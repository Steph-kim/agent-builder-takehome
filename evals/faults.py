"""Fault injection for sims: wraps the real `AvisClient` and fails one call the way production can.

Evals-only, passed in through `cli.chat(client=...)` — the product has no fault switch. Everything not faulted
goes to the live API, so the model, gates, card, commit and trace all run for real up to the fault. Faulted
writes never reach the API (no charge is attempted).

    extend_402           extend → 402 PAYMENT_DECLINED
    extend_409           extend → 409 RESERVATION_NOT_ACTIVE (the booking changed under us)
    extend_unknown       extend → OutcomeUnknown (sent, no answer: may or may not have charged)
    vehicle_unavailable  availability → requested type not available
    price_drift          every quote after the first (and the charge) is $10.00 higher
"""

from __future__ import annotations

import copy
from typing import Any

from avis_agent.client import AvisAPIError, AvisClient, OutcomeUnknown

FAULTS = ("extend_402", "extend_409", "extend_unknown", "vehicle_unavailable", "price_drift")
DRIFT = 10.0


class FaultyClient:
    def __init__(self, inner: AvisClient, fault: str):
        if fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}; expected one of {FAULTS}")
        self._inner, self.fault = inner, fault
        self.quotes = 0

    def __getattr__(self, name: str) -> Any:  # get_reservation, close, … go straight through
        return getattr(self._inner, name)

    def get_availability(self, *args: Any, **kwargs: Any) -> dict:
        result = self._inner.get_availability(*args, **kwargs)
        if self.fault == "vehicle_unavailable":
            result = copy.deepcopy(result)
            result.setdefault("availability", {}).setdefault("requested_type", {})["available"] = False
        return result

    def quote_extension(self, *args: Any, **kwargs: Any) -> dict:
        result = self._inner.quote_extension(*args, **kwargs)
        self.quotes += 1
        if self.fault == "price_drift" and self.quotes > 1:
            result = copy.deepcopy(result)
            _bump(result["quote"]["charges"])
        return result

    def extend(self, reservation_id: str, new_return_datetime: str, **kwargs: Any) -> dict:
        key = kwargs.get("idempotency_key") or "none"
        if self.fault == "extend_402":
            raise AvisAPIError(402, "PAYMENT_DECLINED", "The card on file was declined.")
        if self.fault == "extend_409":
            raise AvisAPIError(409, "RESERVATION_NOT_ACTIVE", "Reservation is no longer active.")
        if self.fault == "extend_unknown":
            raise OutcomeUnknown(f"/reservations/{reservation_id}/extend", key, "ReadTimeout")
        result = self._inner.extend(reservation_id, new_return_datetime, **kwargs)
        if self.fault == "price_drift":
            result = copy.deepcopy(result)
            _bump(result["charges"])
        return result


def _bump(charges: dict) -> None:
    """The price really changed: taxes and total move together, so the card still adds up."""
    charges["taxes_and_fees"] = round(float(charges["taxes_and_fees"]) + DRIFT, 2)
    charges["total_charged"] = round(float(charges["total_charged"]) + DRIFT, 2)
