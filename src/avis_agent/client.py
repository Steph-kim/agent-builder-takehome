"""Avis API client — the single owner of retries for Avis calls.

Reads retry 5xx/timeouts with jittered backoff; 4xx is never retried. The extend
write mints one Idempotency-Key per call (one exact body) and reuses it only for
transport retries of that body, because the API replays a cached success for a
reused key *regardless of body*. Write retries exhausted → OutcomeUnknown: the
charge may or may not have happened, and callers must never say "failed".
"""

from __future__ import annotations

import random
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

RETRYABLE_STATUS = {500, 502, 503, 504}
_PATH_ID = re.compile(r"^[A-Za-z0-9-]{1,40}$")


@dataclass(frozen=True)
class RetryPolicy:
    timeout_s: float
    retries: int


# Availability 504s surface after ~8.5s, so it gets a longer per-try timeout and one retry.
READ_POLICY = RetryPolicy(timeout_s=5.0, retries=2)
AVAILABILITY_POLICY = RetryPolicy(timeout_s=10.0, retries=1)
WRITE_POLICY = RetryPolicy(timeout_s=15.0, retries=2)


class AvisError(Exception):
    """Base for every error this client raises."""


class AvisAPIError(AvisError):
    """A non-retryable answer from the API (4xx), carrying the envelope's code."""

    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}


class AvisUnavailable(AvisError):
    """A read whose retries were exhausted on 5xx/timeouts — the answer is unknowable right now."""

    def __init__(self, path: str, last_error: str):
        super().__init__(f"{path} unavailable after retries: {last_error}")
        self.path = path
        self.last_error = last_error


class OutcomeUnknown(AvisError):
    """A write whose retries were exhausted — it may have been processed. Hand off with the key."""

    def __init__(self, path: str, idempotency_key: str, last_error: str):
        super().__init__(f"{path} outcome unknown (key {idempotency_key}): {last_error}")
        self.path = path
        self.idempotency_key = idempotency_key
        self.last_error = last_error


Observer = Callable[[dict[str, Any]], None]


class AvisClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        transport: httpx.BaseTransport | None = None,
        observer: Observer | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            transport=transport,
        )
        self._observer = observer
        self._sleep = sleep

    def close(self) -> None:
        self._http.close()

    # --- endpoints -------------------------------------------------------------------------

    def get_reservation(self, reservation_id: str) -> dict:
        return self._read("GET", f"/reservations/{_path_id(reservation_id)}", READ_POLICY)

    def get_availability(self, location: str, vehicle_type: str, start_date: str, end_date: str) -> dict:
        params = {
            "location": location,
            "vehicle_type": vehicle_type,
            "start_date": start_date,
            "end_date": end_date,
        }
        return self._read("GET", "/availability", AVAILABILITY_POLICY, params=params)

    def quote_extension(self, reservation_id: str, new_return_datetime: str) -> dict:
        body = {"change_type": "extend", "new_return_datetime": new_return_datetime}
        # Quote has no side effects, so it is retried like a read.
        return self._read("POST", f"/reservations/{_path_id(reservation_id)}/quote", READ_POLICY, json=body)

    def extend(
        self,
        reservation_id: str,
        new_return_datetime: str,
        *,
        email: str,
        cvv: str,
        billing_zip: str,
        idempotency_key: str | None = None,
    ) -> dict:
        body = {
            "new_return_datetime": new_return_datetime,
            "email": email,
            "payment": {"use_card_on_file": True, "cvv": cvv, "billing_zip": billing_zip},
        }
        key = idempotency_key or str(uuid.uuid4())
        path = f"/reservations/{_path_id(reservation_id)}/extend"
        try:
            return self._request("POST", path, WRITE_POLICY, json=body, idempotency_key=key)
        except _RetriesExhausted as e:
            raise OutcomeUnknown(path, key, e.last_error) from None

    # --- plumbing --------------------------------------------------------------------------

    def _read(self, method: str, path: str, policy: RetryPolicy, **kwargs: Any) -> dict:
        try:
            return self._request(method, path, policy, **kwargs)
        except _RetriesExhausted as e:
            raise AvisUnavailable(path, e.last_error) from None

    def _request(
        self,
        method: str,
        path: str,
        policy: RetryPolicy,
        *,
        params: dict | None = None,
        json: dict | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
        last_error = ""
        for attempt in range(1, policy.retries + 2):
            if attempt > 1:
                self._sleep(_backoff(attempt))
            started = time.monotonic()
            event: dict[str, Any] = {
                "method": method,
                "path": path,
                "attempt": attempt,
                "idempotency_key": idempotency_key,
            }
            try:
                resp = self._http.request(
                    method, path, params=params, json=json, headers=headers, timeout=policy.timeout_s
                )
            except httpx.TransportError as e:  # timeouts, connection resets, DNS
                last_error = type(e).__name__
                self._emit(event, started, status=None, error_code=last_error)
                continue
            payload = _json_or_none(resp)
            if resp.status_code in RETRYABLE_STATUS:
                last_error = f"HTTP {resp.status_code}"
                self._emit(event, started, status=resp.status_code, error_code=_error_code(payload))
                continue
            if resp.is_success and isinstance(payload, dict):
                self._emit(event, started, status=resp.status_code, error_code=None)
                return payload
            err = (payload or {}).get("error") if isinstance(payload, dict) else None
            err = err if isinstance(err, dict) else {}
            code = err.get("code") or f"HTTP_{resp.status_code}"
            self._emit(event, started, status=resp.status_code, error_code=code)
            raise AvisAPIError(
                resp.status_code, code, err.get("message") or resp.text[:200], err.get("details")
            )
        raise _RetriesExhausted(last_error)

    def _emit(self, event: dict, started: float, *, status: int | None, error_code: str | None) -> None:
        if self._observer is None:
            return
        event.update(
            status=status, error_code=error_code, latency_ms=round((time.monotonic() - started) * 1000)
        )
        try:
            self._observer(event)
        except Exception:  # a broken logger must never break an API call
            pass


class _RetriesExhausted(Exception):
    def __init__(self, last_error: str):
        self.last_error = last_error


def _path_id(value: str) -> str:
    """Model-supplied ids go into URL paths — reject anything that isn't a plain id."""
    value = value.strip().upper()
    if not _PATH_ID.match(value):
        raise AvisAPIError(404, "RESERVATION_NOT_FOUND", "Malformed reservation id.")
    return value


def _backoff(attempt: int) -> float:
    return 0.5 * 2 ** (attempt - 2) + random.uniform(0, 0.25)


def _json_or_none(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def _error_code(payload: Any) -> str | None:
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        return payload["error"].get("code")
    return None
