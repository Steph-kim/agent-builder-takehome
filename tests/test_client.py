"""Fault injection against AvisClient via httpx.MockTransport (offline)."""

import json

import httpx
import pytest

from avis_agent.client import AvisAPIError, AvisClient, AvisUnavailable, OutcomeUnknown
from avis_agent.config import ConfigError, load_settings

OK_RES = {"reservation_id": "AVS-29471835", "status": "active"}
OK_EXTEND = {"success": True, "confirmation_number": "EXT-1"}


def envelope(status, code, message="x"):
    return httpx.Response(
        status, json={"success": False, "error": {"code": code, "message": message, "details": {}}}
    )


def scripted(*steps):
    """Transport that plays `steps` in order (a Response, or an exception to raise) and records requests."""
    seen = []
    queue = list(steps)

    def handler(request: httpx.Request):
        seen.append(request)
        step = queue.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    return httpx.MockTransport(handler), seen


def make(transport, events=None):
    return AvisClient(
        "https://avis.test",
        "k",
        transport=transport,
        sleep=lambda s: None,
        observer=(events.append if events is not None else None),
    )


def extend(client, **kw):
    return client.extend(
        "AVS-29471835", "2027-06-17T14:00:00-07:00", email="a@b.co", cvv="123", billing_zip="90210", **kw
    )


def test_read_retries_transient_5xx_then_succeeds():
    transport, seen = scripted(
        envelope(503, "SERVICE_UNAVAILABLE"),
        httpx.Response(502, text="<html>"),
        httpx.Response(200, json=OK_RES),
    )
    assert make(transport).get_reservation("AVS-29471835") == OK_RES
    assert len(seen) == 3


def test_read_timeouts_exhaust_into_unavailable():
    timeout = httpx.ReadTimeout("slow")
    transport, seen = scripted(timeout, timeout, timeout)
    with pytest.raises(AvisUnavailable):
        make(transport).get_reservation("AVS-29471835")
    assert len(seen) == 3  # 1 try + 2 retries


def test_availability_gets_exactly_one_retry():
    transport, seen = scripted(httpx.Response(504), httpx.Response(504))
    with pytest.raises(AvisUnavailable):
        make(transport).get_availability("OAK", "suv", "2027-06-13", "2027-06-15")
    assert len(seen) == 2


def test_4xx_is_not_retried_and_keeps_envelope_code():
    transport, seen = scripted(envelope(404, "RESERVATION_NOT_FOUND", "No reservation found"))
    with pytest.raises(AvisAPIError) as e:
        make(transport).get_reservation("AVS-00000000")
    assert (e.value.status, e.value.code) == (404, "RESERVATION_NOT_FOUND")
    assert len(seen) == 1


def test_write_reuses_one_idempotency_key_across_transport_retries():
    transport, seen = scripted(
        httpx.ReadTimeout("lost"), envelope(503, "X"), httpx.Response(200, json=OK_EXTEND)
    )
    assert extend(make(transport)) == OK_EXTEND
    keys = {r.headers["Idempotency-Key"] for r in seen}
    assert len(seen) == 3 and len(keys) == 1


def test_separate_write_calls_mint_separate_keys():
    # The API replays a cached success for a reused key regardless of body, so a re-quoted
    # commit (new body) must never share a key with the previous attempt.
    transport, seen = scripted(httpx.Response(200, json=OK_EXTEND), httpx.Response(200, json=OK_EXTEND))
    client = make(transport)
    extend(client)
    extend(client)
    assert seen[0].headers["Idempotency-Key"] != seen[1].headers["Idempotency-Key"]


def test_write_exhausted_is_outcome_unknown_with_key():
    timeout = httpx.ReadTimeout("lost")
    transport, seen = scripted(timeout, timeout, timeout)
    with pytest.raises(OutcomeUnknown) as e:
        extend(make(transport), idempotency_key="key-1")
    assert e.value.idempotency_key == "key-1"
    assert all(r.headers["Idempotency-Key"] == "key-1" for r in seen)


def test_write_4xx_is_a_definite_failure_not_unknown():
    transport, _ = scripted(envelope(402, "PAYMENT_DECLINED"))
    with pytest.raises(AvisAPIError) as e:
        extend(make(transport))
    assert e.value.code == "PAYMENT_DECLINED"


def test_extend_body_uses_card_on_file():
    transport, seen = scripted(httpx.Response(200, json=OK_EXTEND))
    extend(make(transport))
    body = json.loads(seen[0].content)
    assert body["payment"] == {"use_card_on_file": True, "cvv": "123", "billing_zip": "90210"}


def test_observer_events_carry_no_request_body():
    events = []
    transport, _ = scripted(envelope(503, "X"), httpx.Response(200, json=OK_EXTEND))
    extend(make(transport, events))
    assert [e["status"] for e in events] == [503, 200]
    flat = json.dumps(events)
    assert "a@b.co" not in flat and "90210" not in flat and '"123"' not in flat
    assert {"latency_ms", "attempt", "idempotency_key", "error_code"} <= events[0].keys()


def test_broken_observer_does_not_break_the_call():
    transport, _ = scripted(httpx.Response(200, json=OK_RES))
    client = AvisClient("https://avis.test", "k", transport=transport, observer=lambda e: 1 / 0)
    assert client.get_reservation("AVS-29471835") == OK_RES


@pytest.mark.parametrize("bad", ["../admin", "AVS 1", "", "a/b"])
def test_malformed_ids_never_reach_the_network(bad):
    transport, seen = scripted()
    with pytest.raises(AvisAPIError) as e:
        make(transport).get_reservation(bad)
    assert e.value.code == "RESERVATION_NOT_FOUND" and not seen


def test_missing_env_names_every_missing_variable(tmp_path, monkeypatch):
    for name in ("OPENAI_API_KEY", "AVIS_API_KEY", "AVIS_API_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AVIS_API_URL", "https://avis.test")
    with pytest.raises(ConfigError) as e:
        load_settings(tmp_path / "missing.env")
    assert "OPENAI_API_KEY" in str(e.value) and "AVIS_API_KEY" in str(e.value)
    assert "AVIS_API_URL" not in str(e.value)
