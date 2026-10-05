import io
import json
from pathlib import Path

import httpx
import pytest

from avis_agent.client import AvisClient
from avis_agent.trace import Tracer

FIXTURES = Path(__file__).parent / "fixtures"
SARAH = json.loads((FIXTURES / "AVS-29471835.json").read_text())


def lines(tracer: Tracer) -> list[dict]:
    return [json.loads(line) for line in tracer.path.read_text().splitlines()]


@pytest.fixture
def tracer(tmp_path):
    return Tracer("s1", tmp_path)


def test_reservation_payload_loses_pii_but_keeps_what_debugging_needs(tracer):
    tracer.emit("tool.result", name="lookup_reservation", result=SARAH)
    raw = tracer.path.read_text()
    for pii in (
        "Sarah",
        "Johnson",
        "9217 Airport Blvd",
        "8ABC123",
        "4832",
        "Chevrolet",
        "Silver",
        "CUST-847291",
    ):
        assert pii not in raw, pii
    result = lines(tracer)[0]["result"]
    assert result["reservation_id"] == "AVS-29471835"
    assert result["dates"]["current_return_datetime"] == "2027-06-15T14:00:00-07:00"
    assert result["vehicle"] == {"type": "midsize_sedan"}
    assert result["pickup_location"]["code"] == "LAX"


def test_tool_args_drop_identity_fields(tracer):
    args = {"reservation_id": "AVS-29471835", "last_name": "Johnson", "email": "sarah@example.com"}
    tracer.emit("tool.call", name="lookup_reservation", args=args)
    assert lines(tracer)[0]["args"] == {"reservation_id": "AVS-29471835"}


def test_fields_outside_an_events_allowlist_are_dropped(tracer):
    tracer.emit("customer.msg", text="hi", redacted=["cvv"], email="sarah@example.com")
    tracer.emit("not.an.event", text="secret")
    first, second = lines(tracer)
    assert first.keys() == {"ts", "event", "text", "redacted"}
    assert second.keys() == {"ts", "event"}


def test_client_observer_writes_api_requests_without_the_body(tracer):
    replies = iter(
        [
            httpx.Response(503, json={}),
            httpx.Response(200, json={"success": True, "confirmation_number": "EXT-1"}),
        ]
    )
    client = AvisClient(
        "https://avis.test",
        "k",
        transport=httpx.MockTransport(lambda r: next(replies)),
        observer=tracer.api_request,
        sleep=lambda s: None,
    )
    client.extend(
        "AVS-29471835", "2027-06-17T14:00:00-07:00",
        email="sarah@example.com", cvv="847", billing_zip="90210", idempotency_key="key-1",
    )  # fmt: skip
    events = lines(tracer)
    assert [(e["event"], e["status"], e["attempt"], e["idempotency_key"]) for e in events] == [
        ("api.request", 503, 1, "key-1"),
        ("api.request", 200, 2, "key-1"),
    ]
    raw = tracer.path.read_text()
    assert "sarah@example.com" not in raw and "90210" not in raw and '"847"' not in raw


def test_unwritable_log_warns_once_and_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    err = io.StringIO()
    t = Tracer("s1", blocker, stderr=err)  # log dir is a file → every write fails
    t.emit("agent.msg", text="hello")
    t.emit("agent.msg", text="again")
    t.close()
    assert err.getvalue().count("[trace]") == 1


def test_outcomes_are_one_per_request_in_order(tracer):
    tracer.add_outcome("resolved_extension")
    tracer.add_outcome("handed_off", "overdue")
    tracer.add_outcome("info_only")
    tracer.close()
    tracer.close()  # idempotent: one ledger line
    outcome_lines = [e for e in lines(tracer) if e["event"] == "outcome"]
    assert outcome_lines == [
        {**outcome_lines[0], "outcomes": ["resolved_extension", "handed_off:overdue", "info_only"]}
    ]


def test_ctrl_c_mid_session_still_writes_the_ledger_and_the_key(tracer):
    with pytest.raises(KeyboardInterrupt), tracer:
        tracer.add_outcome("info_only")
        tracer.emit("tool.call", name="extend_reservation", args={"idempotency_key": "key-9"})
        raise KeyboardInterrupt
    events = lines(tracer)
    assert events[0]["args"] == {"idempotency_key": "key-9"}
    assert events[-1]["outcomes"] == ["info_only", "interrupted"]


def test_crash_records_error_and_empty_session_records_abandoned(tmp_path):
    crashed = Tracer("a", tmp_path)
    with pytest.raises(RuntimeError), crashed:
        raise RuntimeError
    empty = Tracer("b", tmp_path)
    empty.close()
    assert lines(crashed)[-1]["outcomes"] == ["error"]
    assert lines(empty)[-1]["outcomes"] == ["abandoned"]


def test_session_start_carries_versions(tracer):
    tracer.start(model="gpt-5-mini", prompt_hash="p1", kb_hash="k1")
    start = lines(tracer)[0]
    assert start["event"] == "session.start"
    assert {"git_sha", "model", "prompt_hash", "kb_hash"} <= start.keys()
    assert start["git_sha"]
