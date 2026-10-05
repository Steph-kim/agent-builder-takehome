"""Extend path (AOP §4–6) against the real AvisClient over httpx.MockTransport, so retries and parsing are
under test too. Every test ends with the PII oracle: no email, CVV, ZIP, card last-4 or auth code anywhere on
disk."""

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from avis_agent import extend
from avis_agent.client import AvisClient
from avis_agent.config import Thresholds
from avis_agent.extend import Payment, check, commit, render_card
from avis_agent.handoff import customer_summary, request_transfer
from avis_agent.reasons import CUSTOMER_COPY, ReasonCode
from avis_agent.tools import AgentContext
from avis_agent.trace import Tracer

FIXTURES = Path(__file__).parent / "fixtures"
SARAH = json.loads((FIXTURES / "AVS-29471835.json").read_text())
NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
NEW = "2027-06-17T14:00"  # Sarah: Tuesday 2pm → Thursday 2pm, LAX
EMAIL, CVV, ZIP = "pii.probe@example.com", "8641", "97035"
PAY = Payment(email=EMAIL, cvv=CVV, billing_zip=ZIP)
SECRETS = (EMAIL, CVV, ZIP, "4832", "AUTH-")


def charges(total=100.49, days=2):
    return {
        "daily_rate": 45.99,
        "extension_days": days,
        "subtotal": 91.98,
        "late_fee": 0.0,
        "one_way_fee": 0.0,
        "taxes_and_fees": 8.51,
        "total_charged": total,
        "currency": "USD",
    }


def extend_ok(total=100.49, new_return="2027-06-17T21:00:00+00:00"):  # same instant, other offset
    return httpx.Response(
        200,
        json={
            "success": True,
            "confirmation_number": "EXT-1",
            "reservation_id": "AVS-29471835",
            "extension_details": {"new_return_datetime": new_return, "extension_days": 2},
            "charges": charges(total),
            "payment": {"card_type": "Visa", "last_four": "4832", "authorization_code": "AUTH-9"},
        },
    )


def err(status, code):
    return httpx.Response(status, json={"success": False, "error": {"code": code, "message": f"bad {EMAIL}"}})


class FakeAvis:
    """Serves Sarah; quotes and extend answers are scripted queues. Records every request."""

    def __init__(self, extend_steps=(), quotes=(100.49,), availability=None, record=SARAH):
        self.extend_steps, self.quotes = list(extend_steps), list(quotes)
        self.availability, self.record = availability, record
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request):
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/extend"):
            step = self.extend_steps.pop(0)
            if isinstance(step, BaseException):
                raise step
            return step
        if path.endswith("/quote"):
            total = self.quotes.pop(0) if len(self.quotes) > 1 else self.quotes[0]
            return httpx.Response(200, json={"quote": {"charges": charges(total)}})
        if path == "/availability":
            return self.availability or httpx.Response(
                200, json={"availability": {"requested_type": {"available": True}}}
            )
        return httpx.Response(200, json=self.record)

    def count(self, suffix):
        return sum(r.url.path.endswith(suffix) for r in self.requests)

    def extend_bodies(self):
        return [r for r in self.requests if r.url.path.endswith("/extend")]


@pytest.fixture
def make(tmp_path):
    made = []

    def _make(fake=None, now=NOW):
        fake = fake or FakeAvis()
        tracer = Tracer("s1", tmp_path)
        client = AvisClient(
            "http://avis.test",
            "k",
            transport=httpx.MockTransport(fake),
            sleep=lambda s: None,
            observer=tracer.api_request,
        )
        clock = {"now": now}
        ctx = AgentContext(
            client=client,
            tracer=tracer,
            thresholds=Thresholds(),
            handoff_log=tmp_path / "handoffs.jsonl",
            now=lambda: clock["now"],
            notify=lambda m: ctx.notes.append(m),
        )
        ctx.notes, ctx.fake, ctx.clock = [], fake, clock
        ctx.reservation = copy.deepcopy(fake.record)  # verified by lookup
        made.append(ctx)
        return ctx

    yield _make
    # PII oracle: nothing secret reached the session log, the handoff log, or a pending/quote/packet.
    on_disk = "".join(p.read_text() for p in tmp_path.glob("*.jsonl"))
    for secret in SECRETS:
        assert secret not in on_disk, secret
    for ctx in made:
        assert not any(s in json.dumps([ctx.quote, ctx.last_write], default=str) for s in SECRETS)


def approved(ctx, new=NEW):
    result = check(ctx, new)
    assert result.get("ready"), result
    ctx.pending.shown = ctx.pending.approved = True
    return ctx.pending


def events(ctx):
    return [json.loads(line) for line in ctx.tracer.path.read_text().splitlines()]


# --- check ---------------------------------------------------------------------------------------


def test_check_prices_and_stores_a_pending_the_model_sees_without_card(make):
    ctx = make()
    result = check(ctx, NEW)
    assert result["ready"] and result["charges"]["total_charged"] == 100.49
    assert result["new_return"] == "Thursday, June 17, 2027 at 2:00 PM PDT"
    assert "4832" not in json.dumps(result)
    assert ctx.pending.total == extend.money(100.49) and ctx.quote["total"] == 100.49
    assert ctx.notes == ["Checking availability…"]
    avail = next(r for r in ctx.fake.requests if r.url.path == "/availability")
    assert dict(avail.url.params) == {
        "location": "LAX",
        "vehicle_type": "midsize_sedan",
        "start_date": "2027-06-15",
        "end_date": "2027-06-17",
    }
    card = render_card(ctx.pending)
    assert "$100.49 USD" in card and "Visa ending 4832" in card and "Thursday, June 17" in card


def test_availability_unknown_stops_before_the_quote(make):
    ctx = make(FakeAvis(availability=httpx.Response(504)))
    assert check(ctx, NEW)["reason_code"] == "availability_unknown"
    assert ctx.fake.count("/quote") == 0 and ctx.pending is None
    assert ctx.pending_handoff is ReasonCode.AVAILABILITY_UNKNOWN


def test_vehicle_unavailable(make):
    no = httpx.Response(200, json={"availability": {"requested_type": {"available": False}}})
    assert check(make(FakeAvis(availability=no)), NEW)["reason_code"] == "vehicle_unavailable"


@pytest.mark.parametrize(
    "new, expected",
    [("2027-06-15T13:00", "not_an_extension"), ("Friday 2pm", "bad_time"), ("2027-03-14T02:30", "bad_time")],
)
def test_refusals_before_any_pricing(make, new, expected):
    ctx = make()
    result = check(ctx, new)
    assert expected in (result.get("reason_code"), result.get("error"))
    assert ctx.fake.count("/quote") == 0


def test_a_time_that_already_passed_is_refused(make):
    late = datetime(2027, 6, 16, 19, 0, tzinfo=UTC)  # Sarah 22h late; 10am local is already gone
    assert check(make(now=late), "2027-06-16T10:00")["error"] == "time_passed"


def test_high_value_handoff_carries_the_quote(make):
    ctx = make(FakeAvis(quotes=(600.0,)))
    assert check(ctx, NEW)["reason_code"] == "high_value" and ctx.pending is None
    request_transfer(ctx, "customer_requested", "")
    assert "the quote ($600.00)" in customer_summary(ctx.transfer)


def test_a_failed_check_drops_an_earlier_card(make):
    ctx = make()
    check(ctx, NEW)
    assert check(ctx, "2027-06-15T13:00")["reason_code"] == "not_an_extension"
    assert ctx.pending is None


# --- commit --------------------------------------------------------------------------------------


def test_happy_path_commits_once_and_receipt_comes_from_the_response(make):
    ctx = make(FakeAvis([extend_ok()]))
    p = approved(ctx)
    result = commit(ctx, p, PAY, retry_allowed=True)
    assert result.kind == "resolved" and "EXT-1" in result.message and "$100.49 USD" in result.message
    assert ctx.write_state == "committed" and ctx.pending is None
    assert ctx.tracer.outcomes == ["resolved_extension"]
    (req,) = ctx.fake.extend_bodies()
    assert json.loads(req.content) == {
        "new_return_datetime": "2027-06-17T14:00:00-07:00",
        "email": EMAIL,
        "payment": {"use_card_on_file": True, "cvv": CVV, "billing_zip": ZIP},
    }
    # The key is logged before the request goes out (Ctrl-C mid-write still leaves it).
    ev = events(ctx)
    call = next(i for i, e in enumerate(ev) if e["event"] == "tool.call" and e["name"] == "extend")
    sent = next(i for i, e in enumerate(ev) if e["event"] == "api.request" and e["path"].endswith("/extend"))
    assert call < sent and ev[call]["args"]["idempotency_key"] == req.headers["Idempotency-Key"]
    assert ctx.fake.count("/quote") == 2  # re-quoted at commit
    assert check(ctx, NEW)["reason_code"] == "write_already_done"


def test_commit_refuses_without_an_approved_pending(make):
    ctx = make(FakeAvis([extend_ok()]))
    check(ctx, NEW)
    with pytest.raises(RuntimeError):
        commit(ctx, ctx.pending, PAY, retry_allowed=True)  # shown but never approved
    assert ctx.fake.count("/extend") == 0


@pytest.mark.parametrize(
    "steps",
    [
        [httpx.Response(200, text="<html>oops</html>")],  # charged, body unreadable
        [httpx.Response(503)] * 3,  # retries exhausted
        [httpx.ReadTimeout("t"), err(400, "INVALID_EXTENSION")],  # refusal after an uncertain attempt
    ],
    ids=["malformed-2xx", "5xx-exhausted", "timeout-then-4xx"],
)
def test_uncertain_writes_transfer_as_outcome_unknown(make, steps):
    ctx = make(FakeAvis(steps))
    result = commit(ctx, approved(ctx), PAY, retry_allowed=True)
    assert result.kind == "transfer" and ctx.transfer.reason_code == "outcome_unknown"
    assert ctx.write_state == "unknown"
    assert ctx.transfer.extend["idempotency_key"] and ctx.transfer.extend["approved_total"] == 100.49
    assert "may have" in customer_summary(ctx.transfer)
    assert ctx.tracer.outcomes == ["handed_off:outcome_unknown"]


@pytest.mark.parametrize(
    "response",
    [extend_ok(total=90.0), extend_ok(new_return="2027-06-18T14:00:00-07:00")],
    ids=["total", "date"],
)
def test_response_that_differs_from_the_card_is_a_mismatch_transfer(make, response):
    ctx = make(FakeAvis([response]))
    assert commit(ctx, approved(ctx), PAY, retry_allowed=True).kind == "transfer"
    assert ctx.transfer.reason_code == "confirmation_mismatch" and ctx.write_state == "committed"
    assert ctx.transfer.extend["confirmation_number"] == "EXT-1"
    summary = customer_summary(ctx.transfer)
    assert "submitted and charged" in summary and "confirmation number is EXT-1" in summary


def test_2xx_missing_charges_is_a_mismatch(make):
    body = json.loads(extend_ok().content)
    del body["charges"]
    ctx = make(FakeAvis([httpx.Response(200, json=body)]))
    commit(ctx, approved(ctx), PAY, retry_allowed=True)
    assert ctx.transfer.reason_code == "confirmation_mismatch"


def test_wrong_details_get_one_retry_then_lock_extensions(make):
    ctx = make(FakeAvis([err(403, "VERIFICATION_FAILED"), err(403, "VERIFICATION_FAILED")]))
    p = approved(ctx)
    assert commit(ctx, p, PAY, retry_allowed=True).kind == "retry"
    assert ctx.pending is p and ctx.write_state is None
    result = commit(ctx, p, PAY, retry_allowed=False)
    assert result.kind == "offer" and result.message == CUSTOMER_COPY[ReasonCode.VERIFICATION_FAILED]
    keys = [r.headers["Idempotency-Key"] for r in ctx.fake.extend_bodies()]
    assert len(keys) == 2 and keys[0] != keys[1]
    # No guessing emails through a fresh card.
    assert check(ctx, "2027-06-18T14:00")["reason_code"] == "verification_failed"
    assert ctx.fake.count("/extend") == 2


@pytest.mark.parametrize(
    "response, reason",
    [
        (err(402, "PAYMENT_DECLINED"), ReasonCode.PAYMENT_DECLINED),
        (err(409, "RESERVATION_NOT_ACTIVE"), ReasonCode.NOT_ACTIVE),
        (err(404, "RESERVATION_NOT_FOUND"), ReasonCode.INTERNAL_ERROR),
        (err(400, "INVALID_EXTENSION"), ReasonCode.INTERNAL_ERROR),
    ],
)
def test_definite_refusals_charge_nothing_and_offer(make, response, reason):
    ctx = make(FakeAvis([response]))
    result = commit(ctx, approved(ctx), PAY, retry_allowed=True)
    assert result.kind == "offer" and result.message == CUSTOMER_COPY[reason]
    assert ctx.write_state is None and ctx.pending is None and ctx.transfer is None


def test_price_drift_shows_one_new_card_then_hands_off(make):
    ctx = make(FakeAvis([extend_ok()], quotes=(100.49, 129.0, 150.0)))
    first = approved(ctx)
    assert commit(ctx, first, PAY, retry_allowed=True).kind == "recard"
    second = ctx.pending
    assert second is not first and second.total == extend.money(129.0) and not second.shown
    second.shown = second.approved = True
    result = commit(ctx, second, PAY, retry_allowed=True)
    assert result.kind == "offer" and result.message == CUSTOMER_COPY[ReasonCode.INTERNAL_ERROR]
    assert ctx.fake.count("/extend") == 0


def test_gates_rerun_at_commit(make):
    ctx = make(FakeAvis([extend_ok()]))
    p = approved(ctx)
    ctx.clock["now"] = datetime(2027, 6, 16, 22, 0, tzinfo=UTC) + timedelta(hours=1)  # now overdue
    result = commit(ctx, p, PAY, retry_allowed=True)
    assert result.message == CUSTOMER_COPY[ReasonCode.OVERDUE_BEYOND_POLICY]
    assert ctx.fake.count("/extend") == 0
    assert any(e.get("stage") == "commit" and e.get("allowed") is False for e in events(ctx))


def test_ctrl_c_after_send_files_outcome_unknown_with_the_key(make):
    ctx = make(FakeAvis([KeyboardInterrupt()]))
    with pytest.raises(KeyboardInterrupt):
        commit(ctx, approved(ctx), PAY, retry_allowed=True)
    assert ctx.transfer.reason_code == "outcome_unknown" and ctx.write_state == "unknown"
    assert ctx.transfer.extend["idempotency_key"] in ctx.tracer.path.read_text()


def test_payment_repr_hides_everything():
    assert not any(s in repr(PAY) for s in (EMAIL, CVV, ZIP))
