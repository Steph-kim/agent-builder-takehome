"""Handoffs (D10, AOP §7): offered vs transferred, code's reason wins, packet is scrubbed and PII-free."""

import asyncio
import io
import json
from pathlib import Path

import pytest
from agents.tool_context import ToolContext

from avis_agent.config import Thresholds
from avis_agent.handoff import customer_summary, offer, request_transfer, transfer
from avis_agent.reasons import CUSTOMER_COPY, ReasonCode
from avis_agent.tools import AgentContext, handoff_to_human, lookup
from avis_agent.trace import Tracer

FIXTURES = Path(__file__).parent / "fixtures"
SARAH = json.loads((FIXTURES / "AVS-29471835.json").read_text())
CARD = "4111 1111 1111 1111"


@pytest.fixture
def ctx(tmp_path):
    return AgentContext(
        client=None,
        tracer=Tracer("s1", tmp_path, stderr=io.StringIO()),
        thresholds=Thresholds(),
        handoff_log=tmp_path / "handoffs.jsonl",
    )


def filed(ctx) -> list[dict]:
    if not ctx.handoff_log.exists():
        return []
    return [json.loads(line) for line in ctx.handoff_log.read_text().splitlines()]


def test_accepting_an_offer_files_codes_reason_not_models(ctx):
    offer(ctx, ReasonCode.VERIFICATION_FAILED)
    out = request_transfer(ctx, "customer_requested", "Wants to extend but couldn't verify")
    assert out["handed_off"] is True and out["reason_code"] == "verification_failed"
    [packet] = filed(ctx)
    assert packet["reason_code"] == "verification_failed"
    assert packet["model_reason"] == "customer_requested"
    assert packet["gates_hit"] == ["verification_failed"]
    assert ctx.pending_handoff is None and ctx.transfer is not None
    assert ctx.tracer.outcomes == ["handed_off:verification_failed"]


@pytest.mark.parametrize("bad", ["high_value", "outcome_unknown", "vip_override", ""])
def test_model_cannot_name_a_code_reason_or_invent_one(ctx, bad):
    out = request_transfer(ctx, bad, "x")
    assert out["error"] == "invalid_reason"
    assert ctx.transfer is None and filed(ctx) == [] and ctx.tracer.outcomes == []


def test_safety_beats_a_pending_offer(ctx):
    offer(ctx, ReasonCode.HIGH_VALUE)
    request_transfer(ctx, "safety_incident", "Was in a collision")
    assert ctx.transfer.reason_code == "safety_incident"
    assert customer_summary(ctx.transfer).startswith(CUSTOMER_COPY[ReasonCode.SAFETY_INCIDENT])


def test_model_reason_used_when_nothing_pending(ctx):
    request_transfer(ctx, "dispute", "Disputes a fuel charge")
    assert ctx.transfer.reason_code == "dispute" and ctx.transfer.model_reason is None


def test_offer_does_not_transfer_and_rejects_transfer_now_reasons(ctx):
    out = offer(ctx, ReasonCode.HIGH_VALUE)
    assert out["message"] == CUSTOMER_COPY[ReasonCode.HIGH_VALUE]
    assert ctx.transfer is None and ctx.tracer.outcomes == ["offered:high_value"]
    with pytest.raises(ValueError):
        offer(ctx, ReasonCode.OUTCOME_UNKNOWN)  # must transfer, never wait on the customer


def test_note_is_scrubbed_in_packet_and_trace(ctx):
    ctx.reservation = SARAH
    tool_ctx = ToolContext(context=ctx, tool_name="handoff_to_human", tool_call_id="c1", tool_arguments="{}")
    args = json.dumps({"reason_code": "payment_change", "note": f"Wants to pay with {CARD}"})
    asyncio.run(handoff_to_human.on_invoke_tool(tool_ctx, args))
    assert CARD not in ctx.handoff_log.read_text()
    assert CARD not in ctx.tracer.path.read_text()
    assert "[redacted]" in filed(ctx)[0]["note"]


def test_packet_carries_ids_not_pii(ctx):
    ctx.reservation = SARAH
    transfer(ctx, ReasonCode.CUSTOMER_REQUESTED, "Wants a person")
    raw = ctx.handoff_log.read_text()
    for pii in ("Sarah", "Johnson", "9217 Airport", "8ABC123", "4832", "CUST-847291"):
        assert pii not in raw, pii
    packet = filed(ctx)[0]
    assert packet["reservation_id"] == "AVS-29471835" and packet["session_id"] == "s1"
    assert packet["log_path"].endswith("s1.jsonl")


def test_second_transfer_keeps_the_first(ctx):
    transfer(ctx, ReasonCode.DISPUTE, "first")
    transfer(ctx, ReasonCode.KB_GAP, "second")
    assert len(filed(ctx)) == 1 and ctx.transfer.reason_code == "dispute"
    assert ctx.tracer.outcomes == ["handed_off:dispute"]


def test_summary_repeats_copy_only_for_transfer_reasons(ctx):
    ctx.reservation, ctx.quote = SARAH, {"total": 91.98}
    offer(ctx, ReasonCode.HIGH_VALUE)
    request_transfer(ctx, "customer_requested", "Extend to June 30")
    text = customer_summary(ctx.transfer)
    assert CUSTOMER_COPY[ReasonCode.HIGH_VALUE] not in text  # the offer was already said
    assert "AVS-29471835" in text and "$91.98" in text and "won't need to repeat" in text


def test_transfer_now_summary_keeps_may_have_charged_wording(ctx):
    transfer(ctx, ReasonCode.OUTCOME_UNKNOWN, "Extend timed out")
    assert "It may have." in customer_summary(ctx.transfer)


def test_summary_lead_comes_from_code_except_where_the_model_must_speak(ctx):
    transfer(ctx, ReasonCode.SAFETY_INCIDENT, "accident")
    text = customer_summary(ctx.transfer, fallback_lead="ok bye")
    assert "call 911" in text and "ok bye" not in text
    ctx.transfer = None
    transfer(ctx, ReasonCode.LANGUAGE_UNSUPPORTED, "")
    hola = "Solo puedo ayudar en inglés. Le conecto con un representante."
    assert customer_summary(ctx.transfer, fallback_lead=hola).startswith(hola)
    assert customer_summary(ctx.transfer).startswith(CUSTOMER_COPY[ReasonCode.LANGUAGE_UNSUPPORTED])
    ctx.transfer = None
    offer(ctx, ReasonCode.HIGH_VALUE)
    request_transfer(ctx, "customer_requested", "")
    assert customer_summary(ctx.transfer, fallback_lead="Connecting you.").startswith("Connecting you.")


def test_transient_offer_cleared_once_lookup_succeeds(ctx):
    offer(ctx, ReasonCode.SERVICE_UNAVAILABLE)

    class Up:
        def get_reservation(self, rid):
            return SARAH

    ctx.client = Up()
    assert lookup(ctx, "AVS-29471835", "Johnson")["verified"] is True
    assert ctx.pending_handoff is None and ctx.tracer.outcomes == []


def test_declined_offer_closes_as_offered_not_abandoned(ctx):
    offer(ctx, ReasonCode.OVERDUE_BEYOND_POLICY)
    offer(ctx, ReasonCode.OVERDUE_BEYOND_POLICY)  # re-offered on a later turn: still one request
    ctx.tracer.close()
    assert ctx.tracer.outcomes == ["offered:overdue_beyond_policy"]


def test_accepting_after_other_turns_replaces_the_offer(ctx):
    offer(ctx, ReasonCode.OVERDUE_BEYOND_POLICY)
    ctx.tracer.add_outcome("info_only")
    request_transfer(ctx, "customer_requested", "")
    assert ctx.tracer.outcomes == ["info_only", "handed_off:overdue_beyond_policy"]


def test_unwritable_handoff_log_still_transfers(ctx, tmp_path):
    ctx.handoff_log = tmp_path / "blocker" / "handoffs.jsonl"
    (tmp_path / "blocker").write_text("a file where a directory should be")
    transfer(ctx, ReasonCode.CUSTOMER_REQUESTED, "x")
    assert ctx.transfer is not None and ctx.tracer.outcomes == ["handed_off:customer_requested"]
