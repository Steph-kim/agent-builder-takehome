"""CLI loop with the model stubbed: scrubbing, transfer ends the chat, outcome ledger, no evals backdoor."""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import openai
import pytest

from avis_agent import agent as agent_mod
from avis_agent import cli
from avis_agent.cli import CARD_CLOSED, CHARGED
from avis_agent.config import Settings
from avis_agent.extend import NOTHING_CHANGED, CommitResult, PendingExtension, render_receipt
from avis_agent.handoff import request_transfer, transfer
from avis_agent.reasons import ReasonCode
from avis_agent.trace import Tracer

SETTINGS = Settings(avis_api_url="http://avis.test", avis_api_key="k", openai_api_key="k")


def run_chat(tmp_path, monkeypatch, lines, respond, secrets=(), histories=None):
    """Feed `lines` as customer input (and `secrets` to hidden prompts); `respond(history, ctx)` stands in
    for the model. `histories` collects what the model was sent each turn."""

    async def fake_respond(agent, history, ctx):
        if histories is not None:
            histories.append(list(history))
        reply = respond(history, ctx)
        return reply, [*history, {"role": "assistant", "content": reply}]

    monkeypatch.setattr(cli, "respond", fake_respond)
    feed, out = iter(lines), []

    def read(prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    hidden = iter(secrets)
    tracer = Tracer("s1", tmp_path)
    asyncio.run(
        cli.chat(
            SETTINGS,
            kb=None,
            read=read,
            write=out.append,
            read_secret=lambda prompt: next(hidden),
            tracer=tracer,
            handoff_log=tmp_path / "h.jsonl",
        )
    )
    events = [json.loads(line) for line in tracer.path.read_text().splitlines()]
    return out, events


def test_card_number_never_reaches_model_or_log(tmp_path, monkeypatch):
    seen = []
    out, events = run_chat(
        tmp_path, monkeypatch, ["my card is 4111 1111 1111 1111"], lambda h, ctx: seen.append(h[-1]) or "ok"
    )
    assert "4111" not in json.dumps(seen) and "4111" not in json.dumps(events)
    assert [e["redacted"] for e in events if e["event"] == "customer.msg"] == [["card_number"]]
    assert events[-1] == {**events[-1], "event": "outcome", "outcomes": ["abandoned"]}


def test_transfer_prints_code_summary_and_ends_chat(tmp_path, monkeypatch):
    def model(history, ctx):
        if "human" in history[-1]["content"]:
            request_transfer(ctx, "customer_requested", "wants a person.")
            return "Connecting you."
        return "ok"

    out, events = run_chat(tmp_path, monkeypatch, ["hi", "get me a human", "are you there?"], model)
    assert out[0] == "Agent: How can I help?"
    assert "— Connecting you with a representative —" in out[-1]
    assert "Connecting you." not in out  # the model's closing line is replaced, not printed
    assert "person." in out[-1] and "person.." not in out[-1]
    assert sum(e["event"] == "customer.msg" for e in events) == 2  # third line never read
    assert events[-1]["outcomes"] == ["handed_off:customer_requested"]


def test_exit_word_and_blank_lines(tmp_path, monkeypatch):
    out, events = run_chat(tmp_path, monkeypatch, ["", "  ", "exit", "never read"], lambda h, ctx: "ok")
    assert out == ["Agent: How can I help?"]
    assert events[0]["event"] == "session.start" and events[0]["prompt_hash"] and events[0]["kb_hash"]


def test_model_outage_transfers(tmp_path, monkeypatch):
    async def down(*args, **kwargs):
        raise openai.APIConnectionError(request=None)

    monkeypatch.setattr(agent_mod.Runner, "run", down)
    tracer, out = Tracer("s1", tmp_path), []
    feed = iter(["hello"])
    asyncio.run(
        cli.chat(
            SETTINGS,
            kb=None,
            read=lambda p: next(feed),
            write=out.append,
            tracer=tracer,
            handoff_log=tmp_path / "h.jsonl",
        )
    )
    assert len(out) == 2 and out[1].startswith(f"Agent: {agent_mod.MODEL_DOWN}\n— Connecting you")
    assert json.loads(tracer.path.read_text().splitlines()[-1])["outcomes"] == ["handed_off:internal_error"]


def test_ctrl_c_mid_await_is_interrupted_not_error(tmp_path):
    t = Tracer("s1", tmp_path)
    with pytest.raises(asyncio.CancelledError), t:
        raise asyncio.CancelledError
    assert json.loads(t.path.read_text())["outcomes"] == ["interrupted"]


def test_cli_never_imports_evals():
    imports = re.compile(r"^\s*(from|import)\s+evals\b", re.M)
    assert not [p.name for p in Path(cli.__file__).parent.glob("*.py") if imports.search(p.read_text())]


# --- extension card → y/n → payment → commit (commit stubbed; tests/test_extend.py covers it) -------

LAX = ZoneInfo("America/Los_Angeles")
CHARGES = {
    "daily_rate": 45.99,
    "extension_days": 2,
    "subtotal": 91.98,
    "late_fee": 0.0,
    "one_way_fee": 0.0,
    "taxes_and_fees": 8.51,
    "total_charged": 100.49,
    "currency": "USD",
}
RESPONSE = {
    "confirmation_number": "EXT-1",
    "extension_details": {"new_return_datetime": "2027-06-17T14:00:00-07:00"},
    "charges": CHARGES,
}


def pending():
    return PendingExtension(
        reservation_id="AVS-29471835",
        new_return_local="2027-06-17T14:00",
        new_return=datetime(2027, 6, 17, 14, tzinfo=LAX),
        current_return=datetime(2027, 6, 15, 14, tzinfo=LAX),
        charges=CHARGES,
        card={"type": "Visa", "last_four": "4832"},
    )


def model_offering_card(history, ctx):
    if "extend" in history[-1]["content"]:
        ctx.pending = pending()
        return "Please review the card below."
    return f"model saw: {history[-1]['content']}"


def stub_commit(monkeypatch, *results):
    calls, queue = [], list(results)

    def fake(ctx, p, payment, *, retry_allowed):
        calls.append((payment, retry_allowed))
        result = queue.pop(0)
        if result.kind in ("resolved", "offer", "transfer"):
            ctx.pending = None
        if result.kind == "transfer":
            transfer(ctx, ReasonCode.OUTCOME_UNKNOWN, "")
        return result

    monkeypatch.setattr(cli, "commit", fake)
    return calls


def test_yes_collects_payment_off_model_and_prints_the_receipt(tmp_path, monkeypatch):
    receipt = render_receipt(RESPONSE, pending())
    calls = stub_commit(monkeypatch, CommitResult("resolved", receipt))
    histories = []
    out, events = run_chat(
        tmp_path,
        monkeypatch,
        ["extend to Thursday", "Y!", "a@b.co", "thanks"],
        model_offering_card,
        secrets=["8641", "97035"],
        histories=histories,
    )
    card = next(o for o in out if o.startswith("──── Confirm"))
    assert "$100.49 USD" in card and "Visa ending 4832" in card
    assert receipt in out
    ((payment, retry_allowed),) = calls
    assert (payment.email, payment.cvv, payment.billing_zip) == ("a@b.co", "8641", "97035") and retry_allowed
    after = json.dumps(histories[-1])
    assert "EXT-1" in after and "Extension confirmed" in after
    for secret in ("a@b.co", "8641", "97035", "4832"):
        assert secret not in after and secret not in tmp_path.joinpath("s1.jsonl").read_text()
    assert [e["decision"] for e in events if e["event"] == "approval"] == ["y"]


def test_no_prints_one_code_line_and_waits(tmp_path, monkeypatch):
    calls = stub_commit(monkeypatch)
    histories = []
    out, events = run_chat(
        tmp_path, monkeypatch, ["extend to Thursday", "n"], model_offering_card, histories=histories
    )
    assert out[-1] == f"Agent: {NOTHING_CHANGED}" and not calls
    assert len(histories) == 1  # the model wasn't run again for the "n"


def test_a_declined_card_tells_the_model_it_is_gone(tmp_path, monkeypatch):
    """Without the note the model saw check_extension's old "ready" and pointed to a phantom card."""
    stub_commit(monkeypatch)
    histories = []
    run_chat(
        tmp_path,
        monkeypatch,
        ["extend to Thursday", "n", "go ahead with it after all"],
        model_offering_card,
        histories=histories,
    )
    after = histories[-1]
    note = after.index({"role": "system", "content": CARD_CLOSED})
    assert after[note - 1]["content"] == NOTHING_CHANGED and after[note + 1]["role"] == "user"


@pytest.mark.parametrize("decisions", [["n"], ["n", "n"], ["y"], ["n", "y"], ["n", "n", "y"]])
def test_card_closed_note_iff_the_last_card_closed_uncharged(tmp_path, monkeypatch, decisions):
    """Invariant at the next model call: one CARD_CLOSED note if the last card was declined, none otherwise,
    and a CHARGED note right after every receipt.
    kills: the note left beside a later receipt (seen live: the model then denied a real charge);
    kills: notes piling up across declines; kills: the note never added;
    kills: a receipt the model doesn't trust (seen live after n → y: a charged customer told to approve)."""
    receipt = render_receipt(RESPONSE, pending())
    stub_commit(monkeypatch, *[CommitResult("resolved", receipt) for d in decisions if d == "y"])
    lines = []
    for d in decisions:
        lines += ["extend to Thursday", d] + (["a@b.co"] if d == "y" else [])
    histories = []
    run_chat(
        tmp_path,
        monkeypatch,
        [*lines, "thanks"],
        model_offering_card,
        secrets=["8641", "97035"] * decisions.count("y"),
        histories=histories,
    )
    notes = sum(h.get("content") == CARD_CLOSED for h in histories[-1])
    assert notes == (1 if decisions[-1] == "n" else 0)
    after = histories[-1]
    receipts = [i for i, h in enumerate(after) if str(h.get("content", "")).startswith("Extension confirmed")]
    assert len(receipts) == decisions.count("y")
    assert all(after[i + 1] == {"role": "system", "content": CHARGED} for i in receipts)


def test_other_text_at_the_card_is_a_no_scrubbed_and_answered_once(tmp_path, monkeypatch):
    stub_commit(monkeypatch)
    out, events = run_chat(
        tmp_path,
        monkeypatch,
        ["extend to Thursday", "yes cvv 8641 but does it include tax?"],
        model_offering_card,
    )
    assert out[-2] == f"Agent: {NOTHING_CHANGED}"
    assert out[-1].startswith("Agent: model saw:") and "8641" not in out[-1]
    assert sum(o.startswith("Agent: model saw:") for o in out) == 1
    assert [e["decision"] for e in events if e["event"] == "approval"] == ["other"]
    assert "8641" not in tmp_path.joinpath("s1.jsonl").read_text()


def test_wrong_details_retry_once(tmp_path, monkeypatch):
    calls = stub_commit(
        monkeypatch, CommitResult("retry", "Try again."), CommitResult("offer", "Want a representative?")
    )
    out, _ = run_chat(
        tmp_path,
        monkeypatch,
        ["extend to Thursday", "y", "a@b.co", "a@b.co"],
        model_offering_card,
        secrets=["1", "2", "3", "4"],
    )
    assert [r for _, r in calls] == [True, False]
    assert out[-1] == "Agent: Want a representative?"


def test_transfer_during_commit_prints_summary_and_ends(tmp_path, monkeypatch):
    stub_commit(monkeypatch, CommitResult("transfer"))
    out, events = run_chat(
        tmp_path,
        monkeypatch,
        ["extend to Thursday", "y", "a@b.co", "never read"],
        model_offering_card,
        secrets=["8641", "97035"],
    )
    assert "It may have." in out[-1] and "— Connecting you with a representative —" in out[-1]
    assert sum(e["event"] == "customer.msg" for e in events) == 1  # "never read" was never read
    assert events[-1]["outcomes"] == ["handed_off:outcome_unknown"]


def test_no_card_when_the_same_turn_transferred(tmp_path, monkeypatch):
    def model(history, ctx):
        ctx.pending = pending()
        request_transfer(ctx, "safety_incident", "car is making a grinding noise")
        return "Getting you help."

    out, _ = run_chat(tmp_path, monkeypatch, ["extend it, also the car is grinding"], model)
    assert not any(o.startswith("──── Confirm") for o in out)


def test_citations_are_logged_but_never_printed(tmp_path, monkeypatch):
    """kills: [kb_…] ids shown to the customer; kills: the ids dropped from the log (grounding audit)."""
    reply = "Grace is 30 minutes [kb_ext_01]. Extending is cheaper [kb_ext_05][kb_fee_02] [kb_a_01, kb_b_02]."
    out, events = run_chat(tmp_path, monkeypatch, ["grace?"], lambda h, c: reply)
    shown = next(o for o in out if o.startswith("Agent: Grace"))
    assert shown == "Agent: Grace is 30 minutes. Extending is cheaper."
    msg = next(e for e in events if e["event"] == "agent.msg" and e["text"].startswith("Grace"))
    assert msg["text"] == shown.removeprefix("Agent: ")
    assert msg["cites"] == ["kb_ext_01", "kb_ext_05", "kb_fee_02", "kb_a_01", "kb_b_02"]
