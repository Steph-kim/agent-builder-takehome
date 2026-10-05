"""CLI loop with the model stubbed: scrubbing, transfer ends the chat, outcome ledger, no evals backdoor."""

import asyncio
import json
import re
from pathlib import Path

import openai
import pytest

from avis_agent import agent as agent_mod
from avis_agent import cli
from avis_agent.config import Settings
from avis_agent.handoff import request_transfer
from avis_agent.trace import Tracer

SETTINGS = Settings(avis_api_url="http://avis.test", avis_api_key="k", openai_api_key="k")


def run_chat(tmp_path, monkeypatch, lines, respond):
    """Feed `lines` as customer input; `respond(history, ctx)` stands in for the model."""

    async def fake_respond(agent, history, ctx):
        reply = respond(history, ctx)
        return reply, [*history, {"role": "assistant", "content": reply}]

    monkeypatch.setattr(cli, "respond", fake_respond)
    feed, out = iter(lines), []

    def read(prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    tracer = Tracer("s1", tmp_path)
    asyncio.run(
        cli.chat(
            SETTINGS, kb=None, read=read, write=out.append, tracer=tracer, handoff_log=tmp_path / "h.jsonl"
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
