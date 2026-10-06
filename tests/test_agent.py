"""Agent wiring: thresholds never reach the model (D6), tool set, and the turn-limit fallback."""

import asyncio
import dataclasses
import json
import re

import pytest
from agents import MaxTurnsExceeded, Runner

from avis_agent import agent as agent_mod
from avis_agent.agent import INSTRUCTIONS, build_agent, respond
from avis_agent.config import Settings, Thresholds
from avis_agent.reasons import CUSTOMER_COPY, ReasonCode
from avis_agent.tools import AgentContext
from avis_agent.trace import Tracer

SETTINGS = Settings(avis_api_url="http://avis.test", avis_api_key="k", openai_api_key="k")


def model_visible_text(a) -> str:
    return INSTRUCTIONS + "".join(t.description + json.dumps(t.params_json_schema) for t in a.tools)


@pytest.mark.parametrize("name", [f.name for f in dataclasses.fields(Thresholds)])
def test_no_threshold_value_reaches_the_model(name):
    value = getattr(Thresholds(), name)
    text = model_visible_text(build_agent(SETTINGS))
    numbers = {float(n) for n in re.findall(r"(?<![\w.-])\d+(?:\.\d+)?(?![\w.])", text)}
    assert float(value) not in numbers


def test_tools_and_settings():
    a = build_agent(dataclasses.replace(SETTINGS, reasoning_effort="low"))
    # No tool can charge: the extend write is reachable only from the CLI after a "y" (extend.commit).
    names = [t.name for t in a.tools]
    assert names == ["lookup_reservation", "search_kb", "check_extension", "handoff_to_human"]
    assert a.model_settings.parallel_tool_calls is False
    assert a.model_settings.reasoning.effort == "low"
    assert build_agent(SETTINGS).model_settings.reasoning is None
    assert agent_mod.RUN_CONFIG.tracing_disabled is True


def test_turn_limit_becomes_internal_error_offer(tmp_path, monkeypatch):
    async def looping(*args, **kwargs):
        raise MaxTurnsExceeded("loop")

    monkeypatch.setattr(Runner, "run", looping)
    ctx = AgentContext(
        client=None, tracer=Tracer("s1", tmp_path), thresholds=Thresholds(), handoff_log=tmp_path / "h.jsonl"
    )
    history = [{"role": "user", "content": "hi"}]
    reply, new_history = asyncio.run(respond(build_agent(SETTINGS), history, ctx))
    assert reply == CUSTOMER_COPY[ReasonCode.INTERNAL_ERROR]
    assert ctx.pending_handoff == ReasonCode.INTERNAL_ERROR and ctx.transfer is None
    assert new_history == [*history, {"role": "assistant", "content": reply}]
    assert '"gate": "max_turns"' in ctx.tracer.path.read_text()


def test_a_silent_model_server_transfers_within_the_timeout(tmp_path, monkeypatch):
    """kills: the SDK's 10-minute default timeout (a dropped connection hung sims 15+ min, 2026-10-06)."""
    import socket
    import time

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()  # accepts the connection, never answers
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{srv.getsockname()[1]}/v1")
    monkeypatch.setattr(agent_mod, "MODEL_TIMEOUT_S", 0.3)
    ctx = AgentContext(
        client=None, tracer=Tracer("s1", tmp_path), thresholds=Thresholds(), handoff_log=tmp_path / "h.jsonl"
    )
    started = time.monotonic()
    reply, _ = asyncio.run(respond(build_agent(SETTINGS), [{"role": "user", "content": "hi"}], ctx))
    srv.close()
    assert reply == agent_mod.MODEL_DOWN and ctx.transfer is not None
    assert time.monotonic() - started < 5
