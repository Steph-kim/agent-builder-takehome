"""Lookup tool (D5): last-name gate, generic failure, 3-failure lock. Offline via httpx.MockTransport."""

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from agents.tool_context import ToolContext

from avis_agent.client import AvisClient
from avis_agent.config import Thresholds
from avis_agent.kb import LEGACY_NOTE, Article, KnowledgeBase
from avis_agent.reasons import CUSTOMER_COPY, ReasonCode
from avis_agent.tools import (
    NO_KB_MATCH,
    AgentContext,
    last_name_matches,
    lookup,
    lookup_reservation,
    normalize_reservation_id,
    search_kb,
    traced,
)
from avis_agent.trace import Tracer

FIXTURES = Path(__file__).parent / "fixtures"
SARAH = json.loads((FIXTURES / "AVS-29471835.json").read_text())
TOMAS = json.loads((FIXTURES / "AVS-99004050.json").read_text())
RECORDS = {r["reservation_id"]: r for r in (SARAH, TOMAS)}


def avis(status_override: int | None = None):
    """Serves fixtures by id; unknown ids 404. Records every request path."""
    seen: list[str] = []

    def handler(request: httpx.Request):
        seen.append(request.url.path)
        if status_override:
            return httpx.Response(status_override, json={"error": {"code": "X", "message": "x"}})
        rid = request.url.path.rsplit("/", 1)[-1]
        if rid in RECORDS:
            return httpx.Response(200, json=RECORDS[rid])
        return httpx.Response(404, json={"error": {"code": "RESERVATION_NOT_FOUND", "message": "nope"}})

    client = AvisClient("http://avis.test", "k", transport=httpx.MockTransport(handler), sleep=lambda s: None)
    return client, seen


@pytest.fixture
def ctx(tmp_path):
    client, seen = avis()
    c = AgentContext(
        client=client,
        tracer=Tracer("s1", tmp_path),
        thresholds=Thresholds(),
        handoff_log=tmp_path / "handoffs.jsonl",
    )
    c.seen = seen  # test-only handle on request log
    return c


def test_verified_view_has_local_dates_and_no_pii(ctx):
    out = lookup(ctx, "avs-29471835", "johnson")
    assert out["verified"] is True
    assert out["current_return_datetime"] == "Tuesday, June 15, 2027 at 2:00 PM PDT"
    assert out["vehicle_type"] == "midsize_sedan"
    assert out["vehicle_description"] == "Chevrolet Malibu or similar"
    raw = json.dumps(out)
    for pii in (
        "Sarah",
        "Johnson",
        "9217 Airport",
        "8ABC123",
        "4832",
        "2025 Chevrolet",  # exact make/model; the generic "Malibu or similar" is allowed
        "Silver",
        "CUST-847291",
        "Visa",
    ):
        assert pii not in raw, pii
    assert ctx.reservation == SARAH


@pytest.mark.parametrize(
    ("given", "ok"),
    [
        ("Rivera", True),
        ("RIVERA ", True),
        ("Tomas Rivera", True),  # accent folded
        ("Tomás Rivera", True),
        ("rivera.", True),
        ("Tomás", False),  # first name alone
        ("vera", False),  # suffix, not a whole word
        ("", False),
        ("Rivera Smith", False),
    ],
)
def test_last_name_match_rules(given, ok):
    assert last_name_matches("Tomás Rivera", given) is ok


@pytest.mark.parametrize(
    ("booked", "given", "ok"),
    [
        ("María García López", "García", True),  # two-part surname, first half
        ("María García López", "lopez", True),
        ("María García López", "Garcia-Lopez", True),
        ("María García López", "María", False),
        ("Ann Smith-Jones", "Smith", True),  # hyphenated
        ("Ann Smith-Jones", "Smith Jones", True),
        ("Ana de la Cruz", "Cruz", True),
        ("Ana de la Cruz", "de la Cruz", True),
        ("Ana de la Cruz", "de", False),  # particle alone
        ("Ana de la Cruz", "la", False),
        ("Sarah Johnson", "Jonson", False),  # typos stay misses
        ("Sarah Johnson", "Johnson Sarah", False),  # order matters
        ("Cher", "Cher", True),  # single-word booking
    ],
)
def test_multi_part_surnames(booked, given, ok):
    assert last_name_matches(booked, given) is ok


def test_wrong_name_and_unknown_id_are_indistinguishable(ctx):
    wrong_name = lookup(ctx, "AVS-29471835", "Smith")
    unknown_id = lookup(ctx, "AVS-00000000", "Johnson")
    assert wrong_name == unknown_id == {"verified": False, "message": wrong_name["message"]}
    assert ctx.reservation is None


def test_last_allowed_failure_locks_and_offers_handoff_without_more_api_calls(ctx):
    cap = ctx.thresholds.max_failed_lookups
    assert cap == 5
    lookup(ctx, "not an id!", "Johnson")  # malformed: counts, never hits the API
    for i in range(cap - 2):
        lookup(ctx, f"AVS-0000000{i}", "Johnson")
    assert ctx.pending_handoff is None and ctx.failed_lookups == cap - 1
    last = lookup(ctx, "AVS-29471835", "Smith")
    assert last["locked"] is True and ctx.pending_handoff is ReasonCode.VERIFICATION_FAILED
    assert last["message"] == CUSTOMER_COPY[ReasonCode.VERIFICATION_FAILED]
    assert ctx.transfer is None  # offered, not forced: the chat goes on
    calls = len(ctx.seen)
    after = lookup(ctx, "AVS-29471835", "Johnson")  # correct now, but too late
    assert after["locked"] is True and after.get("verified") is False
    assert len(ctx.seen) == calls and ctx.reservation is None


def test_outage_does_not_count_toward_cap(tmp_path):
    client, _ = avis(status_override=503)
    ctx = AgentContext(client=client, tracer=Tracer("s1", tmp_path), thresholds=Thresholds())
    for _ in range(5):
        out = lookup(ctx, "AVS-29471835", "Johnson")
    assert out["reason_code"] == "service_unavailable" and ctx.failed_lookups == 0


def test_second_reservation_refused_without_fetch_or_penalty(ctx):
    lookup(ctx, "AVS-29471835", "Johnson")
    calls = len(ctx.seen)
    out = lookup(ctx, "AVS-99004050", "Rivera")
    assert out["error"] == "second_reservation"
    assert len(ctx.seen) == calls and ctx.failed_lookups == 0 and ctx.reservation == SARAH


@pytest.mark.parametrize(
    "typed", ["AVS-29471835", "29471835", "AVS 2947 1835", "avs29471835", " AVS–29471835 ", "AVS_2947-1835"]
)
def test_reservation_id_format_slips_still_verify(ctx, typed):
    assert lookup(ctx, typed, "Johnson")["verified"] is True and ctx.failed_lookups == 0


@pytest.mark.parametrize("typed", ["2947183", "AVS-294718351", "XYZ-29471835", "AVS-2947183A"])
def test_wrong_length_or_prefix_is_not_rescued(typed):
    # kills: a lenient digit count that turns a 7- or 9-digit typo into some other well-formed id
    assert normalize_reservation_id(typed) == typed.upper()


def test_empty_last_name_does_not_hit_api_or_count(ctx):
    out = lookup(ctx, "AVS-29471835", "  ")
    assert out["verified"] is False and ctx.seen == [] and ctx.failed_lookups == 0


def test_sdk_tool_traces_without_last_name(ctx):
    wrapper = ToolContext(context=ctx, tool_name="lookup_reservation", tool_call_id="c1", tool_arguments="{}")
    args = json.dumps({"reservation_id": "AVS-29471835", "last_name": "Johnson"})
    out = asyncio.run(lookup_reservation.on_invoke_tool(wrapper, args))
    assert "verified" in str(out)
    raw = ctx.tracer.path.read_text()
    assert "Johnson" not in raw and "Sarah" not in raw
    events = [json.loads(line) for line in raw.splitlines()]
    assert [e["event"] for e in events] == ["tool.call", "tool.result"]
    assert events[-1]["result"]["verified"] is True
    assert events[0]["args"] == {"reservation_id": "AVS-29471835"}


def test_tool_crash_becomes_generic_error(ctx):
    ctx.client = None  # any attribute access on the client now raises
    out = traced(ctx, "lookup_reservation", {}, lambda: lookup(ctx, "AVS-29471835", "Johnson"))
    assert out["reason_code"] == "internal_error"
    assert json.loads(ctx.tracer.path.read_text().splitlines()[-1])["error"] == "AttributeError"


# --- search_kb ---------------------------------------------------------------------------------


def two_article_kb(query_vector):
    """Legacy article on axis 0, official on axis 1; every query embeds to `query_vector`."""
    articles = [
        Article(
            id="kb_old", title="Old grace", body="2 hours", last_updated="2020-01-01", authority="legacy"
        ),
        Article(
            id="kb_new",
            title="Grace",
            body="30 minutes",
            last_updated="2026-01-01",
            authority="official-policy",
        ),
    ]

    async def embed(texts):
        return np.array([query_vector] * len(texts))

    kb = KnowledgeBase(articles, np.eye(2), [0, 1], embed)
    kb.superseded_by = {}
    return kb


def call_search(ctx, query):
    wrapper = ToolContext(context=ctx, tool_name="search_kb", tool_call_id="c1", tool_arguments="{}")
    return asyncio.run(search_kb.on_invoke_tool(wrapper, json.dumps({"query": query})))


def test_search_kb_returns_bodies_official_first_but_logs_no_bodies(ctx):
    ctx.kb = two_article_kb([0.9, 0.4])
    ctx.kb.floor = 0.1
    out = str(call_search(ctx, "grace period"))
    assert out.index("kb_new") < out.index("kb_old")
    assert "30 minutes" in out and LEGACY_NOTE in out
    events = [json.loads(line) for line in ctx.tracer.path.read_text().splitlines()]
    assert events[0]["args"] == {"query": "grace period"}
    assert [r["id"] for r in events[1]["result"]["results"]] == ["kb_new", "kb_old"]
    assert "30 minutes" not in ctx.tracer.path.read_text()


def test_search_kb_with_no_match_says_so_instead_of_returning_nothing(ctx):
    ctx.kb = two_article_kb([0.1, 0.1])
    ctx.kb.floor = 0.9
    assert NO_KB_MATCH in str(call_search(ctx, "pets"))


def test_search_kb_crash_becomes_generic_error(ctx):
    ctx.kb = None
    assert "internal_error" in str(call_search(ctx, "grace period"))
    assert ctx.pending_handoff == ReasonCode.INTERNAL_ERROR
