"""Agent tools. Each one is a plain function over `AgentContext` (tested directly) plus a thin
SDK wrapper that traces the call.

Lookup (D5): the model sees nothing about a reservation until code has matched the last name.
Wrong name, unknown id and malformed id all return the same generic result, so the reply never
says which field was wrong. Hitting `Thresholds.max_failed_lookups` locks lookup for the session
and *offers* a handoff (handoff.py) — the customer can keep asking policy questions.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agents import RunContextWrapper, function_tool

from . import extend
from .client import AvisAPIError, AvisClient, AvisUnavailable
from .config import Thresholds
from .handoff import HANDOFF_LOG, Handoff, offer, request_transfer
from .kb import KnowledgeBase
from .privacy import scrub
from .reasons import ReasonCode
from .timeutil import format_for_customer, resolve_zone, to_local
from .trace import Tracer

NOT_VERIFIED = (
    "I couldn't verify a reservation with those details. Please check the reservation number "
    "and the last name on the booking."
)
ONE_RESERVATION = (
    "I can only work on one reservation per chat. Once we're done here, you can start a new chat "
    "for the other one, or I can pass it to a representative now."
)


@dataclass
class AgentContext:
    """Per-session state handed to every tool via the SDK run context. One process = one session."""

    client: AvisClient
    tracer: Tracer
    thresholds: Thresholds
    kb: KnowledgeBase | None = None  # built once at startup (one embedding call); search_kb needs it
    reservation: dict | None = None  # raw API record, set only after the last-name check passes
    failed_lookups: int = 0
    quote: dict | None = None  # latest quote {total, new_return, extension_days}; goes in the handoff packet
    pending: extend.PendingExtension | None = None  # priced extension awaiting the customer's y/n on the card
    write_state: str | None = None  # None | in_flight | committed | unknown — one extend write per session
    last_write: dict | None = None  # key + approved vs actual, for the on-call packet
    pilot_locations: frozenset[str] = frozenset()
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))  # frozen in tests
    notify: Callable[[str], None] = field(
        default=lambda msg: None
    )  # progress lines ("Checking availability…")
    gates_hit: list[str] = field(default_factory=list)
    # A code-raised reason offered to the customer; `handoff_to_human` files under it if they accept.
    pending_handoff: ReasonCode | None = None
    # Set once a transfer is filed; the CLI prints the summary and ends the chat after this turn.
    transfer: Handoff | None = None
    handoff_log: Path = HANDOFF_LOG

    @property
    def lookup_locked(self) -> bool:
        return self.failed_lookups >= self.thresholds.max_failed_lookups


# --- lookup ------------------------------------------------------------------------------------


def lookup(ctx: AgentContext, reservation_id: str, last_name: str) -> dict[str, Any]:
    if ctx.lookup_locked:
        return {"verified": False, "locked": True, **offer(ctx, ReasonCode.VERIFICATION_FAILED)}
    if not _words(last_name):
        return {"verified": False, "message": "Ask the customer for the last name on the booking."}

    wanted = normalize_reservation_id(reservation_id)
    if ctx.reservation is not None:
        if wanted != ctx.reservation["reservation_id"]:
            return {
                "verified": False,
                "error": ReasonCode.SECOND_RESERVATION.value,
                "message": ONE_RESERVATION,
            }
        record = ctx.reservation  # re-check the name, but don't refetch
    else:
        try:
            record = ctx.client.get_reservation(wanted)
        except AvisUnavailable:
            return {"verified": False, **offer(ctx, ReasonCode.SERVICE_UNAVAILABLE)}
        except AvisAPIError as e:
            if e.status != 404:
                return {"verified": False, **offer(ctx, ReasonCode.INTERNAL_ERROR)}
            record = None

    if record is None or not last_name_matches(record.get("customer_name", ""), last_name):
        return _failed(ctx)
    ctx.reservation = record
    if ctx.pending_handoff in _TRANSIENT:  # the outage passed; don't file a stale reason later
        ctx.tracer.withdraw_offer(ctx.pending_handoff.value)
        ctx.pending_handoff = None
    return reservation_view(record, ctx.now())


def normalize_reservation_id(raw: str) -> str:
    """'29471835', 'avs 2947 1835', 'AVS–29471835' → 'AVS-29471835'. Customers type the number the way
    it looks on paper; a format slip shouldn't burn a verification attempt. Anything else passes through
    (upper-cased) and fails as an unknown id."""
    compact = re.sub(r"[\W_]", "", raw).upper()
    if re.fullmatch(r"(AVS)?\d{8}", compact):
        return "AVS-" + compact[-8:]
    return raw.strip().upper()


_TRANSIENT = frozenset({ReasonCode.SERVICE_UNAVAILABLE, ReasonCode.INTERNAL_ERROR})


def _failed(ctx: AgentContext) -> dict[str, Any]:
    ctx.failed_lookups += 1
    if ctx.lookup_locked:
        return {"verified": False, "locked": True, **offer(ctx, ReasonCode.VERIFICATION_FAILED)}
    return {"verified": False, "message": NOT_VERIFIED}


NAME_PARTICLES = frozenset(
    {"de", "del", "la", "las", "los", "da", "di", "du", "van", "von", "der", "den", "le"}
)


def last_name_matches(customer_name: str, given: str) -> bool:
    """A run of whole words from the booked name that includes a surname word (not the first name,
    not a bare particle). 'García' and 'López' match 'María García López'; 'María', 'de', 'arc' don't.
    Covers two-part and hyphenated surnames; typos still fail (a near-miss is still a miss)."""
    full, words = _words(customer_name), _words(given)
    if not words or not full:
        return False
    surname_idx = range(1, len(full)) if len(full) > 1 else range(1)
    for start in range(len(full) - len(words) + 1):
        if full[start : start + len(words)] == words and any(
            i in surname_idx and full[i] not in NAME_PARTICLES for i in range(start, start + len(words))
        ):
            return True
    return False


def _words(name: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^\w']", " ", folded).split()


def reservation_view(record: dict, now: datetime) -> dict[str, Any]:
    """What the model may see once verified: dates (local, with weekday), locations, vehicle class and
    its generic description, rate, and the current time at the return location (for "Friday", "tomorrow").
    Never the name, address, plate, exact make/model, color, customer id or card."""
    pickup, ret = record["pickup_location"], record["return_location"]
    dates = record["dates"]
    return {
        "verified": True,
        "reservation_id": record["reservation_id"],
        "status": record["status"],
        "membership_status": record.get("membership_status"),
        "vehicle_type": record["vehicle"]["type"],
        "vehicle_description": record["vehicle"].get("description"),  # class text, e.g. "… or similar"
        "pickup_location": {"code": pickup["code"], "name": pickup["name"]},
        "return_location": {"code": ret["code"], "name": ret["name"]},
        "pickup_datetime": _local(dates["pickup_datetime"], pickup["code"], dates),
        "current_return_datetime": _local(dates["current_return_datetime"], ret["code"], dates),
        "daily_rate": record["pricing"]["daily_rate"],
        "currency": record["pricing"]["currency"],
        "now_at_return_location": _local(now.isoformat(), ret["code"], dates),
    }


def _local(iso: str, location_code: str, dates: dict) -> str:
    zone = resolve_zone(location_code, list(dates.values()))
    return format_for_customer(to_local(iso, zone[0])) if zone else iso


# --- search_kb ---------------------------------------------------------------------------------

NO_KB_MATCH = (
    "Nothing in the policy information covers this. Tell the customer it isn't covered; don't answer "
    "from general knowledge. If they need an answer, offer a representative (kb_gap)."
)


async def search(ctx: AgentContext, query: str) -> dict[str, Any]:
    hits = await ctx.kb.search(query)
    if not hits:
        return {"results": [], "message": NO_KB_MATCH}
    # One info_only per run of policy questions; a later handoff or extension adds its own outcome.
    if not ctx.tracer.outcomes or ctx.tracer.outcomes[-1] != "info_only":
        ctx.tracer.add_outcome("info_only")
    return {"results": [h.to_tool_dict() for h in hits]}


# --- tracing + SDK wrappers --------------------------------------------------------------------


def traced(ctx: AgentContext, name: str, args: dict, fn: Callable[[], dict]) -> dict:
    """Emit tool.call/tool.result around a tool body. An unexpected crash becomes a generic error
    result (logged by type only) so the model can hand off instead of the chat dying."""
    ctx.tracer.emit("tool.call", name=name, args=args)
    try:
        result = fn()
    except Exception as e:
        ctx.tracer.emit("tool.result", name=name, error=type(e).__name__)
        return offer(ctx, ReasonCode.INTERNAL_ERROR)
    ctx.tracer.emit("tool.result", name=name, result=result)
    return result


async def traced_async(ctx: AgentContext, name: str, args: dict, fn: Callable[[], Awaitable[dict]]) -> dict:
    """`traced` for async tool bodies."""
    ctx.tracer.emit("tool.call", name=name, args=args)
    try:
        result = await fn()
    except Exception as e:
        ctx.tracer.emit("tool.result", name=name, error=type(e).__name__)
        return offer(ctx, ReasonCode.INTERNAL_ERROR)
    ctx.tracer.emit("tool.result", name=name, result=result)
    return result


@function_tool
def lookup_reservation(wrapper: RunContextWrapper[AgentContext], reservation_id: str, last_name: str) -> dict:
    """Look up the customer's reservation. Needs the reservation number (e.g. AVS-12345678) and the
    last name on the booking, both as the customer gave them. Returns the reservation only if they
    match; otherwise a message to relay as-is."""
    ctx = wrapper.context
    args = {"reservation_id": reservation_id, "last_name": last_name}
    return traced(ctx, "lookup_reservation", args, lambda: lookup(ctx, reservation_id, last_name))


@function_tool
async def search_kb(wrapper: RunContextWrapper[AgentContext], query: str) -> dict:
    """Search Avis policy articles. Use for any policy question (fees, grace periods, extensions, fuel,
    tolls, etc.) and answer only from the results, citing article ids. Phrase the query as the policy
    topic, not the customer's whole message. Results are ordered most authoritative first; an article
    with a `note` is outdated — prefer the official one it conflicts with."""
    ctx = wrapper.context
    args = {"query": scrub(query).text}
    return await traced_async(ctx, "search_kb", args, lambda: search(ctx, query))


@function_tool
def check_extension(wrapper: RunContextWrapper[AgentContext], new_return_local: str) -> dict:
    """Check and price an extension for the verified reservation. Call only after the customer has confirmed
    the new return date and time. new_return_local: local wall-clock time at the return location, formatted
    YYYY-MM-DDTHH:MM. If it's ready, the system shows the customer a confirmation card to approve; otherwise
    relay the message."""
    ctx = wrapper.context
    args = {"new_return_local": new_return_local}
    return traced(ctx, "check_extension", args, lambda: extend.check(ctx, new_return_local))


@function_tool
def handoff_to_human(wrapper: RunContextWrapper[AgentContext], reason_code: str, note: str) -> dict:
    """Transfer the customer to a representative. Call only when the customer asks for a person, accepts
    an offered handoff, or the situation needs one (accident, injury or a car that may be unsafe, dispute,
    cancel/modify/upgrade, different card, unsupported language, a policy question the KB doesn't cover).
    reason_code: one of customer_requested, safety_incident, dispute, unsupported_intent, payment_change,
    second_reservation, language_unsupported, kb_gap (when accepting an offer, use customer_requested).
    note: one sentence on what the customer asked for, as they put it (claims as claims, not facts), no
    names, card or contact details. After this, say one
    short closing line only."""
    ctx = wrapper.context
    args = {"reason_code": reason_code, "note": scrub(note).text}
    return traced(ctx, "handoff_to_human", args, lambda: request_transfer(ctx, reason_code, note))
