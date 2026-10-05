"""The extend path (AOP §4–6): evaluate → card → y/n → commit → receipt.

The model can only *check* (`tools.check_extension` → `check`). The charge itself, `commit`, is called by
the CLI after the customer answers yes to a code-rendered card — no agent tool can reach it, so the model
can't charge anyone, however it's prompted. Every assertion lives in `commit`: the pending is the one the
customer approved, every gate still passes, the re-quote equals the card, and nothing was written before in
this session.

Fail-closed after send: once the request may have reached Avis, anything other than a definite first-attempt
refusal (client.py) ends in `outcome_unknown` or `confirmation_mismatch`, never "nothing was charged".
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .client import AvisAPIError, AvisUnavailable, OutcomeUnknown
from .handoff import offer, transfer
from .policy import money, reservation_gate, value_gate
from .reasons import ReasonCode
from .timeutil import (
    InvalidLocalTime,
    availability_dates,
    format_for_customer,
    parse_local,
    resolve_zone,
    to_api,
    to_local,
)

if TYPE_CHECKING:
    from .tools import AgentContext

CARD_READY = (
    "The system is now showing the customer a confirmation card with this price; they answer it there. "
    "Say one short line pointing them to it. Don't say the change is made — the system confirms it."
)
BAD_TIME = (
    "Use the local date and time at the return location as YYYY-MM-DDTHH:MM, e.g. 2027-06-17T14:00, and make "
    "sure that time exists (clocks don't skip it)."
)
TIME_PASSED = "That time has already passed at the return location. Ask the customer for a later time."
NOT_VERIFIED = "Verify the reservation first (lookup_reservation)."
DETAILS_RETRY = "Those details didn't match our records. Please enter them once more."
PRICE_CHANGED = "The price changed since I showed it, so here is the updated total. Nothing has been charged."
NOTHING_CHANGED = "No problem — nothing has been changed."

# Raised by an earlier gate in this session, they block any further extension (no guessing emails/CVVs through
# fresh cards, no second charge).
LOCKS = (ReasonCode.VERIFICATION_FAILED, ReasonCode.PAYMENT_DECLINED)


@dataclass
class PendingExtension:
    """A priced, gate-passed extension waiting for the customer's answer on the card."""

    reservation_id: str
    new_return_local: str  # what the model sent; commit re-evaluates this exact input
    new_return: datetime  # aware, in the return location's zone
    current_return: datetime
    charges: dict
    card: dict  # card_on_file {type, last_four} — shown on the card only, never to the model or the log
    shown: bool = False
    approved: bool = False
    recards: int = 0  # price changes between card and commit for this approval (capped at 1)

    @property
    def total(self):
        return money(self.charges["total_charged"])


@dataclass(frozen=True)
class Payment:
    """Collected by the CLI after "y", passed to `commit`, dropped. Never stored on ctx, logged or repr'd."""

    email: str = field(repr=False)
    cvv: str = field(repr=False)
    billing_zip: str = field(repr=False)


@dataclass
class CommitResult:
    kind: str  # resolved | offer | transfer | recard | retry
    message: str = ""


# --- check ---------------------------------------------------------------------------------------


def check(ctx: AgentContext, new_return_local: str) -> dict[str, Any]:
    """`check_extension` body: the only place that stores a pending; a new check drops any stale card."""
    ctx.pending = None
    result = evaluate(ctx, new_return_local, stage="check")
    if not isinstance(result, PendingExtension):
        return result
    ctx.pending = result
    return {
        "ready": True,
        "new_return": format_for_customer(result.new_return),
        "charges": result.charges,
        "message": CARD_READY,
    }


def evaluate(ctx: AgentContext, new_return_local: str, *, stage: str) -> PendingExtension | dict[str, Any]:
    """Run every gate in order (AOP §4) and price the change. Stores nothing except `ctx.quote` (for the
    handoff packet). Returns a pending, or the dict the model relays."""
    for lock in LOCKS:
        if lock.value in ctx.gates_hit:
            return _refuse(ctx, stage, "locked", lock)
    if ctx.write_state is not None:
        return _refuse(ctx, stage, "one_write", ReasonCode.WRITE_ALREADY_DONE)
    if ctx.reservation is None:
        return {"error": "not_verified", "message": NOT_VERIFIED}
    rid = ctx.reservation["reservation_id"]
    try:
        record = ctx.client.get_reservation(rid)  # fresh: status and return time can change
    except AvisUnavailable:
        return _refuse(ctx, stage, "reservation", ReasonCode.SERVICE_UNAVAILABLE)
    except AvisAPIError:
        return _refuse(ctx, stage, "reservation", ReasonCode.INTERNAL_ERROR)
    ctx.reservation = record

    now = ctx.now()
    reason = reservation_gate(record, now, ctx.thresholds, ctx.pilot_locations)
    if reason:
        return _refuse(ctx, stage, "reservation", reason)

    ret = record["return_location"]["code"]
    zone = resolve_zone(ret, list(record["dates"].values()))
    if zone is None:
        return _refuse(ctx, stage, "time_zone", ReasonCode.INTERNAL_ERROR)
    tz = zone[0]
    try:
        new_return = parse_local(new_return_local, tz)
    except InvalidLocalTime:
        return {"error": "bad_time", "message": BAD_TIME}
    current_return = to_local(record["dates"]["current_return_datetime"], tz)
    if new_return <= current_return:
        return _refuse(ctx, stage, "later", ReasonCode.NOT_AN_EXTENSION)
    if new_return <= now:
        return {"error": "time_passed", "message": TIME_PASSED}

    vehicle_type = record["vehicle"]["type"]
    start, end = availability_dates(current_return, new_return, tz)
    if stage == "check":  # at commit the customer has just entered payment; no second progress line
        ctx.notify("Checking availability…")
    try:
        avail = ctx.client.get_availability(ret, vehicle_type, start, end)
    except AvisUnavailable:
        return _refuse(ctx, stage, "availability", ReasonCode.AVAILABILITY_UNKNOWN)
    except AvisAPIError:  # a 4xx here is our bug, not "unknown" (D6)
        return _refuse(ctx, stage, "availability", ReasonCode.INTERNAL_ERROR)
    if not avail.get("availability", {}).get("requested_type", {}).get("available", False):
        return _refuse(ctx, stage, "availability", ReasonCode.VEHICLE_UNAVAILABLE)

    try:
        charges = ctx.client.quote_extension(rid, to_api(new_return))["quote"]["charges"]
    except AvisUnavailable:
        return _refuse(ctx, stage, "quote", ReasonCode.SERVICE_UNAVAILABLE)
    except AvisAPIError:
        return _refuse(ctx, stage, "quote", ReasonCode.INTERNAL_ERROR)
    ctx.quote = {
        "total": float(money(charges["total_charged"])),
        "new_return": to_api(new_return),
        "extension_days": charges["extension_days"],
    }
    reason = value_gate(charges, ctx.thresholds)
    if reason:
        return _refuse(ctx, stage, "value", reason)

    ctx.tracer.emit("gate.decision", gate="all", allowed=True, stage=stage)
    return PendingExtension(
        reservation_id=rid,
        new_return_local=new_return_local,
        new_return=new_return,
        current_return=current_return,
        charges=charges,
        card=(record.get("payment") or {}).get("card_on_file") or {},
    )


def _refuse(ctx: AgentContext, stage: str, gate: str, reason: ReasonCode) -> dict[str, Any]:
    ctx.tracer.emit("gate.decision", gate=gate, allowed=False, reason_code=reason.value, stage=stage)
    return offer(ctx, reason)


# --- commit --------------------------------------------------------------------------------------


def commit(
    ctx: AgentContext, pending: PendingExtension, payment: Payment, *, retry_allowed: bool
) -> CommitResult:
    """Charge the approved extension. Called only by the CLI after "y" on the card."""
    if pending is not ctx.pending or not pending.approved:
        raise RuntimeError("commit without an approved pending")  # a bug — never write

    fresh = evaluate(ctx, pending.new_return_local, stage="commit")
    if not isinstance(fresh, PendingExtension):
        ctx.pending = None
        if "reason_code" not in fresh:  # e.g. the time passed while the card was open
            fresh = offer(ctx, ReasonCode.INTERNAL_ERROR)
        return CommitResult("offer", fresh["message"])
    if fresh.total != pending.total or fresh.charges.get("extension_days") != pending.charges.get(
        "extension_days"
    ):
        if pending.recards >= 1:
            ctx.pending = None
            ctx.tracer.emit("gate.decision", gate="price_drift", allowed=False, reason_code="internal_error")
            return CommitResult("offer", offer(ctx, ReasonCode.INTERNAL_ERROR)["message"])
        fresh.recards = pending.recards + 1
        ctx.pending = fresh  # the CLI shows this new card; no write
        return CommitResult("recard", PRICE_CHANGED)

    key = str(uuid.uuid4())
    new_return_api = to_api(pending.new_return)
    ctx.write_state = "in_flight"
    ctx.last_write = {
        "idempotency_key": key,
        "approved_total": float(pending.total),
        "approved_return": new_return_api,
    }
    # Logged before the request so a Ctrl-C mid-write still leaves the key (D11).
    ctx.tracer.emit(
        "tool.call",
        name="extend",
        args={
            "reservation_id": pending.reservation_id,
            "new_return_datetime": new_return_api,
            "idempotency_key": key,
        },
    )
    try:
        response = ctx.client.extend(
            pending.reservation_id,
            new_return_api,
            email=payment.email,
            cvv=payment.cvv,
            billing_zip=payment.billing_zip,
            idempotency_key=key,
        )
    except AvisAPIError as e:  # definite: refused on the first attempt, nothing charged
        ctx.write_state = None
        ctx.last_write = None
        ctx.tracer.emit("tool.result", name="extend", error=e.code)  # code only — messages may echo inputs
        return _definite_failure(ctx, e, retry_allowed)
    except OutcomeUnknown:
        return _unknown(ctx, "OutcomeUnknown")
    except BaseException as e:  # Ctrl-C or a crash after the request may have gone out
        _unknown(ctx, type(e).__name__)
        if isinstance(e, Exception):
            return CommitResult("transfer")
        raise

    ctx.write_state = "committed"
    ctx.pending = None
    ctx.tracer.emit("tool.result", name="extend", result=response)
    if not _matches(response, pending):
        ctx.last_write.update(_actual(response))
        ctx.tracer.emit(
            "gate.decision", gate="confirmation", allowed=False, reason_code="confirmation_mismatch"
        )
        transfer(ctx, ReasonCode.CONFIRMATION_MISMATCH, "")
        return CommitResult("transfer")
    ctx.last_write.update(_actual(response))
    ctx.quote = {**(ctx.quote or {}), "total": float(pending.total)}
    ctx.tracer.add_outcome("resolved_extension")
    return CommitResult("resolved", render_receipt(response, pending))


def _definite_failure(ctx: AgentContext, e: AvisAPIError, retry_allowed: bool) -> CommitResult:
    details_wrong = e.status == 403 or (e.status == 400 and e.code == "PAYMENT_VALIDATION_ERROR")
    if details_wrong and retry_allowed:
        return CommitResult("retry", DETAILS_RETRY)  # the pending stays approved for one more go
    ctx.pending = None
    if e.status == 403:
        reason = ReasonCode.VERIFICATION_FAILED
    elif e.status == 402:
        reason = ReasonCode.PAYMENT_DECLINED
    elif e.status == 409:
        reason = ReasonCode.NOT_ACTIVE
    else:
        reason = ReasonCode.INTERNAL_ERROR
    ctx.tracer.emit("gate.decision", gate="extend", allowed=False, reason_code=reason.value, stage="commit")
    return CommitResult("offer", offer(ctx, reason)["message"])


def _unknown(ctx: AgentContext, error: str) -> CommitResult:
    ctx.write_state = "unknown"
    ctx.pending = None
    ctx.tracer.emit("tool.result", name="extend", error=error)
    transfer(ctx, ReasonCode.OUTCOME_UNKNOWN, "")
    return CommitResult("transfer")


def _matches(response: dict, pending: PendingExtension) -> bool:
    """The response is what the customer approved: same reservation, instant, currency and total."""
    try:
        charges = response["charges"]
        return (
            response["reservation_id"] == pending.reservation_id
            and datetime.fromisoformat(response["extension_details"]["new_return_datetime"])
            == pending.new_return
            and charges["currency"] == pending.charges.get("currency")
            and money(charges["total_charged"]) == pending.total
        )
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return False


def _actual(response: dict) -> dict[str, Any]:
    charges = response.get("charges") if isinstance(response.get("charges"), dict) else {}
    return {
        "confirmation_number": response.get("confirmation_number"),
        "charged_total": charges.get("total_charged"),
    }


# --- what the customer sees (printed by the CLI, never by the model) -------------------------------


def render_card(p: PendingExtension) -> str:
    c = p.charges
    cur = c.get("currency", "USD")
    lines = [
        "──── Confirm your extension ────",
        f"Reservation    {p.reservation_id}",
        f"Return now     {format_for_customer(p.current_return)}",
        f"New return     {format_for_customer(p.new_return)}",
        f"Extra days     {c['extension_days']} × ${money(c['daily_rate'])} = ${money(c['subtotal'])}",
    ]
    for label, key in (("Late fee", "late_fee"), ("One-way fee", "one_way_fee")):
        if money(c.get(key) or 0):
            lines.append(f"{label:<15}${money(c[key])}")
    lines.append(f"Taxes & fees   ${money(c['taxes_and_fees'])}")
    lines.append(f"Total          ${p.total} {cur}")
    if p.card.get("last_four"):
        lines.append(f"Charged to     {p.card.get('type', 'Card')} ending {p.card['last_four']}")
    else:
        lines.append("Charged to     the card on file")
    return "\n".join(lines)


def render_receipt(response: dict, p: PendingExtension) -> str:
    charges = response["charges"]
    total = f"${money(charges['total_charged'])} {charges.get('currency', 'USD')}"
    when = to_local(response["extension_details"]["new_return_datetime"], p.new_return.tzinfo)
    return "\n".join(
        [
            "──── Extension confirmed ────",
            f"Confirmation   {response['confirmation_number']}",
            f"New return     {format_for_customer(when)}",
            f"Charged        {total} to the card on file",
        ]
    )


def receipt_note(receipt: str) -> str:
    """History entry after a commit: the receipt minus layout, so the model knows what happened. The receipt
    holds no card digits or auth code by construction."""
    return (
        "Extension confirmed — "
        + "; ".join(" ".join(line.split()) for line in receipt.splitlines()[1:])
        + "."
    )
