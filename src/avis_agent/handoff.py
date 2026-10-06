"""Handoffs (D10, AOP §7): a warm transfer, never a dead end.

Two steps, kept apart. *Lock*: a code-raised reason blocks its action for the rest of the chat
(enforced by the tool that raised it). *Transfer*: for OFFERED reasons code parks the reason and the
agent relays an offer; when the customer accepts, `request_transfer` files the packet under the
*code's* reason, whatever the model asked for (safety always wins). TRANSFER_NOW and model-raised
reasons transfer as soon as they're filed. The CLI ends the chat after the turn in which
`ctx.transfer` was set, and prints `customer_summary` — the model never writes that text.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import REPO_ROOT
from .privacy import scrub
from .reasons import CUSTOMER_COPY, MODEL_RAISED, OFFERED, TRANSFER_NOW, ReasonCode

if TYPE_CHECKING:
    from .tools import AgentContext

HANDOFF_LOG = REPO_ROOT / "logs" / "handoffs.jsonl"
NOTE_MAX = 300


@dataclass
class Handoff:
    """The packet a representative receives. Code adds no name, email, address or card (the rep looks those
    up); the model's `note` is scrubbed of contact and card details but may still mention a surname."""

    reason_code: str
    session_id: str
    reservation_id: str | None
    note: str  # what the customer wanted, model-written then scrubbed
    gates_hit: list[str] = field(default_factory=list)
    quote: dict | None = None
    log_path: str = ""
    model_reason: str | None = None  # what the model asked for, when code's reason overrode it
    # After an extend write: idempotency key, approved vs charged total, confirmation number (for on-call).
    extend: dict | None = None
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def _display_path(path: Path) -> str:
    """Repo-relative, so packets don't carry a home directory and still resolve on another machine."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def offer(ctx: AgentContext, reason: ReasonCode) -> dict[str, Any]:
    """Park an OFFERED reason and give the agent the offer to relay. Doesn't end anything."""
    if reason not in OFFERED:
        raise ValueError(f"{reason} is not an offered reason")
    ctx.pending_handoff = reason
    # D11: a gate stop the customer declines is still a protective handoff, not `abandoned`.
    if f"offered:{reason.value}" not in ctx.tracer.outcomes:
        ctx.tracer.add_outcome("offered", reason.value)
    if reason.value not in ctx.gates_hit:
        ctx.gates_hit.append(reason.value)
    return {"reason_code": reason.value, "message": CUSTOMER_COPY[reason]}


def request_transfer(ctx: AgentContext, model_reason: str, note: str) -> dict[str, Any]:
    """`handoff_to_human` body. The model may only name MODEL_RAISED reasons; a parked code reason
    takes precedence over what it names, except a safety incident."""
    try:
        asked = ReasonCode(model_reason)
    except ValueError:
        asked = None
    if asked is ReasonCode.SAFETY_INCIDENT or (asked in MODEL_RAISED and ctx.pending_handoff is None):
        reason = asked
    elif ctx.pending_handoff is not None:
        reason = ctx.pending_handoff
    else:
        allowed = ", ".join(sorted(c.value for c in MODEL_RAISED))
        return {"error": "invalid_reason", "message": f"reason_code must be one of: {allowed}"}
    transfer(ctx, reason, note, model_reason=model_reason if asked is not reason else None)
    return {
        "handed_off": True,
        "reason_code": reason.value,
        "message": "Transfer filed. Say one short line to close; the system shows the customer the summary.",
    }


def transfer(ctx: AgentContext, reason: ReasonCode, note: str, *, model_reason: str | None = None) -> Handoff:
    """File the packet and mark the chat for transfer. Idempotent: a second call keeps the first."""
    if ctx.transfer is not None:
        return ctx.transfer
    reservation = ctx.reservation or {}
    packet = Handoff(
        reason_code=reason.value,
        session_id=ctx.tracer.session_id,
        reservation_id=reservation.get("reservation_id"),
        note=scrub(note.strip()).text[:NOTE_MAX] if note else "",
        gates_hit=list(ctx.gates_hit),
        quote=ctx.quote,
        log_path=_display_path(ctx.tracer.path),
        model_reason=model_reason,
        extend=dict(ctx.last_write) if ctx.last_write else None,
    )
    ctx.transfer = packet
    if ctx.pending_handoff is not None:
        ctx.tracer.withdraw_offer(ctx.pending_handoff.value)
    ctx.pending_handoff = None
    _append(ctx.handoff_log, asdict(packet), ctx)
    ctx.tracer.add_outcome("handed_off", reason.value)
    return packet


def customer_summary(packet: Handoff, fallback_lead: str = "") -> str:
    """What the CLI prints at transfer, in place of the model's closing line — code-written, so the 911
    line and 'may have charged' wording can't be dropped or softened by the model. Reasons without
    transfer copy (an accepted offer), and language_unsupported (copy must be in the customer's language),
    lead with `fallback_lead`, the model's closing line."""
    reason = ReasonCode(packet.reason_code)
    lines = []
    if reason is ReasonCode.LANGUAGE_UNSUPPORTED and fallback_lead:
        lines.append(fallback_lead)  # the model already said it in the customer's language (AOP)
    elif reason in TRANSFER_NOW or reason in MODEL_RAISED:
        lines.append(CUSTOMER_COPY[reason])
    elif fallback_lead:
        lines.append(fallback_lead)
    passed = []
    if packet.reservation_id:
        passed.append(f"reservation {packet.reservation_id}")
    if packet.note:
        passed.append("what you asked for")  # the note itself is for the representative, not read back
    if packet.quote and packet.quote.get("total") is not None:
        passed.append(f"the quote (${packet.quote['total']:.2f})")
    confirmation = (packet.extend or {}).get("confirmation_number")
    if confirmation:
        lines.append(f"Your confirmation number is {confirmation}.")
    lines.append("— Connecting you with a representative —")
    if passed:
        lines.append("I've passed on " + "; ".join(passed) + ". You won't need to repeat any of it.")
    return "\n".join(lines)


def _append(path: Path, record: dict, ctx: AgentContext) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    except Exception as e:  # the transfer still happens; the session log keeps the outcome
        ctx.tracer.warn(f"handoff write failed ({type(e).__name__})")
