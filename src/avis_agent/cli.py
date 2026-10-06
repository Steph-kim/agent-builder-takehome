"""Terminal chat. One process = one session (no reset: it would bypass the lookup cap and one-write guard).

The whole chat runs inside one `asyncio.run` — the SDK's HTTP client binds to the first event loop.
Customer text is scrubbed before the model or the log sees it. A transfer ends the chat: code, not the
model, prints what happens next.

The extension card, the y/n, the payment prompts, the charge and the receipt all happen here, between model
turns (AOP §5–6). The model only *checks* an extension; it has no tool that can charge.
"""

from __future__ import annotations

import asyncio
import getpass
import hashlib
import re
import string
import sys
from collections.abc import Callable

import openai
from openai import AsyncOpenAI

from .agent import INSTRUCTIONS, build_agent, respond
from .client import AvisClient
from .config import ConfigError, Settings, load_settings
from .extend import NOTHING_CHANGED, Payment, commit, receipt_note, render_card
from .handoff import customer_summary
from .kb import KnowledgeBase, kb_hash, openai_embedder
from .privacy import scrub
from .tools import AgentContext
from .trace import Tracer

GREETING = "How can I help?"
EXIT_WORDS = frozenset({"exit", "quit"})
YES = frozenset({"y", "yes"})
NO = frozenset({"", "n", "no"})
# Model-only, after a card closes without a charge. The model's history still holds check_extension's
# "ready", so without this it points the customer to a card that's gone (seen in sims, 2026-10-06).
CARD_CLOSED = (
    "The last card was closed without a charge, so never point the customer to it. If they still want that "
    "change, call check_extension again; when it returns ready, the system shows them a new card."
)
# Model-only, after a commit. The receipt sits in history as an assistant line, which the prompt says not to
# trust ("the system confirms it"), so without this the model asked a charged customer to approve the card.
CHARGED = (
    "The system charged the customer and confirmed this extension; the receipt above is from the API. The "
    "change is made: no card is showing and nothing is left to approve, so never ask them to approve or "
    "confirm anything for it."
)
# The prompt asks for [kb_…] citations so every policy claim is traceable. They mean nothing to a customer, so
# they're logged (agent.msg "cites") and stripped from what's printed. Handles "[kb_a_01][kb_b_02]" and
# "[kb_a_01, kb_b_02]".
CITATION = re.compile(r"\s*\[(kb_[a-z]+_\d+(?:\s*,\s*kb_[a-z]+_\d+)*)\]")
PAYMENT_INTRO = (
    "To confirm, enter the email on the booking, then the card's security code and billing ZIP. "
    "They go straight to Avis — the assistant never sees them."
)


async def chat(
    settings: Settings,
    kb: KnowledgeBase,
    *,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    read_secret: Callable[[str], str] = getpass.getpass,
    tracer: Tracer | None = None,
    client: AvisClient | None = None,
    **ctx_overrides,
) -> None:
    with tracer or Tracer() as t:
        t.start(model=settings.model, prompt_hash=_short_hash(INSTRUCTIONS), kb_hash=kb_hash())
        client = client or AvisClient(settings.avis_api_url, settings.avis_api_key, observer=t.api_request)
        ctx = AgentContext(
            client=client,
            tracer=t,
            thresholds=settings.thresholds,
            kb=kb,
            pilot_locations=settings.pilot_locations,
            notify=lambda msg: write(f"({msg})"),
            **ctx_overrides,
        )
        agent = build_agent(settings)
        history = []
        queued: str | None = None  # text typed at the card prompt, run as the next customer turn
        _say(write, t, GREETING)
        while True:
            if queued is None:
                try:
                    raw = read("You: ").strip()
                except EOFError:
                    break
            else:
                raw, queued = queued, None
            if raw.lower() in EXIT_WORDS:
                break
            if not raw:
                continue
            clean = scrub(raw)
            t.emit("customer.msg", text=clean.text, redacted=clean.kinds)
            history.append({"role": "user", "content": clean.text})
            reply, history = await respond(agent, history, ctx)
            if ctx.transfer is not None:
                _say(write, t, customer_summary(ctx.transfer, fallback_lead=reply))
                break
            _say(write, t, reply)
            queued = _confirm_pending(ctx, history, read, write, read_secret)
            if ctx.transfer is not None:  # the write was unconfirmed or mismatched
                _say(write, t, customer_summary(ctx.transfer))
                break


def _confirm_pending(
    ctx: AgentContext,
    history: list,
    read: Callable[[str], str],
    write: Callable[[str], None],
    read_secret: Callable[[str], str],
) -> str | None:
    """Show the card for an unshown pending, take y/n, collect payment off-model and commit. Appends what
    happened to `history` (no card digits, no payment data). Returns non-y/n text to run as the next turn."""
    t = ctx.tracer
    while (pending := ctx.pending) is not None and not pending.shown and ctx.transfer is None:
        pending.shown = True
        history[:] = [h for h in history if h.get("content") != CARD_CLOSED]  # a card is showing again
        write(render_card(pending))  # not _say: the card's last four stays out of agent.msg
        try:
            raw = read("Approve this charge? (y/n): ").strip()
        except EOFError:
            raw = ""
        answer = raw.lower().strip(string.punctuation + " ")
        decision = "y" if answer in YES else "n" if answer in NO else "other"
        t.emit("approval", shown_total=float(pending.total), decision=decision)
        if decision != "y":
            ctx.pending = None
            _closed(history, write, t)
            return raw if decision == "other" else None

        pending.approved = True
        retry_allowed = True
        while True:
            payment = _collect_payment(read, read_secret, write)
            if payment is None:
                ctx.pending = None
                _closed(history, write, t)
                return None
            result = commit(ctx, pending, payment, retry_allowed=retry_allowed)
            del payment
            if result.kind != "retry":
                break
            write(f"Agent: {result.message}")
            retry_allowed = False

        if result.kind == "recard":
            _note(history, write, t, result.message)  # loop shows the new card
        elif result.kind == "resolved":
            write(result.message)  # the receipt, from the API response
            note = receipt_note(result.message)
            t.emit("agent.msg", text=note)
            history.append({"role": "assistant", "content": note})
            history.append({"role": "system", "content": CHARGED})
        elif result.kind == "offer":
            _note(history, write, t, result.message)
    return None


def _collect_payment(
    read: Callable[[str], str], read_secret: Callable[[str], str], write: Callable[[str], None]
) -> Payment | None:
    """Email + CVV + ZIP straight from the terminal to the commit call — never the model, history or log."""
    write(f"Agent: {PAYMENT_INTRO}")
    try:
        email = read("Email on the booking: ").strip()
        cvv = read_secret("Card security code (hidden): ").strip()
        billing_zip = read_secret("Billing ZIP (hidden): ").strip()
    except EOFError:
        return None
    if not (email and cvv and billing_zip):
        return None
    return Payment(email=email, cvv=cvv, billing_zip=billing_zip)


def _note(history: list, write: Callable[[str], None], t: Tracer, text: str) -> None:
    """A code-written agent line: shown, logged, and added to history so the model knows."""
    _say(write, t, text)
    history.append({"role": "assistant", "content": text})


def _closed(history: list, write: Callable[[str], None], t: Tracer) -> None:
    _note(history, write, t, NOTHING_CHANGED)
    history.append({"role": "system", "content": CARD_CLOSED})


def _say(write: Callable[[str], None], t: Tracer, text: str) -> None:
    cites = [c.strip() for m in CITATION.finditer(text) for c in m.group(1).split(",")]
    shown = CITATION.sub("", text)
    if cites:
        t.emit("agent.msg", text=scrub(shown).text, cites=cites)
    else:
        t.emit("agent.msg", text=scrub(shown).text)
    write(f"Agent: {shown}")


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


async def _run(settings: Settings) -> None:
    embed = openai_embedder(AsyncOpenAI(api_key=settings.openai_api_key), settings.embed_model)
    kb = await KnowledgeBase.build(embed)
    await chat(settings, kb)


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    print("Avis support (type 'exit' or Ctrl-D to leave)\n")
    try:
        asyncio.run(_run(settings))
    except openai.APIError as e:  # KB embedding at startup; mid-chat failures are handled in respond()
        print(
            f"Can't reach the language model ({type(e).__name__}). Please try again shortly.", file=sys.stderr
        )
        return 1
    except KeyboardInterrupt:
        print("\n[session ended]")
        return 130
    return 0
