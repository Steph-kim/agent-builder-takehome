"""Terminal chat. One process = one session (no reset: it would bypass the lookup cap and one-write guard).

The whole chat runs inside one `asyncio.run` — the SDK's HTTP client binds to the first event loop.
Customer text is scrubbed before the model or the log sees it. A transfer ends the chat: code, not the
model, prints what happens next.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from collections.abc import Callable

import openai
from openai import AsyncOpenAI

from .agent import INSTRUCTIONS, build_agent, respond
from .client import AvisClient
from .config import ConfigError, Settings, load_settings
from .handoff import customer_summary
from .kb import KnowledgeBase, kb_hash, openai_embedder
from .privacy import scrub
from .tools import AgentContext
from .trace import Tracer

GREETING = "How can I help?"
EXIT_WORDS = frozenset({"exit", "quit"})


async def chat(
    settings: Settings,
    kb: KnowledgeBase,
    *,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    tracer: Tracer | None = None,
    **ctx_overrides,
) -> None:
    with tracer or Tracer() as t:
        t.start(model=settings.model, prompt_hash=_short_hash(INSTRUCTIONS), kb_hash=kb_hash())
        client = AvisClient(settings.avis_api_url, settings.avis_api_key, observer=t.api_request)
        ctx = AgentContext(client=client, tracer=t, thresholds=settings.thresholds, kb=kb, **ctx_overrides)
        agent = build_agent(settings)
        history = []
        _say(write, t, GREETING)
        while True:
            try:
                raw = read("You: ").strip()
            except EOFError:
                break
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


def _say(write: Callable[[str], None], t: Tracer, text: str) -> None:
    t.emit("agent.msg", text=scrub(text).text)
    write(f"Agent: {text}")


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
