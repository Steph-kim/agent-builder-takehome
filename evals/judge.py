"""CX judge for sim transcripts: tone, clarity, concision and next step, scored 1-5. Reported, never gated.

Correctness and safety are graded from the trace (`sim.grade`); a judge is noisy on those. It earns its
place on the subjective half — would a customer find this clear and pleasant? Uncalibrated against human
labels, so it's a triage signal: read the lowest-scoring transcripts first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

DIMENSIONS = ("clarity", "concision", "tone", "next_step")
DEFAULT_JUDGE_MODEL = "gpt-5.4"  # not the agent's model, so it isn't grading its own habits

RUBRIC = """You review a chat between a rental-car customer and Avis's support assistant. Score the ASSISTANT
only, 1 (poor) to 5 (excellent), on:
- clarity: plain words, dates/prices stated unambiguously, no jargon or internal codes.
- concision: as short as the moment allows; no repeated boilerplate or padding.
- tone: warm and professional, not robotic or over-apologetic; respects the customer's choices.
- next_step: at every point the customer knows what happens next and what (if anything) to do.
Do NOT judge policy correctness or whether the right action was taken — that is graded elsewhere.
Lines in [brackets] are the system (cards, receipts, payment prompts), not the assistant; don't score them.
Return JSON: {"clarity": n, "concision": n, "tone": n, "next_step": n, "weakest": "<one sentence>"}"""

SCHEMA = {
    "type": "object",
    "properties": {
        **{d: {"type": "integer", "minimum": 1, "maximum": 5} for d in DIMENSIONS},
        "weakest": {"type": "string"},
    },
    "required": [*DIMENSIONS, "weakest"],
    "additionalProperties": False,
}


@dataclass
class CX:
    scores: dict[str, int]
    weakest: str

    @property
    def mean(self) -> float:
        return sum(self.scores.values()) / len(self.scores)


def transcript(events: list[dict]) -> str:
    """Only what the customer saw from each side, scrubbed as logged. Tool payloads stay out."""
    lines = []
    for e in events:
        if e.get("event") == "customer.msg":
            lines.append(f"Customer: {e.get('text', '')}")
        elif e.get("event") == "agent.msg":
            lines.append(f"Assistant: {e.get('text', '')}")
        elif e.get("event") == "approval":
            lines.append(
                f"[card shown, total ${e.get('shown_total')}; customer answered {e.get('decision')}]"
            )
    return "\n".join(lines)


def judge(client: OpenAI, model: str, events: list[dict]) -> CX | None:
    text = transcript(events)
    if not text:
        return None
    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": RUBRIC}, {"role": "user", "content": text}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "cx", "schema": SCHEMA, "strict": True},
        },
    )
    data = json.loads(reply.choices[0].message.content or "{}")
    return CX({d: int(data[d]) for d in DIMENSIONS}, str(data.get("weakest", "")))
