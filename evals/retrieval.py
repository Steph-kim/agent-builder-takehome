"""Retrieval eval: recall@k on the gold set, authority precedence, off-topic rejection, floor sweep.

    python -m evals.retrieval

Exits nonzero if authority precedence or off-topic rejection is below 100% (recall is reported, not gated).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import yaml
from openai import AsyncOpenAI

from avis_agent.config import load_settings
from avis_agent.kb import SIMILARITY_FLOOR, TOP_K, Hit, KnowledgeBase, openai_embedder

QUERIES = Path(__file__).with_name("queries.yaml")
SWEEP = (0.25, 0.30, 0.35, 0.40, 0.45)


def score(cases: list[dict], results: list[list[Hit]]) -> dict:
    found = total = prec_ok = prec_n = empty_ok = empty_n = 0
    failures = []
    for case, hits in zip(cases, results, strict=True):
        shown = [h.article.id for h in hits]
        if case.get("empty"):
            empty_n += 1
            empty_ok += not shown
            if shown:
                failures.append(f"off-topic returned {shown}: {case['q']}")
            continue
        gold = case.get("gold", [])
        hit = [g for g in gold if g in shown]
        found, total = found + len(hit), total + len(gold)
        if len(hit) < len(gold):
            failures.append(f"missed {sorted(set(gold) - set(hit))}: {case['q']}")
        for official, legacy in case.get("outrank", []):
            prec_n += 1
            ok = legacy not in shown or (official in shown and shown.index(official) < shown.index(legacy))
            prec_ok += ok
            if not ok:
                failures.append(f"{legacy} shown without {official} above it: {case['q']}")
    return {
        "recall": found / total if total else 1.0,
        "recall_n": f"{found}/{total}",
        "precedence": prec_ok / prec_n if prec_n else 1.0,
        "precedence_n": f"{prec_ok}/{prec_n}",
        "empty": empty_ok / empty_n if empty_n else 1.0,
        "empty_n": f"{empty_ok}/{empty_n}",
        "failures": failures,
    }


async def run() -> int:
    settings = load_settings()
    embed = openai_embedder(AsyncOpenAI(api_key=settings.openai_api_key), settings.embed_model)
    cases = yaml.safe_load(QUERIES.read_text())
    kb = await KnowledgeBase.build(embed)
    qvecs = [np.asarray(v) for v in await embed([c["q"] for c in cases])]

    def rank_all(floor: float) -> list[list[Hit]]:
        kb.floor = floor  # rank() is pure; re-run it per floor so supersession sees the floored set
        return [kb.rank(v) for v in qvecs]

    results = rank_all(SIMILARITY_FLOOR)
    print(f"model={settings.embed_model} top_k={TOP_K} floor={SIMILARITY_FLOOR} queries={len(cases)}")
    print("(+ = added because it supersedes a shown legacy article)\n")
    for case, hits in zip(cases, results, strict=True):
        tops = ", ".join(f"{h.article.id}({h.similarity:.2f}){'+' if h.superseding else ''}" for h in hits)
        print(f"- {case['q']}\n    {tops or '(nothing above floor)'}")

    print("\nfloor  recall@k     precedence  off-topic-empty")
    for f in SWEEP:
        s = score(cases, rank_all(f))
        mark = "  <- shipped" if abs(f - SIMILARITY_FLOOR) < 1e-9 else ""
        recall = f"{s['recall']:.0%} ({s['recall_n']})"
        print(f"{f:.2f}   {recall}  {s['precedence_n']:>9}  {s['empty_n']:>9}{mark}")

    shipped = score(cases, rank_all(SIMILARITY_FLOOR))
    for line in shipped["failures"]:
        print(f"FAIL {line}")
    ok = shipped["precedence"] == 1.0 and shipped["empty"] == 1.0
    print(
        f"\n{'PASS' if ok else 'FAIL'}: precedence {shipped['precedence_n']}, "
        f"off-topic {shipped['empty_n']}, "
        f"recall@{TOP_K} {shipped['recall']:.0%} ({shipped['recall_n']}) at floor {SIMILARITY_FLOOR}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
