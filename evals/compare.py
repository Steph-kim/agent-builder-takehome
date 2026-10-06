"""Model comparison from finished sim runs: quality from the result tables, speed and cost from the logs.

    python -m evals.compare 20261006T122129 20261006T130112 ...

Each stamp is a full `python -m evals.sim -k N` run (one agent model per run, set with AVIS_MODEL).
Latency is one `llm.turn` per customer message: the whole agent reply, Avis API calls included. Cost uses
the token counts logged per turn at the prices below; cached input isn't logged, so it's priced as
uncached (an upper bound).
"""

from __future__ import annotations

import json
import re
import statistics
import sys

from evals.sim import RESULTS_DIR, SIM_LOGS

# USD per 1M tokens (input, output), standard tier. Snapshot — re-check before relying on it.
PRICES_AS_OF = "2026-10-06, developers.openai.com/api/docs/pricing"
PRICES = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.5": (5.00, 30.00),
}
SUMMARY = re.compile(
    r"pass\^(\d+): (\d+)/(\d+) scenarios.*?Runs passed: (\d+)/(\d+)\. Safety: (\d+)/(\d+) sessions clean"
)


def summarise(stamp: str) -> dict:
    report = next(RESULTS_DIR.glob(f"sim-{stamp}-k*.md"))
    k, all_pass, n_ids, runs_ok, runs, safe, _ = map(int, SUMMARY.search(report.read_text()).groups())
    model, latencies, tokens_in, tokens_out = None, [], 0, 0
    sessions = sorted((SIM_LOGS / stamp).glob("*-r*.jsonl"))
    for path in sessions:
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "session.start":
                model = event["model"]
            elif event["event"] == "llm.turn":
                latencies.append(event["latency_ms"])
                tokens_in += event["input_tokens"]
                tokens_out += event["output_tokens"]
    price_in, price_out = PRICES[model]
    cost = (tokens_in * price_in + tokens_out * price_out) / 1e6
    p95 = statistics.quantiles(latencies, n=20)[18]
    return {
        "model": model,
        "pass_k": f"{all_pass}/{n_ids} (k={k})",
        "runs": f"{runs_ok}/{runs}",
        "safety": f"{safe}/{runs}",
        "p50": f"{statistics.median(latencies) / 1000:.1f}s",
        "p95": f"{p95 / 1000:.1f}s",
        "per_conv": f"${cost / len(sessions):.4f}",
        "per_1k": f"${cost / len(latencies) * 1000:.2f}",
    }


def main(stamps: list[str]) -> None:
    print(
        "| Agent model | pass^k | Runs passed | Safety clean | p50 reply | p95 reply "
        "| $ / conversation | $ / 1k turns |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for s in map(summarise, stamps):
        print(
            f"| `{s['model']}` | {s['pass_k']} | {s['runs']} | {s['safety']} | {s['p50']} | {s['p95']} "
            f"| {s['per_conv']} | {s['per_1k']} |"
        )
    print(f"\nPrices: {PRICES_AS_OF}. Cached input priced as uncached (upper bound).")


if __name__ == "__main__":
    main(sys.argv[1:])
