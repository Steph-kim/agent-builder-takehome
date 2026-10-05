"""Per-session JSONL trace — the source of truth for debugging and for the billing ledger (D11).

Every event type has a field allowlist, and tool payloads are reduced to an allowlist of
keys, so a name, address, plate, email or card detail can't reach `logs/` by accident:
new fields are dropped until someone adds them here on purpose. Customer text is logged
only after `privacy.scrub`.

A failed write never ends the conversation: it warns once on stderr and keeps serving.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from .config import REPO_ROOT

LOG_DIR = REPO_ROOT / "logs" / "sessions"

EVENT_FIELDS: dict[str, frozenset[str]] = {
    "session.start": frozenset({"git_sha", "model", "prompt_hash", "kb_hash"}),
    "customer.msg": frozenset({"text", "redacted"}),
    "agent.msg": frozenset({"text"}),
    "llm.turn": frozenset({"latency_ms", "input_tokens", "output_tokens"}),
    "tool.call": frozenset({"name", "args"}),
    "tool.result": frozenset({"name", "result", "error"}),
    "api.request": frozenset(
        {"method", "path", "status", "error_code", "latency_ms", "attempt", "idempotency_key"}
    ),
    "gate.decision": frozenset({"gate", "allowed", "reason_code"}),
    "approval": frozenset({"shown_total", "decision"}),
    "outcome": frozenset({"outcomes"}),
}

# Keys allowed anywhere inside tool args/results. Deliberately absent: customer_name,
# last_name, email, address, license_plate, make_model, color, card_on_file, cvv, billing_zip.
PAYLOAD_KEYS = frozenset(
    {
        # reservation
        "reservation_id", "verified", "locked", "membership_status", "status", "vehicle", "type",
        "pickup_location", "return_location", "code", "name",
        "dates", "pickup_datetime", "current_return_datetime", "original_return_datetime",
        "pricing", "daily_rate", "currency",
        # availability / quote / extend
        "location", "vehicle_type", "vehicle_description", "start_date", "end_date", "available",
        "new_return_datetime", "quote", "charges", "extension_days", "subtotal", "late_fee",
        "one_way_fee", "taxes_and_fees", "total_charged", "success", "confirmation_number",
        "extension_details", "late_return", "change_type", "idempotency_key",
        # kb
        "query", "results", "id", "title", "authority", "last_updated", "note",
        # handoff / errors
        "reason", "reason_code", "handed_off", "error", "message",
    }
)  # fmt: skip


def reduce_payload(value: Any) -> Any:
    """Keep only allowlisted keys, recursively. Fail-closed: unknown keys vanish."""
    if isinstance(value, dict):
        return {k: reduce_payload(v) for k, v in value.items() if k in PAYLOAD_KEYS}
    if isinstance(value, list | tuple):
        return [reduce_payload(v) for v in value]
    return value


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class Tracer:
    def __init__(
        self,
        session_id: str | None = None,
        log_dir: Path = LOG_DIR,
        *,
        stderr: TextIO = sys.stderr,
    ):
        self.session_id = session_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6]
        self.path = log_dir / f"{self.session_id}.jsonl"
        self.outcomes: list[str] = []
        self._stderr = stderr
        self._warned = False
        self._closed = False

    # --- events ----------------------------------------------------------------------------

    def start(self, *, model: str, prompt_hash: str, kb_hash: str) -> None:
        self.emit("session.start", git_sha=git_sha(), model=model, prompt_hash=prompt_hash, kb_hash=kb_hash)

    def emit(self, event: str, **fields: Any) -> None:
        allowed = EVENT_FIELDS.get(event, frozenset())
        record: dict[str, Any] = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), "event": event}
        for k, v in fields.items():
            if k not in allowed:
                continue
            record[k] = reduce_payload(v) if k in ("args", "result") else v
        self._write(record)

    def api_request(self, event: dict[str, Any]) -> None:
        """`AvisClient` observer: pass `observer=tracer.api_request`."""
        self.emit("api.request", **event)

    # --- outcomes (one per customer request; the billing ledger) ---------------------------

    def add_outcome(self, kind: str, reason: str | None = None) -> None:
        """kind: resolved_extension | info_only | handed_off (with reason) | error."""
        self.outcomes.append(f"handed_off:{reason}" if kind == "handed_off" else kind)

    def close(self, *, interrupted: bool = False) -> None:
        """Write the outcome line. Call from a `finally` so Ctrl-C still leaves a ledger entry."""
        if self._closed:
            return
        self._closed = True
        if interrupted or not self.outcomes:
            self.outcomes.append("interrupted" if interrupted else "abandoned")
        self.emit("outcome", outcomes=self.outcomes)

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Ctrl-C during an await reaches the coroutine as CancelledError (asyncio.run re-raises it after).
        interrupted = exc_type is not None and issubclass(
            exc_type, KeyboardInterrupt | asyncio.CancelledError
        )
        if exc_type is not None and not interrupted:
            self.add_outcome("error")
        self.close(interrupted=interrupted)

    # --- plumbing --------------------------------------------------------------------------

    def _write(self, record: dict[str, Any]) -> None:
        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:  # a broken log must never break the chat
            self.warn(f"log write failed ({type(e).__name__}); continuing without it")

    def warn(self, msg: str) -> None:
        """One stderr warning per session; also used for side files (handoffs.jsonl) that fail to write."""
        if self._warned:
            return
        self._warned = True
        try:
            print(f"[trace] {msg}", file=self._stderr)
        except Exception:
            pass
