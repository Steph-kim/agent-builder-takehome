"""Settings loaded once at startup from the environment / `.env`.

Policy thresholds live here (not in the prompt) so the model never sees them and
can't coach a customer to stay under a cap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED = ("OPENAI_API_KEY", "AVIS_API_KEY", "AVIS_API_URL")


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Thresholds:
    # Assumptions, stated in the README. Overdue: late-but-recent customers can still extend.
    overdue_hours: float = 24.0
    max_total_usd: float = 500.0
    max_added_days: int = 14
    max_failed_lookups: int = 5


@dataclass(frozen=True)
class Settings:
    avis_api_url: str
    avis_api_key: str
    openai_api_key: str
    model: str = "gpt-5-mini"
    sim_model: str = "gpt-5-mini"
    embed_model: str = "text-embedding-3-small"
    reasoning_effort: str | None = None
    # Empty = no market restriction. Demo default is the US pilot (see README assumptions).
    pilot_locations: frozenset[str] = frozenset()
    thresholds: Thresholds = field(default_factory=Thresholds)


def load_settings(env_file: Path | None = None) -> Settings:
    """Read settings; raise ConfigError naming every missing variable (not a bare KeyError)."""
    load_dotenv(env_file or REPO_ROOT / ".env")
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        raise ConfigError(
            f"Missing {', '.join(missing)}. Copy env.example to .env and fill them in (see README)."
        )
    pilot = os.environ.get("AVIS_PILOT_LOCATIONS", "")
    return Settings(
        avis_api_url=os.environ["AVIS_API_URL"].rstrip("/"),
        avis_api_key=os.environ["AVIS_API_KEY"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ.get("AVIS_MODEL") or Settings.model,
        sim_model=os.environ.get("AVIS_SIM_MODEL") or Settings.sim_model,
        embed_model=os.environ.get("AVIS_EMBED_MODEL") or Settings.embed_model,
        reasoning_effort=os.environ.get("AVIS_REASONING_EFFORT") or None,
        pilot_locations=frozenset(c.strip().upper() for c in pilot.split(",") if c.strip()),
    )
