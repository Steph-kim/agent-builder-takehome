"""Knowledge-base retrieval: embed every article sentence once at startup, score each article by its best
sentence, cut to top-k, then order by authority.

Order of operations matters (plan D9):
1. Score each article by its **best-matching sentence** (title-prefixed). Whole-article embeddings missed the
   official 30-minute grace period: it is one sentence inside "Extending Your Rental", while the legacy
   "2 hours" article is entirely about grace periods (evals/retrieval.py: recall 14/17 → 17/17).
2. Cut to the top-k by that score, dropping anything under the floor — relevance decides *what* is shown.
3. If a legacy article made the cut, add the official articles that supersede it (`SUPERSEDED_BY`), so an
   outdated answer is never shown without its correction. Retrieval alone left one trap open.
4. Order the set by authority (official > help-center > legacy), then recency — authority decides *which
   answer wins* when two shown articles conflict. Sorting by authority before the cut would bury a relevant
   help-center article under a barely related official one.
Legacy articles stay (kb_fee_05 is the only fuel source) but carry a "defer to official" note. The whole
article is returned, never a lone sentence, so the model sees the context.

RAG explains, the API decides: nothing here is ever the source of a number about a specific rental.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avis_agent.config import REPO_ROOT

ARTICLES_PATH = REPO_ROOT / "data" / "knowledge-base" / "articles.json"

AUTHORITY_RANK = {"official-policy": 0, "help-center": 1, "legacy": 2}
LEGACY_NOTE = "Outdated article — where it conflicts with an official-policy article, the official one wins."

# Legacy article → the official articles that replace it. Curated: part of the KB conflict check that runs
# before a KB change ships (README, post-deploy). kb_fee_05 (fuel) has no official replacement.
SUPERSEDED_BY: dict[str, tuple[str, ...]] = {
    "kb_fee_01": ("kb_ext_01", "kb_fee_02"),  # grace 2h → 30 min; late fee → $29 flat
}

TOP_K = 4
# Cosine floor for text-embedding-3-small on sentence passages, from the evals/retrieval.py sweep: weakest
# gold match 0.48, strongest uncovered near-domain query 0.42. Thin margin — erring high is the safe side
# (an empty result means "not in the KB" + kb_gap handoff, not a stretched answer).
SIMILARITY_FLOOR = 0.45

Embedder = Callable[[Sequence[str]], Awaitable[np.ndarray]]

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    body: str
    last_updated: str  # ISO date; lexical order == chronological
    authority: str

    @property
    def passages(self) -> list[str]:
        return [f"{self.title}\n{s}" for s in _SENTENCE_END.split(self.body.strip()) if s]


@dataclass(frozen=True)
class Hit:
    article: Article
    similarity: float  # best sentence's cosine similarity to the query
    superseding: bool = False  # added because it replaces a shown legacy article, not on its own score

    def to_tool_dict(self) -> dict:
        a = self.article
        out = {
            "id": a.id,
            "title": a.title,
            "authority": a.authority,
            "last_updated": a.last_updated,
            "body": a.body,
        }
        if a.authority == "legacy":
            out["note"] = LEGACY_NOTE
        return out


def load_articles(path: Path = ARTICLES_PATH) -> list[Article]:
    raw = json.loads(path.read_text())
    articles = [
        Article(
            id=r["id"],
            title=r["title"],
            body=r["body"],
            last_updated=r["last_updated"],
            authority=r["authority"],
        )
        for r in raw
    ]
    return articles


def kb_hash(path: Path = ARTICLES_PATH) -> str:
    """Short content hash for the session.start log line — ties every answer to a KB version."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class KnowledgeBase:
    def __init__(
        self, articles: Sequence[Article], vectors: np.ndarray, owners: Sequence[int], embed: Embedder
    ):
        """`vectors[j]` embeds a passage (sentence) of `articles[owners[j]]`."""
        self.articles = list(articles)
        self._owners = np.asarray(owners)
        self._index = {a.id: i for i, a in enumerate(self.articles)}
        self._vectors = _normalize(vectors)
        self._embed = embed
        self.superseded_by: Mapping[str, Sequence[str]] = SUPERSEDED_BY
        self.top_k = TOP_K
        self.floor = SIMILARITY_FLOOR

    @classmethod
    async def build(cls, embed: Embedder, articles: Sequence[Article] | None = None) -> KnowledgeBase:
        """One embedding call for every sentence in the KB (~160 strings — no cache, no vector store)."""
        articles = list(articles) if articles is not None else load_articles()
        texts, owners = [], []
        for i, a in enumerate(articles):
            for p in a.passages:
                texts.append(p)
                owners.append(i)
        vectors = await embed(texts)
        return cls(articles, vectors, owners, embed)

    def rank(self, query_vector: np.ndarray) -> list[Hit]:
        """Best-sentence score → top-k cut → add supersessors → authority/recency order. Pure."""
        passage_sims = self._vectors @ _normalize(query_vector)
        sims = np.full(len(self.articles), -np.inf)
        np.maximum.at(sims, self._owners, passage_sims)
        top = [i for i in np.argsort(-sims, kind="stable")[: self.top_k] if sims[i] >= self.floor]
        hits = [Hit(self.articles[i], float(sims[i])) for i in top]
        shown = {h.article.id for h in hits}
        for h in list(hits):
            for official in self.superseded_by.get(h.article.id, ()):
                if official not in shown:
                    shown.add(official)
                    i = self._index[official]
                    hits.append(Hit(self.articles[i], float(sims[i]), superseding=True))
        # Two stable sorts: recency (newest first), then authority — authority dominates, recency breaks ties.
        hits.sort(key=lambda h: h.article.last_updated, reverse=True)
        hits.sort(key=lambda h: AUTHORITY_RANK[h.article.authority])
        return hits

    async def search(self, query: str) -> list[Hit]:
        vector = await self._embed([query])
        return self.rank(np.asarray(vector)[0])


def openai_embedder(client, model: str) -> Embedder:
    """Adapter over `openai.AsyncOpenAI` — the client keeps its own retries (plan D8)."""

    async def embed(texts: Sequence[str]) -> np.ndarray:
        resp = await client.embeddings.create(model=model, input=list(texts))
        return np.array([d.embedding for d in sorted(resp.data, key=lambda d: d.index)])

    return embed
