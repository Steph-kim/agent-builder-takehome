"""KB ranking, offline. Articles sit on orthogonal axes, so a query's weights *are* its cosine similarities
(after normalisation) and each test controls exactly which article is closest."""

import asyncio

import numpy as np
import pytest

from avis_agent.kb import LEGACY_NOTE, SUPERSEDED_BY, Article, KnowledgeBase, load_articles


def art(id, authority, updated="2026-01-01"):
    return Article(id=id, title=id, body=f"body {id}", last_updated=updated, authority=authority)


async def no_embed(texts):
    raise AssertionError("rank() must not embed")


def kb_of(articles, vectors=None, owners=None, **attrs):
    """One vector per article unless given; no supersession unless given; then override top_k/floor."""
    vectors = np.eye(len(articles)) if vectors is None else vectors
    kb = KnowledgeBase(articles, vectors, range(len(articles)) if owners is None else owners, no_embed)
    kb.superseded_by = {}
    for name, value in attrs.items():
        assert hasattr(kb, name), name
        setattr(kb, name, value)
    return kb


def ids(hits):
    return [h.article.id for h in hits]


def test_official_outranks_a_more_similar_legacy_article():
    # The grace-period trap: legacy "2 hours" is the closer match, official "30 minutes" must still lead.
    kb = kb_of([art("legacy_grace", "legacy"), art("official_grace", "official-policy")], floor=0.1)
    assert ids(kb.rank(np.array([0.9, 0.4]))) == ["official_grace", "legacy_grace"]


def test_similarity_cut_happens_before_authority_order():
    # Four relevant help-center/legacy articles and one barely related official one: the official one is
    # 5th by similarity and must not displace a relevant article from the top 4.
    arts = [art(f"hc{i}", "help-center") for i in range(3)] + [
        art("leg", "legacy"),
        art("off_far", "official-policy"),
    ]
    kb = kb_of(arts, top_k=4, floor=0.0)
    hits = kb.rank(np.array([0.9, 0.8, 0.7, 0.6, 0.35]))
    assert "off_far" not in ids(hits)
    assert ids(hits)[-1] == "leg"


def test_recency_breaks_ties_within_an_authority_level():
    arts = [art("old", "official-policy", "2025-01-01"), art("new", "official-policy", "2026-05-01")]
    assert ids(kb_of(arts, floor=0.1).rank(np.array([0.9, 0.5]))) == ["new", "old"]


def test_authority_dominates_recency():
    arts = [art("new_hc", "help-center", "2026-09-01"), art("old_off", "official-policy", "2024-01-01")]
    assert ids(kb_of(arts, floor=0.1).rank(np.array([0.9, 0.5]))) == ["old_off", "new_hc"]


def test_floor_drops_weak_matches_and_can_return_nothing():
    kb = kb_of([art("a", "official-policy"), art("b", "help-center"), art("c", "legacy")], floor=0.3)
    # Normalised [0.9, 0.2, 0.1] → sims ≈ 0.97, 0.22, 0.11: only "a" clears the floor.
    assert ids(kb.rank(np.array([0.9, 0.2, 0.1]))) == ["a"]
    # Off-topic: nothing clears it → empty, so the agent says "not in the KB" instead of guessing.
    assert kb.rank(np.array([0.1, 0.1, -1.0])) == []


def test_only_legacy_hits_carry_the_outdated_note():
    kb = kb_of([art("leg", "legacy"), art("off", "official-policy")], floor=0.1)
    by_id = {h.article.id: h.to_tool_dict() for h in kb.rank(np.array([0.7, 0.7]))}
    assert by_id["leg"]["note"] == LEGACY_NOTE
    assert "note" not in by_id["off"]
    assert set(by_id["off"]) == {"id", "title", "authority", "last_updated", "body"}


def test_build_embeds_the_whole_kb_in_one_call_and_search_embeds_the_query():
    calls = []
    arts = [art("a", "official-policy"), art("b", "legacy")]

    async def fake_embed(texts):
        calls.append(list(texts))
        if len(texts) == 2:
            return np.eye(2)
        return np.array([[0.2, 0.9]])

    async def go():
        kb = await KnowledgeBase.build(fake_embed, arts)
        kb.superseded_by, kb.floor = {}, 0.1
        return await kb.search("anything")

    hits = asyncio.run(go())
    assert calls == [["a\nbody a", "b\nbody b"], ["anything"]]
    assert ids(hits) == ["a", "b"]


def test_article_scores_by_its_best_sentence_not_its_average():
    # The buried-grace-period bug: "ext" has one strongly matching sentence among off-topic ones; "leg" is
    # moderately on-topic throughout. Best-sentence scoring must rank ext's 0.95 above leg's 0.6.
    arts = [art("ext", "help-center"), art("leg", "help-center")]
    vectors = np.array([[0.95, 0.3], [0.0, 1.0], [0.0, 1.0], [0.6, 0.8]])
    kb = kb_of(arts, vectors, owners=[0, 0, 0, 1], top_k=1, floor=0.1)
    hits = kb.rank(np.array([1.0, 0.0]))
    assert ids(hits) == ["ext"]
    assert hits[0].similarity == pytest.approx(0.95 / np.hypot(0.95, 0.3))


def test_a_shown_legacy_article_brings_its_official_replacement_above_it():
    # Official grace article is 3rd by similarity, outside top_k=1 — it must still be shown, and lead.
    arts = [art("leg", "legacy"), art("other", "help-center"), art("off", "official-policy")]
    kb = kb_of(arts, superseded_by={"leg": ["off"]}, top_k=1, floor=0.5)
    hits = kb.rank(np.array([0.9, 0.3, 0.1]))
    assert ids(hits) == ["off", "leg"]
    assert [h.superseding for h in hits] == [True, False]


def test_replacement_is_not_added_when_the_legacy_article_is_not_shown():
    arts = [art("leg", "legacy"), art("other", "help-center"), art("off", "official-policy")]
    kb = kb_of(arts, superseded_by={"leg": ["off"]}, top_k=1, floor=0.5)
    assert ids(kb.rank(np.array([0.1, 0.9, 0.0]))) == ["other"]


def test_shipped_supersession_map_points_legacy_to_official():
    by_id = {a.id: a for a in load_articles()}
    assert set(SUPERSEDED_BY) == {"kb_fee_01"}
    for legacy, officials in SUPERSEDED_BY.items():
        assert by_id[legacy].authority == "legacy"
        assert all(by_id[o].authority == "official-policy" for o in officials)


def test_shipped_kb_loads_with_known_authorities():
    articles = load_articles()
    assert len(articles) == 30
    assert {a.authority for a in articles} == {"official-policy", "help-center", "legacy"}
    assert len({a.id for a in articles}) == len(articles)
