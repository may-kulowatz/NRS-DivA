"""Tests for normalized (EBNeRD-style) content diversity and the per-impression
recommendation readers that feed it.

The pools here are tiny (C(n, k) well under the sampling budget), so the min/max
achievable diversity is computed by full enumeration — making every normalized
score exact and hand-checkable.
"""

import sys
import os
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dataset_module.common import Impression
from diversity_module.content_diversity_normalized import normalized_content_diversity
from recommender_module.common.io import (
    recommended_per_impression_from_ranks,
    recommended_per_impression_from_topk,
)

logger = logging.getLogger(__name__)


# Four articles: N1/N3 point one way, N2/N4 the orthogonal way. So among the
# C(4,2)=6 candidate pairs the achievable cosine-distance ILD ranges over
# {0.0 (identical pair), 1.0 (orthogonal pair)} → min 0.0, max 1.0.
EMB = {
    "N1": np.array([1.0, 0.0]),
    "N2": np.array([0.0, 1.0]),
    "N3": np.array([1.0, 0.0]),
    "N4": np.array([0.0, 1.0]),
}
POOL = ["N1", "N2", "N3", "N4"]


def _imp(impr_id, candidates, labels=None):
    labels = labels if labels is not None else [0] * len(candidates)
    return Impression(impr_id, f"U{impr_id}", "t", candidates, labels)


# ---------------------------------------------------------------------------
# normalized_content_diversity
# ---------------------------------------------------------------------------

def test_normalized_is_one_for_most_diverse_recommendation():
    # Recommending the orthogonal pair = the most diverse 2-subset of the pool.
    impressions = [_imp(1, POOL)]
    score = normalized_content_diversity(impressions, {1: ["N1", "N2"]}, EMB)

    assert abs(score - 1.0) < 1e-9
    logger.info(
        "Most-diverse recommendation normalizes to 1.0 — expected 1.0000, actual %.4f", score
    )


def test_normalized_is_zero_for_least_diverse_recommendation():
    # Recommending two identical-direction articles = the least diverse 2-subset.
    impressions = [_imp(1, POOL)]
    score = normalized_content_diversity(impressions, {1: ["N1", "N3"]}, EMB)

    assert abs(score - 0.0) < 1e-9
    logger.info(
        "Least-diverse recommendation normalizes to 0.0 — expected 0.0000, actual %.4f", score
    )


def test_normalized_averages_across_impressions():
    # imp1 most diverse (1.0), imp2 least diverse (0.0) → mean 0.5.
    impressions = [_imp(1, POOL), _imp(2, POOL)]
    recommended = {1: ["N1", "N2"], 2: ["N1", "N3"]}
    score = normalized_content_diversity(impressions, recommended, EMB)

    assert abs(score - 0.5) < 1e-9
    logger.info(
        "Normalized diversity averages per-impression scores — expected 0.5000, actual %.4f", score
    )


def test_normalized_skips_when_pool_has_no_freedom():
    # Pool size == number recommended: only one possible selection, so min==max
    # and the impression contributes nothing → no scorable impressions → 0.0.
    impressions = [_imp(1, ["N1", "N2"])]
    score = normalized_content_diversity(impressions, {1: ["N1", "N2"]}, EMB)

    assert score == 0.0
    logger.info(
        "An impression whose pool offers no alternative selection is skipped — "
        "expected 0.0000, actual %.4f", score
    )


def test_normalized_skips_recommendation_with_fewer_than_two_vectors():
    impressions = [_imp(1, POOL)]
    score = normalized_content_diversity(impressions, {1: ["N1"]}, EMB)

    assert score == 0.0
    logger.info(
        "A single-item recommendation has no intra-list diversity and is skipped — "
        "expected 0.0000, actual %.4f", score
    )


def test_normalized_ignores_ids_without_embeddings():
    # "X" has no vector: dropped from both the recommendation and the pool, leaving
    # the same orthogonal-vs-pool computation → 1.0.
    emb = dict(EMB)
    impressions = [_imp(1, ["N1", "N2", "N3", "N4", "X"])]
    score = normalized_content_diversity(impressions, {1: ["N1", "N2", "X"]}, emb)

    assert abs(score - 1.0) < 1e-9
    logger.info(
        "Article ids without an embedding are ignored — expected 1.0000, actual %.4f", score
    )


def test_normalized_score_in_unit_range_random_vectors():
    rng = np.random.default_rng(0)
    emb = {f"N{i}": rng.normal(size=8) for i in range(12)}
    pool = list(emb)
    impressions = [_imp(1, pool)]
    score = normalized_content_diversity(impressions, {1: pool[:4]}, emb)

    assert 0.0 <= score <= 1.0
    logger.info(
        "Normalized diversity stays within [0, 1] — actual %.4f", score
    )


# ---------------------------------------------------------------------------
# per-impression recommendation readers (io.py)
# ---------------------------------------------------------------------------

def test_recommended_from_ranks_takes_topk_by_clicks(tmp_path):
    # Candidates [A,B,C], ranks [2,3,1] (C best, then A, then B); k = 2 clicks
    # → top-2 candidates are C and A, in rank order. Prediction files are
    # "{impr_id} [ranks]" (no user_id column).
    impressions = [_imp(1, ["A", "B", "C"], labels=[1, 0, 1])]
    pred = tmp_path / "prediction_random.txt"
    pred.write_text("1 [2,3,1]\n", encoding="utf-8")

    recommended = recommended_per_impression_from_ranks(str(pred), impressions)

    assert recommended == {1: ["C", "A"]}
    logger.info(
        "from_ranks picks the k=clicks best-ranked candidates — expected {1: ['C','A']}, actual %s",
        recommended,
    )


def test_recommended_from_topk_reads_id_token(tmp_path):
    # Ground-truth/top-k format: "{impr_id} {user_id} [positions] [ids]".
    topk = tmp_path / "ground_truth.txt"
    topk.write_text("1 U1 [1,2] [C,A]\n2 U2 [1] [B]\n", encoding="utf-8")

    recommended = recommended_per_impression_from_topk(str(topk))

    assert recommended == {1: ["C", "A"], 2: ["B"]}
    logger.info(
        "from_topk reads the chosen-ids token per impression — actual %s", recommended
    )
