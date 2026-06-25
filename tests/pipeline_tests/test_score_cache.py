"""Tests for the diversity-score cache in pipeline.py.

The cache exists so the score-calculation step (the only step with no file
output) is not redone on every pipeline run. These tests exercise the cache
helper directly with a counting metric function, so they're fast and don't need
a full dataset.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from pipeline import _compute_run_scores, _file_sig, _load_score_cache, _save_score_cache

logger = logging.getLogger(__name__)


def write_file(tmp_path, text):
    p = tmp_path / "user_articles.txt"
    p.write_text(text, encoding="utf-8")
    return str(p)


class Counter:
    """A metric fn that records how many times it actually ran."""
    def __init__(self, value=0.5):
        self.calls = 0
        self.value = value

    def __call__(self, path):
        self.calls += 1
        return self.value


def test_first_run_computes(tmp_path):
    f = write_file(tmp_path, "U1 [N1,N2] [a,b]\n")
    fn = Counter(0.42)

    scores, cache, n_computed = _compute_run_scores(f, [("topic_diversity", fn)], {})

    assert n_computed == 1
    assert fn.calls == 1
    assert scores["topic_diversity"] == 0.42
    logger.info(
        "First evaluation computes the score — expected 1 computed call, actual %d", fn.calls
    )


def test_reuses_when_file_unchanged(tmp_path):
    f = write_file(tmp_path, "U1 [N1,N2] [a,b]\n")
    fn = Counter(0.42)
    defs = [("topic_diversity", fn)]

    _, cache, _ = _compute_run_scores(f, defs, {})
    scores, _, n_computed = _compute_run_scores(f, defs, cache)

    assert n_computed == 0
    assert fn.calls == 1  # not called again
    assert scores["topic_diversity"] == 0.42
    logger.info(
        "Unchanged input reuses the cached score — expected 0 recomputes, actual %d", n_computed
    )


def test_recomputes_when_file_changes(tmp_path):
    f = write_file(tmp_path, "U1 [N1,N2] [a,b]\n")
    fn = Counter()
    defs = [("topic_diversity", fn)]

    _, cache, _ = _compute_run_scores(f, defs, {})
    # Change the file content (and size), which changes its signature.
    write_file(tmp_path, "U1 [N1,N2,N3] [a,b,c]\nU2 [N4,N5] [a,b]\n")
    _, _, n_computed = _compute_run_scores(f, defs, cache)

    assert n_computed == 1
    assert fn.calls == 2
    logger.info(
        "Changed input triggers a recompute — expected 1 recompute, actual %d", n_computed
    )


def test_new_metric_computed_others_reused(tmp_path):
    f = write_file(tmp_path, "U1 [N1,N2] [a,b]\n")
    topic = Counter(0.3)
    content = Counter(0.7)

    # First run with only topic cached.
    _, cache, _ = _compute_run_scores(f, [("topic_diversity", topic)], {})
    # Second run adds a content metric; topic should be reused, content computed.
    scores, _, n_computed = _compute_run_scores(
        f, [("topic_diversity", topic), ("content_diversity", content)], cache
    )

    assert n_computed == 1
    assert topic.calls == 1    # reused
    assert content.calls == 1  # newly computed
    assert scores == {"topic_diversity": 0.3, "content_diversity": 0.7}
    logger.info(
        "Only the newly added metric is computed — expected topic reused, content computed"
    )


def test_cache_roundtrip_json(tmp_path):
    path = str(tmp_path / "diversity_scores.json")
    cache = {"random": {"topic_diversity": {"value": 0.5, "sig": "123-45"}}}

    _save_score_cache(path, cache)
    loaded = _load_score_cache(path)

    assert loaded == cache
    logger.info("Cache round-trips through JSON unchanged.")


def test_load_missing_cache_returns_empty(tmp_path):
    loaded = _load_score_cache(str(tmp_path / "does_not_exist.json"))
    assert loaded == {}
    logger.info("Loading a missing cache file returns an empty dict.")


def test_signature_changes_with_content(tmp_path):
    f = write_file(tmp_path, "short\n")
    sig1 = _file_sig(f)
    write_file(tmp_path, "a much longer line of content\n")
    sig2 = _file_sig(f)
    assert sig1 != sig2
    logger.info("File signature changes when the file content changes.")
