"""Tests for the diversity stage's compute internals in diversity_module/compute.py.

The stage recomputes every metric on every run (there is no reuse cache), so these
tests exercise the compute helper with a counting metric function plus the
_stale rebuild check. They're fast and need no dataset.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from diversity_module.compute import MetricUnavailable, _compute_scores, _stale

logger = logging.getLogger(__name__)


def write_file(tmp_path, text, name="user_articles.txt"):
    p = tmp_path / name
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


def test_computes_every_metric(tmp_path):
    f = write_file(tmp_path, "U1 [N1,N2] [a,b]\n")
    topic = Counter(0.3)
    content = Counter(0.7)

    scores = _compute_scores(f, [("topic_diversity", topic), ("content_diversity", content)])

    assert scores == {"topic_diversity": 0.3, "content_diversity": 0.7}
    assert topic.calls == 1 and content.calls == 1
    logger.info("Every metric is computed once — expected both called, actual %d/%d",
                topic.calls, content.calls)


def test_recomputes_on_every_call(tmp_path):
    f = write_file(tmp_path, "U1 [N1,N2] [a,b]\n")
    fn = Counter(0.42)
    defs = [("topic_diversity", fn)]

    _compute_scores(f, defs)
    _compute_scores(f, defs)  # no cache: runs again even though nothing changed

    assert fn.calls == 2
    logger.info("Scores recompute every call (no reuse cache) — expected 2 calls, actual %d",
                fn.calls)


def test_unavailable_metric_is_skipped(tmp_path):
    f = write_file(tmp_path, "U1 [N1,N2] [a,b]\n")

    def boom(_path):
        raise MetricUnavailable("embeddings unavailable")

    scores = _compute_scores(f, [("topic_diversity", Counter(0.5)), ("content_diversity", boom)])

    # The failing metric is left out; the other still computes.
    assert scores == {"topic_diversity": 0.5}
    logger.info("A MetricUnavailable metric is skipped, others still compute.")


def test_stale_when_output_missing_or_older(tmp_path):
    src = write_file(tmp_path, "source\n", name="src.txt")
    out = write_file(tmp_path, "out\n", name="out.txt")

    # out is newer than src -> not stale; a missing output -> stale.
    assert _stale(out, src) is False
    assert _stale(str(tmp_path / "missing.txt"), src) is True

    # Touch src so it becomes newer than out -> stale.
    os.utime(src, (os.path.getmtime(out) + 10, os.path.getmtime(out) + 10))
    assert _stale(out, src) is True
    logger.info("_stale is True when the output is missing or older than an input.")