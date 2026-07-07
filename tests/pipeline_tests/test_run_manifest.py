"""Tests for the run-manifest I/O in run_manifest.py — the shared contract the
recommender and diversity stages write and the dashboard reads.

Covers the JSON load/save round-trip, the nested manifest schema, and the record
helpers. Fast and need no dataset.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from run_manifest import (
    _load_scores,
    _save_scores,
    load_manifest,
    save_manifest,
    metric_value,
    record_metric,
    record_stage_times,
)

logger = logging.getLogger(__name__)


def write_file(tmp_path, text, name="user_articles.txt"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_scores_roundtrip_json(tmp_path):
    path = str(tmp_path / "run_manifest.json")
    scores = {"random": {"topic_diversity": 0.5, "content_diversity": 0.7}}

    _save_scores(path, scores)
    loaded = _load_scores(path)

    assert loaded == scores
    logger.info("Scores round-trip through JSON unchanged.")


def test_load_missing_scores_returns_empty(tmp_path):
    assert _load_scores(str(tmp_path / "does_not_exist.json")) == {}
    logger.info("Loading a missing scores file returns an empty dict.")


def test_manifest_roundtrip_and_metric_value(tmp_path):
    manifest = {}
    record_metric(manifest.setdefault("nrms", {}), "topic_diversity", 0.91, "2026-06-27T12:00:00")
    save_manifest(str(tmp_path), manifest)

    loaded = load_manifest(str(tmp_path))
    assert loaded["nrms"]["metrics"]["topic_diversity"] == {
        "value": 0.91, "calculated_at": "2026-06-27T12:00:00"
    }
    assert metric_value(loaded, "nrms", "topic_diversity") == 0.91
    assert metric_value(loaded, "nrms", "content_diversity") is None
    assert metric_value(loaded, "missing_rec", "topic_diversity") is None
    logger.info("Manifest round-trips and metric_value reads the nested value.")


def test_record_stage_times_from_file_mtimes(tmp_path):
    raw = write_file(tmp_path, "1 [1]\n", name="prediction_random.txt")
    proc = write_file(tmp_path, "U1 [N1] [a]\n", name="processed_random.txt")
    entry = {}
    record_stage_times(entry, raw, proc)
    # Both files exist -> both timestamps are ISO strings; a missing file -> None.
    assert entry["predictions_generated_at"] is not None
    assert entry["predictions_processed_at"] is not None
    record_stage_times(entry, raw, str(tmp_path / "nope.txt"))
    assert entry["predictions_processed_at"] is None
    logger.info("record_stage_times reads mtimes, None for missing files.")


def test_load_missing_manifest_returns_empty(tmp_path):
    assert load_manifest(str(tmp_path)) == {}
    logger.info("Loading a dir with no run_manifest.json returns an empty dict.")