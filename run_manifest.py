"""Run-manifest I/O — the shared contract between the pipeline stages and the dashboard.

The diversity scores are the one pipeline step with no natural file output, so the
results are persisted to ``data_processed/<dataset>/run_manifest.json`` — a
per-recommender **run manifest** that records both the results and *when* each
stage ran::

    {
      "<recommender>": {
        "predictions_generated_at": "<iso>",   # mtime of its prediction file
        "predictions_processed_at": "<iso>",   # mtime of its processed file (or null)
        "metrics": {
          "<metric_key>": {"value": <float>, "calculated_at": "<iso>"}
        }
      }
    }

The manifest is written by two stages and read by a third, which is why it lives
here rather than inside any one of them: ``recommender_module`` stamps the stage
timestamps, ``diversity_module`` records the metric values, and the dashboard is a
read-only consumer — it only reads this file, never recomputes.

The two stage timestamps are read straight from the on-disk files' mtimes; only
each metric's ``calculated_at`` is stamped at compute time. Metrics are merged
into the manifest, so a measure not recomputed this run keeps its previous value +
timestamp.
"""

import json
import os
from datetime import datetime

MANIFEST_FILENAME = "run_manifest.json"


def _load_scores(path):
    """Load a JSON object from ``path``, or ``{}`` if missing/unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_scores(path, scores):
    """Persist a JSON object to ``path``."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)


def _iso_mtime(path):
    """File's modification time as an ISO-8601 string (to seconds), or None."""
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
    except OSError:
        return None


def load_manifest(out_dir):
    """Load the run manifest for a dataset's output dir, or ``{}`` if none exists."""
    return _load_scores(os.path.join(out_dir, MANIFEST_FILENAME))


def save_manifest(out_dir, manifest):
    """Persist the run manifest to ``out_dir/run_manifest.json``."""
    _save_scores(os.path.join(out_dir, MANIFEST_FILENAME), manifest)


def metric_value(manifest, recommender, metric_key):
    """The recorded value for one (recommender, metric) in the manifest, or None."""
    return (
        manifest.get(recommender, {})
        .get("metrics", {})
        .get(metric_key, {})
        .get("value")
    )


def record_metric(rec_entry, metric_key, value, when):
    """Record a metric's ``value`` + ``calculated_at`` into a recommender entry
    (``manifest[rec]``), creating the ``metrics`` block if needed."""
    rec_entry.setdefault("metrics", {})[metric_key] = {
        "value": value,
        "calculated_at": when,
    }


def record_stage_times(rec_entry, raw_path, processed_path):
    """Set a recommender entry's stage timestamps from the mtimes of its
    prediction file and processed per-user file (``None`` if a file is missing)."""
    rec_entry["predictions_generated_at"] = _iso_mtime(raw_path)
    rec_entry["predictions_processed_at"] = _iso_mtime(processed_path)