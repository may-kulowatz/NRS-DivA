"""Diversity-score computation, run-manifest I/O, and small file helpers.

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

That file is the **dashboard's data source** — the dashboard only reads it, never
recomputes. The two stage timestamps are read straight from the on-disk files'
mtimes (no extra plumbing); only each metric's ``calculated_at`` is stamped at
compute time. Metrics are merged into the manifest, so a measure not recomputed
this run keeps its previous value + timestamp.

Earlier versions stored a flat ``{recommender: {metric: value}}`` map in
``diversity_scores.json``; ``load_manifest`` migrates such a file into the nested
shape on read, so existing datasets keep working until the next pipeline run.
"""

import json
import os
from datetime import datetime

# Current manifest filename and the legacy flat-scores filename it superseded.
MANIFEST_FILENAME = "run_manifest.json"
_LEGACY_FILENAME = "diversity_scores.json"


def _file_sig(path):
    """Cheap change-signature for a file: modification time + size.

    Still used by the dashboard to key its in-memory parse of the (large)
    user-article files, so a changed file invalidates that parse.
    """
    st = os.stat(path)
    return f"{st.st_mtime_ns}-{st.st_size}"


def _stale(output, *inputs):
    """True if output is missing or older than any existing input.

    Lets a derived file (the processed per-user file) be rebuilt when its source
    changed, instead of being skipped just because some version already exists.
    """
    if not os.path.exists(output):
        return True
    out_mtime = os.path.getmtime(output)
    return any(os.path.exists(i) and os.path.getmtime(i) > out_mtime for i in inputs)


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


# --------------------------------------------------------------------------- #
# Run manifest: results + "what ran when", per recommender.
# --------------------------------------------------------------------------- #

def _iso_mtime(path):
    """File's modification time as an ISO-8601 string (to seconds), or None."""
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
    except OSError:
        return None


def _migrate_flat_scores(flat):
    """Convert a legacy ``{recommender: {metric: value}}`` map into the nested
    manifest shape. Stage timestamps are unknown for legacy data (filled in from
    file mtimes the next time the pipeline saves), and ``calculated_at`` is null."""
    manifest = {}
    for rec, metrics in flat.items():
        manifest[rec] = {
            "predictions_generated_at": None,
            "predictions_processed_at": None,
            "metrics": {
                key: {"value": value, "calculated_at": None}
                for key, value in metrics.items()
            },
        }
    return manifest


def load_manifest(out_dir):
    """Load the run manifest for a dataset's output dir.

    Prefers ``run_manifest.json``; if only the legacy ``diversity_scores.json`` is
    present it is migrated into the nested shape in memory (so the dashboard keeps
    working before the next pipeline run). Returns ``{}`` if neither exists.
    """
    manifest_path = os.path.join(out_dir, MANIFEST_FILENAME)
    if os.path.exists(manifest_path):
        return _load_scores(manifest_path)
    legacy_path = os.path.join(out_dir, _LEGACY_FILENAME)
    if os.path.exists(legacy_path):
        return _migrate_flat_scores(_load_scores(legacy_path))
    return {}


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


class MetricUnavailable(Exception):
    """A metric's inputs couldn't be obtained (e.g. embeddings can't be
    downloaded). The metric is skipped for the affected run rather than failing
    the whole pipeline."""


def _compute_scores(path, metric_defs):
    """Compute every metric for one user-article file.

    metric_defs : list of (metric_key, fn) where fn(path) -> float.

    Returns ``{metric_key: value}``. A metric whose fn raises MetricUnavailable is
    skipped (left out) so the others still compute.
    """
    scores = {}
    for key, fn in metric_defs:
        try:
            scores[key] = fn(path)
        except MetricUnavailable as exc:
            print(f"  Skipping {key}: {exc}")
    return scores