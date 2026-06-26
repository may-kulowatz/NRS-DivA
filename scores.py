"""Diversity-score computation, results I/O, and small file helpers.

The diversity scores are the one pipeline step with no natural file output, so the
results are persisted to ``data_processed/<dataset>/diversity_scores.json`` as a
plain ``{recommender: {metric: value}}`` map. That file is the **dashboard's data
source** — the dashboard only reads it, never recomputes.

Scores are recomputed from scratch on every run: there is no reuse cache. The
interactive front-end already reports what exists and asks what to (re)run, so a
signature-keyed "skip if unchanged" cache on top of that was redundant.
"""

import json
import os


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
    """Load the persisted ``{recommender: {metric: value}}`` map, or ``{}``."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_scores(path, scores):
    """Persist the ``{recommender: {metric: value}}`` map as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)


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