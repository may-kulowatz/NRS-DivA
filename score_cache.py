"""Diversity-score cache and staleness helpers.

Computing the diversity scores is the only pipeline step with no file output, so
without a cache it re-runs on every invocation even when nothing changed. Scores
are persisted to ``data_processed/<dataset>/diversity_scores.json``, keyed per
recommender and metric, alongside a signature of the user-article file the score
was computed from. A score is reused as long as that signature still matches.

This module is dataset- and pipeline-agnostic: it knows only about files,
signatures, and ``(metric_key, fn)`` pairs, so it is unit-tested on its own.
"""

import json
import os


def _file_sig(path):
    """Cheap change-signature for a file: modification time + size."""
    st = os.stat(path)
    return f"{st.st_mtime_ns}-{st.st_size}"


def _stale(output, *inputs):
    """True if output is missing or older than any existing input.

    Lets a derived file be rebuilt when its source changed, instead of being
    skipped just because some version of it already exists.
    """
    if not os.path.exists(output):
        return True
    out_mtime = os.path.getmtime(output)
    return any(os.path.exists(i) and os.path.getmtime(i) > out_mtime for i in inputs)


def _load_score_cache(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_score_cache(path, cache):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


class MetricUnavailable(Exception):
    """A metric's inputs couldn't be obtained (e.g. embeddings can't be
    downloaded). The metric is skipped for the affected runs rather than failing
    the whole pipeline."""


def _compute_run_scores(path, metric_defs, prev_cache, force=False):
    """Compute (or reuse) every metric for one user-article file.

    metric_defs : list of (metric_key, fn) where fn(path) -> float.
    prev_cache  : the cached {metric_key: {"value", "sig"}} for this run.
    force       : when True, recompute every metric even if the cached signature
                  still matches (the interactive "recalculate diversity" option).

    Returns (scores, cache, n_computed): scores is {metric_key: value} for
    display; cache is the refreshed {metric_key: {"value", "sig"}} to persist;
    n_computed counts how many metrics were actually (re)calculated. A metric
    whose fn raises MetricUnavailable is skipped (left out of scores/cache) so
    the others still compute.
    """
    sig = _file_sig(path)
    scores, cache, n_computed = {}, {}, 0
    for key, fn in metric_defs:
        prev = prev_cache.get(key)
        if not force and prev is not None and prev.get("sig") == sig:
            value = prev["value"]
        else:
            try:
                value = fn(path)
            except MetricUnavailable as exc:
                print(f"  Skipping {key}: {exc}")
                continue
            n_computed += 1
        scores[key] = value
        cache[key] = {"value": value, "sig": sig}
    return scores, cache, n_computed