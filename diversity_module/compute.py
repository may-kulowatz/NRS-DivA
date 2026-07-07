"""Diversity-score computation and the file-staleness check it relies on.

These are diversity-stage internals: the driver in ``__main__`` computes every
metric for each processed user-article file (``_compute_scores``) and rebuilds a
processed file only when its source changed (``_stale``). The computed values are
then handed to :mod:`run_manifest` for persistence — this module knows nothing
about how or where results are stored.
"""

import os


def _stale(output, *inputs):
    """True if output is missing or older than any existing input.

    Lets a derived file (the processed per-user file) be rebuilt when its source
    changed, instead of being skipped just because some version already exists.
    """
    if not os.path.exists(output):
        return True
    out_mtime = os.path.getmtime(output)
    return any(os.path.exists(i) and os.path.getmtime(i) > out_mtime for i in inputs)


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