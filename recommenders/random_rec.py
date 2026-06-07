"""Random recommender (dataset-agnostic).

Assigns each candidate in an impression a uniform random score. Output writing
is handled by recommenders/io.py.
"""

import numpy as np


def random_recommend(impressions, seed=42):
    """Score every candidate of every impression uniformly at random.

    Returns [(impr_id, user_id, scores)] with one score array per impression.
    The same seed reproduces identical scores.
    """
    rng = np.random.default_rng(seed)
    return [
        (imp.impr_id, imp.user_id, rng.random(len(imp.candidate_ids)))
        for imp in impressions
    ]