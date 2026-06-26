"""Normalized intra-list content diversity (EBNeRD-style).

The plain intra-list diversity (ILD) of a recommended set says how varied it is,
but not how varied it *could* have been: a small candidate pool of near-identical
articles caps the achievable diversity no matter how good the recommender is.
This metric removes that ceiling by normalizing, **per impression**, against the
range of diversity reachable from that impression's candidate pool:

    normalized = (ILD(recommended) - ILD_min) / (ILD_max - ILD_min)

where ILD_min / ILD_max are the least / most diverse selections of the same size
from the impression's candidates (estimated by ``IntralistDiversity._candidate_diversity``).
A score near 1 means the recommender picked about as diverse a set as the pool
allowed; near 0 means about as homogeneous as possible. The per-impression scores
are averaged.

This needs each impression's candidate pool (``Impression.candidate_ids``) and the
recommender's per-impression choices, so — unlike the per-user ``content_diversity``
— it operates on impressions, not the aggregated per-user files.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from diversity_module.content_diversity_ebnerd import IntralistDiversity


def normalized_content_diversity(
    impressions,
    recommended_by_impr,
    embeddings,
    *,
    lookup_key="vector",
    max_combinations=1000,
    seed=42,
):
    """Average normalized intra-list content diversity across impressions.

    impressions          : list of Impression records (carry candidate_ids).
    recommended_by_impr  : {impr_id: [article_id, ...]} the recommender chose.
    embeddings           : {article_id: vector} (e.g. from load_precomputed_embeddings
                           or load_news_embeddings).
    max_combinations     : per-impression budget for estimating the min/max
                           achievable diversity; above it the candidate subsets are
                           randomly sampled rather than fully enumerated.

    An impression is scored only when its recommended set has >= 2 embeddable
    articles and its candidate pool has more embeddable articles than were
    recommended (otherwise there is no room for the selection to be more or less
    diverse). Returns the mean normalized score, or 0.0 if nothing was scorable.
    """
    div = IntralistDiversity()
    lookup = {aid: {lookup_key: v} for aid, v in embeddings.items()}

    per_impr = []
    for imp in impressions:
        rec_ids = [r for r in recommended_by_impr.get(imp.impr_id, []) if r in lookup]
        if len(rec_ids) < 2:
            continue
        pool = [c for c in imp.candidate_ids if c in lookup]
        n = len(rec_ids)
        if len(pool) <= n:
            continue  # no freedom: every n-subset is the whole pool

        actual = div([rec_ids], lookup, lookup_key, cosine_distances)[0]
        d_min, d_max = div._candidate_diversity(
            pool,
            n,
            lookup,
            lookup_key,
            cosine_distances,
            max_number_combinations=max_combinations,
            seed=seed,
        )
        if np.isnan(actual) or d_max <= d_min:
            continue
        # Clip: with sampling the estimated min/max can sit just inside the true
        # range, which would push the ratio slightly outside [0, 1].
        per_impr.append(float(np.clip((actual - d_min) / (d_max - d_min), 0.0, 1.0)))

    return float(np.mean(per_impr)) if per_impr else 0.0
