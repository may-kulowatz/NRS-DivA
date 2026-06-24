"""Popularity recommender (dataset-agnostic).

Scores each candidate by how many clicks it has accumulated in chronologically
earlier impressions, so no future information leaks into a score. Output writing
is handled by recommender_module/common/io.py.
"""

from collections import defaultdict

import numpy as np


def popular_recommend(impressions):
    """Score candidates by prior click count, processed in timestamp order.

    Returns [(impr_id, user_id, scores)] in the *original* input order. The
    chronologically first impression scores all-zero (nothing observed yet);
    later impressions reflect clicks seen strictly before them.
    """
    sorted_imps = sorted(impressions, key=lambda imp: imp.timestamp)

    click_counts = defaultdict(int)
    scores_by_id = {}

    for imp in sorted_imps:
        scores_by_id[imp.impr_id] = np.array(
            [float(click_counts[aid]) for aid in imp.candidate_ids]
        )
        # Update counts with clicks observed in this impression.
        for aid, label in zip(imp.candidate_ids, imp.labels):
            if label == 1:
                click_counts[aid] += 1

    # Return results in original (input) order, not chronological order.
    return [(imp.impr_id, imp.user_id, scores_by_id[imp.impr_id]) for imp in impressions]