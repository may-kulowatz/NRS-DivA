"""Ground-truth extraction (dataset-agnostic).

Turns normalized Impression records into the set of articles each user actually
clicked, written in the same top-k line format the recommenders use so it can be
compared against them directly.
"""

from tqdm import tqdm


def extract_ground_truth(impressions):
    """Return [(impr_id, user_id, positions, ids)] of the clicked candidates.

    positions are 1-indexed locations within the impression; ids are the
    matching article ids. Both are empty for impressions with no clicks.
    """
    results = []
    for imp in impressions:
        clicked = [
            (i + 1, imp.candidate_ids[i])
            for i, label in enumerate(imp.labels)
            if label == 1
        ]
        positions = [pos for pos, _ in clicked]
        ids = [aid for _, aid in clicked]
        results.append((imp.impr_id, imp.user_id, positions, ids))
    return results


def save_ground_truth(results, output_file):
    """Write extract_ground_truth() output as "{impr_id} {user_id} [pos] [ids]"."""
    with open(output_file, "w", encoding="utf-8") as f:
        for impr_id, user_id, positions, ids in tqdm(results):
            f.write(
                f"{impr_id} {user_id} ["
                + ",".join(map(str, positions))
                + "] ["
                + ",".join(ids)
                + "]\n"
            )