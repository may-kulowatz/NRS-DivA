"""Shared, dataset-agnostic writers for prediction and user-article files.

Every recommender produces the same kind of output, so the writing logic lives here once instead of being copy-pasted
into each recommender module.
"""

from collections import defaultdict

import numpy as np
from tqdm import tqdm


def processed_filename(name):
    """File name for a recommender's processed per-user file.

    Ground truth is the users' actual clicks, not a prediction, so it drops the
    "prediction_" prefix: ``processed_ground_truth.txt`` vs
    ``prediction_processed_<rec>.txt``. Single source of truth for this naming,
    shared by the pipeline (which writes the files) and the dashboard (which
    reads them).
    """
    if name == "ground_truth":
        return "processed_ground_truth.txt"
    return f"prediction_processed_{name}.txt"


def save_predictions(results, output_file):
    """Write full per-candidate rankings (rank 1 = highest score).

    results: iterable of (impr_id, user_id, scores) where scores is a 1-D array.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for impr_id, user_id, scores in tqdm(results):
            ranks = (np.argsort(np.argsort(scores)[::-1]) + 1).tolist()
            f.write(f"{impr_id} {user_id} [" + ",".join(map(str, ranks)) + "]\n")


def save_predictions_topk(results, impressions, output_file):
    """Write the top-K recommended candidates per impression.

    K for each impression is the number of clicks it actually received,
    so every recommender is asked for exactly as many items as the user clicked
    — making the outputs directly comparable to the ground truth.
    """
    candidate_ids = {imp.impr_id: imp.candidate_ids for imp in impressions}
    k_by_impr = {imp.impr_id: sum(imp.labels) for imp in impressions}

    with open(output_file, "w", encoding="utf-8") as f:
        for impr_id, user_id, scores in tqdm(results):
            k = k_by_impr.get(impr_id, 0)
            ids = candidate_ids[impr_id]
            top_k_idx = np.argsort(scores)[::-1][:k]
            positions = [i + 1 for i in top_k_idx]
            chosen_ids = [ids[i] for i in top_k_idx]
            f.write(
                f"{impr_id} {user_id} ["
                + ",".join(map(str, positions))
                + "] ["
                + ",".join(chosen_ids)
                + "]\n"
            )


def _write_user_article_map(selections, article_meta, output_file):
    """Aggregate per-impression (user_id, chosen_ids) pairs into the per-user map.

    selections : iterable of (user_id, [article_id, ...]) — one item per
                 impression; ids for the same user are concatenated in order.
    article_meta : {article_id: (topic, subtopic)} from a dataset adapter; only
                 the topic (index 0) is written.

    This is the single writer shared by every ``save_user_article_map*`` entry
    point; they differ only in how they produce the (user_id, chosen_ids) stream.
    """
    user_articles = defaultdict(list)
    for user_id, ids in selections:
        if ids:
            user_articles[user_id].extend(ids)

    with open(output_file, "w", encoding="utf-8") as f:
        for user_id, articles in user_articles.items():
            topics = [article_meta.get(a, ("unknown", "none"))[0] for a in articles]
            f.write(
                f"{user_id} ["
                + ",".join(articles)
                + "] ["
                + ",".join(topics)
                + "]\n"
            )


def save_user_article_map(topk_file, article_meta, output_file):
    """Aggregate an existing top-k file into per-user article/topic lists.

    topk_file    : a "{impr_id} {user_id} [pos] [ids]" file (top-k or ground truth)
    article_meta : {article_id: (topic, subtopic)} from a dataset adapter
    """
    def selections():
        with open(topk_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                inner_ids = parts[3][1:-1]
                yield parts[1], inner_ids.split(",") if inner_ids else []

    _write_user_article_map(selections(), article_meta, output_file)


def _topk_indices(ranks, k):
    """Candidate indices of the k best-ranked items (rank 1 = best)."""
    return np.argsort(ranks)[:k]


def _read_rank_file(prediction_file):
    """Parse a full-rank prediction file into {impr_id: [rank, ...]}.

    The rank list is the last bracketed token, so both the model format
    ("impr_id [ranks]") and the random/popular format ("impr_id user_id [ranks]")
    parse the same way.
    """
    pred_ranks = {}
    with open(prediction_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            pred_ranks[int(parts[0])] = list(map(int, parts[-1][1:-1].split(",")))
    return pred_ranks


def recommended_per_impression_from_ranks(prediction_file, impressions):
    """{impr_id: [article_id, ...]} — the top-k a rank file chose per impression.

    K per impression is the number of clicks it kept (``sum(imp.labels)``), the
    same selection used to build the per-user map. Impressions absent from the
    prediction file are omitted. The user_id is not needed here, so it is dropped.
    """
    pred_ranks = _read_rank_file(prediction_file)
    recommended = {}
    for imp in impressions:
        ranks = pred_ranks.get(imp.impr_id)
        if ranks is None:
            continue
        k = sum(imp.labels)
        recommended[imp.impr_id] = [imp.candidate_ids[i] for i in _topk_indices(ranks, k)]
    return recommended


def recommended_per_impression_from_topk(topk_file):
    """{impr_id: [article_id, ...]} from a "{impr_id} {user_id} [pos] [ids]" file.

    Used for ground truth (and any top-k file): the chosen ids are the last
    bracketed token, the impression id is column 0.
    """
    recommended = {}
    with open(topk_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            inner_ids = parts[3][1:-1]
            recommended[int(parts[0])] = inner_ids.split(",") if inner_ids else []
    return recommended


def save_user_article_map_from_ranks(
    prediction_file, impressions, article_meta, output_file
):
    """Build the per-user map straight from a full-rank prediction file.

    prediction_file : a recommender's full-rank output. K per impression is the
                      number of clicks it kept; the user_id always comes from the
                      impressions, not the file.
    """
    recommended = recommended_per_impression_from_ranks(prediction_file, impressions)

    def selections():
        for imp in impressions:
            yield imp.user_id, recommended.get(imp.impr_id, [])

    _write_user_article_map(selections(), article_meta, output_file)