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
    ``prediction_processed_<rec>.txt``.
    """
    if name == "ground_truth":
        return "processed_ground_truth.txt"
    return f"prediction_processed_{name}.txt"


def save_predictions(results, output_file):
    """Write full per-candidate rankings (rank 1 = highest score).

    results: iterable of (impr_id, user_id, scores) where scores is a 1-D array.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for impr_id, _user_id, scores in tqdm(results):
            ranks = (np.argsort(np.argsort(scores)[::-1]) + 1).tolist()
            f.write(f"{impr_id} [" + ",".join(map(str, ranks)) + "]\n")


def _write_user_article_map(selections, article_meta, output_file):
    """Aggregate per-impression (user_id, chosen_ids) pairs into the per-user map.

    selections : iterable of (user_id, [article_id, ...]) — one item per
                 impression; ids for the same user are concatenated in order.
    article_meta : {article_id: topic} from a dataset adapter.

    This is the single writer shared by every ``save_user_article_map*`` entry
    point; they differ only in how they produce the (user_id, chosen_ids) stream.
    """
    user_articles = defaultdict(list)
    for user_id, ids in selections:
        if ids:
            user_articles[user_id].extend(ids)

    with open(output_file, "w", encoding="utf-8") as f:
        for user_id, articles in user_articles.items():
            topics = [article_meta.get(a, "unknown") for a in articles]
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
    article_meta : {article_id: topic} from a dataset adapter
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

    Every prediction file uses the ``"impr_id [ranks]"`` layout (the impression id
    first, the rank list as the last bracketed token), so only those two tokens are
    read — robust to any extra columns a file might carry.
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