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
    article_meta : {article_id: (topic, subtopic)} from a dataset adapter.

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
            subtopics = [article_meta.get(a, ("unknown", "none"))[1] for a in articles]
            f.write(
                f"{user_id} ["
                + ",".join(articles)
                + "] ["
                + ",".join(topics)
                + "] ["
                + ",".join(subtopics)
                + "]\n"
            )


def save_user_article_map(topk_file, article_meta, output_file):
    """Aggregate an existing top-k file into per-user article/topic/subtopic lists.

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


def save_user_article_map_from_results(results, impressions, article_meta, output_file):
    """Build the per-user map straight from scored results, skipping the top-k file.

    Used for the score-based recommenders (random, popular). The top-K selection
    per impression is identical to ``save_predictions_topk`` (the k highest-score
    candidates, k = number of clicks); only the on-disk intermediate is dropped.
    """
    candidate_ids = {imp.impr_id: imp.candidate_ids for imp in impressions}
    k_by_impr = {imp.impr_id: sum(imp.labels) for imp in impressions}

    def selections():
        for impr_id, user_id, scores in tqdm(results):
            k = k_by_impr.get(impr_id, 0)
            ids = candidate_ids[impr_id]
            top_k_idx = np.argsort(scores)[::-1][:k]
            yield user_id, [ids[i] for i in top_k_idx]

    _write_user_article_map(selections(), article_meta, output_file)


def save_user_article_map_from_ranks(
    prediction_file, impressions, article_meta, output_file, positions_by_impr=None
):
    """Build the per-user map straight from a full-rank prediction file.

    prediction_file : a recommender's full-rank output. The rank list is the last
                      bracketed token, so both the model format ("impr_id [ranks]")
                      and the random/popular format ("impr_id user_id [ranks]")
                      parse the same way; the user_id always comes from the
                      impressions, not the file.
    positions_by_impr : optional {impr_id: [orig_index, ...]}. When given, each
                      impression's ranks are sliced to those candidate positions
                      before ranking (the subtopic news subset reusing a model's
                      full-dataset scores); otherwise the full candidate list is
                      used.

    K per impression is the number of clicks it kept. The selection matches the
    old top-k step (np.argsort(ranks)[:k]); only the on-disk intermediate is gone.
    """
    pred_ranks = {}
    with open(prediction_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            pred_ranks[int(parts[0])] = list(map(int, parts[-1][1:-1].split(",")))

    def selections():
        for imp in impressions:
            full_ranks = pred_ranks.get(imp.impr_id, [])
            if positions_by_impr is not None:
                positions = positions_by_impr.get(imp.impr_id, [])
                ranks = [full_ranks[i] for i in positions] if full_ranks else []
            else:
                ranks = full_ranks
            k = sum(imp.labels)
            top_k_idx = np.argsort(ranks)[:k]
            yield imp.user_id, [imp.candidate_ids[i] for i in top_k_idx]

    _write_user_article_map(selections(), article_meta, output_file)


def save_user_article_map_from_ground_truth(gt_results, article_meta, output_file):
    """Build the per-user map straight from extract_ground_truth() records.

    gt_results : [(impr_id, user_id, positions, ids)] — the clicked candidates.
    Lets the ground-truth map be built without writing the intermediate top-k file.
    """
    _write_user_article_map(
        ((user_id, ids) for _, user_id, _, ids in gt_results), article_meta, output_file
    )