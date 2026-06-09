"""Shared, dataset-agnostic writers for prediction and user-article files.

Every recommender (random, popular, NRMS, ground truth) produces the same kind
of output, so the writing logic lives here once instead of being copy-pasted
into each recommender module. These functions operate purely on the normalized
in-memory structures (Impression records, scored results, {id: (topic, subtopic)}
metadata) and never touch a dataset-specific file format.

Output line formats
-------------------
  predictions      : "{impr_id} {user_id} [rank,rank,...]"          (1 = best)
  top-k / ground   : "{impr_id} {user_id} [pos,pos,...] [id,id,...]"
  user-article map : "{user_id} [ids] [topics] [subtopics]"
"""

from collections import defaultdict

import numpy as np
from tqdm import tqdm


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

    K for each impression is the number of clicks it actually received
    (sum of its labels), so every recommender is asked for exactly as many
    items as the user clicked — making the outputs directly comparable to the
    ground truth. Candidate ids come from the impressions themselves, so no
    dataset file needs re-reading.
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


def save_user_article_map(topk_file, article_meta, output_file):
    """Aggregate a top-k file into per-user article/topic/subtopic lists.

    topk_file    : a "{impr_id} {user_id} [pos] [ids]" file (top-k or ground truth)
    article_meta : {article_id: (topic, subtopic)} from a dataset adapter
    """
    user_articles = defaultdict(list)
    with open(topk_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            user_id = parts[1]
            inner_ids = parts[3][1:-1]
            if inner_ids:
                user_articles[user_id].extend(inner_ids.split(","))

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