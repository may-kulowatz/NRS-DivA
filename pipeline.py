import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommenders.ground_truth import (
    load_ground_truth_mind,
    save_ground_truth_mind,
    save_user_article_map as gt_save_map,
)
from recommenders.random_rec import (
    load_impressions_mind,
    random_recommend,
    save_predictions_mind,
    save_predictions_mind_topk,
    save_user_article_map as random_save_map,
)
from recommenders.popular_rec import (
    load_impressions_mind as popular_load_impressions,
    popular_recommend,
    save_predictions_mind as popular_save_predictions,
    save_predictions_mind_topk as popular_save_topk,
    save_user_article_map as popular_save_map,
)
from diversityScores.topic_diversity import topic_diversity, subtopic_diversity


def _nrms_topk(nrms_file, behaviors_file, ground_truth_file, output_topk):
    """Convert prediction_nrms.txt (impr_id [ranks]) to a topk file
    (impr_id user_id [positions] [ids]) using K from the ground truth.
    The nrms file has no user_id column, so user_ids are loaded from behaviors.tsv.
    """
    # impr_id → list of per-candidate ranks (rank 1 = best)
    nrms_ranks = {}
    with open(nrms_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            impr_id = int(parts[0])
            nrms_ranks[impr_id] = list(map(int, parts[1][1:-1].split(",")))

    user_ids = {}
    article_ids = {}
    impression_order = []
    with open(behaviors_file, encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("\t")
            impr_id = int(cols[0])
            user_ids[impr_id] = cols[1]
            article_ids[impr_id] = [c.split("-")[0] for c in cols[4].split()]
            impression_order.append(impr_id)

    k_by_impr = {}
    with open(ground_truth_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            impr_id = int(parts[0])
            inner = parts[2][1:-1]
            k_by_impr[impr_id] = len(inner.split(",")) if inner else 0

    with open(output_topk, "w", encoding="utf-8") as f:
        for impr_id in impression_order:
            ranks = nrms_ranks.get(impr_id, [])
            k = k_by_impr.get(impr_id, 0)
            ids = article_ids.get(impr_id, [])
            user_id = user_ids.get(impr_id, "unknown")
            top_k_idx = np.argsort(ranks)[:k]
            positions = [i + 1 for i in top_k_idx]
            chosen_ids = [ids[i] for i in top_k_idx]
            f.write(f"{impr_id} {user_id} [" + ",".join(map(str, positions)) + "] [" + ",".join(chosen_ids) + "]\n")


def _exists(*paths):
    return all(os.path.exists(p) for p in paths)


def run_pipeline(mind_dir, seed=42):
    behaviors_file      = os.path.join(mind_dir, "MINDsmall_dev", "behaviors.tsv")
    news_file           = os.path.join(mind_dir, "MINDsmall_dev", "news.tsv")

    pred_dir            = os.path.join(mind_dir, "predictions")

    gt_file             = os.path.join(pred_dir, "prediction_ground_truth.txt")
    random_file         = os.path.join(pred_dir, "prediction_random.txt")
    random_topk_file    = os.path.join(pred_dir, "prediction_random_topk.txt")
    popular_file        = os.path.join(pred_dir, "prediction_popular.txt")
    popular_topk_file   = os.path.join(pred_dir, "prediction_popular_topk.txt")
    nrms_file           = os.path.join(pred_dir, "prediction_nrms.txt")
    nrms_topk_file      = os.path.join(pred_dir, "prediction_nrms_topk.txt")
    user_articles_gt     = os.path.join(pred_dir, "user_articles_ground_truth.txt")
    user_articles_random = os.path.join(pred_dir, "user_articles_random.txt")
    user_articles_popular = os.path.join(pred_dir, "user_articles_popular.txt")
    user_articles_nrms   = os.path.join(pred_dir, "user_articles_nrms.txt")

    # -------------------------------------------------------------------------
    # Step 0 — Check which files already exist
    # -------------------------------------------------------------------------
    skip_gt      = _exists(gt_file)
    skip_random  = _exists(random_file, random_topk_file)
    skip_nrms    = _exists(nrms_topk_file)
    skip_popular = _exists(popular_file, popular_topk_file)
    skip_maps    = _exists(user_articles_gt, user_articles_random,
                           user_articles_popular, user_articles_nrms)

    print("Step 0/6 — Checking existing files...")
    for label, skip in [
        ("Ground truth",          skip_gt),
        ("Random predictions",    skip_random),
        ("NRMS topk",             skip_nrms),
        ("Popular predictions",   skip_popular),
        ("User article maps",     skip_maps),
    ]:
        print(f"  {'SKIP' if skip else 'RUN '} — {label}")

    # -------------------------------------------------------------------------
    # Step 1 — Ground truth
    # Depends on: behaviors.tsv
    # -------------------------------------------------------------------------
    if skip_gt:
        print("Step 1/6 — Ground truth already exists, skipping.")
    else:
        print("Step 1/6 — Generating ground truth...")
        gt_results = load_ground_truth_mind(behaviors_file)
        save_ground_truth_mind(gt_results, gt_file)

    # -------------------------------------------------------------------------
    # Step 2 — Random recommendations
    # Depends on: behaviors.tsv, gt_file (for topk K values)
    # -------------------------------------------------------------------------
    if skip_random:
        print("Step 2/6 — Random predictions already exist, skipping.")
    else:
        print("Step 2/6 — Generating random recommendations...")
        impressions = load_impressions_mind(behaviors_file)
        random_results = random_recommend(impressions, seed=seed)
        save_predictions_mind(random_results, random_file)
        save_predictions_mind_topk(random_results, behaviors_file, gt_file, random_topk_file)

    # -------------------------------------------------------------------------
    # Step 3 — NRMS topk
    # Depends on: prediction_nrms.txt, behaviors.tsv, gt_file
    # -------------------------------------------------------------------------
    if skip_nrms:
        print("Step 3/6 — NRMS topk already exists, skipping.")
    else:
        print("Step 3/6 — Converting NRMS predictions to top-k format...")
        _nrms_topk(nrms_file, behaviors_file, gt_file, nrms_topk_file)

    # -------------------------------------------------------------------------
    # Step 4 — Popular recommendations
    # Depends on: behaviors.tsv, gt_file (for topk K values)
    # -------------------------------------------------------------------------
    if skip_popular:
        print("Step 4/6 — Popular predictions already exist, skipping.")
    else:
        print("Step 4/6 — Generating popular recommendations...")
        rows = popular_load_impressions(behaviors_file)
        popular_results = popular_recommend(rows)
        popular_save_predictions(popular_results, popular_file)
        popular_save_topk(popular_results, behaviors_file, gt_file, popular_topk_file)

    # -------------------------------------------------------------------------
    # Step 5 — User article maps
    # Depends on: gt_file, random_topk_file, popular_topk_file, nrms_topk_file, news.tsv
    # -------------------------------------------------------------------------
    if skip_maps:
        print("Step 5/6 — User article maps already exist, skipping.")
    else:
        print("Step 5/6 — Building user article maps...")
        gt_save_map(gt_file, news_file, user_articles_gt)
        random_save_map(random_topk_file, news_file, user_articles_random)
        popular_save_map(popular_topk_file, news_file, user_articles_popular)
        random_save_map(nrms_topk_file, news_file, user_articles_nrms)

    # -------------------------------------------------------------------------
    # Step 6 — Diversity scores
    # Depends on: user article maps (always runs — no file output)
    # -------------------------------------------------------------------------
    print("Step 6/6 — Calculating diversity scores...")
    scores = {}
    for name, path in [
        ("random",       user_articles_random),
        ("popular",      user_articles_popular),
        ("nrms",         user_articles_nrms),
        ("ground_truth", user_articles_gt),
    ]:
        scores[name] = {
            "topic_diversity":           topic_diversity(path),
            "subtopic_diversity_news":   subtopic_diversity(path, category="news"),
        }

    print("\n=== Diversity Scores ===")
    for name, s in scores.items():
        print(f"\n  {name}:")
        print(f"    Topic diversity:           {s['topic_diversity']:.4f}")
        print(f"    Subtopic diversity (news): {s['subtopic_diversity_news']:.4f}")

    return scores


if __name__ == "__main__":
    _project_dir = os.path.dirname(os.path.abspath(__file__))
    _mind_dir = os.path.join(_project_dir, "data", "MIND")
    run_pipeline(_mind_dir)