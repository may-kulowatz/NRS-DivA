import os
import sys

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


def run_pipeline(mind_dir, seed=42):
    behaviors_file      = os.path.join(mind_dir, "MINDsmall_dev", "behaviors.tsv")
    news_file           = os.path.join(mind_dir, "MINDsmall_dev", "news.tsv")

    gt_file             = os.path.join(mind_dir, "prediction_ground_truth.txt")
    random_file         = os.path.join(mind_dir, "prediction_random.txt")
    random_topk_file    = os.path.join(mind_dir, "prediction_random_topk.txt")
    popular_file        = os.path.join(mind_dir, "prediction_popular.txt")
    popular_topk_file   = os.path.join(mind_dir, "prediction_popular_topk.txt")
    user_articles_gt     = os.path.join(mind_dir, "user_articles_ground_truth.txt")
    user_articles_random = os.path.join(mind_dir, "user_articles_random.txt")
    user_articles_popular = os.path.join(mind_dir, "user_articles_popular.txt")

    # -------------------------------------------------------------------------
    # Step 1 — Ground truth
    # Depends on: behaviors.tsv
    # -------------------------------------------------------------------------
    print("Step 1/5 — Generating ground truth...")
    gt_results = load_ground_truth_mind(behaviors_file)
    save_ground_truth_mind(gt_results, gt_file)

    # -------------------------------------------------------------------------
    # Step 2 — Random recommendations
    # Depends on: behaviors.tsv, gt_file (for topk K values)
    # -------------------------------------------------------------------------
    print("Step 2/5 — Generating random recommendations...")
    impressions = load_impressions_mind(behaviors_file)
    random_results = random_recommend(impressions, seed=seed)
    save_predictions_mind(random_results, random_file)
    save_predictions_mind_topk(random_results, behaviors_file, gt_file, random_topk_file)

    # -------------------------------------------------------------------------
    # Step 3 — Popular recommendations
    # Depends on: behaviors.tsv, gt_file (for topk K values)
    # -------------------------------------------------------------------------
    print("Step 3/5 — Generating popular recommendations...")
    rows = popular_load_impressions(behaviors_file)
    popular_results = popular_recommend(rows)
    popular_save_predictions(popular_results, popular_file)
    popular_save_topk(popular_results, behaviors_file, gt_file, popular_topk_file)

    # -------------------------------------------------------------------------
    # Step 4 — User article maps
    # Depends on: gt_file, random_topk_file, popular_topk_file, news.tsv
    # -------------------------------------------------------------------------
    print("Step 4/5 — Building user article maps...")
    gt_save_map(gt_file, news_file, user_articles_gt)
    random_save_map(random_topk_file, news_file, user_articles_random)
    popular_save_map(popular_topk_file, news_file, user_articles_popular)

    # -------------------------------------------------------------------------
    # Step 5 — Diversity scores
    # Depends on: user article maps
    # -------------------------------------------------------------------------
    print("Step 5/5 — Calculating diversity scores...")
    scores = {}
    for name, path in [
        ("random",       user_articles_random),
        ("popular",      user_articles_popular),
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