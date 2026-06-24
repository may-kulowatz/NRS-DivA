"""
Cross-file validation tests.

These tests generate all three pipelines (random, popular, ground truth) from
the same small dataset, then check consistency across the resulting
prediction_processed_xxx.txt files.
"""

import sys
import os
import logging
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dataset_module.mind_adapter import load_impressions, load_article_meta
from recommender_module.common.random_rec import random_recommend
from recommender_module.common.popular_rec import popular_recommend
from recommender_module.common.ground_truth import extract_ground_truth, save_ground_truth
from recommender_module.common.io import save_predictions_topk, save_user_article_map

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared dataset
# ---------------------------------------------------------------------------

BEHAVIORS = [
    "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1",
    "2\tU2\t11/15/2019 11:00:00\t\tN4-1 N5-0",
    "3\tU1\t11/15/2019 12:00:00\t\tN4-0 N6-1",
    "4\tU3\t11/15/2019 13:00:00\t\tN7-1 N8-1 N9-1",
]

NEWS = [
    "N1\tsports\tgolf",
    "N2\tfinance\tinvesting",
    "N3\tsports\ttennis",
    "N4\tnews\tpolitics",
    "N5\tlifestyle\ttravel",
    "N6\tfinance\tstocks",
    "N7\tsports\tfootball",
    "N8\tnews\tworld",
    "N9\tlifestyle\t\tN/A",
]

# ---------------------------------------------------------------------------
# Module-scoped fixture — generate all files once for all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generated_files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cross")

    behaviors_file  = str(tmp / "behaviors.tsv")
    news_file       = str(tmp / "news.tsv")
    gt_file         = str(tmp / "ground_truth.txt")
    random_topk     = str(tmp / "random_topk.txt")
    popular_topk    = str(tmp / "popular_topk.txt")
    random_map      = str(tmp / "prediction_processed_random.txt")
    popular_map     = str(tmp / "prediction_processed_popular.txt")
    gt_map          = str(tmp / "processed_ground_truth.txt")

    (tmp / "behaviors.tsv").write_text("\n".join(BEHAVIORS), encoding="utf-8")
    (tmp / "news.tsv").write_text("\n".join(NEWS), encoding="utf-8")

    impressions = load_impressions(behaviors_file)
    article_meta = load_article_meta(news_file)

    # Ground truth
    save_ground_truth(extract_ground_truth(impressions), gt_file)
    save_user_article_map(gt_file, article_meta, gt_map)

    # Random
    random_results = random_recommend(impressions, seed=42)
    save_predictions_topk(random_results, impressions, random_topk)
    save_user_article_map(random_topk, article_meta, random_map)

    # Popular
    popular_results = popular_recommend(impressions)
    save_predictions_topk(popular_results, impressions, popular_topk)
    save_user_article_map(popular_topk, article_meta, popular_map)

    return {
        "random_map":  random_map,
        "popular_map": popular_map,
        "gt_map":      gt_map,
        "news_file":   news_file,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_map_file(path):
    """Return {user_id: (ids, topics, subtopics)} for every line in a map file."""
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            user_id   = parts[0]
            ids       = parts[1][1:-1].split(",") if parts[1][1:-1] else []
            topics    = parts[2][1:-1].split(",") if parts[2][1:-1] else []
            subtopics = parts[3][1:-1].split(",") if parts[3][1:-1] else []
            result[user_id] = (ids, topics, subtopics)
    return result


def load_news_ids(news_file):
    ids = set()
    with open(news_file, encoding="utf-8") as f:
        for line in f:
            ids.add(line.strip().split("\t")[0])
    return ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_same_user_ids_across_all_files(generated_files):
    random_users  = set(parse_map_file(generated_files["random_map"]))
    popular_users = set(parse_map_file(generated_files["popular_map"]))
    gt_users      = set(parse_map_file(generated_files["gt_map"]))

    assert random_users == gt_users
    assert popular_users == gt_users
    logger.info(
        "User IDs are identical across all three files — "
        "expected %s, actual random=%s, popular=%s, ground_truth=%s",
        gt_users, random_users, popular_users, gt_users
    )


def test_article_count_per_user_matches_ground_truth(generated_files):
    random_map  = parse_map_file(generated_files["random_map"])
    popular_map = parse_map_file(generated_files["popular_map"])
    gt_map      = parse_map_file(generated_files["gt_map"])

    for user_id, (gt_ids, _, _) in gt_map.items():
        random_count  = len(random_map[user_id][0])
        popular_count = len(popular_map[user_id][0])
        expected      = len(gt_ids)
        assert random_count == expected
        assert popular_count == expected
        logger.info(
            "Article count for %s matches ground truth — "
            "expected %d, actual random=%d, popular=%d",
            user_id, expected, random_count, popular_count
        )


def test_ids_topics_subtopics_same_length_in_all_files(generated_files):
    for label, path in [
        ("random",       generated_files["random_map"]),
        ("popular",      generated_files["popular_map"]),
        ("ground_truth", generated_files["gt_map"]),
    ]:
        for user_id, (ids, topics, subtopics) in parse_map_file(path).items():
            assert len(ids) == len(topics) == len(subtopics)
            logger.info(
                "%s — %s: article IDs, topics, and subtopics all have the same length — "
                "expected equal lengths, actual ids=%d, topics=%d, subtopics=%d",
                label, user_id, len(ids), len(topics), len(subtopics)
            )


def test_all_article_ids_exist_in_news(generated_files):
    valid_ids = load_news_ids(generated_files["news_file"])

    for label, path in [
        ("random",       generated_files["random_map"]),
        ("popular",      generated_files["popular_map"]),
        ("ground_truth", generated_files["gt_map"]),
    ]:
        for user_id, (ids, _, _) in parse_map_file(path).items():
            unknown = [a for a in ids if a not in valid_ids]
            assert unknown == []
            logger.info(
                "%s — %s: all article IDs exist in news.tsv — "
                "expected no unknown IDs, actual unknown=%s",
                label, user_id, unknown
            )


def test_no_empty_topics_or_subtopics(generated_files):
    for label, path in [
        ("random",       generated_files["random_map"]),
        ("popular",      generated_files["popular_map"]),
        ("ground_truth", generated_files["gt_map"]),
    ]:
        for user_id, (_, topics, subtopics) in parse_map_file(path).items():
            assert all(t != "" for t in topics)
            assert all(s != "" for s in subtopics)
            logger.info(
                "%s — %s: no empty topic or subtopic strings — "
                "expected all non-empty, actual topics=%s, subtopics=%s",
                label, user_id, topics, subtopics
            )


def test_missing_subcategory_written_as_none(generated_files):
    # N9 has an empty subcategory in news.tsv; wherever it appears it must show "none".
    for label, path in [
        ("random",       generated_files["random_map"]),
        ("popular",      generated_files["popular_map"]),
        ("ground_truth", generated_files["gt_map"]),
    ]:
        for user_id, (ids, _, subtopics) in parse_map_file(path).items():
            for article_id, subtopic in zip(ids, subtopics):
                if article_id == "N9":
                    assert subtopic == "none"
                    logger.info(
                        "%s — %s: N9 has empty subcategory in news.tsv — "
                        "expected 'none', actual '%s'",
                        label, user_id, subtopic
                    )


def test_users_with_no_clicks_absent_from_all_files(generated_files):
    # All impressions in the test dataset have at least one click, so every
    # user that appears in behaviors must appear in all three map files.
    random_users  = set(parse_map_file(generated_files["random_map"]))
    popular_users = set(parse_map_file(generated_files["popular_map"]))
    gt_users      = set(parse_map_file(generated_files["gt_map"]))

    expected_users = {"U1", "U2", "U3"}
    assert gt_users == expected_users
    assert random_users == expected_users
    assert popular_users == expected_users
    logger.info(
        "All users with clicks appear in every file — "
        "expected %s, actual random=%s, popular=%s, ground_truth=%s",
        expected_users, random_users, popular_users, gt_users
    )