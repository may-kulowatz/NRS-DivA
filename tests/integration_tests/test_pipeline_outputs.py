"""
Integration tests for the generated prediction and user-article map files.
All tests read the real files in data/MIND/predictions/ — no fake data.
Tests that require file content skip automatically if the files are absent.
"""

import os
import logging
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRED_DIR     = os.path.join(PROJECT_ROOT, "data", "MIND", "predictions")

PREDICTION_FILES = {
    "ground_truth": os.path.join(PRED_DIR, "prediction_ground_truth.txt"),
    "random":       os.path.join(PRED_DIR, "prediction_random.txt"),
    "popular":      os.path.join(PRED_DIR, "prediction_popular.txt"),
    "nrms":         os.path.join(PRED_DIR, "prediction_nrms.txt"),
}
TOPK_FILES = {
    "ground_truth": os.path.join(PRED_DIR, "prediction_ground_truth.txt"),
    "random_topk":  os.path.join(PRED_DIR, "prediction_random_topk.txt"),
    "popular_topk": os.path.join(PRED_DIR, "prediction_popular_topk.txt"),
    "nrms_topk":    os.path.join(PRED_DIR, "prediction_nrms_topk.txt"),
}
USER_ARTICLE_FILES = {
    "ground_truth": os.path.join(PRED_DIR, "user_articles_ground_truth.txt"),
    "random":       os.path.join(PRED_DIR, "user_articles_random.txt"),
    "popular":      os.path.join(PRED_DIR, "user_articles_popular.txt"),
    "nrms":         os.path.join(PRED_DIR, "user_articles_nrms.txt"),
}
ALL_FILES = {**PREDICTION_FILES, **TOPK_FILES, **USER_ARTICLE_FILES}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def parse_user_articles(path):
    """Return {user_id: (ids, topics, subtopics)} from a user_articles file."""
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


def skip_if_missing(*paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        names = [os.path.basename(p) for p in missing]
        pytest.skip(f"File(s) not found: {', '.join(names)} — run pipeline.py first")


# ---------------------------------------------------------------------------
# Step 1 — File existence (fails hard if missing, does not skip)
# ---------------------------------------------------------------------------

def test_all_prediction_files_exist():
    missing = [name for name, path in ALL_FILES.items() if not os.path.exists(path)]
    assert missing == [], (
        f"Missing prediction files: {missing}. Run pipeline.py to generate them."
    )
    logger.info(
        "All prediction files exist — expected all present, actual missing=%s", missing
    )


# ---------------------------------------------------------------------------
# Step 2 — Line counts
# ---------------------------------------------------------------------------

def test_prediction_files_have_same_line_count():
    skip_if_missing(*PREDICTION_FILES.values())
    counts = {name: count_lines(path) for name, path in PREDICTION_FILES.items()}
    unique = set(counts.values())
    assert len(unique) == 1, (
        f"Prediction files have different line counts: {counts}"
    )
    logger.info(
        "All prediction files have the same line count — expected identical, actual %s",
        counts
    )


def test_topk_files_have_same_line_count():
    skip_if_missing(*TOPK_FILES.values())
    counts = {name: count_lines(path) for name, path in TOPK_FILES.items()}
    unique = set(counts.values())
    assert len(unique) == 1, (
        f"Top-k files have different line counts: {counts}"
    )
    logger.info(
        "All top-k files have the same line count — expected identical, actual %s",
        counts
    )


def test_user_article_files_have_same_line_count():
    skip_if_missing(*USER_ARTICLE_FILES.values())
    counts = {name: count_lines(path) for name, path in USER_ARTICLE_FILES.items()}
    unique = set(counts.values())
    assert len(unique) == 1, (
        f"User article files have different line counts (user counts): {counts}"
    )
    logger.info(
        "All user article files have the same number of users — expected identical, actual %s",
        counts
    )


# ---------------------------------------------------------------------------
# Step 3 — User IDs
# ---------------------------------------------------------------------------

def test_user_articles_same_user_ids():
    skip_if_missing(*USER_ARTICLE_FILES.values())
    user_id_sets = {name: set(parse_user_articles(path)) for name, path in USER_ARTICLE_FILES.items()}
    reference = user_id_sets["ground_truth"]
    for name, uid_set in user_id_sets.items():
        assert uid_set == reference, (
            f"User IDs in '{name}' differ from ground truth: "
            f"only in {name}={uid_set - reference}, only in gt={reference - uid_set}"
        )
    logger.info(
        "All user article files contain the same user IDs — expected %d users, all match",
        len(reference)
    )


# ---------------------------------------------------------------------------
# Step 4 — Article counts per user match ground truth
# ---------------------------------------------------------------------------

def test_article_count_per_user_matches_ground_truth():
    skip_if_missing(*USER_ARTICLE_FILES.values())
    gt_map = parse_user_articles(USER_ARTICLE_FILES["ground_truth"])
    for name, path in USER_ARTICLE_FILES.items():
        if name == "ground_truth":
            continue
        rec_map = parse_user_articles(path)
        mismatches = {
            uid: (len(gt_map[uid][0]), len(rec_map[uid][0]))
            for uid in gt_map
            if len(rec_map[uid][0]) != len(gt_map[uid][0])
        }
        assert mismatches == {}, (
            f"Article counts in '{name}' differ from ground truth for users: {mismatches}"
        )
        logger.info(
            "Article count per user matches ground truth in '%s' — "
            "expected 0 mismatches, actual %d mismatches",
            name, len(mismatches)
        )


# ---------------------------------------------------------------------------
# Step 5 — Internal consistency within user article files
# ---------------------------------------------------------------------------

def test_user_articles_ids_topics_subtopics_same_length():
    skip_if_missing(*USER_ARTICLE_FILES.values())
    for name, path in USER_ARTICLE_FILES.items():
        for user_id, (ids, topics, subtopics) in parse_user_articles(path).items():
            assert len(ids) == len(topics) == len(subtopics), (
                f"[{name}] user {user_id}: ids={len(ids)}, topics={len(topics)}, subtopics={len(subtopics)}"
            )
    logger.info(
        "IDs, topics, and subtopics lists are equal length for every user in all files — "
        "expected 0 inconsistencies, actual 0"
    )


# ---------------------------------------------------------------------------
# Step 6 — NRMS-specific: topk file must have user IDs (raw file does not)
# ---------------------------------------------------------------------------

def test_nrms_topk_has_user_ids():
    skip_if_missing(TOPK_FILES["nrms_topk"])
    with open(TOPK_FILES["nrms_topk"], encoding="utf-8") as f:
        first_line = f.readline().strip()
    parts = first_line.split()
    # topk format: impr_id user_id [positions] [ids] — at least 4 tokens
    assert len(parts) >= 4, (
        f"nrms_topk first line has only {len(parts)} token(s); expected impr_id user_id [pos] [ids]"
    )
    logger.info(
        "NRMS topk file contains user IDs — "
        "expected ≥4 tokens per line, actual first line has %d tokens: %s",
        len(parts), parts[:4]
    )