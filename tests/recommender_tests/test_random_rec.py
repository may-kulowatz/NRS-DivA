import sys
import os
import logging
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dataset_module.mind_adapter import load_impressions, load_article_meta
from recommender_module.random_rec import random_recommend
from recommender_module.io import save_predictions_topk, save_user_article_map

logger = logging.getLogger(__name__)


def write_behaviors(tmp_path, lines):
    p = tmp_path / "behaviors.tsv"
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def parse_topk_line(line):
    parts = line.strip().split()
    inner_pos = parts[2][1:-1]
    inner_ids = parts[3][1:-1]
    positions = inner_pos.split(",") if inner_pos else []
    ids = inner_ids.split(",") if inner_ids else []
    return int(parts[0]), parts[1], positions, ids


# ---------------------------------------------------------------------------
# load + recommend
# ---------------------------------------------------------------------------

def test_output_shape(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1",
        "2\tU2\t11/15/2019 11:00:00\t\tN4-0 N5-1",
    ])
    results = random_recommend(load_impressions(f))

    assert len(results) == 2
    assert len(results[0][2]) == 3
    assert len(results[1][2]) == 2
    logger.info(
        "Number of impressions and scores per impression match the input behaviors file — "
        "expected 2 impressions with [3, 2] scores, actual %d impressions with [%d, %d] scores",
        len(results), len(results[0][2]), len(results[1][2])
    )


def test_user_ids_preserved(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0",
        "2\tU2\t11/15/2019 11:00:00\t\tN3-1",
    ])
    results = random_recommend(load_impressions(f))

    assert results[0][1] == "U1"
    assert results[1][1] == "U2"
    logger.info(
        "User IDs are passed through from the behaviors file to the recommend output without modification — "
        "expected [U1, U2], actual [%s, %s]",
        results[0][1], results[1][1]
    )


def test_scores_in_range(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1",
        "2\tU2\t11/15/2019 11:00:00\t\tN4-0 N5-1",
    ])
    results = random_recommend(load_impressions(f))

    for _, _, scores in results:
        assert np.all(scores >= 0) and np.all(scores <= 1)
    all_scores = np.concatenate([s for _, _, s in results])
    logger.info(
        "Random scores are bounded within a valid probability range — "
        "expected all scores in [0.0, 1.0], actual min=%.4f, max=%.4f",
        all_scores.min(), all_scores.max()
    )


def test_same_seed_reproducible(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1"])
    impressions = load_impressions(f)

    scores_a = random_recommend(impressions, seed=42)[0][2]
    scores_b = random_recommend(impressions, seed=42)[0][2]

    np.testing.assert_array_equal(scores_a, scores_b)
    logger.info(
        "Using the same random seed produces identical scores across multiple runs — "
        "expected identical arrays, actual run1=%s, run2=%s",
        scores_a, scores_b
    )


def test_different_seeds_differ(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1"])
    impressions = load_impressions(f)

    scores_a = random_recommend(impressions, seed=1)[0][2]
    scores_b = random_recommend(impressions, seed=2)[0][2]

    assert not np.array_equal(scores_a, scores_b)
    logger.info(
        "Different random seeds produce different score arrays — "
        "expected differing scores, actual seed=1: %s, seed=2: %s",
        scores_a, scores_b
    )


# ---------------------------------------------------------------------------
# save_predictions_topk
# ---------------------------------------------------------------------------

def test_topk_list_length_matches_clicks(tmp_path):
    behaviors = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1 N4-0 N5-0",
    ])
    output = str(tmp_path / "out.txt")

    impressions = load_impressions(behaviors)
    results = random_recommend(impressions, seed=0)
    save_predictions_topk(results, impressions, output)

    _, _, positions, ids = parse_topk_line(open(output).readline())
    assert len(positions) == 2
    assert len(ids) == 2
    logger.info(
        "Top-k output contains exactly as many recommendations as there are clicks in the impression — "
        "expected 2 positions and 2 IDs, actual %d positions and %d IDs",
        len(positions), len(ids)
    )


def test_topk_article_ids_are_valid_candidates(tmp_path):
    behaviors = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1 N4-0 N5-0",
    ])
    output = str(tmp_path / "out.txt")

    impressions = load_impressions(behaviors)
    results = random_recommend(impressions, seed=0)
    save_predictions_topk(results, impressions, output)

    _, _, _, ids = parse_topk_line(open(output).readline())
    assert all(aid in {"N1", "N2", "N3", "N4", "N5"} for aid in ids)
    logger.info(
        "Article IDs in top-k output are drawn only from the impression's candidate pool — "
        "expected candidates from {N1, N2, N3, N4, N5}, actual %s",
        ids
    )


def test_topk_positions_within_valid_range(tmp_path):
    behaviors = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-0 N2-1 N3-0 N4-1 N5-0",
    ])
    output = str(tmp_path / "out.txt")

    impressions = load_impressions(behaviors)
    results = random_recommend(impressions, seed=0)
    save_predictions_topk(results, impressions, output)

    _, _, positions, _ = parse_topk_line(open(output).readline())
    assert all(1 <= int(p) <= 5 for p in positions)
    logger.info(
        "Candidate positions in top-k output are valid 1-indexed positions within the impression — "
        "expected all in range [1, 5], actual %s",
        positions
    )


def test_topk_zero_clicks_gives_empty_lists(tmp_path):
    behaviors = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-0 N2-0 N3-0",
    ])
    output = str(tmp_path / "out.txt")

    impressions = load_impressions(behaviors)
    results = random_recommend(impressions, seed=0)
    save_predictions_topk(results, impressions, output)

    _, _, positions, ids = parse_topk_line(open(output).readline())
    assert positions == []
    assert ids == []
    logger.info(
        "Impression with no clicks produces empty recommendation lists — "
        "expected positions=[], IDs=[], actual positions=%s, IDs=%s",
        positions, ids
    )


# ---------------------------------------------------------------------------
# save_user_article_map
# ---------------------------------------------------------------------------

def write_topk(tmp_path, lines):
    p = tmp_path / "topk.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def write_news(tmp_path, rows):
    p = tmp_path / "news.tsv"
    p.write_text("\n".join(f"{nid}\t{cat}\t{sub}" for nid, cat, sub in rows), encoding="utf-8")
    return load_article_meta(str(p))


def parse_user_map_line(line):
    parts = line.strip().split()
    user_id = parts[0]
    ids = parts[1][1:-1].split(",") if parts[1][1:-1] else []
    topics = parts[2][1:-1].split(",") if parts[2][1:-1] else []
    subtopics = parts[3][1:-1].split(",") if parts[3][1:-1] else []
    return user_id, ids, topics, subtopics


def test_user_map_groups_articles_by_user(tmp_path):
    topk = write_topk(tmp_path, [
        "1 U1 [1] [N1]",
        "2 U1 [1] [N2]",
    ])
    meta = write_news(tmp_path, [("N1", "sports", "golf"), ("N2", "finance", "investing")])
    output = str(tmp_path / "out.txt")

    save_user_article_map(topk, meta, output)

    lines = open(output).readlines()
    user_id, ids, _, _ = parse_user_map_line(lines[0])
    assert user_id == "U1"
    assert ids == ["N1", "N2"]
    logger.info(
        "Articles from multiple impressions are grouped under one user entry — "
        "expected U1: [N1, N2], actual %s: %s",
        user_id, ids
    )


def test_user_map_correct_topics(tmp_path):
    topk = write_topk(tmp_path, ["1 U1 [1,2] [N1,N2]"])
    meta = write_news(tmp_path, [("N1", "sports", "golf"), ("N2", "finance", "investing")])
    output = str(tmp_path / "out.txt")

    save_user_article_map(topk, meta, output)

    _, _, topics, _ = parse_user_map_line(open(output).readline())
    assert topics == ["sports", "finance"]
    logger.info(
        "Article topics are correctly looked up from article metadata — "
        "expected ['sports', 'finance'], actual %s",
        topics
    )


def test_user_map_correct_subtopics(tmp_path):
    topk = write_topk(tmp_path, ["1 U1 [1,2] [N1,N2]"])
    meta = write_news(tmp_path, [("N1", "sports", "golf"), ("N2", "finance", "investing")])
    output = str(tmp_path / "out.txt")

    save_user_article_map(topk, meta, output)

    _, _, _, subtopics = parse_user_map_line(open(output).readline())
    assert subtopics == ["golf", "investing"]
    logger.info(
        "Article subcategories are correctly looked up from article metadata — "
        "expected ['golf', 'investing'], actual %s",
        subtopics
    )


def test_user_map_multiple_users_have_separate_entries(tmp_path):
    topk = write_topk(tmp_path, [
        "1 U1 [1] [N1]",
        "2 U2 [1] [N2]",
    ])
    meta = write_news(tmp_path, [("N1", "sports", "golf"), ("N2", "finance", "investing")])
    output = str(tmp_path / "out.txt")

    save_user_article_map(topk, meta, output)

    lines = open(output).readlines()
    user_ids = {parse_user_map_line(l)[0] for l in lines}
    assert user_ids == {"U1", "U2"}
    logger.info(
        "Each user gets a separate line in the output — "
        "expected users {U1, U2}, actual %s",
        user_ids
    )