import sys
import os
import logging
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from general.random_rec import load_impressions_mind, random_recommend, save_predictions_mind_topk

logger = logging.getLogger(__name__)


def write_behaviors(tmp_path, lines):
    p = tmp_path / "behaviors.tsv"
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def write_ground_truth(tmp_path, lines):
    p = tmp_path / "ground_truth.txt"
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
    results = random_recommend(load_impressions_mind(f))

    assert len(results) == 2
    assert len(results[0][2]) == 3
    assert len(results[1][2]) == 2
    logger.info("Output contains %d impressions with %d and %d scores respectively",
                len(results), len(results[0][2]), len(results[1][2]))


def test_user_ids_preserved(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0",
        "2\tU2\t11/15/2019 11:00:00\t\tN3-1",
    ])
    results = random_recommend(load_impressions_mind(f))

    assert results[0][1] == "U1"
    assert results[1][1] == "U2"
    logger.info("User IDs correctly passed through: %s, %s", results[0][1], results[1][1])


def test_scores_in_range(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1",
        "2\tU2\t11/15/2019 11:00:00\t\tN4-0 N5-1",
    ])
    results = random_recommend(load_impressions_mind(f))

    for _, _, scores in results:
        assert np.all(scores >= 0) and np.all(scores <= 1)
    logger.info("All scores are within [0, 1] across %d impressions", len(results))


def test_same_seed_reproducible(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1"])
    impressions = load_impressions_mind(f)

    scores_a = random_recommend(impressions, seed=42)[0][2]
    scores_b = random_recommend(impressions, seed=42)[0][2]

    np.testing.assert_array_equal(scores_a, scores_b)
    logger.info("Scores with seed=42 are identical across two runs: %s", scores_a)


def test_different_seeds_differ(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1"])
    impressions = load_impressions_mind(f)

    scores_a = random_recommend(impressions, seed=1)[0][2]
    scores_b = random_recommend(impressions, seed=2)[0][2]

    assert not np.array_equal(scores_a, scores_b)
    logger.info("Scores differ between seed=1 %s and seed=2 %s", scores_a, scores_b)


# ---------------------------------------------------------------------------
# save_predictions_mind_topk
# ---------------------------------------------------------------------------

def test_topk_list_length_matches_ground_truth(tmp_path):
    behaviors = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1 N4-0 N5-0",
    ])
    gt = write_ground_truth(tmp_path, ["1 U1 [1,3] [N1,N3]"])
    output = str(tmp_path / "out.txt")

    results = random_recommend(load_impressions_mind(behaviors), seed=0)
    save_predictions_mind_topk(results, behaviors, gt, output)

    _, _, positions, ids = parse_topk_line(open(output).readline())
    assert len(positions) == 2
    assert len(ids) == 2
    logger.info("Top-k output has %d positions and %d article IDs, matching ground truth K=2",
                len(positions), len(ids))


def test_topk_article_ids_are_valid_candidates(tmp_path):
    behaviors = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1 N4-0 N5-0",
    ])
    gt = write_ground_truth(tmp_path, ["1 U1 [1,3] [N1,N3]"])
    output = str(tmp_path / "out.txt")

    results = random_recommend(load_impressions_mind(behaviors), seed=0)
    save_predictions_mind_topk(results, behaviors, gt, output)

    _, _, _, ids = parse_topk_line(open(output).readline())
    assert all(aid in {"N1", "N2", "N3", "N4", "N5"} for aid in ids)
    logger.info("Recommended article IDs %s are all valid candidates", ids)


def test_topk_positions_within_valid_range(tmp_path):
    behaviors = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-0 N2-1 N3-0 N4-1 N5-0",
    ])
    gt = write_ground_truth(tmp_path, ["1 U1 [2,4] [N2,N4]"])
    output = str(tmp_path / "out.txt")

    results = random_recommend(load_impressions_mind(behaviors), seed=0)
    save_predictions_mind_topk(results, behaviors, gt, output)

    _, _, positions, _ = parse_topk_line(open(output).readline())
    assert all(1 <= int(p) <= 5 for p in positions)
    logger.info("All positions %s are within valid range [1, 5]", positions)


def test_topk_zero_clicks_gives_empty_lists(tmp_path):
    behaviors = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-0 N2-0 N3-0",
    ])
    gt = write_ground_truth(tmp_path, ["1 U1 [] []"])
    output = str(tmp_path / "out.txt")

    results = random_recommend(load_impressions_mind(behaviors), seed=0)
    save_predictions_mind_topk(results, behaviors, gt, output)

    _, _, positions, ids = parse_topk_line(open(output).readline())
    assert positions == []
    assert ids == []
    logger.info("Impression with 0 clicks produces empty position and ID lists")
