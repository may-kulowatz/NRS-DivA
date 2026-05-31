import sys
import os
import logging
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from general.popular_rec import load_impressions_mind, popular_recommend, save_predictions_mind_topk

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

def test_first_impression_all_zero(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0",
        "2\tU2\t11/15/2019 11:00:00\t\tN1-0 N2-1",
    ])
    results = {impr_id: scores for impr_id, _, scores in popular_recommend(load_impressions_mind(f))}

    np.testing.assert_array_equal(results[1], [0.0, 0.0])
    logger.info("Earliest impression scores are all zero (no prior clicks): %s", results[1])


def test_clicked_article_scores_higher(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0",
        "2\tU2\t11/15/2019 11:00:00\t\tN1-0 N2-0",
    ])
    results = {impr_id: scores for impr_id, _, scores in popular_recommend(load_impressions_mind(f))}

    assert results[2][0] > results[2][1]
    logger.info("N1 score %.1f > N2 score %.1f after N1 was clicked in impression 1",
                results[2][0], results[2][1])


def test_no_future_click_leak(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-0 N2-0",
        "2\tU2\t11/15/2019 11:00:00\t\tN1-1 N2-0",
    ])
    results = {impr_id: scores for impr_id, _, scores in popular_recommend(load_impressions_mind(f))}

    assert results[1][0] == 0.0
    logger.info("N1 score in impression 1 is %.1f — click from later impression 2 did not leak",
                results[1][0])


def test_counts_accumulate(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0",
        "2\tU2\t11/15/2019 11:00:00\t\tN1-1 N2-0",
        "3\tU3\t11/15/2019 12:00:00\t\tN1-0 N2-0",
    ])
    results = {impr_id: scores for impr_id, _, scores in popular_recommend(load_impressions_mind(f))}

    assert results[3][0] == 2.0
    logger.info("N1 accumulated %.0f clicks by impression 3 (clicked in impressions 1 and 2)",
                results[3][0])


def test_preserves_input_order(tmp_path):
    f = write_behaviors(tmp_path, [
        "2\tU2\t11/15/2019 11:00:00\t\tN1-0",
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1",
    ])
    results = popular_recommend(load_impressions_mind(f))

    assert results[0][0] == 2
    assert results[1][0] == 1
    logger.info("Output order matches file order: impression %d first, impression %d second",
                results[0][0], results[1][0])


def test_user_ids_preserved(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0",
        "2\tU2\t11/15/2019 11:00:00\t\tN1-0",
    ])
    results = popular_recommend(load_impressions_mind(f))

    assert results[0][1] == "U1"
    assert results[1][1] == "U2"
    logger.info("User IDs correctly passed through: %s, %s", results[0][1], results[1][1])


def test_multiple_articles_clicked_ranked_correctly(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-1",
        "2\tU2\t11/15/2019 11:00:00\t\tN1-1 N2-0",
        "3\tU3\t11/15/2019 12:00:00\t\tN1-1 N2-0",
        "4\tU4\t11/15/2019 13:00:00\t\tN1-0 N2-0",
    ])
    results = {impr_id: scores for impr_id, _, scores in popular_recommend(load_impressions_mind(f))}

    assert results[4][0] > results[4][1]
    logger.info("N1 score %.0f > N2 score %.0f in impression 4 (N1 clicked 3×, N2 clicked 1×)",
                results[4][0], results[4][1])


# ---------------------------------------------------------------------------
# save_predictions_mind_topk
# ---------------------------------------------------------------------------

def test_topk_list_length_matches_ground_truth(tmp_path):
    behaviors = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1 N4-0 N5-0",
        "2\tU2\t11/15/2019 11:00:00\t\tN1-1 N2-0 N3-0 N4-1 N5-0",
    ])
    gt = write_ground_truth(tmp_path, [
        "1 U1 [1,3] [N1,N3]",
        "2 U2 [1,4] [N1,N4]",
    ])
    output = str(tmp_path / "out.txt")

    rows = load_impressions_mind(behaviors)
    save_predictions_mind_topk(popular_recommend(rows), behaviors, gt, output)

    for line in open(output):
        _, _, positions, ids = parse_topk_line(line)
        assert len(positions) == 2
        assert len(ids) == 2
    logger.info("Both impressions produced top-k lists of length 2, matching ground truth K=2")


def test_topk_article_ids_are_valid_candidates(tmp_path):
    behaviors = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1",
    ])
    gt = write_ground_truth(tmp_path, ["1 U1 [1,3] [N1,N3]"])
    output = str(tmp_path / "out.txt")

    rows = load_impressions_mind(behaviors)
    save_predictions_mind_topk(popular_recommend(rows), behaviors, gt, output)

    _, _, _, ids = parse_topk_line(open(output).readline())
    assert all(aid in {"N1", "N2", "N3"} for aid in ids)
    logger.info("Recommended article IDs %s are all valid candidates", ids)


def test_topk_zero_clicks_gives_empty_lists(tmp_path):
    behaviors = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-0 N2-0",
    ])
    gt = write_ground_truth(tmp_path, ["1 U1 [] []"])
    output = str(tmp_path / "out.txt")

    rows = load_impressions_mind(behaviors)
    save_predictions_mind_topk(popular_recommend(rows), behaviors, gt, output)

    _, _, positions, ids = parse_topk_line(open(output).readline())
    assert positions == []
    assert ids == []
    logger.info("Impression with 0 clicks produces empty position and ID lists")
