import sys
import os
import logging
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from general.popular_rec import load_impressions_mind, popular_recommend, save_predictions_mind_topk, save_user_article_map

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
    logger.info(
        "The chronologically first impression receives zero scores since no clicks have been observed yet — "
        "expected [0.0, 0.0], actual %s",
        list(results[1])
    )


def test_clicked_article_scores_higher(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0",
        "2\tU2\t11/15/2019 11:00:00\t\tN1-0 N2-0",
    ])
    results = {impr_id: scores for impr_id, _, scores in popular_recommend(load_impressions_mind(f))}

    assert results[2][0] > results[2][1]
    logger.info(
        "An article clicked in a prior impression scores higher than an unclicked article — "
        "expected N1 > N2, actual N1=%.1f, N2=%.1f",
        results[2][0], results[2][1]
    )


def test_no_future_click_leak(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-0 N2-0",
        "2\tU2\t11/15/2019 11:00:00\t\tN1-1 N2-0",
    ])
    results = {impr_id: scores for impr_id, _, scores in popular_recommend(load_impressions_mind(f))}

    assert results[1][0] == 0.0
    logger.info(
        "Clicks from later impressions do not influence scores of earlier impressions — "
        "expected N1 score=0.0 in impression 1, actual %.1f",
        results[1][0]
    )


def test_counts_accumulate(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0",
        "2\tU2\t11/15/2019 11:00:00\t\tN1-1 N2-0",
        "3\tU3\t11/15/2019 12:00:00\t\tN1-0 N2-0",
    ])
    results = {impr_id: scores for impr_id, _, scores in popular_recommend(load_impressions_mind(f))}

    assert results[3][0] == 2.0
    logger.info(
        "Click counts accumulate correctly across multiple prior impressions — "
        "expected N1 score=2.0 in impression 3 (clicked in impressions 1 and 2), actual %.1f",
        results[3][0]
    )


def test_preserves_input_order(tmp_path):
    f = write_behaviors(tmp_path, [
        "2\tU2\t11/15/2019 11:00:00\t\tN1-0",
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1",
    ])
    results = popular_recommend(load_impressions_mind(f))

    assert results[0][0] == 2
    assert results[1][0] == 1
    logger.info(
        "Results are returned in the original file order, not sorted by timestamp — "
        "expected [2, 1], actual [%d, %d]",
        results[0][0], results[1][0]
    )


def test_user_ids_preserved(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0",
        "2\tU2\t11/15/2019 11:00:00\t\tN1-0",
    ])
    results = popular_recommend(load_impressions_mind(f))

    assert results[0][1] == "U1"
    assert results[1][1] == "U2"
    logger.info(
        "User IDs are passed through from loader to recommender output without modification — "
        "expected [U1, U2], actual [%s, %s]",
        results[0][1], results[1][1]
    )


def test_multiple_articles_clicked_ranked_correctly(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-1",
        "2\tU2\t11/15/2019 11:00:00\t\tN1-1 N2-0",
        "3\tU3\t11/15/2019 12:00:00\t\tN1-1 N2-0",
        "4\tU4\t11/15/2019 13:00:00\t\tN1-0 N2-0",
    ])
    results = {impr_id: scores for impr_id, _, scores in popular_recommend(load_impressions_mind(f))}

    assert results[4][0] > results[4][1]
    logger.info(
        "Article with more historical clicks scores higher than one with fewer clicks — "
        "expected N1 (3 clicks) > N2 (1 click), actual N1=%.0f, N2=%.0f",
        results[4][0], results[4][1]
    )


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
    logger.info(
        "Top-k output contains exactly as many recommendations as there are clicks in the ground truth — "
        "expected 2 positions and 2 IDs per impression, actual matches for all impressions"
    )


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
    logger.info(
        "Article IDs in top-k output are drawn only from the impression's candidate pool — "
        "expected candidates from {N1, N2, N3}, actual %s",
        ids
    )


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
    logger.info(
        "Impression with no clicks in ground truth produces empty recommendation lists — "
        "expected positions=[], IDs=[], actual positions=%s, IDs=%s",
        positions, ids
    )


# ---------------------------------------------------------------------------
# save_user_article_map
# ---------------------------------------------------------------------------

def write_news(tmp_path, rows):
    p = tmp_path / "news.tsv"
    p.write_text("\n".join(f"{nid}\t{cat}\t{sub}" for nid, cat, sub in rows), encoding="utf-8")
    return str(p)


def parse_user_map_line(line):
    parts = line.strip().split()
    user_id = parts[0]
    ids = parts[1][1:-1].split(",") if parts[1][1:-1] else []
    topics = parts[2][1:-1].split(",") if parts[2][1:-1] else []
    subtopics = parts[3][1:-1].split(",") if parts[3][1:-1] else []
    return user_id, ids, topics, subtopics


def test_user_map_groups_articles_by_user(tmp_path):
    # U1 appears in two impressions; both sets of articles should be combined.
    topk = write_ground_truth(tmp_path, [
        "1 U1 [1] [N1]",
        "2 U1 [1] [N2]",
    ])
    news = write_news(tmp_path, [("N1", "sports", "golf"), ("N2", "finance", "investing")])
    output = str(tmp_path / "out.txt")

    save_user_article_map(topk, news, output)

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
    topk = write_ground_truth(tmp_path, ["1 U1 [1,2] [N1,N2]"])
    news = write_news(tmp_path, [("N1", "sports", "golf"), ("N2", "finance", "investing")])
    output = str(tmp_path / "out.txt")

    save_user_article_map(topk, news, output)

    _, _, topics, _ = parse_user_map_line(open(output).readline())
    assert topics == ["sports", "finance"]
    logger.info(
        "Article topics are correctly looked up from news.tsv — "
        "expected ['sports', 'finance'], actual %s",
        topics
    )


def test_user_map_correct_subtopics(tmp_path):
    topk = write_ground_truth(tmp_path, ["1 U1 [1,2] [N1,N2]"])
    news = write_news(tmp_path, [("N1", "sports", "golf"), ("N2", "finance", "investing")])
    output = str(tmp_path / "out.txt")

    save_user_article_map(topk, news, output)

    _, _, _, subtopics = parse_user_map_line(open(output).readline())
    assert subtopics == ["golf", "investing"]
    logger.info(
        "Article subcategories are correctly looked up from news.tsv — "
        "expected ['golf', 'investing'], actual %s",
        subtopics
    )


def test_user_map_empty_subcategory_gives_none(tmp_path):
    # N1 has no subcategory in news.tsv; output should show "none".
    topk = write_ground_truth(tmp_path, ["1 U1 [1] [N1]"])
    news = write_news(tmp_path, [("N1", "news", "")])
    output = str(tmp_path / "out.txt")

    save_user_article_map(topk, news, output)

    _, _, _, subtopics = parse_user_map_line(open(output).readline())
    assert subtopics == ["none"]
    logger.info(
        "Empty subcategory in news.tsv is written as 'none' — "
        "expected ['none'], actual %s",
        subtopics
    )


def test_user_map_multiple_users_have_separate_entries(tmp_path):
    topk = write_ground_truth(tmp_path, [
        "1 U1 [1] [N1]",
        "2 U2 [1] [N2]",
    ])
    news = write_news(tmp_path, [("N1", "sports", "golf"), ("N2", "finance", "investing")])
    output = str(tmp_path / "out.txt")

    save_user_article_map(topk, news, output)

    lines = open(output).readlines()
    user_ids = {parse_user_map_line(l)[0] for l in lines}
    assert user_ids == {"U1", "U2"}
    logger.info(
        "Each user gets a separate line in the output — "
        "expected users {U1, U2}, actual %s",
        user_ids
    )
