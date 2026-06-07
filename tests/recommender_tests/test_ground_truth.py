import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from datasets.mind import load_impressions, load_article_meta
from recommenders.ground_truth import extract_ground_truth, save_ground_truth
from recommenders.io import save_user_article_map

logger = logging.getLogger(__name__)


def write_behaviors(tmp_path, lines):
    p = tmp_path / "behaviors.tsv"
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def parse_output_line(line):
    parts = line.strip().split()
    inner_pos = parts[2][1:-1]
    inner_ids = parts[3][1:-1]
    positions = [int(p) for p in inner_pos.split(",")] if inner_pos else []
    ids = inner_ids.split(",") if inner_ids else []
    return int(parts[0]), parts[1], positions, ids


# ---------------------------------------------------------------------------
# load_impressions (MIND adapter)
# ---------------------------------------------------------------------------

def test_load_returns_correct_labels(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1"])
    impressions = load_impressions(f)

    assert impressions[0].labels == [1, 0, 1]
    logger.info(
        "Adapter correctly parses click labels (1=clicked, 0=not clicked) from the candidates column — "
        "expected [1, 0, 1], actual %s",
        impressions[0].labels
    )


def test_load_returns_correct_article_ids(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1"])
    impressions = load_impressions(f)

    assert impressions[0].candidate_ids == ["N1", "N2", "N3"]
    logger.info(
        "Adapter correctly parses article IDs from the candidates column — "
        "expected ['N1', 'N2', 'N3'], actual %s",
        impressions[0].candidate_ids
    )


# ---------------------------------------------------------------------------
# extract_ground_truth + save_ground_truth
# ---------------------------------------------------------------------------

def test_only_clicked_articles_in_output(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1"])
    output = str(tmp_path / "out.txt")

    save_ground_truth(extract_ground_truth(load_impressions(f)), output)

    _, _, _, ids = parse_output_line(open(output).readline())
    assert ids == ["N1", "N3"]
    logger.info(
        "Only articles with label=1 appear in the output, unclicked articles are excluded — "
        "expected ['N1', 'N3'], actual %s",
        ids
    )


def test_correct_positions_of_clicked_articles(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU1\t11/15/2019 10:00:00\t\tN1-0 N2-1 N3-0 N4-1"])
    output = str(tmp_path / "out.txt")

    save_ground_truth(extract_ground_truth(load_impressions(f)), output)

    _, _, positions, _ = parse_output_line(open(output).readline())
    assert positions == [2, 4]
    logger.info(
        "Positions reflect the 1-indexed locations of clicked articles within the impression — "
        "expected [2, 4], actual %s",
        positions
    )


def test_position_and_id_lists_same_length(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1 N4-1",
        "2\tU2\t11/15/2019 11:00:00\t\tN5-0 N6-1",
    ])
    output = str(tmp_path / "out.txt")

    save_ground_truth(extract_ground_truth(load_impressions(f)), output)

    for line in open(output):
        _, _, positions, ids = parse_output_line(line)
        assert len(positions) == len(ids)
        logger.info(
            "Position list and article ID list always have equal length for each impression — "
            "expected equal lengths, actual positions=%d, IDs=%d",
            len(positions), len(ids)
        )


def test_no_clicks_gives_empty_lists(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU1\t11/15/2019 10:00:00\t\tN1-0 N2-0 N3-0"])
    output = str(tmp_path / "out.txt")

    save_ground_truth(extract_ground_truth(load_impressions(f)), output)

    _, _, positions, ids = parse_output_line(open(output).readline())
    assert positions == []
    assert ids == []
    logger.info(
        "Impression with no clicked articles produces empty lists for both positions and article IDs — "
        "expected positions=[], IDs=[], actual positions=%s, IDs=%s",
        positions, ids
    )


def test_user_id_written_to_output(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU42\t11/15/2019 10:00:00\t\tN1-1 N2-0"])
    output = str(tmp_path / "out.txt")

    save_ground_truth(extract_ground_truth(load_impressions(f)), output)

    _, user_id, _, _ = parse_output_line(open(output).readline())
    assert user_id == "U42"
    logger.info(
        "User ID from behaviors file is correctly written to the ground truth output — "
        "expected 'U42', actual '%s'",
        user_id
    )


# ---------------------------------------------------------------------------
# save_user_article_map
# ---------------------------------------------------------------------------

def write_prediction_file(tmp_path, lines):
    p = tmp_path / "pred.txt"
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
    # U1 appears in two impressions; both sets of articles should be combined.
    pred = write_prediction_file(tmp_path, [
        "1 U1 [1] [N1]",
        "2 U1 [1] [N2]",
    ])
    meta = write_news(tmp_path, [("N1", "sports", "golf"), ("N2", "finance", "investing")])
    output = str(tmp_path / "out.txt")

    save_user_article_map(pred, meta, output)

    lines = open(output).readlines()
    user_id, ids, _, _ = parse_user_map_line(lines[0])
    assert user_id == "U1"
    assert ids == ["N1", "N2"]
    logger.info(
        "Clicked articles from multiple impressions are grouped under one user entry — "
        "expected U1: [N1, N2], actual %s: %s",
        user_id, ids
    )


def test_user_map_correct_topics(tmp_path):
    pred = write_prediction_file(tmp_path, ["1 U1 [1,2] [N1,N2]"])
    meta = write_news(tmp_path, [("N1", "sports", "golf"), ("N2", "finance", "investing")])
    output = str(tmp_path / "out.txt")

    save_user_article_map(pred, meta, output)

    _, _, topics, _ = parse_user_map_line(open(output).readline())
    assert topics == ["sports", "finance"]
    logger.info(
        "Article topics are correctly looked up from article metadata — "
        "expected ['sports', 'finance'], actual %s",
        topics
    )


def test_user_map_correct_subtopics(tmp_path):
    pred = write_prediction_file(tmp_path, ["1 U1 [1,2] [N1,N2]"])
    meta = write_news(tmp_path, [("N1", "sports", "golf"), ("N2", "finance", "investing")])
    output = str(tmp_path / "out.txt")

    save_user_article_map(pred, meta, output)

    _, _, _, subtopics = parse_user_map_line(open(output).readline())
    assert subtopics == ["golf", "investing"]
    logger.info(
        "Article subcategories are correctly looked up from article metadata — "
        "expected ['golf', 'investing'], actual %s",
        subtopics
    )


def test_user_map_multiple_users_have_separate_entries(tmp_path):
    pred = write_prediction_file(tmp_path, [
        "1 U1 [1] [N1]",
        "2 U2 [1] [N2]",
    ])
    meta = write_news(tmp_path, [("N1", "sports", "golf"), ("N2", "finance", "investing")])
    output = str(tmp_path / "out.txt")

    save_user_article_map(pred, meta, output)

    lines = open(output).readlines()
    user_ids = {parse_user_map_line(l)[0] for l in lines}
    assert user_ids == {"U1", "U2"}
    logger.info(
        "Each user gets a separate line in the output — "
        "expected users {U1, U2}, actual %s",
        user_ids
    )