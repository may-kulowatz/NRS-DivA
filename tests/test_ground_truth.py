import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from general.ground_truth import load_ground_truth_mind, save_ground_truth_mind

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
# load_ground_truth_mind
# ---------------------------------------------------------------------------

def test_load_returns_correct_labels(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1"])
    results = load_ground_truth_mind(f)

    assert results[0][2] == [1, 0, 1]
    logger.info("Loaded labels correctly: %s", results[0][2])


def test_load_returns_correct_article_ids(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1"])
    results = load_ground_truth_mind(f)

    assert results[0][3] == ["N1", "N2", "N3"]
    logger.info("Loaded article IDs correctly: %s", results[0][3])


# ---------------------------------------------------------------------------
# save_ground_truth_mind
# ---------------------------------------------------------------------------

def test_only_clicked_articles_in_output(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1"])
    output = str(tmp_path / "out.txt")

    save_ground_truth_mind(load_ground_truth_mind(f), output)

    _, _, positions, ids = parse_output_line(open(output).readline())
    assert ids == ["N1", "N3"]
    logger.info("Only clicked articles appear in output: %s (N2 correctly excluded)", ids)


def test_correct_positions_of_clicked_articles(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU1\t11/15/2019 10:00:00\t\tN1-0 N2-1 N3-0 N4-1"])
    output = str(tmp_path / "out.txt")

    save_ground_truth_mind(load_ground_truth_mind(f), output)

    _, _, positions, _ = parse_output_line(open(output).readline())
    assert positions == [2, 4]
    logger.info("Clicked article positions are correct: %s (N2 at 2, N4 at 4)", positions)


def test_position_and_id_lists_same_length(tmp_path):
    f = write_behaviors(tmp_path, [
        "1\tU1\t11/15/2019 10:00:00\t\tN1-1 N2-0 N3-1 N4-1",
        "2\tU2\t11/15/2019 11:00:00\t\tN5-0 N6-1",
    ])
    output = str(tmp_path / "out.txt")

    save_ground_truth_mind(load_ground_truth_mind(f), output)

    for line in open(output):
        _, _, positions, ids = parse_output_line(line)
        assert len(positions) == len(ids)
        logger.info("Impression has %d positions and %d IDs — lists are the same length",
                    len(positions), len(ids))


def test_no_clicks_gives_empty_lists(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU1\t11/15/2019 10:00:00\t\tN1-0 N2-0 N3-0"])
    output = str(tmp_path / "out.txt")

    save_ground_truth_mind(load_ground_truth_mind(f), output)

    _, _, positions, ids = parse_output_line(open(output).readline())
    assert positions == []
    assert ids == []
    logger.info("Impression with no clicks produces empty lists for both positions and IDs")


def test_user_id_written_to_output(tmp_path):
    f = write_behaviors(tmp_path, ["1\tU42\t11/15/2019 10:00:00\t\tN1-1 N2-0"])
    output = str(tmp_path / "out.txt")

    save_ground_truth_mind(load_ground_truth_mind(f), output)

    _, user_id, _, _ = parse_output_line(open(output).readline())
    assert user_id == "U42"
    logger.info("User ID correctly written to output: %s", user_id)
