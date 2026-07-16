"""Tests for the EB-NeRD dataset adapter.

These build tiny synthetic Parquet files (so no large dataset download is
needed) and check that dataset_module/ebnerd/adapter.py normalizes them into the same
Impression / article-metadata shape the rest of the pipeline expects.

Skipped automatically if pandas + a Parquet engine are not installed.
"""

import sys
import os
import logging

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from dataset_module.ebnerd.adapter import load_impressions, load_article_meta, load_titles

logger = logging.getLogger(__name__)


def write_behaviors(tmp_path, rows):
    """rows: list of (impression_id, user_id, time, inview, clicked)."""
    df = pd.DataFrame(rows, columns=[
        "impression_id", "user_id", "impression_time",
        "article_ids_inview", "article_ids_clicked",
    ])
    p = str(tmp_path / "behaviors.parquet")
    df.to_parquet(p)
    return p


def write_articles(tmp_path, rows):
    """rows: list of (article_id, topics_list)."""
    df = pd.DataFrame(rows, columns=["article_id", "topics"])
    p = str(tmp_path / "articles.parquet")
    df.to_parquet(p)
    return p


# ---------------------------------------------------------------------------
# load_impressions
# ---------------------------------------------------------------------------

def test_labels_derived_from_clicked_list(tmp_path):
    f = write_behaviors(tmp_path, [
        (1, 100, "2023-01-01", [9001, 9002, 9003], [9002]),
    ])
    imps = load_impressions(f)

    assert imps[0].labels == [0, 1, 0]
    logger.info(
        "Click labels are derived by matching candidates against article_ids_clicked — "
        "expected [0, 1, 0], actual %s",
        imps[0].labels
    )


def test_ids_are_stringified(tmp_path):
    f = write_behaviors(tmp_path, [
        (1, 100, "2023-01-01", [9001, 9002], [9001]),
    ])
    imps = load_impressions(f)

    assert imps[0].candidate_ids == ["9001", "9002"]
    assert imps[0].user_id == "100"
    logger.info(
        "Integer article and user ids are normalized to strings — "
        "expected candidate_ids ['9001', '9002'] and user_id '100', actual %s / %s",
        imps[0].candidate_ids, imps[0].user_id
    )


def test_multiple_clicks_in_one_impression(tmp_path):
    f = write_behaviors(tmp_path, [
        (7, 100, "2023-01-01", [1, 2, 3, 4], [2, 4]),
    ])
    imps = load_impressions(f)

    assert imps[0].labels == [0, 1, 0, 1]
    assert sum(imps[0].labels) == 2
    logger.info(
        "An impression with several clicks marks each clicked candidate — "
        "expected [0, 1, 0, 1] (2 clicks), actual %s",
        imps[0].labels
    )


def test_no_clicks_all_zero(tmp_path):
    f = write_behaviors(tmp_path, [
        (1, 100, "2023-01-01", [1, 2, 3], []),
    ])
    imps = load_impressions(f)

    assert imps[0].labels == [0, 0, 0]
    logger.info(
        "An impression with no clicks yields all-zero labels — expected [0, 0, 0], actual %s",
        imps[0].labels
    )


# ---------------------------------------------------------------------------
# load_article_meta
# ---------------------------------------------------------------------------

def test_multiple_topics_joined_with_pipe(tmp_path):
    f = write_articles(tmp_path, [
        (3001353, ["Crime", "Violent crime"]),
    ])
    meta = load_article_meta(f)

    # Several topics per article are kept, joined by "|", with spaces in a label
    # replaced by "_" so the whitespace-delimited file stays parseable.
    assert meta["3001353"] == "Crime|Violent_crime"
    logger.info(
        "Multiple topics are joined with '|' and spaces become '_' — "
        "expected 'Crime|Violent_crime', actual %s",
        meta["3001353"]
    )


def test_single_topic_has_no_pipe(tmp_path):
    f = write_articles(tmp_path, [
        (3012771, ["Sport"]),
    ])
    meta = load_article_meta(f)

    assert meta["3012771"] == "Sport"
    logger.info(
        "A single-topic article produces a plain topic with no separator — "
        "expected 'Sport', actual %s",
        meta["3012771"]
    )


def test_empty_topics_is_none(tmp_path):
    f = write_articles(tmp_path, [
        (3001353, []),
    ])
    meta = load_article_meta(f)

    assert meta["3001353"] == "none"
    logger.info(
        "An article with no topics normalizes its topic to 'none' — expected 'none', actual %s",
        meta["3001353"]
    )


# ---------------------------------------------------------------------------
# load_titles
# ---------------------------------------------------------------------------

def test_titles_keyed_by_stringified_id(tmp_path):
    df = pd.DataFrame(
        [(3001353, "Skud i natten"), (3012771, "Sejr til FCK")],
        columns=["article_id", "title"],
    )
    p = str(tmp_path / "articles.parquet")
    df.to_parquet(p)

    titles = load_titles(p)

    assert titles == {"3001353": "Skud i natten", "3012771": "Sejr til FCK"}
    logger.info(
        "Titles are returned keyed by stringified article id — actual %s", titles
    )