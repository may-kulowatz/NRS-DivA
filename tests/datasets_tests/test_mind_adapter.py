"""Tests for the MIND dataset adapter's metadata loaders."""

import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dataset_module.mind.adapter import load_article_meta, load_titles

logger = logging.getLogger(__name__)


def write_news(tmp_path, lines):
    p = tmp_path / "news.tsv"
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def test_article_meta_topic_and_subtopic(tmp_path):
    # news.tsv: id, category, subcategory, title, ...
    f = write_news(tmp_path, [
        "N1\tsports\tgolf\tTiger wins again\tabstract\turl",
        "N2\tnews\t\tBreaking story\tabstract\turl",  # empty subcategory -> "none"
    ])
    meta = load_article_meta(f)

    assert meta["N1"] == ("sports", "golf")
    assert meta["N2"] == ("news", "none")
    logger.info("Article meta parses topic/subtopic; empty subcategory -> 'none'. actual %s", meta)


def test_titles_from_column_three(tmp_path):
    f = write_news(tmp_path, [
        "N1\tsports\tgolf\tTiger wins again\tabstract\turl",
        "N2\tnews\tworld\tBreaking story\tabstract\turl",
    ])
    titles = load_titles(f)

    assert titles == {"N1": "Tiger wins again", "N2": "Breaking story"}
    logger.info("Titles are read from column 3 of news.tsv — actual %s", titles)


def test_titles_skips_rows_without_title(tmp_path):
    # A malformed row with fewer than 4 columns has no title and is skipped.
    f = write_news(tmp_path, [
        "N1\tsports\tgolf\tHas a title\tabstract\turl",
        "N2\tnews\tworld",  # no title column
    ])
    titles = load_titles(f)

    assert titles == {"N1": "Has a title"}
    logger.info("Rows without a title column are skipped — actual %s", titles)
