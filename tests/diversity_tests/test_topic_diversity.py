import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from diversityScores.topic_diversity import topic_diversity, subtopic_diversity

logger = logging.getLogger(__name__)


def write_user_articles(tmp_path, lines):
    p = tmp_path / "user_articles.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# topic_diversity
# ---------------------------------------------------------------------------

def test_topic_diversity_all_same_topics(tmp_path):
    # 3 articles, all "news" → 1 unique / 3 total = 0.333
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2,N3] [news,news,news] [politics,world,economy]",
    ])
    score = topic_diversity(f)

    assert abs(score - 1/3) < 1e-9
    logger.info(
        "User with all same topics has minimum diversity — "
        "expected %.4f, actual %.4f", 1/3, score
    )


def test_topic_diversity_all_different_topics(tmp_path):
    # 3 articles, all different topics → 3 unique / 3 total = 1.0
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2,N3] [news,sports,finance] [politics,golf,stocks]",
    ])
    score = topic_diversity(f)

    assert abs(score - 1.0) < 1e-9
    logger.info(
        "User with all different topics has maximum diversity — "
        "expected 1.0000, actual %.4f", score
    )


def test_topic_diversity_mixed(tmp_path):
    # U1: news,sports,news → 2/3; U2: news,sports,finance,sports → 3/4; avg = (2/3 + 3/4) / 2
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2,N3] [news,sports,news] [p1,g1,p2]",
        "U2 [N4,N5,N6,N7] [news,sports,finance,sports] [p3,g2,s1,g3]",
    ])
    expected = (2/3 + 3/4) / 2
    score = topic_diversity(f)

    assert abs(score - expected) < 1e-9
    logger.info(
        "Average topic diversity across users is computed correctly — "
        "expected %.4f, actual %.4f", expected, score
    )


def test_topic_diversity_excludes_single_click_users(tmp_path):
    # U1 has 1 click (excluded); U2 has 3 clicks with 2 unique → 2/3
    f = write_user_articles(tmp_path, [
        "U1 [N1] [news] [politics]",
        "U2 [N2,N3,N4] [news,sports,news] [p1,g1,p2]",
    ])
    score = topic_diversity(f)

    assert abs(score - 2/3) < 1e-9
    logger.info(
        "Users with only one click are excluded from diversity calculation — "
        "expected %.4f (U1 excluded), actual %.4f", 2/3, score
    )


def test_topic_diversity_no_eligible_users_returns_zero(tmp_path):
    # All users have only 1 click → no eligible users → 0.0
    f = write_user_articles(tmp_path, [
        "U1 [N1] [news] [politics]",
        "U2 [N2] [sports] [golf]",
    ])
    score = topic_diversity(f)

    assert score == 0.0
    logger.info(
        "No eligible users (all have ≤1 click) returns 0.0 — "
        "expected 0.0000, actual %.4f", score
    )


def test_topic_diversity_score_between_zero_and_one(tmp_path):
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2,N3,N4] [news,news,sports,finance] [p1,p2,g1,s1]",
        "U2 [N5,N6,N7] [sports,sports,sports] [g2,g3,g4]",
    ])
    score = topic_diversity(f)

    assert 0.0 <= score <= 1.0
    logger.info(
        "Topic diversity score is always in [0, 1] — "
        "expected range [0.0, 1.0], actual %.4f", score
    )


# ---------------------------------------------------------------------------
# subtopic_diversity
# ---------------------------------------------------------------------------

def test_subtopic_diversity_default_category_news(tmp_path):
    # U1 has 2 news articles with different subtopics → 2/2 = 1.0
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2,N3] [news,news,sports] [politics,world,golf]",
    ])
    score = subtopic_diversity(f)

    assert abs(score - 1.0) < 1e-9
    logger.info(
        "Subtopic diversity defaults to category='news' and counts unique subtopics — "
        "expected 1.0000, actual %.4f", score
    )


def test_subtopic_diversity_custom_category(tmp_path):
    # category="sports": U1 has 3 sports articles, 2 unique subtopics → 2/3
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2,N3,N4] [sports,news,sports,sports] [golf,politics,golf,tennis]",
    ])
    score = subtopic_diversity(f, category="sports")

    assert abs(score - 2/3) < 1e-9
    logger.info(
        "Custom category parameter correctly filters to the specified category — "
        "expected %.4f (sports: golf,golf,tennis → 2/3), actual %.4f", 2/3, score
    )


def test_subtopic_diversity_user_with_no_category_articles_contributes_zero(tmp_path):
    # U1 has 2 articles but none in "news" → contributes 0.0; U2 has 2 news with 2 unique → 1.0; avg = 0.5
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2] [sports,finance] [golf,stocks]",
        "U2 [N3,N4] [news,news] [politics,world]",
    ])
    score = subtopic_diversity(f, category="news")

    assert abs(score - 0.5) < 1e-9
    logger.info(
        "User with no articles in the target category contributes 0.0 to the average — "
        "expected 0.5000 (U1=0.0, U2=1.0), actual %.4f", score
    )


def test_subtopic_diversity_all_same_subtopics(tmp_path):
    # U1: 3 news articles all "politics" → 1 unique / 3 total = 1/3
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2,N3] [news,news,news] [politics,politics,politics]",
    ])
    score = subtopic_diversity(f, category="news")

    assert abs(score - 1/3) < 1e-9
    logger.info(
        "All same subtopics in the category gives minimum diversity — "
        "expected %.4f, actual %.4f", 1/3, score
    )


def test_subtopic_diversity_excludes_single_click_users(tmp_path):
    # U1 has 1 click (excluded); U2 has 2 news articles with 2 unique subtopics → 1.0
    f = write_user_articles(tmp_path, [
        "U1 [N1] [news] [politics]",
        "U2 [N2,N3] [news,news] [politics,world]",
    ])
    score = subtopic_diversity(f, category="news")

    assert abs(score - 1.0) < 1e-9
    logger.info(
        "Users with only one click overall are excluded from subtopic diversity — "
        "expected 1.0000 (U1 excluded), actual %.4f", score
    )


def test_subtopic_diversity_score_between_zero_and_one(tmp_path):
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2,N3] [news,news,sports] [politics,politics,golf]",
        "U2 [N4,N5,N6] [news,finance,news] [world,stocks,economy]",
    ])
    score = subtopic_diversity(f, category="news")

    assert 0.0 <= score <= 1.0
    logger.info(
        "Subtopic diversity score is always in [0, 1] — "
        "expected range [0.0, 1.0], actual %.4f", score
    )