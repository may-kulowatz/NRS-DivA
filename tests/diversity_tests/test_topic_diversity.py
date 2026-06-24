import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from diversity_module.topic_diversity import topic_diversity, subtopic_diversity
from recommender_module.common.subtopic import build_subtopic_subset
from dataset_module.common import Impression

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


def test_topic_diversity_multiple_topics_per_article(tmp_path):
    # An article can carry several topics, encoded as a "|"-separated group.
    # N1 -> {a, b}, N2 -> {b, c}: flattened = [a, b, b, c] -> 3 unique / 4 = 0.75
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2] [a|b,b|c] [s1,s2]",
    ])
    score = topic_diversity(f)

    assert abs(score - 0.75) < 1e-9
    logger.info(
        "Articles with multiple topics flatten all topics before counting — "
        "expected 0.7500 ([a,b,b,c] -> 3/4), actual %.4f", score
    )


def test_topic_diversity_filters_empty_topics(tmp_path):
    # N2 has no topics ("none") and must be ignored entirely:
    # only [a, b] count -> 2 unique / 2 = 1.0 (N2 affects neither count).
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2,N3] [a,none,b] [s1,s2,s3]",
    ])
    score = topic_diversity(f)

    assert abs(score - 1.0) < 1e-9
    logger.info(
        "Articles with no topics ('none') are filtered out of the calculation — "
        "expected 1.0000 ([a,b] -> 2/2, N2 ignored), actual %.4f", score
    )


def test_topic_diversity_user_with_only_empty_topics_skipped(tmp_path):
    # U1's articles are all topic-less -> U1 contributes nothing.
    # U2 has [a, b] -> 1.0; average over eligible users = 1.0.
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2] [none,none] [s1,s2]",
        "U2 [N3,N4] [a,b] [s3,s4]",
    ])
    score = topic_diversity(f)

    assert abs(score - 1.0) < 1e-9
    logger.info(
        "A user left with no topics after filtering is skipped entirely — "
        "expected 1.0000 (U1 skipped, U2=1.0), actual %.4f", score
    )


# ---------------------------------------------------------------------------
# build_subtopic_subset — the news-only subset that subtopic diversity scores
# ---------------------------------------------------------------------------
# article_meta maps {article_id: (topic, subtopic)}. The subset keeps only the
# parent-category ("news") articles and promotes each one's subtopic into the
# topic slot, so topic_diversity on the resulting user-article file measures
# subcategory variety.

ARTICLE_META = {
    "N1": ("news", "politics"),
    "N2": ("news", "world"),
    "N3": ("news", "politics"),
    "N4": ("sports", "golf"),
    "N5": ("finance", "stocks"),
}


def _impr(impr_id, user_id, pairs):
    """Build an Impression from [(article_id, clicked), ...]."""
    candidate_ids = [aid for aid, _ in pairs]
    labels = [c for _, c in pairs]
    return Impression(impr_id, user_id, "t", candidate_ids, labels)


def test_subset_keeps_only_category_articles_with_subtopic_as_topic():
    impr = _impr(1, "U1", [("N1", 1), ("N4", 1), ("N2", 0)])
    sub_imps, sub_meta, positions = build_subtopic_subset(
        [impr], ARTICLE_META, "news"
    )

    # Only the two news candidates survive, in original order.
    assert sub_imps[0].candidate_ids == ["N1", "N2"]
    assert sub_imps[0].labels == [1, 0]
    assert positions[1] == [0, 2]
    # subtopic promoted to the topic slot; non-news articles dropped from meta.
    assert sub_meta == {"N1": ("politics", "none"),
                        "N2": ("world", "none"),
                        "N3": ("politics", "none")}
    logger.info(
        "Subset keeps only category articles and promotes subtopic to topic — "
        "candidates %s, meta keys %s", sub_imps[0].candidate_ids, sorted(sub_meta)
    )


def test_subset_drops_impressions_with_no_category_candidates():
    impr = _impr(1, "U1", [("N4", 1), ("N5", 0)])  # no news candidates
    sub_imps, _, positions = build_subtopic_subset([impr], ARTICLE_META, "news")

    assert sub_imps == []
    assert positions == {}
    logger.info("Impressions with no candidate in the category are dropped — got %d", len(sub_imps))


def test_subset_drops_impressions_with_no_category_clicks():
    # Has a news candidate, but it wasn't clicked → not sensible, dropped.
    impr = _impr(1, "U1", [("N1", 0), ("N4", 1)])
    sub_imps, _, _ = build_subtopic_subset([impr], ARTICLE_META, "news")

    assert sub_imps == []
    logger.info("Impressions with no clicked article in the category are dropped — got %d", len(sub_imps))


# ---------------------------------------------------------------------------
# subtopic_diversity — topic diversity on the subset's user-article file
# ---------------------------------------------------------------------------

def test_subtopic_diversity_delegates_to_topic_diversity(tmp_path):
    # On a subset file (topic slot already holds subcategories) subtopic diversity
    # is exactly topic diversity: 2 unique of 3 → 2/3.
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2,N3] [politics,world,politics] [none,none,none]",
    ])
    assert abs(subtopic_diversity(f) - topic_diversity(f)) < 1e-12
    assert abs(subtopic_diversity(f) - 2/3) < 1e-9
    logger.info(
        "subtopic_diversity equals topic_diversity on the subset file — "
        "expected %.4f, actual %.4f", 2/3, subtopic_diversity(f)
    )