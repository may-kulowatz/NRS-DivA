import sys
import os
import pickle
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from diversity_module.content_diversity import (
    content_diversity,
    load_news_embeddings,
    _ild,
    _word_tokenize,
)

logger = logging.getLogger(__name__)


def write_user_articles(tmp_path, lines):
    p = tmp_path / "user_articles.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# _word_tokenize
# ---------------------------------------------------------------------------

def test_word_tokenize_lowercases_and_splits_punctuation(tmp_path):
    # Matches the MIND word_dict tokenizer: lowercased words + stand-alone punctuation
    tokens = _word_tokenize("The DEA's Take-Back Day!")

    assert tokens == ["the", "dea", "s", "take", "back", "day", "!"]
    logger.info(
        "Tokenizer lowercases and separates punctuation like the MIND word_dict — "
        "expected 7 tokens, actual %d: %s", len(tokens), tokens
    )


def test_word_tokenize_non_string_returns_empty():
    assert _word_tokenize(None) == []
    logger.info("Non-string input tokenizes to an empty list — actual %s", [])


# ---------------------------------------------------------------------------
# _ild  (intra-list diversity of one user's content vectors)
# ---------------------------------------------------------------------------

def test_ild_orthogonal_vectors_is_one():
    # Two orthogonal vectors → cosine similarity 0 → ILD = 1 - 0 = 1.0
    score = _ild([np.array([1.0, 0.0]), np.array([0.0, 1.0])])

    assert abs(score - 1.0) < 1e-9
    logger.info(
        "Orthogonal content vectors give maximum diversity — "
        "expected 1.0000, actual %.4f", score
    )


def test_ild_identical_vectors_is_zero():
    # Two identical vectors → cosine similarity 1 → ILD = 1 - 1 = 0.0
    score = _ild([np.array([1.0, 0.0]), np.array([1.0, 0.0])])

    assert abs(score - 0.0) < 1e-9
    logger.info(
        "Identical content vectors give minimum diversity — "
        "expected 0.0000, actual %.4f", score
    )


def test_ild_magnitude_does_not_matter():
    # Cosine ignores magnitude: [2,0] and [5,0] are still identical in direction → ILD 0
    score = _ild([np.array([2.0, 0.0]), np.array([5.0, 0.0])])

    assert abs(score - 0.0) < 1e-9
    logger.info(
        "ILD uses cosine similarity, so vector magnitude is irrelevant — "
        "expected 0.0000, actual %.4f", score
    )


def test_ild_three_vectors_averages_over_pairs():
    # Pairs: (a,b) cos=1, (a,c) cos=0, (b,c) cos=0 → ILS = 1/3 → ILD = 2/3
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    score = _ild([a, b, c])

    assert abs(score - 2/3) < 1e-9
    logger.info(
        "ILD averages cosine similarity over all unordered pairs — "
        "expected %.4f, actual %.4f", 2/3, score
    )


# ---------------------------------------------------------------------------
# content_diversity
# ---------------------------------------------------------------------------

def test_content_diversity_all_identical_articles_is_zero(tmp_path):
    embeddings = {"N1": np.array([1.0, 0.0]), "N2": np.array([1.0, 0.0])}
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2] [news,news] [politics,world]",
    ])
    score = content_diversity(f, embeddings)

    assert abs(score - 0.0) < 1e-9
    logger.info(
        "A list of content-identical articles has zero diversity — "
        "expected 0.0000, actual %.4f", score
    )


def test_content_diversity_orthogonal_articles_is_one(tmp_path):
    embeddings = {"N1": np.array([1.0, 0.0]), "N2": np.array([0.0, 1.0])}
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2] [news,sports] [politics,golf]",
    ])
    score = content_diversity(f, embeddings)

    assert abs(score - 1.0) < 1e-9
    logger.info(
        "A list of content-orthogonal articles has maximum diversity — "
        "expected 1.0000, actual %.4f", score
    )


def test_content_diversity_averages_across_users(tmp_path):
    # U1: orthogonal → ILD 1.0 ; U2: identical → ILD 0.0 ; avg = 0.5
    embeddings = {
        "N1": np.array([1.0, 0.0]), "N2": np.array([0.0, 1.0]),
        "N3": np.array([1.0, 0.0]), "N4": np.array([1.0, 0.0]),
    }
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2] [news,sports] [p1,g1]",
        "U2 [N3,N4] [news,news] [p2,p3]",
    ])
    score = content_diversity(f, embeddings)

    assert abs(score - 0.5) < 1e-9
    logger.info(
        "Content diversity averages per-user ILD across users — "
        "expected 0.5000 (U1=1.0, U2=0.0), actual %.4f", score
    )


def test_content_diversity_excludes_single_click_users(tmp_path):
    # U1 has 1 click (excluded); U2 orthogonal → ILD 1.0
    embeddings = {
        "N1": np.array([1.0, 0.0]),
        "N2": np.array([1.0, 0.0]), "N3": np.array([0.0, 1.0]),
    }
    f = write_user_articles(tmp_path, [
        "U1 [N1] [news] [politics]",
        "U2 [N2,N3] [news,sports] [p1,g1]",
    ])
    score = content_diversity(f, embeddings)

    assert abs(score - 1.0) < 1e-9
    logger.info(
        "Users with only one click are excluded from content diversity — "
        "expected 1.0000 (U1 excluded), actual %.4f", score
    )


def test_content_diversity_skips_users_with_too_few_embeddable_articles(tmp_path):
    # U1 has 2 clicks but only N1 has an embedding → skipped (need ≥2 vectors)
    # U2 orthogonal → ILD 1.0
    embeddings = {
        "N1": np.array([1.0, 0.0]),  # N2 intentionally missing
        "N3": np.array([1.0, 0.0]), "N4": np.array([0.0, 1.0]),
    }
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2] [news,sports] [p1,g1]",
        "U2 [N3,N4] [news,sports] [p2,g2]",
    ])
    score = content_diversity(f, embeddings)

    assert abs(score - 1.0) < 1e-9
    logger.info(
        "Users left with fewer than two embeddable articles are skipped — "
        "expected 1.0000 (U1 skipped), actual %.4f", score
    )


def test_content_diversity_no_eligible_users_returns_zero(tmp_path):
    embeddings = {"N1": np.array([1.0, 0.0]), "N2": np.array([0.0, 1.0])}
    f = write_user_articles(tmp_path, [
        "U1 [N1] [news] [politics]",
        "U2 [N2] [sports] [golf]",
    ])
    score = content_diversity(f, embeddings)

    assert score == 0.0
    logger.info(
        "No eligible users (all have ≤1 click) returns 0.0 — "
        "expected 0.0000, actual %.4f", score
    )


def test_content_diversity_score_between_zero_and_one(tmp_path):
    embeddings = {
        "N1": np.array([1.0, 0.0]), "N2": np.array([0.7, 0.3]),
        "N3": np.array([0.2, 0.9]), "N4": np.array([1.0, 1.0]),
    }
    f = write_user_articles(tmp_path, [
        "U1 [N1,N2,N3,N4] [news,news,sports,finance] [p1,p2,g1,s1]",
    ])
    score = content_diversity(f, embeddings)

    assert 0.0 <= score <= 1.0
    logger.info(
        "Content diversity score is always in [0, 1] — "
        "expected range [0.0, 1.0], actual %.4f", score
    )


# ---------------------------------------------------------------------------
# load_news_embeddings
# ---------------------------------------------------------------------------

def _write_embedding_fixtures(tmp_path):
    # Row 0 is padding; words map to rows 1..3
    word_embeddings = np.array([
        [0.0, 0.0],   # 0: padding
        [1.0, 0.0],   # 1: "cat"
        [0.0, 1.0],   # 2: "dog"
        [2.0, 2.0],   # 3: "fish"
    ])
    word_dict = {"cat": 1, "dog": 2, "fish": 3}

    emb_file = tmp_path / "embedding.npy"
    np.save(str(emb_file), word_embeddings)
    wd_file = tmp_path / "word_dict.pkl"
    with open(wd_file, "wb") as f:
        pickle.dump(word_dict, f)
    return str(emb_file), str(wd_file)


def _write_news_tsv(tmp_path, rows):
    # Minimal news.tsv: id, category, subcategory, title (+ trailing columns ignored)
    p = tmp_path / "news.tsv"
    lines = ["\t".join([nid, "news", "sub", title, "abstract", "url", "[]", "[]"])
             for nid, title in rows]
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def test_load_news_embeddings_averages_title_word_vectors(tmp_path):
    emb_file, wd_file = _write_embedding_fixtures(tmp_path)
    news_file = _write_news_tsv(tmp_path, [
        ("N1", "Cat Dog"),   # mean([1,0],[0,1]) = [0.5, 0.5]
        ("N2", "Fish"),      # [2, 2]
    ])
    embeddings = load_news_embeddings(news_file, emb_file, wd_file)

    assert set(embeddings) == {"N1", "N2"}
    assert np.allclose(embeddings["N1"], [0.5, 0.5])
    assert np.allclose(embeddings["N2"], [2.0, 2.0])
    logger.info(
        "News embedding is the mean of its known title-word vectors — "
        "N1 expected [0.5, 0.5], actual %s", embeddings["N1"].tolist()
    )


def test_load_news_embeddings_ignores_unknown_words(tmp_path):
    emb_file, wd_file = _write_embedding_fixtures(tmp_path)
    news_file = _write_news_tsv(tmp_path, [
        ("N1", "Cat unknownword"),  # only "cat" is known → [1, 0]
    ])
    embeddings = load_news_embeddings(news_file, emb_file, wd_file)

    assert np.allclose(embeddings["N1"], [1.0, 0.0])
    logger.info(
        "Words absent from word_dict are ignored when averaging — "
        "N1 expected [1.0, 0.0], actual %s", embeddings["N1"].tolist()
    )


def test_load_news_embeddings_skips_news_with_no_known_words(tmp_path):
    emb_file, wd_file = _write_embedding_fixtures(tmp_path)
    news_file = _write_news_tsv(tmp_path, [
        ("N1", "Cat"),
        ("N2", "qux quux"),  # no known words → skipped (no content vector)
    ])
    embeddings = load_news_embeddings(news_file, emb_file, wd_file)

    assert "N2" not in embeddings
    assert set(embeddings) == {"N1"}
    logger.info(
        "News whose title has no known words is skipped — "
        "expected only {'N1'}, actual %s", set(embeddings)
    )


def _write_news_tsv_ta(tmp_path, rows):
    # news.tsv with distinct title (col 3) and abstract (col 4) per row
    p = tmp_path / "news_ta.tsv"
    lines = ["\t".join([nid, "news", "sub", title, abstract, "url", "[]", "[]"])
             for nid, title, abstract in rows]
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def test_load_news_embeddings_text_col_selects_abstract(tmp_path):
    # text_col=4 averages the abstract words instead of the title words.
    emb_file, wd_file = _write_embedding_fixtures(tmp_path)
    news_file = _write_news_tsv_ta(tmp_path, [
        ("N1", "Cat", "Dog Fish"),  # title→[1,0] ; abstract→mean([0,1],[2,2])=[1,1.5]
    ])

    titles = load_news_embeddings(news_file, emb_file, wd_file)                # default col 3
    abstracts = load_news_embeddings(news_file, emb_file, wd_file, text_col=4)

    assert np.allclose(titles["N1"], [1.0, 0.0])       # default unchanged (title)
    assert np.allclose(abstracts["N1"], [1.0, 1.5])    # abstract-based vector
    logger.info(
        "text_col selects which news.tsv field is averaged — title(col3) N1=%s vs "
        "abstract(col4) N1=%s", titles["N1"].tolist(), abstracts["N1"].tolist()
    )


def test_load_news_embeddings_text_col_skips_rows_missing_that_column(tmp_path):
    # A row with no abstract column present is skipped when text_col=4.
    emb_file, wd_file = _write_embedding_fixtures(tmp_path)
    p = tmp_path / "news_short.tsv"
    # N1 has only 4 columns (no abstract at index 4); N2 has an abstract
    p.write_text("N1\tnews\tsub\tCat\n"
                 "N2\tnews\tsub\tDog\tFish\turl\t[]\t[]", encoding="utf-8")

    embeddings = load_news_embeddings(str(p), emb_file, wd_file, text_col=4)

    assert set(embeddings) == {"N2"}
    assert np.allclose(embeddings["N2"], [2.0, 2.0])   # "Fish" → [2,2]
    logger.info(
        "Rows without the requested text column are skipped — "
        "expected only {'N2'}, actual %s", set(embeddings)
    )
