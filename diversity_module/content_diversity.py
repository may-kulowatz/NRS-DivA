"""Intra-list content diversity (ILD) from article-title embeddings.

Represents each news article by a content vector — the mean of the word
embeddings of one of its text fields (title by default) — and scores a user's
recommended list by the mean pairwise cosine *distance* between those vectors.
Embeddings are either built from the MIND word_dict / embedding.npy bundle
(``load_news_embeddings``) or read from a precomputed store
(``load_precomputed_embeddings``). See ``content_diversity_normalized`` for the
per-impression normalized variant.

Adapted from https://github.com/recommenders-team/recommenders/
"""

import re
import pickle

import numpy as np

try:
    # When imported as part of the package (e.g. from diversity_module)
    from diversity_module.topic_diversity import _parse_user_articles
except ImportError:
    # When run directly from inside the diversity_module/ directory
    from topic_diversity import _parse_user_articles


# Same tokenizer the MIND word_dict / embedding.npy were built with
# (Microsoft Recommenders `word_tokenize`): keeps words and stand-alone
# punctuation so token-to-embedding lookups line up with word_dict.
_TOKEN_PATTERN = re.compile(r"[\w]+|[.,!?;|]")


def _word_tokenize(text):
    return _TOKEN_PATTERN.findall(text.lower()) if isinstance(text, str) else []


def load_news_embeddings(news_file, embedding_file, word_dict_file, text_col=3):
    """Build a {news_id: embedding vector} mapping.

    Each news article is represented by the mean of the word embeddings of the
    words in one of its text fields (content-based representation). ``text_col``
    selects which ``news.tsv`` column supplies that text: the default ``3`` is
    the title (the historical behaviour); ``4`` is the abstract. News whose
    chosen field contains no known word are skipped, since they have no content
    vector.
    """
    word_embeddings = np.load(embedding_file)
    with open(word_dict_file, "rb") as f:
        word_dict = pickle.load(f)

    news_embeddings = {}
    with open(news_file, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) <= text_col:
                continue
            news_id, text = cols[0], cols[text_col]
            indices = [word_dict[w] for w in _word_tokenize(text) if w in word_dict]
            if not indices:
                continue
            news_embeddings[news_id] = word_embeddings[indices].mean(axis=0)
    return news_embeddings


def load_precomputed_embeddings(
    vector_file, id_column="article_id", vector_column="contrastive_vector"
):
    """Build a {article_id: embedding vector} mapping from a Parquet file.

    Unlike load_news_embeddings (which averages per-word vectors over a title),
    this reads one ready-made document embedding per article — e.g. eb-nerd's
    contrastive_vector.parquet, a 768-dim vector per article_id. There is no
    tokenizer or word dictionary involved, so it sidesteps the language/vocab
    mismatch that makes MIND's English word embeddings unusable for eb-nerd.

    Article ids are stringified to match the ids used everywhere else in the
    pipeline (the adapters stringify ids, MIND ids are already strings).
    """
    import pyarrow.parquet as pq

    table = pq.read_table(vector_file, columns=[id_column, vector_column])
    ids = table.column(id_column).to_pylist()
    vectors = table.column(vector_column).to_pylist()
    return {str(aid): np.asarray(v, dtype=np.float32) for aid, v in zip(ids, vectors)}


def _ild(vectors):
    """Intra-list diversity of one user's list of content vectors.

    ILS is the mean pairwise cosine similarity (upper triangle of the
    similarity matrix); diversity is 1 - ILS.
    """
    matrix = np.vstack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.where(norms == 0, 1.0, norms)
    similarity = normalized @ normalized.T
    i, j = np.triu_indices(len(vectors), k=1)
    ils = similarity[i, j].mean()
    return 1.0 - ils


def content_diversity(processed_file, news_embeddings):
    """Average intra-list content diversity (ILD) across users.

    Mirrors topic_diversity: only users with more than one click count, and
    only articles with a known content embedding contribute. Users left with
    fewer than two embeddable articles are skipped.
    """
    per_user = []
    for _, (ids, _) in _parse_user_articles(processed_file).items():
        if len(ids) <= 1:
            continue
        vectors = [news_embeddings[n] for n in ids if n in news_embeddings]
        if len(vectors) <= 1:
            continue
        per_user.append(_ild(vectors))
    return sum(per_user) / len(per_user) if per_user else 0.0