# TODO: verify!! Also, add license and information about where this came from.

import os
import re
import pickle

import numpy as np

try:
    # When imported as part of the package (e.g. from pipeline.py)
    from diversityScores.topic_diversity import _parse_user_articles
except ImportError:
    # When run directly from inside the diversityScores/ directory
    from topic_diversity import _parse_user_articles


# Same tokenizer the MIND word_dict / embedding.npy were built with
# (Microsoft Recommenders `word_tokenize`): keeps words and stand-alone
# punctuation so token-to-embedding lookups line up with word_dict.
_TOKEN_PATTERN = re.compile(r"[\w]+|[.,!?;|]")


def _word_tokenize(text):
    return _TOKEN_PATTERN.findall(text.lower()) if isinstance(text, str) else []


def load_news_embeddings(news_file, embedding_file, word_dict_file):
    """Build a {news_id: embedding vector} mapping.

    Each news article is represented by the mean of the word embeddings of
    the words in its title (content-based representation). News whose title
    contains no known word are skipped, since they have no content vector.
    """
    word_embeddings = np.load(embedding_file)
    with open(word_dict_file, "rb") as f:
        word_dict = pickle.load(f)

    news_embeddings = {}
    with open(news_file, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 4:
                continue
            news_id, title = cols[0], cols[3]
            indices = [word_dict[w] for w in _word_tokenize(title) if w in word_dict]
            if not indices:
                continue
            news_embeddings[news_id] = word_embeddings[indices].mean(axis=0)
    return news_embeddings


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


def content_diversity(user_articles_file, news_embeddings):
    """Average intra-list content diversity (ILD) across users.

    Mirrors topic_diversity: only users with more than one click count, and
    only articles with a known content embedding contribute. Users left with
    fewer than two embeddable articles are skipped.
    """
    per_user = []
    for _, (ids, _, _) in _parse_user_articles(user_articles_file).items():
        if len(ids) <= 1:
            continue
        vectors = [news_embeddings[n] for n in ids if n in news_embeddings]
        if len(vectors) <= 1:
            continue
        per_user.append(_ild(vectors))
    return sum(per_user) / len(per_user) if per_user else 0.0


if __name__ == "__main__":
    _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mind_dir = os.path.join(_project_dir, "data", "MIND")

    news_embeddings = load_news_embeddings(
        os.path.join(mind_dir, "MINDsmall_dev", "news.tsv"),
        os.path.join(mind_dir, "utils", "embedding.npy"),
        os.path.join(mind_dir, "utils", "word_dict.pkl"),
    )

    pred_dir = os.path.join(mind_dir, "predictions")
    files = {
        "random":       os.path.join(pred_dir, "user_articles_random.txt"),
        "popular":      os.path.join(pred_dir, "user_articles_popular.txt"),
        "nrms":         os.path.join(pred_dir, "user_articles_nrms.txt"),
        "ground_truth": os.path.join(pred_dir, "user_articles_ground_truth.txt"),
    }

    for name, path in files.items():
        print(f"\n=== {name} ===")
        print(f"  Content diversity (ILD):      {content_diversity(path, news_embeddings):.4f}")