"""mind_news dataset adapter.

mind_news is a news-only subset of MIND (built by mind_news/prepare.py): every
article sits in the MIND ``news`` category, so this adapter uses the subcategory
as the topic — the diversity metrics then measure variety among news
subcategories.

The files share MIND's format (tab-separated ``behaviors.tsv`` / ``news.tsv``
with inline ``Nxxxx-1`` / ``Nxxxx-0`` click labels), but this adapter parses them
independently so it carries no dependency on the MIND adapter.
"""

from dataset_module.common import Impression

__all__ = ["load_impressions", "load_article_meta", "load_titles"]


def load_impressions(behaviors_file):
    """Read behaviors.tsv into a list of normalized Impression records.

    behaviors.tsv columns: impr_id, user_id, time, history, impressions
    where `impressions` is space-separated `Nxxxx-1` (clicked) / `Nxxxx-0`.
    """
    impressions = []
    with open(behaviors_file, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            candidates = cols[4].split()
            candidate_ids = [c.split("-")[0] for c in candidates]
            labels = [int(c.split("-")[1]) for c in candidates]
            impressions.append(
                Impression(int(cols[0]), cols[1], cols[2], candidate_ids, labels)
            )
    return impressions


def load_article_meta(news_file):
    """Read news.tsv into ``{news_id: subcategory}``.

    news.tsv columns: news_id, category, subcategory, title, ... The subcategory
    (column 2) is used as the topic so ``topic_diversity`` measures
    news-subcategory variety. Missing subcategory is normalized to "none".
    """
    meta = {}
    with open(news_file, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            meta[cols[0]] = cols[2] if cols[2] else "none"
    return meta


def load_titles(news_file):
    """Read news.tsv into {news_id: title}. Title is column 3."""
    titles = {}
    with open(news_file, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) > 3:
                titles[cols[0]] = cols[3]
    return titles