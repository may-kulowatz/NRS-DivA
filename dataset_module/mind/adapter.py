"""MIND dataset adapter.

Parses the MIND TSV files (behaviors.tsv, news.tsv) into the normalized
structures defined in dataset_module/common.py. All MIND-specific format knowledge
(tab-separated columns, inline `Nxxxx-1` / `Nxxxx-0` click labels) lives here
and nowhere else.
"""

from dataset_module.common import Impression


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
    """Read news.tsv into {news_id: topic}.

    news.tsv columns: news_id, category, subcategory, title, ...
    The topic is the category (column 1).
    """
    meta = {}
    with open(news_file, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            meta[cols[0]] = cols[1]
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