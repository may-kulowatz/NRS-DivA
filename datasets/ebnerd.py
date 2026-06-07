"""eb-nerd (Ekstra Bladet) dataset adapter.

Parses eb-nerd's Parquet files into the same normalized structures the MIND
adapter produces, so the recommenders and output writers can be reused
unchanged. All eb-nerd-specific format knowledge lives here:

  * behaviors.parquet — one row per impression. Candidates are in
    `article_ids_inview`; clicked articles are listed explicitly in
    `article_ids_clicked` (rather than MIND's inline `-1` / `-0` labels).
  * articles.parquet — `topics` is a LIST of human-readable topic labels per
    article (e.g. ["Crime", "Violent crime"]); all of them are kept so topic
    diversity can account for multiple topics. The numeric `subcategory` codes
    are opaque (they do not map to any parent category), so they are NOT usable
    as MIND-style nested subtopics; the pipeline skips subtopic diversity for
    eb-nerd and the subtopic field is left as "none".

Article ids are integers in eb-nerd; we stringify them so downstream code can
treat ids uniformly across datasets.

Requires pandas + a Parquet engine (pyarrow).
"""

import pandas as pd

from datasets.common import Impression


def load_impressions(behaviors_file):
    """Read behaviors.parquet into a list of normalized Impression records."""
    df = pd.read_parquet(
        behaviors_file,
        columns=[
            "impression_id",
            "user_id",
            "impression_time",
            "article_ids_inview",
            "article_ids_clicked",
        ],
    )
    impressions = []
    for row in df.itertuples(index=False):
        candidate_ids = [str(a) for a in row.article_ids_inview]
        clicked = {str(a) for a in row.article_ids_clicked}
        labels = [1 if aid in clicked else 0 for aid in candidate_ids]
        impressions.append(
            Impression(
                int(row.impression_id),
                str(row.user_id),
                row.impression_time,
                candidate_ids,
                labels,
            )
        )
    return impressions


def load_article_meta(articles_file):
    """Read articles.parquet into {article_id: (topics, subtopic)}.

    topics   = all of the article's `topics` labels, joined with "|" into one
               group (e.g. "Crime|Violent_crime"). Whitespace inside a label is
               replaced with "_" so the whitespace-delimited user-article file
               stays parseable. "none" when the article has no topics.
    subtopic = always "none": eb-nerd's numeric subcategory codes cannot be
               mapped to a parent category, so they are not valid subtopics and
               the pipeline does not compute subtopic diversity for eb-nerd.
    """
    df = pd.read_parquet(articles_file, columns=["article_id", "topics"])
    meta = {}
    for row in df.itertuples(index=False):
        topics = row.topics
        if topics is None or len(topics) == 0:
            topics_str = "none"
        else:
            topics_str = "|".join("_".join(t.split()) for t in topics)
        meta[str(row.article_id)] = (topics_str, "none")
    return meta


def load_titles(articles_file):
    """Read articles.parquet into {article_id: title} (ids stringified)."""
    df = pd.read_parquet(articles_file, columns=["article_id", "title"])
    return {str(row.article_id): row.title for row in df.itertuples(index=False)}