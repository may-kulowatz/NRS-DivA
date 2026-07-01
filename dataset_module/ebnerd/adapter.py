"""EB-NeRD (Ekstra Bladet) dataset adapter.

Parses EB-NeRD's parquet files into the same normalized structures the MIND
adapter produces. All eb-nerd-specific format knowledge lives here:

  * behaviors.parquet — one row per impression. Candidates are in
    `article_ids_inview`; clicked articles are listed explicitly in
    `article_ids_clicked`.
  * articles.parquet — `topics` is a LIST of human-readable topic labels per
    article (e.g. ["Crime", "Violent crime"]); all of them are kept so topic
    diversity can account for multiple topics.

Article ids are integers in EB-NeRD; we stringify them so downstream code can
treat ids uniformly across datasets.

Requires pyarrow.
"""

import pyarrow.parquet as pq

from dataset_module.common import Impression


def _read_columns(path, columns):
    """Read selected columns of a Parquet file as {column: python_list}."""
    table = pq.read_table(path, columns=columns)
    return {name: table.column(name).to_pylist() for name in columns}


def load_impressions(behaviors_file):
    """Read behaviors.parquet into a list of normalized Impression records."""
    cols = _read_columns(
        behaviors_file,
        [
            "impression_id",
            "user_id",
            "impression_time",
            "article_ids_inview",
            "article_ids_clicked",
        ],
    )
    impressions = []
    for impr_id, user_id, timestamp, inview, clicked in zip(
        cols["impression_id"],
        cols["user_id"],
        cols["impression_time"],
        cols["article_ids_inview"],
        cols["article_ids_clicked"],
    ):
        candidate_ids = [str(a) for a in inview]
        clicked_set = {str(a) for a in clicked}
        labels = [1 if aid in clicked_set else 0 for aid in candidate_ids]
        impressions.append(
            Impression(int(impr_id), str(user_id), timestamp, candidate_ids, labels)
        )
    return impressions


def load_article_meta(articles_file):
    """Read articles.parquet into {article_id: topic}.

    topic = all of the article's `topics` labels, joined with "|" into one group
            (e.g. "Crime|Violent_crime"). Whitespace inside a label is replaced
            with "_" so the whitespace-delimited user-article file stays
            parseable. "none" when the article has no topics.
    """
    cols = _read_columns(articles_file, ["article_id", "topics"])
    meta = {}
    for article_id, topics in zip(cols["article_id"], cols["topics"]):
        if topics is None or len(topics) == 0:
            topics_str = "none"
        else:
            topics_str = "|".join("_".join(t.split()) for t in topics)
        meta[str(article_id)] = topics_str
    return meta


def load_titles(articles_file):
    """Read articles.parquet into {article_id: title} (ids stringified)."""
    cols = _read_columns(articles_file, ["article_id", "title"])
    return {str(aid): title for aid, title in zip(cols["article_id"], cols["title"])}
