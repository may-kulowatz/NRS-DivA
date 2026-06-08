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

Parquet is read via pyarrow directly (not pandas.read_parquet). Going through
pandas' parquet engine can re-register pandas' pyarrow extension types and
raise "A type extension with name pandas.period already defined" when modules
are re-imported (e.g. Solara's dev-server hot-reload). Reading with pyarrow
avoids that path entirely.

Requires pyarrow.
"""

import pyarrow.parquet as pq

from datasets.common import Impression


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
    """Read articles.parquet into {article_id: (topics, subtopic)}.

    topics   = all of the article's `topics` labels, joined with "|" into one
               group (e.g. "Crime|Violent_crime"). Whitespace inside a label is
               replaced with "_" so the whitespace-delimited user-article file
               stays parseable. "none" when the article has no topics.
    subtopic = always "none": eb-nerd's numeric subcategory codes cannot be
               mapped to a parent category, so they are not valid subtopics and
               the pipeline does not compute subtopic diversity for eb-nerd.
    """
    cols = _read_columns(articles_file, ["article_id", "topics"])
    meta = {}
    for article_id, topics in zip(cols["article_id"], cols["topics"]):
        if topics is None or len(topics) == 0:
            topics_str = "none"
        else:
            topics_str = "|".join("_".join(t.split()) for t in topics)
        meta[str(article_id)] = (topics_str, "none")
    return meta


def load_titles(articles_file):
    """Read articles.parquet into {article_id: title} (ids stringified)."""
    cols = _read_columns(articles_file, ["article_id", "title"])
    return {str(aid): title for aid, title in zip(cols["article_id"], cols["title"])}
