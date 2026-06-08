"""Explore the eb-nerd dataset.

TODO: Delete in the end! Not really needed for the project. Was mainly generated to find differences between ebnerd and MIND.

eb-nerd (Ekstra Bladet News Recommendation Dataset) is the second news source
we want to add to EchoBench. Before wiring it into the pipeline this script just
*logs* what the data looks like, and — importantly — calls out where it differs
from the MIND dataset the pipeline is currently built around.

Run with:
    python explore_ebnerd.py

Requires pandas + pyarrow (parquet engine).
"""

import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("explore_ebnerd")

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_EBNERD_DIR = os.path.join(_PROJECT_DIR, "data", "ebnerd")

# How many sample values / rows to print so the log stays readable.
_PREVIEW = 60


def _section(title):
    log.info("\n" + "=" * 78)
    log.info(title)
    log.info("=" * 78)


def _short(value):
    """Trim long text fields (article body, etc.) so the log stays readable."""
    text = str(value).replace("\n", " ")
    return text if len(text) <= _PREVIEW else text[:_PREVIEW] + "..."


def _describe(name, df):
    log.info("\n%s - %d rows x %d columns", name, len(df), df.shape[1])
    log.info("%-26s %-18s %s", "column", "dtype", "example value")
    log.info("%-26s %-18s %s", "-" * 26, "-" * 18, "-" * 30)
    first = df.iloc[0]
    for col in df.columns:
        log.info("%-26s %-18s %s", col, str(df[col].dtype), _short(first[col]))


def explore_articles():
    _section("ARTICLES  (data/ebnerd/articles.parquet)")
    df = pd.read_parquet(os.path.join(_EBNERD_DIR, "articles.parquet"))
    _describe("articles", df)

    log.info("\nUnique articles:        %d", df["article_id"].nunique())
    log.info("Article types:          %s", sorted(df["article_type"].dropna().unique()))
    log.info("Categories (numeric):   %d distinct codes", df["category"].nunique())
    cat_examples = (
        df[["category", "category_str"]].drop_duplicates().head(10).values.tolist()
    )
    log.info("Category code -> name:   %s", cat_examples)
    log.info("Sentiment labels:       %s", sorted(df["sentiment_label"].dropna().unique()))
    log.info("Sample titles:          %s", [_short(t) for t in df["title"].head(3)])
    return df


def explore_behaviors_and_history():
    _section("BEHAVIORS + HISTORY  (data/ebnerd/{train,validation}/)")
    for split in ("train", "validation"):
        behaviors = pd.read_parquet(
            os.path.join(_EBNERD_DIR, split, "behaviors.parquet")
        )
        history = pd.read_parquet(os.path.join(_EBNERD_DIR, split, "history.parquet"))

        _describe(f"{split}/behaviors", behaviors)
        log.info("  unique users:        %d", behaviors["user_id"].nunique())
        log.info("  unique impressions:  %d", behaviors["impression_id"].nunique())
        log.info(
            "  candidates/impr.:    avg %.1f (article_ids_inview)",
            behaviors["article_ids_inview"].map(len).mean(),
        )
        log.info(
            "  clicks/impr.:        avg %.2f (article_ids_clicked)",
            behaviors["article_ids_clicked"].map(len).mean(),
        )

        _describe(f"{split}/history", history)
        log.info("  unique users:        %d", history["user_id"].nunique())
        log.info(
            "  history length:      avg %.1f articles/user (article_id_fixed)",
            history["article_id_fixed"].map(len).mean(),
        )


def log_differences_from_mind():
    _section("KEY DIFFERENCES FROM MIND")
    diffs = [
        ("File format",
         "eb-nerd ships Parquet (articles/behaviors/history.parquet); "
         "MIND ships TSV (news.tsv, behaviors.tsv)."),
        ("Language",
         "eb-nerd is Danish (Ekstra Bladet); MIND is English (Microsoft News)."),
        ("Article IDs",
         "eb-nerd article_id is an INTEGER (e.g. 3001353); "
         "MIND news IDs are STRINGS (e.g. 'N55528')."),
        ("Categories",
         "eb-nerd has a numeric `category` code plus `category_str`, and "
         "`subcategory` is a LIST of codes; MIND has single string "
         "category/subcategory columns."),
        ("Article richness",
         "eb-nerd articles include full `body`, `sentiment_score/label`, "
         "`topics`, `total_inviews/pageviews/read_time`, premium flag, etc. "
         "MIND news.tsv only has title, abstract, url and entity annotations."),
        ("Impressions vs history",
         "eb-nerd SPLITS interactions into behaviors.parquet (one row per "
         "impression: article_ids_inview = candidates, article_ids_clicked = "
         "clicks) and a SEPARATE history.parquet (per-user click history in "
         "*_fixed arrays). MIND packs both the click history and the labelled "
         "impressions into a single behaviors.tsv row."),
        ("Click labels",
         "eb-nerd lists clicked IDs explicitly in `article_ids_clicked`; "
         "MIND encodes labels inline as 'Nxxxx-1' / 'Nxxxx-0' per candidate."),
        ("User signals",
         "eb-nerd behaviors carry rich signals: read_time, scroll_percentage, "
         "device_type, gender, age, postcode, is_subscriber. MIND has none of "
         "these - only user id, time, history and impressions."),
        ("Pre-built assets",
         "MIND ships utils/ with embedding.npy + word_dict.pkl that the content "
         "diversity score relies on. eb-nerd ships no such embeddings/dicts - "
         "these would need to be built before content diversity can run."),
        ("Train/validation split",
         "eb-nerd is pre-split into train/ and validation/ folders, each with "
         "its own behaviors + history. MIND uses MINDsmall_train / MINDsmall_dev."),
    ]
    for topic, text in diffs:
        log.info("\n- %s:\n    %s", topic, text)


def main():
    if not os.path.isdir(_EBNERD_DIR):
        log.error("eb-nerd directory not found at %s", _EBNERD_DIR)
        return
    explore_articles()
    explore_behaviors_and_history()
    log_differences_from_mind()


if __name__ == "__main__":
    main()