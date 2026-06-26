"""Build the ``mind_news`` dataset: a news-only subset of MIND.

The preparation analog of ``mind_news/adapter`` — it *derives* the mind_news
files the adapter parses. ``mind_news`` mirrors MIND's layout — a
``MINDnews_train`` and a ``MINDnews_dev`` folder, each holding ``news.tsv`` +
``behaviors.tsv`` — but keeps far less of the data. It is the slice of MIND that
is purely about the *news* category:

  * Only impressions in which the user actually **clicked at least one article
    whose category is "news"** are kept.
  * Among those, only impressions that show **at least two "news" candidates**
    survive (a single candidate has nothing to be diverse against).
  * Every non-news article is **stripped from the impression's candidate list and
    from the user's reading history**, so the only article ids that remain refer
    to news articles.
  * ``news.tsv`` is reduced to just the news-category rows.

The result is a smaller dataset, in the exact MIND format, that the rest of the
pipeline can read through the mind_news adapter (see the ``mind_news`` entry in
``config.DATASETS``).

The source MIND splits live under ``data/datasets/mind`` (``MINDsmall_train`` /
``MINDsmall_dev``); the subset is written under ``data/datasets/mind_news``.

Like the other ``dataset_module`` prepare modules it exposes ``ensure_raw_data``
and ``ensure_utils``; both derive what they need from the sibling ``mind``
dataset (via ``mind/prepare``), so mind_news ends up fully self-contained.
"""

import os
import shutil

from dataset_module.common import default_input_dir

# The dataset's folder name under data/datasets/ (its default standalone location).
DIR = "mind_news"

# MIND's parent category we keep; everything else is dropped.
NEWS_CATEGORY = "news"

# (source split dir under data/datasets/mind, destination dir under data/datasets/mind_news)
_SPLITS = (
    ("MINDsmall_train", "MINDnews_train"),
    ("MINDsmall_dev", "MINDnews_dev"),
)

_SPLIT_FILES = ("news.tsv", "behaviors.tsv")


def _news_ids(news_file):
    """Return the set of news_ids whose category (column 2) is ``news``."""
    ids = set()
    with open(news_file, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if cols[1] == NEWS_CATEGORY:
                ids.add(cols[0])
    return ids


def _filter_news_file(src_news, dst_news, news_ids):
    """Copy only the news-category rows of ``src_news`` into ``dst_news``."""
    with open(src_news, encoding="utf-8") as fin, \
            open(dst_news, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.split("\t", 1)[0] in news_ids:
                fout.write(line)


def _filter_behaviors_file(src_behaviors, dst_behaviors, news_ids):
    """Write the news-only subset of a behaviors.tsv file.

    behaviors.tsv columns: impr_id, user_id, time, history, impressions
    where ``impressions`` is space-separated ``Nxxxx-1`` (clicked) / ``Nxxxx-0``.

    An impression is kept only when, restricted to its news candidates, it still
    has at least two of them and at least one was clicked. The candidate list is
    rewritten to the news candidates (labels preserved, original order kept) and
    the history to its news-only ids. Returns the number of impressions kept.

    Kept impressions are renumbered ``1..N`` in output order: the MIND model
    scripts key their predictions by an impression's line position, and MIND
    itself numbers the impr_id column that way, so following the same convention
    keeps mind_news a drop-in independent dataset.
    """
    kept = 0
    with open(src_behaviors, encoding="utf-8") as fin, \
            open(dst_behaviors, "w", encoding="utf-8") as fout:
        for line in fin:
            cols = line.rstrip("\n").split("\t")
            _orig_impr_id, user_id, time, history, impressions = cols[:5]

            news_cands = [
                c for c in impressions.split() if c.rsplit("-", 1)[0] in news_ids
            ]
            if len(news_cands) < 2:
                continue
            if not any(c.rsplit("-", 1)[1] == "1" for c in news_cands):
                continue

            kept += 1
            news_history = " ".join(h for h in history.split() if h in news_ids)
            fout.write(
                "\t".join([str(kept), user_id, time, news_history, " ".join(news_cands)])
                + "\n"
            )
    return kept


def build_mind_news(mind_dir, out_dir):
    """Build the ``mind_news`` splits under ``out_dir`` from the MIND ``mind_dir``.

    ``mind_dir`` holds the source MINDsmall_train / MINDsmall_dev folders;
    ``out_dir`` is where MINDnews_train / MINDnews_dev are written.
    """
    # Make sure the source dev split exists, fetching it if missing (the train
    # split has no auto-download, so it must already be present).
    from dataset_module.mind import prepare as mind_prepare
    mind_prepare.ensure_raw_data(mind_dir)

    for src, dst in _SPLITS:
        src_dir = os.path.join(mind_dir, src)
        src_news = os.path.join(src_dir, "news.tsv")
        src_behaviors = os.path.join(src_dir, "behaviors.tsv")
        if not (os.path.exists(src_news) and os.path.exists(src_behaviors)):
            raise FileNotFoundError(
                f"MIND source split missing: {src_dir}. mind_news is derived from "
                "the MIND 'small' splits, so both MINDsmall_train and MINDsmall_dev "
                "must be present under data/datasets/mind."
            )

        dst_dir = os.path.join(out_dir, dst)
        os.makedirs(dst_dir, exist_ok=True)

        ids = _news_ids(src_news)
        _filter_news_file(src_news, os.path.join(dst_dir, "news.tsv"), ids)
        kept = _filter_behaviors_file(
            src_behaviors, os.path.join(dst_dir, "behaviors.tsv"), ids
        )
        print(f"  {dst}: kept {kept} impressions, {len(ids)} news articles")


def ensure_raw_data(in_dir):
    """Ensure mind_news's splits exist under ``in_dir``, building them if missing.

    ``in_dir`` is the dataset's input directory (e.g. data/datasets/mind_news).
    The subset is derived from the sibling ``mind`` dataset directory. Returns
    True if a build happened, False if everything was already present.
    """
    needed = [
        os.path.join(in_dir, dst, f)
        for _, dst in _SPLITS
        for f in _SPLIT_FILES
    ]
    if all(os.path.exists(p) for p in needed):
        return False

    mind_dir = os.path.join(os.path.dirname(in_dir), "mind")
    print(f"mind_news splits missing — building from {mind_dir} ...")
    build_mind_news(mind_dir, in_dir)
    return True


# If either of these is present we assume mind_news already has its own utils.
_UTILS_REQUIRED = ("embedding.npy", "word_dict.pkl")


def ensure_utils(in_dir):
    """Ensure mind_news has its own ``utils`` bundle, copying MIND's if missing.

    ``in_dir`` is the mind_news input dir (data/datasets/mind_news). mind_news is
    treated as a fully independent dataset, so it keeps its own copy of the utils
    (word embeddings, dictionaries, model .yaml configs) under ``in_dir/utils``
    rather than reaching into the sibling MIND folder. The bundle is vocabulary-
    level and identical to MIND's, so it is copied from there (downloading the
    MIND utils first if they aren't present). Returns True if a copy happened.
    """
    utils_dir = os.path.join(in_dir, "utils")
    if all(os.path.exists(os.path.join(utils_dir, f)) for f in _UTILS_REQUIRED):
        return False

    from dataset_module.mind import prepare as mind_prepare
    mind_dir = os.path.join(os.path.dirname(in_dir), "mind")
    mind_prepare.ensure_utils(mind_dir)  # make sure the source bundle exists (download if needed)

    print(f"Copying utils bundle into {utils_dir} ...")
    shutil.copytree(os.path.join(mind_dir, "utils"), utils_dir, dirs_exist_ok=True)
    return True


if __name__ == "__main__":
    _in_dir = default_input_dir(DIR)
    print(f"Preparing mind_news in {_in_dir} ...")
    ensure_raw_data(_in_dir)
    ensure_utils(_in_dir)
