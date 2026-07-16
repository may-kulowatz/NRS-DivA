"""Dataset registry and path helpers.

The single place that describes every dataset the tool knows about — its on-disk
folder, which adapter parses it, where its raw inputs live, and which optional
steps apply. Kept separate from the run machinery (``recommender_module`` /
``diversity_module``) and the run-manifest I/O (``run_manifest.py``) so that read-only
consumers (e.g. the dashboard) can import the configuration on its own.
"""

import os

# Each dataset is its own package under dataset_module, holding an `adapter`
# (parses the raw format) and a `prepare` module. The prepare modules are wired
# in via the "prepare" hook below: each exposes ensure_raw_data(in_dir) (the
# essential inputs) and ensure_utils(in_dir) (the optional content-diversity
# bundle), mirroring how each adapter exposes the same load functions.
from dataset_module.mind import adapter as mind_adapter, prepare as mind_prepare
from dataset_module.ebnerd import adapter as ebnerd_adapter, prepare as ebnerd_prepare
from dataset_module.mind_news import adapter as mind_news_adapter, prepare as mind_news_prepare


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
# Each entry describes a dataset's on-disk folder name ("dir"), where its raw
# input files live (relative to data/datasets/<dir>/) and which optional steps
# apply to it. Paths are tuples joined with os.path.join so they work on any
# platform. Generated outputs go to data/data_processed/<dir>/ (see output_dir).
DATASETS = {
    "MIND": {
        "dir": "mind",
        "adapter": mind_adapter,
        "behaviors": ("MINDsmall_dev", "behaviors.tsv"),
        "articles": ("MINDsmall_dev", "news.tsv"),
        "model_recs": ["nrms", "lstur", "naml"],
        "model_trainers": {
            "nrms": ("recommender_module.mind_specific.nrms_mind", "run"),
            "lstur": ("recommender_module.mind_specific.lstur_mind", "run"),
            "naml": ("recommender_module.mind_specific.naml_mind", "run"),
        },
        # Training split the model scripts read when a prediction file must be
        # (re)built; the dev split is taken from "behaviors" above.
        "train_split": "MINDsmall_train",
        "content_diversity": {
            "kind": "word_average",
            "embedding": ("utils", "embedding.npy"),
            "word_dict": ("utils", "word_dict.pkl"),
        },
        # Extra word-average content spaces built from a different news.tsv text
        # column than the primary (title) one, as name -> (column_index, label).
        # Each adds content_diversity_<name> (+ _normalized_<name>), computed
        # exactly like the primary title space but averaging that column's word
        # embeddings, so title- and abstract-based diversity can be compared.
        "content_text_variants": {
            "abstract": (4, "abstract"),
        },
        "prepare": mind_prepare,
    },
    "ebnerd": {
        "dir": "ebnerd",
        "adapter": ebnerd_adapter,
        "behaviors": ("validation", "behaviors.parquet"),
        "articles": ("articles.parquet",),
        # NRMS / LSTUR, trained on demand by the EB-NeRD-specific scripts (which
        # write the same "{impr_id} [ranks]" prediction file MIND's models do).
        # Topic diversity also uses the multi-valued `topics` field.
        "model_recs": ["nrms", "lstur", "naml"],
        "model_trainers": {
            "nrms": ("recommender_module.ebnerd_specific.nrms_ebnerd", "run"),
            "lstur": ("recommender_module.ebnerd_specific.lstur_ebnerd", "run"),
            "naml": ("recommender_module.ebnerd_specific.naml_ebnerd", "run"),
        },
        "train_split": "train",
        "content_diversity": {
            "kind": "precomputed",
            "vectors": ("contrastive_vector.parquet",),
        },
        # Extra precomputed article-embedding spaces EB-NeRD ships, as
        # name -> (parquet_file, vector_column). Each adds its own
        # content_diversity_<name> (+ _normalized_<name>) measures alongside the
        # primary contrastive one above, so diversity can be compared across
        # representations. All keyed by article_id, one vector per article.
        "content_embeddings": {
            "xlmr": ("xlm_roberta_base.parquet", "FacebookAI/xlm-roberta-base"),
            "bert": ("bert_base_multilingual_cased.parquet",
                     "google-bert/bert-base-multilingual-cased"),
            "docvec": ("document_vector.parquet", "document_vector"),
        },
        # Verifies inputs are present; ensure_utils is a no-op (the contrastive
        # vectors are shipped with the dataset, not fetched).
        "prepare": ebnerd_prepare,
    },
    "mind_news": {
        "dir": "mind_news",
        # Every article is in the "news" category, so the adapter promotes each
        # article's subcategory into the topic slot — topic diversity then measures
        # variety among news subcategories.
        "adapter": mind_news_adapter,
        # A news-only subset of MIND built by mind_news/prepare: every impression
        # kept clicked at least one "news" article and showed >=2 news candidates,
        # with all non-news articles stripped from the candidates and history.
        "behaviors": ("MINDnews_dev", "behaviors.tsv"),
        "articles": ("MINDnews_dev", "news.tsv"),
        # NRMS / LSTUR, retrained on mind_news by the same scripts MIND uses (the
        # pipeline hands them the mind_news paths). Their full-rank prediction
        # files aren't shipped; the pipeline builds them on demand.
        "model_recs": ["nrms", "lstur", "naml"],
        "model_trainers": {
            "nrms": ("recommender_module.mind_specific.nrms_mind", "run"),
            "lstur": ("recommender_module.mind_specific.lstur_mind", "run"),
            "naml": ("recommender_module.mind_specific.naml_mind", "run"),
        },
        "train_split": "MINDnews_train",
        # mind_news keeps its own copy of the utils bundle (see "prepare"), so its
        # embeddings are read from the dataset's own utils/ like MIND's.
        "content_diversity": {
            "kind": "word_average",
            "embedding": ("utils", "embedding.npy"),
            "word_dict": ("utils", "word_dict.pkl"),
        },
        # As for MIND: an abstract-based word-average space alongside the primary
        # title one (name -> (news.tsv column_index, label)). mind_news shares
        # MIND's news.tsv layout, so the abstract is column 4 here too.
        "content_text_variants": {
            "abstract": (4, "abstract"),
        },
        # Prepares mind_news: ensure_raw_data builds the splits from the sibling
        # MIND data; ensure_utils builds mind_news's own copy of the utils bundle
        # on demand before content diversity reads it.
        "prepare": mind_news_prepare,
    },
}


_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(_PROJECT_DIR, "data")


def resolve_dataset(name):
    """Return the canonical ``DATASETS`` key for ``name``, case-insensitively.

    Accepts the registry key (``MIND``), any casing of it (``mind``), or the
    on-disk folder name (``dir``), so every stage takes the same spelling and
    users don't have to remember that prepare used the folder name. Raises
    ``ValueError`` for an unknown dataset.
    """
    if name in DATASETS:
        return name
    aliases = {k.lower(): k for k in DATASETS}
    aliases.update({cfg["dir"].lower(): k for k, cfg in DATASETS.items()})
    key = aliases.get(name.lower()) if isinstance(name, str) else None
    if key is None:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(DATASETS)}")
    return key


def input_dir(dataset, data_root=DATA_ROOT):
    """Directory holding a dataset's raw input files."""
    return os.path.join(data_root, "datasets", DATASETS[resolve_dataset(dataset)]["dir"])


def output_dir(dataset, data_root=DATA_ROOT):
    """Directory holding a dataset's generated outputs."""
    return os.path.join(data_root, "data_processed", DATASETS[resolve_dataset(dataset)]["dir"])