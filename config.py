"""Dataset registry and path helpers.

The single place that describes every dataset the pipeline knows about — its
on-disk folder, which adapter parses it, where its raw inputs live, and which
optional steps apply. Kept separate from the orchestration (``pipeline.py``) and
the diversity-score computation (``scores.py``) so that read-only consumers (e.g.
the dashboard) can import the configuration without pulling in the run machinery.
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
        # MIND ships pre-computed NRMS and LSTUR prediction files plus the
        # embeddings/word-dict that content diversity needs.
        "model_recs": ["nrms", "lstur", "naml"],
        # Per-model training scripts (module, function), imported lazily when a
        # prediction must be (re)built. MIND-format datasets use the mind_specific
        # scripts. (NRMS/LSTUR prediction files are shipped; NAML is trained on
        # demand.)
        "model_trainers": {
            "nrms": ("recommender_module.mind_specific.nrms_mind", "run"),
            "lstur": ("recommender_module.mind_specific.lstur_mind", "run"),
            "naml": ("recommender_module.mind_specific.naml_mind", "run"),
        },
        # Training split the model scripts read when a prediction file must be
        # (re)built; the dev split is taken from "behaviors" above.
        "train_split": "MINDsmall_train",
        # "word_average": each article vector is the mean of its title's word
        # embeddings, built from the (gitignored) utils bundle.
        "content_diversity": {
            "kind": "word_average",
            "embedding": ("utils", "embedding.npy"),
            "word_dict": ("utils", "word_dict.pkl"),
        },
        # Prepares MIND: ensure_raw_data fetches the dev split; ensure_utils
        # fetches the (gitignored) embeddings/dicts on demand before content
        # diversity reads them.
        "prepare": mind_prepare,
    },
    "ebnerd": {
        "dir": "ebnerd",
        "adapter": ebnerd_adapter,
        "behaviors": ("validation", "behaviors.parquet"),
        "articles": ("articles.parquet",),
        # NRMS / LSTUR, trained on demand by the eb-nerd-specific scripts (which
        # write the same "{impr_id} [ranks]" prediction file MIND's models do).
        # Topic diversity also uses the multi-valued `topics` field.
        "model_recs": ["nrms", "lstur"],
        "model_trainers": {
            "nrms": ("recommender_module.ebnerd_specific.nrms_ebnerd", "run"),
            "lstur": ("recommender_module.ebnerd_specific.lstur_ebnerd", "run"),
        },
        # Split sub-folders under data/datasets/ebnerd/ (each holds
        # behaviors.parquet + history.parquet); the dev split is "behaviors" above.
        "train_split": "train",
        # "precomputed": one ready-made document embedding per article, read
        # straight from contrastive_vector.parquet (768-dim contrastive vectors).
        "content_diversity": {
            "kind": "precomputed",
            "vectors": ("contrastive_vector.parquet",),
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
        # Prepares mind_news: ensure_raw_data builds the splits from the sibling
        # MIND data; ensure_utils builds mind_news's own copy of the utils bundle
        # on demand before content diversity reads it.
        "prepare": mind_news_prepare,
    },
}


_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(_PROJECT_DIR, "data")


def input_dir(dataset, data_root=DATA_ROOT):
    """Directory holding a dataset's raw input files."""
    return os.path.join(data_root, "datasets", DATASETS[dataset]["dir"])


def output_dir(dataset, data_root=DATA_ROOT):
    """Directory holding a dataset's generated outputs."""
    return os.path.join(data_root, "data_processed", DATASETS[dataset]["dir"])