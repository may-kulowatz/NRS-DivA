"""A uniform interface over the baseline recommenders.

Every recommender does the same two jobs: (1) generate its raw prediction file
and (2) build the processed per-user file the diversity metrics read. They differ
only in *how* — random/popular score candidates and write full-rank files, ground
truth writes the clicks directly, and the model recommenders train a network.

Modelling that as a small ``Recommender`` hierarchy lets callers iterate a list
instead of special-casing each one (separate branches, a "gt"/"ranks" switch, a
trainer dispatch table). Adding a recommender is a new class here.

``build_context`` is the shared entry point both command-line stages
(``python -m recommender_module`` and ``python -m diversity_module``) call to load
a dataset once and get the recommenders + the ``RunContext`` they operate on.

All file I/O is delegated to ``common/io.py`` and the ``common`` recommenders, so
this module only wires those together behind the shared contract.
"""

import importlib
import os
from dataclasses import dataclass

from config import DATASETS, DATA_ROOT, input_dir, output_dir, resolve_dataset
from recommender_module.common.ground_truth import (
    extract_ground_truth,
    save_ground_truth,
)
from recommender_module.common.random_rec import random_recommend
from recommender_module.common.popular_rec import popular_recommend
from recommender_module.common.io import (
    processed_filename,
    recommended_per_impression_from_ranks,
    recommended_per_impression_from_topk,
    save_predictions,
    save_user_article_map,
    save_user_article_map_from_ranks,
)


@dataclass
class RunContext:
    """Everything a recommender needs to generate and process its output.

    Bundled into one handle so the recommender methods take a single argument
    instead of the long, repeated list the pipeline used to thread through. The
    model-training fields are unused by the cheap recommenders.
    """
    impressions: list
    article_meta: dict
    out_dir: str          # data_processed/<dataset>/ (ground truth lives here)
    raw_dir: str          # data_processed/<dataset>/predictions/
    processed_dir: str    # data_processed/<dataset>/predictions_processed/
    seed: int = 42
    in_dir: str = None        # dataset input dir (model training)
    train_split: str = None   # training split sub-folder (model training)
    dev_split: str = None     # validation split sub-folder (model training)
    # {model_name: (module_path, fn_name)} of the per-dataset training scripts,
    # from the dataset config. Different datasets train the same model name with
    # different scripts (MIND-format vs EB-NeRD), so the trainer is dataset-driven.
    model_trainers: dict = None


class Recommender:
    """Base contract shared by every recommender.

    Subclasses set ``name`` and implement ``generate`` (write the raw prediction
    file) and ``build_user_map`` (write the processed per-user file). ``expensive``
    marks recommenders whose (re)build means training a model (needs TensorFlow),
    which the pipeline gates behind a separate policy flag.
    """
    name = None
    expensive = False

    def raw_path(self, ctx):
        """Path of this recommender's raw, full-rank prediction file."""
        return os.path.join(ctx.raw_dir, f"prediction_{self.name}.txt")

    def processed_path(self, ctx):
        """Path of this recommender's processed per-user file (diversity input)."""
        return os.path.join(ctx.processed_dir, processed_filename(self.name))

    def generate(self, ctx):
        raise NotImplementedError

    def build_user_map(self, ctx):
        raise NotImplementedError

    def recommended_by_impr(self, ctx):
        """{impr_id: [article_id, ...]} this recommender chose per impression.

        The per-impression view the normalized content-diversity metric needs (it
        scores each recommended set against its impression's candidate pool),
        unlike the per-user map which aggregates across impressions.
        """
        raise NotImplementedError


class _RankRecommender(Recommender):
    """A recommender whose processed map is built from its full-rank file.

    Shared by random, popular, and the model recommenders — they emit different
    scores but the same ``{impr_id} ... [ranks]`` format, so the per-user map is
    built identically (top-k per impression, k = clicks it received).
    """
    def build_user_map(self, ctx):
        save_user_article_map_from_ranks(
            self.raw_path(ctx), ctx.impressions, ctx.article_meta, self.processed_path(ctx)
        )

    def recommended_by_impr(self, ctx):
        return recommended_per_impression_from_ranks(self.raw_path(ctx), ctx.impressions)


class RandomRecommender(_RankRecommender):
    name = "random"

    def generate(self, ctx):
        save_predictions(random_recommend(ctx.impressions, seed=ctx.seed), self.raw_path(ctx))


class PopularRecommender(_RankRecommender):
    name = "popular"

    def generate(self, ctx):
        save_predictions(popular_recommend(ctx.impressions), self.raw_path(ctx))


class GroundTruthRecommender(Recommender):
    """The reference baseline: the articles users actually clicked.

    Not a prediction, so its raw file is ``ground_truth.txt`` at the dataset
    output root (not under predictions/) and its per-user map is built straight
    from that top-k file rather than from a full-rank ranking.
    """
    name = "ground_truth"

    def raw_path(self, ctx):
        return os.path.join(ctx.out_dir, "ground_truth.txt")

    def generate(self, ctx):
        save_ground_truth(extract_ground_truth(ctx.impressions), self.raw_path(ctx))

    def build_user_map(self, ctx):
        save_user_article_map(self.raw_path(ctx), ctx.article_meta, self.processed_path(ctx))

    def recommended_by_impr(self, ctx):
        return recommended_per_impression_from_topk(self.raw_path(ctx))


# Model name -> (module path, function) of the training script that builds its
class ModelRecommender(_RankRecommender):
    """A neural recommender trained on demand by its dataset's training script.

    ``generate`` looks up this model's training script for the current dataset
    (``ctx.model_trainers``, from the dataset config — each is a
    ``(module_path, fn_name)`` imported lazily so TensorFlow is only pulled in when
    a model is actually (re)trained) and hands it the dataset's paths. The script
    trains the model and writes the full-rank ``"{impr_id} [ranks]"`` prediction
    file. Marked ``expensive`` so the pipeline only (re)trains it when asked.
    """
    expensive = True

    def __init__(self, name):
        self.name = name

    def generate(self, ctx):
        module_path, fn_name = ctx.model_trainers[self.name]
        trainer = getattr(importlib.import_module(module_path), fn_name)
        trainer(ctx.in_dir, ctx.train_split, ctx.dev_split, self.raw_path(ctx))


def build_recommenders(model_recs):
    """Build a dataset's recommenders in scoring/display order.

    Order is random, popular, the dataset's model recommenders, then ground truth
    last (it is the reference every other system is judged against). ``model_recs``
    is the list of model names from the dataset config (empty when a dataset ships
    no models).
    """
    recs = [RandomRecommender(), PopularRecommender()]
    recs += [ModelRecommender(name) for name in model_recs]
    recs.append(GroundTruthRecommender())
    return recs


def build_context(dataset, seed=42, data_root=DATA_ROOT):
    """Load a dataset once and build the ``RunContext`` + its recommenders.

    The shared setup both command-line stages start from, so neither has to know
    how a dataset is wired together.

    PRE  : ``dataset`` is a key in ``config.DATASETS`` and its raw inputs exist
           (fetched here via the dataset's ``prepare`` hook if a known download
           source is configured; otherwise they must already be on disk — run
           ``python -m dataset_module`` first). The behaviors + articles files are
           read into memory.
    POST : the output folders ``data_processed/<dataset>/predictions`` and
           ``.../predictions_processed`` exist. Returns ``(cfg, ctx, recommenders)``
           where ``recommenders`` is in scoring order. No prediction or score files
           are written.

    Raises ``ValueError`` for an unknown dataset name. The name is matched
    case-insensitively (``MIND`` / ``mind`` / the folder name all resolve).
    """
    dataset = resolve_dataset(dataset)
    cfg = DATASETS[dataset]
    adapter = cfg["adapter"]

    in_dir = input_dir(dataset, data_root)
    out_dir = output_dir(dataset, data_root)

    # Make sure the dataset's essential raw inputs exist before anything reads
    # them (the optional content-diversity bundle is fetched later, lazily).
    cfg["prepare"].ensure_raw_data(in_dir)

    behaviors_file = os.path.join(in_dir, *cfg["behaviors"])
    articles_file = os.path.join(in_dir, *cfg["articles"])

    raw_dir = os.path.join(out_dir, "predictions")
    processed_dir = os.path.join(out_dir, "predictions_processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    ctx = RunContext(
        impressions=adapter.load_impressions(behaviors_file),
        article_meta=adapter.load_article_meta(articles_file),
        out_dir=out_dir, raw_dir=raw_dir, processed_dir=processed_dir,
        seed=seed, in_dir=in_dir,
        train_split=cfg.get("train_split"), dev_split=cfg["behaviors"][0],
        model_trainers=cfg.get("model_trainers", {}),
    )
    return cfg, ctx, build_recommenders(cfg["model_recs"])