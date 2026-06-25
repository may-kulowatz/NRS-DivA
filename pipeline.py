"""End-to-end pipeline: load a dataset, run the baseline recommenders, write
prediction / processed files, and report diversity scores.

"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_module import mind_adapter, ebnerd_adapter, mind_news_adapter
from recommender_module.common.ground_truth import extract_ground_truth, save_ground_truth
from recommender_module.common.random_rec import random_recommend
from recommender_module.common.popular_rec import popular_recommend
from recommender_module.common.io import (
    processed_filename,
    save_predictions,
    save_user_article_map,
    save_user_article_map_from_ranks,
)
# Raw-data fetchers: ensure_raw_data guarantees the essential inputs up front;
# ensure_mind_utils is wired in per-dataset via the "prepare" config hook for the
# optional (content-diversity) embeddings.
from prepare import ensure_raw_data, ensure_mind_utils
from prepare_mind_news import ensure_utils as ensure_mind_news_utils
from diversity_module.topic_diversity import topic_diversity
from diversity_module.content_diversity import (
    content_diversity,
    load_news_embeddings,
    load_precomputed_embeddings,
)


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
        "model_recs": ["nrms", "lstur"],
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
        # Fetches the (gitignored) embeddings/dicts on demand before content
        # diversity reads them. Called with the dataset's input_dir.
        "prepare": ensure_mind_utils,
    },
    "ebnerd": {
        "dir": "ebnerd",
        "adapter": ebnerd_adapter,
        "behaviors": ("validation", "behaviors.parquet"),
        "articles": ("articles.parquet",),
        # eb-nerd ships no model prediction files, so the model recommenders are
        # skipped. Topic diversity uses the multi-valued `topics` field instead.
        "model_recs": [],
        # "precomputed": one ready-made document embedding per article, read
        # straight from contrastive_vector.parquet (768-dim contrastive vectors).
        "content_diversity": {
            "kind": "precomputed",
            "vectors": ("contrastive_vector.parquet",),
        },
        # contrastive_vector.parquet is shipped with the dataset, not fetched.
        "prepare": None,
    },
    "mind_news": {
        "dir": "mind_news",
        # Every article is in the "news" category, so the adapter promotes each
        # article's subcategory into the topic slot — topic diversity then measures
        # variety among news subcategories.
        "adapter": mind_news_adapter,
        # A news-only subset of MIND built by prepare_mind_news: every impression
        # kept clicked at least one "news" article and showed >=2 news candidates,
        # with all non-news articles stripped from the candidates and history.
        "behaviors": ("MINDnews_dev", "behaviors.tsv"),
        "articles": ("MINDnews_dev", "news.tsv"),
        # NRMS / LSTUR, retrained on mind_news by the same scripts MIND uses (the
        # pipeline hands them the mind_news paths). Their full-rank prediction
        # files aren't shipped; the pipeline builds them on demand.
        "model_recs": ["nrms", "lstur"],
        "train_split": "MINDnews_train",
        # mind_news keeps its own copy of the utils bundle (see "prepare"), so its
        # embeddings are read from the dataset's own utils/ like MIND's.
        "content_diversity": {
            "kind": "word_average",
            "embedding": ("utils", "embedding.npy"),
            "word_dict": ("utils", "word_dict.pkl"),
        },
        # Builds mind_news's own utils bundle on demand before content diversity
        # reads it (the dataset splits themselves are built by ensure_raw_data).
        "prepare": ensure_mind_news_utils,
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


# Model name -> (module path, function) of the training script that builds its
# prediction file. Imported lazily and only when a prediction file is missing,
# because the scripts pull in TensorFlow + the recommenders library.
_MODEL_TRAINERS = {
    "nrms": ("recommender_module.mind_specific.nrms_mind", "run"),
    "lstur": ("recommender_module.mind_specific.lstur_mind", "run"),
}


def _train_model(name, dataset_dir, train_split, dev_split, prediction_file):
    """Hand a dataset's paths to a model's training script to build its
    prediction file. Returns True if the file now exists."""
    import importlib
    module_path, fn_name = _MODEL_TRAINERS[name]
    trainer = getattr(importlib.import_module(module_path), fn_name)
    trainer(dataset_dir, train_split, dev_split, prediction_file)
    return os.path.exists(prediction_file)


def _stale(output, *inputs):
    """True if output is missing or older than any existing input.

    Lets a derived file be rebuilt when its source changed, instead of being
    skipped just because some version of it already exists.
    """
    if not os.path.exists(output):
        return True
    out_mtime = os.path.getmtime(output)
    return any(os.path.exists(i) and os.path.getmtime(i) > out_mtime for i in inputs)


# ---------------------------------------------------------------------------
# Diversity-score cache
# ---------------------------------------------------------------------------
# Computing the diversity scores is the only pipeline step with no file output,
# so without a cache it re-runs on every invocation even when nothing changed.
# We persist scores to predictions/diversity_scores.json, keyed per recommender
# and metric, alongside a signature of the user-article file the score was
# computed from. A score is reused as long as that signature still matches.
def _file_sig(path):
    """Cheap change-signature for a file: modification time + size."""
    st = os.stat(path)
    return f"{st.st_mtime_ns}-{st.st_size}"


def _load_score_cache(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_score_cache(path, cache):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


class MetricUnavailable(Exception):
    """A metric's inputs couldn't be obtained (e.g. embeddings can't be
    downloaded). The metric is skipped for the affected runs rather than failing
    the whole pipeline."""


def _compute_run_scores(path, metric_defs, prev_cache, force=False):
    """Compute (or reuse) every metric for one user-article file.

    metric_defs : list of (metric_key, fn) where fn(path) -> float.
    prev_cache  : the cached {metric_key: {"value", "sig"}} for this run.
    force       : when True, recompute every metric even if the cached signature
                  still matches (the interactive "recalculate diversity" option).

    Returns (scores, cache, n_computed): scores is {metric_key: value} for
    display; cache is the refreshed {metric_key: {"value", "sig"}} to persist;
    n_computed counts how many metrics were actually (re)calculated. A metric
    whose fn raises MetricUnavailable is skipped (left out of scores/cache) so
    the others still compute.
    """
    sig = _file_sig(path)
    scores, cache, n_computed = {}, {}, 0
    for key, fn in metric_defs:
        prev = prev_cache.get(key)
        if not force and prev is not None and prev.get("sig") == sig:
            value = prev["value"]
        else:
            try:
                value = fn(path)
            except MetricUnavailable as exc:
                print(f"  Skipping {key}: {exc}")
                continue
            n_computed += 1
        scores[key] = value
        cache[key] = {"value": value, "sig": sig}
    return scores, cache, n_computed


def run_pipeline(dataset="MIND", seed=42, data_root=DATA_ROOT, *,
                 force_recommenders=None, force_processed=False,
                 force_diversity=False, generate_missing=True, train_missing=True):
    """Run a dataset's recommenders + diversity scoring.

    By default a recommender's prediction is (re)built only when missing and its
    processed file / diversity score only when out of date — the cheap "just make
    sure everything's there" behaviour used by code and tests. The keyword flags
    give the interactive front-end finer control:

    force_recommenders : iterable of recommender names whose raw predictions are
                         regenerated even if they already exist (model recs are
                         retrained — needs TensorFlow).
    force_processed    : rebuild every processed per-user file, not just stale ones.
    force_diversity    : recompute diversity scores even when the cache is valid.
    generate_missing   : auto-generate a *cheap* recommender (ground truth, random,
                         popular) when its file is missing. Set False to only build
                         what force_recommenders asks for.
    train_missing      : auto-train a *model* recommender when its prediction file
                         is missing. Set False to only train what's forced.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(DATASETS)}")
    cfg = DATASETS[dataset]
    adapter = cfg["adapter"]
    force_recommenders = set(force_recommenders or ())

    in_dir = input_dir(dataset, data_root)
    out_dir = output_dir(dataset, data_root)

    # Make sure the dataset's essential raw inputs exist before anything reads
    # them, fetching whatever is missing (prepare.ensure_raw_data). The optional
    # MIND embeddings are fetched later, lazily, via the "prepare" hook.
    ensure_raw_data(dataset, in_dir)

    behaviors_file = os.path.join(in_dir, *cfg["behaviors"])
    articles_file = os.path.join(in_dir, *cfg["articles"])

    # Outputs split into two parallel folders under data_processed/<dir>/:
    #   predictions/            — full-rank output each recommender emits directly
    #   predictions_processed/  — the per-user files built from it (the diversity
    #                             input); ground_truth.txt + diversity_scores.json
    #                             sit alongside them at the dataset root.
    raw_dir = os.path.join(out_dir, "predictions")
    processed_dir = os.path.join(out_dir, "predictions_processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    gt_file               = os.path.join(out_dir, "ground_truth.txt")
    random_file           = os.path.join(raw_dir, "prediction_random.txt")
    popular_file          = os.path.join(raw_dir, "prediction_popular.txt")
    processed_gt      = os.path.join(processed_dir, processed_filename("ground_truth"))
    processed_random  = os.path.join(processed_dir, processed_filename("random"))
    processed_popular = os.path.join(processed_dir, processed_filename("popular"))

    model_paths = {
        name: {
            "pred":      os.path.join(raw_dir, f"prediction_{name}.txt"),
            "processed": os.path.join(processed_dir, processed_filename(name)),
        }
        for name in cfg["model_recs"]
    }

    # Load the dataset once; every step below reuses these in-memory structures.
    impressions = adapter.load_impressions(behaviors_file)
    article_meta = adapter.load_article_meta(articles_file)

    # -------------------------------------------------------------------------
    # Generate raw predictions (forced ones always; missing ones per policy)
    # -------------------------------------------------------------------------
    def _wanted(name, path, expensive=False):
        if name in force_recommenders:
            return True
        if os.path.exists(path):
            return False
        return train_missing if expensive else generate_missing

    print("Generating recommender predictions...")
    if _wanted("ground_truth", gt_file):
        print("  ground_truth: generating...")
        save_ground_truth(extract_ground_truth(impressions), gt_file)
    if _wanted("random", random_file):
        print("  random: generating...")
        save_predictions(random_recommend(impressions, seed=seed), random_file)
    if _wanted("popular", popular_file):
        print("  popular: generating...")
        save_predictions(popular_recommend(impressions), popular_file)

    # Model recommenders are trained by handing their dataset paths to the
    # training scripts (needs TensorFlow); guarded so a missing/broken training
    # environment skips the model instead of failing the whole run.
    _utils_ready = False
    for name in cfg["model_recs"]:
        pred = model_paths[name]["pred"]
        if not _wanted(name, pred, expensive=True):
            continue
        if cfg.get("prepare") and not _utils_ready:
            cfg["prepare"](in_dir)   # ensure the utils bundle the trainer reads
            _utils_ready = True
        print(f"  {name}: training on '{dataset}' (needs TensorFlow; can take a while)...")
        try:
            _train_model(name, in_dir, cfg["train_split"], cfg["behaviors"][0], pred)
        except Exception as exc:
            print(f"    could not train {name} "
                  f"({exc.__class__.__name__}: {exc}); skipping it.")

    # Active recommenders = those whose raw prediction now exists. Ground truth is
    # scored last (it's the reference), matching the original display order.
    active = []  # (name, raw_path, processed_path, kind)
    if os.path.exists(random_file):
        active.append(("random", random_file, processed_random, "ranks"))
    if os.path.exists(popular_file):
        active.append(("popular", popular_file, processed_popular, "ranks"))
    for name in cfg["model_recs"]:
        if os.path.exists(model_paths[name]["pred"]):
            active.append((name, model_paths[name]["pred"],
                           model_paths[name]["processed"], "ranks"))
    if os.path.exists(gt_file):
        active.append(("ground_truth", gt_file, processed_gt, "gt"))

    unavailable = [n for n in cfg["model_recs"] if not os.path.exists(model_paths[n]["pred"])]
    if unavailable:
        print("  (no prediction file for " + ", ".join(unavailable)
              + " — not scored; (re)run it to include it.)")
    if not active:
        print("No recommender predictions available — nothing to score.")
        return {}

    # -------------------------------------------------------------------------
    # Processed per-user files (the diversity input)
    # -------------------------------------------------------------------------
    to_build = [(n, r, p, k) for (n, r, p, k) in active
                if force_processed or _stale(p, r)]
    if to_build:
        print("Building processed per-user files...")
        for name, raw, proc, kind in to_build:
            if kind == "gt":
                save_user_article_map(raw, article_meta, proc)
            else:
                save_user_article_map_from_ranks(raw, impressions, article_meta, proc)
    else:
        print("Processed per-user files already up to date.")

    # -------------------------------------------------------------------------
    # Diversity scores (cached; only recomputed when inputs change or forced)
    # -------------------------------------------------------------------------
    print("Calculating diversity scores...")
    cd_cfg = cfg["content_diversity"]

    # Embeddings are only needed to (re)compute content diversity, and loading
    # them is expensive, so load lazily and at most once per run.
    _embeddings = {}
    def get_embeddings():
        if "error" in _embeddings:           # fetch already failed once this run
            raise MetricUnavailable(_embeddings["error"])
        if "value" not in _embeddings:
            try:
                if cd_cfg["kind"] == "precomputed":
                    # Ready-made document vectors shipped with the dataset
                    # (e.g. eb-nerd's contrastive_vector.parquet) — just load them.
                    _embeddings["value"] = load_precomputed_embeddings(
                        os.path.join(in_dir, *cd_cfg["vectors"])
                    )
                else:
                    # word_average: the embeddings/dicts are large gitignored
                    # inputs; fetch them on demand (dataset-specific hook) before
                    # the first content-diversity compute.
                    if cfg["prepare"] is not None:
                        cfg["prepare"](in_dir)
                    _embeddings["value"] = load_news_embeddings(
                        articles_file,
                        os.path.join(in_dir, *cd_cfg["embedding"]),
                        os.path.join(in_dir, *cd_cfg["word_dict"]),
                    )
            except Exception as exc:
                # If the vectors can't be obtained (missing file, no network, host
                # down, Recommenders not installed), record it and let content
                # diversity be skipped rather than crashing the whole run.
                _embeddings["error"] = (
                    f"content diversity needs article embeddings, which couldn't "
                    f"be obtained ({exc.__class__.__name__})"
                )
                raise MetricUnavailable(_embeddings["error"]) from exc
        return _embeddings["value"]

    # Which metrics apply to this dataset, as (key, fn(path) -> float).
    # Content diversity only applies when embeddings are shipped
    # (content_diversity config present); eb-nerd and mind_news only have topic.
    metric_defs = [("topic_diversity", lambda p: topic_diversity(p))]
    if cd_cfg is not None:
        metric_defs.append(
            ("content_diversity", lambda p: content_diversity(p, get_embeddings()))
        )

    # Score every active recommender (in the active order: random, popular,
    # models..., ground_truth) from its processed per-user file.
    runs = [(name, proc) for (name, _raw, proc, _kind) in active]

    cache_file = os.path.join(out_dir, "diversity_scores.json")
    old_cache = _load_score_cache(cache_file)
    new_cache = {}
    scores = {}
    total_computed = 0
    for name, path in runs:
        entry, run_cache, n_computed = _compute_run_scores(
            path, metric_defs, old_cache.get(name, {}), force=force_diversity
        )
        scores[name] = entry
        new_cache[name] = run_cache
        total_computed += n_computed

    if new_cache != old_cache:
        _save_score_cache(cache_file, new_cache)

    reused = len(runs) * len(metric_defs) - total_computed
    print(f"  {total_computed} score(s) computed, {reused} reused from cache.")

    print("\n=== Diversity Scores ===")
    for name, s in scores.items():
        print(f"\n  {name}:")
        print(f"    Topic diversity:      {s['topic_diversity']:.4f}")
        if "content_diversity" in s:
            print(f"    Content diversity:    {s['content_diversity']:.4f}")

    return scores


# ---------------------------------------------------------------------------
# Interactive command-line front-end
# ---------------------------------------------------------------------------
def _ask_yes_no(question, default=False):
    """Prompt for a yes/no answer; empty input takes the default."""
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        try:
            answer = input(question + suffix).strip().lower()
        except EOFError:
            return default
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please answer 'y' or 'n'.")


def _choose_dataset(argv):
    """Pick the dataset: use a valid CLI argument, otherwise a numbered menu."""
    names = list(DATASETS)
    arg = argv[1] if len(argv) > 1 else None
    if arg in DATASETS:
        return arg
    if arg is not None:
        print(f"Unknown dataset {arg!r}.")
    print("Available datasets:")
    for i, name in enumerate(names, 1):
        print(f"  {i}) {name}")
    while True:
        answer = input(f"Select a dataset [1-{len(names)} or name]: ").strip()
        if answer in DATASETS:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        print("  Invalid selection.")


def interactive_main(argv):
    """Interactive driver: choose a dataset, show what already exists, then ask
    which recommenders to (re)run, whether to rebuild the processed files, and
    whether to recompute the diversity scores — and run accordingly."""
    dataset = _choose_dataset(argv)
    cfg = DATASETS[dataset]
    out_dir = output_dir(dataset)
    raw_dir = os.path.join(out_dir, "predictions")
    processed_dir = os.path.join(out_dir, "predictions_processed")
    json_file = os.path.join(out_dir, "diversity_scores.json")

    # Ground truth + the recommenders applicable to this dataset.
    recommenders = ["ground_truth", "random", "popular"] + cfg["model_recs"]

    def raw_path(name):
        if name == "ground_truth":
            return os.path.join(out_dir, "ground_truth.txt")
        return os.path.join(raw_dir, f"prediction_{name}.txt")

    def proc_path(name):
        return os.path.join(processed_dir, processed_filename(name))

    def mark(path):
        return "present" if os.path.exists(path) else "missing"

    # 1-3 — show what's already there for this dataset.
    print(f"\n=== EchoBench pipeline — dataset '{dataset}' ===")
    print("\nCurrent state:")
    print(f"  diversity_scores.json : {mark(json_file)}")
    print("  raw predictions:")
    for name in recommenders:
        print(f"      {name:<13}: {mark(raw_path(name))}")
    print("  processed predictions:")
    for name in recommenders:
        print(f"      {name:<13}: {mark(proc_path(name))}")

    # 4 — one question per recommender (default: build the missing cheap ones;
    # models must be opted into explicitly because training is expensive).
    print("\nWhich recommenders should be (re)run?")
    force_recommenders = set()
    for name in recommenders:
        is_model = name in cfg["model_recs"]
        missing = not os.path.exists(raw_path(name))
        default = missing and not is_model
        extra = " (trains a model, needs TensorFlow)" if is_model else ""
        if _ask_yes_no(f"  (re)run {name}{extra}?", default=default):
            force_recommenders.add(name)

    # 3 / 5 — rebuild processed files, and recompute diversity.
    force_processed = _ask_yes_no("\nRebuild processed per-user files?", default=False)
    force_diversity = _ask_yes_no("(Re)calculate diversity scores?", default=False)

    print()
    run_pipeline(
        dataset,
        force_recommenders=force_recommenders,
        force_processed=force_processed,
        force_diversity=force_diversity,
        # Only build what was explicitly asked for; the prompts already defaulted
        # missing cheap recommenders to "yes".
        generate_missing=False,
        train_missing=False,
    )


if __name__ == "__main__":
    interactive_main(sys.argv)