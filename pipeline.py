"""End-to-end pipeline: load a dataset, run the baseline recommenders, write
prediction / processed files, and report diversity scores.

"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_module import mind_adapter, ebnerd_adapter
from recommender_module.common.ground_truth import extract_ground_truth, save_ground_truth
from recommender_module.common.random_rec import random_recommend
from recommender_module.common.popular_rec import popular_recommend
from recommender_module.common.io import (
    processed_filename,
    save_predictions,
    save_user_article_map,
    save_user_article_map_from_results,
    save_user_article_map_from_ranks,
    save_user_article_map_from_ground_truth,
)
from recommender_module.common.subtopic import (
    build_subtopic_subset,
    subtopic_subset_path,
)
from diversity_module.topic_diversity import topic_diversity, subtopic_diversity
from diversity_module.content_diversity import content_diversity, load_news_embeddings


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
        "subtopic_category": "news",
        "content_diversity": {
            "embedding": ("utils", "embedding.npy"),
            "word_dict": ("utils", "word_dict.pkl"),
        },
    },
    "ebnerd": {
        "dir": "ebnerd",
        "adapter": ebnerd_adapter,
        "behaviors": ("validation", "behaviors.parquet"),
        "articles": ("articles.parquet",),
        # eb-nerd ships no model prediction files and no embeddings yet, so both
        # the model recommenders and content diversity are skipped. Its numeric
        # subcategory codes don't map to a parent category, so subtopic diversity
        # is skipped too (subtopic_category=None); topic diversity uses the
        # multi-valued `topics` field instead.
        "model_recs": [],
        "subtopic_category": None,
        "content_diversity": None,
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


def _exists(*paths):
    return all(os.path.exists(p) for p in paths)


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


def _compute_run_scores(path, metric_defs, prev_cache):
    """Compute (or reuse) every metric for one user-article file.

    metric_defs : list of (metric_key, fn) where fn(path) -> float.
    prev_cache  : the cached {metric_key: {"value", "sig"}} for this run.

    Returns (scores, cache, n_computed): scores is {metric_key: value} for
    display; cache is the refreshed {metric_key: {"value", "sig"}} to persist;
    n_computed counts how many metrics were actually (re)calculated.
    """
    sig = _file_sig(path)
    scores, cache, n_computed = {}, {}, 0
    for key, fn in metric_defs:
        prev = prev_cache.get(key)
        if prev is not None and prev.get("sig") == sig:
            value = prev["value"]
        else:
            value = fn(path)
            n_computed += 1
        scores[key] = value
        cache[key] = {"value": value, "sig": sig}
    return scores, cache, n_computed


def run_pipeline(dataset="MIND", seed=42, data_root=DATA_ROOT):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(DATASETS)}")
    cfg = DATASETS[dataset]
    adapter = cfg["adapter"]

    in_dir = input_dir(dataset, data_root)
    out_dir = output_dir(dataset, data_root)

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

    # Model recommenders (NRMS, LSTUR, ...) each ship a full-rank prediction file
    # in predictions/; the processed per-user file is built straight from it.
    model_recs = cfg["model_recs"]
    model_paths = {
        name: {
            "pred":      os.path.join(raw_dir, f"prediction_{name}.txt"),
            "processed": os.path.join(processed_dir, processed_filename(name)),
        }
        for name in model_recs
    }

    # -------------------------------------------------------------------------
    # Step 0 — Check which files already exist
    # -------------------------------------------------------------------------
    skip_gt      = _exists(gt_file)
    skip_random  = _exists(random_file)
    skip_popular = _exists(popular_file)

    print(f"Step 0/5 — Checking existing files for dataset '{dataset}'...")
    checks = [
        ("Ground truth",        skip_gt),
        ("Random predictions",  skip_random),
        ("Popular predictions", skip_popular),
    ]
    # Model full-rank files are shipped, not generated; flag any that are missing.
    checks += [
        (f"{name.upper()} predictions (shipped)", _exists(model_paths[name]["pred"]))
        for name in model_recs
    ]
    for label, present in checks:
        print(f"  {'SKIP' if present else 'RUN '} — {label}")

    # Load the dataset once; every step below reuses these in-memory structures.
    impressions = adapter.load_impressions(behaviors_file)
    article_meta = adapter.load_article_meta(articles_file)

    # -------------------------------------------------------------------------
    # Step 1 — Ground truth
    # -------------------------------------------------------------------------
    if skip_gt:
        print("Step 1/5 — Ground truth already exists, skipping.")
    else:
        print("Step 1/5 — Generating ground truth...")
        save_ground_truth(extract_ground_truth(impressions), gt_file)

    # -------------------------------------------------------------------------
    # Step 2 — Random recommendations (full-rank output only)
    # -------------------------------------------------------------------------
    if skip_random:
        print("Step 2/5 — Random predictions already exist, skipping.")
    else:
        print("Step 2/5 — Generating random recommendations...")
        save_predictions(random_recommend(impressions, seed=seed), random_file)

    # -------------------------------------------------------------------------
    # Step 3 — Popular recommendations (full-rank output only)
    # -------------------------------------------------------------------------
    if skip_popular:
        print("Step 3/5 — Popular predictions already exist, skipping.")
    else:
        print("Step 3/5 — Generating popular recommendations...")
        save_predictions(popular_recommend(impressions), popular_file)

    # -------------------------------------------------------------------------
    # Step 4 — Processed per-user files (the diversity input)
    # -------------------------------------------------------------------------

    map_specs = [
        (processed_gt, gt_file, "gt"),
        (processed_random, random_file, "ranks"),
        (processed_popular, popular_file, "ranks"),
    ]
    map_specs += [
        (model_paths[name]["processed"], model_paths[name]["pred"], "ranks")
        for name in model_recs
    ]

    stale_maps = [(out, src, kind) for out, src, kind in map_specs if _stale(out, src)]
    if not stale_maps:
        print("Step 4/5 — Processed per-user files already exist, skipping.")
    else:
        print("Step 4/5 — Building processed per-user files...")
        for out, src, kind in stale_maps:
            if kind == "gt":
                save_user_article_map(src, article_meta, out)
            else:
                save_user_article_map_from_ranks(src, impressions, article_meta, out)

    # -------------------------------------------------------------------------
    # Step 5b — Subtopic news subset
    # -------------------------------------------------------------------------

    subtopic_category = cfg["subtopic_category"]
    if subtopic_category is None:
        print("Step 4b/5 — No subtopic category for this dataset, skipping.")
    else:
        sub_dir = os.path.join(processed_dir, "subtopic")
        os.makedirs(sub_dir, exist_ok=True)
        sub_impressions, sub_meta, positions_by_impr = build_subtopic_subset(
            impressions, article_meta, subtopic_category
        )

        def _write_random_map(out):
            save_user_article_map_from_results(
                random_recommend(sub_impressions, seed=seed), sub_impressions, sub_meta, out
            )

        def _write_popular_map(out):
            save_user_article_map_from_results(
                popular_recommend(sub_impressions), sub_impressions, sub_meta, out
            )

        def _write_gt_map(out):
            save_user_article_map_from_ground_truth(
                extract_ground_truth(sub_impressions), sub_meta, out
            )

        # (name, source file that triggers a rebuild, map writer(out)). Model
        # subsets reuse the full-dataset ranks, sliced to the in-category
        # candidate positions via positions_by_impr.
        sub_runs = [
            ("random",       random_file,  _write_random_map),
            ("popular",      popular_file, _write_popular_map),
            ("ground_truth", gt_file,      _write_gt_map),
        ]
        for name in model_recs:
            pred_file = model_paths[name]["pred"]
            sub_runs.append((
                name, pred_file,
                lambda out, pf=pred_file: save_user_article_map_from_ranks(
                    pf, sub_impressions, sub_meta, out, positions_by_impr=positions_by_impr
                ),
            ))

        built = False
        for name, source, write_map in sub_runs:
            sub_processed = os.path.join(sub_dir, processed_filename(name))
            if _stale(sub_processed, source):
                write_map(sub_processed)
                built = True
        print(
            "Step 4b/5 — Subtopic news subset "
            + ("rebuilt." if built else "already up to date, skipping.")
        )

    # -------------------------------------------------------------------------
    # Step 5 — Diversity scores (cached; only recomputed when inputs change)
    # -------------------------------------------------------------------------
    print("Step 5/5 — Calculating diversity scores...")
    cd_cfg = cfg["content_diversity"]

    # Embeddings are only needed to (re)compute content diversity, and loading
    # them is expensive, so load lazily and at most once per run.
    _embeddings = {}
    def get_embeddings():
        if "value" not in _embeddings:
            _embeddings["value"] = load_news_embeddings(
                articles_file,
                os.path.join(in_dir, *cd_cfg["embedding"]),
                os.path.join(in_dir, *cd_cfg["word_dict"]),
            )
        return _embeddings["value"]

    # Which metrics apply to this dataset, as (key, fn(path) -> float).
    # Subtopic diversity only applies when subcategories nest under a parent
    # category (subtopic_category set); content diversity only when embeddings
    # are shipped (content_diversity config present). eb-nerd has neither.
    # Subtopic reads the news-subset sibling of each run's user-article file
    # (predictions/subtopic/...). The run's signature stays the main file: the
    # subset is regenerated from the same source in the same run, so the two move
    # together and a stale main file already forces a subtopic recompute.
    metric_defs = [("topic_diversity", lambda p: topic_diversity(p))]
    if subtopic_category is not None:
        metric_defs.append(
            ("subtopic_diversity", lambda p: subtopic_diversity(subtopic_subset_path(p)))
        )
    if cd_cfg is not None:
        metric_defs.append(
            ("content_diversity", lambda p: content_diversity(p, get_embeddings()))
        )

    runs = [
        ("random",  processed_random),
        ("popular", processed_popular),
    ]
    runs += [(name, model_paths[name]["processed"]) for name in model_recs]
    runs.append(("ground_truth", processed_gt))

    cache_file = os.path.join(out_dir, "diversity_scores.json")
    old_cache = _load_score_cache(cache_file)
    new_cache = {}
    scores = {}
    total_computed = 0
    for name, path in runs:
        entry, run_cache, n_computed = _compute_run_scores(
            path, metric_defs, old_cache.get(name, {})
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
        if "subtopic_diversity" in s:
            print(f"    Subtopic diversity:   {s['subtopic_diversity']:.4f}")
        if "content_diversity" in s:
            print(f"    Content diversity:    {s['content_diversity']:.4f}")

    return scores


if __name__ == "__main__":
    # Default to MIND; pass a dataset name to run another, e.g.:
    #   python pipeline.py ebnerd
    _dataset = sys.argv[1] if len(sys.argv) > 1 else "MIND"
    run_pipeline(dataset=_dataset)