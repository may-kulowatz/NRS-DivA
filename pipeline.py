"""End-to-end pipeline: load a dataset, run the baseline recommenders, write
prediction / user-article files, and report diversity scores.

The pipeline is dataset-agnostic. A dataset adapter (datasets/mind_adapter.py,
datasets/ebnerd_adapter.py) normalizes the raw files into Impression records and
article metadata; everything downstream is shared. Adding a dataset means
adding an adapter and a DATASETS entry — no recommender or writer changes.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import mind_adapter, ebnerd_adapter
from recommender_systems.ground_truth import extract_ground_truth, save_ground_truth
from recommender_systems.random_rec import random_recommend
from recommender_systems.popular_rec import popular_recommend
from recommender_systems.io import (
    save_predictions,
    save_predictions_topk,
    save_user_article_map,
)
from diversityScores.topic_diversity import topic_diversity, subtopic_diversity
from diversityScores.content_diversity import content_diversity, load_news_embeddings


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
# Each entry describes where a dataset's raw files live (relative to its
# data_dir) and which optional steps apply to it. Paths are tuples joined with
# os.path.join so they work on any platform.
DATASETS = {
    "MIND": {
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


def _prediction_topk(prediction_file, impressions, output_topk):
    """Convert a raw model prediction file (impr_id [ranks]) to the top-k format
    (impr_id user_id [positions] [ids]).

    Model prediction files (NRMS, LSTUR, ...) have no user_id column and only
    per-candidate ranks, so user_ids, candidate ids and K (number of clicks) all
    come from the normalized impressions.
    """
    pred_ranks = {}
    with open(prediction_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            pred_ranks[int(parts[0])] = list(map(int, parts[1][1:-1].split(",")))

    with open(output_topk, "w", encoding="utf-8") as f:
        for imp in impressions:
            ranks = pred_ranks.get(imp.impr_id, [])
            k = sum(imp.labels)
            top_k_idx = np.argsort(ranks)[:k]
            positions = [i + 1 for i in top_k_idx]
            chosen_ids = [imp.candidate_ids[i] for i in top_k_idx]
            f.write(
                f"{imp.impr_id} {imp.user_id} ["
                + ",".join(map(str, positions))
                + "] ["
                + ",".join(chosen_ids)
                + "]\n"
            )


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


def run_pipeline(data_dir, dataset="MIND", seed=42):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(DATASETS)}")
    cfg = DATASETS[dataset]
    adapter = cfg["adapter"]

    behaviors_file = os.path.join(data_dir, *cfg["behaviors"])
    articles_file = os.path.join(data_dir, *cfg["articles"])

    pred_dir = os.path.join(data_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    gt_file               = os.path.join(pred_dir, "prediction_ground_truth.txt")
    random_file           = os.path.join(pred_dir, "prediction_random.txt")
    random_topk_file      = os.path.join(pred_dir, "prediction_random_topk.txt")
    popular_file          = os.path.join(pred_dir, "prediction_popular.txt")
    popular_topk_file     = os.path.join(pred_dir, "prediction_popular_topk.txt")
    user_articles_gt      = os.path.join(pred_dir, "user_articles_ground_truth.txt")
    user_articles_random  = os.path.join(pred_dir, "user_articles_random.txt")
    user_articles_popular = os.path.join(pred_dir, "user_articles_popular.txt")

    # Model recommenders (NRMS, LSTUR, ...) each ship a raw prediction file that
    # is converted to top-k and then to a user-article map, exactly like NRMS.
    model_recs = cfg["model_recs"]
    model_paths = {
        name: {
            "pred": os.path.join(pred_dir, f"prediction_{name}.txt"),
            "topk": os.path.join(pred_dir, f"prediction_{name}_topk.txt"),
            "map":  os.path.join(pred_dir, f"user_articles_{name}.txt"),
        }
        for name in model_recs
    }

    # -------------------------------------------------------------------------
    # Step 0 — Check which files already exist
    # -------------------------------------------------------------------------
    skip_gt      = _exists(gt_file)
    skip_random  = _exists(random_file, random_topk_file)
    skip_popular = _exists(popular_file, popular_topk_file)
    skip_model   = {name: not _stale(p["topk"], p["pred"]) for name, p in model_paths.items()}

    print(f"Step 0/6 — Checking existing files for dataset '{dataset}'...")
    checks = [
        ("Ground truth",        skip_gt),
        ("Random predictions",  skip_random),
        ("Popular predictions", skip_popular),
    ]
    checks += [(f"{name.upper()} topk", skip_model[name]) for name in model_recs]
    for label, skip in checks:
        print(f"  {'SKIP' if skip else 'RUN '} — {label}")

    # Load the dataset once; every step below reuses these in-memory structures.
    impressions = adapter.load_impressions(behaviors_file)
    article_meta = adapter.load_article_meta(articles_file)

    # -------------------------------------------------------------------------
    # Step 1 — Ground truth
    # -------------------------------------------------------------------------
    if skip_gt:
        print("Step 1/6 — Ground truth already exists, skipping.")
    else:
        print("Step 1/6 — Generating ground truth...")
        save_ground_truth(extract_ground_truth(impressions), gt_file)

    # -------------------------------------------------------------------------
    # Step 2 — Random recommendations
    # -------------------------------------------------------------------------
    if skip_random:
        print("Step 2/6 — Random predictions already exist, skipping.")
    else:
        print("Step 2/6 — Generating random recommendations...")
        random_results = random_recommend(impressions, seed=seed)
        save_predictions(random_results, random_file)
        save_predictions_topk(random_results, impressions, random_topk_file)

    # -------------------------------------------------------------------------
    # Step 3 — Model recommender topk (NRMS, LSTUR, ...; MIND only)
    # -------------------------------------------------------------------------
    if not model_recs:
        print("Step 3/6 — No model recommenders for this dataset, skipping.")
    else:
        for name in model_recs:
            p = model_paths[name]
            if skip_model[name]:
                print(f"Step 3/6 — {name.upper()} topk already up to date, skipping.")
            else:
                print(f"Step 3/6 — Converting {name.upper()} predictions to top-k format...")
                _prediction_topk(p["pred"], impressions, p["topk"])

    # -------------------------------------------------------------------------
    # Step 4 — Popular recommendations
    # -------------------------------------------------------------------------
    if skip_popular:
        print("Step 4/6 — Popular predictions already exist, skipping.")
    else:
        print("Step 4/6 — Generating popular recommendations...")
        popular_results = popular_recommend(impressions)
        save_predictions(popular_results, popular_file)
        save_predictions_topk(popular_results, impressions, popular_topk_file)

    # -------------------------------------------------------------------------
    # Step 5 — User article maps
    # -------------------------------------------------------------------------
    # Each map is rebuilt only when its source top-k file is newer (checked here,
    # after Steps 1-4 have refreshed those sources), so replacing a model's
    # prediction file rebuilds just that model's map and leaves the others'
    # cached scores intact.
    map_specs = [
        (user_articles_gt, gt_file),
        (user_articles_random, random_topk_file),
        (user_articles_popular, popular_topk_file),
    ]
    map_specs += [(model_paths[name]["map"], model_paths[name]["topk"]) for name in model_recs]

    stale_maps = [(out, src) for out, src in map_specs if _stale(out, src)]
    if not stale_maps:
        print("Step 5/6 — User article maps already exist, skipping.")
    else:
        print("Step 5/6 — Building user article maps...")
        for out, src in stale_maps:
            save_user_article_map(src, article_meta, out)

    # -------------------------------------------------------------------------
    # Step 6 — Diversity scores (cached; only recomputed when inputs change)
    # -------------------------------------------------------------------------
    print("Step 6/6 — Calculating diversity scores...")
    subtopic_category = cfg["subtopic_category"]
    cd_cfg = cfg["content_diversity"]

    # Embeddings are only needed to (re)compute content diversity, and loading
    # them is expensive, so load lazily and at most once per run.
    _embeddings = {}
    def get_embeddings():
        if "value" not in _embeddings:
            _embeddings["value"] = load_news_embeddings(
                articles_file,
                os.path.join(data_dir, *cd_cfg["embedding"]),
                os.path.join(data_dir, *cd_cfg["word_dict"]),
            )
        return _embeddings["value"]

    # Which metrics apply to this dataset, as (key, fn(path) -> float).
    # Subtopic diversity only applies when subcategories nest under a parent
    # category (subtopic_category set); content diversity only when embeddings
    # are shipped (content_diversity config present). eb-nerd has neither.
    metric_defs = [("topic_diversity", lambda p: topic_diversity(p))]
    if subtopic_category is not None:
        metric_defs.append(
            ("subtopic_diversity", lambda p: subtopic_diversity(p, category=subtopic_category))
        )
    if cd_cfg is not None:
        metric_defs.append(
            ("content_diversity", lambda p: content_diversity(p, get_embeddings()))
        )

    runs = [
        ("random",  user_articles_random),
        ("popular", user_articles_popular),
    ]
    runs += [(name, model_paths[name]["map"]) for name in model_recs]
    runs.append(("ground_truth", user_articles_gt))

    cache_file = os.path.join(pred_dir, "diversity_scores.json")
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
    _project_dir = os.path.dirname(os.path.abspath(__file__))
    # Default to MIND; pass a dataset name to run another, e.g.:
    #   python pipeline.py ebnerd
    _dataset = sys.argv[1] if len(sys.argv) > 1 else "MIND"
    _data_dir = os.path.join(_project_dir, "data", _dataset)
    run_pipeline(_data_dir, dataset=_dataset)