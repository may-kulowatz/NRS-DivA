"""End-to-end pipeline: load a dataset, run the baseline recommenders, write
prediction / processed files, and report diversity scores.

The supporting concerns live in their own modules: the dataset registry and path
helpers in ``config.py``, the recommender interface in
``recommender_module/base.py``, the diversity-score computation/IO in
``scores.py``, and the interactive front-end in ``cli.py``. This module is just
the orchestration that ties them together.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASETS, DATA_ROOT, input_dir, output_dir
from recommender_module.base import build_recommenders, RunContext
from scores import (
    MetricUnavailable,
    _compute_scores,
    _save_scores,
    _stale,
)
# Each dataset's preparation is driven through its config "prepare" module:
# ensure_raw_data guarantees the essential inputs up front; ensure_utils fetches
# the optional content-diversity bundle on demand.
from diversity_module.topic_diversity import topic_diversity
from diversity_module.content_diversity import (
    content_diversity,
    load_news_embeddings,
    load_precomputed_embeddings,
)
from diversity_module.content_diversity_normalized import normalized_content_diversity


def run_pipeline(dataset="MIND", seed=42, data_root=DATA_ROOT, *,
                 force_recommenders=None, force_processed=False,
                 generate_missing=True, train_missing=True,
                 normalized_diversity=False, normalized_max_combinations=1000):
    """Run a dataset's recommenders + diversity scoring.

    By default a recommender's prediction is (re)built only when missing and its
    processed file only when out of date — the cheap "just make sure everything's
    there" behaviour used by code and tests. Diversity scores are always
    recomputed from the processed files (there is no reuse cache; see
    ``scores.py``). The keyword flags give the interactive front-end finer control:

    force_recommenders : iterable of recommender names whose raw predictions are
                         regenerated even if they already exist (model recs are
                         retrained — needs TensorFlow).
    force_processed    : rebuild every processed per-user file, not just stale ones.
    generate_missing   : auto-generate a *cheap* recommender (ground truth, random,
                         popular) when its file is missing. Set False to only build
                         what force_recommenders asks for.
    train_missing      : auto-train a *model* recommender when its prediction file
                         is missing. Set False to only train what's forced.
    normalized_diversity : also compute the (expensive, per-impression) normalized
                         content-diversity metric for datasets that ship content
                         embeddings. Off by default.
    normalized_max_combinations : per-impression budget for the min/max estimate in
                         the normalized metric (sampled above it; smaller = faster).
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(DATASETS)}")
    cfg = DATASETS[dataset]
    adapter = cfg["adapter"]
    force_recommenders = set(force_recommenders or ())

    in_dir = input_dir(dataset, data_root)
    out_dir = output_dir(dataset, data_root)

    # Make sure the dataset's essential raw inputs exist before anything reads
    # them, fetching/building whatever is missing. The optional content-diversity
    # bundle is fetched later, lazily, via the same prepare module's ensure_utils.
    cfg["prepare"].ensure_raw_data(in_dir)

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

    # Load the dataset once; every step below reuses these in-memory structures.
    impressions = adapter.load_impressions(behaviors_file)
    article_meta = adapter.load_article_meta(articles_file)

    # The recommenders this dataset has, in scoring/display order (random, popular,
    # its models..., ground_truth), plus the shared context they each operate on.
    recommenders = build_recommenders(cfg["model_recs"])
    ctx = RunContext(
        impressions=impressions, article_meta=article_meta,
        out_dir=out_dir, raw_dir=raw_dir, processed_dir=processed_dir,
        seed=seed, in_dir=in_dir,
        train_split=cfg.get("train_split"), dev_split=cfg["behaviors"][0],
    )

    # -------------------------------------------------------------------------
    # Generate raw predictions (forced ones always; missing ones per policy)
    # -------------------------------------------------------------------------
    def _wanted(rec):
        if rec.name in force_recommenders:
            return True
        if os.path.exists(rec.raw_path(ctx)):
            return False
        return train_missing if rec.expensive else generate_missing

    print("Generating recommender predictions...")
    # The prepare hook (fetches the utils bundle a model trainer reads) is run at
    # most once, just before the first model is trained.
    _utils_ready = False
    for rec in recommenders:
        if not _wanted(rec):
            continue
        if rec.expensive:
            # Model recommenders are trained by handing their dataset paths to the
            # training scripts (needs TensorFlow); guarded so a missing/broken
            # training environment skips the model instead of failing the run.
            if not _utils_ready:
                cfg["prepare"].ensure_utils(in_dir)
                _utils_ready = True
            print(f"  {rec.name}: training on '{dataset}' (needs TensorFlow; can take a while)...")
            try:
                rec.generate(ctx)
            except Exception as exc:
                print(f"    could not train {rec.name} "
                      f"({exc.__class__.__name__}: {exc}); skipping it.")
        else:
            print(f"  {rec.name}: generating...")
            rec.generate(ctx)

    # Active recommenders = those whose raw prediction now exists (kept in the
    # scoring order above, so ground truth is scored last as the reference).
    active = [rec for rec in recommenders if os.path.exists(rec.raw_path(ctx))]

    unavailable = [rec.name for rec in recommenders
                   if rec.expensive and not os.path.exists(rec.raw_path(ctx))]
    if unavailable:
        print("  (no prediction file for " + ", ".join(unavailable)
              + " — not scored; (re)run it to include it.)")
    if not active:
        print("No recommender predictions available — nothing to score.")
        return {}

    # -------------------------------------------------------------------------
    # Processed per-user files (the diversity input)
    # -------------------------------------------------------------------------
    to_build = [rec for rec in active
                if force_processed or _stale(rec.processed_path(ctx), rec.raw_path(ctx))]
    if to_build:
        print("Building processed per-user files...")
        for rec in to_build:
            rec.build_user_map(ctx)
    else:
        print("Processed per-user files already up to date.")

    # -------------------------------------------------------------------------
    # Diversity scores (recomputed every run; persisted for the dashboard)
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
                    cfg["prepare"].ensure_utils(in_dir)
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
    scores_file = os.path.join(out_dir, "diversity_scores.json")
    scores = {rec.name: _compute_scores(rec.processed_path(ctx), metric_defs)
              for rec in active}

    # Optional, expensive: normalized (EBNeRD-style) content diversity. Needs the
    # candidate pools, so it works from each recommender's per-impression choices
    # rather than the per-user file. Only when asked and embeddings are available.
    if normalized_diversity and cd_cfg is not None:
        print("  computing normalized content diversity (per-impression; can be slow)...")
        try:
            embeddings = get_embeddings()
        except MetricUnavailable as exc:
            print(f"    skipping content_diversity_normalized: {exc}")
        else:
            for rec in active:
                scores[rec.name]["content_diversity_normalized"] = normalized_content_diversity(
                    impressions, rec.recommended_by_impr(ctx), embeddings,
                    max_combinations=normalized_max_combinations, seed=seed,
                )

    _save_scores(scores_file, scores)
    print(f"  scored {len(scores)} recommender(s).")

    print("\n=== Diversity Scores ===")
    for name, s in scores.items():
        print(f"\n  {name}:")
        print(f"    Topic diversity:      {s['topic_diversity']:.4f}")
        if "content_diversity" in s:
            print(f"    Content diversity:    {s['content_diversity']:.4f}")
        if "content_diversity_normalized" in s:
            print(f"    Content div. (norm.): {s['content_diversity_normalized']:.4f}")

    return scores


if __name__ == "__main__":
    # The interactive front-end lives in cli.py; imported here (lazily) so
    # `python pipeline.py <dataset>` keeps working as the documented entry point.
    from cli import interactive_main
    interactive_main(sys.argv)