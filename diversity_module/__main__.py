"""Compute diversity scores for a dataset's recommender predictions.

Compute one measure on a dataset, or every measure at once:

    python -m diversity_module <dataset> <measure> [<recommender>]   # just that one
    python -m diversity_module <dataset> --all                        # all of them

``<measure>`` is one of ``topic_diversity``, ``content_diversity``, or
``content_diversity_normalized`` (plus per-embedding-space variants such as
``content_diversity_xlmr``); which measures apply depends on the dataset. By
default a measure is scored across every recommender that has a prediction file;
pass an optional ``<recommender>`` (e.g. ``random``) to score only that one.

PRE  : the recommenders have been generated (run ``python -m recommender_module
       <dataset> --all`` first). Recommenders without a prediction file are
       skipped; if none exist there is nothing to score.
POST : ``data_processed/<dataset>/run_manifest.json`` is updated with the computed
       measure(s) (measures not computed this run keep their stored value) and the
       scores are printed. Any stale processed per-user file is rebuilt from its
       prediction first. No predictions are generated and no model is trained.
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASETS
from recommender_module.base import build_context
from scores import (
    MetricUnavailable,
    _compute_scores,
    _stale,
    load_manifest,
    save_manifest,
    metric_value,
    record_metric,
    record_stage_times,
)
from diversity_module.topic_diversity import topic_diversity
from diversity_module.content_diversity import (
    content_diversity,
    load_news_embeddings,
    load_precomputed_embeddings,
)
from diversity_module.content_diversity_normalized import normalized_content_diversity

# Per-impression sampling budget for the normalized metric's min/max estimate.
_NORMALIZED_MAX_COMBINATIONS = 1000


def _content_spaces(cfg, ctx, articles_file):
    """The content-embedding spaces to score, as ``(suffix, lazy_loader)`` pairs.

    The primary space (bare ``content_diversity`` keys) comes from the dataset's
    ``content_diversity`` config; each extra precomputed space in
    ``content_embeddings`` adds a suffixed ``content_diversity_<name>`` key.
    Loading an embedding map is expensive, so each loader runs at most once and a
    load failure becomes ``MetricUnavailable`` (only that measure is skipped).
    """
    def _make_loader(load_fn):
        cache = {}
        def loader():
            if "error" in cache:
                raise MetricUnavailable(cache["error"])
            if "value" not in cache:
                try:
                    cache["value"] = load_fn()
                except Exception as exc:
                    cache["error"] = (
                        f"content diversity needs article embeddings, which "
                        f"couldn't be obtained ({exc.__class__.__name__})"
                    )
                    raise MetricUnavailable(cache["error"]) from exc
            return cache["value"]
        return loader

    cd_cfg = cfg["content_diversity"]

    def _load_primary():
        if cd_cfg["kind"] == "precomputed":
            return load_precomputed_embeddings(os.path.join(ctx.in_dir, *cd_cfg["vectors"]))
        # word_average: fetch the (gitignored) embeddings/dicts on demand.
        cfg["prepare"].ensure_utils(ctx.in_dir)
        return load_news_embeddings(
            articles_file,
            os.path.join(ctx.in_dir, *cd_cfg["embedding"]),
            os.path.join(ctx.in_dir, *cd_cfg["word_dict"]),
        )

    spaces = []
    if cd_cfg is not None:
        spaces.append(("", _make_loader(_load_primary)))
    for name, (vec_file, vec_col) in cfg.get("content_embeddings", {}).items():
        spaces.append((
            f"_{name}",
            _make_loader(
                lambda vf=vec_file, vc=vec_col: load_precomputed_embeddings(
                    os.path.join(ctx.in_dir, vf), vector_column=vc
                )
            ),
        ))
    return spaces


def score(dataset, only=None, recommender=None):
    """Compute ``dataset``'s diversity scores — one measure, or all.

    only        : a single measure key, or ``None`` for every applicable measure.
    recommender : score only this recommender's prediction, or ``None`` for all
                  recommenders that have one.
    """
    cfg, ctx, recs = build_context(dataset)
    articles_file = os.path.join(ctx.in_dir, *cfg["articles"])

    active = [rec for rec in recs if os.path.exists(rec.raw_path(ctx))]
    if not active:
        print(f"No recommender predictions found for '{dataset}'. Generate them "
              f"first with:  python -m recommender_module {dataset} --all")
        return {}

    if recommender is not None:
        if recommender not in {rec.name for rec in recs}:
            raise SystemExit(
                f"Unknown recommender {recommender!r} for '{dataset}'. "
                f"Available: {[rec.name for rec in recs]}"
            )
        active = [rec for rec in active if rec.name == recommender]
        if not active:
            print(f"No prediction for '{recommender}' on '{dataset}' yet. Generate "
                  f"it first with:  python -m recommender_module {dataset} {recommender}")
            return {}

    # Diversity reads the per-user files; rebuild any that are missing or stale.
    for rec in active:
        if _stale(rec.processed_path(ctx), rec.raw_path(ctx)):
            print(f"  ({rec.name}: processed file out of date — rebuilding)")
            rec.build_user_map(ctx)

    content_spaces = _content_spaces(cfg, ctx, articles_file)

    # Per-user measures as (key, fn(path) -> float): topic + one ILD per space.
    metric_defs = [("topic_diversity", lambda p: topic_diversity(p))]
    for suffix, loader in content_spaces:
        metric_defs.append((
            f"content_diversity{suffix}",
            lambda p, ld=loader: content_diversity(p, ld()),
        ))
    # Normalized variants are computed separately (they need the candidate pools).
    normalized_keys = [f"content_diversity_normalized{suffix}" for suffix, _ in content_spaces]
    applicable = [k for k, _ in metric_defs] + normalized_keys

    if only is None:
        selected = set(applicable)
    else:
        if only not in applicable:
            raise SystemExit(
                f"Unknown measure {only!r} for '{dataset}'. Available: {applicable}"
            )
        selected = {only}

    # Merge into the manifest already on disk so measures we are NOT computing this
    # run keep their stored value.
    manifest = load_manifest(ctx.out_dir)
    now = datetime.now().isoformat(timespec="seconds")
    for rec in active:
        record_stage_times(manifest.setdefault(rec.name, {}),
                            rec.raw_path(ctx), rec.processed_path(ctx))

    print(f"Calculating diversity scores for '{dataset}'...")
    per_user_defs = [(k, fn) for k, fn in metric_defs if k in selected]
    if per_user_defs:
        for rec in active:
            entry = manifest.setdefault(rec.name, {})
            for key, value in _compute_scores(rec.processed_path(ctx), per_user_defs).items():
                record_metric(entry, key, value, now)

    for (suffix, loader), nkey in zip(content_spaces, normalized_keys):
        if nkey not in selected:
            continue
        print(f"  computing {nkey} (per-impression; can be slow)...")
        try:
            embeddings = loader()
        except MetricUnavailable as exc:
            print(f"    skipping {nkey}: {exc}")
            continue
        for rec in active:
            value = normalized_content_diversity(
                ctx.impressions, rec.recommended_by_impr(ctx), embeddings,
                max_combinations=_NORMALIZED_MAX_COMBINATIONS, seed=ctx.seed,
            )
            record_metric(manifest.setdefault(rec.name, {}), nkey, value, now)

    save_manifest(ctx.out_dir, manifest)

    # Display the measure(s) computed this run, in canonical order.
    display_order = ["topic_diversity"]
    for suffix, _ in content_spaces:
        display_order.append(f"content_diversity{suffix}")
        display_order.append(f"content_diversity_normalized{suffix}")
    display_keys = [k for k in display_order if k in selected]
    print("\n=== Diversity Scores ===")
    for rec in active:
        print(f"\n  {rec.name}:")
        for key in display_keys:
            value = metric_value(manifest, rec.name, key)
            if value is not None:
                print(f"    {key:<37} {value:.4f}")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m diversity_module",
        description="Compute a dataset's diversity scores (no generation).",
    )
    p.add_argument("dataset", choices=list(DATASETS),
                   help="dataset to score")
    p.add_argument("measure", nargs="?",
                   help="a single measure to compute, e.g. topic_diversity, "
                        "content_diversity, content_diversity_normalized (which "
                        "ones apply depends on the dataset). Omit and use --all to "
                        "compute them all.")
    p.add_argument("recommender", nargs="?",
                   help="score only this recommender's prediction, e.g. random "
                        "(default: every recommender with a prediction)")
    p.add_argument("--all", action="store_true",
                   help="compute every measure on the dataset — WARNING: this can "
                        "take a while, as it includes the per-impression normalized "
                        "content diversity")
    args = p.parse_args(argv)

    if bool(args.measure) == args.all:
        p.error("specify exactly one of: a measure name, or --all")

    if args.all:
        print(f"Computing every diversity measure on '{args.dataset}'. This can "
              f"take a while — it includes the slow per-impression normalized metric.")
    score(args.dataset, only=None if args.all else args.measure,
          recommender=args.recommender)


if __name__ == "__main__":
    main()