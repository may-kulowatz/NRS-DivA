"""Generate recommender predictions for a dataset.

Run one recommender on a dataset, or every recommender at once:

    python -m recommender_module <dataset> <recommender>   # just that one
    python -m recommender_module <dataset> --all           # all of them

``<recommender>`` is one of ``random``, ``popular``, ``ground_truth`` or a model
(``nrms`` / ``lstur`` / ``naml``); which models exist depends on the dataset.

PRE  : the dataset's raw inputs exist (run ``python -m dataset_module`` first).
POST : each recommender that ran has written its raw prediction file
       (``data_processed/<dataset>/predictions/prediction_<name>.txt``, or
       ``ground_truth.txt``) and its per-user processed file
       (``predictions_processed/…``), and the run manifest's stage timestamps are
       refreshed. No diversity scores are written — use ``python -m
       diversity_module`` for that.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASETS
from recommender_module.base import build_context
from scores import load_manifest, save_manifest, record_stage_times


def generate(dataset, only=None):
    """(Re)generate ``dataset``'s recommenders — one of them, or all.

    only : a single recommender name, or ``None`` to run every recommender.
    """
    cfg, ctx, recs = build_context(dataset)

    if only is None:
        selected = recs
    else:
        selected = [rec for rec in recs if rec.name == only]
        if not selected:
            raise SystemExit(
                f"Unknown recommender {only!r} for '{dataset}'. "
                f"Available: {[rec.name for rec in recs]}"
            )

    for rec in selected:
        if rec.expensive:
            print(f"  {rec.name}: training on '{dataset}' (needs TensorFlow)...")
            try:
                rec.generate(ctx)
            except Exception as exc:
                print(f"    could not train {rec.name} "
                      f"({exc.__class__.__name__}: {exc}); skipping it.")
                continue
        else:
            print(f"  {rec.name}: generating...")
            rec.generate(ctx)
        rec.build_user_map(ctx)

    # Refresh the manifest's stage timestamps for every recommender that has a
    # prediction on disk (metrics are the diversity stage's job). Merged into
    # whatever is already there.
    manifest = load_manifest(ctx.out_dir)
    for rec in recs:
        if os.path.exists(rec.raw_path(ctx)):
            record_stage_times(manifest.setdefault(rec.name, {}),
                               rec.raw_path(ctx), rec.processed_path(ctx))
    save_manifest(ctx.out_dir, manifest)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m recommender_module",
        description="Generate a dataset's recommender predictions (no scoring).",
    )
    p.add_argument("dataset", choices=list(DATASETS),
                   help="dataset to run on")
    p.add_argument("recommender", nargs="?",
                   help="a single recommender to generate, e.g. random, popular, "
                        "ground_truth, nrms, lstur, naml (which ones exist depends "
                        "on the dataset). Omit and use --all to run them all.")
    p.add_argument("--all", action="store_true",
                   help="run every recommender on the dataset — WARNING: this can "
                        "take a long time, as it trains every model")
    args = p.parse_args(argv)

    if bool(args.recommender) == args.all:
        p.error("specify exactly one of: a recommender name, or --all")

    if args.all:
        print(f"Running every recommender on '{args.dataset}'. This can take a "
              f"while — it trains every model (needs TensorFlow).")
    generate(args.dataset, only=None if args.all else args.recommender)


if __name__ == "__main__":
    main()