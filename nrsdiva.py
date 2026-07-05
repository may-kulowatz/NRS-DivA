"""NRS-DivA — a one-command front door over the three stages.

Each stage stays independently runnable, for when you want just one:

    python -m dataset_module <dataset>            # prepare
    python -m recommender_module <dataset> --all  # generate
    python -m diversity_module <dataset> --all     # score

This wrapper simply runs them in order for a dataset, so a newcomer can benchmark
in a single command. It holds no pipeline logic of its own — it only delegates to
the stage entry points above.

    python -m nrsdiva <dataset>              # prepare -> generate -> score
    python -m nrsdiva <dataset> --train      # ...also (re)train the neural models
    python -m nrsdiva <dataset> --dashboard  # ...then open the dashboard

By default the generate step runs only the cheap recommenders (random, popular,
ground truth); any model predictions already on disk are still scored. Pass
``--train`` to (re)train the neural models too (needs TensorFlow; slow).
"""

import argparse
import subprocess
import sys

from config import DATASETS, resolve_dataset
from dataset_module.__main__ import main as prepare_stage
from recommender_module.__main__ import generate
from diversity_module.__main__ import score


def run(dataset, train=False, dashboard=False):
    """Run prepare -> generate -> score for one dataset (optionally then dashboard)."""
    dataset = resolve_dataset(dataset)
    print(f"=== NRS-DivA · {dataset} ===")

    print("\n[1/3] Preparing raw inputs...")
    prepare_stage([dataset])

    detail = "training every model" if train else "cheap recommenders only"
    print(f"\n[2/3] Generating recommendations ({detail})...")
    generate(dataset, skip_expensive=not train)

    print("\n[3/3] Scoring diversity...")
    score(dataset)

    manifest = f"data/data_processed/{DATASETS[dataset]['dir']}/run_manifest.json"
    print(f"\nDone. Scores written to {manifest}")

    if dashboard:
        print("Opening the dashboard (Ctrl-C to stop)...")
        subprocess.run([sys.executable, "-m", "solara", "run", "dashboard.py"])


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m nrsdiva",
        description="Run all three NRS-DivA stages for a dataset, in order.",
    )
    p.add_argument("dataset",
                   help=f"dataset to benchmark (one of {list(DATASETS)}, any case)")
    p.add_argument("--train", action="store_true",
                   help="also (re)train the neural models in the generate step "
                        "(needs TensorFlow; slow). Without it, only the cheap "
                        "recommenders are generated and any model predictions "
                        "already on disk are scored as-is.")
    p.add_argument("--dashboard", action="store_true",
                   help="open the Solara dashboard once scoring is done")
    args = p.parse_args(argv)

    try:
        run(args.dataset, train=args.train, dashboard=args.dashboard)
    except ValueError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
