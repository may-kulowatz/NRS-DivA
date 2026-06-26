"""Interactive command-line for the pipeline.

Run with ``python cli.py [dataset]`` (or ``python pipeline.py [dataset]``, which
delegates here).
"""

import os
import sys

from config import DATASETS, output_dir
from recommender_module.common.io import processed_filename
from pipeline import run_pipeline


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
    which recommenders to (re)run and whether to rebuild the processed files — and
    run accordingly. Diversity scores are always recomputed from the processed
    files."""
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

    # 3 — rebuild processed files. (Diversity scores are always recomputed from
    # the processed files, so there is no separate "recalculate" prompt.)
    force_processed = _ask_yes_no("\nRebuild processed per-user files?", default=False)

    print()
    run_pipeline(
        dataset,
        force_recommenders=force_recommenders,
        force_processed=force_processed,
        # Only build what was explicitly asked for; the prompts already defaulted
        # missing cheap recommenders to "yes".
        generate_missing=False,
        train_missing=False,
    )


if __name__ == "__main__":
    interactive_main(sys.argv)