"""Interactive command-line for the pipeline.

Run with ``python cli.py [dataset]`` (or ``python pipeline.py [dataset]``, which
delegates here).
"""

import os
import sys

from config import DATASETS, output_dir
from recommender_module.common.io import processed_filename
from scores import load_manifest
from pipeline import run_pipeline


# Short labels for the extra content-embedding spaces (mirrors the dashboard).
_SPACE_LABELS = {"xlmr": "XLM-R", "bert": "BERT", "docvec": "doc2vec"}


def _metric_label(key):
    """Human label for a manifest metric key (incl. per-embedding-space variants)."""
    if key == "topic_diversity":
        return "topic diversity"
    if key == "content_diversity":
        return "content diversity"
    if key == "content_diversity_normalized":
        return "content diversity (normalized)"
    # content_diversity_<space> or content_diversity_normalized_<space>
    norm = "_normalized_" in key
    name = key.rsplit("_", 1)[-1]
    lbl = _SPACE_LABELS.get(name, name)
    return f"content diversity ({lbl}, normalized)" if norm else f"content diversity ({lbl})"


def _dataset_measures(cfg):
    """Diversity measures applicable to a dataset, as manifest metric keys.

    topic is always available; the default content measures need a
    `content_diversity` config, and each extra embedding space in
    `content_embeddings` adds its own content_diversity_<name> (+ _normalized_<name>)."""
    measures = ["topic_diversity"]
    if cfg["content_diversity"] is not None:
        measures += ["content_diversity", "content_diversity_normalized"]
    for name in cfg.get("content_embeddings", {}):
        measures += [f"content_diversity_{name}", f"content_diversity_normalized_{name}"]
    return measures


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
    # First positional argument that isn't a --flag (e.g. --normalized).
    positionals = [a for a in argv[1:] if not a.startswith("--")]
    arg = positionals[0] if positionals else None
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
    # --normalized makes the (expensive) normalized measure default to "yes".
    normalized = "--normalized" in argv
    cfg = DATASETS[dataset]
    out_dir = output_dir(dataset)
    raw_dir = os.path.join(out_dir, "predictions")
    processed_dir = os.path.join(out_dir, "predictions_processed")

    # Ground truth + the recommenders applicable to this dataset.
    recommenders = ["ground_truth", "random", "popular"] + cfg["model_recs"]

    # Diversity measures applicable to this dataset (content needs embeddings), and
    # which of them already have a value in the run manifest.
    measures = _dataset_measures(cfg)
    existing_scores = load_manifest(out_dir)

    def measure_present(m):
        return any(m in entry.get("metrics", {}) for entry in existing_scores.values())

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
    print("  raw predictions:")
    for name in recommenders:
        print(f"      {name:<13}: {mark(raw_path(name))}")
    print("  processed predictions:")
    for name in recommenders:
        print(f"      {name:<13}: {mark(proc_path(name))}")
    print("  diversity measures:")
    for m in measures:
        status = "present" if measure_present(m) else "missing"
        print(f"      {_metric_label(m):<34}: {status}")

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

    # 5 — rebuild processed files.
    force_processed = _ask_yes_no("\nRebuild processed per-user files?", default=False)

    # 6 — one question per diversity measure (default: (re)calculate the missing
    # cheap ones; the expensive normalized one is opt-in, defaulting to whether
    # --normalized was passed). Measures left unticked keep their existing scores.
    print("\nWhich diversity measures should be (re)calculated?")
    force_metrics = set()
    for m in measures:
        is_expensive = m.startswith("content_diversity_normalized")
        default = normalized if is_expensive else not measure_present(m)
        extra = " (slow; per-impression)" if is_expensive else ""
        if _ask_yes_no(f"  (re)calculate {_metric_label(m)}{extra}?", default=default):
            force_metrics.add(m)

    print()
    run_pipeline(
        dataset,
        force_recommenders=force_recommenders,
        force_processed=force_processed,
        # Only build what was explicitly asked for; the prompts already defaulted
        # missing cheap recommenders / measures to "yes".
        generate_missing=False,
        train_missing=False,
        metrics=force_metrics,
    )


if __name__ == "__main__":
    interactive_main(sys.argv)