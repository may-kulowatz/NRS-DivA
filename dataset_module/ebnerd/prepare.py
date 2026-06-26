"""Ensure the eb-nerd dataset's raw inputs exist.

The preparation analog of ``ebnerd/adapter``. eb-nerd has no public
direct-download URL, so unlike MIND there is nothing to fetch — preparation only
*verifies* the inputs are present and raises with download instructions if not.
It exposes the two functions every ``dataset_module`` prepare module shares so
the pipeline can call them interchangeably (see ``config.DATASETS``):

  * ``ensure_raw_data(in_dir)`` — checks the essential inputs the adapter reads
    (articles.parquet + the validation behaviors.parquet); raises if missing.
  * ``ensure_utils(in_dir)`` — no-op: eb-nerd's content diversity uses the
    contrastive vectors shipped with the dataset (contrastive_vector.parquet),
    so there is no optional bundle to fetch.
"""

import os

from dataset_module.common import default_input_dir

# The dataset's folder name under data/datasets/ (its default standalone location).
DIR = "ebnerd"

# eb-nerd inputs the pipeline reads, as (subdir..., filename) under the dataset
# directory. There is no public direct-download URL, so these are only checked.
_EBNERD_REQUIRED = (
    ("articles.parquet",),
    ("validation", "behaviors.parquet"),
)


def ensure_raw_data(in_dir):
    """Verify eb-nerd's inputs are present under ``in_dir``; raise if not.

    ``in_dir`` is the dataset's input directory (e.g. data/datasets/ebnerd).
    Always returns False (nothing is ever downloaded).
    """
    missing = [
        os.path.join(*parts)
        for parts in _EBNERD_REQUIRED
        if not os.path.exists(os.path.join(in_dir, *parts))
    ]
    if missing:
        raise FileNotFoundError(
            f"eb-nerd inputs missing under {in_dir}: {', '.join(missing)}. "
            "eb-nerd has no public direct-download URL; download the dataset from "
            "https://recsys.eb.dk/dataset/ and unpack it there."
        )
    return False


def ensure_utils(in_dir):
    """No optional bundle to fetch for eb-nerd; content diversity is precomputed."""
    return False


if __name__ == "__main__":
    _in_dir = default_input_dir(DIR)
    print(f"Checking eb-nerd inputs in {_in_dir} ...")
    ensure_raw_data(_in_dir)
    print("  eb-nerd inputs present.")
