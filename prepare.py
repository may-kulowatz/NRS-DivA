"""Ensure the pipeline's raw input data exists, fetching whatever is missing.

The pipeline reads several large inputs that are kept out of git (the dataset
splits and the MIND embedding/dictionary bundle; see .gitignore). A fresh
checkout — or a stray ``git clean`` — can therefore leave them absent. This
module is the single place responsible for getting them back:

  * ``ensure_raw_data(dataset, in_dir)`` — guarantees the *essential* inputs a
    run can't proceed without (the news/behaviors splits). Called once, eagerly,
    at the start of a run; a failure here aborts the run, because there is
    nothing to compute without them.

  * ``ensure_mind_utils(mind_dir)`` — fetches the *optional* MIND utils bundle
    (word embeddings, dictionaries, model .yaml configs) that only content
    diversity needs. Called lazily, just before content diversity reads the
    embeddings, so that if it can't be obtained the metric is skipped rather than
    the whole run failing.

The MIND resources are downloaded from Hugging Face: the Recommenders library's
built-in URL (recodatasets.z20.web.core.windows.net) is dead, so we pass this
base URL explicitly and use get_mind_data_set only for the bundle file names
(e.g. "MINDsmall_dev.zip"). eb-nerd has no public direct-download URL, so its
inputs can only be checked for presence, with an instructive error if missing.

The Recommenders library (and the TensorFlow it pulls in) is imported lazily, so
importing this module — and the pipeline that uses it — stays cheap and only needs
Recommenders installed when a download actually has to happen.
"""

import os

_MIND_RESOURCES_URL = "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/"

# The MIND 'small' dev split the pipeline reads (news.tsv + behaviors.tsv), and
# the subfolder the MINDsmall_dev.zip bundle is unpacked into.
_MIND_SPLIT_DIR = "MINDsmall_dev"
_MIND_SPLIT_FILES = ("news.tsv", "behaviors.tsv")

# If either of these utils is absent we re-download the whole utils bundle, which
# also restores the dictionaries and .yaml configs the training scripts use.
_MIND_UTILS_REQUIRED = ("embedding.npy", "word_dict.pkl")

# eb-nerd inputs the pipeline reads, as (subdir..., filename) under the dataset
# directory. There is no public direct-download URL, so these are only checked.
_EBNERD_REQUIRED = (
    ("articles.parquet",),
    ("validation", "behaviors.parquet"),
)


def ensure_raw_data(dataset, in_dir):
    """Ensure ``dataset``'s essential pipeline inputs exist under ``in_dir``.

    ``in_dir`` is the dataset's input directory (e.g. data/datasets/mind).
    Downloads any missing inputs that have a known source; raises if an input is
    missing and cannot be fetched automatically. Returns True if a download
    happened, False if everything was already present.
    """
    if dataset == "MIND":
        return _ensure_mind_split(in_dir)
    if dataset == "ebnerd":
        _require_ebnerd_inputs(in_dir)
        return False
    raise ValueError(f"Unknown dataset {dataset!r}")


def _ensure_mind_split(mind_dir):
    """Download the MIND 'small' dev split into ``mind_dir/MINDsmall_dev`` if missing."""
    split_dir = os.path.join(mind_dir, _MIND_SPLIT_DIR)
    if all(os.path.exists(os.path.join(split_dir, f)) for f in _MIND_SPLIT_FILES):
        return False

    # Imported lazily — only needed when something is actually missing.
    from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
    from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

    _, _, mind_dev, _ = get_mind_data_set("small")
    print(f"MIND dev split missing — downloading into {split_dir} ...")
    download_deeprec_resources(_MIND_RESOURCES_URL, split_dir, mind_dev)
    return True


def _require_ebnerd_inputs(ebnerd_dir):
    """Verify eb-nerd's inputs are present; raise with instructions if not."""
    missing = [
        os.path.join(*parts)
        for parts in _EBNERD_REQUIRED
        if not os.path.exists(os.path.join(ebnerd_dir, *parts))
    ]
    if missing:
        raise FileNotFoundError(
            f"eb-nerd inputs missing under {ebnerd_dir}: {', '.join(missing)}. "
            "eb-nerd has no public direct-download URL; download the dataset from "
            "https://recsys.eb.dk/dataset/ and unpack it there."
        )


def ensure_mind_utils(mind_dir):
    """Download the MIND 'small' utils into ``mind_dir/utils`` if they're missing.

    ``mind_dir`` is a dataset input directory (e.g. data/datasets/mind). Returns
    True if a download happened, False if the files were already present. Raises
    on network/download errors — callers that want to degrade gracefully should
    catch the exception.
    """
    utils_dir = os.path.join(mind_dir, "utils")
    if all(os.path.exists(os.path.join(utils_dir, f)) for f in _MIND_UTILS_REQUIRED):
        return False

    # Imported lazily — only needed when something is actually missing.
    from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
    from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

    os.makedirs(utils_dir, exist_ok=True)
    _, _, _, mind_utils = get_mind_data_set("small")
    print(f"MIND utils missing — downloading into {utils_dir} ...")
    download_deeprec_resources(_MIND_RESOURCES_URL, utils_dir, mind_utils)
    return True