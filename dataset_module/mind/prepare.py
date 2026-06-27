"""Ensure the MIND dataset's raw inputs exist, fetching whatever is missing.

The preparation analog of ``mind/adapter`` — where the adapter *parses* MIND's
files, this module *acquires* them. It exposes the two functions every
``dataset_module`` prepare module shares, so the pipeline can call them
interchangeably (see the ``"prepare"`` entries in ``config.DATASETS``):

  * ``ensure_raw_data(in_dir)`` — fetches the MIND 'small' dev + train splits.
    The dev split is the *essential* input a run can't proceed without, so a
    failure to obtain it aborts the run; the train split is only needed to
    (re)train a model, so it is fetched best-effort and its absence doesn't block
    scoring. Called once, eagerly, at the start of a run.

  * ``ensure_utils(in_dir)`` — fetches the *optional* MIND utils bundle (word
    embeddings, dictionaries, model .yaml configs) that only content diversity
    needs. Called lazily, just before content diversity reads the embeddings, so
    that if it can't be obtained the metric is skipped rather than the whole run
    failing.

The Recommenders library (and the TensorFlow it pulls in) is imported lazily, so
importing this module stays cheap and only needs Recommenders installed when a
download actually has to happen.
"""

import os

from dataset_module.common import default_input_dir

# The dataset's folder name under data/datasets/ (its default standalone location).
DIR = "mind"

_MIND_RESOURCES_URL = "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/"

# The MIND 'small' splits the pipeline reads — each unpacks into its own subfolder
# holding news.tsv + behaviors.tsv. The dev split is the *essential* input (scored
# every run); the train split is only read when a model is (re)trained.
_MIND_DEV_DIR = "MINDsmall_dev"
_MIND_TRAIN_DIR = "MINDsmall_train"
_MIND_SPLIT_FILES = ("news.tsv", "behaviors.tsv")

# If either of these utils is absent we re-download the whole utils bundle, which
# also restores the dictionaries and .yaml configs the training scripts use.
_MIND_UTILS_REQUIRED = ("embedding.npy", "word_dict.pkl")


def _split_present(in_dir, split_dir):
    """True if a split subfolder has both news.tsv and behaviors.tsv."""
    return all(
        os.path.exists(os.path.join(in_dir, split_dir, f)) for f in _MIND_SPLIT_FILES
    )


def ensure_raw_data(in_dir):
    """Download the missing MIND 'small' splits (dev + train) into ``in_dir``.

    ``in_dir`` is the dataset's input directory (e.g. data/datasets/mind).

    The **dev** split (``MINDsmall_dev``) is essential — it is scored on every run —
    so a failure to obtain it propagates and aborts the run. The **train** split
    (``MINDsmall_train``) is only read when a model is (re)trained, so it is fetched
    *best-effort*: if it can't be downloaded the run still continues (scoring works;
    a later training attempt would report it missing).

    Returns True if any download happened, False if everything was already present.
    """
    dev_present = _split_present(in_dir, _MIND_DEV_DIR)
    train_present = _split_present(in_dir, _MIND_TRAIN_DIR)
    if dev_present and train_present:
        return False

    # Imported lazily — only needed when something is actually missing.
    from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
    from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

    _, mind_train, mind_dev, _ = get_mind_data_set("small")
    downloaded = False

    if not dev_present:
        dev_dir = os.path.join(in_dir, _MIND_DEV_DIR)
        print(f"MIND dev split missing — downloading into {dev_dir} ...")
        download_deeprec_resources(_MIND_RESOURCES_URL, dev_dir, mind_dev)
        downloaded = True

    if not train_present:
        train_dir = os.path.join(in_dir, _MIND_TRAIN_DIR)
        print(f"MIND train split missing — downloading into {train_dir} ...")
        try:
            download_deeprec_resources(_MIND_RESOURCES_URL, train_dir, mind_train)
            downloaded = True
        except Exception as exc:
            # Non-fatal: scoring doesn't need the train split. Only training does,
            # and that path reports the missing files itself.
            print(f"  could not download the MIND train split "
                  f"({exc.__class__.__name__}: {exc}); scoring still works, "
                  f"but training a model will need it.")

    return downloaded


def ensure_utils(in_dir):
    """Download the MIND 'small' utils into ``in_dir/utils`` if they're missing.

    ``in_dir`` is the dataset's input directory (e.g. data/datasets/mind). Returns
    True if a download happened, False if the files were already present. Raises
    on network/download errors — callers that want to degrade gracefully should
    catch the exception.
    """
    utils_dir = os.path.join(in_dir, "utils")
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


if __name__ == "__main__":
    _in_dir = default_input_dir(DIR)
    print(f"Preparing MIND in {_in_dir} ...")
    ensure_raw_data(_in_dir)
    ensure_utils(_in_dir)
