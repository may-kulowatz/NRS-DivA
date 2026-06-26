"""Ensure the MIND dataset's raw inputs exist, fetching whatever is missing.

The preparation analog of ``mind/adapter`` — where the adapter *parses* MIND's
files, this module *acquires* them. It exposes the two functions every
``dataset_module`` prepare module shares, so the pipeline can call them
interchangeably (see the ``"prepare"`` entries in ``config.DATASETS``):

  * ``ensure_raw_data(in_dir)`` — guarantees the *essential* inputs a run can't
    proceed without (the MIND 'small' dev news/behaviors split). Called once,
    eagerly, at the start of a run; a failure here aborts the run, because there
    is nothing to compute without them.

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

# The MIND 'small' dev split the pipeline reads (news.tsv + behaviors.tsv), and
# the subfolder the MINDsmall_dev.zip bundle is unpacked into.
_MIND_SPLIT_DIR = "MINDsmall_dev"
_MIND_SPLIT_FILES = ("news.tsv", "behaviors.tsv")

# If either of these utils is absent we re-download the whole utils bundle, which
# also restores the dictionaries and .yaml configs the training scripts use.
_MIND_UTILS_REQUIRED = ("embedding.npy", "word_dict.pkl")


def ensure_raw_data(in_dir):
    """Download the MIND 'small' dev split into ``in_dir/MINDsmall_dev`` if missing.

    ``in_dir`` is the dataset's input directory (e.g. data/datasets/mind).
    Returns True if a download happened, False if everything was already present.
    """
    split_dir = os.path.join(in_dir, _MIND_SPLIT_DIR)
    if all(os.path.exists(os.path.join(split_dir, f)) for f in _MIND_SPLIT_FILES):
        return False

    # Imported lazily — only needed when something is actually missing.
    from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
    from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

    _, _, mind_dev, _ = get_mind_data_set("small")
    print(f"MIND dev split missing — downloading into {split_dir} ...")
    download_deeprec_resources(_MIND_RESOURCES_URL, split_dir, mind_dev)
    return True


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
