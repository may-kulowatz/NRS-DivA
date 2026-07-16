# NAML news recommender (from the Microsoft Recommenders quickstart).
# Adapted from: https://github.com/recommenders-team/recommenders/
#
# Parameterized by dataset path: `run()` trains NAML on any MIND-format dataset
# (MIND, mind_news, ...), so the pipeline can hand it whichever dataset's paths it
# needs. Run this file directly (no args) to train on MIND.

import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # only show error messages

from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.naml import NAMLModel
from recommenders.models.newsrec.io.mind_all_iterator import MINDAllIterator

# NAML uses the full uid2index user mapping.
_USER_DICT = "uid2index.pkl"


def _embedding_size(dict_file):
    """Rows a category embedding needs for a 1-based {name: index} dict.

    Indices start at 1 (0 is the padding slot), so the table size is the largest
    index plus one — not len(dict), which would be one short whenever the indices
    are contiguous from 1.
    """
    with open(dict_file, "rb") as f:
        cat_dict = pickle.load(f)
    return max(cat_dict.values()) + 1


def run(dataset_dir, train_split, dev_split, prediction_file,
        *, epochs=2, seed=42, batch_size=32):
    """Train NAML on a MIND-format dataset and write its full-rank predictions.

    dataset_dir     : the dataset's input dir (holds the split folders + utils/).
    train_split     : training split sub-folder name (e.g. "MINDsmall_train").
    dev_split       : validation split sub-folder name (e.g. "MINDsmall_dev").
    prediction_file : output path for the "{impr_id} [ranks]" full-rank file.

    Epochs default to 2 (the original Microsoft Recommenders repo uses 5). NAML reads more of the utils/
    bundle than NRMS — the "_all" word embeddings/dict plus the vertical and
    sub-vertical category dicts — but the bundle ships them all together, so the
    same ensure_utils that the pipeline runs (and the __main__ block here) suffices.
    """
    train_news_file = os.path.join(dataset_dir, train_split, "news.tsv")
    train_behaviors_file = os.path.join(dataset_dir, train_split, "behaviors.tsv")
    valid_news_file = os.path.join(dataset_dir, dev_split, "news.tsv")
    valid_behaviors_file = os.path.join(dataset_dir, dev_split, "behaviors.tsv")

    utils_dir = os.path.join(dataset_dir, "utils")
    vert_dict_file = os.path.join(utils_dir, "vert_dict.pkl")
    subvert_dict_file = os.path.join(utils_dir, "subvert_dict.pkl")

    # The (sub-)vertical category counts in naml.yaml are the Microsoft MINDdemo
    # defaults and don't necessarily match the dicts this dataset shipped. Derive
    # the embedding sizes from the actual dicts: indices are 1-based (0 is the
    # padding slot), so the table needs max(index) + 1 rows.
    vert_num = _embedding_size(vert_dict_file)
    subvert_num = _embedding_size(subvert_dict_file)

    hparams = prepare_hparams(
        os.path.join(utils_dir, "naml.yaml"),
        wordEmb_file=os.path.join(utils_dir, "embedding_all.npy"),
        wordDict_file=os.path.join(utils_dir, "word_dict_all.pkl"),
        userDict_file=os.path.join(utils_dir, _USER_DICT),
        vertDict_file=vert_dict_file,
        subvertDict_file=subvert_dict_file,
        vert_num=vert_num,
        subvert_num=subvert_num,
        batch_size=batch_size,
        epochs=epochs,
        show_step=10,
    )
    print(hparams)

    # NAML uses the "all" iterator (it reads the vertical/sub-vertical + body).
    model = NAMLModel(hparams, MINDAllIterator, seed=seed)
    print(model.run_eval(valid_news_file, valid_behaviors_file))

    # Fit the model
    model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)
    print(model.run_eval(valid_news_file, valid_behaviors_file))

    # Save the model checkpoint alongside the dataset.
    model_dir = os.path.join(dataset_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model.model.save_weights(os.path.join(model_dir, "naml_ckpt"))

    # Write full-rank predictions ("{impr_id} [ranks]").
    os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
    group_impr_indexes, _group_labels, group_preds = model.run_fast_eval(
        valid_news_file, valid_behaviors_file
    )
    with open(prediction_file, "w") as f:
        for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
            impr_index += 1
            pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
            f.write(f"{impr_index} [" + ",".join(str(i) for i in pred_rank) + "]\n")


if __name__ == "__main__":
    print("System version: {}".format(sys.version))
    print("Tensorflow version: {}".format(tf.__version__))

    # This script lives in recommender_module/mind_specific/, two levels below root.
    _project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _project_dir)
    from dataset_module.mind.prepare import ensure_raw_data, ensure_utils

    mind_dir = os.path.join(_project_dir, "data", "datasets", "mind")
    ensure_raw_data(mind_dir)   # dev + train splits (train fetched best-effort)
    ensure_utils(mind_dir)

    prediction_file = os.path.join(
        _project_dir, "data", "data_processed", "mind", "predictions", "prediction_naml.txt"
    )
    run(mind_dir, "MINDsmall_train", "MINDsmall_dev", prediction_file)
