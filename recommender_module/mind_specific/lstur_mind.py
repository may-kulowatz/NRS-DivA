# LSTUR news recommender (from the Microsoft Recommenders quickstart).
# TODO: link and give props
#
# Parameterized by dataset path: `run()` trains LSTUR on any MIND-format dataset
# (MIND, mind_news, ...), so the pipeline can hand it whichever dataset's paths it
# needs. Run this file directly (no args) to train on MIND.

import os
import sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # only show error messages

from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.lstur import LSTURModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator

# LSTUR has a per-user embedding sized len(uid2index); the shipped uid2index.pkl
# is the MIND-large dict (indices up to 230117) and overflows it on MINDsmall.
# uid2index_small.pkl is a contiguous small-dataset mapping (0 = unknown user).
_USER_DICT = "uid2index_small.pkl"


def run(dataset_dir, train_split, dev_split, prediction_file,
        *, epochs=2, seed=40, batch_size=32):
    """Train LSTUR on a MIND-format dataset and write its full-rank predictions.

    dataset_dir     : the dataset's input dir (holds the split folders + utils/).
    train_split     : training split sub-folder name (e.g. "MINDsmall_train").
    dev_split       : validation split sub-folder name (e.g. "MINDsmall_dev").
    prediction_file : output path for the "{impr_id} [ranks]" full-rank file.

    Epochs should be 5; left at 2 for quick testing. The dataset's utils/ bundle
    (embeddings, dicts, lstur.yaml) must already exist — the pipeline ensures it
    before calling, and the __main__ block ensures it for standalone MIND runs.
    """
    train_news_file = os.path.join(dataset_dir, train_split, "news.tsv")
    train_behaviors_file = os.path.join(dataset_dir, train_split, "behaviors.tsv")
    valid_news_file = os.path.join(dataset_dir, dev_split, "news.tsv")
    valid_behaviors_file = os.path.join(dataset_dir, dev_split, "behaviors.tsv")

    utils_dir = os.path.join(dataset_dir, "utils")
    hparams = prepare_hparams(
        os.path.join(utils_dir, "lstur.yaml"),
        wordEmb_file=os.path.join(utils_dir, "embedding.npy"),
        wordDict_file=os.path.join(utils_dir, "word_dict.pkl"),
        userDict_file=os.path.join(utils_dir, _USER_DICT),
        batch_size=batch_size,
        epochs=epochs,
    )
    print(hparams)

    model = LSTURModel(hparams, MINDIterator, seed=seed)
    print(model.run_eval(valid_news_file, valid_behaviors_file))

    # Fit the model
    model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)
    print(model.run_eval(valid_news_file, valid_behaviors_file))

    # Save the model checkpoint alongside the dataset.
    model_dir = os.path.join(dataset_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model.model.save_weights(os.path.join(model_dir, "lstur_ckpt"))

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
    ensure_raw_data(mind_dir)   # dev split (train split must already exist)
    ensure_utils(mind_dir)

    prediction_file = os.path.join(
        _project_dir, "data", "data_processed", "mind", "predictions", "prediction_lstur.txt"
    )
    run(mind_dir, "MINDsmall_train", "MINDsmall_dev", prediction_file)