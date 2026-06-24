# Taken from recommenders notebook in quickstart
# TODO: link and give props

import os
import sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

# Epochs should be set to 5, only set to 2 for testing reasons
epochs = 2
seed = 42
batch_size = 32

# This script lives in recommender_module/mind_specific/, two levels below root.
_project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Raw inputs (train/dev splits, utils, model checkpoints) live under datasets/.
mind_dir = os.path.join(_project_dir, "data", "datasets", "mind")

train_news_file = os.path.join(mind_dir, "MINDsmall_train", "news.tsv")
train_behaviors_file = os.path.join(mind_dir, "MINDsmall_train", "behaviors.tsv")
valid_news_file = os.path.join(mind_dir, "MINDsmall_dev", "news.tsv")
valid_behaviors_file = os.path.join(mind_dir, "MINDsmall_dev", "behaviors.tsv")

utils_dir = os.path.join(mind_dir, "utils")
wordEmb_file = os.path.join(utils_dir, "embedding.npy")
userDict_file = os.path.join(utils_dir, "uid2index.pkl")
wordDict_file = os.path.join(utils_dir, "word_dict.pkl")
yaml_file = os.path.join(utils_dir, "nrms.yaml")

# Download utils (embedding, dictionaries, yaml) only if not already present
if not os.path.exists(yaml_file):
    _, _, _, mind_utils = get_mind_data_set('small')
    download_deeprec_resources(
        r'https://huggingface.co/dataset_module/Recommenders/MIND/resolve/main/',
        utils_dir, mind_utils
    )

# Set parameters
hparams = prepare_hparams(yaml_file,
    wordEmb_file=wordEmb_file,
    wordDict_file=wordDict_file,
    userDict_file=userDict_file,
    batch_size=batch_size,
    epochs=epochs,
    show_step=10)

print(hparams)

# Training
iterator = MINDIterator
model = NRMSModel(hparams, iterator, seed=seed)
print(model.run_eval(valid_news_file, valid_behaviors_file))

# Fit the model
model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)

res_syn = model.run_eval(valid_news_file, valid_behaviors_file)
print(res_syn)

# Save the model
model_path = os.path.join(mind_dir, "model")
os.makedirs(model_path, exist_ok=True)

model.model.save_weights(os.path.join(model_path, "nrms_ckpt"))

# Output: full-rank predictions live under data_processed/mind/predictions/.
predictions_dir = os.path.join(
    _project_dir, "data", "data_processed", "mind", "predictions"
)
os.makedirs(predictions_dir, exist_ok=True)

group_impr_indexes, group_labels, group_preds = model.run_fast_eval(valid_news_file, valid_behaviors_file)

with open(os.path.join(predictions_dir, 'prediction_nrms.txt'), 'w') as f:
    for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
        impr_index += 1
        pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
        pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
        f.write(' '.join([str(impr_index), pred_rank]) + '\n')