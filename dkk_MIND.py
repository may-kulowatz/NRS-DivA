"""
EchoBench: MIND_demo
Author: recommenders team and Marlene Kulowatz

This code basis was taken from the github respository of the recommenders team and adjusted.
It is an example implementation of the DKN Recommender System on the MIND_demo dataset.

It downloads and loads the data automatically, so no raw data is required prior to running this file.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from collections import defaultdict
from tempfile import TemporaryDirectory
import tensorflow as tf
tf.get_logger().setLevel("ERROR") # only show error messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources, prepare_hparams
from recommenders.models.deeprec.models.dkn import DKN
from recommenders.models.deeprec.io.dkn_iterator import DKNTextIterator

REC_SCORE_THRESHOLD = 0.8  # articles scoring >= this are considered recommended

_project_dir = os.path.dirname(os.path.abspath(__file__))
MIND_NEWS_FILES = [
    os.path.join(_project_dir, "data", "MIND", "MINDsmall_train", "news.tsv"),
    os.path.join(_project_dir, "data", "MIND", "MINDsmall_dev", "news.tsv"),
]

# Since this uses mind_demo, there is no news.tsv. We took those from the MINDsmall_dataset and hope it works.
def load_news_topics(news_files):
    """Load news_id -> {category, subcategory} from one or more MIND news.tsv files."""
    topics = {}
    for path in news_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                cols = line.strip().split("\t")
                if cols[0] not in topics:
                    topics[cols[0]] = {"category": cols[1], "subcategory": cols[2]}
    return topics


def parse_test_file(filepath):
    """Parse DKN test file into (impression_id, news_id, label) tuples."""
    records = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("%")
            imp_id = parts[1].strip() if len(parts) == 2 else "0"
            cols = parts[0].strip().split(" ")
            records.append((imp_id, cols[2], float(cols[0])))
    return records


def _unique_topic_ratio(topics):
    if not topics:
        return 0.0
    return len(set(topics)) / len(topics)

# Computes 4 scores: topic-diversity for rec-results and actual results for both main category and subcategory
def compute_diversity_metrics(records, scores, news_topics, threshold):
    """Compute topic-diversity-rec and topic-diversity-actual at category and subcategory level."""
    impressions = defaultdict(list)
    for (imp_id, news_id, label), score in zip(records, scores):
        impressions[imp_id].append((score, label, news_id))

    rec_cat, rec_sub, act_cat, act_sub = [], [], [], []
    for items in impressions.values():
        ranked = sorted(items, key=lambda x: x[0], reverse=True)

        rec_ids = [nid for score, _, nid in ranked if score >= threshold]
        if not rec_ids:
            rec_ids = [ranked[0][2]]

        clicked_ids = [nid for _, lbl, nid in items if lbl == 1.0]

        rec_cat.append(_unique_topic_ratio(
            [news_topics[n]["category"] for n in rec_ids if n in news_topics]
        ))
        rec_sub.append(_unique_topic_ratio(
            [news_topics[n]["subcategory"] for n in rec_ids if n in news_topics]
        ))
        if clicked_ids:
            act_cat.append(_unique_topic_ratio(
                [news_topics[n]["category"] for n in clicked_ids if n in news_topics]
            ))
            act_sub.append(_unique_topic_ratio(
                [news_topics[n]["subcategory"] for n in clicked_ids if n in news_topics]
            ))

    return {
        "topic-diversity-rec-cat": sum(rec_cat) / len(rec_cat),
        "topic-diversity-rec-sub": sum(rec_sub) / len(rec_sub),
        "topic-diversity-actual-cat": sum(act_cat) / len(act_cat) if act_cat else 0.0,
        "topic-diversity-actual-sub": sum(act_sub) / len(act_sub) if act_sub else 0.0,
    }

print(f"System version: {sys.version}")
print(f"Tensorflow version: {tf.__version__}")
print("Hello Recommender")


# Download and load the data

tmpdir = TemporaryDirectory()
data_path = os.path.join(tmpdir.name, "mind-demo-dkn")

yaml_file = os.path.join(data_path, "dkn.yaml")
train_file = os.path.join(data_path, "train_mind_demo.txt")
valid_file = os.path.join(data_path, "valid_mind_demo.txt")
test_file = os.path.join(data_path, "test_mind_demo.txt")
news_feature_file = os.path.join(data_path, "doc_feature.txt")
user_history_file = os.path.join(data_path, "user_history.txt")
wordEmb_file = os.path.join(data_path, "word_embeddings_100.npy")
entityEmb_file = os.path.join(data_path, "TransE_entity2vec_100.npy")
contextEmb_file = os.path.join(data_path, "TransE_context2vec_100.npy")
if not os.path.exists(yaml_file):
    download_deeprec_resources("https://raw.githubusercontent.com/recommenders-team/resources/main/deeprec/",
                               tmpdir.name, "mind-demo-dkn.zip")

# Train the model
# Setup parameters, can be changed
EPOCHS = 1
HISTORY_SIZE = 50
BATCH_SIZE = 500

hparams = prepare_hparams(yaml_file,
                          news_feature_file = news_feature_file,
                          user_history_file = user_history_file,
                          wordEmb_file=wordEmb_file,
                          entityEmb_file=entityEmb_file,
                          contextEmb_file=contextEmb_file,
                          epochs=EPOCHS,
                          history_size=HISTORY_SIZE,
                          batch_size=BATCH_SIZE)
print(hparams)

# Set up the model
print("Setting up model...", flush=True)
model = DKN(hparams, DKNTextIterator)
print("Running pre-train eval on valid set...", flush=True)
print(model.run_eval(valid_file))

# Fit the model
print("Starting model.fit...", flush=True)
model.fit(train_file, valid_file)
print("model.fit done.", flush=True)

# Print results
print("Running eval on test set...", flush=True)
res = model.run_eval(test_file)
print(res)

# Topic diversity metrics
print("Computing topic diversity...", flush=True)
news_topics = load_news_topics(MIND_NEWS_FILES)
pred_file = os.path.join(tmpdir.name, "predictions.txt")
model.predict(test_file, pred_file)
scores = [float(line) for line in open(pred_file, encoding="utf-8") if line.strip()]
records = parse_test_file(test_file)
div_res = compute_diversity_metrics(records, scores, news_topics, REC_SCORE_THRESHOLD)
print(div_res)