from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import polars as pl
import datetime

from data.datasets.ebnerd.utils._constants import *

from data.datasets.ebnerd.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_prediction_scores,
    truncate_history,
    ebnerd_from_path,
)
from data.datasets.ebnerd.utils import MetricEvaluator, AucScore, NdcgScore, MrrScore
from data.datasets.ebnerd.utils._articles import convert_text2encoding_with_transformers
from data.datasets.ebnerd.utils._polars import concat_str_columns, slice_join_dataframes
from data.datasets.ebnerd.utils._articles import create_article_id_to_value_mapping
from data.datasets.ebnerd.utils._nlp import get_transformers_word_embeddings
from data.datasets.ebnerd.utils._python import write_submission_file, rank_predictions_by_score

from data.datasets.ebnerd.utils.dataloader import NRMSDataLoader
from data.datasets.ebnerd.utils.model_config import hparams_nrms
from data.datasets.ebnerd.utils import NRMSModel

# List all physical devices
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

physical_devices = tf.config.list_physical_devices()
print("Available devices:", physical_devices)

PATH = Path("~/data/ebnerd").expanduser()
#
DATASPLIT = "ebnerd_small"
DUMP_DIR = Path("ebnerd_predictions")
DUMP_DIR.mkdir(exist_ok=True, parents=True)

HISTORY_SIZE = 20
hparams_nrms.history_size = HISTORY_SIZE

# We just want to load the necessary columns
COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
]
# This notebook is just a simple 'get-started'; we down sample the number of samples to just run quickly through it.
FRACTION = 0.01

df = (
    ebnerd_from_path(
        PATH.joinpath(DATASPLIT, "train"),
        history_size=HISTORY_SIZE,
        padding=0,
    )
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=4,
        shuffle=True,
        with_replacement=True,
        seed=123,
    )
    .pipe(create_binary_labels_column)
    .sample(fraction=FRACTION)
)

dt_split = pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).max() - datetime.timedelta(days=1)
df_train = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL) < dt_split)
df_validation = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL) >= dt_split)

print(f"Train samples: {df_train.height}\nValidation samples: {df_validation.height}")
df_train.head(2)




df_test = (
    ebnerd_from_path(
        PATH.joinpath(DATASPLIT, "validation"),
        history_size=HISTORY_SIZE,
        padding=0,
    )
    .select(COLUMNS)
    .pipe(create_binary_labels_column)
    .sample(fraction=FRACTION)
)


df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))
df_articles.head(2)

TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
MAX_TITLE_LENGTH = 30

# LOAD HUGGINGFACE:
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

# We'll init the word embeddings using the
word2vec_embedding = get_transformers_word_embeddings(transformer_model)
#
df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
# =>
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=token_col_title
)


# DATALOADING STUFF

BATCH_SIZE = 32

train_dataloader = NRMSDataLoader(
    behaviors=df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BATCH_SIZE,
)
val_dataloader = NRMSDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BATCH_SIZE,
)

# List all physical devices
physical_devices = tf.config.list_physical_devices()
print("Available devices:", physical_devices)


model = NRMSModel(
    hparams=hparams_nrms,
    word2vec_embedding=word2vec_embedding,
    seed=42,
)
model.model.compile(
    optimizer=model.model.optimizer,
    loss=model.model.loss,
    metrics=["AUC"],
)

MODEL_NAME = model.__class__.__name__
MODEL_WEIGHTS = DUMP_DIR.joinpath(f"state_dict/{MODEL_NAME}/weights")
LOG_DIR = DUMP_DIR.joinpath(f"runs/{MODEL_NAME}")


# Tensorboard:
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=LOG_DIR,
    histogram_freq=1,
)

# Earlystopping:
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=3,
    restore_best_weights=True,
)

# ModelCheckpoint:
modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_WEIGHTS,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

# Learning rate scheduler:
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc",
    mode="max",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
)

callbacks = [tensorboard_callback, early_stopping, modelcheckpoint, lr_scheduler]

USE_CALLBACKS = True
EPOCHS = 1

hist = model.model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=EPOCHS,
    callbacks=callbacks if USE_CALLBACKS else [],
)

if USE_CALLBACKS:
    _ = model.model.load_weights(filepath=MODEL_WEIGHTS)


BATCH_SIZE_TEST = 16

test_dataloader = NRMSDataLoader(
    behaviors=df_test,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=BATCH_SIZE_TEST,
)

pred_test = model.scorer.predict(test_dataloader)

df_test = add_prediction_scores(df_test, pred_test.tolist())
df_test.head(2)

# Metrics stuff ...

metrics = MetricEvaluator(
    labels=df_test["labels"].to_list(),
    predictions=df_test["scores"].to_list(),
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
metrics.evaluate()

df_test = df_test.with_columns(
    pl.col("scores")
    .map_elements(lambda x: list(rank_predictions_by_score(x)))
    .alias("ranked_scores")
)
df_test.head(2)