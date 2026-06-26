# LSTUR news recommender for eb-nerd.
#
# This is the LSTUR analogue of `nrms_ebnerd.py`, the way `lstur_mind.py` mirrors
# `nrms_mind.py` on the MIND side: same `run(...)` contract, same data pipeline,
# same MIND-format `"{impr_id} [ranks]"` output — only the *model* (and the bits
# LSTUR needs that NRMS doesn't) differ.
#
# The one real difference from NRMS: LSTUR ("Neural News Recommendation with
# Long- and Short-term User Representations") keeps a **long-term, per-user
# embedding**. So on top of the article-title encoding that NRMS already uses,
# LSTUR also needs:
#   * a user -> integer-index mapping (so each user has a row in the user
#     embedding table), built here with `create_user_id_to_int_mapping`;
#   * `hparams_lstur.n_users` set to the size of that table; and
#   * the `LSTURDataLoader` (it additionally yields the user index per sample),
#     instead of the `NRMSDataLoader`.
# Everything else (article tokenisation, training loop, writing the prediction
# file) is identical to the NRMS trainer.
#
# As in `nrms_ebnerd.py`, everything from the EBNeRD quick-start that isn't needed
# to *produce the prediction file* (metrics evaluation, TensorBoard, checkpoint /
# early-stopping callbacks, notebook `.head()` previews) is left out / commented.

import os
import datetime
from pathlib import Path

import polars as pl
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel

# Column-name constants (DEFAULT_USER_COL, DEFAULT_TITLE_COL, ...).
from data.datasets.ebnerd.utils._constants import *

from data.datasets.ebnerd.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_prediction_scores,
    ebnerd_from_path,
    # LSTUR-specific: builds {user_id: contiguous integer index} for the user
    # embedding table. (NRMS doesn't need this — it has no per-user parameters.)
    create_user_id_to_int_mapping,
)
# Metrics are not needed to generate the prediction file:
# from data.datasets.ebnerd.utils import MetricEvaluator, AucScore, NdcgScore, MrrScore
from data.datasets.ebnerd.utils._articles import (
    convert_text2encoding_with_transformers,
    create_article_id_to_value_mapping,
)
from data.datasets.ebnerd.utils._polars import concat_str_columns
from data.datasets.ebnerd.utils._nlp import get_transformers_word_embeddings
from data.datasets.ebnerd.utils._python import rank_predictions_by_score

# LSTUR-specific imports: its own dataloader, hyper-parameters, and model — the
# direct counterparts of the NRMS trio used in nrms_ebnerd.py.
from data.datasets.ebnerd.utils.dataloader import LSTURDataLoader
from data.datasets.ebnerd.utils.model_config import hparams_lstur
from data.datasets.ebnerd.utils import LSTURModel

# Title text encoder + how much of the article title / user history to use.
# (Same as the NRMS trainer — the article side of the model is identical.)
TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
MAX_TITLE_LENGTH = 30
HISTORY_SIZE = 20


def run(dataset_dir, train_split, dev_split, prediction_file,
        *, epochs=1, seed=42, batch_size=32, fraction=0.01):
    """Train LSTUR on eb-nerd and write its full-rank predictions.

    Signature is identical to `nrms_ebnerd.run` (and to the MIND trainers), so the
    pipeline calls it the same way:

    dataset_dir     : the dataset's input dir (data/datasets/ebnerd) — holds
                      articles.parquet plus the split sub-folders, each with
                      behaviors.parquet + history.parquet.
    train_split     : training split sub-folder (e.g. "train").
    dev_split       : validation split sub-folder (e.g. "validation"); the
                      prediction file is written for *its* impressions.
    prediction_file : output path for the "{impr_id} [ranks]" file. Same format the
                      MIND model trainers write: ranks align to article_ids_inview
                      order and the line is keyed by the real impression_id, so the
                      pipeline reads it through the ordinary rank readers.

    `fraction` down-samples the *training* data for speed (the quick-start default);
    predictions are written for the full dev split so every impression is covered.
    """
    # ebnerd_from_path / pl.read_parquet take a pathlib.Path (it calls
    # `.joinpath(...)` internally), so use Path for the dataset paths.
    base = Path(dataset_dir)

    # Let TF grow GPU memory on demand instead of grabbing it all up front.
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # The history length the model attends over. Set on the hparams before the
    # model is built.
    hparams_lstur.history_size = HISTORY_SIZE

    # Only load the columns the dataloader needs (user, impression id + time,
    # history, clicked, in-view candidates).
    columns = [
        DEFAULT_USER_COL,
        DEFAULT_IMPRESSION_ID_COL,
        DEFAULT_IMPRESSION_TIMESTAMP_COL,
        DEFAULT_HISTORY_ARTICLE_ID_COL,
        DEFAULT_CLICKED_ARTICLES_COL,
        DEFAULT_INVIEW_ARTICLES_COL,
    ]

    # --- Training data: down-sampled, split by time into a train + small val set ---
    # `sampling_strategy_wu2019` builds the negative samples (npratio negatives per
    # positive) the news models train on; `create_binary_labels_column` adds the
    # 0/1 label list aligned to the in-view candidates.
    df = (
        ebnerd_from_path(
            base / train_split, history_size=HISTORY_SIZE, padding=0
        )
        .select(columns)
        .pipe(
            sampling_strategy_wu2019,
            npratio=4, shuffle=True, with_replacement=True, seed=seed,
        )
        .pipe(create_binary_labels_column)
        .sample(fraction=fraction)
    )
    dt_split = pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).max() - datetime.timedelta(days=1)
    df_train = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL) < dt_split)
    df_validation = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL) >= dt_split)
    print(f"Train samples: {df_train.height}\nValidation samples: {df_validation.height}")

    # --- LSTUR-specific: the user embedding table -----------------------------
    # LSTUR learns a long-term vector per user, so each user needs a stable integer
    # index. Build the mapping from the *whole* training set `df` (so both the
    # train and the in-fit validation users get a real row), then size the user
    # embedding to match. The model allocates `n_users + 1` rows internally, and
    # the dataloader maps any user it doesn't know (e.g. dev-split users unseen in
    # training) to index 0, so there is always room for them.
    user_id_mapping = create_user_id_to_int_mapping(df)
    hparams_lstur.n_users = len(user_id_mapping)

    # --- Dev split: the impressions we write predictions for (full, not sampled) ---
    df_test = (
        ebnerd_from_path(
            base / dev_split, history_size=HISTORY_SIZE, padding=0
        )
        .select(columns)
        .pipe(create_binary_labels_column)
    )

    # --- Article title embeddings via the HuggingFace transformer -------------
    # Identical to NRMS: tokenise the (sub)title with XLM-RoBERTa, take the model's
    # word embeddings as the news encoder's input table, and map article_id -> token
    # ids.
    df_articles = pl.read_parquet(base / "articles.parquet")
    transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
    transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    word2vec_embedding = get_transformers_word_embeddings(transformer_model)

    df_articles, cat_col = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
    df_articles, token_col_title = convert_text2encoding_with_transformers(
        df_articles, transformer_tokenizer, cat_col, max_length=MAX_TITLE_LENGTH
    )
    article_mapping = create_article_id_to_value_mapping(
        df=df_articles, value_col=token_col_title
    )

    # --- Dataloaders ----------------------------------------------------------
    # LSTURDataLoader == NRMSDataLoader plus `user_id_mapping`: it additionally
    # yields the per-sample user index that feeds the long-term user embedding.
    train_dataloader = LSTURDataLoader(
        behaviors=df_train, article_dict=article_mapping, user_id_mapping=user_id_mapping,
        unknown_representation="zeros", history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=False, batch_size=batch_size,
    )
    val_dataloader = LSTURDataLoader(
        behaviors=df_validation, article_dict=article_mapping, user_id_mapping=user_id_mapping,
        unknown_representation="zeros", history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=False, batch_size=batch_size,
    )

    # --- Model + training -----------------------------------------------------
    # LSTURModel takes the same (hparams, word2vec_embedding, seed) constructor as
    # NRMSModel and exposes the same `.model` (trainable Keras model) and `.scorer`
    # (inference model), so the fit/predict code below is unchanged from NRMS.
    model = LSTURModel(hparams=hparams_lstur, word2vec_embedding=word2vec_embedding, seed=seed)
    model.model.compile(
        optimizer=model.model.optimizer, loss=model.model.loss, metrics=["AUC"],
    )
    # Callbacks (TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau) are
    # not needed to produce the prediction file — train directly for `epochs` and use
    # the resulting in-memory weights.
    model.model.fit(train_dataloader, validation_data=val_dataloader, epochs=epochs)

    # --- Predict on the dev split and write the MIND-format prediction file ----
    test_dataloader = LSTURDataLoader(
        behaviors=df_test, article_dict=article_mapping, user_id_mapping=user_id_mapping,
        unknown_representation="zeros", history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=True, batch_size=batch_size,
    )
    pred_test = model.scorer.predict(test_dataloader)
    # `scores` is one score per in-view candidate, aligned to article_ids_inview.
    df_test = add_prediction_scores(df_test, pred_test.tolist())

    impression_ids = df_test[DEFAULT_IMPRESSION_ID_COL].to_list()
    scores = df_test["scores"].to_list()
    os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
    with open(prediction_file, "w") as f:
        for impr_id, candidate_scores in zip(impression_ids, scores):
            # rank_predictions_by_score: 1 = highest score, aligned to the in-view
            # order — exactly the "{impr_id} [ranks]" line the MIND trainers emit.
            ranks = rank_predictions_by_score(candidate_scores)
            f.write(f"{impr_id} [" + ",".join(str(int(r)) for r in ranks) + "]\n")

    # --- Metrics (commented out — not needed for the prediction file) ---
    # metrics = MetricEvaluator(
    #     labels=df_test["labels"].to_list(),
    #     predictions=df_test["scores"].to_list(),
    #     metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
    # )
    # print(metrics.evaluate())


if __name__ == "__main__":
    import sys
    # This file lives in recommender_module/ebnerd_specific/, two levels below root.
    _project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _project_dir)
    ebnerd_dir = os.path.join(_project_dir, "data", "datasets", "ebnerd")
    prediction_file = os.path.join(
        _project_dir, "data", "data_processed", "ebnerd", "predictions", "prediction_lstur.txt"
    )
    run(ebnerd_dir, "train", "validation", prediction_file)
