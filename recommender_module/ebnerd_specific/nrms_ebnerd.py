# NRMS news recommender for eb-nerd — adapted from the EBNeRD benchmark quick-start
# (examples/00_quick_start/nrms_ebnerd). Wrapped into `run()` so the pipeline can
# train it on demand and read its full-rank "{impr_id} [ranks]" prediction file —
# the SAME format the MIND model trainers write, so the pipeline parses it the same
# way. Everything not needed to produce that file (metrics evaluation, TensorBoard,
# checkpoint/early-stopping callbacks, notebook `.head()` previews) is commented out.

import os
import datetime
from pathlib import Path

import polars as pl
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel

from data.datasets.ebnerd.utils._constants import *

from data.datasets.ebnerd.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_prediction_scores,
    ebnerd_from_path,
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

from data.datasets.ebnerd.utils.dataloader import NRMSDataLoader
from data.datasets.ebnerd.utils.model_config import hparams_nrms
from data.datasets.ebnerd.utils import NRMSModel

# Title text encoder + how much of the article title / user history to use.
TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
MAX_TITLE_LENGTH = 30
HISTORY_SIZE = 20


def run(dataset_dir, train_split, dev_split, prediction_file,
        *, epochs=2, seed=42, batch_size=32):
    """Train NRMS on eb-nerd and write its full-rank predictions.

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

    Trains on the whole training split and predicts on the whole dev split — the
    parameters mirror the MIND NRMS trainer (epochs=2, seed=42, batch_size=32).
    """
    # ebnerd_from_path / pl.read_parquet take a pathlib.Path (it calls
    # `.joinpath(...)` internally), so use Path for the dataset paths.
    base = Path(dataset_dir)

    # Let TF grow GPU memory on demand instead of grabbing it all up front.
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    hparams_nrms.history_size = HISTORY_SIZE
    columns = [
        DEFAULT_USER_COL,
        DEFAULT_IMPRESSION_ID_COL,
        DEFAULT_IMPRESSION_TIMESTAMP_COL,
        DEFAULT_HISTORY_ARTICLE_ID_COL,
        DEFAULT_CLICKED_ARTICLES_COL,
        DEFAULT_INVIEW_ARTICLES_COL,
    ]

    # --- Training data: whole split, split by time into a train + small val set ---
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
    )
    dt_split = pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).max() - datetime.timedelta(days=1)
    df_train = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL) < dt_split)
    df_validation = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL) >= dt_split)
    print(f"Train samples: {df_train.height}\nValidation samples: {df_validation.height}")

    # --- Dev split: the impressions we write predictions for (whole split) ---
    df_test = (
        ebnerd_from_path(
            base / dev_split, history_size=HISTORY_SIZE, padding=0
        )
        .select(columns)
        .pipe(create_binary_labels_column)
    )
    print(f"Dev (prediction) samples: {df_test.height}")

    # --- Article title embeddings via the HuggingFace transformer ---
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

    # --- Dataloaders ---
    train_dataloader = NRMSDataLoader(
        behaviors=df_train, article_dict=article_mapping, unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL, eval_mode=False, batch_size=batch_size,
    )
    val_dataloader = NRMSDataLoader(
        behaviors=df_validation, article_dict=article_mapping, unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL, eval_mode=False, batch_size=batch_size,
    )

    # --- Model + training ---
    model = NRMSModel(hparams=hparams_nrms, word2vec_embedding=word2vec_embedding, seed=seed)
    model.model.compile(
        optimizer=model.model.optimizer, loss=model.model.loss, metrics=["AUC"],
    )
    # Callbacks (TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau) are
    # not needed to produce the prediction file — train directly for `epochs` and use
    # the resulting in-memory weights.
    model.model.fit(train_dataloader, validation_data=val_dataloader, epochs=epochs)

    # --- Predict on the dev split and write the MIND-format prediction file ---
    test_dataloader = NRMSDataLoader(
        behaviors=df_test, article_dict=article_mapping, unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL, eval_mode=True, batch_size=batch_size,
    )
    pred_test = model.scorer.predict(test_dataloader)
    df_test = add_prediction_scores(df_test, pred_test.tolist())

    impression_ids = df_test[DEFAULT_IMPRESSION_ID_COL].to_list()
    scores = df_test["scores"].to_list()
    os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
    with open(prediction_file, "w") as f:
        for impr_id, candidate_scores in zip(impression_ids, scores):
            ranks = rank_predictions_by_score(candidate_scores)  # 1 = highest score
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
        _project_dir, "data", "data_processed", "ebnerd", "predictions", "prediction_nrms.txt"
    )
    # Trains on the whole training split and predicts on the whole dev split, the
    # same way the pipeline invokes it.
    run(ebnerd_dir, "train", "validation", prediction_file)
