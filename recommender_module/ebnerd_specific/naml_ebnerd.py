# NAML news recommender for eb-nerd.
#
# This is the NAML analogue of `nrms_ebnerd.py` / `lstur_ebnerd.py`: same `run(...)`
# contract, same data pipeline, same MIND-format `"{impr_id} [ranks]"` output — only
# the *model* (and the extra article views NAML needs) differ.
#
# NAML ("Neural News Recommendation with Attentive Multi-View Learning") encodes each
# article from **four views** instead of NRMS's single title view:
#   * title  — tokenised (sub)title (same as NRMS);
#   * body   — tokenised article body;
#   * vertical (category) and sub-vertical (subcategory) — categorical ids embedded
#     in their own tables.
# So on top of the title token mapping NRMS already builds, NAML also needs a body
# token mapping plus article -> category / article -> subcategory index mappings, and
# `hparams_naml.vert_num` / `subvert_num` sized to those category tables. The
# `NAMLDataLoader` yields all four views per sample (history + candidates).
#
# As in the NRMS/LSTUR trainers, everything from the EBNeRD quick-start that isn't
# needed to *produce the prediction file* (metrics evaluation, TensorBoard, checkpoint
# / early-stopping callbacks, notebook `.head()` previews) is left out / commented.

import os
import datetime
from pathlib import Path

import polars as pl
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel

# Column-name constants (DEFAULT_USER_COL, DEFAULT_TITLE_COL, DEFAULT_BODY_COL, ...).
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

# NAML-specific imports: its own dataloader (four article views), hyper-parameters,
# and model — the direct counterparts of the NRMS trio used in nrms_ebnerd.py.
from data.datasets.ebnerd.utils.dataloader import NAMLDataLoader
from data.datasets.ebnerd.utils.model_config import hparams_naml
from data.datasets.ebnerd.utils import NAMLModel

# Title text encoder + how much of the article title / body / user history to use.
# (The title side is the same as the NRMS trainer; NAML adds the body view.)
TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
TEXT_COLUMNS_TO_USE = [DEFAULT_SUBTITLE_COL, DEFAULT_TITLE_COL]
MAX_TITLE_LENGTH = 30
MAX_BODY_LENGTH = 40
HISTORY_SIZE = 20


def _create_category_mapping(df_articles, category_col):
    """article_id -> contiguous 1-based category index (0 reserved for unknown/pad).

    The raw eb-nerd category ids are sparse (a couple dozen distinct values spread
    over a wide range), so dense-rank them into 1..K. Returns the mapping plus the
    embedding-table size it needs (K + 1, since index 0 is the unknown/padding slot
    the dataloader fills missing/empty values with).
    """
    idx_col = f"{category_col}_idx"
    df_idx = df_articles.select(DEFAULT_ARTICLE_ID_COL, category_col).with_columns(
        pl.col(category_col).rank("dense").fill_null(0).cast(pl.Int32).alias(idx_col)
    )
    mapping = create_article_id_to_value_mapping(
        df=df_idx, value_col=idx_col, article_col=DEFAULT_ARTICLE_ID_COL
    )
    num = int(df_idx[idx_col].max()) + 1
    return mapping, num


def run(dataset_dir, train_split, dev_split, prediction_file,
        *, epochs=2, seed=42, batch_size=32):
    """Train NAML on eb-nerd and write its full-rank predictions.

    Signature is identical to `nrms_ebnerd.run` / `lstur_ebnerd.run` (and to the MIND
    trainers), so the pipeline calls it the same way:

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
    parameters mirror the NRMS/LSTUR trainers (epochs=2, seed=42, batch_size=32).
    """
    # ebnerd_from_path / pl.read_parquet take a pathlib.Path (it calls
    # `.joinpath(...)` internally), so use Path for the dataset paths.
    base = Path(dataset_dir)

    # Let TF grow GPU memory on demand instead of grabbing it all up front.
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Input dimensions the model is built with. Set on the hparams before the model
    # is constructed (vert_num/subvert_num are filled in once the category tables are
    # known, below).
    hparams_naml.history_size = HISTORY_SIZE
    hparams_naml.title_size = MAX_TITLE_LENGTH
    hparams_naml.body_size = MAX_BODY_LENGTH

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

    # --- Training data: whole split, split by time into a train + small val set ---
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

    # --- Article views ---------------------------------------------------------
    # NAML encodes four views per article. Title + body are tokenised with the same
    # XLM-RoBERTa as NRMS; the model's word embeddings are the news encoder's input
    # table. Vertical/sub-vertical are categorical ids mapped to their own embeddings.
    df_articles = pl.read_parquet(base / "articles.parquet")
    transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
    transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    word2vec_embedding = get_transformers_word_embeddings(transformer_model)

    # Title view: subtitle + title concatenated (same as NRMS).
    df_articles, cat_col = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
    df_articles, token_col_title = convert_text2encoding_with_transformers(
        df_articles, transformer_tokenizer, cat_col, max_length=MAX_TITLE_LENGTH
    )
    # Body view: the article body, tokenised to MAX_BODY_LENGTH.
    df_articles, token_col_body = convert_text2encoding_with_transformers(
        df_articles, transformer_tokenizer, DEFAULT_BODY_COL, max_length=MAX_BODY_LENGTH
    )
    title_mapping = create_article_id_to_value_mapping(
        df=df_articles, value_col=token_col_title
    )
    body_mapping = create_article_id_to_value_mapping(
        df=df_articles, value_col=token_col_body
    )

    # Vertical (category) / sub-vertical (first subcategory) -> embedding indices.
    # Size the embedding tables to the actual category counts and feed them to the
    # hparams the model is built from.
    category_mapping, hparams_naml.vert_num = _create_category_mapping(
        df_articles, DEFAULT_CATEGORY_COL
    )
    df_articles = df_articles.with_columns(
        pl.col(DEFAULT_SUBCATEGORY_COL).list.first().alias("_subcategory_first")
    )
    subcategory_mapping, hparams_naml.subvert_num = _create_category_mapping(
        df_articles, "_subcategory_first"
    )

    # --- Dataloaders -----------------------------------------------------------
    # NAMLDataLoader == NRMSDataLoader plus the body + (sub)category mappings: it
    # yields all four views per sample.
    naml_kwargs = dict(
        article_dict=title_mapping, body_mapping=body_mapping,
        category_mapping=category_mapping, subcategory_mapping=subcategory_mapping,
        unknown_representation="zeros", history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        batch_size=batch_size,
    )
    train_dataloader = NAMLDataLoader(behaviors=df_train, eval_mode=False, **naml_kwargs)
    val_dataloader = NAMLDataLoader(behaviors=df_validation, eval_mode=False, **naml_kwargs)

    # --- Model + training ------------------------------------------------------
    # NAMLModel exposes the same `.model` (trainable Keras model) and `.scorer`
    # (inference model) as NRMS/LSTUR, so the fit/predict code below is unchanged.
    model = NAMLModel(hparams=hparams_naml, word2vec_embedding=word2vec_embedding, seed=seed)
    model.model.compile(
        optimizer=model.model.optimizer, loss=model.model.loss, metrics=["AUC"],
    )
    # Callbacks (TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau) are
    # not needed to produce the prediction file — train directly for `epochs` and use
    # the resulting in-memory weights.
    model.model.fit(train_dataloader, validation_data=val_dataloader, epochs=epochs)

    # --- Predict on the dev split and write the MIND-format prediction file -----
    test_dataloader = NAMLDataLoader(behaviors=df_test, eval_mode=True, **naml_kwargs)
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
        _project_dir, "data", "data_processed", "ebnerd", "predictions", "prediction_naml.txt"
    )
    # Trains on the whole training split and predicts on the whole dev split, the
    # same way the pipeline invokes it.
    run(ebnerd_dir, "train", "validation", prediction_file)