# NAML news recommender for eb-nerd.
#
# This is the NAML analogue of `nrms_ebnerd.py` / `lstur_ebnerd.py`: same
# `run(...)` contract, same data pipeline, and the SAME MIND-format
# `"{impr_id} [ranks]"` output, so the pipeline reads its predictions exactly the
# way it reads NRMS's and LSTUR's. Only the model — and the extra inputs NAML
# needs — differ.
#
# NAML ("Neural News Recommendation with Attentive Multi-View Learning", Wu et
# al., IJCAI 2019) encodes each article from FOUR views, not just the title:
#   * title   (word tokens, like NRMS/LSTUR)
#   * body    (word tokens)
#   * vert    (category, a single categorical id)
#   * subvert (subcategory, a single categorical id)
# So on top of the title tokenisation the other trainers already do, this file
# also builds body tokens and category/subcategory index mappings.
#
# TWO eb-nerd-specific wrinkles the mappings have to handle:
#   * eb-nerd's `category` ids are sparse (values up to ~2975 for only ~25
#     distinct categories), so they are REMAPPED to contiguous indices [1..N];
#     index 0 is reserved for unknown/pad. `hparams_naml.vert_num` /
#     `subvert_num` are then sized from the data so no id overflows the model's
#     category-embedding tables.
#   * eb-nerd's `subcategory` is a LIST per article (multi-valued), while NAML's
#     subvert view expects one id — we take the first subcategory (or unknown if
#     the list is empty), matching how the title/category are single-valued.
#
# THE PREDICTION PATH DIFFERS FROM NRMS/LSTUR ON PURPOSE. Those trainers get
# per-candidate scores from a dev `DataLoader(eval_mode=True)`, but eb-nerd's
# `NAMLDataLoader` explicitly does not implement eval mode. So instead of the
# dataloader we score with the trained sub-encoders directly: run the news
# encoder over every article once to get a news vector per article, run the user
# encoder over each impression's history to get a user vector, and take their dot
# product per in-view candidate. This yields exactly the same per-candidate
# scores the scorer would, and we write them in the identical MIND format — the
# vendored utils are left untouched.
#
# As in the sibling trainers, everything from the EBNeRD quick-start that isn't
# needed to *produce the prediction file* (metrics evaluation, TensorBoard,
# checkpoint / early-stopping callbacks, notebook `.head()` previews) is omitted.

import os
import datetime
from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel

# Column-name constants (DEFAULT_USER_COL, DEFAULT_TITLE_COL, DEFAULT_BODY_COL,
# DEFAULT_CATEGORY_COL, DEFAULT_SUBCATEGORY_COL, ...).
from data.datasets.ebnerd.utils._constants import *

from data.datasets.ebnerd.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    ebnerd_from_path,
)
from data.datasets.ebnerd.utils._articles import (
    convert_text2encoding_with_transformers,
    create_article_id_to_value_mapping,
)
from data.datasets.ebnerd.utils._nlp import get_transformers_word_embeddings
from data.datasets.ebnerd.utils._python import rank_predictions_by_score

# NAML-specific imports: its own dataloader, hyper-parameters, and model — the
# direct counterparts of the NRMS/LSTUR trio used in the sibling trainers.
from data.datasets.ebnerd.utils.dataloader import NAMLDataLoader
from data.datasets.ebnerd.utils.model_config import hparams_naml
from data.datasets.ebnerd.utils import NAMLModel

# Title/body text encoder + how much of the title, body and user history to use.
# (The title side matches the NRMS/LSTUR trainers; NAML adds the body.)
TRANSFORMER_MODEL_NAME = "FacebookAI/xlm-roberta-base"
MAX_TITLE_LENGTH = hparams_naml.title_size   # 30
MAX_BODY_LENGTH = hparams_naml.body_size     # 40
HISTORY_SIZE = 20


def _build_category_index(values):
    """Map sparse categorical ids to contiguous indices [1..N] (0 = unknown/pad).

    eb-nerd category / subcategory ids are sparse (few distinct values, but large
    numbers), which would overflow NAML's category-embedding tables. Remapping to
    a dense range keeps the tables small and the ids in bounds.
    """
    uniques = sorted({v for v in values if v is not None})
    return {v: i for i, v in enumerate(uniques, start=1)}


def _prepare_articles(df_articles, tokenizer):
    """Tokenise title/body and add a single-valued subvert column.

    Returns ``(df, title_col, body_col, cat2idx, subcat2idx)`` — the augmented
    articles frame plus the token column names and the contiguous category /
    subcategory index maps. Shared by both the model's dataloaders and the
    prediction lookups so they encode articles identically.
    """
    df_articles = df_articles.with_columns(
        pl.col(DEFAULT_TITLE_COL).fill_null(""),
        pl.col(DEFAULT_BODY_COL).fill_null(""),
        # subcategory is multi-valued (a list); take the first as the subvert id.
        pl.col(DEFAULT_SUBCATEGORY_COL).list.first().alias("_subvert"),
    )
    df_articles, title_col = convert_text2encoding_with_transformers(
        df_articles, tokenizer, DEFAULT_TITLE_COL, max_length=MAX_TITLE_LENGTH
    )
    df_articles, body_col = convert_text2encoding_with_transformers(
        df_articles, tokenizer, DEFAULT_BODY_COL, max_length=MAX_BODY_LENGTH
    )
    cat2idx = _build_category_index(df_articles[DEFAULT_CATEGORY_COL].to_list())
    subcat2idx = _build_category_index(df_articles["_subvert"].to_list())
    return df_articles, title_col, body_col, cat2idx, subcat2idx


def _article_input_lookups(df_articles, title_col, body_col, cat2idx, subcat2idx):
    """Assemble the per-article NAML input matrix used for prediction.

    Returns ``(concat_lookup, news_row_of)`` where ``concat_lookup`` is an
    ``(n_articles + 1, title+body+2)`` int32 matrix — one row per article in the
    layout NAML's news encoder consumes (title tokens | body tokens | vert |
    subvert), with row 0 the all-zero unknown/pad row — and ``news_row_of`` maps
    ``article_id -> row`` (defaulting to 0 = unknown).
    """
    art_ids = df_articles[DEFAULT_ARTICLE_ID_COL].to_list()
    titles = df_articles[title_col].to_list()
    bodies = df_articles[body_col].to_list()
    verts = df_articles[DEFAULT_CATEGORY_COL].to_list()
    subverts = df_articles["_subvert"].to_list()

    n = len(art_ids)
    title_lookup = np.zeros((n + 1, MAX_TITLE_LENGTH), dtype=np.int32)
    body_lookup = np.zeros((n + 1, MAX_BODY_LENGTH), dtype=np.int32)
    vert_lookup = np.zeros((n + 1,), dtype=np.int32)
    subvert_lookup = np.zeros((n + 1,), dtype=np.int32)
    news_row_of = {}
    for row, (aid, tt, bb, cc, ss) in enumerate(
        zip(art_ids, titles, bodies, verts, subverts), start=1
    ):
        news_row_of[aid] = row
        title_lookup[row] = tt
        body_lookup[row] = bb
        vert_lookup[row] = cat2idx.get(cc, 0)
        subvert_lookup[row] = subcat2idx.get(ss, 0)

    concat_lookup = np.concatenate(
        [title_lookup, body_lookup, vert_lookup[:, None], subvert_lookup[:, None]],
        axis=1,
    ).astype(np.int32)
    return concat_lookup, news_row_of


def run(dataset_dir, train_split, dev_split, prediction_file,
        *, epochs=2, seed=42, batch_size=32):
    """Train NAML on eb-nerd and write its full-rank predictions.

    Signature is identical to `nrms_ebnerd.run` / `lstur_ebnerd.run` (and the MIND
    trainers), so the pipeline calls it the same way:

    dataset_dir     : the dataset's input dir (data/datasets/ebnerd) — holds
                      articles.parquet plus the split sub-folders, each with
                      behaviors.parquet + history.parquet.
    train_split     : training split sub-folder (e.g. "train").
    dev_split       : validation split sub-folder (e.g. "validation"); the
                      prediction file is written for *its* impressions.
    prediction_file : output path for the "{impr_id} [ranks]" file — ranks align to
                      article_ids_inview order and the line is keyed by the real
                      impression_id, so the pipeline reads it through the ordinary
                      rank readers.

    Trains on the whole training split and predicts on the whole dev split.
    """
    # ebnerd_from_path / pl.read_parquet take a pathlib.Path (it calls
    # `.joinpath(...)` internally), so use Path for the dataset paths.
    base = Path(dataset_dir)

    # Let TF grow GPU memory on demand instead of grabbing it all up front.
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    hparams_naml.history_size = HISTORY_SIZE
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

    # --- Article inputs: title + body word tokens, category + subcategory ids ---
    df_articles = pl.read_parquet(base / "articles.parquet")
    transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
    transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    word2vec_embedding = get_transformers_word_embeddings(transformer_model)

    df_articles, title_col, body_col, cat2idx, subcat2idx = _prepare_articles(
        df_articles, transformer_tokenizer
    )
    # Size the category embeddings from the data so no remapped id overflows.
    hparams_naml.vert_num = len(cat2idx) + 1
    hparams_naml.subvert_num = len(subcat2idx) + 1

    # Per-view article_id -> value maps the NAMLDataLoader consumes.
    title_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=title_col)
    body_mapping = create_article_id_to_value_mapping(df=df_articles, value_col=body_col)
    category_mapping = {
        aid: cat2idx.get(c, 0)
        for aid, c in zip(
            df_articles[DEFAULT_ARTICLE_ID_COL].to_list(),
            df_articles[DEFAULT_CATEGORY_COL].to_list(),
        )
    }
    subcategory_mapping = {
        aid: subcat2idx.get(s, 0)
        for aid, s in zip(
            df_articles[DEFAULT_ARTICLE_ID_COL].to_list(),
            df_articles["_subvert"].to_list(),
        )
    }

    # --- Dataloaders (train / val only; NAMLDataLoader has no eval mode) --------
    train_dataloader = NAMLDataLoader(
        behaviors=df_train, article_dict=title_mapping, body_mapping=body_mapping,
        category_mapping=category_mapping, subcategory_mapping=subcategory_mapping,
        unknown_representation="zeros", history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=False, batch_size=batch_size,
    )
    val_dataloader = NAMLDataLoader(
        behaviors=df_validation, article_dict=title_mapping, body_mapping=body_mapping,
        category_mapping=category_mapping, subcategory_mapping=subcategory_mapping,
        unknown_representation="zeros", history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=False, batch_size=batch_size,
    )

    # --- Model + training -------------------------------------------------------
    model = NAMLModel(hparams=hparams_naml, word2vec_embedding=word2vec_embedding, seed=seed)
    model.model.compile(
        optimizer=model.model.optimizer, loss=model.model.loss, metrics=["AUC"],
    )
    model.model.fit(train_dataloader, validation_data=val_dataloader, epochs=epochs)

    # --- Predict on the dev split via the trained sub-encoders ------------------
    # NAMLDataLoader can't run eval mode, so we score without it: one news vector
    # per article (news encoder over every article), one user vector per
    # impression (user encoder over its padded history), then dot each in-view
    # candidate's news vector with the impression's user vector.
    concat_lookup, news_row_of = _article_input_lookups(
        df_articles, title_col, body_col, cat2idx, subcat2idx
    )
    news_vectors = model.newsencoder.predict(concat_lookup, batch_size=1024)

    history = df_test[DEFAULT_HISTORY_ARTICLE_ID_COL].to_list()
    hist_rows = np.array(
        [[news_row_of.get(a, 0) for a in h] for h in history], dtype=np.int64
    )
    hist_concat = concat_lookup[hist_rows]            # (n_impr, history_size, title+body+2)
    user_vectors = model.userencoder.predict(hist_concat, batch_size=batch_size)

    impression_ids = df_test[DEFAULT_IMPRESSION_ID_COL].to_list()
    inview = df_test[DEFAULT_INVIEW_ARTICLES_COL].to_list()
    os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
    with open(prediction_file, "w") as f:
        for impr_id, candidates, user_vec in zip(impression_ids, inview, user_vectors):
            rows = [news_row_of.get(a, 0) for a in candidates]
            candidate_scores = news_vectors[rows] @ user_vec  # one score per candidate
            ranks = rank_predictions_by_score(candidate_scores)  # 1 = highest score
            f.write(f"{impr_id} [" + ",".join(str(int(r)) for r in ranks) + "]\n")


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
