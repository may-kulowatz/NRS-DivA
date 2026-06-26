# `recommender_module`

The recommenders and the shared I/O that turns their output into the per-user
files the diversity metrics consume. Recommenders take normalized `Impression`
records (from `dataset_module`) and produce per-candidate scores or rankings;
the writers in `common/io.py` aggregate those into the on-disk formats.

```
base.py            Recommender interface + RunContext + build_recommenders
common/            dataset-agnostic recommenders + shared writers
  ground_truth.py    the articles users actually clicked (reference baseline)
  random_rec.py      uniform random scores
  popular_rec.py     prior-click-popularity scores (no future leakage)
  io.py              shared prediction / user-article-map writers
ebnerd_specific/   eb-nerd NRMS training script (uses the ebrec library)
mind_specific/     NRMS / LSTUR training scripts (parameterized by dataset path)
```

## `base.py` — the recommender interface

Wraps the `common/` recommenders behind one contract so the pipeline treats them
uniformly. A `Recommender` knows how to `generate(ctx)` its raw prediction file
and `build_user_map(ctx)` the processed per-user file; `RunContext` bundles the
inputs (impressions, article metadata, output paths, and the model-training
paths). `expensive = True` marks the model recommenders (NRMS / LSTUR) whose
(re)build means training a network.

- `RandomRecommender` / `PopularRecommender` — full-rank scorers; share the
  rank-based `build_user_map` via `_RankRecommender`.
- `GroundTruthRecommender` — writes `ground_truth.txt` at the dataset root and
  builds its map straight from that top-k file.
- `ModelRecommender` — trains on demand via the `mind_specific/` scripts (the
  `_MODEL_TRAINERS` dispatch); imported lazily so TensorFlow is only needed when a
  model is actually (re)trained.
- `build_recommenders(model_recs)` — returns a dataset's recommenders in
  scoring/display order: random, popular, its models, then ground truth.

> Raw-input fetching (MIND dev split + utils bundle, eb-nerd presence check)
> lives in the per-dataset `dataset_module/<name>/prepare.py` modules, not here —
> see that module.

---

## `common/` — dataset-agnostic

These operate purely on normalized in-memory structures and never touch a
dataset-specific file format. `numpy` and `tqdm` are required.

### `ground_truth.py`

**`extract_ground_truth(impressions)`**
- **Pre:** `impressions` is an iterable of `Impression` records with aligned
  `candidate_ids` / `labels`.
- **Post:** returns `[(impr_id, user_id, positions, ids)]`, one tuple per
  impression in input order. `positions` are **1-indexed** locations of clicked
  candidates; `ids` are the matching article ids. Both empty for impressions
  with no clicks.

**`save_ground_truth(results, output_file)`**
- **Pre:** `results` is the output of `extract_ground_truth`; `output_file` is a
  writable path.
- **Post:** writes one `"{impr_id} {user_id} [pos] [ids]"` line per record
  (UTF-8), truncating any existing file.

### `random_rec.py`

**`random_recommend(impressions, seed=42)`**
- **Pre:** `impressions` is an iterable of `Impression` records.
- **Post:** returns `[(impr_id, user_id, scores)]`, one per impression, with
  `scores` a float array of length `len(candidate_ids)` drawn uniformly in
  `[0, 1)`. **Deterministic in `seed`** — the same seed reproduces identical
  scores. No global RNG state is touched (uses a local `default_rng`).

### `popular_rec.py`

**`popular_recommend(impressions)`**
- **Pre:** `impressions` is an iterable of `Impression` records whose
  `timestamp` is sortable.
- **Post:** returns `[(impr_id, user_id, scores)]` in the **original input
  order**. Each candidate's score is its click count accumulated over
  impressions strictly **earlier** in timestamp order — so no future
  information leaks. The chronologically first impression scores all-zero.

### `io.py` — shared writers

**`processed_filename(name)`**
- **Pre:** `name` is a recommender key (e.g. `"random"`, `"nrms"`, or
  `"ground_truth"`).
- **Post:** returns the processed file name — `"processed_ground_truth.txt"`
  for ground truth (it's clicks, not a prediction), else
  `"prediction_processed_{name}.txt"`. Single source of truth shared by the
  pipeline (writer) and dashboard (reader).

**`save_predictions(results, output_file)`**
- **Pre:** `results` is an iterable of `(impr_id, user_id, scores)` with `scores`
  a 1-D array.
- **Post:** writes one `"{impr_id} [ranks]"` line per result; ranks are a dense
  1..N ranking of `scores` (rank 1 = highest score). The `user_id` is not written
  — every prediction file shares this layout with the model recommenders' output,
  and readers take the user from the impressions. Truncates the file.

**`save_predictions_topk(results, impressions, output_file)`**
- **Pre:** `results` are scored results; `impressions` are the matching
  `Impression` records (candidate ids come from them, so no dataset file is
  re-read).
- **Post:** writes `"{impr_id} {user_id} [positions] [ids]"`. `K` per impression
  is **the number of clicks it received** (`sum(labels)`), so each recommender is
  asked for exactly as many items as the user clicked — making outputs directly
  comparable to ground truth.

**`save_user_article_map(topk_file, article_meta, output_file)`**
- **Pre:** `topk_file` is a `"{impr_id} {user_id} [pos] [ids]"` file (top-k or
  ground truth); `article_meta` is `{id: (topic, subtopic)}` from an adapter
  (only the topic is written).
- **Post:** writes the per-user map `"{user_id} [ids] [topics]"`, concatenating
  ids across a user's impressions in order. Unknown ids map to
  `("unknown", "none")`.

**`save_user_article_map_from_ranks(prediction_file, impressions, article_meta, output_file)`**
- **Pre:** `prediction_file` is a full-rank `"impr_id [ranks]"` prediction (only
  the impression id and the **last** bracketed rank token are read). `user_id` is
  always taken from `impressions`, not the file.
- **Post:** writes the per-user map, selecting each impression's top-`k`
  candidates (`k` = number of clicks it received).

---

## `mind_specific/` — NRMS / LSTUR for MIND-format datasets

> Raw-input fetching (the utils bundle, the dev split) lives in
> `dataset_module/mind/prepare.py`, not here.

### `nrms_mind.py` / `lstur_mind.py`

Each exposes a **`run(dataset_dir, train_split, dev_split, prediction_file, *,
epochs=2, ...)`** function so the same model can be trained on any MIND-format
dataset (MIND, `mind_news`, ...). The pipeline hands it the dataset's paths when a
prediction file is missing; running the file directly (`__main__`) trains on MIND.

- **Pre:** `recommenders` + TensorFlow are installed. The dataset's
  `train_split` / `dev_split` folders and its `utils/` bundle (embeddings, dicts,
  `{nrms,lstur}.yaml`) already exist under `dataset_dir` — the pipeline ensures
  the utils via the dataset's `prepare` hook before calling, and the `__main__`
  block downloads MIND's for standalone runs. LSTUR uses `uid2index_small.pkl` (a
  contiguous small-dataset user mapping) because the shipped `uid2index.pkl` is
  the MIND-large dict and overflows LSTUR's per-user embedding on MINDsmall.
- **Post:** trains the model, saves checkpoint weights under
  `dataset_dir/model/`, and writes full-rank predictions to `prediction_file` in
  `"{impr_id} [ranks]"` format. Epochs are currently set to `2` for testing
  (intended to be `5`).

---

## `ebnerd_specific/` — eb-nerd-only

### `nrms_ebnerd.py` (executable training script, WIP)

- **Pre:** the `ebrec` library, `transformers`, `polars`, and TensorFlow are
  installed; the eb-nerd dataset lives under `~/data/ebnerd`.
- **Post:** intended to train NRMS on eb-nerd and dump predictions under
  `ebnerd_predictions/`. **Currently incomplete** — sets up paths/hparams only.