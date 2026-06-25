# `recommender_module`

The recommenders and the shared I/O that turns their output into the per-user
files the diversity metrics consume. Recommenders take normalized `Impression`
records (from `dataset_module`) and produce per-candidate scores or rankings;
the writers in `common/io.py` aggregate those into the on-disk formats.

```
common/            dataset-agnostic recommenders + shared writers
  ground_truth.py    the articles users actually clicked (reference baseline)
  random_rec.py      uniform random scores
  popular_rec.py     prior-click-popularity scores (no future leakage)
  io.py              shared prediction / user-article-map writers
  subtopic.py        builds the news-only subset for subtopic diversity
ebnerd_specific/   eb-nerd NRMS training script (uses the ebrec library)
mind_specific/     MIND NRMS / LSTUR training scripts
```

> Raw-input fetching (MIND dev split + utils bundle, eb-nerd presence check)
> lives in `prepare.py` at the project root, not here — see that module.

### On-disk line formats (produced here, read by diversity_module / dashboard)
```
predictions       : {impr_id} {user_id} [rank,rank,...]            (rank 1 = best)
top-k / ground    : {impr_id} {user_id} [pos,pos,...] [id,id,...]
user-article map  : {user_id} [ids] [topics] [subtopics]
```

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
- **Post:** writes one `"{impr_id} {user_id} [ranks]"` line per result; ranks are
  a dense 1..N ranking of `scores` (rank 1 = highest score). Truncates the file.

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
  ground truth); `article_meta` is `{id: (topic, subtopic)}` from an adapter.
- **Post:** writes the per-user map `"{user_id} [ids] [topics] [subtopics]"`,
  concatenating ids across a user's impressions in order. Unknown ids map to
  `("unknown", "none")`.

**`save_user_article_map_from_results(results, impressions, article_meta, output_file)`**
- **Pre:** `results` are scored results (random/popular); `impressions` and
  `article_meta` as above.
- **Post:** same per-user map as above, built directly from scores (skipping the
  intermediate top-k file). Selection is identical to `save_predictions_topk`
  (`k` = number of clicks, highest-scoring candidates).

**`save_user_article_map_from_ranks(prediction_file, impressions, article_meta, output_file, positions_by_impr=None)`**
- **Pre:** `prediction_file` is a full-rank prediction (the rank list is the
  **last** bracketed token, so both `"impr_id [ranks]"` and
  `"impr_id user_id [ranks]"` parse). `user_id` is always taken from
  `impressions`, not the file. `positions_by_impr`, if given, is
  `{impr_id: [orig_index,...]}` (the subtopic subset's kept candidate positions).
- **Post:** writes the per-user map. When `positions_by_impr` is given, each
  impression's ranks are sliced to those positions before selecting top-`k`
  (`k` = clicks kept); otherwise the full candidate list is used.

**`save_user_article_map_from_ground_truth(gt_results, article_meta, output_file)`**
- **Pre:** `gt_results` is `extract_ground_truth` output; `article_meta` as above.
- **Post:** writes the ground-truth per-user map without needing an intermediate
  top-k file.

### `subtopic.py` — news-only subset for subtopic diversity

**`is_in_category(article_meta, article_id, category)`**
- **Pre:** `article_meta` is `{id: (topic, subtopic)}`.
- **Post:** returns `True` iff `article_meta[article_id]`'s topic equals
  `category` (missing id → `False`).

**`build_subtopic_subset(impressions, article_meta, category)`**
- **Pre:** `impressions` are `Impression` records; `article_meta` covers their
  candidate ids; `category` is the parent category to restrict to (e.g. `"news"`
  for MIND).
- **Post:** returns `(sub_impressions, sub_meta, positions_by_impr)`:
  - `sub_impressions`: impressions with candidate lists narrowed to `category`
    articles. Impressions with **no** candidate in the category, or with no
    **clicked** article in it, are dropped (single-click users are left for
    `topic_diversity` to drop downstream).
  - `sub_meta`: `{id: (subtopic, "none")}` for the category's articles — the
    subcategory is promoted into the topic slot so `topic_diversity` measures
    subcategory variety, reusing the writers unchanged.
  - `positions_by_impr`: `{impr_id: [orig_index,...]}` of the kept candidates,
    used to slice a model's per-candidate ranks
    (`save_user_article_map_from_ranks`).

**`subtopic_subset_path(processed_path)`**
- **Pre:** `processed_path` is a recommender's processed per-user file path.
- **Post:** returns the subset counterpart with a `subtopic/` directory inserted
  before the filename (e.g. `.../predictions/prediction_processed_random.txt` →
  `.../predictions/subtopic/prediction_processed_random.txt`). Single source of
  truth for where subset artifacts live.

---

## `mind_specific/` — MIND-only

> Raw-input fetching (the utils bundle, the dev split) lives in the root
> `prepare.py` module, not here.

### `nrms_MIND.py` / `lstur_MIND.py` (executable training scripts)

These are run as scripts, not imported as a library — they have no public
functions, so the contract is **module-level side effects**:

- **Pre:** the MIND `MINDsmall_train` / `MINDsmall_dev` splits exist under
  `data/datasets/mind/`; `recommenders` + TensorFlow are installed. The
  `utils/` bundle is downloaded on demand if the `.yaml` is missing (from the
  Hugging Face base URL — the library's built-in URL is dead). LSTUR uses
  `uid2index_small.pkl` (a contiguous small-dataset user mapping) because the
  shipped `uid2index.pkl` is the MIND-large dict and overflows LSTUR's per-user
  embedding on MINDsmall.
- **Post:** trains the model, saves checkpoint weights under
  `data/datasets/mind/model/`, and writes full-rank predictions to
  `data/data_processed/mind/predictions/prediction_{nrms,lstur}.txt` in
  `"{impr_id} [ranks]"` format. Epochs are currently set to `2` for testing
  (intended to be `5`).

---

## `ebnerd_specific/` — eb-nerd-only

### `nrms_ebnerd.py` (executable training script, WIP)

- **Pre:** the `ebrec` library, `transformers`, `polars`, and TensorFlow are
  installed; the eb-nerd dataset lives under `~/data/ebnerd`.
- **Post:** intended to train NRMS on eb-nerd and dump predictions under
  `ebnerd_predictions/`. **Currently incomplete** — sets up paths/hparams only.