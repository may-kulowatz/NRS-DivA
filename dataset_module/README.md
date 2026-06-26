~~# `dataset_module`

## INPUT
Raw data

## OUTPUT
normalized data for processing

Dataset adapters **and preparers**. For each dataset there is an *adapter* that
parses its raw on-disk format into the normalized, dataset-agnostic structures
defined in `common.py`, and a *preparer* that makes sure that raw format is
present on disk in the first place (downloading or building it as needed). So the
recommenders, output writers, and diversity metrics can be reused unchanged
across datasets. **All dataset-specific knowledge lives here and nowhere else.**

Each dataset is its own package holding an `adapter` and a `prepare` module;
`common.py`, `__main__.py`, and this README sit at the top level:

```
common.py            Impression namedtuple + default_input_dir helper
__main__.py          `python -m dataset_module` — prepare every dataset
mind/
  adapter.py         MIND TSV files  -> normalized structures
  prepare.py         fetch MIND's dev split + utils bundle
ebnerd/
  adapter.py         eb-nerd Parquet -> normalized structures
  prepare.py         verify eb-nerd's inputs are present
mind_news/
  adapter.py         mind_news TSV   -> normalized structures (subcategory as topic)
  prepare.py         build mind_news from the sibling MIND data
```

Every `adapter` exposes the **same three functions** with the same signatures
(`load_impressions`, `load_article_meta`, `load_titles`); every `prepare` module
likewise exposes the **same two functions** (`ensure_raw_data`, `ensure_utils`).
So the pipeline can select a dataset's adapter and preparer and call them
interchangeably (wired per-dataset in `config.DATASETS`).

### Running independently

`dataset_module` can fetch/build all of its data without running the pipeline:

```
python -m dataset_module                     # prepare every dataset
python -m dataset_module.mind.prepare        # just MIND
python -m dataset_module.mind_news.prepare   # build mind_news from local MIND data
```

---

## Preparers (`<dataset>/prepare.py`)

Each preparer is the acquisition analog of the matching adapter and exposes:

### `ensure_raw_data(in_dir)`
- **Post:** the *essential* inputs the adapter reads exist under `in_dir`,
  fetching or building whatever is missing; raises if an input is missing and
  cannot be obtained. Returns `True` if work happened, `False` if already present.
  - `mind/prepare` downloads the MIND 'small' dev split.
  - `ebnerd/prepare` only verifies the inputs are present (eb-nerd has no public
    download URL) and raises with instructions if not.
  - `mind_news/prepare` builds `MINDnews_{train,dev}` from the sibling `mind`
    dataset (fetching MIND's dev split first if missing).

### `ensure_utils(in_dir)`
- **Post:** the *optional* content-diversity bundle exists under `in_dir/utils`,
  fetched/built on demand. Returns `True` if work happened, `False` otherwise.
  - `mind/prepare` downloads the embeddings/word-dict bundle.
  - `mind_news/prepare` copies MIND's bundle into mind_news's own `utils/`.
  - `ebnerd/prepare` is a no-op (content diversity is precomputed).

Each module also defines `DIR` (its folder under `data/datasets/`) and a
`__main__` block so it can be run standalone.

---

## `common.py`

### `Impression` (namedtuple)
The normalized view of a single impression. Every adapter must emit records that
satisfy this contract:

| field           | type    | guarantee |
| --------------- | ------- | --------- |
| `impr_id`       | `int`   | unique per impression |
| `user_id`       | `str`   | the user shown the impression |
| `timestamp`     | sortable| orderable impression time (popularity ordering relies on it) |
| `candidate_ids` | `[str]` | article ids in display order |
| `labels`        | `[int]` | `1` if the candidate at the same index was clicked, else `0` |

**Postcondition (invariant for all producers):** `len(candidate_ids) == len(labels)`
and `labels[i]` corresponds to `candidate_ids[i]`. Article ids are strings in
every dataset (eb-nerd's integer ids are stringified) so downstream code treats
ids uniformly.

---

## `mind/adapter.py`

Parses MIND's tab-separated `behaviors.tsv` / `news.tsv`. Inline `Nxxxx-1`
(clicked) / `Nxxxx-0` (not clicked) labels are decoded here.

### `load_impressions(behaviors_file)`
- **Pre:** `behaviors_file` exists and is UTF-8 MIND `behaviors.tsv`; each line
  has at least 5 tab-separated columns and column 5 (`impressions`) is
  space-separated `Nxxxx-<0|1>` tokens.
- **Post:** returns `list[Impression]`, one per line, in file order. `impr_id`
  is the parsed int from column 1; `candidate_ids`/`labels` are aligned
  (`labels[i] == 1` iff that candidate was clicked).

### `load_article_meta(news_file)`
- **Pre:** `news_file` is UTF-8 MIND `news.tsv`; columns are
  `news_id, category, subcategory, title, ...` (≥3 columns used).
- **Post:** returns `{news_id: (topic, subtopic)}`. `topic` is the category,
  `subtopic` is the subcategory or the sentinel `"none"` when absent. Only
  `topic` is consumed downstream; `subtopic` is retained in the tuple but unused.
  Last line wins on duplicate ids.

### `load_titles(news_file)`
- **Pre:** `news_file` is UTF-8 MIND `news.tsv`; title is column 4 (index 3).
- **Post:** returns `{news_id: title}` containing only rows with ≥4 columns;
  rows without a title column are silently omitted.

---

## `ebnerd/adapter.py`

Parses eb-nerd's Parquet files into the **same** structures the MIND adapter
produces. **Requires `pyarrow`** (read directly via `pyarrow.parquet`, not
`pandas.read_parquet`, to avoid pandas' pyarrow extension-type re-registration
error on module re-import).

### `load_impressions(behaviors_file)`
- **Pre:** `behaviors_file` is an eb-nerd `behaviors.parquet` with columns
  `impression_id, user_id, impression_time, article_ids_inview,
  article_ids_clicked`.
- **Post:** returns `list[Impression]` in row order. Candidates come from
  `article_ids_inview` (stringified); `labels[i] == 1` iff that id appears in
  `article_ids_clicked`. (Clicks are listed explicitly rather than via inline
  labels as in MIND.)

### `load_article_meta(articles_file)`
- **Pre:** `articles_file` is an eb-nerd `articles.parquet` with columns
  `article_id, topics` (`topics` is a list of human-readable labels per article).
- **Post:** returns `{article_id: (topics_str, "none")}`. `topics_str` joins all
  of an article's topics with `"|"` and replaces intra-label whitespace with
  `"_"` (e.g. `"Crime|Violent_crime"`) so the whitespace-delimited user-article
  file stays parseable; `"none"` when the article has no topics. The `subtopic`
  slot is always `"none"` (eb-nerd has no usable subcategory) and is unused
  downstream.

### `load_titles(articles_file)`
- **Pre:** `articles_file` is an eb-nerd `articles.parquet` with columns
  `article_id, title`.
- **Post:** returns `{article_id: title}` with ids stringified.