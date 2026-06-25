~~# `dataset_module`

## INPUT
Raw data

## OUTPUT
normalized data for processing

Dataset adapters. Each adapter parses one dataset's raw on-disk format into the
normalized, dataset-agnostic structures defined in `common.py`, so that the
recommenders, output writers, and diversity metrics can be reused unchanged
across datasets. **All dataset-specific format knowledge lives here and nowhere
else.**

```
common.py          Impression namedtuple (the shared in-memory contract)
mind_adapter.py    MIND TSV files  -> normalized structures
ebnerd_adapter.py  eb-nerd Parquet -> normalized structures
```

The two adapters expose the **same three functions** with the same signatures
(`load_impressions`, `load_article_meta`, `load_titles`), so the pipeline can
select an adapter by dataset and call them interchangeably.

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

## `mind_adapter.py`

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

## `ebnerd_adapter.py`

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