# `diversity_module`

Diversity metrics. Each function scores how varied a recommender's per-user
article lists are, reading the **processed user-article files** produced by the
pipeline (via `recommender_module/common/io.py`). All metrics share the same
shape: average a per-user score across users, counting only users with **more
than one click**, and return `0.0` when no user qualifies.

```
topic_diversity.py     topic diversity (category-share metric)
content_diversity.py   content diversity / intra-list diversity (ILD, embedding-based)
```

### The user-article file format (shared input)
Every metric consumes a whitespace-delimited file with one line per user:

```
{user_id} [id,id,...] [topic,topic,...]
```

The two bracketed lists are positionally aligned per article. A topic field
may itself be a `"|"`-separated group of topics (eb-nerd multi-topic articles);
the sentinel `"none"` marks "no topic". Parsing is done by the private
`_parse_user_articles`, which `content_diversity.py` reuses.

---

## `topic_diversity.py`

### `topic_diversity(user_articles_file)`
Average of `unique topics / total topic assignments` across qualifying users.
- **Pre:** `user_articles_file` exists and follows the format above. Topics may
  be `"|"`-grouped; `"none"` marks an untopiced article.
- **Post:** returns a `float` in `[0.0, 1.0]`.
  - Users with `≤ 1` clicked article are skipped.
  - Each article's topics are flattened; `"none"` entries are dropped from both
    the unique count and the total.
  - A user left with no topics after filtering is skipped.
  - Returns `0.0` when no user qualifies. For single-topic datasets (MIND) this
    reduces to `unique topics / number of articles`.

> Running this file directly (`__main__`) prints topic diversity for the MIND
> random / popular / ground-truth processed files, if present.

---

## `content_diversity.py`

Intra-list diversity based on the mean pairwise cosine **distance** between
article content embeddings. **Requires `numpy`.** The metric is dataset-agnostic;
only the way the `{article_id: vector}` map is built differs per dataset (two
loaders below):

- **MIND** — `load_news_embeddings`, averaging MIND word embeddings
  (`embedding.npy` + `word_dict.pkl`, gitignored; fetched on demand by
  `dataset_module/mind/prepare.py`). **Requires** those utils.
- **eb-nerd** — `load_precomputed_embeddings`, reading ready-made document
  vectors from `contrastive_vector.parquet`. **Requires `pyarrow`.**

### `load_news_embeddings(news_file, embedding_file, word_dict_file)`
Builds the `{news_id: vector}` map by averaging a title's word embeddings.
- **Pre:**
  - `news_file` is a MIND `news.tsv` (id in col 1, title in col 4).
  - `embedding_file` is the `.npy` word-embedding matrix.
  - `word_dict_file` is the pickled `{word: row_index}` dict built with the
    **same** tokenizer used here (Microsoft Recommenders `word_tokenize`), so
    token→embedding lookups line up.
- **Post:** returns `{news_id: np.ndarray}` where each vector is the **mean** of
  its title words' embeddings. Rows with `< 4` columns and articles whose title
  has **no known word** are omitted (they have no content vector).

### `load_precomputed_embeddings(vector_file, id_column="article_id", vector_column="contrastive_vector")`
Reads one ready-made document embedding per article from a Parquet file (no
tokenizer/word dict — sidesteps the language/vocab mismatch that makes MIND's
English word embeddings unusable for eb-nerd's Danish titles).
- **Pre:** `vector_file` is a Parquet file with an id column and a list-of-float
  vector column (e.g. eb-nerd's `contrastive_vector.parquet`: `article_id` +
  768-dim `contrastive_vector`). All vectors share one dimension.
- **Post:** returns `{str(article_id): np.ndarray(float32)}`. Ids are stringified
  to match the ids used throughout the pipeline.

### `content_diversity(user_articles_file, news_embeddings)`
- **Pre:** `user_articles_file` follows the shared format; `news_embeddings` is
  a `{article_id: vector}` map from **either** loader. (Pass the same object
  across calls — building it is expensive; the pipeline caches it per run.)
- **Post:** returns a `float`. For each user: ILD `= 1 - mean pairwise cosine
  similarity` over that user's embeddable articles.
  - Users with `≤ 1` clicked article are skipped.
  - Only articles present in `news_embeddings` contribute; users left with
    `< 2` embeddable articles are skipped.
  - Returns `0.0` when no user qualifies. Zero-norm vectors are guarded against
    division-by-zero in the cosine computation.

> Running this file directly (`__main__`) loads the MIND dev embeddings and
> prints content diversity for the random / popular / nrms / lstur / ground-truth
> processed files.