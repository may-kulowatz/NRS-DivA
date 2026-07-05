# `diversity_module`

Diversity metrics. Each function scores how varied a recommender's per-user
article lists are, reading the **processed user-article files** produced by the
pipeline (via `recommender_module/common/io.py`). All metrics share the same
shape: average a per-user score across users, counting only users with **more
than one click**, and return `0.0` when no user qualifies.

```
topic_diversity.py               topic diversity (category-share metric)
content_diversity.py             content diversity / intra-list diversity (ILD, embedding-based)
content_diversity_normalized.py  normalized (EBNeRD-style) ILD, against the candidate pool
content_diversity_ebnerd.py      EBNeRD IntralistDiversity class (used by the normalized metric)
```

> **Granularity note:** `topic_diversity` and `content_diversity` read the
> per-user files described below. `content_diversity_normalized` is the
> exception — it needs each impression's **candidate pool**, so it works from
> `Impression`s + per-impression recommendations, not the per-user files (see its
> own section).

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

---

## `content_diversity.py`

Intra-list diversity based on the mean pairwise cosine **distance** between
article content embeddings. **Requires `numpy`.** The metric is dataset-agnostic;
only the way the `{article_id: vector}` map is built differs per dataset (two
loaders below):

### Why cosine distance (and not Euclidean or Jaccard)?
- **vs. Euclidean.** The vectors are averaged word / document embeddings, where
  *direction* carries the semantic content and *magnitude* is mostly an artifact
  (title length, token frequency, per-space norm variation). Cosine compares only
  orientation, so two same-topic titles aren't judged "far apart" just because one
  averaged to a longer vector. Cosine similarity is also bounded (≈ `[0, 1]` here),
  which keeps ILD in a fixed, interpretable range — needed for averaging across
  users and for the min/max rescaling in `content_diversity_normalized.py` —
  whereas Euclidean is unbounded and scale-dependent, and tends to *concentrate*
  (all pairs look equidistant) in high dimensions. It also matches the EBNeRD
  leaderboard metric, which defaults to `cosine_distances`; using the same metric
  keeps the two ILD numbers comparable. (If vectors were L2-normalized, cosine and
  squared Euclidean become monotonically equivalent, so the choice would be moot.)
- **vs. Jaccard.** Jaccard is a *set* distance — it needs discrete features
  (bag-of-words, tags, categories), so it can only ask "which items do two articles
  share." These vectors are **dense continuous** embeddings with no meaningful
  set interpretation; Jaccard has nothing to intersect. It would also discard the
  semantic geometry that makes embedding-based ILD worthwhile: two articles using
  different words for the same topic share few tokens (high Jaccard distance = looks
  diverse) yet point the same way in embedding space (low cosine distance = correctly
  judged similar). Set-overlap diversity has its place — that's essentially what the
  **topic** diversity metric captures over discrete categories — but for content
  embeddings cosine is the right tool.

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

---

## `content_diversity_normalized.py`

Normalized intra-list content diversity (the EBNeRD-leaderboard metric). Plain
ILD says how varied a recommended set is; this says how varied it could have been
given the candidate pool, by normalizing **per impression**:

```
normalized = (ILD(recommended) - ILD_min) / (ILD_max - ILD_min)
```

where `ILD_min` / `ILD_max` are the least / most diverse selections of the same
size from the impression's candidates, estimated by
`content_diversity_ebnerd.IntralistDiversity._candidate_diversity`.

### `normalized_content_diversity(impressions, recommended_by_impr, embeddings, *, lookup_key="vector", max_combinations=1000, seed=42)`
- **Pre:** `impressions` are `Impression` records (carry `candidate_ids`);
  `recommended_by_impr` is `{impr_id: [article_id, ...]}` (built by
  `recommender_module/common/io.recommended_per_impression_from_*` / a
  recommender's `recommended_by_impr(ctx)`); `embeddings` is a
  `{article_id: vector}` map from either content loader.
- **Post:** returns a `float` in `[0.0, 1.0]` (clipped — sampled min/max can sit
  just inside the true range). An impression is scored only when its
  recommendation has `≥ 2` embeddable articles **and** its pool has more
  embeddable articles than were recommended; otherwise it is skipped. Returns
  `0.0` when nothing is scorable.

> **Expensive** (per impression it samples up to `max_combinations` candidate
> subsets), so it is computed only when asked:
> `python -m diversity_module <dataset> --normalized`. It is written to
> `run_manifest.json` under each recommender's
> `metrics.content_diversity_normalized`.