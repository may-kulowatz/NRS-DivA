# Welcome to NRS-DiAna (News Recommender System - Diversity Analyser)
... a benchmarking prototype to compare news recommender systems based on different diversity measures.
Right now, it runs with two datasets [MINDsmall](link) and [EB-NeRDsmall](link).
It also includes a subset of MINDsmall, only containing news of category "news".

## Content

 1. Quick start
 2. Components
 3. Add MIND and EB-NeRD
 4. Run the pipeline
 5. Architecture
 6. Add your own dataset
 7. Add your own recommender
 8. Add your own diversity metric
 9. Licence

**If you want to test your prediction for the MIND challenge**:

## Quick start

First of all, install solara. After that, run the full pipeline with:

```bash
python pipeline.py
```

This generates predictions for each recommender, builds per-user article maps,
and prints the diversity scores.

Since loading and predicting can take a very long time (especially for machine learning based recommender systems),
there are txt files already prepared. The pipeline will skip the prediction and calculation.
## Main Components

### Datasets

- **MIND** (`data/datasets/mind/`) — Microsoft News Dataset (`MINDsmall`), the primary
  dataset. Provides `news.tsv` (article id, category, subcategory, title) and
  `behaviors.tsv` (user impressions and clicks), plus pre-built embeddings and
  dictionaries in `utils/`.
  For further information on this dataset, please check out the [documentation](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news?tabs=azureml-opendatasets) and the [github repo](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md).
- **eb-nerd** (`data/datasets/ebnerd/`) — Ekstra Bladet News Recommendation Dataset
  (`articles.parquet`, train/validation splits) used as a second news source.
  Generated outputs for both datasets live under `data/data_processed/<dataset>/`.
  For further information on this dataset, please check out the [documentation](https://recsys.eb.dk/dataset/).
- **mind_news** (`data/datasets/mind_news/`) — a news-only subset of MIND, built
  from the MIND splits by `dataset_module/mind_news/prepare.py` (`MINDnews_train` /
  `MINDnews_dev`). It keeps only impressions where the user clicked at least one
  article in the **news** category and that showed at least two news candidates,
  stripping every non-news article from the candidates and the user's history.
  Built automatically the first time you run `python pipeline.py mind_news`.

### Recommender Systems

Each recommender produces a ranked top-k list per user impression
(`recommender_module/`):

- **Ground truth** (`ground_truth.py`) — the articles users actually clicked,
  used as the reference baseline.
- **Random** (`random_rec.py`) — randomly ranks the candidate articles.
- **Popular** (`popular_rec.py`) — ranks candidates by overall click popularity.
- **NRMS** (`nrms_MIND.py`) — neural news recommender (Neural News Recommendation
  with Multi-Head Self-Attention) from Microsoft Recommenders.

### Diversity Scores

(`diversity_module/`):

- **Topic diversity** (`topic_diversity.py`) — share of unique topics
  (categories) in a user's list.
- **Content diversity / ILD** (`content_diversity.py`) — intra-list diversity
  based on the mean pairwise cosine *distance* between article title embeddings.

Not all scores are compatible with all recommender systems!

## Software Architecture

### Modules at the root
The orchestration is split into small, single-purpose modules so each concern can
be read and tested on its own:

- `config.py` — the dataset registry (`DATASETS`) and the `input_dir` / `output_dir`
  path helpers. Read-only consumers (e.g. the dashboard) import this without
  pulling in the run machinery.
- `recommender_module/base.py` — the `Recommender` interface (`random`, `popular`,
  `ground_truth`, and the trained `nrms` / `lstur` models) plus the `RunContext`
  they operate on. Each recommender knows how to generate its raw prediction file
  and build its processed per-user file, so the pipeline iterates a list instead
  of special-casing each one. Adding a recommender is a new class here.
- `scores.py` — diversity-score computation, results IO, and file-staleness helpers.
- `pipeline.py` — `run_pipeline`, the orchestration that ties the above together.
- `cli.py` — the interactive front-end. `python pipeline.py [dataset]` delegates
  to it, so the documented entry point is unchanged.

### Pipeline
The pipeline does the following:
- starts the prediction processes for each dataset and recommender system and writes the results to txt files. (This step is skipped if the file alreaady exists.)
- calculates the diversity scores and prints them in the console (TODO: skip this step if already calculated and no changes were made)
- transfers the results to GUI (still to be implemented)

TODO:
- explain own recommenders better
- explain topic diversity better (especially formular)
- add option to exclude users with less than x clicked articles
- explain the ranking (topk from where) use
- delete evaluation and explore-ebnerd eventually
- refactor recommender structure (io.py somewhere else?)
- Add more MIND recommenders
- enrich texts in Solara Dashboard + add links
- add information about code taken from other repositories (ebnerd, MIND, recommenders, content-diversity)
- add license information
- test the tests
- clean up the tests
- add requirements to README (solara for GUI, more stuff for pipeline)