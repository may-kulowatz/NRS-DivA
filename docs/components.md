# Components

The datasets, recommenders, and diversity scores NRS-DivA ships with. For how the
modules fit together see [architecture](architecture.md); for how to run them see
[usage](usage.md).

## Datasets

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
  Built by `python -m dataset_module mind_news` (or on first use of the
  `mind_news` recommender stage).

## Recommender systems

Each recommender produces a ranked top-k list per user impression
(`recommender_module/`):

- **Ground truth** (`ground_truth.py`) — the articles users actually clicked,
  used as the reference baseline.
- **Random** (`random_rec.py`) — randomly ranks the candidate articles.
- **Popular** (`popular_rec.py`) — ranks candidates by overall click popularity.
- **NRMS** (`nrms_MIND.py`) — neural news recommender (Neural News Recommendation
  with Multi-Head Self-Attention) from Microsoft Recommenders.

## Diversity scores

(`diversity_module/`):

- **Topic diversity** (`topic_diversity.py`) — share of unique topics
  (categories) in a user's list.
- **Content diversity / ILD** (`content_diversity.py`) — intra-list diversity
  based on the mean pairwise cosine distance between article title embeddings.

Not all scores are compatible with all recommender systems!
