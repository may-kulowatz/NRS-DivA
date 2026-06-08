# EchoBench

A benchmark for measuring the **diversity** of news recommender systems. EchoBench
runs several recommenders over a news dataset, then scores how diverse each
recommender's lists are compared to one another and to the ground truth.

## Quick start

Run the full pipeline with:

```bash
python pipeline.py
```

This generates predictions for each recommender, builds per-user article maps,
and prints the diversity scores.

Since loading and predicting can take a very long time (especially for machine learning based recommender systems),
there are txt files already prepared. The pipeline will skip the prediction and calculation.
## Main Components

### Datasets

- **MIND** (`data/MIND/`) — Microsoft News Dataset (`MINDsmall`), the primary
  dataset. Provides `news.tsv` (article id, category, subcategory, title) and
  `behaviors.tsv` (user impressions and clicks), plus pre-built embeddings and
  dictionaries in `utils/`.
  For further information on this dataset, please check out the [documentation](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news?tabs=azureml-opendatasets) and the [github repo](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md).
- **eb-nerd** (`data/ebnerd/`) — Ekstra Bladet News Recommendation Dataset
  (`articles.parquet`, train/validation splits) used as a second news source.
  For further information on this dataset, please check out the [documentation](https://recsys.eb.dk/dataset/).

### Recommender Systems

Each recommender produces a ranked top-k list per user impression
(`recommenders/`):

- **Ground truth** (`ground_truth.py`) — the articles users actually clicked,
  used as the reference baseline.
- **Random** (`random_rec.py`) — randomly ranks the candidate articles.
- **Popular** (`popular_rec.py`) — ranks candidates by overall click popularity.
- **NRMS** (`nrms_MIND.py`) — neural news recommender (Neural News Recommendation
  with Multi-Head Self-Attention) from Microsoft Recommenders.

### Diversity Scores

Each recommender's lists are scored on three diversity metrics
(`diversityScores/`):

- **Topic diversity** (`topic_diversity.py`) — share of unique topics
  (categories) in a user's list.
- **Subtopic diversity** (`topic_diversity.py`) — share of unique subtopics
  within a category.
- **Content diversity / ILD** (`content_diversity.py`) — intra-list diversity
  based on the mean pairwise cosine *distance* between article title embeddings.

## Software Architecture

### Pipeline
The pipeline does the following:
- starts the prediction processes for each dataset and recommender system and writes the results to txt files. (This step is skipped if the file alreaady exists.)
- calculates the diversity scores and prints them in the console (TODO: skip this step if already calculated and no changes were made)
- transfers the results to GUI (still to be implemented)

## To-Dos
- fix subtopics implementation
- explain own recommenders better
- explain topic diversity better (especially formular)
- add option to exclude users with less than x clicked articles
- explain the ranking (topk from where) use
- delete evaluation and explore-ebnerd eventually
- refactor recommender structure (io.py somewhere else?)
- Add more MIND recommenders
- Add content-diversity to ebnerd
- enrich texts in Solara Dashboard + add links
- add information about code taken from other repositories (ebnerd, MIND, recommenders, content-diversity)
- add license information
- test the tests
- clean up the tests
- add requirements to README