# Welcome to NRS-DivA (News Recommender System - Diversity Analyser)
... a benchmarking prototype to compare news recommender systems based on different diversity measures.
Right now, it runs with two datasets [MINDsmall](link) and [EB-NeRDsmall](link).
It also includes a subset of MINDsmall, only containing news of category "news".

IMPORTANT: Keep in MIND (haha): https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf

## Content

 1. Quick start
 2. Command reference (the stages, in order)
 3. Components
 4. Architecture
 5. Add your own dataset
 6. Add your own recommender
 7. Add your own diversity metric
 8. Licence

**If you want to test your prediction for the MIND challenge**:

## Quick start

### Requirements

The project targets **Python 3.11**. After cloning, install the third-party
packages below (a virtual environment is recommended):

```bash
pip install \
    numpy scipy scikit-learn \
    polars pyarrow \
    tensorflow transformers recommenders \
    matplotlib tqdm solara
```

What each is for:

| Package | Used for |
| --- | --- |
| `numpy`, `scipy`, `scikit-learn` | numeric arrays, cosine distances, and the diversity-score math |
| `polars`, `pyarrow` | reading the EB-NeRD Parquet files (`articles.parquet`, `behaviors.parquet`) |
| `tensorflow`, `transformers`, `recommenders` | the neural recommenders (NRMS / LSTUR) built on [Microsoft Recommenders](https://github.com/recommenders-team/recommenders); `recommenders` is also used to download the MIND resources in `dataset_module/mind/prepare.py` |
| `matplotlib` | plotting the diversity results |
| `solara` | the interactive dashboard (`dashboard.py`) |
| `tqdm` | progress bars during prediction |

Everything else the code imports (`os`, `json`, `pathlib`, `pickle`,
`itertools`, …) ships with the Python standard library. To run the test suite
you also need `pytest` (`pip install pytest`).

Tested versions: numpy 1.26.4, scipy 1.17.1, scikit-learn 1.8.0, polars 1.41.2,
pyarrow 24.0.0, tensorflow 2.15.1, transformers 4.57.6, recommenders 1.2.1,
matplotlib 3.10.9, tqdm 4.67.3, solara 1.57.4.

### Run

**There is no single orchestrator script.** Each stage of the workflow is its own
command, and you run the stages in order. Every command is **non-interactive**
(no prompts). Stage 1 runs one recommender at a time (or `--all`); stage 2 scores
them.

The whole workflow for the MIND dataset is just:

```bash
python -m dataset_module                 # stage 0 — get the raw data (once)
python -m recommender_module MIND --all  # stage 1 — generate every prediction (slow: trains models)
python -m diversity_module MIND          # stage 2 — score them
solara run dashboard.py                  # view the results (optional)
```

Prepared prediction files are shipped with the repo, so for MIND you can usually
skip straight to stage 2, or generate a single recommender with
`python -m recommender_module MIND random`. See the
[Command reference](#command-reference) for each stage's pre/post-conditions.

## Command reference

In every command `<dataset>` is one of `MIND`, `ebnerd`, or `mind_news`. Run the
stages in the order below; each stage's **PRE** is produced by the stage before
it. (Note the dataset name is spelled `mind` — the folder name — for the prepare
stage, but `MIND` for the recommender and diversity stages.)

### Example workflows

Each line is one command; run them top to bottom.

**A. One recommender, one measure on MIND** (prepare → random → content diversity):

```bash
python -m dataset_module mind                       # 1. prepare MIND's raw data
python -m recommender_module MIND random            # 2. generate the random recommender
python -m diversity_module MIND content_diversity random   # 3. score content diversity, random only
```

**B. The full run on MIND** (every recommender, every measure, then browse):

```bash
python -m dataset_module mind                       # 1. prepare the data
python -m recommender_module MIND --all             # 2. all recommenders (slow: trains the models)
python -m diversity_module MIND --all               # 3. all measures (slow: normalized metric)
solara run dashboard.py                             # 4. view the results
```

**C. Compare recommenders on a single measure:**

```bash
python -m recommender_module MIND random            # 1. generate two recommenders
python -m recommender_module MIND popular           #    (data already shipped for MIND)
python -m diversity_module MIND topic_diversity     # 2. scores every recommender with a prediction
```

**D. Just one measure on one recommender** (MIND ships predictions, so no prepare needed):

```bash
python -m diversity_module MIND content_diversity random   # content diversity, random only
```

**E. eb-nerd** (its files must already be on disk — no public download):

```bash
python -m dataset_module ebnerd                            # 1. verify the raw inputs
python -m recommender_module ebnerd nrms                   # 2. train + predict NRMS
python -m diversity_module ebnerd content_diversity_xlmr nrms   # 3. score one space for NRMS
```

### Stage 0 — Prepare data · `python -m dataset_module`

```bash
python -m dataset_module                        # all datasets
python -m dataset_module mind                   # one dataset (folder name: mind / ebnerd / mind_news)
python -m dataset_module.mind.prepare           # equivalent single-dataset form
```

- **PRE:** network access for datasets with a known download source (MIND, and
  mind_news which is derived from it). eb-nerd has no public URL — its Parquet
  files must already be on disk.
- **POST:** each dataset's raw inputs live under `data/datasets/<dataset>/`. A
  dataset that cannot be prepared is reported and skipped, not fatal. Idempotent:
  anything already present is left untouched.

### Stage 1 — Generate predictions · `python -m recommender_module`

Run one recommender, or every recommender on the dataset:

```bash
python -m recommender_module <dataset> <recommender>   # just that one
python -m recommender_module <dataset> --all           # all of them
```

- **PRE:** the dataset's raw inputs exist (stage 0).
- **POST:** each recommender that ran has written its raw prediction file
  (`data_processed/<dataset>/predictions/prediction_<name>.txt`, or
  `ground_truth.txt`) and its per-user processed file
  (`predictions_processed/…`), and `run_manifest.json`'s stage timestamps are
  refreshed. **No diversity scores are written.**
- `<recommender>` is one of `random`, `popular`, `ground_truth`, or a model
  (`nrms` / `lstur` / `naml`) — which models exist depends on the dataset.
- `--all` runs them all. ⚠️ It **trains every model** (needs TensorFlow), so it
  can take a long time; running a single cheap recommender is instant.

```bash
python -m recommender_module MIND random      # just the random recommender
python -m recommender_module ebnerd nrms      # train + predict NRMS on ebnerd
python -m recommender_module MIND --all       # every recommender (slow: trains models)
```

### Stage 2 — Score diversity · `python -m diversity_module`

Compute one measure, or every measure on the dataset:

```bash
python -m diversity_module <dataset> <measure> [<recommender>]   # just that one
python -m diversity_module <dataset> --all                        # all of them
```

- **PRE:** the recommenders have been generated (stage 1). Recommenders without a
  prediction file are skipped; if none exist there is nothing to score.
- **POST:** `data_processed/<dataset>/run_manifest.json` is updated with the
  computed measure(s) (measures not computed this run keep their stored value) and
  the scores are printed. Any stale processed per-user file is rebuilt from its
  prediction first. **No predictions are generated and no model is trained.**
- `<measure>` is one of `topic_diversity`, `content_diversity`, or
  `content_diversity_normalized` (plus per-embedding-space variants such as
  `content_diversity_xlmr` on ebnerd) — which measures apply depends on the dataset.
- `<recommender>` is optional: a measure is scored across every recommender with a
  prediction by default, or pass one (e.g. `random`) to score only that one.
- `--all` computes them all. ⚠️ It includes the **per-impression normalized**
  content diversity, which is slow; a single cheap measure is fast.

```bash
python -m diversity_module MIND topic_diversity              # one measure, all recommenders
python -m diversity_module MIND content_diversity random     # one measure, one recommender
python -m diversity_module ebnerd content_diversity_xlmr     # one embedding space
python -m diversity_module MIND --all                        # every measure (slow: normalized)
```

### View results · `solara run dashboard.py`

```bash
solara run dashboard.py
```

- **PRE:** stages 1 and 2 have been run for the datasets you want to browse.
- **POST:** nothing — the dashboard is strictly a viewer; it only reads
  `run_manifest.json` and the processed files.

### Extras

Train a single model recommender directly (bypasses the stage-1 CLI; writes the
same prediction file):

```bash
python -m recommender_module.mind_specific.nrms_mind      # NRMS on MIND
python -m recommender_module.mind_specific.lstur_mind     # LSTUR on MIND
python -m recommender_module.mind_specific.naml_mind      # NAML on MIND
python -m recommender_module.ebnerd_specific.nrms_ebnerd  # NRMS on eb-nerd
python -m recommender_module.ebnerd_specific.lstur_ebnerd # LSTUR on eb-nerd
python -m recommender_module.ebnerd_specific.naml_ebnerd  # NAML on eb-nerd
```

Dataset statistics table (CSV + JSON):

```bash
python statistic.py [<dataset>]
```

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
  Built by `python -m dataset_module mind_news` (or on first use of the
  `mind_news` recommender stage).

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

### Modules
There is no orchestrator module. The work is split into small, single-purpose
modules that each own one command-line stage; the "pipeline" is just the order you
run them in (see the [Command reference](#command-reference)).

- `config.py` — the dataset registry (`DATASETS`) and the `input_dir` / `output_dir`
  path helpers. Read-only consumers (e.g. the dashboard) import this without
  pulling in the run machinery.
- `dataset_module/` — stage 0. `python -m dataset_module` prepares the raw inputs;
  each dataset has its own `adapter` (parses the format) and `prepare` module
  (acquires the files).
- `recommender_module/` — stage 1. `python -m recommender_module` generates the
  predictions. `base.py` holds the `Recommender` interface (`random`, `popular`,
  `ground_truth`, and the trained `nrms` / `lstur` / `naml` models), the
  `RunContext` they operate on, and `build_context`, the shared "load a dataset"
  helper both stages call. Each recommender knows how to generate its raw
  prediction file and build its processed per-user file, so the stage iterates a
  list instead of special-casing each one. Adding a recommender is a new class here.
- `diversity_module/` — stage 2. `python -m diversity_module` computes the
  diversity measures from the processed files and writes them to the run manifest.
- `scores.py` — run-manifest I/O, score-computation primitives, and file-staleness
  helpers shared by the scoring stage.
- `dashboard.py` — the Solara viewer (read-only).

### Data flow
Each stage consumes the previous stage's output:
- **stage 0** writes raw inputs to `data/datasets/<dataset>/`;
- **stage 1** reads those and writes prediction + per-user files to
  `data/data_processed/<dataset>/predictions{,_processed}/` (skipping any file
  that already exists unless forced);
- **stage 2** reads the per-user files and writes/updates the diversity scores in
  `data/data_processed/<dataset>/run_manifest.json`, which the dashboard reads.

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