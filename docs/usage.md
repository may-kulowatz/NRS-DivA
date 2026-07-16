# Usage

How to install NRS-DivA and run the pipeline. For the overview see the
[README](../README.md); for the internals see [architecture](architecture.md).

## Requirements

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

## Run

The quickest way is the one-command wrapper, which runs all three stages in order:

```bash
python -m nrsdiva MIND               # prepare -> generate -> score
python -m nrsdiva MIND --dashboard   # ...then open the dashboard
python -m nrsdiva MIND --train       # ...also (re)train the neural models (slow)
```

By default this generates only the cheap recommenders and scores any model
predictions already shipped with the repo — so it finishes fast. Add `--train`
to (re)train the neural models (needs TensorFlow).

The wrapper is just a front door: each stage is still its own command, which you
run directly for finer control (a single recommender, a single measure). Every
command is non-interactive. See the [Command reference](#command-reference).

```bash
python -m dataset_module MIND            # stage 0 — get the raw data (once)
python -m recommender_module MIND --all  # stage 1 — every prediction (slow: trains models)
python -m diversity_module MIND --all    # stage 2 — every measure
solara run dashboard.py                  # view the results (optional)
```

## Command reference

`<dataset>` is `MIND`, `ebnerd`, or `mind_news` — spelled in **any case** (the
folder name works too). Run the stages in order; each stage's **PRE** is produced
by the stage before it. Every command is non-interactive.

### Example workflows

```bash
# One recommender, one measure on MIND:
python -m recommender_module MIND random                   # generate random
python -m diversity_module MIND content_diversity random   # score content diversity for it

# eb-nerd, one model (its files must already be on disk — no public download):
python -m recommender_module ebnerd nrms                   # train + predict NRMS
python -m diversity_module ebnerd content_diversity_xlmr nrms   # score one embedding space
```

For the full run, prefer `python -m nrsdiva <dataset>` (see [Run](#run)).

### Stage 0 — Prepare data · `python -m dataset_module`

```bash
python -m dataset_module                        # all datasets
python -m dataset_module MIND                   # one dataset (MIND / ebnerd / mind_news)
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

Significance of each recommender's diversity vs. the ground truth — a paired
t-test per metric, with one figure (mean difference ± 95% CI) written to
`data_processed/<dataset>/statistics/`:

```bash
python statistic.py [<dataset>]
```