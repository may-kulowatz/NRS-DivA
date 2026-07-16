# NRS-DivA (News Recommender System – Diversity Analyser)

A benchmarking prototype to compare news recommender systems based on different
diversity measures. It runs a three-stage pipeline — **prepare** datasets →
**generate** recommendations → **score** diversity — with a dashboard for
browsing the results.

Right now it ships with two datasets, [MINDsmall](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news)
and [EB-NeRDsmall](https://recsys.eb.dk/dataset/), plus `mind_news`, a news-only
subset of MIND.

> **License note:** the dataset data is covered by its own licenses:
>
> - MIND — [MSR Data License](https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf)
> - EB-NeRD — [General License Terms](https://recsys.eb.dk/assets/pdf/general_license_terms.pdf)

## Quick start

Scoring diversity is slow, but the repo already ships computed results
(`run_manifest.json` per dataset). To just explore those, launch the dashboard on
its own — it only reads the manifests, it never runs the pipeline:

```bash
solara run dashboard.py   # browse the already-generated results, no scoring
```

To (re)run the pipeline yourself, use the one-command wrapper:

```bash
python -m nrsdiva MIND               # prepare -> generate -> score
python -m nrsdiva MIND --dashboard   # ...then open the dashboard
```

By default this generates the cheap recommenders and scores any model predictions
already in the repo, so it finishes fast. Each stage is also its own command for
finer control. See [docs/usage.md](docs/usage.md) for requirements, the full
command reference, and per-stage details.

**Already have a `prediction.txt` in MIND-challenge format?** You can score it
without writing any recommender logic — see
[docs/extending.md](docs/extending.md), option B.

## Documentation

- [Usage](docs/usage.md) — requirements, running the pipeline, full command reference
- [Components](docs/components.md) — the datasets, recommenders, and diversity scores
- [Architecture](docs/architecture.md) — modules, data flow, and core types
- [Add your own recommender](docs/extending.md) — plug in scoring logic or a prediction file

### Module references

Per-module docs, each covering the stage it owns:

- [`dataset_module`](dataset_module/README.md) — stage 0: dataset adapters and preparers
- [`recommender_module`](recommender_module/README.md) — stage 1: recommenders and shared prediction I/O
- [`diversity_module`](diversity_module/README.md) — stage 2: the diversity metrics