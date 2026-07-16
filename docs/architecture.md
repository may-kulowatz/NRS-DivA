# NRS-DivA architecture

NRS-DivA (News Recommender System – Diversity Analyser) is a three-stage
news-recommender **diversity benchmark**: prepare
datasets → generate recommendations → score diversity, with a dashboard reading
the results.

The three stages are separate CLI entry points that hand off through files on
disk; `config.py` is the shared registry describing every dataset.

```mermaid
flowchart LR
    subgraph stages["Pipeline stages"]
        direction TB
        S1["1 · Prepare<br/>python -m dataset_module"]
        S2["2 · Generate<br/>python -m recommender_module"]
        S3["3 · Score<br/>python -m diversity_module"]
    end

    RAW[("Raw inputs<br/>data/datasets/")]
    PRED[("Predictions<br/>data_processed/.../predictions")]
    MAN[("run_manifest.json<br/>diversity scores")]
    DASH["Dashboard<br/>dashboard.py"]
    CFG["config.py<br/>DATASETS registry"]

    S1 --> RAW
    RAW --> S2
    S2 --> PRED
    PRED --> S3
    S3 --> MAN
    MAN -.reads.-> DASH
    CFG -.configures.-> S1 & S2 & S3 & DASH
```