# NRS-DivA architecture

NRS-DivA (News Recommender System – Diversity Analyser) is a three-stage
news-recommender **diversity benchmark**: prepare
datasets → generate recommendations → score diversity, with a dashboard reading
the results.

## Overview

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

## Class diagram — recommender hierarchy & core types

```mermaid
classDiagram
    direction TB

    class RunContext {
        <<dataclass>>
        +list impressions
        +dict article_meta
        +str out_dir
        +str raw_dir
        +str processed_dir
        +int seed
        +str in_dir
        +str train_split
        +str dev_split
        +dict model_trainers
    }

    class Recommender {
        <<abstract>>
        +str name
        +bool expensive = False
        +raw_path(ctx) str
        +processed_path(ctx) str
        +generate(ctx)*
        +build_user_map(ctx)*
        +recommended_by_impr(ctx)*
    }

    class _RankRecommender {
        <<abstract>>
        +build_user_map(ctx)
        +recommended_by_impr(ctx)
    }

    class RandomRecommender {
        +name = "random"
        +generate(ctx)
    }

    class PopularRecommender {
        +name = "popular"
        +generate(ctx)
    }

    class ModelRecommender {
        +expensive = True
        +__init__(name)
        +generate(ctx)
    }

    class GroundTruthRecommender {
        +name = "ground_truth"
        +raw_path(ctx) str
        +generate(ctx)
        +build_user_map(ctx)
        +recommended_by_impr(ctx)
    }

    class Impression {
        <<namedtuple>>
        +int impr_id
        +str user_id
        +timestamp
        +list~str~ candidate_ids
        +list~int~ labels
    }

    Recommender <|-- _RankRecommender
    Recommender <|-- GroundTruthRecommender
    _RankRecommender <|-- RandomRecommender
    _RankRecommender <|-- PopularRecommender
    _RankRecommender <|-- ModelRecommender

    Recommender ..> RunContext : operates on
    RunContext o-- "many" Impression : impressions
```

## Module — `dataset_module` (prepare & parse)

Each dataset is its own package with a matching pair of interfaces: an `adapter`
(parses the raw format into `Impression` records) and a `prepare` module
(downloads / builds the raw inputs). `__main__` prepares one or all of them.

```mermaid
flowchart TB
    MAIN["__main__<br/>_prepare() · main()"]

    subgraph mind["mind/"]
        MA["adapter<br/>load_impressions()<br/>load_article_meta()<br/>load_titles()"]
        MP["prepare<br/>DIR = 'mind'<br/>ensure_raw_data()<br/>ensure_utils()"]
    end

    subgraph ebnerd["ebnerd/"]
        EA["adapter<br/>load_impressions()<br/>load_article_meta()<br/>load_titles()"]
        EP["prepare<br/>DIR = 'ebnerd'<br/>ensure_raw_data()<br/>ensure_utils()"]
    end

    subgraph mnews["mind_news/"]
        NA["adapter<br/>load_impressions()<br/>load_article_meta()<br/>load_titles()"]
        NP["prepare<br/>DIR = 'mind_news'<br/>build_mind_news()<br/>ensure_raw_data()<br/>ensure_utils()"]
    end

    COMMON["common<br/>Impression (namedtuple)<br/>default_input_dir()"]

    MAIN --> MP & EP & NP
    MA & EA & NA --> COMMON
    NP -. derives from .-> MP
```

## Module — `recommender_module` (generate predictions)

`base.py` is the hub: `build_context()` loads a dataset once and
`build_recommenders()` assembles the `Recommender` hierarchy. The cheap
recommenders live in `common/`; the neural models are per-dataset training
scripts imported lazily (they pull in TensorFlow) via `config.model_trainers`.

```mermaid
flowchart TB
    MAIN["__main__<br/>generate(dataset, only)"]
    BASE["base.py<br/>build_context()<br/>build_recommenders()<br/>Recommender hierarchy · RunContext"]

    subgraph common["common/"]
        IO["io.py<br/>save_predictions()<br/>save_user_article_map*()<br/>recommended_per_impression_*()"]
        RND["random_rec<br/>random_recommend()"]
        POP["popular_rec<br/>popular_recommend()"]
        GT["ground_truth<br/>extract_ground_truth()<br/>save_ground_truth()"]
    end

    subgraph models["model trainers (lazy · TensorFlow)"]
        MS["mind_specific/<br/>nrms_mind.run()<br/>lstur_mind.run()<br/>naml_mind.run()"]
        ES["ebnerd_specific/<br/>nrms_ebnerd.run()<br/>lstur_ebnerd.run()<br/>naml_ebnerd.run()"]
    end

    MAIN --> BASE
    BASE --> IO
    BASE --> RND & POP & GT
    BASE -. imports via config.model_trainers .-> MS & ES
```

## Module — `diversity_module` (score diversity)

`__main__` computes each measure across every recommender that has a prediction.
The normalized metric delegates to the `IntralistDiversity` class; the content
metric reuses `topic_diversity`'s per-user file parser.

```mermaid
flowchart TB
    MAIN["__main__<br/>score(dataset, only, recommender)"]

    TD["topic_diversity.py<br/>topic_diversity()<br/>_parse_user_articles()"]
    CD["content_diversity.py<br/>content_diversity()<br/>load_news_embeddings()<br/>load_precomputed_embeddings()"]
    NCD["content_diversity_normalized.py<br/>normalized_content_diversity()"]
    ILD["content_diversity_ebnerd.py<br/>IntralistDiversity"]

    MAIN --> TD & CD & NCD
    NCD --> ILD
    CD -. reuses parser .-> TD
```

## How it fits together

- **`RunContext`** is the single handle threaded through every recommender —
  `build_context()` in `recommender_module/base.py` loads a dataset once (via its
  configured adapter/prepare hook) and hands back `(cfg, ctx, recommenders)`.
- The **`Recommender` hierarchy** is the main polymorphism point:
  `_RankRecommender` subclasses share the "build the per-user map from a full-rank
  file" logic (random/popular/model), while `GroundTruthRecommender` overrides the
  file layout. `ModelRecommender` is the only `expensive` one — it lazily imports a
  per-dataset TensorFlow training script from `config.model_trainers`.
- **`scores.py`** owns the `run_manifest.json` — the one shared data contract
  between the generate stage, the score stage, and the read-only **dashboard**.
