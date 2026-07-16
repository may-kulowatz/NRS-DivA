# EB-NeRD dataset files

The pipeline reads its raw **EB-NeRD** inputs from this folder. EB-NeRD has **no
public direct-download URL**, so nothing here is fetched automatically — the
prepare step only *verifies* the files are present. Everything below is
git-ignored and must be **added manually**:

```
data/datasets/ebnerd/
├── articles.parquet            # article metadata (title, subtitle, topics, …)
├── contrastive_vector.parquet  # 768-dim document embeddings (content diversity)
├── train/                      # training split: behaviors.parquet + history.parquet
└── validation/                 # dev split the pipeline scores: behaviors.parquet + history.parquet
```

Notes:

- **Where to get them** - download the dataset from <https://recsys.eb.dk/dataset/>
  and unpack these files/folders here.
- **Which version to use** - the prebuild recommendations were derived from 
  the ebnerd_small folder, containing: `articles.parquet`, `validation/` and `train/`.
  The `contrastive_vector.parquet` was taken from the folder `Ekstra_Bladet_contrastive_vector`.
- **What needs what** - `articles.parquet` + `validation/` are the essential
  inputs every run checks (see `dataset_module/ebnerd/prepare.py`). `train/` is
  required to train the NRMS/LSTUR recommenders (EB-NeRD ships no pre-built
  predictions). `contrastive_vector.parquet` is required for content diversity.
- **Not data** - the `utils/` folder here is vendored *code* (the EB-NeRD-benchmark
  helpers) and is tracked in git, so you don't add it. `__MACOSX/` is unzip junk
  and is ignored.
