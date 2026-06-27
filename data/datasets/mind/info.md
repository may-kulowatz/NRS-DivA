# MIND dataset files

The pipeline reads its raw **MIND** inputs from this folder. Everything listed
below is git-ignored (too large to commit), so a fresh clone starts without them.
To fully run the application this folder should contain:

```
data/datasets/mind/
├── MINDsmall_dev/     # dev split the pipeline scores (news.tsv + behaviors.tsv)
├── MINDsmall_train/   # training split — needed to (re)train models on demand
├── utils/             # embeddings + word/user dicts + model .yaml configs
└── model/             # model checkpoints written during training
```

How each part gets here:

- **Auto-downloaded** — `MINDsmall_dev/`, `MINDsmall_train/`, and `utils/` are
  fetched on demand by `dataset_module/mind/prepare.py` (from the HuggingFace
  `Recommenders/MIND` mirror) the first time a run needs them, so you normally
  don't add anything by hand. The dev split + `utils/` are fetched by
  `ensure_raw_data`/`ensure_utils`; the train split is fetched *best-effort* by
  `ensure_raw_data` (it's only needed to (re)train a model — e.g. NAML, or
  rebuilding the NRMS/LSTUR predictions — so scoring still works if it can't be
  obtained). `utils/` holds the embeddings/dicts content diversity reads
  (`embedding.npy`, `word_dict.pkl`, …).
- **If you need to add it manually** — should the train download fail (no network,
  mirror down), get the MIND **small** dataset from <https://msnews.github.io/>
  (or the same HuggingFace mirror) and unpack the train split into
  `MINDsmall_train/`.
- **Generated** — `model/` is created automatically the first time a model is
  trained; you never add it by hand.
- **If you want to use another MIND dataset** - like "demo" or "large", you 
  might have to adapt some paths (the application uses `/data/datasets/mind/MINDsmall_xxx`)
  or rename the folder you downloaded to MINDsmall, even though it includes MINDlarge or demo.
