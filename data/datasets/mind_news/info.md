# mind_news dataset files

**mind_news** is a news-only subset of MIND — every impression kept clicked at
least one `news` article and showed ≥2 news candidates, with all non-news
articles stripped from the candidates and history. The whole folder is git-ignored
and **generated automatically**, so you normally add nothing here by hand:

```
data/datasets/mind_news/
├── MINDnews_dev/     # news-only dev split   (news.tsv + behaviors.tsv)
├── MINDnews_train/   # news-only train split (news.tsv + behaviors.tsv)
├── utils/            # mind_news's own embeddings/dicts + model .yaml configs
└── model/            # model checkpoints written during training
```

How it gets here:

- **Built from MIND** — `dataset_module/mind_news/prepare.py` derives the
  `MINDnews_*` splits from the sibling MIND data, and builds mind_news's own
  copy of `utils/` on demand (before content diversity reads its embeddings).
- **Prerequisite** — a populated `data/datasets/mind/` (see its `info.md`).
  mind_news is built from MIND's splits and utils, so MIND must be available
  first.
- **Generated** — `model/` is created automatically when a model is trained.
