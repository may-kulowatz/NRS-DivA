# Add your own recommender

There are two ways to evaluate your own recommender, depending on whether you
want to plug in **scoring logic** (option A) or you already have a **prediction
file** (option B). Both reuse the existing recommender contract, so once your
recommender is registered every diversity measure and the dashboard pick it up
automatically.

First, the one format everything hinges on. A **raw prediction file** lives at
`data/data_processed/<dataset>/predictions/prediction_<name>.txt` and has one
line per impression:

```
<impr_id> [r1,r2,...,rn]
```

`r1..rn` are the ranks your recommender assigns to that impression's candidate
articles, **in the same order the dataset lists them**, with `1` = top
recommendation. This is exactly the MIND-challenge leaderboard format. The
pipeline then keeps, per impression, the top *k* candidates (where *k* is the
number of articles the user actually clicked) and measures diversity over that
set — so your ranking is compared to the ground truth on an equal footing.

## Option A — paste your scoring logic

Add a class to `recommender_module/base.py`. Subclass **`_RankRecommender`**
(it already builds the processed per-user file and the per-impression view from
your rank file), so the only method you write is `generate` — produce one score
per candidate and hand the results to `save_predictions`, which turns scores into
the rank file above:

```python
class MyRecommender(_RankRecommender):
    name = "myrec"

    def generate(self, ctx):
        def results():
            for imp in ctx.impressions:          # imp: (impr_id, user_id, timestamp,
                                                 #       candidate_ids, labels)
                scores = my_score(imp)           # 1-D array, one score per imp.candidate_ids
                yield imp.impr_id, imp.user_id, scores
        save_predictions(results(), self.raw_path(ctx))
```

Then register it in `build_recommenders` (any position before ground truth):

```python
recs = [RandomRecommender(), PopularRecommender(), MyRecommender()]
```

Now run it like any other recommender:

```bash
python -m dataset_module <dataset>                 # 0. prepare data (once)
python -m recommender_module <dataset> myrec       # 1. generate your prediction
python -m diversity_module <dataset> --all         # 2. score it
```

## Option B — you already have a `prediction.txt` (e.g. from the MIND challenge)

Because the raw format *is* the MIND-challenge format, you skip generation
entirely and let the scoring stage read your file:

1. **Register a stub recommender** in `recommender_module/base.py` so the pipeline
   knows the name (and won't try to regenerate it):

   ```python
   class MyRecommender(_RankRecommender):
       name = "myrec"

       def generate(self, ctx):
           raise RuntimeError("myrec is supplied manually — skip stage 1")
   ```

   Add it to `build_recommenders` exactly as in option A.

2. **Drop your file** at
   `data/data_processed/<dataset>/predictions/prediction_myrec.txt`, in the
   `<impr_id> [ranks]` format above. The impression ids and candidate ordering
   must match the dataset's `behaviors` file.

3. **Score it — stage 2 only** (it rebuilds the per-user file from your ranks; no
   generation, no training):

   ```bash
   python -m dataset_module <dataset>            # once, so the adapter can read the data
   python -m diversity_module <dataset> --all    # scores every recommender with a file, incl. yours
   ```

In both cases, add your recommender's diversity numbers to the significance tests
by including `"myrec"` in the `RECOMMENDERS` list at the top of `statistic.py`.