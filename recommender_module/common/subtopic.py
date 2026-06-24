"""Subtopic = topic diversity on a news-only subset of a dataset.

Subtopic diversity is just topic diversity measured one level finer: within a
single parent category (``news`` for MIND), how varied are the subcategories?
Instead of filtering a recommender's full output after the fact, we build a
genuine subset of the data — only the articles in the parent category, only the
impressions/clicks that still make sense — and run the *same* recommenders on
it, with each article's subcategory promoted into the topic slot. Running
``topic_diversity`` on that subset then yields subtopic diversity.

All subtopic-specific logic lives here so the pipeline and the dashboard share
one definition (including the on-disk layout via ``subtopic_subset_path``).
"""

import os


def is_in_category(article_meta, article_id, category):
    """True when ``article_id``'s topic (article_meta[id][0]) equals ``category``."""
    return article_meta.get(article_id, ("",))[0] == category


def build_subtopic_subset(impressions, article_meta, category):
    """Restrict impressions + metadata to a single parent category.

    Returns ``(sub_impressions, sub_meta, positions_by_impr)``:

      sub_impressions   : the impressions with their candidate lists narrowed to
                          the ``category`` articles. Impressions that no longer
                          make sense are dropped — those with no candidate in the
                          category, or with no *clicked* article in it (nothing to
                          recommend or to compare against). Single-click users are
                          left for ``topic_diversity`` to drop downstream, exactly
                          as for topic diversity.
      sub_meta          : {article_id: (subtopic, "none")} for the category's
                          articles — the subcategory is promoted into the topic
                          slot so ``topic_diversity`` measures subcategory variety.
                          Reuses save_user_article_map unchanged (it reads [0] as
                          topic, [1] as subtopic).
      positions_by_impr : {impr_id: [orig_index, ...]} the kept candidates'
                          positions in the original impression, used to slice a
                          model prediction's per-candidate ranks
                          (save_user_article_map_from_ranks).
    """
    sub_impressions = []
    positions_by_impr = {}
    for imp in impressions:
        positions = [
            i for i, aid in enumerate(imp.candidate_ids)
            if is_in_category(article_meta, aid, category)
        ]
        if not positions:
            continue
        labels = [imp.labels[i] for i in positions]
        if sum(labels) == 0:
            continue
        candidate_ids = [imp.candidate_ids[i] for i in positions]
        sub_impressions.append(imp._replace(candidate_ids=candidate_ids, labels=labels))
        positions_by_impr[imp.impr_id] = positions

    sub_meta = {
        aid: (sub, "none")
        for aid, (top, sub) in article_meta.items()
        if top == category
    }
    return sub_impressions, sub_meta, positions_by_impr


def subtopic_subset_path(processed_path):
    """Map a recommender's processed per-user file to its news-subset counterpart.

    e.g. ``.../predictions/prediction_processed_random.txt``
      -> ``.../predictions/subtopic/prediction_processed_random.txt``

    Single source of truth for where subset artifacts live, shared by the
    pipeline (which writes them) and the dashboard (which reads them).
    """
    pred_dir, filename = os.path.split(processed_path)
    return os.path.join(pred_dir, "subtopic", filename)
