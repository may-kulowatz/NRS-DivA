"""EchoBench — Solara dashboard.

A single-page dashboard for exploring the diversity of the baseline
recommenders across datasets. The user picks a Dataset, a Recommender System and
a Diversity Score; the Diversity Score section then shows the real score for the
chosen dataset + recommender. Scores are read from the same
predictions/diversity_scores.json cache the pipeline writes, and only computed
(and written back) when the cache has no fresh entry — so the dashboard and the
pipeline never disagree and expensive metrics aren't recomputed needlessly.

Run with:
    solara run dashboard.py
"""

import functools
import os
import random

import solara
import solara.lab
from matplotlib.figure import Figure

# App primary color (Material green). Used for the vuetify theme as well as the
# matplotlib chart and score number, which aren't theme-aware.
PRIMARY_GREEN = "#2e7d32"

from pipeline import (
    DATASETS,
    input_dir,
    output_dir,
    _compute_run_scores,
    _file_sig,
    _load_score_cache,
    _save_score_cache,
)
from recommender_module.common.io import processed_filename
from recommender_module.common.subtopic import subtopic_subset_path
from diversity_module.topic_diversity import (
    topic_diversity,
    subtopic_diversity,
    _parse_user_articles,
)
from diversity_module.content_diversity import content_diversity, load_news_embeddings

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Options + display labels
# ---------------------------------------------------------------------------
DATASET_LABELS = {"MIND": "MIND", "ebnerd": "EB-NeRD"}

# Recommenders available per dataset (eb-nerd ships no model predictions).
RECOMMENDERS = {
    "MIND": ["random", "popular", "nrms", "lstur", "ground_truth"],
    "ebnerd": ["random", "popular", "ground_truth"],
}
REC_LABELS = {
    "random": "Random",
    "popular": "Popular",
    "nrms": "NRMS",
    "lstur": "LSTUR",
    "ground_truth": "Ground truth",
}

METRICS = ["topic", "subtopic", "content"]
METRIC_LABELS = {
    "topic": "Topic diversity",
    "subtopic": "Subtopic diversity",
    "content": "Content diversity (ILD)",
}


# ---------------------------------------------------------------------------
# Explanatory copy shown for each dataset / recommender / metric choice
# ---------------------------------------------------------------------------
DATASET_TEXT = {
    "MIND": (
        "**MIND** (Microsoft News Dataset) — English news from *Microsoft News*. "
        "Each impression records the candidate articles shown to a user, which "
        "one(s) they clicked, and the user's reading history. Every article has a "
        "**category** (e.g. sports, finance, news) and a **subcategory**, plus a "
        "title and abstract. This app uses the small dev split."
    ),
    "ebnerd": (
        "**EB-NeRD** (Ekstra Bladet News Recommendation Dataset) — Danish news from "
        "the tabloid *Ekstra Bladet*. Each impression lists the articles in view "
        "and which were clicked. Articles carry one **category** plus several "
        "free-text **topics**, along with richer signals (full body text, read "
        "time, sentiment). Its subcategories are opaque numeric codes with no "
        "parent category, so only topic diversity is computed for it."
    ),
}

RECOMMENDER_TEXT = {
    "random": (
        "**Random** — assigns each candidate article a uniform random score. A "
        "no-personalization baseline; it tends to look highly diverse precisely "
        "because it ignores relevance."
    ),
    "popular": (
        "**Popular** — ranks candidates only by how often they were clicked in "
        "*earlier* impressions (global popularity), identically for every user. "
        "Counts are accumulated in time order, so no future clicks leak into a "
        "score."
    ),
    "nrms": (
        "**NRMS** — a neural recommender (*Neural News Recommendation with "
        "Multi-Head Self-Attention*). It builds a user representation from the "
        "articles in their history and scores each candidate by predicted "
        "relevance. Pre-computed predictions ship with MIND only."
    ),
    "lstur": (
        "**LSTUR** — a neural recommender (*Neural News Recommendation with "
        "Long- and Short-term User Representations*). It combines a long-term "
        "user embedding with a short-term interest built from the user's recent "
        "reading history, then scores each candidate by predicted relevance. "
        "Pre-computed predictions ship with MIND only."
    ),
    "ground_truth": (
        "**Ground truth** — not a recommender but the reference point: the articles "
        "the user *actually* clicked. Every other system's diversity is judged "
        "against this."
    ),
}

METRIC_TEXT = {
    "topic": (
        "**Topic diversity** — the share of *distinct* topics among a user's "
        "articles (unique topics ÷ total topic assignments), averaged over users "
        "with more than one click. 1.0 means every article has a different topic; "
        "low means many share the same one. *Topic* = the article's category "
        "(MIND) or its set of topic labels (EB-NeRD)."
    ),
    "subtopic": (
        "**Subtopic diversity** — the same idea one level finer: within a parent "
        "category (here *news* for MIND), the share of distinct subcategories. It "
        "captures variety *inside* a single topic. Defined for MIND only — "
        "EB-NeRD's subcategory codes don't map to a parent category."
    ),
    "content": (
        "**Content diversity (ILD)** — *intra-list diversity*: 1 minus the average "
        "pairwise cosine similarity of the articles' title embeddings. It measures "
        "how semantically different the articles are, independent of category "
        "labels. Needs embeddings, so MIND only."
    ),
}

INTRO_MD = """
# EchoBench

Welcome to **EchoBench**, a small workbench for comparing how *diverse* the
recommendations of different baseline recommender systems are, across different
news datasets.

Use the controls below to pick a **dataset**, a **recommender system** and a
**diversity score**. The diversity score is computed live from the recommender's
output for the dataset you selected.
"""


# ---------------------------------------------------------------------------
# Score computation (reuses the pipeline's configuration + metric functions)
# ---------------------------------------------------------------------------
def _processed_path(dataset, recommender):
    return os.path.join(
        output_dir(dataset), "predictions_processed",
        processed_filename(recommender),
    )


@functools.lru_cache(maxsize=None)
def _get_embeddings(dataset):
    """Load (and cache) the news embeddings a dataset's content diversity needs."""
    cfg = DATASETS[dataset]
    in_dir = input_dir(dataset)
    cd = cfg["content_diversity"]
    return load_news_embeddings(
        os.path.join(in_dir, *cfg["articles"]),
        os.path.join(in_dir, *cd["embedding"]),
        os.path.join(in_dir, *cd["word_dict"]),
    )


@functools.lru_cache(maxsize=None)
def _get_titles(dataset):
    """Load (and cache) {article_id: title} via the dataset's adapter."""
    cfg = DATASETS[dataset]
    articles_file = os.path.join(input_dir(dataset), *cfg["articles"])
    return cfg["adapter"].load_titles(articles_file)


@functools.lru_cache(maxsize=None)
def _parse_user_articles_cached(path, sig):
    """Parse a user_articles file into {user: (ids, topics, subtopics)}.

    Cached on the file's change-signature so repeated dashboard renders don't
    re-read the (potentially large) file; a changed file invalidates the entry.
    """
    return _parse_user_articles(path)


# Dashboard metric name -> the key used in predictions/diversity_scores.json.
_CACHE_KEY = {
    "topic": "topic_diversity",
    "subtopic": "subtopic_diversity",
    "content": "content_diversity",
}


def _metric_fn(dataset, metric):
    """Return the metric function fn(path) -> float for the chosen metric."""
    if metric == "topic":
        return topic_diversity
    if metric == "subtopic":
        # Subtopic is topic diversity on the news-subset sibling file built by
        # the pipeline (predictions/subtopic/...).
        return lambda p: subtopic_diversity(subtopic_subset_path(p))
    if metric == "content":
        return lambda p: content_diversity(p, _get_embeddings(dataset))
    raise ValueError(f"Unknown metric '{metric}'")


def compute_score(dataset, recommender, metric):
    """Return (value, message).

    The score is read from the shared predictions/diversity_scores.json cache
    written by the pipeline. It is only recomputed if the cache has no entry for
    this (recommender, metric) or the cached entry is stale (the recommender's
    user-article file changed since it was cached); a freshly computed score is
    written back so both the dashboard and the pipeline stay in sync.

    value is the float diversity score, or None when it can't be computed — in
    which case message explains why (and is shown to the user).
    """
    cfg = DATASETS[dataset]
    path = _processed_path(dataset, recommender)
    if not os.path.exists(path):
        return None, (
            f"No output for '{REC_LABELS[recommender]}' on {DATASET_LABELS[dataset]} yet. "
            f"Generate it with:  python pipeline.py {dataset}"
        )
    if metric == "subtopic" and cfg["subtopic_category"] is None:
        return None, (
            f"Subtopic diversity isn't defined for {DATASET_LABELS[dataset]} — "
            "its subcategories don't map to a parent category."
        )
    if metric == "subtopic" and not os.path.exists(subtopic_subset_path(path)):
        return None, (
            f"No subtopic subset for '{REC_LABELS[recommender]}' on "
            f"{DATASET_LABELS[dataset]} yet. Generate it with:  python pipeline.py {dataset}"
        )
    if metric == "content" and cfg["content_diversity"] is None:
        return None, (
            f"Content diversity isn't available for {DATASET_LABELS[dataset]} — "
            "no article embeddings are shipped for this dataset."
        )

    # Reuse the pipeline's cache logic: it returns the cached value when the
    # input file's signature still matches, and only calls the metric function
    # on a miss.
    key = _CACHE_KEY[metric]
    cache_file = os.path.join(output_dir(dataset), "diversity_scores.json")
    cache = _load_score_cache(cache_file)
    prev_run = cache.get(recommender, {})

    scores, run_cache, n_computed = _compute_run_scores(
        path, [(key, _metric_fn(dataset, metric))], prev_run
    )

    if n_computed:
        # Persist the freshly computed score without dropping other cached
        # metrics already stored for this recommender.
        cache[recommender] = {**prev_run, **run_cache}
        _save_score_cache(cache_file, cache)

    return scores[key], None


# ---------------------------------------------------------------------------
# Reactive state
# ---------------------------------------------------------------------------
dataset = solara.reactive("MIND")
recommender = solara.reactive("random")
metric = solara.reactive("topic")


def select_dataset(value):
    dataset.set(value)
    # Keep the recommender valid for the new dataset (e.g. NRMS only on MIND).
    if recommender.value not in RECOMMENDERS[value]:
        recommender.set(RECOMMENDERS[value][0])


# ---------------------------------------------------------------------------
# Reusable UI pieces
# ---------------------------------------------------------------------------
@solara.component
def PillGroup(options, selected, labels, on_select):
    """A row of pill-shaped, single-select buttons."""
    with solara.Row(style={"flex-wrap": "wrap", "gap": "8px", "margin-bottom": "8px"}):
        for opt in options:
            is_selected = selected == opt
            solara.Button(
                labels.get(opt, opt),
                on_click=lambda opt=opt: on_select(opt),
                color="primary" if is_selected else None,
                outlined=not is_selected,
                classes=["rounded-pill", "text-none"],
            )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
@solara.component
def Page():
    solara.Title("EchoBench — Diversity Dashboard")
    # Make the app's primary accent green (pills, outlined buttons, etc.).
    solara.lab.theme.themes.light.primary = PRIMARY_GREEN
    solara.lab.theme.themes.dark.primary = PRIMARY_GREEN

    with solara.Column(style={"max-width": "860px", "margin": "0 auto", "padding": "24px"}):
        solara.Markdown(INTRO_MD)

        # Each section is a foldable panel (open by default, click the header to
        # collapse). The current selection is shown in the header so it stays
        # visible even when the panel is collapsed.
        with solara.Details(f"Dataset  ·  {DATASET_LABELS[dataset.value]}", expand=True):
            with solara.Column():
                PillGroup(list(DATASETS.keys()), dataset.value, DATASET_LABELS, select_dataset)
                solara.Markdown(DATASET_TEXT[dataset.value])

        with solara.Details(
            f"Recommender System  ·  {REC_LABELS[recommender.value]}", expand=True
        ):
            with solara.Column():
                PillGroup(
                    RECOMMENDERS[dataset.value], recommender.value, REC_LABELS, recommender.set
                )
                solara.Markdown(RECOMMENDER_TEXT[recommender.value])

        with solara.Details(
            f"Diversity Score  ·  {METRIC_LABELS[metric.value]}", expand=True
        ):
            with solara.Column():
                PillGroup(METRICS, metric.value, METRIC_LABELS, metric.set)
                solara.Markdown(METRIC_TEXT[metric.value])
                value, message = compute_score(dataset.value, recommender.value, metric.value)
                ScoreCard(value, message)

        with solara.Details("Visualization and Examples", expand=True):
            with solara.Column():
                solara.Markdown(
                    f"How the recommenders compare on **{METRIC_LABELS[metric.value]}** for "
                    f"**{DATASET_LABELS[dataset.value]}**. Your selected recommender "
                    f"(**{REC_LABELS[recommender.value]}**) is highlighted."
                )
                ComparisonChart(dataset.value, recommender.value, metric.value)

                solara.Markdown(
                    "### Example — a random user\n"
                    "The articles this user actually clicked, next to what "
                    f"**{REC_LABELS[recommender.value]}** would have recommended them."
                )
                ExamplesPanel(dataset.value, recommender.value, metric.value)


@solara.component
def ScoreCard(value, message):
    caption = (
        f"{METRIC_LABELS[metric.value]} — {DATASET_LABELS[dataset.value]} / "
        f"{REC_LABELS[recommender.value]}"
    )
    with solara.Card(style={"margin-top": "8px"}):
        solara.Markdown(f"**{caption}**")
        if value is None:
            solara.Info(message)
        else:
            solara.Markdown(
                f"<div style='font-size:48px;font-weight:700;color:{PRIMARY_GREEN}'>{value:.4f}</div>",
            )
            solara.Markdown(
                "_Higher means the recommended articles are more diverse "
                "(range 0–1)._"
            )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
_CHOSEN_COLOR = PRIMARY_GREEN   # highlight for the selected recommender's bar
_OTHER_COLOR = "#cfd8dc"        # muted grey for the rest

# Solara alert component to use per interpretation severity.
_ALERT = {"warning": solara.Warning, "info": solara.Info, "success": solara.Success}


def scores_for_all_recommenders(dataset, metric):
    """Return {recommender: value} for every recommender that has this metric.

    Reuses compute_score (and therefore the JSON cache), so this is cheap.
    """
    result = {}
    for rec in RECOMMENDERS[dataset]:
        value, _ = compute_score(dataset, rec, metric)
        if value is not None:
            result[rec] = value
    return result


def interpret_vs_ground_truth(recommender, scores, metric):
    """Compare the chosen recommender's score against ground truth.

    Returns (message, severity) where severity is a key of _ALERT, or
    (None, None) when no meaningful comparison can be made.
    """
    chosen = scores.get(recommender)
    if chosen is None:
        return None, None

    label = METRIC_LABELS[metric]
    if recommender == "ground_truth":
        return (
            f"This is the ground-truth reference — the {label.lower()} of the "
            "articles users actually clicked.",
            "info",
        )

    gt = scores.get("ground_truth")
    if not gt:  # no ground truth, or ground truth is 0 → ratio undefined
        return None, None

    ratio = chosen / gt
    if ratio < 0.75:
        return (
            f"{label} much lower than ground truth: might lead into echo chambers!",
            "warning",
        )
    if ratio < 0.95:
        return (f"{label} somewhat lower than ground truth.", "info")
    if ratio <= 1.05:
        return (f"{label} about the same as ground truth.", "success")
    return (
        f"{label} higher than ground truth — more diverse than what users "
        "actually clicked.",
        "info",
    )


@solara.component
def ComparisonChart(dataset, recommender, metric):
    scores = scores_for_all_recommenders(dataset, metric)
    if not scores:
        solara.Info(
            f"{METRIC_LABELS[metric]} isn't available for {DATASET_LABELS[dataset]}, "
            "so there's nothing to compare here."
        )
        return

    # Keep a stable recommender order; drop any without a value for this metric.
    recs = [r for r in RECOMMENDERS[dataset] if r in scores]
    values = [scores[r] for r in recs]
    # Mark ground truth with a trailing "*" — it's the reference, not a recommender.
    labels = [REC_LABELS[r] + (" *" if r == "ground_truth" else "") for r in recs]
    colors = [_CHOSEN_COLOR if r == recommender else _OTHER_COLOR for r in recs]

    # Object-oriented Figure (not pyplot) to avoid global state on the server.
    fig = Figure(figsize=(6.2, 3.4))
    ax = fig.subplots()
    bars = ax.bar(labels, values, color=colors, edgecolor="#90a4ae")
    # Ground truth isn't a recommender, so always set it apart with a hatch
    # pattern (and a darker edge so the stripes read on both green and grey
    # fills) — independent of which recommender is currently selected.
    for bar, r in zip(bars, recs):
        if r == "ground_truth":
            bar.set_hatch("//")
            bar.set_edgecolor("#607d8b")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_title(f"{METRIC_LABELS[metric]} by recommender — {DATASET_LABELS[dataset]}")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value,
            f"{value:.3f}", ha="center", va="bottom", fontsize=9,
        )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    solara.FigureMatplotlib(fig)

    if "ground_truth" in recs:
        solara.Markdown(
            "_\\* reference — the articles users actually clicked, not a "
            "recommender (shown hatched)._"
        )

    message, severity = interpret_vs_ground_truth(recommender, scores, metric)
    if message:
        _ALERT[severity](message)


# ---------------------------------------------------------------------------
# Concrete examples: one random user's clicked vs recommended articles
# ---------------------------------------------------------------------------
# Incremented by the "Pick another user" button to draw a fresh random user.
example_reshuffle = solara.reactive(0)


def _fmt_topic(topic):
    """Make a stored topic field human-readable (undo the '|' / '_' encoding)."""
    if not topic or topic == "none":
        return ""
    return topic.replace("_", " ").replace("|", ", ")


def _articles_for_user(dataset, recommender, kind):
    """Return {user: (ids, topics, subtopics)} for kind 'clicked' or 'recommended'.

    'clicked' reads the ground-truth user-article map; 'recommended' reads the
    chosen recommender's map. Returns None when the file doesn't exist yet.
    """
    rec = "ground_truth" if kind == "clicked" else recommender
    path = _processed_path(dataset, rec)
    if not os.path.exists(path):
        return None
    return _parse_user_articles_cached(path, _file_sig(path))


@solara.component
def ArticleList(title, ids, topics, subtopics, titles, show_subtopic, category):
    with solara.Card(title, style={"height": "100%"}):
        with solara.Column(gap="4px"):
            for aid, topic, subtopic in zip(ids, topics, subtopics):
                name = titles.get(aid) or f"(article {aid})"
                if len(name) > 80:
                    name = name[:80] + "…"
                # topic always; subtopic only when subtopic diversity is selected.
                bits = []
                tp = _fmt_topic(topic)
                if tp:
                    bits.append(tp)
                if show_subtopic:
                    st = _fmt_topic(subtopic)
                    if st:
                        bits.append(st)
                suffix = f"  ·  _{' › '.join(bits)}_" if bits else ""
                # When subtopic diversity is selected, the articles it actually
                # considers are those in the parent category — mark them in bold.
                considered = show_subtopic and category is not None and topic == category
                name_md = f"**{name}**" if considered else name
                solara.Markdown(f"- {name_md}{suffix}")


@solara.component
def ExamplesPanel(dataset, recommender, metric):
    clicked_map = _articles_for_user(dataset, recommender, "clicked")
    rec_map = _articles_for_user(dataset, recommender, "recommended")

    # Hooks must run unconditionally (no early return before this).
    users = sorted(set(clicked_map) & set(rec_map)) if clicked_map and rec_map else []
    counter = example_reshuffle.value
    user = solara.use_memo(
        lambda: random.choice(users) if users else None,
        [dataset, counter, len(users)],
    )

    if clicked_map is None or rec_map is None:
        solara.Info(f"Generate outputs first with:  python pipeline.py {dataset}")
        return
    if not users or user is None:
        solara.Info("No users with both clicks and recommendations are available.")
        return

    titles = _get_titles(dataset)
    clicked_ids, clicked_topics, clicked_subtopics = clicked_map[user]
    rec_ids, rec_topics, rec_subtopics = rec_map[user]

    show_subtopic = metric == "subtopic"
    category = DATASETS[dataset]["subtopic_category"]

    with solara.Row(style={"align-items": "center", "gap": "12px", "margin-bottom": "8px"}):
        solara.Markdown(f"**User `{user}`** — {len(clicked_ids)} clicked article(s)")
        solara.Button(
            "Pick another user",
            on_click=lambda: example_reshuffle.set(example_reshuffle.value + 1),
            outlined=True,
            classes=["rounded-pill", "text-none"],
        )

    if show_subtopic and category is not None:
        solara.Markdown(
            f"Articles in the **{category}** category — the ones subtopic diversity "
            "is measured on — are shown in **bold**."
        )

    with solara.Columns([1, 1]):
        ArticleList(
            "Actually clicked (ground truth)",
            clicked_ids, clicked_topics, clicked_subtopics, titles, show_subtopic, category,
        )
        ArticleList(
            f"Recommended by {REC_LABELS[recommender]}",
            rec_ids, rec_topics, rec_subtopics, titles, show_subtopic, category,
        )
