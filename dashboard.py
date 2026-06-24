"""EchoBench — Solara dashboard.

A single-page dashboard for exploring the diversity of the baseline
recommenders across datasets. The user picks a Dataset, a Recommender System and
a Diversity Score; the Diversity Score section then shows the real score for the
chosen dataset + recommender.

The dashboard is strictly a *viewer*: it never computes or writes anything. Scores
are read straight from the diversity_scores.json file the pipeline writes, and the
example article lists from the pipeline's predictions_processed/ files. If a
dataset's outputs haven't been generated yet, it shows a message pointing you at
`python pipeline.py <dataset>`.

Run with:
    solara run dashboard.py
"""

import functools
import os
import random

import solara
import solara.lab
from matplotlib.figure import Figure


PRIMARY_GREEN = "#2e7d32"

# The dashboard only reads what the pipeline produced — it reuses the pipeline's
# dataset config and path helpers, plus the cache reader, but never the compute
# or write helpers.
from pipeline import (
    DATASETS,
    input_dir,
    output_dir,
    _file_sig,
    _load_score_cache,
)
from recommender_module.common.io import processed_filename
from diversity_module.topic_diversity import _parse_user_articles

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Options + display labels
# ---------------------------------------------------------------------------
DATASET_LABELS = {"MIND": "MIND", "ebnerd": "EB-NeRD"}

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
recommendations of different recommender systems are, across different
news datasets.

Use the controls below to pick a **dataset**, a **recommender system** and a
**diversity score**. The diversity score is computed from the recommender's
output for the dataset you selected.
"""


# ---------------------------------------------------------------------------
# Reading the pipeline's outputs (scores + per-user files). Read-only.
# ---------------------------------------------------------------------------
def _processed_path(dataset, recommender):
    return os.path.join(
        output_dir(dataset), "predictions_processed",
        processed_filename(recommender),
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


def read_score(dataset, recommender, metric):
    """Return (value, message) for one (dataset, recommender, metric).

    Read-only: the score is looked up in the diversity_scores.json file the
    pipeline writes. The dashboard never computes or stores anything — if the
    score isn't there yet, it returns a message telling the user to run the
    pipeline. value is the float score, or None (with a message) otherwise.
    """
    cfg = DATASETS[dataset]
    # Metrics that simply don't apply to a dataset get an explanatory message,
    # not a "run the pipeline" one — running it wouldn't produce them.
    if metric == "subtopic" and cfg["subtopic_category"] is None:
        return None, (
            f"Subtopic diversity isn't defined for {DATASET_LABELS[dataset]} — "
            "its subcategories don't map to a parent category."
        )
    if metric == "content" and cfg["content_diversity"] is None:
        return None, (
            f"Content diversity isn't available for {DATASET_LABELS[dataset]} — "
            "no article embeddings are shipped for this dataset."
        )

    not_generated = (
        f"No {METRIC_LABELS[metric].lower()} for '{REC_LABELS[recommender]}' on "
        f"{DATASET_LABELS[dataset]} yet. Generate it with:  python pipeline.py {dataset}"
    )
    cache_file = os.path.join(output_dir(dataset), "diversity_scores.json")
    if not os.path.exists(cache_file):
        return None, not_generated

    cache = _load_score_cache(cache_file)
    entry = cache.get(recommender, {}).get(_CACHE_KEY[metric])
    if entry is None:
        return None, not_generated
    return entry["value"], None


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

    with solara.Column(style={"max-width": "1400px", "margin": "0 auto", "padding": "24px"}):
        solara.Markdown(INTRO_MD)

        # Two columns: the controls (Dataset / Recommender / Diversity Score) on
        # the left, the visualization next to them on the right. Solara stacks
        # them automatically on narrow screens.
        with solara.Columns([1, 1]):
            # --- Left column: controls ---------------------------------------
            # Each section is a foldable panel (open by default, click the header
            # to collapse). The current selection is shown in the header so it
            # stays visible even when the panel is collapsed.
            with solara.Column():
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
                        value, message = read_score(dataset.value, recommender.value, metric.value)
                        ScoreCard(value, message)

            # --- Right column: visualization ---------------------------------
            with solara.Column():
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

    Reads from the diversity_scores.json cache via read_score, so this is cheap
    and recommenders the pipeline hasn't scored yet are simply omitted.
    """
    result = {}
    for rec in RECOMMENDERS[dataset]:
        value, _ = read_score(dataset, rec, metric)
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
