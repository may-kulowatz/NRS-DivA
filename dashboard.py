"""NRS-DivA — Solara dashboard.

A single-page dashboard for exploring the diversity of the baseline
recommenders across datasets. The user picks a Dataset, a Recommender System and
a Diversity Score; the Diversity Score section then shows the real score for the
chosen dataset + recommender.

The dashboard is strictly a *viewer*: it never computes or writes anything. Scores
are read straight from the run_manifest.json file the diversity stage writes, and
the example article lists from the predictions_processed/ files the recommender
stage writes. If a dataset's outputs haven't been generated yet, it shows a
message pointing you at `python -m recommender_module <dataset>` and
`python -m diversity_module <dataset>`.

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

# The dashboard only reads what the pipeline produced — it reuses the dataset
# config and path helpers, plus the score-cache reader, but never the compute or
# write helpers.
from config import DATASETS, input_dir, output_dir
from scores import _file_sig, load_manifest, metric_value
from recommender_module.common.io import processed_filename
from diversity_module.topic_diversity import _parse_user_articles

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Options + display labels
# ---------------------------------------------------------------------------
DATASET_LABELS = {"MIND": "MIND", "ebnerd": "EB-NeRD", "mind_news": "MIND-News"}

RECOMMENDERS = {
    "MIND": ["random", "popular", "nrms", "lstur", "naml", "ground_truth"],
    "ebnerd": ["random", "popular", "nrms", "lstur", "ground_truth"],
    "mind_news": ["random", "popular", "nrms", "lstur", "naml", "ground_truth"],
}
# Every recommender across datasets, in display order. A dataset that doesn't
# support one (e.g. LSTUR / NAML on eb-nerd) shows it greyed-out, not hidden.
ALL_RECOMMENDERS = ["random", "popular", "nrms", "lstur", "naml", "ground_truth"]
REC_LABELS = {
    "random": "Random",
    "popular": "Popular",
    "nrms": "NRMS",
    "lstur": "LSTUR",
    "naml": "NAML",
    "ground_truth": "Ground truth",
}

# Extra content-embedding spaces (beyond the default contrastive one), discovered
# from the dataset configs so the dashboard's metric list/labels/text/cache-keys
# stay in sync with the pipeline without hardcoding each space. Each space `name`
# yields two dashboard metric ids: content_<name> (ILD) and content_normalized_<name>,
# mapping to manifest keys content_diversity_<name> / content_diversity_normalized_<name>.
_SPACE_LABELS = {"xlmr": "XLM-R", "bert": "BERT", "docvec": "doc2vec"}
_SPACE_DESC = {
    "xlmr": "the multilingual **XLM-RoBERTa** transformer (the same encoder the "
            "NRMS/LSTUR models use for titles)",
    "bert": "**multilingual BERT** (`bert-base-multilingual-cased`)",
    "docvec": "the classical 300-dim **document vector** (non-contextual) embedding",
}
_CONTENT_SPACES = []  # ordered union of embedding-space names across all datasets
for _cfg in DATASETS.values():
    for _name in _cfg.get("content_embeddings", {}):
        if _name not in _CONTENT_SPACES:
            _CONTENT_SPACES.append(_name)

# Extra word-average content spaces built from a different article text field than
# the title (e.g. the abstract), discovered from each dataset's `content_text_variants`.
# Each variant `name` yields the same two dashboard metric ids as an embedding space.
_TEXT_VARIANT_LABELS = {"abstract": "abstract"}
_TEXT_VARIANT_DESC = {
    "abstract": "the article **abstract** (the short summary) instead of the title",
}
_TEXT_VARIANTS = []  # ordered union of text-variant names across all datasets
for _cfg in DATASETS.values():
    for _name in _cfg.get("content_text_variants", {}):
        if _name not in _TEXT_VARIANTS:
            _TEXT_VARIANTS.append(_name)

METRICS = ["topic", "content", "content_normalized"]
METRIC_LABELS = {
    "topic": "Topic diversity",
    "content": "Content diversity (ILD)",
    "content_normalized": "Content diversity (normalized)",
}
for _name in _CONTENT_SPACES:
    _lbl = _SPACE_LABELS.get(_name, _name)
    METRICS += [f"content_{_name}", f"content_normalized_{_name}"]
    METRIC_LABELS[f"content_{_name}"] = f"Content diversity ({_lbl}, ILD)"
    METRIC_LABELS[f"content_normalized_{_name}"] = f"Content diversity ({_lbl}, normalized)"
for _name in _TEXT_VARIANTS:
    _lbl = _TEXT_VARIANT_LABELS.get(_name, _name)
    METRICS += [f"content_{_name}", f"content_normalized_{_name}"]
    METRIC_LABELS[f"content_{_name}"] = f"Content diversity ({_lbl}, ILD)"
    METRIC_LABELS[f"content_normalized_{_name}"] = f"Content diversity ({_lbl}, normalized)"

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
        "time, sentiment). It also ships ready-made 768-dimensional contrastive "
        "document embeddings (`contrastive_vector.parquet`), so content diversity "
        "is computed for it too, not just topic diversity."
    ),
    "mind_news": (
        "**MIND-News** — a news-only slice of MIND. It keeps only impressions in "
        "which the user clicked at least one article in the **news** category and "
        "that showed at least two news candidates; every non-news article is "
        "removed from the candidates and the user's history. It is built from the "
        "MIND splits by `dataset_module/mind_news/prepare.py`. Because every article is in the "
        "*news* category, topic diversity here is measured over the news "
        "**subcategories** instead."
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
        "relevance. Available for MIND, MIND-News and EB-NeRD; trained on demand."
    ),
    "lstur": (
        "**LSTUR** — a neural recommender (*Neural News Recommendation with "
        "Long- and Short-term User Representations*). It combines a long-term "
        "user embedding with a short-term interest built from the user's recent "
        "reading history, then scores each candidate by predicted relevance. "
        "Available for MIND, MIND-News and EB-NeRD; trained on demand."
    ),
    "naml": (
        "**NAML** — a neural recommender (*Neural News Recommendation with "
        "Attentive Multi-View Learning*). It encodes each article from several "
        "views — title, body, category and sub-category — and scores candidates "
        "against a user vector built from their reading history. Available for the "
        "MIND-based datasets (MIND, MIND-News); trained on demand."
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
    "content": (
        "**Content diversity (ILD)** — *intra-list diversity*: 1 minus the average "
        "pairwise cosine similarity of the articles' title embeddings. It measures "
        "how semantically different the articles are, independent of category "
        "labels. Needs article embeddings."
    ),
    "content_normalized": (
        "**Content diversity (normalized)** — the intra-list content diversity of "
        "the recommended set, rescaled *per impression* against the most and least "
        "diverse selections possible from that impression's candidate pool. 1.0 "
        "means the recommender picked about as varied a set as the candidates "
        "allowed; 0.0 about as uniform as possible. This isolates the recommender's "
        "choice from how (un)diverse the candidates happened to be. Computed only "
        "when the pipeline is run with `--normalized`."
    ),
}
# Per-embedding-space variants reuse the same wording, noting which embedding space
# they are measured in (so the user can compare representations).
for _name in _CONTENT_SPACES:
    _lbl = _SPACE_LABELS.get(_name, _name)
    _desc = _SPACE_DESC.get(_name, f"the `{_name}` embedding")
    METRIC_TEXT[f"content_{_name}"] = (
        f"**Content diversity ({_lbl}, ILD)** — the same *intra-list diversity* as "
        f"*Content diversity (ILD)*, but computed in {_desc} space instead of the "
        f"default contrastive one. Comparing the spaces shows how representation-"
        f"dependent the measured diversity is."
    )
    METRIC_TEXT[f"content_normalized_{_name}"] = (
        f"**Content diversity ({_lbl}, normalized)** — the per-impression normalized "
        f"content diversity (rescaled against each impression's candidate pool), "
        f"computed in {_desc} space. Computed only when the pipeline is run with "
        f"`--normalized`."
    )
# Text-field variants (e.g. abstract) reuse the same wording, noting which article
# text represents each article (so title- vs abstract-based diversity can be compared).
for _name in _TEXT_VARIANTS:
    _lbl = _TEXT_VARIANT_LABELS.get(_name, _name)
    _desc = _TEXT_VARIANT_DESC.get(_name, f"the article `{_name}` text")
    METRIC_TEXT[f"content_{_name}"] = (
        f"**Content diversity ({_lbl}, ILD)** — the same *intra-list diversity* as "
        f"*Content diversity (ILD)*, but each article is represented by {_desc}. "
        f"Comparing the two shows how much the measured diversity depends on which "
        f"article text is used."
    )
    METRIC_TEXT[f"content_normalized_{_name}"] = (
        f"**Content diversity ({_lbl}, normalized)** — the per-impression normalized "
        f"content diversity (rescaled against each impression's candidate pool), with "
        f"each article represented by {_desc}. Computed only when the pipeline is run "
        f"with `--normalized`."
    )

INTRO_MD = """
# NRS-DivA

Welcome to **NRS-DivA**, a small workbench for comparing how *diverse* the
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
    """Load (and cache) {article_id: title} via the dataset's adapter.

    Titles come from the raw dataset files under ``data/datasets/`` — which are
    *optional* for the dashboard (only the Examples panel uses them, purely for
    nicer labels). If they are absent — e.g. you kept only the processed pipeline
    outputs and deleted the raw inputs — return an empty map instead of raising;
    the article lists then fall back to showing article ids rather than the whole
    panel erroring out. A corrupt/unreadable file is treated the same way.
    """
    cfg = DATASETS[dataset]
    articles_file = os.path.join(input_dir(dataset), *cfg["articles"])
    if not os.path.exists(articles_file):
        return {}
    try:
        return cfg["adapter"].load_titles(articles_file)
    except Exception:
        return {}


@functools.lru_cache(maxsize=None)
def _parse_user_articles_cached(path, sig):
    """Parse a user_articles file into {user: (ids, topics)}.

    Cached on the file's change-signature so repeated dashboard renders don't
    re-read the (potentially large) file; a changed file invalidates the entry.
    """
    return _parse_user_articles(path)


# Dashboard metric name -> the metric key used in run_manifest.json.
_CACHE_KEY = {
    "topic": "topic_diversity",
    "content": "content_diversity",
    "content_normalized": "content_diversity_normalized",
}
for _name in _CONTENT_SPACES:
    _CACHE_KEY[f"content_{_name}"] = f"content_diversity_{_name}"
    _CACHE_KEY[f"content_normalized_{_name}"] = f"content_diversity_normalized_{_name}"
for _name in _TEXT_VARIANTS:
    _CACHE_KEY[f"content_{_name}"] = f"content_diversity_{_name}"
    _CACHE_KEY[f"content_normalized_{_name}"] = f"content_diversity_normalized_{_name}"


def read_score(dataset, recommender, metric):
    """Return (value, message) for one (dataset, recommender, metric).

    Read-only: the score is looked up in the run_manifest.json file the
    pipeline writes. The dashboard never computes or stores anything — if the
    score isn't there yet, it returns a message telling the user to run the
    pipeline. value is the float score, or None (with a message) otherwise.
    """
    # Metrics that simply don't apply to a dataset get an explanatory message, not
    # a "run the pipeline" one — running it wouldn't produce them. All content
    # metrics (default contrastive + each embedding space) need article embeddings;
    # available_metrics() already encodes which apply to this dataset.
    if metric not in available_metrics(dataset):
        return None, (
            f"{METRIC_LABELS[metric]} isn't available for {DATASET_LABELS[dataset]} — "
            "the matching article embeddings aren't shipped for this dataset."
        )

    # The normalized metrics are opt-in, so the scoring hint must include the flag
    # that produces them; the others are emitted by a plain scoring run.
    score_cmd = f"python -m diversity_module {dataset}"
    if metric.startswith("content_normalized"):
        score_cmd += " --normalized"
    generate_cmd = f"python -m recommender_module {dataset}  then  {score_cmd}"
    not_generated = (
        f"No {METRIC_LABELS[metric].lower()} for '{REC_LABELS[recommender]}' on "
        f"{DATASET_LABELS[dataset]} yet. Generate it with:  {generate_cmd}"
    )
    manifest = load_manifest(output_dir(dataset))
    value = metric_value(manifest, recommender, _CACHE_KEY[metric])
    if value is None:
        return None, not_generated
    return value, None


def available_metrics(dataset):
    """Diversity metrics that *apply* to a dataset. Topic always applies; the
    default content metrics need a `content_diversity` config, and each extra
    embedding space in `content_embeddings` or text field in
    `content_text_variants` adds its own content_<name> pair."""
    cfg = DATASETS[dataset]
    metrics = ["topic"]
    if cfg["content_diversity"] is not None:
        metrics += ["content", "content_normalized"]
    for name in cfg.get("content_embeddings", {}):
        metrics += [f"content_{name}", f"content_normalized_{name}"]
    for name in cfg.get("content_text_variants", {}):
        metrics += [f"content_{name}", f"content_normalized_{name}"]
    return metrics


def _dataset_scores(dataset):
    """The dataset's run manifest, or {} if none yet."""
    return load_manifest(output_dir(dataset))


def enabled_recommenders(dataset):
    """Recommenders to leave clickable: applicable to the dataset *and* already
    scored. Ones that don't apply, or haven't been run/scored yet, are greyed-out."""
    scored = {rec for rec, entry in _dataset_scores(dataset).items() if entry.get("metrics")}
    return [r for r in RECOMMENDERS[dataset] if r in scored]


def enabled_metrics(dataset):
    """Metrics to leave clickable: applicable to the dataset *and* already computed
    for at least one recommender. Ones not applicable, or not calculated yet, are
    greyed-out."""
    present = {k for entry in _dataset_scores(dataset).values() for k in entry.get("metrics", {})}
    return [m for m in available_metrics(dataset) if _CACHE_KEY[m] in present]


# ---------------------------------------------------------------------------
# Reactive state
# ---------------------------------------------------------------------------
dataset = solara.reactive("MIND")
recommender = solara.reactive("random")
metric = solara.reactive("topic")


def select_dataset(value):
    dataset.set(value)
    # Keep the selection valid for the new dataset so the highlighted pill is never a
    # greyed-out one: prefer an option that's clickable (applicable + has data), but
    # fall back to any applicable one if the dataset has no scores yet.
    recs = enabled_recommenders(value) or RECOMMENDERS[value]
    if recommender.value not in recs:
        recommender.set(recs[0])
    mets = enabled_metrics(value) or available_metrics(value)
    if metric.value not in mets:
        metric.set(mets[0])


# ---------------------------------------------------------------------------
# Reusable UI pieces
# ---------------------------------------------------------------------------
@solara.component
def PillGroup(options, selected, labels, on_select, enabled=None):
    """A row of pill-shaped, single-select buttons.

    `enabled` (optional) is the subset of `options` that apply right now; options
    outside it are rendered greyed-out and unclickable — e.g. a recommender or
    diversity metric not available for the selected dataset. When `enabled` is
    None every option is clickable.
    """
    with solara.Row(style={"flex-wrap": "wrap", "gap": "8px", "margin-bottom": "8px"}):
        for opt in options:
            is_enabled = enabled is None or opt in enabled
            is_selected = is_enabled and selected == opt
            solara.Button(
                labels.get(opt, opt),
                on_click=lambda opt=opt: on_select(opt),
                color="primary" if is_selected else None,
                outlined=not is_selected,
                disabled=not is_enabled,
                classes=["rounded-pill", "text-none"],
            )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
@solara.component
def Page():
    solara.Title("NRS-DivA — Diversity Dashboard")
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
                            ALL_RECOMMENDERS, recommender.value, REC_LABELS, recommender.set,
                            enabled=enabled_recommenders(dataset.value),
                        )
                        solara.Markdown(RECOMMENDER_TEXT[recommender.value])

                with solara.Details(
                    f"Diversity Score  ·  {METRIC_LABELS[metric.value]}", expand=True
                ):
                    with solara.Column():
                        PillGroup(
                            METRICS, metric.value, METRIC_LABELS, metric.set,
                            enabled=enabled_metrics(dataset.value),
                        )
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
                        ExamplesPanel(dataset.value, recommender.value)


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

    Reads from the run_manifest.json cache via read_score, so this is cheap
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
    """Return {user: (ids, topics)} for kind 'clicked' or 'recommended'.

    'clicked' reads the ground-truth user-article map; 'recommended' reads the
    chosen recommender's map. Returns None when the file doesn't exist yet.
    """
    rec = "ground_truth" if kind == "clicked" else recommender
    path = _processed_path(dataset, rec)
    if not os.path.exists(path):
        return None
    return _parse_user_articles_cached(path, _file_sig(path))


@solara.component
def ArticleList(title, ids, topics, titles):
    with solara.Card(title, style={"height": "100%"}):
        with solara.Column(gap="4px"):
            for aid, topic in zip(ids, topics):
                name = titles.get(aid) or f"(article {aid})"
                if len(name) > 80:
                    name = name[:80] + "…"
                tp = _fmt_topic(topic)
                suffix = f"  ·  _{tp}_" if tp else ""
                solara.Markdown(f"- {name}{suffix}")


@solara.component
def ExamplesPanel(dataset, recommender):
    clicked_map = _articles_for_user(dataset, recommender, "clicked")
    rec_map = _articles_for_user(dataset, recommender, "recommended")

    # Hooks must run unconditionally (no early return before this). Only show
    # users that actually contribute to the diversity score — i.e. with at least
    # two clicked articles (single-click users are excluded from the metrics, so
    # showing one here would be misleading).
    users = sorted(
        u for u in (set(clicked_map) & set(rec_map))
        if len(clicked_map[u][0]) >= 2
    ) if clicked_map and rec_map else []
    counter = example_reshuffle.value
    user = solara.use_memo(
        lambda: random.choice(users) if users else None,
        [dataset, counter, len(users)],
    )

    if clicked_map is None or rec_map is None:
        solara.Info(f"Generate outputs first with:  python -m recommender_module "
                    f"{dataset}  then  python -m diversity_module {dataset}")
        return
    if not users or user is None:
        solara.Info("No users with both clicks and recommendations are available.")
        return

    titles = _get_titles(dataset)
    clicked_ids, clicked_topics = clicked_map[user]
    rec_ids, rec_topics = rec_map[user]

    with solara.Row(style={"align-items": "center", "gap": "12px", "margin-bottom": "8px"}):
        solara.Markdown(f"**User `{user}`** — {len(clicked_ids)} clicked article(s)")
        solara.Button(
            "Pick another user",
            on_click=lambda: example_reshuffle.set(example_reshuffle.value + 1),
            outlined=True,
            classes=["rounded-pill", "text-none"],
        )

    with solara.Columns([1, 1]):
        ArticleList(
            "Actually clicked (ground truth)",
            clicked_ids, clicked_topics, titles,
        )
        ArticleList(
            f"Recommended by {REC_LABELS[recommender]}",
            rec_ids, rec_topics, titles,
        )
