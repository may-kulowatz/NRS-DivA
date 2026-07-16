"""Statistical significance of recommender diversity vs. the ground truth.

The run manifest stores a single aggregate diversity number per recommender (e.g.
NRMS topic diversity = 0.703) next to the ground-truth number (0.804). Eyeballing
"0.703 < 0.804" tells you NRMS *looks* less diverse than the users' real clicks —
but not whether that gap is **statistically significant** or just sampling noise.

This script answers, per recommender and per diversity metric:

    "Does this recommender's diversity sit significantly ABOVE or BELOW the
     ground truth?"

The same user IDs appear in every recommender's processed file and in the
ground-truth file, so the observations are *paired*: for each user we have both
the diversity of what the recommender served them and the diversity of their real
clicks. We test the difference d = recommender - ground_truth with a **paired
t-test** (scipy.stats.ttest_rel, H0: mean(d) = 0). A positive mean means the
recommender is more diverse than the users' real clicks; negative means less.

Alongside the test we report the paired effect size (Cohen's d_z = mean(d) /
sd(d)): with ~50 000 users the t-test flags almost any non-zero gap as
significant, so d_z is what says whether the gap is *practically* big (< 0.2
negligible, < 0.5 small, < 0.8 medium, else large).

Metrics (rebuilt with the same definitions the pipeline uses):
  * topic_diversity   : per user, (# unique topics) / (# topic assignments), for
                        users with >1 article and >=1 non-"none" topic.
  * content_diversity : per user, intra-list diversity (ILD) = 1 - mean pairwise
                        cosine similarity of the user's article title embeddings,
                        for users with >=2 embeddable articles.
  * content_diversity_normalized : per impression, the recommended set's ILD
                        rescaled to [0, 1] against the min/max ILD achievable from
                        that impression's candidate pool. Rebuilt from the
                        impressions + each recommender's picks and paired by
                        impression. Slower, but a first-class diversity measure.

Both content metrics are computed once per content-embedding space the dataset
defines (matching the pipeline): the primary space built from article titles,
plus any ``content_text_variants`` — for MIND / mind_news, the abstract. Those
add ``content_diversity_abstract`` and ``content_diversity_normalized_abstract``,
built exactly like the title metrics but averaging the abstract's word
embeddings, so title- and abstract-based diversity can be compared side by side.

Output (per metric, to data_processed/<dataset>/statistics/):
  * fig_paired_ttest_<metric>.png : mean difference (recommender − GT) with 95% CI
                                    and a zero reference line, each row annotated
                                    with Cohen's d_z. A CI that excludes 0 is
                                    significant at the 5% level. ``<metric>`` gains
                                    a space suffix for text variants, e.g.
                                    fig_paired_ttest_content_diversity_abstract.png.

Run:  python statistic.py [DATASET]      (DATASET defaults to "MIND")
"""

import os
import sys
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")  # file output only, no interactive display
import matplotlib.pyplot as plt
from scipy import stats

# Reuse the project's parsing + diversity definitions so our per-user values
# match the pipeline's aggregates exactly.
from config import DATASETS, input_dir, output_dir, resolve_dataset
from recommender_module.base import build_context
from diversity_module.topic_diversity import _parse_user_articles
from diversity_module.content_diversity import (
    load_news_embeddings,
    load_precomputed_embeddings,
    _ild,
)
from diversity_module.content_diversity_normalized import (
    per_impression_normalized_content_diversity,
)

# Per-impression sampling budget for the normalized metric's min/max estimate —
# the same default the diversity stage uses (diversity_module.__main__).
_NORMALIZED_MAX_COMBINATIONS = 1000


# Recommenders to test against the ground-truth baseline, in display order.
RECOMMENDERS = ["random", "popular", "nrms", "lstur", "naml"]
GROUND_TRUTH = "ground_truth"

REC_LABELS = {
    "random": "Random",
    "popular": "Popular",
    "nrms": "NRMS",
    "lstur": "LSTUR",
    "naml": "NAML",
    "ground_truth": "Ground truth",
}
METRIC_LABELS = {
    "topic_diversity": "Topic diversity",
    "content_diversity": "Content diversity (ILD)",
    "content_diversity_normalized": "Normalized content diversity",
}


def _metric_label(base_key, space_label, suffix):
    """Human label for a content metric in a given text space.

    The primary space (empty suffix) keeps the plain label; a text variant such
    as the abstract appends its label, e.g. "Content diversity (ILD) — abstract".
    """
    base = METRIC_LABELS[base_key]
    return base if not suffix else f"{base} — {space_label}"


# --------------------------------------------------------------------------- #
# Locating the processed per-user files
# --------------------------------------------------------------------------- #
def _processed_path(out_dir, recommender):
    """Path to a recommender's processed per-user file.

    Ground truth uses ``processed_ground_truth.txt``; every other recommender
    uses ``prediction_processed_<rec>.txt`` (matching the pipeline's naming).
    """
    sub = os.path.join(out_dir, "predictions_processed")
    if recommender == GROUND_TRUTH:
        return os.path.join(sub, "processed_ground_truth.txt")
    return os.path.join(sub, f"prediction_processed_{recommender}.txt")


# --------------------------------------------------------------------------- #
# Per-user diversity values (keyed by user id, so we can pair across files)
# --------------------------------------------------------------------------- #
def per_user_topic_diversity(path):
    """{user_id: topic-diversity ratio} for the users that qualify.

    Same rule as diversity_module.topic_diversity, but kept per user instead of
    averaged: a user counts only with >1 article and at least one non-"none"
    topic; the value is (unique topics) / (total topic assignments).
    """
    out = {}
    for uid, (ids, topics) in _parse_user_articles(path).items():
        if len(ids) <= 1:
            continue
        flat = [t for group in topics for t in group.split("|") if t != "none"]
        if not flat:
            continue
        out[uid] = len(set(flat)) / len(flat)
    return out


def per_user_content_diversity(path, news_embeddings):
    """{user_id: intra-list diversity (ILD)} for the users that qualify.

    Same rule as diversity_module.content_diversity, kept per user: a user counts
    only with >=2 articles that have a known title embedding; the value is
    1 - mean pairwise cosine similarity of those embeddings.
    """
    out = {}
    for uid, (ids, _) in _parse_user_articles(path).items():
        if len(ids) <= 1:
            continue
        vectors = [news_embeddings[n] for n in ids if n in news_embeddings]
        if len(vectors) <= 1:
            continue
        out[uid] = _ild(vectors)
    return out


# --------------------------------------------------------------------------- #
# The paired t-test of one recommender against ground truth
# --------------------------------------------------------------------------- #
def _cohen_d_label(dz):
    """Conventional plain-language size band for a paired Cohen's d_z."""
    a = abs(dz)
    if math.isnan(a):
        return "n/a"
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def paired_ttest(rec_vals, gt_vals):
    """Paired t-test of a recommender's values against ground truth.

    rec_vals, gt_vals : {key: value} dicts, keyed by user id (the per-user
    metrics) or impression id (the per-impression normalized metric). The paired
    sample is over the keys present in both. Returns the group means, the mean
    paired difference d = recommender - ground_truth with its 95% t-interval, the
    t-test result, and the paired effect size (Cohen's d_z). A positive mean
    difference means the recommender is MORE diverse than the users' real clicks,
    negative means LESS.

    The effect size answers "how big is the gap?" separately from "is it
    significant?": with ~50 000 users the t-test flags almost any non-zero gap, so
    Cohen's d_z = mean(d) / sd(d) is what tells you whether the gap is
    *practically* meaningful (|d_z| < 0.2 negligible, < 0.5 small, < 0.8 medium,
    else large).
    """
    keys = sorted(rec_vals.keys() & gt_vals.keys())
    rec = np.array([rec_vals[k] for k in keys], dtype=float)
    gt = np.array([gt_vals[k] for k in keys], dtype=float)
    d = rec - gt
    n = len(d)

    mean_diff = float(d.mean())
    sd_diff = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd_diff / math.sqrt(n) if n > 1 else float("nan")
    # 95% t-interval for the mean difference (used by the figure).
    tcrit = stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
    ci_lo, ci_hi = mean_diff - tcrit * se, mean_diff + tcrit * se

    t_stat, t_p = stats.ttest_rel(rec, gt)

    # Paired effect size (Cohen's d_z): the standardized magnitude of the gap.
    dz = mean_diff / sd_diff if sd_diff and not math.isnan(sd_diff) else float("nan")

    return {
        "n": n,
        "mean_recommender": float(rec.mean()),
        "mean_ground_truth": float(gt.mean()),
        "mean_difference": mean_diff,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "direction": "above" if mean_diff > 0 else "below",
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_p),
        "cohen_dz": float(dz),
        "effect_size_label": _cohen_d_label(dz),
    }


def stars(p):
    """Significance stars from a p-value."""
    if p is None or math.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# --------------------------------------------------------------------------- #
# The single figure
# --------------------------------------------------------------------------- #
def plot_paired_ttest(results, metric, label, stats_dir):
    """Forest plot of the mean difference (recommender - ground truth) with its
    95% CI, and a zero reference line. A CI that excludes 0 is a significant
    direction; colour marks above (green) / below (red) / n.s. (grey). Each row
    is annotated with the paired effect size (Cohen's d_z) and its size band, so
    the plot shows both *whether* and *how big* the gap is."""
    recs = [r for r in RECOMMENDERS if r in results]
    means = [results[r]["mean_difference"] for r in recs]
    los = [results[r]["ci95_low"] for r in recs]
    his = [results[r]["ci95_high"] for r in recs]
    y = np.arange(len(recs))

    fig, ax = plt.subplots(figsize=(8, 5))
    xmax = max(his)
    for yi, r, m, lo, hi in zip(y, recs, means, los, his):
        color = "#c0392b" if hi < 0 else ("#27ae60" if lo > 0 else "#7f8c8d")
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2)
        ax.plot(m, yi, "o", color=color, markersize=7)
        s = results[r]
        ax.annotate(f"$d_z$ = {s['cohen_dz']:+.3f} ({s['effect_size_label']})",
                    xy=(hi, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=color)
    ax.axvline(0, color="#2c3e50", linestyle="--", linewidth=1.5,
               label="no difference (d = 0)")
    # Headroom on the right so the effect-size labels aren't clipped.
    ax.set_xlim(right=xmax + 0.6 * (xmax - min(los)) + 1e-9)
    ax.set_yticks(y)
    ax.set_yticklabels([REC_LABELS[r] for r in recs])
    ax.set_xlabel("Mean difference  (recommender − ground truth)  ±95% CI")
    ax.set_title(f"Paired t-test: {label.lower()} vs. ground truth\n"
                 "(CI excluding 0 = significant; label = Cohen's $d_z$ effect size)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(stats_dir, f"fig_paired_ttest_{metric}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# One metric: pair, test, print, plot
# --------------------------------------------------------------------------- #
def _analyze_and_report(metric, label, gt_vals, rec_vals_by_rec, out_dir, stats_dir):
    """Run the paired t-test of each recommender against ground truth on one
    metric, print a short table, and write the one figure."""
    print(f"=== {label} ===")
    results = {r: paired_ttest(vals, gt_vals) for r, vals in rec_vals_by_rec.items()}
    recs = [r for r in RECOMMENDERS if r in results]

    hdr = (f"{'recommender':<12}{'n':>7}{'mean':>9}{'GT':>9}{'d_mean':>10}"
           f"{'dir':>7}{'t_p':>12}{'':>5}{'d_z':>8}{'size':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        s = results[r]
        print(f"{REC_LABELS[r]:<12}{s['n']:>7}{s['mean_recommender']:>9.4f}"
              f"{s['mean_ground_truth']:>9.4f}{s['mean_difference']:>+10.4f}"
              f"{s['direction']:>7}{s['t_pvalue']:>12.2e}{stars(s['t_pvalue']):>5}"
              f"{s['cohen_dz']:>+8.3f}{s['effect_size_label']:>12}")
    print()

    path = plot_paired_ttest(results, metric, label, stats_dir)
    print(f"  wrote {os.path.relpath(path, out_dir)}\n")


def _normalized_per_impression_scores(ctx, recs, embeddings, recommenders):
    """Per-impression normalized content diversity for ground truth + recommenders.

    Takes an already-built context (from ``build_context``) so the impressions
    and per-impression picks are parsed once and reused across every content
    space — the normalized metric cannot be rebuilt from the per-user files.
    Returns ``(gt_vals, {recommender_name: vals})`` as ``{impr_id: score}``
    dicts, restricted to the recommenders that have a prediction on disk. Every
    call uses the same seed, so a given impression's min/max (the normalisation
    denominator) is shared across recommenders and cancels in the
    recommender-vs-ground-truth paired difference.
    """
    by_name = {rec.name: rec for rec in recs}

    def scores_for(rec):
        return per_impression_normalized_content_diversity(
            ctx.impressions, rec.recommended_by_impr(ctx), embeddings,
            max_combinations=_NORMALIZED_MAX_COMBINATIONS, seed=ctx.seed,
        )

    gt_rec = by_name.get(GROUND_TRUTH)
    if gt_rec is None or not os.path.exists(gt_rec.raw_path(ctx)):
        return {}, {}
    gt_vals = scores_for(gt_rec)

    rec_vals = {}
    for name in recommenders:
        rec = by_name.get(name)
        if rec is not None and os.path.exists(rec.raw_path(ctx)):
            rec_vals[name] = scores_for(rec)
    return gt_vals, rec_vals


# --------------------------------------------------------------------------- #
# The content-embedding spaces to test (title + any text variants)
# --------------------------------------------------------------------------- #
def _content_spaces(cfg, in_dir):
    """``(suffix, label, lazy_loader)`` for each content space to run stats on.

    Mirrors ``diversity_module.__main__._content_spaces`` so the metric keys and
    figures line up with the pipeline's manifest: the primary space (empty
    suffix) plus, for word-average datasets (MIND / mind_news), one space per
    ``content_text_variants`` entry — e.g. the abstract (news.tsv column 4)
    alongside the title (column 3). Precomputed datasets (eb-nerd) contribute
    their primary contrastive space plus one space per ``content_embeddings``
    entry (xlmr / bert / docvec), each a ready-made article-embedding parquet, so
    the standard per-recommender forest plots can be produced in any of those
    spaces (``statistic_for_embeddings.py`` instead compares the spaces
    side-by-side in one grid). Loaders are lazy so each (large) embedding map is
    built at most once, only when it is scored.
    """
    cd_cfg = cfg.get("content_diversity")
    if not cd_cfg:
        return []

    def _load_word_average(text_col):
        cfg["prepare"].ensure_utils(in_dir)
        return load_news_embeddings(
            os.path.join(in_dir, *cfg["articles"]),
            os.path.join(in_dir, *cd_cfg["embedding"]),
            os.path.join(in_dir, *cd_cfg["word_dict"]),
            text_col=text_col,
        )

    if cd_cfg["kind"] == "word_average":
        spaces = [("", "title", lambda: _load_word_average(3))]  # column 3 = title
        for name, (text_col, label) in cfg.get("content_text_variants", {}).items():
            spaces.append((f"_{name}", label,
                           lambda tc=text_col: _load_word_average(tc)))
        return spaces
    if cd_cfg["kind"] == "precomputed":
        spaces = [("", "contrastive",
                   lambda: load_precomputed_embeddings(
                       os.path.join(in_dir, *cd_cfg["vectors"])))]
        for name, (vec_file, vec_col) in cfg.get("content_embeddings", {}).items():
            path = os.path.join(in_dir, vec_file)
            spaces.append((f"_{name}", name,
                           lambda p=path, c=vec_col: load_precomputed_embeddings(
                               p, vector_column=c)))
        return spaces
    return []


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(dataset="MIND", only_spaces=None):
    """Run the paired t-tests for ``dataset``.

    only_spaces : if given, an iterable of content-space labels (e.g.
    ``["abstract"]``) — only those content spaces are scored and topic diversity
    is skipped, so you can compute just the abstract variants without recomputing
    the title / topic figures you already have. ``None`` runs everything.
    """
    try:
        dataset = resolve_dataset(dataset)  # accept MIND / mind / folder name
    except ValueError as exc:
        raise SystemExit(str(exc))

    out_dir = output_dir(dataset)
    in_dir = input_dir(dataset)
    stats_dir = os.path.join(out_dir, "statistics")
    os.makedirs(stats_dir, exist_ok=True)

    cfg = DATASETS[dataset]

    # Which recommenders actually have a processed file present.
    available = [r for r in RECOMMENDERS if os.path.exists(_processed_path(out_dir, r))]
    gt_path = _processed_path(out_dir, GROUND_TRUTH)
    if not os.path.exists(gt_path):
        raise SystemExit(f"No ground-truth file at {gt_path}")
    print(f"Dataset {dataset}: testing {available} against ground truth\n")

    # Topic diversity: per user, no embeddings, paired by user. Skipped when the
    # caller asked for specific content spaces only.
    if only_spaces is None:
        gt_topic = per_user_topic_diversity(gt_path)
        rec_topic = {r: per_user_topic_diversity(_processed_path(out_dir, r)) for r in available}
        _analyze_and_report("topic_diversity", METRIC_LABELS["topic_diversity"],
                            gt_topic, rec_topic, out_dir, stats_dir)

    # Content diversity, once per content-embedding space (matching the pipeline:
    # the primary title space plus any text variants such as the abstract). Each
    # space yields two metrics — per-user ILD and per-impression normalized ILD —
    # with the same suffixed keys/figures the pipeline's manifest uses.
    spaces = _content_spaces(cfg, in_dir)
    if only_spaces is not None:
        wanted = set(only_spaces)
        spaces = [s for s in spaces if s[1] in wanted]
        missing = wanted - {s[1] for s in _content_spaces(cfg, in_dir)}
        if missing:
            raise SystemExit(
                f"Unknown content space(s) {sorted(missing)} for '{dataset}'. "
                f"Available: {[s[1] for s in _content_spaces(cfg, in_dir)]}"
            )
    if not spaces:
        return

    # The impressions + per-impression picks (for the normalized metric) are the
    # same across spaces, so build the context once and reuse it.
    _, ctx, recs = build_context(dataset)

    for suffix, space_label, load in spaces:
        embeddings = load()

        # Per-user content diversity (ILD), paired by user.
        cd_key = f"content_diversity{suffix}"
        gt_vals = per_user_content_diversity(gt_path, embeddings)
        rec_vals = {r: per_user_content_diversity(_processed_path(out_dir, r), embeddings)
                    for r in available}
        _analyze_and_report(cd_key, _metric_label("content_diversity", space_label, suffix),
                            gt_vals, rec_vals, out_dir, stats_dir)

        # Per-impression normalized content diversity, paired by impression. This
        # is the slow one — per-impression min/max sampling.
        norm_key = f"content_diversity_normalized{suffix}"
        norm_label = _metric_label("content_diversity_normalized", space_label, suffix)
        print(f"(computing {norm_label.lower()} - per-impression, slower...)")
        gt_norm, rec_norm = _normalized_per_impression_scores(ctx, recs, embeddings, available)
        if gt_norm and rec_norm:
            _analyze_and_report(norm_key, norm_label, gt_norm, rec_norm, out_dir, stats_dir)
        else:
            print(f"  no scorable impressions for {norm_label.lower()}; skipping.\n")

        del embeddings  # free before loading the next space


if __name__ == "__main__":
    # Usage: python statistic.py [DATASET] [SPACE ...]
    #   python statistic.py MIND            -> everything (topic + all content spaces)
    #   python statistic.py MIND abstract   -> only the abstract content space
    #                                          (per-user + normalized), no topic/title
    dataset = sys.argv[1] if len(sys.argv) > 1 else "MIND"
    only_spaces = sys.argv[2:] or None
    run(dataset, only_spaces=only_spaces)