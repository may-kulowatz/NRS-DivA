"""Statistical significance of recommender diversity vs. the ground truth (MIND).

WHAT THIS SCRIPT ANSWERS
------------------------
The dashboard / run-manifest stores a single aggregate diversity number per
recommender (e.g. NRMS topic diversity = 0.703) next to the ground-truth number
(0.804). Eyeballing "0.703 < 0.804" tells you NRMS *looks* less diverse than the
users' real clicks — but not whether that gap is **statistically significant** or
just sampling noise. This script answers, for every recommender and every
diversity metric:

    "Does this recommender's diversity sit significantly ABOVE or BELOW the
     ground truth, and how large is the effect?"

THE DATA WE TEST ON
-------------------
Each processed prediction file (``predictions_processed/``) holds ONE LINE PER
USER, and the SAME 50 000 user IDs appear in every recommender's file and in the
ground-truth file. That gives us *paired* observations: for user ``u`` we have
both the diversity of what the recommender served ``u`` and the diversity of
``u``'s real clicks. Pairing is the whole point — it removes between-user
variability (some users simply click broader topics than others) and asks the
sharper question: *for the same user*, does the recommender raise or lower
diversity relative to that user's own behavior?

We rebuild the *per-user* diversity values (not the dataset-level average) using
exactly the same definitions the pipeline uses, so the mean of our per-user
sample reproduces the manifest's aggregate number:

  * topic_diversity   : per user, (# unique topics) / (# topic assignments),
                        over users with >1 article and >=1 non-"none" topic.
                        1.0 = every article a different topic; low = repetitive.
  * content_diversity : per user, intra-list diversity (ILD) = 1 - mean pairwise
                        cosine similarity of the title embeddings of the user's
                        articles. Needs >=2 embeddable articles.

THE STATISTICS WE COMPUTE  (per recommender x metric)
-----------------------------------------------------
Let d_i = value_recommender(u_i) - value_ground_truth(u_i) over the users u_i
present (and scorable) in BOTH files. We summarize and test this difference
sample d:

  1. Means + mean difference (mean(d)). Its SIGN is the headline: d>0 the
     recommender is MORE diverse than ground truth, d<0 LESS diverse.
  2. 95% confidence interval for mean(d) (t-interval). If it excludes 0, the
     direction is significant at the 5% level.
  3. Paired t-test  (scipy.stats.ttest_rel): parametric, H0: mean(d)=0. Valid
     here because n is large (CLT makes the mean's sampling distribution normal
     regardless of d's shape).
  4. Wilcoxon signed-rank test (scipy.stats.wilcoxon): non-parametric companion,
     H0: the differences are symmetric about 0. Reported because per-user
     diversity is bounded/skewed, not normal; if both tests agree we can trust
     the conclusion isn't an artefact of the normality assumption.
  5. Cohen's d_z (paired effect size) = mean(d) / std(d). Significance with
     50 000 users is almost guaranteed for any non-zero gap, so the effect size
     tells us whether the gap is *practically* meaningful:
        |d_z| < 0.2 negligible, <0.5 small, <0.8 medium, else large.
  6. Shapiro-Wilk normality flag on a subsample of d (purely diagnostic: it
     justifies leaning on Wilcoxon when the differences aren't normal).

MULTIPLE COMPARISONS
--------------------
We run many tests (recommenders x metrics), so a raw p<0.05 would over-fire by
chance. We apply Holm-Bonferroni correction across all tests of a metric family
and report the adjusted p-values; significance stars use the adjusted values.

OUTPUTS  (written to data_processed/<dataset>/statistics/)
----------------------------------------------------------
  * stat_tests.csv / stat_tests.json : the full results table.
  * fig_means_vs_ground_truth.png    : mean per-user diversity per recommender
                                       with 95% CI error bars, the ground-truth
                                       reference line, and significance stars.
  * fig_distributions_box.png        : per-user distribution (box + jittered
                                       points) per recommender vs. ground truth.
  * fig_paired_differences.png       : distribution of d = recommender - GT per
                                       recommender, with the 0 reference line.
  * fig_effect_sizes.png             : forest plot of mean(d) with 95% CI.

Run:  python statistic.py [DATASET]      (DATASET defaults to "MIND")
"""

import os
import sys
import json
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")  # file output only, no interactive display
import matplotlib.pyplot as plt
from scipy import stats

# Reuse the project's parsing + diversity definitions so our per-user values
# match the pipeline's aggregates exactly.
from config import DATASETS, input_dir, output_dir
from diversity_module.topic_diversity import _parse_user_articles
from diversity_module.content_diversity import load_news_embeddings, _ild


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
}


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
# The statistical comparison of one recommender against ground truth
# --------------------------------------------------------------------------- #
def _cohen_d_label(dz):
    """Conventional plain-language size band for a paired Cohen's d_z."""
    a = abs(dz)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def paired_comparison(rec_vals, gt_vals):
    """Compare a recommender's per-user values to ground truth on shared users.

    rec_vals, gt_vals : {user_id: value} dicts.

    Returns a dict of summary statistics + test results. ``d`` is the paired
    difference (recommender - ground_truth); a positive mean means the
    recommender is MORE diverse than the users' real clicks, negative means LESS.
    """
    users = sorted(rec_vals.keys() & gt_vals.keys())
    rec = np.array([rec_vals[u] for u in users], dtype=float)
    gt = np.array([gt_vals[u] for u in users], dtype=float)
    d = rec - gt
    n = len(d)

    mean_diff = float(d.mean())
    sd_diff = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd_diff / math.sqrt(n) if n > 1 else float("nan")
    # 95% t-interval for the mean difference.
    tcrit = stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
    ci_lo, ci_hi = mean_diff - tcrit * se, mean_diff + tcrit * se

    # Paired parametric test.
    t_stat, t_p = stats.ttest_rel(rec, gt)

    # Non-parametric companion. Wilcoxon errors if every difference is zero;
    # guard so a degenerate metric doesn't crash the whole run.
    try:
        w_stat, w_p = stats.wilcoxon(rec, gt)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    # Paired effect size (Cohen's d_z).
    dz = mean_diff / sd_diff if sd_diff and not math.isnan(sd_diff) else float("nan")

    # Normality diagnostic on the differences (Shapiro caps at 5000 samples).
    if n >= 3:
        sample = d if n <= 5000 else np.random.default_rng(42).choice(d, 5000, replace=False)
        _, shapiro_p = stats.shapiro(sample)
        diffs_normal = bool(shapiro_p > 0.05)
    else:
        shapiro_p, diffs_normal = float("nan"), False

    return {
        "n_users": n,
        "mean_recommender": float(rec.mean()),
        "mean_ground_truth": float(gt.mean()),
        "mean_difference": mean_diff,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "direction": "above" if mean_diff > 0 else "below",
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_p),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
        "cohen_dz": float(dz),
        "effect_size_label": _cohen_d_label(dz),
        "shapiro_pvalue": float(shapiro_p),
        "differences_normal": diffs_normal,
        # carried for plotting, not serialised to the table
        "_d": d,
        "_rec": rec,
        "_gt": gt,
    }


def holm_bonferroni(pvalues):
    """Holm-Bonferroni step-down adjusted p-values for a list of raw p-values.

    Controls the family-wise error rate without assuming independence. Returns a
    list aligned with the input order; adjusted values are clipped to <=1 and
    kept monotonic in the sorted order (the standard Holm enforcement).
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvalues[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def stars(p):
    """Significance stars from a (corrected) p-value."""
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
# Plots
# --------------------------------------------------------------------------- #
def plot_means_vs_ground_truth(results, samples, metric, gt_mean, stats_dir):
    """Bar of each recommender's mean per-user diversity (95% CI error bars),
    with the ground-truth value as a horizontal reference line and significance
    stars (Holm-adjusted Wilcoxon p) above each bar."""
    recs = [r for r in RECOMMENDERS if r in results]
    means = [samples[r]["mean_recommender"] for r in recs]
    # Half-width of each recommender's own per-user 95% CI (for the error bar).
    errs = []
    for r in recs:
        s = samples[r]
        vals = s["_rec"]
        n = len(vals)
        se = vals.std(ddof=1) / math.sqrt(n)
        errs.append(stats.t.ppf(0.975, df=n - 1) * se)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(recs))
    colors = ["#c0392b" if m < gt_mean else "#27ae60" for m in means]
    bars = ax.bar(x, means, yerr=errs, capsize=5, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.6)
    ax.axhline(gt_mean, color="#2c3e50", linestyle="--", linewidth=1.5,
               label=f"Ground truth = {gt_mean:.3f}")

    top = max(m + e for m, e in zip(means, errs))
    for xi, r, m, e in zip(x, recs, means, errs):
        ax.text(xi, m + e + top * 0.01, stars(results[r]["wilcoxon_p_adj"]),
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([REC_LABELS[r] for r in recs])
    ax.set_ylabel(f"Mean per-user {METRIC_LABELS[metric].lower()}")
    ax.set_title(f"{METRIC_LABELS[metric]} vs. ground truth\n"
                 "(green = above GT, red = below; stars = Holm-adj. Wilcoxon)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(stats_dir, f"fig_means_vs_ground_truth_{metric}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_distributions_box(samples, gt_sample, metric, stats_dir):
    """Per-user distribution (box plot) for each recommender and ground truth,
    so the spread/skew behind the means is visible."""
    recs = [r for r in RECOMMENDERS if r in samples] + [GROUND_TRUTH]
    data = []
    for r in recs:
        data.append(samples[r]["_rec"] if r != GROUND_TRUTH else gt_sample)

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black"))
    palette = plt.cm.tab10(np.linspace(0, 1, len(recs)))
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticklabels([REC_LABELS[r] for r in recs], rotation=15)
    ax.set_ylabel(f"Per-user {METRIC_LABELS[metric].lower()}")
    ax.set_title(f"Per-user {METRIC_LABELS[metric].lower()} distributions")
    fig.tight_layout()
    path = os.path.join(stats_dir, f"fig_distributions_box_{metric}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_paired_differences(samples, metric, stats_dir):
    """Distribution of the paired difference d = recommender - ground_truth per
    recommender (violin), with the d=0 line. Mass left of 0 = below GT."""
    recs = [r for r in RECOMMENDERS if r in samples]
    data = [samples[r]["_d"] for r in recs]

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc, c in zip(parts["bodies"], plt.cm.tab10(np.linspace(0, 1, len(recs)))):
        pc.set_facecolor(c)
        pc.set_alpha(0.6)
    ax.axhline(0, color="#2c3e50", linestyle="--", linewidth=1.5,
               label="no difference (d = 0)")
    ax.set_xticks(np.arange(1, len(recs) + 1))
    ax.set_xticklabels([REC_LABELS[r] for r in recs])
    ax.set_ylabel(f"Per-user difference  (recommender − ground truth)")
    ax.set_title(f"Paired differences in {METRIC_LABELS[metric].lower()}\n"
                 "(below the dashed line = less diverse than real clicks)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(stats_dir, f"fig_paired_differences_{metric}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_effect_sizes(results, samples, metric, stats_dir):
    """Forest plot: mean difference (recommender - GT) with its 95% CI per
    recommender. CIs crossing 0 = no significant direction."""
    recs = [r for r in RECOMMENDERS if r in results]
    means = [samples[r]["mean_difference"] for r in recs]
    los = [samples[r]["ci95_low"] for r in recs]
    his = [samples[r]["ci95_high"] for r in recs]
    y = np.arange(len(recs))

    fig, ax = plt.subplots(figsize=(8, 5))
    for yi, m, lo, hi in zip(y, means, los, his):
        color = "#c0392b" if hi < 0 else ("#27ae60" if lo > 0 else "#7f8c8d")
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2)
        ax.plot(m, yi, "o", color=color, markersize=7)
    ax.axvline(0, color="#2c3e50", linestyle="--", linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels([REC_LABELS[r] for r in recs])
    ax.set_xlabel("Mean difference  (recommender − ground truth)  with 95% CI")
    ax.set_title(f"Effect of each recommender on {METRIC_LABELS[metric].lower()}")
    fig.tight_layout()
    path = os.path.join(stats_dir, f"fig_effect_sizes_{metric}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(dataset="MIND"):
    if dataset not in DATASETS:
        raise SystemExit(f"Unknown dataset {dataset!r}; choose from {list(DATASETS)}")

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

    # Load title embeddings once for content diversity (word-average datasets).
    cd_cfg = cfg.get("content_diversity")
    news_embeddings = None
    if cd_cfg and cd_cfg["kind"] == "word_average":
        cfg["prepare"].ensure_utils(in_dir)
        news_embeddings = load_news_embeddings(
            os.path.join(in_dir, *cfg["articles"]),
            os.path.join(in_dir, *cd_cfg["embedding"]),
            os.path.join(in_dir, *cd_cfg["word_dict"]),
        )

    # (metric_key, per-user extractor) pairs we can compute for this dataset.
    metric_defs = [("topic_diversity", lambda p: per_user_topic_diversity(p))]
    if news_embeddings is not None:
        metric_defs.append(
            ("content_diversity", lambda p: per_user_content_diversity(p, news_embeddings))
        )

    all_rows = []  # flat result table for CSV/JSON

    for metric, extractor in metric_defs:
        print(f"=== {METRIC_LABELS[metric]} ===")
        gt_vals = extractor(gt_path)
        gt_mean = float(np.mean(list(gt_vals.values())))

        results, samples = {}, {}
        for r in available:
            rec_vals = extractor(_processed_path(out_dir, r))
            res = paired_comparison(rec_vals, gt_vals)
            results[r] = res
            samples[r] = res

        # Holm-Bonferroni correction across recommenders, per test family.
        recs = list(results)
        t_adj = holm_bonferroni([results[r]["t_pvalue"] for r in recs])
        w_adj = holm_bonferroni([results[r]["wilcoxon_pvalue"] for r in recs])
        for r, ta, wa in zip(recs, t_adj, w_adj):
            results[r]["t_p_adj"] = ta
            results[r]["wilcoxon_p_adj"] = wa

        # Console summary.
        hdr = (f"{'recommender':<12}{'n':>7}{'mean':>9}{'GT':>9}{'d_mean':>10}"
               f"{'dir':>7}{'t_p_adj':>11}{'wil_p_adj':>11}{'d_z':>8}{'size':>11}{'':>5}")
        print(hdr)
        print("-" * len(hdr))
        for r in recs:
            s = results[r]
            print(f"{REC_LABELS[r]:<12}{s['n_users']:>7}{s['mean_recommender']:>9.4f}"
                  f"{s['mean_ground_truth']:>9.4f}{s['mean_difference']:>+10.4f}"
                  f"{s['direction']:>7}{s['t_p_adj']:>11.2e}{s['wilcoxon_p_adj']:>11.2e}"
                  f"{s['cohen_dz']:>+8.3f}{s['effect_size_label']:>11}"
                  f"{stars(s['wilcoxon_p_adj']):>5}")
            row = {k: v for k, v in s.items() if not k.startswith("_")}
            row.update({"dataset": dataset, "metric": metric, "recommender": r})
            all_rows.append(row)
        print()

        # Figures for this metric.
        gt_sample = np.array(list(gt_vals.values()), dtype=float)
        for fn, args in [
            (plot_means_vs_ground_truth, (results, samples, metric, gt_mean, stats_dir)),
            (plot_distributions_box, (samples, gt_sample, metric, stats_dir)),
            (plot_paired_differences, (samples, metric, stats_dir)),
            (plot_effect_sizes, (results, samples, metric, stats_dir)),
        ]:
            path = fn(*args)
            print(f"  wrote {os.path.relpath(path, out_dir)}")
        print()

    # Persist the full table.
    csv_path = os.path.join(stats_dir, "stat_tests.csv")
    json_path = os.path.join(stats_dir, "stat_tests.json")
    if all_rows:
        columns = list(all_rows[0].keys())
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(columns) + "\n")
            for row in all_rows:
                f.write(",".join(str(row[c]) for c in columns) + "\n")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)
        print(f"Wrote results table -> {os.path.relpath(csv_path, out_dir)} (+ .json)")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "MIND")