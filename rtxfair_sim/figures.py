# revision_outputs/rtxfair_sim/figures.py
"""
High-resolution, large, readable scenario-response figures for the revision.
All figures: large fonts, 300 dpi, saved as PNG + PDF. These replace the
unreadable Figs 2-4.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 2.5,
    "lines.markersize": 8,
})

TEAL, RED, ORANGE, GRAY = "#008080", "#d1495b", "#edae49", "#5a5a5a"


def _save(fig, out_base):
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return out_base + ".png"


def response_vs_factor(agg, factor, y_mean, y_ci, out_base, title=None,
                       xlabel=None, ylabel=None, color=TEAL, logx=False):
    """Single response with 95% CI band vs an ordered factor."""
    d = agg.dropna(subset=[factor]).sort_values(factor)
    x = d[factor].values.astype(float)
    m = d[y_mean].values.astype(float)
    h = d[y_ci].values.astype(float)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, m, "-o", color=color)
    ax.fill_between(x, m - h, m + h, color=color, alpha=0.20, label="95% CI")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel or factor)
    ax.set_ylabel(ylabel or y_mean)
    ax.set_title(title or f"{y_mean} vs {factor}")
    ax.legend()
    return _save(fig, out_base)


def dual_response_vs_lambda(agg, out_base, dataset_pretty=""):
    """AUC (left) and DP gap (right) vs lambda_fair, each with CI band."""
    d = agg.dropna(subset=["lambda_fair"]).sort_values("lambda_fair")
    x = d["lambda_fair"].values.astype(float)
    fig, ax1 = plt.subplots(figsize=(10, 6.5))
    a_m, a_h = d["AUC_mean"].values, d["AUC_ci95"].values
    ax1.plot(x, a_m, "-o", color=TEAL, label="AUC")
    ax1.fill_between(x, a_m - a_h, a_m + a_h, color=TEAL, alpha=0.18)
    ax1.set_xlabel(r"Fairness pressure $\lambda_{fair}$")
    ax1.set_ylabel("AUC", color=TEAL)
    ax1.tick_params(axis="y", labelcolor=TEAL)
    ax2 = ax1.twinx()
    g_m, g_h = d["DP_gap_mean"].values, d["DP_gap_ci95"].values
    ax2.plot(x, g_m, "-s", color=RED, label="DP gap")
    ax2.fill_between(x, g_m - g_h, g_m + g_h, color=RED, alpha=0.18)
    ax2.set_ylabel("Demographic-parity gap", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.grid(False)
    ax1.set_title(f"Accuracy-fairness response to $\\lambda_{{fair}}$  {dataset_pretty}")
    return _save(fig, out_base)


def stability_vs_minority(agg, out_base, dataset_pretty=""):
    """Gradient-norm (max) and masked-batch fraction vs minority fraction."""
    d = agg.dropna(subset=["minority_fraction"]).sort_values("minority_fraction")
    x = d["minority_fraction"].values.astype(float) * 100
    fig, ax1 = plt.subplots(figsize=(10, 6.5))
    m, h = d["grad_norm_max_mean"].values, d["grad_norm_max_ci95"].values
    ax1.plot(x, m, "-o", color=ORANGE, label="max grad norm")
    ax1.fill_between(x, m - h, m + h, color=ORANGE, alpha=0.18)
    ax1.set_xlabel("Minority fraction (%)")
    ax1.set_ylabel("Max gradient norm", color=ORANGE)
    ax1.tick_params(axis="y", labelcolor=ORANGE)
    ax2 = ax1.twinx()
    mm = d["masked_batch_frac_mean"].values
    ax2.plot(x, mm, "-s", color=GRAY, label="masked-batch frac")
    ax2.set_ylabel("Fraction of safety-net-masked batches", color=GRAY)
    ax2.tick_params(axis="y", labelcolor=GRAY)
    ax2.grid(False)
    ax1.set_title(f"Training stability vs minority fraction  {dataset_pretty}")
    return _save(fig, out_base)


def pareto_auc_dpgap(agg, out_base, dataset_pretty=""):
    """Accuracy-fairness Pareto: AUC vs DP gap, annotated by lambda."""
    d = agg.dropna(subset=["lambda_fair"]).sort_values("lambda_fair")
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(d["DP_gap_mean"], d["AUC_mean"], c=d["lambda_fair"],
                    cmap="coolwarm", s=160, edgecolor="k", zorder=3)
    for _, r in d.iterrows():
        ax.annotate(f"$\\lambda$={r['lambda_fair']:g}",
                    (r["DP_gap_mean"], r["AUC_mean"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=12)
    ax.set_xlabel("Demographic-parity gap  (lower = fairer)")
    ax.set_ylabel("AUC  (higher = better)")
    ax.set_title(f"Accuracy-fairness Pareto front  {dataset_pretty}")
    fig.colorbar(sc, ax=ax, label=r"$\lambda_{fair}$")
    return _save(fig, out_base)
