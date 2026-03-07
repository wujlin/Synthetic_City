"""
论文全部主图的统一绘图脚本。

用法：
    python Essay/plot_all_figures.py

输出：
    Essay/figures/legacy/fig1_heterogeneity.pdf
    Essay/figures/fig_06_main_results.pdf
    Essay/figures/fig_05_ablation.pdf
    Essay/figures/fig_s01_scaling.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

import matplotlib.pyplot as plt
from plot_style import (
    OKABE_ITO,
    paper_style,
    add_panel_label,
    save_figure,
    despine,
)

OUT_DIR = REPO / "Essay" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_DIR = OUT_DIR / "legacy"
LEGACY_DIR.mkdir(parents=True, exist_ok=True)

C_DIFF = OKABE_ITO["blue"]
C_IPF = OKABE_ITO["vermillion"]
C_INDEP = OKABE_ITO["gray"]
C_PAIR = OKABE_ITO["bluish_green"]
C_ACCENT = OKABE_ITO["orange"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ======================================================================
# Data loading
# ======================================================================

def load_heterogeneity() -> dict:
    p = REPO / "outputs" / "_tmp_puma5var_us_smoke" / "heterogeneity_diagnostic.json"
    return _load_json(p)


def load_ablation_k32() -> dict:
    p = REPO / "outputs" / "_us_puma_5var_k32_leaveMI_20260220T172356Z" / "metrics" / "ablation_summary.json"
    return _load_json(p)


def load_ablation_k128() -> dict:
    p = REPO / "outputs" / "_us_puma_5var_k128_leaveMI_20260220T173653Z" / "metrics" / "ablation_summary.json"
    return _load_json(p)


def load_mi_kfold() -> dict:
    p = REPO / "outputs" / "_us_puma_b19037_mikfold_fixscale_20260220T094927Z" / "metrics" / "mi_kfold_significance.json"
    return _load_json(p)


def load_ablation_csv() -> list[dict]:
    import csv
    p = REPO / "outputs" / "_us_puma_ablation_summary_20260220T121355Z" / "ablation_long.csv"
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ======================================================================
# Figure 1: Copula heterogeneity
# ======================================================================

def _load_choropleth_data():
    """Load PUMA boundaries + TVD data for choropleth panel."""
    import zipfile
    from urllib.request import urlretrieve
    import pandas as pd
    import geopandas as gpd

    CACHE_DIR = REPO / "data" / "geo_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CB_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_puma20_500k.zip"
    CB_ZIP = CACHE_DIR / "cb_2020_us_puma20_500k.zip"
    CB_DIR = CACHE_DIR / "cb_2020_us_puma20_500k"

    if not CB_ZIP.exists():
        print(f"  Downloading PUMA boundaries...")
        urlretrieve(CB_URL, CB_ZIP)
    if not CB_DIR.exists():
        CB_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(CB_ZIP, "r") as zf:
            zf.extractall(CB_DIR)

    gdf = gpd.read_file(list(CB_DIR.glob("*.shp"))[0])
    data = load_heterogeneity()
    tvd_df = pd.DataFrame(data["by_puma"])
    tvd_df["GEOID20"] = tvd_df["statefp"].str.zfill(2) + tvd_df["puma"].str.zfill(5)

    if "GEOID20" not in gdf.columns:
        gdf["GEOID20"] = gdf["STATEFP20"] + gdf["PUMACE20"]
    gdf = gdf.merge(tvd_df[["GEOID20", "tvd_to_global"]], on="GEOID20", how="left")

    TERRITORIES = {"60", "66", "69", "72", "78"}
    gdf = gdf[~gdf["STATEFP20"].isin(TERRITORIES)]
    return gdf


def fig1_heterogeneity():
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import Normalize

    data = load_heterogeneity()
    tvds = [r["tvd_to_global"] for r in data["by_puma"]]
    mean_tvd = data["mean_tvd_to_global"]
    p90 = data["p90_tvd_to_global"]

    mi_p = REPO / "outputs" / "_tmp_puma5var_mi_smoke" / "heterogeneity_diagnostic.json"
    mi_data = _load_json(mi_p) if mi_p.exists() else None
    mi_tvds = [r["tvd_to_global"] for r in mi_data["by_puma"]] if mi_data else []

    gdf = _load_choropleth_data()

    vmin, vmax = 0.05, 0.40
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.YlOrRd
    plot_kw = dict(
        column="tvd_to_global", cmap=cmap, norm=norm,
        linewidth=0.05, edgecolor="0.6",
        missing_kwds={"color": "white", "edgecolor": "0.85", "linewidth": 0.05},
    )

    with paper_style():
        fig = plt.figure(figsize=(7.0, 7.5))
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1.1, 1],
                      hspace=0.30, wspace=0.35,
                      left=0.08, right=0.92, top=0.98, bottom=0.06)

        # --- (a) Choropleth: full top row ---
        ax_map = fig.add_subplot(gs[0, :])
        conus = gdf[(gdf["STATEFP20"] != "02") & (gdf["STATEFP20"] != "15")]
        alaska = gdf[gdf["STATEFP20"] == "02"]
        hawaii = gdf[gdf["STATEFP20"] == "15"]

        conus.plot(ax=ax_map, **plot_kw)
        ax_map.set_xlim(-128, -65)
        ax_map.set_ylim(23, 52)
        ax_map.axis("off")
        add_panel_label(ax_map, "a", dx=-20, dy=0)

        # Alaska inset
        ax_ak = fig.add_axes([0.02, 0.46, 0.16, 0.18])
        if not alaska.empty:
            alaska.plot(ax=ax_ak, **plot_kw)
            ax_ak.set_xlim(-180, -130)
            ax_ak.set_ylim(51, 72)
        ax_ak.axis("off")
        ax_ak.set_title("AK", fontsize=7, pad=1)

        # Hawaii inset
        ax_hi = fig.add_axes([0.17, 0.47, 0.13, 0.11])
        if not hawaii.empty:
            hawaii.plot(ax=ax_hi, **plot_kw)
            ax_hi.set_xlim(-160.5, -154.5)
            ax_hi.set_ylim(18.8, 22.5)
        ax_hi.axis("off")
        ax_hi.set_title("HI", fontsize=7, pad=1)

        # Horizontal colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.55, 0.50, 0.30, 0.015])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("TVD to national-average copula", fontsize=8, labelpad=3)
        cb.ax.tick_params(labelsize=7)

        # --- (b) Histogram ---
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(tvds, bins=40, color=C_DIFF, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(mean_tvd, color=C_IPF, linestyle="--", linewidth=1.8, label=f"Mean = {mean_tvd:.3f}")
        ax.axvline(p90, color=C_ACCENT, linestyle=":", linewidth=1.5, alpha=0.8, label=f"P90 = {p90:.3f}")
        ax.set_xlabel("TVD to global average")
        ax.set_ylabel("Number of PUMAs")
        ax.legend(frameon=False, fontsize=7)
        despine(ax)
        add_panel_label(ax, "b")

        # --- (c) Rank scatter ---
        ax = fig.add_subplot(gs[1, 1])
        ax.scatter(range(len(tvds)), sorted(tvds), s=3, alpha=0.4, color=C_DIFF, rasterized=True)
        ax.axhline(mean_tvd, color=C_IPF, linestyle="--", linewidth=1.5)
        ax.set_xlabel("PUMA rank (sorted by TVD)")
        ax.set_ylabel("TVD to global average")
        despine(ax)
        add_panel_label(ax, "c")

        # --- (d) Michigan ---
        ax = fig.add_subplot(gs[1, 2])
        if mi_tvds:
            ax.hist(mi_tvds, bins=20, color=C_PAIR, alpha=0.75, edgecolor="white", linewidth=0.5)
            mi_mean = np.mean(mi_tvds)
            ax.axvline(mi_mean, color=C_IPF, linestyle="--", linewidth=1.8,
                       label=f"MI mean = {mi_mean:.3f}")
            ax.set_xlabel("TVD to global average")
            ax.set_ylabel("Number of MI PUMAs")
            ax.legend(frameon=False, fontsize=7)
        else:
            ax.text(0.5, 0.5, "MI data\nnot available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color=C_INDEP)
        despine(ax)
        add_panel_label(ax, "d")

        save_figure(fig, LEGACY_DIR / "fig1_heterogeneity.pdf")
        fig.savefig(LEGACY_DIR / "fig1_heterogeneity.png", dpi=200)
        print(f"  → {LEGACY_DIR / 'fig1_heterogeneity.pdf'}")
    plt.close(fig)


# ======================================================================
# Figure 2: Main results
# ======================================================================

def fig2_main_results():
    kfold = load_mi_kfold()

    configs = ["5v K=32", "5v K=128", "3v K=256", "3v K=64", "2v K=64"]
    diff_tvd = [0.024, 0.051, 0.084, 0.055, 0.069]
    ipf_tvd  = [0.058, 0.081, 0.103, 0.070, 0.074]
    ks       = [32, 128, 256, 64, 64]

    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0))
        fig.subplots_adjust(hspace=0.42, wspace=0.35)

        # (a) Scatter: diffusion vs IPF per config — each config gets distinct color + legend
        ax = axes[0, 0]
        cfg_colors = [C_DIFF, C_IPF, C_PAIR, C_ACCENT, C_INDEP]
        cfg_markers = ["o", "s", "^", "D", "v"]
        lim = [0, max(max(ipf_tvd), max(diff_tvd)) * 1.15]
        ax.plot(lim, lim, color=C_INDEP, linestyle="--", linewidth=1.0, alpha=0.6)
        for i, cfg in enumerate(configs):
            ax.scatter(ipf_tvd[i], diff_tvd[i], s=60, color=cfg_colors[i],
                       marker=cfg_markers[i], zorder=5, edgecolors="white",
                       linewidth=0.8, label=cfg)
        ax.set_xlabel("IPF TVD")
        ax.set_ylabel("Diffusion TVD")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.legend(frameon=False, fontsize=6.5,
                  loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
        despine(ax)
        add_panel_label(ax, "a")

        # (b) 5-fold delta
        ax = axes[0, 1]
        folds = kfold["folds"]
        fold_ids = [f["fold"].replace("mi_fold_", "Fold ") for f in folds]
        deltas = [f["diff_minus_ipf"] for f in folds]
        colors = [C_DIFF if d < 0 else C_IPF for d in deltas]
        bars = ax.barh(fold_ids, deltas, color=colors, height=0.6, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color=C_INDEP, linewidth=0.8)
        mean_d = np.mean(deltas)
        ax.axvline(mean_d, color=C_DIFF, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.annotate(f"Mean Δ = {mean_d:.4f}", xy=(mean_d, 0),
                    xytext=(0.02, 0.96), textcoords="axes fraction",
                    fontsize=7, color=C_DIFF, ha="left", va="top")
        ax.set_xlabel("Δ TVD (Diffusion − IPF)")
        ax.invert_yaxis()
        despine(ax)
        add_panel_label(ax, "b")

        # (c) Relative gain vs K
        ax = axes[1, 0]
        k_vals = [32, 64, 64, 128, 256, 512]
        rel_gain = [-59, -8, -22, -37, -19, 25]
        labels_k = ["5v", "2v", "3v", "5v", "3v", "5v"]
        colors_k = [C_DIFF if g < 0 else C_IPF for g in rel_gain]
        ax.bar(range(len(k_vals)), rel_gain, color=colors_k, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(k_vals)))
        ax.set_xticklabels([f"{k}({l})" for k, l in zip(k_vals, labels_k)], fontsize=7)
        ax.axhline(0, color=C_INDEP, linewidth=0.8)
        ax.set_xlabel("K (variables)", fontsize=9)
        ax.set_ylabel("Relative gain (%)")
        despine(ax)
        add_panel_label(ax, "c")

        # (d) Paired TVD comparison
        ax = axes[1, 1]
        short_labels = ["32(5v)", "128(5v)", "256(3v)", "64(3v)", "64(2v)"]
        x_pos = np.arange(len(configs))
        w = 0.35
        ax.bar(x_pos - w/2, diff_tvd, w, color=C_DIFF, label="Diffusion", edgecolor="white", linewidth=0.5)
        ax.bar(x_pos + w/2, ipf_tvd, w, color=C_IPF, label="IPF", edgecolor="white", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_labels, fontsize=7)
        ax.set_xlabel("K (variables)", fontsize=9)
        ax.set_ylabel("TVD")
        ax.legend(frameon=False, fontsize=8)
        despine(ax)
        add_panel_label(ax, "d")

        save_figure(fig, OUT_DIR / "fig_06_main_results.pdf")
        fig.savefig(OUT_DIR / "fig_06_main_results.png", dpi=200)
        print(f"  → {OUT_DIR / 'fig_06_main_results.pdf'}")
    plt.close(fig)


# ======================================================================
# Figure 3: Condition ablation
# ======================================================================

def fig3_ablation():
    abl32 = load_ablation_k32()
    abl128 = load_ablation_k128()

    def _get_tvd(abl, cond):
        return abl["conditions"][cond]["tvd_joint"]["mean"]

    conds = ["none", "marginal", "pairwise", "marginal_pairwise"]
    labels = ["none", "marginal", "pairwise", "marg+pair"]

    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.4))
        fig.subplots_adjust(hspace=0.55, wspace=0.35)

        # (a) K=32 ablation
        ax = axes[0, 0]
        tvds_32 = [_get_tvd(abl32, c) for c in conds]
        ipf_32 = abl32["baselines"]["ipf_train_seed"]["tvd_joint"]["mean"]
        indep_32 = abl32["baselines"]["independence"]["tvd_joint"]["mean"]

        bars = ax.bar(range(len(conds)), tvds_32, color=[C_INDEP, C_ACCENT, C_PAIR, C_DIFF],
                      edgecolor="white", linewidth=0.5)
        ax.axhline(ipf_32, color=C_IPF, linestyle="--", linewidth=1.5, label=f"IPF = {ipf_32:.3f}")
        ax.axhline(indep_32, color=C_INDEP, linestyle=":", linewidth=1.2, alpha=0.6, label=f"Indep = {indep_32:.3f}")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("TVD")
        ax.set_title("5-var, K=32", fontsize=10)
        ax.legend(frameon=False, fontsize=7,
                  loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        despine(ax)
        add_panel_label(ax, "a")

        # (b) K=128 ablation
        ax = axes[0, 1]
        tvds_128 = [_get_tvd(abl128, c) for c in conds]
        ipf_128 = abl128["baselines"]["ipf_train_seed"]["tvd_joint"]["mean"]
        indep_128 = abl128["baselines"]["independence"]["tvd_joint"]["mean"]

        bars = ax.bar(range(len(conds)), tvds_128, color=[C_INDEP, C_ACCENT, C_PAIR, C_DIFF],
                      edgecolor="white", linewidth=0.5)
        ax.axhline(ipf_128, color=C_IPF, linestyle="--", linewidth=1.5, label=f"IPF = {ipf_128:.3f}")
        ax.axhline(indep_128, color=C_INDEP, linestyle=":", linewidth=1.2, alpha=0.6, label=f"Indep = {indep_128:.3f}")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("TVD")
        ax.set_title("5-var, K=128", fontsize=10)
        ax.legend(frameon=False, fontsize=7,
                  loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        despine(ax)
        add_panel_label(ax, "b")

        # (c) Pairwise contribution vs D
        ax = axes[1, 0]
        d_vals = [2, 3, 5, 5]
        d_labels = ["D=2\nK=64", "D=3\nK=256", "D=5\nK=32", "D=5\nK=128"]
        marg_only = [0.069, 0.101, 0.043, 0.070]
        best_pair = [0.069, 0.084, 0.024, 0.051]
        pct_change = [0, -17, -44, -27]

        x_pos = np.arange(len(d_vals))
        w = 0.35
        ax.bar(x_pos - w/2, marg_only, w, color=C_ACCENT, label="Marginal", edgecolor="white", linewidth=0.5)
        ax.bar(x_pos + w/2, best_pair, w, color=C_PAIR, label="+Pair", edgecolor="white", linewidth=0.5)
        for i, pct in enumerate(pct_change):
            if pct != 0:
                ax.text(x_pos[i] + w/2, best_pair[i] + 0.003, f"{pct}%",
                        ha="center", fontsize=6, color=C_PAIR, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(d_labels, fontsize=7)
        ax.set_ylabel("TVD")
        ax.set_ylim(0, max(marg_only) * 1.25)
        ax.legend(frameon=False, fontsize=7,
                  loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        despine(ax)
        add_panel_label(ax, "c")

        # (d) Raw marginal TVD under pairwise (K=32)
        ax = axes[1, 1]
        vars_list = ["age", "sex", "income", "schl", "esr"]
        var_labels = ["Age", "Sex", "Income", "Edu.", "Empl."]
        raw_tvds = []
        for v in vars_list:
            key = f"tvd_{v}_raw"
            raw_tvds.append(abl32["conditions"]["pairwise"].get(key, {}).get("mean", 0.0))

        ax.bar(range(len(vars_list)), raw_tvds, color=C_PAIR, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(vars_list)))
        ax.set_xticklabels(var_labels, fontsize=9)
        ax.set_ylabel("Marginal TVD (raw)")
        ax.set_ylim(0, 0.003)
        ax.axhline(0.002, color=C_INDEP, linestyle=":", linewidth=1.0, alpha=0.5, label="0.002 threshold")
        ax.legend(frameon=False, fontsize=8)
        despine(ax)
        add_panel_label(ax, "d")

        save_figure(fig, OUT_DIR / "fig_05_ablation.pdf")
        fig.savefig(OUT_DIR / "fig_05_ablation.png", dpi=200)
        print(f"  → {OUT_DIR / 'fig_05_ablation.pdf'}")
    plt.close(fig)


# ======================================================================
# Figure 4: Scaling
# ======================================================================

def fig4_scaling():
    k_vals = np.array([32, 64, 64, 128, 256, 512])
    diff_tvds = np.array([0.024, 0.069, 0.055, 0.051, 0.084, 0.139])
    ipf_tvds  = np.array([0.058, 0.074, 0.070, 0.081, 0.103, 0.112])
    k_labels  = ["32\n(5v)", "64\n(2v)", "64\n(3v)", "128\n(5v)", "256\n(3v)", "512\n(5v)"]

    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0))
        fig.subplots_adjust(hspace=0.42, wspace=0.35)

        # (a) TVD vs K
        ax = axes[0, 0]
        ax.plot(range(len(k_vals)), diff_tvds, "o-", color=C_DIFF, label="Diffusion", markersize=6, linewidth=2.0)
        ax.plot(range(len(k_vals)), ipf_tvds, "s--", color=C_IPF, label="IPF", markersize=5, linewidth=1.8)
        ax.set_xticks(range(len(k_vals)))
        ax.set_xticklabels(k_labels, fontsize=7)
        ax.set_ylabel("TVD")
        ax.set_xlabel("Joint dimensionality K")
        ax.legend(frameon=False, fontsize=8)
        despine(ax)
        add_panel_label(ax, "a")

        # (b) K=512 training effect
        ax = axes[0, 1]
        epochs = [10000, 30000]
        tvd_512 = [0.182, 0.147]
        ax.bar([0], [tvd_512[0]], color=C_DIFF, edgecolor="white", linewidth=0.5, alpha=0.5)
        ax.bar([1], [tvd_512[1]], color=C_DIFF, edgecolor="white", linewidth=0.5, alpha=1.0)
        ax.axhline(0.112, color=C_IPF, linestyle="--", linewidth=1.8, label="IPF = 0.112")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["10k epochs", "30k epochs"], fontsize=9)
        ax.set_ylabel("TVD")
        ax.set_title("K=512, marginal+IPF", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        despine(ax)
        add_panel_label(ax, "b")

        # (c) Mean persons per cell
        ax = axes[1, 0]
        avg_pop = 7000.0
        k_range = np.array([32, 64, 128, 256, 512])
        persons_per_cell = avg_pop / k_range
        ax.plot(k_range, persons_per_cell, "o-", color=C_ACCENT, markersize=6, linewidth=2.0)
        ax.axhline(30, color=C_INDEP, linestyle=":", linewidth=1.2, alpha=0.6, label="~30 (noise floor)")
        ax.set_xlabel("K")
        ax.set_ylabel("Mean persons per cell")
        ax.set_xscale("log", base=2)
        ax.set_xticks(k_range)
        ax.set_xticklabels([str(k) for k in k_range])
        ax.legend(frameon=False, fontsize=8)
        despine(ax)
        add_panel_label(ax, "c")

        # (d) Constraint coverage: pairwise dims vs joint free params
        ax = axes[1, 1]
        bins_configs = {
            32:  (2, 2, 2, 2, 2),
            128: (4, 2, 4, 2, 2),
            512: (4, 2, 4, 4, 4),
        }
        k_plot = [32, 128, 512]
        free_params = [k - 1 for k in k_plot]
        pair_dims = []
        for k in k_plot:
            bs = bins_configs[k]
            pd_val = sum(bs[i] * bs[j] for i in range(len(bs)) for j in range(i+1, len(bs)))
            pair_dims.append(pd_val)

        x_pos = np.arange(len(k_plot))
        w = 0.35
        ax.bar(x_pos - w/2, free_params, w, color=C_IPF, label="Free parameters (K−1)", edgecolor="white", linewidth=0.5)
        ax.bar(x_pos + w/2, pair_dims, w, color=C_PAIR, label="Pairwise cond. dims", edgecolor="white", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"K={k}" for k in k_plot], fontsize=9)
        ax.set_ylabel("Dimensionality")
        ax.legend(frameon=False, fontsize=8)
        despine(ax)
        add_panel_label(ax, "d")

        save_figure(fig, OUT_DIR / "fig_s01_scaling.pdf")
        fig.savefig(OUT_DIR / "fig_s01_scaling.png", dpi=200)
        print(f"  → {OUT_DIR / 'fig_s01_scaling.pdf'}")
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print("Generating figures...")
    fig1_heterogeneity()
    fig2_main_results()
    fig3_ablation()
    fig4_scaling()
    print("Done.")
