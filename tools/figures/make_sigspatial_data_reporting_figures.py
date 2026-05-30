#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import PercentFormatter
from shapely import affinity

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthpop.plot_style import OKABE_ITO, PaperStyle, add_panel_label, despine, paper_style, save_figure


VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop", "EARN_16p_bin"]
VARIABLE_TITLES = {
    "AGEP_bin": "Age",
    "SEX": "Gender",
    "SCHL_allpop": "Education",
    "ESR_allpop": "Employment",
    "EARN_16p_bin": "Income",
}
VARIABLE_CATEGORIES = {
    "AGEP_bin": [
        "0--4",
        "5--17",
        "18--24",
        "25--34",
        "35--44",
        "45--54",
        "55--64",
        "65--74",
        "75--84",
        "85+",
    ],
    "SEX": ["Male", "Female"],
    "SCHL_allpop": [
        "Age <25",
        "Less than high school",
        "High school/GED",
        "Some college",
        "Bachelor+",
    ],
    "ESR_allpop": [
        "Age <16",
        "Employed",
        "Unemployed",
        "Armed forces",
        "Not in labor force",
    ],
    "EARN_16p_bin": [
        "No earnings or <16",
        "<$25k",
        "$25k--$50k",
        "$50k--$75k",
        "$75k--$100k",
        "$100k+",
    ],
}
SHAPE = (10, 2, 5, 5, 6)
TVD_ORANGE_CMAP = LinearSegmentedColormap.from_list(
    "tvd_orange_white",
    ["#f2d6a2", "#f4a261", "#e76f51", "#c44536", "#7f000d"],
)


def _load_joint(npz_path: Path, puma_qc_csv: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    z = np.load(npz_path, allow_pickle=True)
    p_true = np.asarray(z["p_true"], dtype=float)
    pred = np.asarray(z["predicted"], dtype=float)
    keys = pd.DataFrame(
        {
            "puma_uid": pd.Series(z["puma_uid"].astype(str)).str.zfill(7),
            "statefp": pd.Series(z["statefp"].astype(str)).str.zfill(2),
        }
    )
    qc = pd.read_csv(puma_qc_csv, dtype={"puma_uid": str, "statefp": str})
    qc["puma_uid"] = qc["puma_uid"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)
    qc["statefp"] = qc["statefp"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    qc["target_persons"] = pd.to_numeric(qc["target_persons"], errors="coerce").fillna(0.0)
    keys = keys.merge(qc[["puma_uid", "target_persons"]], on="puma_uid", how="left")
    if keys["target_persons"].isna().any():
        missing = keys.loc[keys["target_persons"].isna(), "puma_uid"].head().tolist()
        raise ValueError(f"missing target_persons for PUMAs: {missing}")
    return p_true, pred, keys


def _plot_puma_tvd_map(
    *,
    tvd: pd.DataFrame,
    dist_tvd: np.ndarray | None = None,
    puma_shp: Path,
    out_pdf: Path,
    out_png: Path,
) -> None:
    gdf = gpd.read_file(puma_shp)
    gdf["puma_uid"] = gdf["GEOID20"].astype(str).str.zfill(7)
    gdf["statefp"] = gdf["STATEFP20"].astype(str).str.zfill(2)
    gdf = gdf.merge(tvd, on="puma_uid", how="inner")
    if "statefp_x" in gdf.columns:
        gdf["statefp"] = gdf["statefp_x"].astype(str).str.zfill(2)
    gdf = gdf.to_crs("EPSG:5070")

    conus = gdf[~gdf["statefp"].isin(["02", "15"])].copy()
    alaska = gdf[gdf["statefp"].eq("02")].copy()
    hawaii = gdf[gdf["statefp"].eq("15")].copy()

    q01, q99 = np.nanpercentile(gdf["tvd"], [1, 99])
    norm = Normalize(vmin=float(q01), vmax=float(q99))
    cmap = TVD_ORANGE_CMAP

    style = PaperStyle(font_size=8.5, axes_labelsize=9.0, axes_titlesize=10.0, tick_labelsize=8.0, legend_fontsize=8.0)
    with paper_style(style):
        fig = plt.figure(figsize=(7.2, 5.6))
        ax = fig.add_axes([0.02, 0.34, 0.94, 0.60])
        ax_ak = fig.add_axes([0.06, 0.35, 0.16, 0.16])
        ax_hi = fig.add_axes([0.25, 0.35, 0.12, 0.12])
        ax_dist = fig.add_axes([0.16, 0.12, 0.70, 0.15])

        conus.plot(column="tvd", ax=ax, cmap=cmap, norm=norm, linewidth=0.04, edgecolor="#F3F3F3")
        for collection in ax.collections:
            collection.set_rasterized(True)
        ax.set_axis_off()
        add_panel_label(ax, "a", dx=-8)

        if not alaska.empty:
            alaska.plot(column="tvd", ax=ax_ak, cmap=cmap, norm=norm, linewidth=0.04, edgecolor="#F3F3F3")
            for collection in ax_ak.collections:
                collection.set_rasterized(True)
            ax_ak.text(0.02, 0.92, "AK", transform=ax_ak.transAxes, ha="left", va="top", fontsize=8)
            ax_ak.set_axis_off()
        if not hawaii.empty:
            hawaii.plot(column="tvd", ax=ax_hi, cmap=cmap, norm=norm, linewidth=0.04, edgecolor="#F3F3F3")
            for collection in ax_hi.collections:
                collection.set_rasterized(True)
            ax_hi.text(0.02, 0.92, "HI", transform=ax_hi.transAxes, ha="left", va="top", fontsize=8)
            ax_hi.set_axis_off()

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = ax.inset_axes([0.66, 0.02, 0.24, 0.03])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("TVD", fontsize=7.0, labelpad=1)
        cb.ax.tick_params(labelsize=6.6, length=2, pad=1)
        cb.outline.set_linewidth(0.6)

        vals = tvd["tvd"].to_numpy(dtype=float) if dist_tvd is None else np.asarray(dist_tvd, dtype=float)
        parts = ax_dist.violinplot(
            vals,
            positions=[0],
            vert=False,
            widths=0.72,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor("#f4a261")
            body.set_edgecolor("#c44536")
            body.set_alpha(0.40)
            body.set_linewidth(0.8)
        parts["cmedians"].set_color("#222222")
        parts["cmedians"].set_linewidth(1.1)
        rng = np.random.default_rng(20260527)
        jitter = rng.normal(0.0, 0.035, size=vals.size)
        ax_dist.scatter(vals, jitter, s=4.0, color="#7f000d", alpha=0.16, linewidth=0, rasterized=True)
        ax_dist.set_ylim(-0.55, 0.55)
        ax_dist.set_yticks([])
        ax_dist.set_xlabel("TVD")
        add_panel_label(ax_dist, "b", dx=-30, dy=6)
        despine(ax_dist)
        ax_dist.spines["left"].set_visible(False)

        save_figure(fig, out_pdf)
        save_figure(fig, out_png, dpi=300)
        plt.close(fig)


def _weighted_attribute_distribution(arr: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    x = arr.reshape((arr.shape[0],) + SHAPE)
    axes_to_sum = tuple(i for i in range(1, 6) if i != axis + 1)
    marginal = x.sum(axis=axes_to_sum)
    weighted = (marginal * weights[:, None]).sum(axis=0)
    return weighted / max(float(weighted.sum()), 1e-12)


def _plot_attribute_distribution(
    *,
    p_true: np.ndarray,
    pred: np.ndarray,
    weights: np.ndarray,
    out_pdf: Path,
    out_png: Path,
    out_csv: Path,
) -> None:
    rows = []
    for axis, var in enumerate(VARIABLE_ORDER):
        true_dist = _weighted_attribute_distribution(p_true, weights, axis)
        pred_dist = _weighted_attribute_distribution(pred, weights, axis)
        for cat, truth, synthetic in zip(VARIABLE_CATEGORIES[var], true_dist, pred_dist):
            rows.append(
                {
                    "variable": VARIABLE_TITLES[var],
                    "category": cat,
                    "pums_derived_target": float(truth),
                    "synthetic": float(synthetic),
                    "absolute_gap": float(abs(synthetic - truth)),
                }
            )
    dist_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dist_df.to_csv(out_csv, index=False)

    style = PaperStyle(font_size=8.0, axes_labelsize=8.8, axes_titlesize=9.6, tick_labelsize=7.4, legend_fontsize=8.0)
    with paper_style(style):
        fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.6))
        axes_flat = axes.ravel()
        for idx, var in enumerate(VARIABLE_ORDER):
            ax = axes_flat[idx]
            sub = dist_df[dist_df["variable"].eq(VARIABLE_TITLES[var])].reset_index(drop=True)
            x = np.arange(sub.shape[0])
            width = 0.24 if var == "SEX" else 0.38
            ax.bar(
                x - width / 2,
                sub["pums_derived_target"],
                width=width,
                label="PUMS-derived target",
                color="#F2E6D2",
                edgecolor="#333333",
                linewidth=0.45,
            )
            ax.bar(
                x + width / 2,
                sub["synthetic"],
                width=width,
                label="Synthetic",
                color=OKABE_ITO["orange"],
                edgecolor="#333333",
                linewidth=0.45,
                alpha=0.82,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(sub["category"], rotation=35, ha="right")
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.set_ylabel("Percentage")
            despine(ax)
            panel_label = chr(ord("a") + idx)
            label_offset = {"dx": -30, "dy": 14} if panel_label == "d" else {"dx": -30, "dy": 6}
            add_panel_label(ax, panel_label, **label_offset)
        axes_flat[-1].axis("off")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        axes_flat[-1].legend(handles, labels, frameon=False, loc="center")
        fig.subplots_adjust(left=0.095, right=0.985, top=0.96, bottom=0.12, wspace=0.28, hspace=0.58)
        save_figure(fig, out_pdf)
        save_figure(fig, out_png, dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        type=Path,
        default=REPO_ROOT
        / "outputs/_paper1_full_us_spatial_population_2023_k1440_geoidfix_20260524T091418Z/model/predicted_joint_wide_seed2_fine1440_50states_plus_dc.npz",
    )
    parser.add_argument(
        "--puma-qc",
        type=Path,
        default=REPO_ROOT / "outputs/sigspatial_national_product_assets_20260526/puma_qc_summary.csv",
    )
    parser.add_argument(
        "--puma-shp",
        type=Path,
        default=REPO_ROOT / "data/geo_cache/cb_2020_us_puma20_500k/cb_2020_us_puma20_500k.shp",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "SigSpatial2026_spop/figures")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "outputs/sigspatial_data_reporting_assets_20260526",
    )
    parser.add_argument(
        "--allpuma-tvd-by-seed-csv",
        type=Path,
        default=REPO_ROOT
        / "outputs/sigspatial_data_reporting_assets_20260526/k960_allpuma_seed_metrics_20260527/allpuma_tvd_by_seed_puma.csv",
    )
    args = parser.parse_args()

    p_true, pred, keys = _load_joint(args.npz, args.puma_qc)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    dist_tvd = None
    if args.allpuma_tvd_by_seed_csv.exists():
        tvd_seed_df = pd.read_csv(args.allpuma_tvd_by_seed_csv, dtype={"puma_uid": str, "statefp": str})
        tvd_seed_df["puma_uid"] = tvd_seed_df["puma_uid"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)
        tvd_seed_df["statefp"] = tvd_seed_df["statefp"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
        tvd_seed_df["tvd"] = pd.to_numeric(tvd_seed_df["tvd"], errors="raise")
        tvd_df = (
            tvd_seed_df.groupby(["puma_uid", "statefp"], as_index=False)
            .agg(tvd=("tvd", "mean"), n_seeds=("seed", "nunique"))
            .merge(keys[["puma_uid", "target_persons"]], on="puma_uid", how="left")
        )
        dist_tvd = tvd_seed_df["tvd"].to_numpy(dtype=float)
    else:
        tvd = 0.5 * np.abs(pred - p_true).sum(axis=1)
        tvd_df = keys[["puma_uid", "statefp", "target_persons"]].copy()
        tvd_df["tvd"] = tvd
    tvd_df.to_csv(args.source_dir / "puma_level_tvd.csv", index=False)

    _plot_puma_tvd_map(
        tvd=tvd_df,
        dist_tvd=dist_tvd,
        puma_shp=args.puma_shp,
        out_pdf=args.out_dir / "fig_02_national_puma_tvd_map.pdf",
        out_png=args.out_dir / "fig_02_national_puma_tvd_map.png",
    )
    _plot_attribute_distribution(
        p_true=p_true,
        pred=pred,
        weights=keys["target_persons"].to_numpy(dtype=float),
        out_pdf=args.out_dir / "fig_03_national_attribute_distribution.pdf",
        out_png=args.out_dir / "fig_03_national_attribute_distribution.png",
        out_csv=args.source_dir / "attribute_distribution_comparison.csv",
    )


if __name__ == "__main__":
    main()
