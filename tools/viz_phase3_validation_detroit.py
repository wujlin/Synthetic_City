from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path("/Users/jinlin/Desktop/Project/Synthetic_City")
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs" / "_sync_phase3_detroit_hcenter_20260330" / "metrics_detailed"
DEFAULT_TRACT_ZIP = PROJECT_ROOT / "dataset" / "cache" / "geo" / "tl_2023_26_tract.zip"
DEFAULT_OUTDIR = PROJECT_ROOT / "figures" / "phase3_validation_detroit_latest"

COUNTY_NAME_BY_GEOID = {
    "26049": "Genesee",
    "26059": "Hillsdale",
    "26063": "Huron",
    "26065": "Ingham",
    "26075": "Jackson",
    "26087": "Lapeer",
    "26091": "Lenawee",
    "26093": "Livingston",
    "26099": "Macomb",
    "26115": "Monroe",
    "26125": "Oakland",
    "26147": "St. Clair",
    "26151": "Sanilac",
    "26155": "Shiawassee",
    "26157": "Tuscola",
    "26161": "Washtenaw",
    "26163": "Wayne",
}


def _panel_label(ax, text: str, x: float = 0.0, y: float = 1.02) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def _style_map_ax(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _share_norm(df: pd.DataFrame, cols: list[str]) -> LogNorm:
    vals = []
    for col in cols:
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy()
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size:
            vals.append(arr)
    cat = np.concatenate(vals) if vals else np.array([1e-8, 1e-4])
    vmin = float(cat.min())
    vmax = float(cat.max())
    if vmin == vmax:
        vmax = vmin * 1.5
    return LogNorm(vmin=vmin, vmax=vmax)


def _plot_share_map(ax, gdf: gpd.GeoDataFrame, column: str, norm: LogNorm, title: str | None, cmap: str = "Blues") -> None:
    plot_df = gdf.copy()
    plot_df[column] = plot_df[column].where(plot_df[column] > 0)
    plot_df.plot(
        ax=ax,
        color="#f3f4f6",
        linewidth=0.15,
        edgecolor="#dadde2",
    )
    plot_df.dropna(subset=[column]).plot(
        ax=ax,
        column=column,
        cmap=cmap,
        norm=norm,
        linewidth=0.12,
        edgecolor="#f2f2f2",
    )
    if title:
        ax.set_title(title, fontsize=11)
    _style_map_ax(ax)


def _plot_diverging_map(
    ax,
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str | None,
    vlim: float,
    cmap: str = "RdBu_r",
) -> None:
    plot_df = gdf.copy()
    plot_df.plot(
        ax=ax,
        color="#f3f4f6",
        linewidth=0.15,
        edgecolor="#dadde2",
    )
    plot_df.dropna(subset=[column]).plot(
        ax=ax,
        column=column,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim),
        linewidth=0.12,
        edgecolor="#f2f2f2",
    )
    if title:
        ax.set_title(title, fontsize=11)
    _style_map_ax(ax)


def _plot_spearman_map(ax, gdf: gpd.GeoDataFrame, column: str, title: str) -> None:
    plot_df = gdf.copy()
    plot_df.plot(
        ax=ax,
        color="#f3f4f6",
        linewidth=0.15,
        edgecolor="#dadde2",
    )
    plot_df.dropna(subset=[column]).plot(
        ax=ax,
        column=column,
        cmap="RdYlBu",
        norm=Normalize(vmin=-1.0, vmax=1.0),
        linewidth=0.12,
        edgecolor="#f2f2f2",
    )
    ax.set_title(title, fontsize=11)
    _style_map_ax(ax)


def _save_home_validation(
    tracts: gpd.GeoDataFrame,
    home_tract_df: pd.DataFrame,
    home_bg_df: pd.DataFrame,
    out_path: Path,
    *,
    home_ref_label: str,
    home_ref_map_title: str,
) -> dict:
    merged = tracts.merge(home_tract_df, on="tract_geoid", how="left")
    merged = merged.merge(
        home_bg_df[["tract_geoid", "spearman_bg", "eligible"]],
        on="tract_geoid",
        how="left",
    )
    merged.loc[~merged["eligible"].fillna(False), "spearman_bg"] = np.nan
    share_norm = _share_norm(merged, ["left_share", "right_share"])

    fig = plt.figure(figsize=(10.5, 8.8))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.90,
        top=0.95,
        bottom=0.11,
        wspace=0.10,
        hspace=0.12,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _plot_share_map(ax_a, merged, "left_share", share_norm, "Synthetic home share")
    _plot_share_map(ax_b, merged, "right_share", share_norm, home_ref_map_title)
    _plot_spearman_map(ax_c, merged, "spearman_bg", "Within-tract local rank agreement")

    spearman = (
        pd.to_numeric(home_bg_df.loc[home_bg_df["eligible"], "spearman_bg"], errors="coerce")
        .dropna()
        .sort_values()
        .to_numpy()
    )
    y = np.arange(1, spearman.size + 1) / spearman.size if spearman.size else np.array([])
    if spearman.size:
        ax_d.plot(spearman, y, color="#2c7fb8", lw=2.2)
    ax_d.axvline(0.5, color="#a0a0a0", lw=1.0, ls="--")
    ax_d.axvline(float(np.nanmedian(spearman)) if spearman.size else 0.0, color="#5a5a5a", lw=1.0, ls=":")
    ax_d.set_xlim(-1.0, 1.0)
    ax_d.set_ylim(0.0, 1.0)
    ax_d.set_xlabel("Tract-level BG Spearman")
    ax_d.set_ylabel("Cumulative share of eligible tracts")
    ax_d.grid(axis="both", color="#e6e6e6", lw=0.6)
    ax_d.set_title("Distribution across tracts", fontsize=11)
    for side in ("top", "right"):
        ax_d.spines[side].set_visible(False)

    for label, ax in zip(("a", "b", "c", "d"), (ax_a, ax_b, ax_c, ax_d)):
        _panel_label(ax, label, x=-0.10, y=1.03)

    cax_share = fig.add_axes([0.92, 0.57, 0.018, 0.28])
    sm_share = plt.cm.ScalarMappable(norm=share_norm, cmap="Blues")
    cb_share = fig.colorbar(sm_share, cax=cax_share)
    cb_share.set_label("Share within Detroit validation area")

    cax_corr = fig.add_axes([0.18, 0.06, 0.20, 0.018])
    sm_corr = plt.cm.ScalarMappable(norm=Normalize(vmin=-1.0, vmax=1.0), cmap="RdYlBu")
    cb_corr = fig.colorbar(sm_corr, cax=cax_corr, orientation="horizontal")
    cb_corr.set_label("BG-rank Spearman")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return {
        "eligible_tracts": int(home_bg_df["eligible"].fillna(False).sum()),
        "mean_bg_spearman": float(home_bg_df.loc[home_bg_df["eligible"], "spearman_bg"].mean()),
        "median_bg_spearman": float(home_bg_df.loc[home_bg_df["eligible"], "spearman_bg"].median()),
        "share_ge_05": float((home_bg_df.loc[home_bg_df["eligible"], "spearman_bg"] >= 0.5).mean()),
    }


def _save_work_validation(
    tracts: gpd.GeoDataFrame,
    work_tract_df: pd.DataFrame,
    commute_df: pd.DataFrame,
    out_path: Path,
) -> dict:
    merged = tracts.merge(work_tract_df, on="tract_geoid", how="left")
    merged["share_gap"] = merged["left_share"].fillna(0.0) - merged["right_share"].fillna(0.0)
    share_norm = _share_norm(merged, ["left_share", "right_share"])
    gap_vlim = float(np.nanmax(np.abs(merged["share_gap"]))) if len(merged) else 1e-4
    gap_vlim = max(gap_vlim, 1e-6)

    fig = plt.figure(figsize=(10.5, 8.8))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.90,
        top=0.95,
        bottom=0.08,
        wspace=0.10,
        hspace=0.14,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _plot_share_map(ax_a, merged, "left_share", share_norm, "Synthetic work share")
    _plot_share_map(ax_b, merged, "right_share", share_norm, "Mobility day-anchor share")
    _plot_diverging_map(ax_c, merged, "share_gap", "Synthetic minus mobility share", gap_vlim)

    x_labels = [f"{int(l)}-{int(r)}" if np.isfinite(r) else f"{int(l)}+" for l, r in zip(commute_df["bin_left_km"], commute_df["bin_right_km"])]
    x = np.arange(len(x_labels))
    ax_d.plot(x, commute_df["synthetic_share"], color="#8F7A67", marker="o", lw=1.9, label="Synthetic")
    ax_d.plot(x, commute_df["mobility_share"], color="#8FAFBE", marker="o", lw=1.9, label="Mobility")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(x_labels, rotation=35, ha="right")
    ax_d.set_xlabel("Commute distance bin (km)")
    ax_d.set_ylabel("Share")
    ax_d.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax_d.set_title("Commute distance distribution", fontsize=11)
    ax_d.legend(frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax_d.spines[side].set_visible(False)

    for label, ax in zip(("a", "b", "c", "d"), (ax_a, ax_b, ax_c, ax_d)):
        _panel_label(ax, label)

    cax_share = fig.add_axes([0.92, 0.57, 0.018, 0.28])
    sm_share = plt.cm.ScalarMappable(norm=share_norm, cmap="Blues")
    cb_share = fig.colorbar(sm_share, cax=cax_share)
    cb_share.set_label("Share within Detroit validation area")

    cax_gap = fig.add_axes([0.16, 0.08, 0.20, 0.018])
    sm_gap = plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-gap_vlim, vcenter=0.0, vmax=gap_vlim), cmap="RdBu_r")
    cb_gap = fig.colorbar(sm_gap, cax=cax_gap, orientation="horizontal")
    cb_gap.set_label("Share gap")
    cb_gap.formatter = FuncFormatter(lambda x, _pos: f"{x:.3f}")
    cb_gap.update_ticks()

    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return {
        "max_abs_share_gap": float(gap_vlim),
        "n_work_tracts": int(work_tract_df.shape[0]),
        "synthetic_median_km": float(
            np.nan
            if commute_df.empty
            else np.interp(
                0.5,
                np.cumsum(commute_df["synthetic_share"].to_numpy()),
                commute_df["bin_right_km"].to_numpy(),
            )
        ),
    }


def _county_order(df: pd.DataFrame) -> list[str]:
    tmp = df.copy()
    tmp["home_county"] = tmp["home_tract_geoid"].astype(str).str[:5]
    tmp["work_county"] = tmp["work_tract_geoid"].astype(str).str[:5]
    totals = (
        tmp.groupby("work_county")[["synthetic_count", "mobility_count"]]
        .sum()
        .sum(axis=1)
        .sort_values(ascending=False)
    )
    return totals.index.tolist()


def _matrix_from_od(df: pd.DataFrame, value_col: str, order: list[str]) -> np.ndarray:
    tmp = df.copy()
    tmp["home_county"] = tmp["home_tract_geoid"].astype(str).str[:5]
    tmp["work_county"] = tmp["work_tract_geoid"].astype(str).str[:5]
    mat = (
        tmp.groupby(["home_county", "work_county"])[value_col]
        .sum()
        .reindex(pd.MultiIndex.from_product([order, order], names=["home_county", "work_county"]), fill_value=0.0)
        .unstack(fill_value=0.0)
    )
    arr = mat.to_numpy(dtype=float)
    total = arr.sum()
    return arr / total if total > 0 else arr


def _save_od_validation(work_od_df: pd.DataFrame, out_path: Path) -> dict:
    order = _county_order(work_od_df)
    syn = _matrix_from_od(work_od_df, "synthetic_count", order)
    mob = _matrix_from_od(work_od_df, "mobility_count", order)
    diff = syn - mob

    labels = [COUNTY_NAME_BY_GEOID.get(c, c[-3:]) for c in order]
    positive = np.concatenate([syn[syn > 0], mob[mob > 0]])
    norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max())) if positive.size else LogNorm(vmin=1e-6, vmax=1e-2)
    diff_vlim = max(float(np.abs(diff).max()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    ims = []
    ims.append(axes[0].imshow(np.where(syn > 0, syn, np.nan), cmap="Blues", norm=norm))
    ims.append(axes[1].imshow(np.where(mob > 0, mob, np.nan), cmap="Blues", norm=norm))
    ims.append(axes[2].imshow(diff, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-diff_vlim, vcenter=0.0, vmax=diff_vlim)))

    for idx, ax in enumerate(axes):
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7.2)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=6.8)
        if idx == 0:
            ax.set_ylabel("Home county")
        _panel_label(ax, chr(ord("a") + idx), x=-0.18, y=1.02)

    cax_share = fig.add_axes([0.92, 0.54, 0.015, 0.30])
    cb_share = fig.colorbar(ims[0], cax=cax_share)
    cb_share.set_label("County-OD share")

    cax_gap = fig.add_axes([0.92, 0.14, 0.015, 0.30])
    cb_gap = fig.colorbar(ims[2], cax=cax_gap)
    cb_gap.set_label("Share gap")
    cb_gap.formatter = FuncFormatter(lambda x, _pos: f"{x:.3f}")
    cb_gap.update_ticks()

    fig.text(0.47, 0.045, "Work county", ha="center", va="center", fontsize=10.5)
    fig.subplots_adjust(left=0.08, right=0.89, top=0.93, bottom=0.24, wspace=0.22)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    same_county_syn = float(np.trace(syn))
    same_county_mob = float(np.trace(mob))
    return {
        "n_counties": len(order),
        "county_labels": labels,
        "same_county_synthetic_share": same_county_syn,
        "same_county_mobility_share": same_county_mob,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--tract_zip", type=Path, default=DEFAULT_TRACT_ZIP)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--home_ref_label", default="Mobility")
    parser.add_argument("--home_ref_map_title", default="Mobility night-anchor share")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    home_bg_df = pd.read_csv(args.metrics_dir / "home_bg_spearman_by_tract.csv", dtype={"tract_geoid": str})
    home_tract_df = pd.read_csv(args.metrics_dir / "home_tract_comparison.csv", dtype={"tract_geoid": str})
    work_tract_df = pd.read_csv(args.metrics_dir / "work_tract_comparison.csv", dtype={"tract_geoid": str})
    work_od_df = pd.read_csv(
        args.metrics_dir / "work_od_comparison.csv",
        dtype={"home_tract_geoid": str, "work_tract_geoid": str},
    )
    commute_df = pd.read_csv(args.metrics_dir / "commute_distance_bins.csv")

    tract_ids = sorted(
        set(home_tract_df["tract_geoid"].astype(str))
        | set(work_tract_df["tract_geoid"].astype(str))
        | set(work_od_df["home_tract_geoid"].astype(str))
        | set(work_od_df["work_tract_geoid"].astype(str))
    )
    tracts = gpd.read_file(args.tract_zip)
    tracts["tract_geoid"] = tracts["GEOID"].astype(str)
    tracts = tracts.loc[tracts["tract_geoid"].isin(tract_ids)].copy()

    home_summary = _save_home_validation(
        tracts=tracts,
        home_tract_df=home_tract_df,
        home_bg_df=home_bg_df,
        out_path=args.outdir / "home_validation.png",
        home_ref_label=str(args.home_ref_label),
        home_ref_map_title=str(args.home_ref_map_title),
    )
    work_summary = _save_work_validation(
        tracts=tracts,
        work_tract_df=work_tract_df,
        commute_df=commute_df,
        out_path=args.outdir / "work_validation.png",
    )
    od_summary = _save_od_validation(
        work_od_df=work_od_df,
        out_path=args.outdir / "od_validation.png",
    )

    manifest = {
        "metrics_dir": str(args.metrics_dir),
        "tract_zip": str(args.tract_zip),
        "tract_count": int(tracts.shape[0]),
        "artifacts": {
            "home_validation_png": str(args.outdir / "home_validation.png"),
            "work_validation_png": str(args.outdir / "work_validation.png"),
            "od_validation_png": str(args.outdir / "od_validation.png"),
        },
        "home_summary": home_summary,
        "work_summary": work_summary,
        "od_summary": od_summary,
    }
    with open(args.outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[ok] wrote {args.outdir / 'home_validation.png'}")
    print(f"[ok] wrote {args.outdir / 'work_validation.png'}")
    print(f"[ok] wrote {args.outdir / 'od_validation.png'}")
    print(f"[ok] wrote {args.outdir / 'manifest.json'}")


if __name__ == "__main__":
    main()
