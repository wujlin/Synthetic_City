#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.ticker import FixedLocator, FuncFormatter
from shapely.geometry import box

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.plot_style import paper_style, save_figure
from src.synthpop.data.lodes import aggregate_lodes_wac_to_tract, load_lodes_rac_or_wac


PROJECT_ROOT = Path("/Users/jinlin/Desktop/Project/Synthetic_City")
DEFAULT_OUTDIR = PROJECT_ROOT / "figures" / "phase3_detroit_metro_latest"
DEFAULT_TRACT_ZIP = PROJECT_ROOT / "dataset" / "cache" / "geo" / "tl_2023_26_tract.zip"
DEFAULT_ROAD_ZIP = PROJECT_ROOT / "dataset" / "cache" / "geo" / "MI_road_cleaned.shp.zip"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "outputs" / "_sync_phase3_detroit_hcenter_20260330" / "metrics_detailed"
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "outputs" / "_sync_phase3_detroit_hcenter_20260330" / "overview_samples"
DEFAULT_MICRO_MANIFEST = PROJECT_ROOT / "figures" / "phase3_detroit_metro_latest" / "micro_examples_manifest.json"
DEFAULT_HOME_DECKGL_CROP = PROJECT_ROOT / "figures" / "phase3_detroit_metro_latest" / "home_overview_deckgl_cropped.png"
DEFAULT_ACS_B01001 = PROJECT_ROOT / "dataset" / "census" / "acs5_2022_B01001_tract_michigan.csv.gz"
DEFAULT_WAC_S000 = PROJECT_ROOT / "dataset" / "lodes" / "mi_wac_S000_JT00_2020.csv.gz"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _despine_map(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")


def _panel_label(ax, text: str, x: float = 0.015, y: float = 0.985) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.15},
    )


def _study_tracts(tract_zip: Path, tract_ids: list[str]) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(tract_zip)
    gdf["tract_geoid"] = gdf["GEOID"].astype(str)
    gdf = gdf.loc[gdf["tract_geoid"].isin(set(tract_ids))].copy()
    return gdf


def _load_bounds(manifest_path: Path) -> tuple[float, float, float, float]:
    meta = _read_json(manifest_path)
    bounds = meta["focus_meta"]["bounds"]
    return tuple(float(v) for v in bounds)


def _subset_points(df: pd.DataFrame, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    xmin, ymin, xmax, ymax = bounds
    return df.loc[
        (df["x"] >= xmin)
        & (df["x"] <= xmax)
        & (df["y"] >= ymin)
        & (df["y"] <= ymax)
    ].copy()


def _read_geodata(path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".zip":
        return gpd.read_file(f"zip://{path}")
    return gpd.read_file(path)


def _prepare_roads(path: Path, target_crs) -> gpd.GeoDataFrame:
    roads = _read_geodata(path)
    keep = [c for c in ["MTFCC", "geometry"] if c in roads.columns]
    roads = roads[keep].copy()
    if roads.crs != target_crs:
        roads = roads.to_crs(target_crs)
    roads["MTFCC"] = roads["MTFCC"].astype(str)
    return roads


def _clip_to_bbox(gdf: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = bounds
    geom = box(xmin, ymin, xmax, ymax)
    try:
        idx = list(gdf.sindex.query(geom, predicate="intersects"))
        subset = gdf.iloc[idx].copy()
    except Exception:
        subset = gdf[gdf.intersects(geom)].copy()
    try:
        clipper = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs=gdf.crs)
        return gpd.clip(subset, clipper)
    except Exception:
        return subset


def _micro_meta(path: Path) -> dict:
    return _read_json(path)


def _image_panel(ax, image_path: Path) -> None:
    img = plt.imread(image_path)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _fmt_count(x, _pos=None) -> str:
    if x >= 1000:
        return f"{int(x):,}"
    if x >= 1:
        return f"{int(x)}"
    return f"{x:g}"


def _fmt_compact_count(x, _pos=None) -> str:
    if x >= 1000:
        v = x / 1000.0
        return f"{int(v)}k" if abs(v - round(v)) < 1e-9 else f"{v:.1f}k"
    if x >= 1:
        return f"{int(x)}"
    return f"{x:g}"


def _locator_panel(ax, tracts: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> None:
    metro = tracts.dissolve()
    metro.plot(ax=ax, color="#F5F1EA", edgecolor="#D8D1C5", linewidth=0.45, zorder=0)
    tracts.boundary.plot(ax=ax, color="#DDD6CA", linewidth=0.14, alpha=0.45, zorder=1)
    highlight = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=tracts.crs)
    highlight.boundary.plot(ax=ax, color="#C97A1D", linewidth=1.35, zorder=4)
    _despine_map(ax)


def _small_horizontal_colorbar(
    ax,
    sm,
    label: str,
    ticks: list[float],
    *,
    box: tuple[float, float, float, float] = (0.12, -0.12, 0.52, 0.045),
    formatter=FuncFormatter(_fmt_count),
    tick_labelsize: float = 9.2,
    label_size: float = 10.5,
    outline_lw: float = 0.55,
) -> None:
    cax = ax.inset_axes(list(box))
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(outline_lw)
    cb.outline.set_edgecolor("#8E8678")
    cb.ax.tick_params(length=1.8, width=0.55, pad=1.2, labelsize=tick_labelsize, colors="#6F675B")
    cb.locator = FixedLocator(ticks)
    cb.update_ticks()
    cb.ax.xaxis.set_major_formatter(formatter)
    cb.set_label(label, labelpad=1.8, size=label_size, color="#5B5449")


def _choropleth_panel(
    ax,
    gdf: gpd.GeoDataFrame,
    value_col: str,
    cmap: str,
    label: str,
    *,
    highlight_bounds: tuple[float, float, float, float] | None = None,
    highlight_color: str = "#C97A1D",
    use_log: bool = True,
    ticks: list[float] | None = None,
    colorbar_box: tuple[float, float, float, float] = (0.14, -0.10, 0.48, 0.040),
    formatter=FuncFormatter(_fmt_count),
    colorbar_tick_labelsize: float = 8.0,
    colorbar_label_size: float = 9.0,
    colorbar_outline_lw: float = 0.55,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    values = pd.to_numeric(gdf[value_col], errors="coerce")
    positive = values[np.isfinite(values) & (values > 0)]
    vmin_eff = float(positive.min()) if len(positive) else 1.0
    vmax_eff = float(positive.max()) if len(positive) else 2.0
    if vmin is not None:
        vmin_eff = float(vmin)
    if vmax is not None:
        vmax_eff = float(vmax)
    norm = LogNorm(vmin=vmin_eff, vmax=vmax_eff) if use_log else Normalize(vmin=vmin_eff, vmax=vmax_eff)
    gdf.plot(
        ax=ax,
        column=value_col,
        cmap=cmap,
        norm=norm,
        linewidth=0.12,
        edgecolor="#E4DED2",
        legend=False,
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    if ticks is None:
        ticks = [t for t in [100, 500, 1000, 5000] if vmin_eff <= t <= vmax_eff]
        if len(ticks) < 2:
            ticks = [vmin_eff, vmax_eff]
    _small_horizontal_colorbar(
        ax,
        sm,
        label,
        ticks,
        box=colorbar_box,
        formatter=formatter,
        tick_labelsize=colorbar_tick_labelsize,
        label_size=colorbar_label_size,
        outline_lw=colorbar_outline_lw,
    )
    if highlight_bounds is not None:
        highlight = gpd.GeoDataFrame(geometry=[box(*highlight_bounds)], crs=gdf.crs)
        highlight.boundary.plot(ax=ax, color=highlight_color, linewidth=1.25, zorder=4)
    _despine_map(ax)


def _home_density_panel(
    ax,
    tracts: gpd.GeoDataFrame,
    points: pd.DataFrame,
    *,
    cmap: str = "YlOrBr",
    label: str = "Home-point density",
    ticks: list[float] = [1, 10, 100],
    colorbar_box: tuple[float, float, float, float] = (0.14, -0.08, 0.44, 0.040),
    colorbar_tick_labelsize: float = 8.0,
    colorbar_label_size: float = 9.0,
    colorbar_outline_lw: float = 0.55,
) -> None:
    tracts.plot(ax=ax, color="#FCFAF6", edgecolor="#E3DCCE", linewidth=0.12, zorder=0)
    xmin, ymin, xmax, ymax = tracts.total_bounds
    hb = ax.hexbin(
        points["x"].to_numpy(),
        points["y"].to_numpy(),
        gridsize=95,
        extent=(xmin, xmax, ymin, ymax),
        mincnt=1,
        bins="log",
        cmap=cmap,
        linewidths=0.0,
        alpha=0.95,
        zorder=2,
    )
    _small_horizontal_colorbar(
        ax,
        hb,
        label,
        ticks,
        box=colorbar_box,
        tick_labelsize=colorbar_tick_labelsize,
        label_size=colorbar_label_size,
        outline_lw=colorbar_outline_lw,
    )
    _despine_map(ax)


def _home_distribution_panel(ax, tract_df: pd.DataFrame, *, home_ref_label: str) -> None:
    syn = pd.to_numeric(tract_df["left_share"], errors="coerce")
    mob = pd.to_numeric(tract_df["right_share"], errors="coerce")
    syn = syn[np.isfinite(syn) & (syn > 0)]
    mob = mob[np.isfinite(mob) & (mob > 0)]

    all_vals = np.concatenate([syn.to_numpy(), mob.to_numpy()]) if len(syn) and len(mob) else np.array([1e-6, 1e-2])
    bins = np.geomspace(all_vals.min(), all_vals.max() * 1.02, 28)

    ax.hist(
        syn,
        bins=bins,
        histtype="stepfilled",
        color="#4C94C6",
        alpha=0.42,
        edgecolor="#4C94C6",
        linewidth=0.9,
        label="Synthetic",
    )
    ax.hist(
        mob,
        bins=bins,
        histtype="step",
        color="#C97A1D",
        linewidth=1.35,
        label=home_ref_label,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Tract home share")
    ax.set_ylabel("Number of tracts")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    q50 = float(np.median(syn))
    q90 = float(np.quantile(syn, 0.9))
    ax.axvline(q50, color="#1E5E84", lw=1.0, ls="--", label="Synthetic median")
    ax.axvline(q90, color="#8C510A", lw=1.0, ls="--", label="Synthetic p90")
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9.4,
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.4,
    )


def _share_distribution_panel(
    ax,
    df: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
    x_label: str,
    q_label_prefix: str = "Synthetic",
) -> None:
    left = pd.to_numeric(df[left_col], errors="coerce")
    right = pd.to_numeric(df[right_col], errors="coerce")
    left = left[np.isfinite(left) & (left > 0)]
    right = right[np.isfinite(right) & (right > 0)]
    all_vals = np.concatenate([left.to_numpy(), right.to_numpy()]) if len(left) and len(right) else np.array([1e-6, 1e-2])
    bins = np.geomspace(all_vals.min(), all_vals.max() * 1.02, 28)
    ax.hist(
        left,
        bins=bins,
        histtype="stepfilled",
        color=left_color,
        alpha=0.34,
        edgecolor=left_color,
        linewidth=0.9,
        label=left_label,
    )
    ax.hist(
        right,
        bins=bins,
        histtype="step",
        color=right_color,
        linewidth=1.35,
        label=right_label,
    )
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of tracts")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    q50 = float(np.median(left))
    q90 = float(np.quantile(left, 0.9))
    ax.axvline(q50, color="#1E5E84", lw=1.0, ls="--", label=f"{q_label_prefix} median")
    ax.axvline(q90, color="#8C510A", lw=1.0, ls="--", label=f"{q_label_prefix} p90")
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9.3,
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.35,
    )


def _share_scatter_panel(
    ax,
    df: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    x_label: str,
    y_label: str,
    point_color: str,
) -> None:
    x = pd.to_numeric(df[left_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[right_col], errors="coerce").to_numpy()
    eps = 1e-6
    lo = max(min(float(np.nanmin(x)), float(np.nanmin(y))), eps)
    hi = max(float(np.nanmax(x)), float(np.nanmax(y))) * 1.08
    ax.scatter(x + eps, y + eps, s=8, color=point_color, alpha=0.24, linewidths=0.0)
    ax.plot([lo, hi], [lo, hi], color="#7A7A7A", lw=1.0, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    spearman = pd.Series(x).corr(pd.Series(y), method="spearman")
    cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    text = f"Tract share  ρ={spearman:.3f}\nCosine={cosine:.3f}"
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.6,
        color="#4C443A",
        bbox={"facecolor": "white", "edgecolor": "#DDD5C9", "linewidth": 0.7, "pad": 0.35},
    )


def _home_validation_panel(
    ax,
    tract_df: pd.DataFrame,
    *,
    bg_df: pd.DataFrame | None = None,
    y_label: str = "Mobility residential share",
) -> None:
    x = tract_df["left_share"].to_numpy()
    y = tract_df["right_share"].to_numpy()
    eps = 1e-6
    lo = max(min(float(np.nanmin(x)), float(np.nanmin(y))), eps)
    hi = max(float(np.nanmax(x)), float(np.nanmax(y))) * 1.08
    ax.scatter(x + eps, y + eps, s=8, color="#C97A1D", alpha=0.24, linewidths=0.0)
    ax.plot([lo, hi], [lo, hi], color="#7A7A7A", lw=1.0, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Synthetic tract home share")
    ax.set_ylabel(y_label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    spearman = pd.Series(x).corr(pd.Series(y), method="spearman")
    cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    text = (
        f"Tract share  ρ={spearman:.3f}\n"
        f"Cosine={cosine:.3f}"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.6,
        color="#4C443A",
        bbox={"facecolor": "white", "edgecolor": "#DDD5C9", "linewidth": 0.7, "pad": 0.35},
    )


def _home_rank_panel(ax, bg_df: pd.DataFrame) -> None:
    bg_valid = bg_df.loc[pd.to_numeric(bg_df["spearman_bg"], errors="coerce").notna()].copy()
    bg_valid["spearman_bg"] = pd.to_numeric(bg_valid["spearman_bg"], errors="coerce")
    vals = bg_valid["spearman_bg"].dropna().to_numpy()
    thresholds = np.linspace(-1.0, 1.0, 161)
    survival = np.array([(vals >= t).mean() for t in thresholds])

    share_v = float((vals >= 0.5).mean())

    ax.plot(thresholds, survival, color="#4C94C6", lw=2.0)
    ax.fill_between(thresholds, survival, 0.0, color="#4C94C6", alpha=0.16)
    ax.axvline(0.5, color="#8C510A", lw=1.1, ls="--", label="Threshold = 0.5")
    ax.axhline(share_v, color="#8C510A", lw=1.0, ls=":", alpha=0.95, label="Share at 0.5")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("BG-rank Spearman threshold")
    ax.set_ylabel("Share of tracts ≥ threshold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9.2, handlelength=1.6, labelspacing=0.35)


def _home_rank_validation_figure(
    *,
    bg_df: pd.DataFrame,
    out_path: Path,
) -> None:
    with paper_style():
        fig = plt.figure(figsize=(6.2, 4.5))
        ax = fig.add_subplot(111)
        _home_rank_panel(ax, bg_df)
        _panel_label(ax, "a", x=-0.10, y=1.03)
        save_figure(fig, out_path)
        plt.close(fig)


def _map_panel(
    ax,
    gdf: gpd.GeoDataFrame,
    value_col: str,
    cmap: str,
    cax,
    label: str,
    orientation: str = "vertical",
    highlight_bounds: tuple[float, float, float, float] | None = None,
    highlight_color: str | None = None,
) -> None:
    plot_df = gdf.copy()
    positive = pd.to_numeric(plot_df[value_col], errors="coerce")
    positive = positive[np.isfinite(positive) & (positive > 0)]
    vmin = float(positive.min()) if len(positive) else 1.0
    vmax = float(positive.max()) if len(positive) else 2.0
    norm = LogNorm(vmin=vmin, vmax=vmax)
    plot_df.plot(
        ax=ax,
        column=value_col,
        cmap=cmap,
        norm=norm,
        linewidth=0.12,
        edgecolor="#E7E7E7",
        legend=False,
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax, orientation=orientation)
    if orientation == "horizontal":
        cb.set_label(label, labelpad=3)
    else:
        cb.set_label(label, labelpad=4)
    cb.outline.set_linewidth(0.7)
    cb.ax.tick_params(length=2.5, width=0.8, pad=2)
    if highlight_bounds is not None:
        highlight = gpd.GeoDataFrame(geometry=[box(*highlight_bounds)], crs=gdf.crs)
        highlight.boundary.plot(
            ax=ax,
            color=highlight_color or "#5A5A5A",
            linewidth=1.25,
            zorder=4,
        )
    _despine_map(ax)


def _local_pattern_panel(
    ax,
    tracts_zoom: gpd.GeoDataFrame,
    roads_zoom: gpd.GeoDataFrame,
    pts_zoom: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    point_color: str,
    primary_mtfcc: list[str],
) -> None:
    tracts_zoom.plot(ax=ax, color="#FBFBFB", edgecolor="#D6D6D6", linewidth=0.28, zorder=0)
    roads_bg = roads_zoom.loc[~roads_zoom["MTFCC"].isin(primary_mtfcc)]
    roads_fg = roads_zoom.loc[roads_zoom["MTFCC"].isin(primary_mtfcc)]
    if len(roads_bg):
        roads_bg.plot(ax=ax, color="#DDDDDD", linewidth=0.36, alpha=0.75, zorder=1)
    if len(roads_fg):
        roads_fg.plot(ax=ax, color="#727272", linewidth=0.68, alpha=0.95, zorder=2)
    if len(pts_zoom):
        ax.scatter(
            pts_zoom["x"].to_numpy(),
            pts_zoom["y"].to_numpy(),
            s=7.0,
            c=point_color,
            alpha=0.42,
            linewidths=0.0,
            zorder=3,
        )
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    _despine_map(ax)


def _home_occupancy_panel(
    ax,
    tracts_zoom: gpd.GeoDataFrame,
    selected_zoom: gpd.GeoDataFrame,
    roads_zoom: gpd.GeoDataFrame,
    pts_zoom: pd.DataFrame,
) -> None:
    tracts_3857 = tracts_zoom.to_crs(3857)
    selected_3857 = selected_zoom.to_crs(3857)
    roads_3857 = roads_zoom.to_crs(3857)
    occ = pts_zoom.groupby(["x", "y"], as_index=False).size().rename(columns={"size": "residents"})
    occ_gdf = gpd.GeoDataFrame(occ, geometry=gpd.points_from_xy(occ["x"], occ["y"]), crs=tracts_zoom.crs).to_crs(3857)
    occ_gdf["x_m"] = occ_gdf.geometry.x
    occ_gdf["y_m"] = occ_gdf.geometry.y

    xmin, ymin, xmax, ymax = selected_3857.total_bounds if not selected_3857.empty else tracts_3857.total_bounds
    xpad = (xmax - xmin) * 0.18
    ypad = (ymax - ymin) * 0.18
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad

    occ_local = occ_gdf.loc[
        (occ_gdf["x_m"] >= xmin)
        & (occ_gdf["x_m"] <= xmax)
        & (occ_gdf["y_m"] >= ymin)
        & (occ_gdf["y_m"] <= ymax)
    ].copy()

    span_x = xmax - xmin
    span_y = ymax - ymin
    nx = 26
    ny = max(16, int(round(nx * span_y / max(span_x, 1.0))))
    xedges = np.linspace(xmin, xmax, nx + 1)
    yedges = np.linspace(ymin, ymax, ny + 1)
    heights, _, _ = np.histogram2d(
        occ_local["x_m"].to_numpy(),
        occ_local["y_m"].to_numpy(),
        bins=[xedges, yedges],
        weights=occ_local["residents"].to_numpy(),
    )

    x0 = xmin
    y0 = ymin
    sxy = 1.0 / max(max(span_x, span_y), 1.0)
    zmax = float(max(24.0, np.quantile(heights[heights > 0], 0.99) * 1.05)) if np.any(heights > 0) else 24.0
    zscale = 0.72 / max(zmax, 1.0)

    def _iso(x: float, y: float, z: float = 0.0) -> tuple[float, float]:
        xn = (x - x0) * sxy
        yn = (y - y0) * sxy
        u = (xn - yn) * 0.866
        v = (xn + yn) * 0.50 - z * zscale
        return u, v

    def _add_poly(points, face, edge, lw=0.3, alpha=1.0, zorder=1):
        poly = Polygon(points, closed=True, facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha, joinstyle="round")
        poly.set_zorder(zorder)
        ax.add_patch(poly)

    def _draw_geom_faces(gdf: gpd.GeoDataFrame, facecolor, edgecolor, lw, zorder):
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            polys = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
            for poly in polys:
                x, y = poly.exterior.xy
                pts = [_iso(float(xx), float(yy), 0.0) for xx, yy in zip(x, y)]
                _add_poly(pts, facecolor, edgecolor, lw=lw, alpha=1.0, zorder=zorder)

    _draw_geom_faces(tracts_3857, facecolor="#F6F1E8", edgecolor="#DED5C6", lw=0.42, zorder=0.5)
    if not selected_3857.empty:
        _draw_geom_faces(selected_3857, facecolor="#FCE7C0", edgecolor="#C97A1D", lw=0.9, zorder=0.9)

    def _draw_lines(gdf: gpd.GeoDataFrame, color: str, lw: float, zorder: float) -> None:
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            geoms = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
            for part in geoms:
                x, y = part.xy
                pts = np.array([_iso(float(xx), float(yy), 0.0) for xx, yy in zip(x, y)])
                ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=0.95, zorder=zorder)

    roads_primary = roads_3857.loc[roads_3857["MTFCC"].isin(["S1400", "S1740"])]
    roads_bg = roads_3857.loc[~roads_3857["MTFCC"].isin(["S1400", "S1740"])]
    if len(roads_bg):
        _draw_lines(roads_bg, color="#E1D9CB", lw=0.32, zorder=1.0)
    if len(roads_primary):
        _draw_lines(roads_primary, color="#C8B9A0", lw=0.50, zorder=1.1)

    vals = heights[heights > 0]
    vmax = max(18.0, float(np.quantile(vals, 0.98))) if vals.size else 18.0
    cells = []
    for i in range(nx):
        for j in range(ny):
            h = float(heights[i, j])
            if h <= 0:
                continue
            x1, x2 = xedges[i], xedges[i + 1]
            y1, y2 = yedges[j], yedges[j + 1]
            top = [_iso(x1, y1, h), _iso(x2, y1, h), _iso(x2, y2, h), _iso(x1, y2, h)]
            base = [_iso(x1, y1, 0.0), _iso(x2, y1, 0.0), _iso(x2, y2, 0.0), _iso(x1, y2, 0.0)]
            cells.append((i + j, h, base, top))
    cells.sort(key=lambda t: (t[0], t[1]))
    for _, h, base, top in cells:
        t = np.clip(h / vmax, 0.32, 1.0)
        top_c = cm.YlOrBr(t)
        left_c = tuple(np.clip(np.array(top_c[:3]) * 0.84, 0, 1)) + (0.98,)
        right_c = tuple(np.clip(np.array(top_c[:3]) * 0.68, 0, 1)) + (0.98,)
        left_face = [base[3], top[3], top[0], base[0]]
        right_face = [base[1], top[1], top[2], base[2]]
        _add_poly(left_face, left_c, edge="none", lw=0.0, alpha=0.98, zorder=2.0)
        _add_poly(right_face, right_c, edge="none", lw=0.0, alpha=0.98, zorder=2.05)
        _add_poly(top, top_c, edge=(1, 1, 1, 0), lw=0.0, alpha=0.99, zorder=2.1)

    if not selected_3857.empty:
        for geom in selected_3857.boundary.geometry:
            geoms = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
            for part in geoms:
                x, y = part.xy
                pts = np.array([_iso(float(xx), float(yy), 0.02 * zmax) for xx, yy in zip(x, y)])
                ax.plot(pts[:, 0], pts[:, 1], color="#C97A1D", lw=1.1, alpha=1.0, zorder=2.5)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("auto")

    corners = [_iso(xmin, ymin, 0.0), _iso(xmax, ymin, 0.0), _iso(xmax, ymax, zmax), _iso(xmin, ymax, zmax)]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    ax.set_xlim(min(xs) - 0.03, max(xs) + 0.03)
    ax.set_ylim(min(ys) - 0.02, max(ys) + 0.04)

    ax.text(
        0.98,
        0.92,
        "Residents per\nhome point",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#6E5635",
    )


def _plot_home_overview(
    *,
    tracts: gpd.GeoDataFrame,
    home_points_sample: pd.DataFrame,
    home_tract_validation: pd.DataFrame,
    home_bg_validation: pd.DataFrame,
    out_path: Path,
    home_ref_label: str,
    home_ref_y_label: str,
) -> None:
    with paper_style():
        fig = plt.figure(figsize=(13.4, 8.4))
        gs = fig.add_gridspec(
            2,
            3,
            left=0.050,
            right=0.980,
            top=0.965,
            bottom=0.09,
            wspace=0.18,
            hspace=0.26,
        )

        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])
        ax_d = fig.add_subplot(gs[1, 0])
        ax_e = fig.add_subplot(gs[1, 1])
        ax_f = fig.add_subplot(gs[1, 2])

        acs_vals = pd.to_numeric(tracts["acs_home_count"], errors="coerce")
        vmax_acs = float(np.nanmax(acs_vals)) if np.isfinite(acs_vals).any() else 1.0
        home_vals = pd.to_numeric(tracts["home_count"], errors="coerce")
        vmax_home = float(np.nanmax(home_vals)) if np.isfinite(home_vals).any() else 1.0
        shared_home_vmax = max(vmax_acs, vmax_home)
        ticks_acs = [t for t in [2000, 5000, 8000] if t <= vmax_acs]
        if len(ticks_acs) < 2:
            positive_acs = acs_vals[acs_vals > 0]
            ticks_acs = [float(np.nanmin(positive_acs)), vmax_acs] if len(positive_acs) else [1.0, vmax_acs]
        _choropleth_panel(
            ax_a,
            tracts,
            "acs_home_count",
            "YlOrBr",
            "ACS residents per tract",
            use_log=False,
            ticks=ticks_acs,
            colorbar_box=(0.18, -0.11, 0.44, 0.040),
        )

        home_vals = pd.to_numeric(tracts["home_count"], errors="coerce")
        vmax_home = float(np.nanmax(home_vals)) if np.isfinite(home_vals).any() else 1.0
        ticks_home = [t for t in [2000, 5000, 8000] if t <= vmax_home]
        if len(ticks_home) < 2:
            ticks_home = [float(np.nanmin(home_vals[home_vals > 0])), vmax_home]
        _choropleth_panel(
            ax_b,
            tracts,
            "home_count",
            "YlGnBu",
            "Residents per tract",
            use_log=False,
            ticks=ticks_home,
            colorbar_box=(0.18, -0.11, 0.44, 0.040),
        )

        _panel_label(ax_a, "a", x=-0.10, y=1.03)
        _panel_label(ax_b, "b", x=-0.10, y=1.03)
        _home_density_panel(
            ax_c,
            tracts,
            home_points_sample,
            cmap="Blues",
            label="Sampled home-point density",
            ticks=[1, 10, 100],
            colorbar_box=(0.18, -0.11, 0.44, 0.040),
        )
        _panel_label(ax_c, "c", x=-0.10, y=1.03)

        _home_distribution_panel(ax_d, home_tract_validation, home_ref_label=home_ref_label)
        _home_validation_panel(
            ax_e,
            home_tract_validation,
            bg_df=home_bg_validation,
            y_label=home_ref_y_label,
        )
        _home_rank_panel(ax_f, home_bg_validation)

        _panel_label(ax_d, "d", x=-0.10, y=1.03)
        _panel_label(ax_e, "e", x=-0.10, y=1.03)
        _panel_label(ax_f, "f", x=-0.10, y=1.03)

        save_figure(fig, out_path)
        plt.close(fig)


def _plot_work_overview(
    *,
    tracts: gpd.GeoDataFrame,
    work_points_sample: pd.DataFrame,
    work_tract_validation: pd.DataFrame,
    commute_bins: pd.DataFrame,
    out_path: Path,
) -> None:
    with paper_style():
        fig = plt.figure(figsize=(13.4, 8.4))
        gs = fig.add_gridspec(
            2,
            3,
            left=0.050,
            right=0.980,
            top=0.965,
            bottom=0.09,
            wspace=0.18,
            hspace=0.26,
        )
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])
        ax_d = fig.add_subplot(gs[1, 0])
        ax_e = fig.add_subplot(gs[1, 1])
        ax_f = fig.add_subplot(gs[1, 2])

        emp_vals = pd.to_numeric(tracts["wac_work_count"], errors="coerce")
        vmax_emp = float(np.nanmax(emp_vals)) if np.isfinite(emp_vals).any() else 1.0
        ticks_emp = [t for t in [1000, 10000] if t <= vmax_emp]
        if len(ticks_emp) < 2:
            positive_emp = emp_vals[emp_vals > 0]
            ticks_emp = [float(np.nanmin(positive_emp)), vmax_emp] if len(positive_emp) else [1.0, vmax_emp]
        _choropleth_panel(
            ax_a,
            tracts,
            "wac_work_count",
            "OrRd",
            "LODES WAC workers per tract",
            use_log=False,
            ticks=ticks_emp,
            colorbar_box=(0.18, -0.11, 0.44, 0.040),
            formatter=FuncFormatter(_fmt_compact_count),
            colorbar_tick_labelsize=8.6,
            colorbar_label_size=9.8,
            colorbar_outline_lw=0.42,
        )
        syn_vals = pd.to_numeric(tracts["work_count"], errors="coerce")
        vmax_syn = float(np.nanmax(syn_vals)) if np.isfinite(syn_vals).any() else 1.0
        ticks_syn = [t for t in [1000, 10000] if t <= vmax_syn]
        if len(ticks_syn) < 2:
            positive_syn = syn_vals[syn_vals > 0]
            ticks_syn = [float(np.nanmin(positive_syn)), vmax_syn] if len(positive_syn) else [1.0, vmax_syn]
        _choropleth_panel(
            ax_b,
            tracts,
            "work_count",
            "YlOrBr",
            "Workers per destination tract",
            use_log=False,
            ticks=ticks_syn,
            colorbar_box=(0.18, -0.11, 0.44, 0.040),
            formatter=FuncFormatter(_fmt_compact_count),
            colorbar_tick_labelsize=8.6,
            colorbar_label_size=9.8,
            colorbar_outline_lw=0.42,
        )
        _home_density_panel(
            ax_c,
            tracts,
            work_points_sample,
            cmap="OrRd",
            label="Sampled work-point density",
            ticks=[1, 10, 100],
            colorbar_box=(0.18, -0.11, 0.44, 0.040),
            colorbar_tick_labelsize=8.6,
            colorbar_label_size=9.8,
            colorbar_outline_lw=0.42,
        )

        _share_distribution_panel(
            ax_d,
            work_tract_validation,
            left_col="left_share",
            right_col="right_share",
            left_label="Synthetic",
            right_label="Mobility",
            left_color="#D8872E",
            right_color="#9C2F10",
            x_label="Destination-tract share",
            q_label_prefix="Synthetic",
        )
        _share_scatter_panel(
            ax_e,
            work_tract_validation,
            left_col="left_share",
            right_col="right_share",
            x_label="Synthetic work-tract share",
            y_label="Mobility workplace share",
            point_color="#C45A00",
        )
        x = np.arange(commute_bins.shape[0])
        labels = [
            f"{int(l)}-{int(r)}" if np.isfinite(r) else f"{int(l)}+"
            for l, r in zip(commute_bins["bin_left_km"], commute_bins["bin_right_km"])
        ]
        syn = commute_bins["synthetic_share"].to_numpy()
        mob = commute_bins["mobility_share"].to_numpy()
        line_syn, = ax_f.plot(x, syn, color="#9A7D64", lw=1.9, marker="o", ms=3.8)
        line_mob, = ax_f.plot(x, mob, color="#9AB6C4", lw=1.9, marker="o", ms=3.8)
        ax_f.set_xticks(x)
        ax_f.set_xticklabels(labels, rotation=35, ha="right")
        ax_f.set_xlabel("Commute distance bin (km)")
        ax_f.set_ylabel("Share")
        ax_f.spines["top"].set_visible(False)
        ax_f.spines["right"].set_visible(False)
        ymax = max(float(syn.max()), float(mob.max())) * 1.16
        ax_f.set_ylim(0.0, ymax)
        ax_f.set_xlim(-0.35, x[-1] + 1.05)
        ax_f.legend(
            [line_syn, line_mob],
            ["Synthetic", "Mobility"],
            loc="upper left",
            frameon=False,
            fontsize=9.4,
            handlelength=1.6,
            borderpad=0.2,
            labelspacing=0.35,
        )

        _panel_label(ax_a, "a", x=-0.10, y=1.03)
        _panel_label(ax_b, "b", x=-0.10, y=1.03)
        _panel_label(ax_c, "c", x=-0.10, y=1.03)
        _panel_label(ax_d, "d", x=-0.10, y=1.03)
        _panel_label(ax_e, "e", x=-0.10, y=1.03)
        _panel_label(ax_f, "f", x=-0.10, y=1.03)

        save_figure(fig, out_path)
        plt.close(fig)


def _plot_spatial_product_overview(
    *,
    tracts: gpd.GeoDataFrame,
    home_points_sample: pd.DataFrame,
    work_points_sample: pd.DataFrame,
    out_path: Path,
) -> None:
    with paper_style():
        fig = plt.figure(figsize=(13.4, 8.4))
        gs = fig.add_gridspec(
            2,
            3,
            left=0.050,
            right=0.980,
            top=0.965,
            bottom=0.09,
            wspace=0.18,
            hspace=0.26,
        )

        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])
        ax_d = fig.add_subplot(gs[1, 0])
        ax_e = fig.add_subplot(gs[1, 1])
        ax_f = fig.add_subplot(gs[1, 2])

        acs_vals = pd.to_numeric(tracts["acs_home_count"], errors="coerce")
        vmax_acs = float(np.nanmax(acs_vals)) if np.isfinite(acs_vals).any() else 1.0
        home_vals = pd.to_numeric(tracts["home_count"], errors="coerce")
        vmax_home = float(np.nanmax(home_vals)) if np.isfinite(home_vals).any() else 1.0
        shared_home_vmax = max(vmax_acs, vmax_home)
        ticks_acs = [t for t in [2000, 5000, 8000] if t <= vmax_acs]
        if len(ticks_acs) < 2:
            positive_acs = acs_vals[acs_vals > 0]
            ticks_acs = [float(np.nanmin(positive_acs)), vmax_acs] if len(positive_acs) else [1.0, vmax_acs]
        _choropleth_panel(
            ax_a,
            tracts,
            "acs_home_count",
            "YlOrBr",
            "ACS residents per tract",
            use_log=False,
            ticks=ticks_acs,
            colorbar_box=(0.18, -0.11, 0.44, 0.040),
            vmin=0.0,
            vmax=shared_home_vmax,
        )
        _choropleth_panel(
            ax_b,
            tracts,
            "home_count",
            "YlOrBr",
            "Synthetic residents per tract",
            use_log=False,
            ticks=ticks_acs,
            colorbar_box=(0.18, -0.11, 0.44, 0.040),
            vmin=0.0,
            vmax=shared_home_vmax,
        )
        _home_density_panel(
            ax_c,
            tracts,
            home_points_sample,
            cmap="Blues",
            label="Sampled home-point density",
            ticks=[1, 10, 100],
            colorbar_box=(0.18, -0.11, 0.44, 0.040),
        )

        emp_vals = pd.to_numeric(tracts["wac_work_count"], errors="coerce")
        vmax_emp = float(np.nanmax(emp_vals)) if np.isfinite(emp_vals).any() else 1.0
        syn_vals = pd.to_numeric(tracts["work_count"], errors="coerce")
        vmax_syn = float(np.nanmax(syn_vals)) if np.isfinite(syn_vals).any() else 1.0
        shared_work_vmax = max(vmax_emp, vmax_syn)
        ticks_emp = [t for t in [1000, 10000] if t <= vmax_emp]
        if len(ticks_emp) < 2:
            positive_emp = emp_vals[emp_vals > 0]
            ticks_emp = [float(np.nanmin(positive_emp)), vmax_emp] if len(positive_emp) else [1.0, vmax_emp]
        _choropleth_panel(
            ax_d,
            tracts,
            "wac_work_count",
            "OrRd",
            "LODES WAC workers per tract",
            use_log=False,
            ticks=ticks_emp,
            colorbar_box=(0.18, -0.11, 0.44, 0.040),
            formatter=FuncFormatter(_fmt_compact_count),
            colorbar_tick_labelsize=8.6,
            colorbar_label_size=9.8,
            colorbar_outline_lw=0.42,
            vmin=0.0,
            vmax=shared_work_vmax,
        )
        _choropleth_panel(
            ax_e,
            tracts,
            "work_count",
            "OrRd",
            "Synthetic workers per destination tract",
            use_log=False,
            ticks=ticks_emp,
            colorbar_box=(0.18, -0.11, 0.44, 0.040),
            formatter=FuncFormatter(_fmt_compact_count),
            colorbar_tick_labelsize=8.6,
            colorbar_label_size=9.8,
            colorbar_outline_lw=0.42,
            vmin=0.0,
            vmax=shared_work_vmax,
        )
        _home_density_panel(
            ax_f,
            tracts,
            work_points_sample,
            cmap="OrRd",
            label="Sampled work-point density",
            ticks=[1, 10, 100],
            colorbar_box=(0.18, -0.11, 0.44, 0.040),
            colorbar_tick_labelsize=8.6,
            colorbar_label_size=9.8,
            colorbar_outline_lw=0.42,
        )

        for ax, lab in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], list("abcdef")):
            _panel_label(ax, lab, x=-0.10, y=1.03)

        save_figure(fig, out_path)
        plt.close(fig)


def _plot_spatial_validation_overview(
    *,
    home_tract_validation: pd.DataFrame,
    home_bg_validation: pd.DataFrame,
    work_tract_validation: pd.DataFrame,
    commute_bins: pd.DataFrame,
    out_path: Path,
    home_ref_label: str,
    home_ref_y_label: str,
) -> None:
    with paper_style():
        fig = plt.figure(figsize=(13.4, 8.4))
        gs = fig.add_gridspec(
            2,
            3,
            left=0.050,
            right=0.980,
            top=0.965,
            bottom=0.09,
            wspace=0.18,
            hspace=0.26,
        )

        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])
        ax_d = fig.add_subplot(gs[1, 0])
        ax_e = fig.add_subplot(gs[1, 1])
        ax_f = fig.add_subplot(gs[1, 2])

        _home_distribution_panel(ax_a, home_tract_validation, home_ref_label=home_ref_label)
        _home_validation_panel(
            ax_b,
            home_tract_validation,
            bg_df=home_bg_validation,
            y_label=home_ref_y_label,
        )
        _home_rank_panel(ax_c, home_bg_validation)

        _share_distribution_panel(
            ax_d,
            work_tract_validation,
            left_col="left_share",
            right_col="right_share",
            left_label="Synthetic",
            right_label="Mobility",
            left_color="#D8872E",
            right_color="#9C2F10",
            x_label="Destination-tract share",
            q_label_prefix="Synthetic",
        )
        _share_scatter_panel(
            ax_e,
            work_tract_validation,
            left_col="left_share",
            right_col="right_share",
            x_label="Synthetic work-tract share",
            y_label="Mobility workplace share",
            point_color="#C45A00",
        )

        x = np.arange(commute_bins.shape[0])
        labels = [
            f"{int(l)}-{int(r)}" if np.isfinite(r) else f"{int(l)}+"
            for l, r in zip(commute_bins["bin_left_km"], commute_bins["bin_right_km"])
        ]
        syn = commute_bins["synthetic_share"].to_numpy()
        mob = commute_bins["mobility_share"].to_numpy()
        line_syn, = ax_f.plot(x, syn, color="#9A7D64", lw=1.9, marker="o", ms=3.8)
        line_mob, = ax_f.plot(x, mob, color="#9AB6C4", lw=1.9, marker="o", ms=3.8)
        ax_f.set_xticks(x)
        ax_f.set_xticklabels(labels, rotation=35, ha="right")
        ax_f.set_xlabel("Commute distance bin (km)")
        ax_f.set_ylabel("Share")
        ax_f.spines["top"].set_visible(False)
        ax_f.spines["right"].set_visible(False)
        ymax = max(float(syn.max()), float(mob.max())) * 1.16
        ax_f.set_ylim(0.0, ymax)
        ax_f.set_xlim(-0.35, x[-1] + 1.05)
        ax_f.legend(
            [line_syn, line_mob],
            ["Synthetic", "Mobility"],
            loc="upper left",
            frameon=False,
            fontsize=9.4,
            handlelength=1.6,
            borderpad=0.2,
            labelspacing=0.35,
        )

        for ax, lab in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], list("abcdef")):
            _panel_label(ax, lab, x=-0.10, y=1.03)

        save_figure(fig, out_path)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(prog="viz_phase3_detroit_overviews")
    parser.add_argument("--tract_zip", type=Path, default=DEFAULT_TRACT_ZIP)
    parser.add_argument("--road_zip", type=Path, default=DEFAULT_ROAD_ZIP)
    parser.add_argument("--metrics_dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--sample_dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--micro_manifest", type=Path, default=DEFAULT_MICRO_MANIFEST)
    parser.add_argument("--home_deckgl_crop", type=Path, default=DEFAULT_HOME_DECKGL_CROP)
    parser.add_argument("--acs_b01001_csv", type=Path, default=DEFAULT_ACS_B01001)
    parser.add_argument("--wac_s000_csv", type=Path, default=DEFAULT_WAC_S000)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--home_ref_label", default="Mobility")
    parser.add_argument("--home_ref_y_label", default="Mobility residential share")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    home_tract = pd.read_csv(args.metrics_dir / "home_tract_comparison.csv", dtype={"tract_geoid": str})
    work_tract = pd.read_csv(args.metrics_dir / "work_tract_comparison.csv", dtype={"tract_geoid": str})
    commute_bins = pd.read_csv(args.metrics_dir / "commute_distance_bins.csv")
    acs_b01001 = pd.read_csv(args.acs_b01001_csv, dtype={"GEOID": str})
    tract_ids = sorted(set(home_tract["tract_geoid"].astype(str)) | set(work_tract["tract_geoid"].astype(str)))
    tracts = _study_tracts(args.tract_zip, tract_ids)
    roads = _prepare_roads(args.road_zip, tracts.crs)
    micro_meta = _micro_meta(args.micro_manifest)
    home_points_overview = pd.read_csv(args.sample_dir / "home_sample_160k.csv")
    work_points_overview = pd.read_csv(args.sample_dir / "work_sample_160k.csv")
    home_bounds = tuple(float(v) for v in micro_meta["home_example"]["bounds"])
    tracts = tracts.merge(
        home_tract[["tract_geoid", "synthetic_count"]].rename(columns={"synthetic_count": "home_count"}),
        on="tract_geoid",
        how="left",
    )
    tracts = tracts.merge(
        work_tract[["tract_geoid", "synthetic_count"]].rename(columns={"synthetic_count": "work_count"}),
        on="tract_geoid",
        how="left",
    )
    tracts["home_count"] = pd.to_numeric(tracts["home_count"], errors="coerce").fillna(0.0)
    tracts["work_count"] = pd.to_numeric(tracts["work_count"], errors="coerce").fillna(0.0)

    home_bg = pd.read_csv(args.metrics_dir / "home_bg_spearman_by_tract.csv", dtype={"tract_geoid": str})
    acs_home = acs_b01001[["GEOID", "B01001_001E"]].rename(
        columns={"GEOID": "tract_geoid", "B01001_001E": "acs_home_count"}
    )
    acs_home["tract_geoid"] = acs_home["tract_geoid"].astype(str)
    acs_home["acs_home_count"] = pd.to_numeric(acs_home["acs_home_count"], errors="coerce").fillna(0.0)
    tracts = tracts.merge(
        acs_home,
        on="tract_geoid",
        how="left",
    )
    tracts["acs_home_count"] = pd.to_numeric(tracts["acs_home_count"], errors="coerce").fillna(0.0)

    wac_block = load_lodes_rac_or_wac(path=args.wac_s000_csv, geocode_col="w_geocode", usecols=["w_geocode", "C000"])
    tract_wac = aggregate_lodes_wac_to_tract(wac_block)[["tract_geoid", "C000"]].rename(columns={"C000": "wac_work_count"})
    tract_wac = tract_wac.loc[tract_wac["tract_geoid"].isin(set(tract_ids))].copy()
    tracts = tracts.merge(tract_wac, on="tract_geoid", how="left")
    tracts["wac_work_count"] = pd.to_numeric(tracts["wac_work_count"], errors="coerce").fillna(0.0)

    work_tract_lodes = (
        tracts[["tract_geoid", "work_count", "wac_work_count"]]
        .rename(columns={"work_count": "synthetic_count", "wac_work_count": "empirical_count"})
        .copy()
    )
    syn_total = float(work_tract_lodes["synthetic_count"].sum())
    emp_total = float(work_tract_lodes["empirical_count"].sum())
    work_tract_lodes["left_share"] = work_tract_lodes["synthetic_count"] / syn_total if syn_total > 0 else 0.0
    work_tract_lodes["right_share"] = work_tract_lodes["empirical_count"] / emp_total if emp_total > 0 else 0.0

    _plot_home_overview(
        tracts=tracts,
        home_points_sample=home_points_overview,
        home_tract_validation=home_tract,
        home_bg_validation=home_bg,
        out_path=args.outdir / "home_overview.png",
        home_ref_label=str(args.home_ref_label),
        home_ref_y_label=str(args.home_ref_y_label),
    )
    _home_rank_validation_figure(
        bg_df=home_bg,
        out_path=args.outdir / "home_rank_validation.png",
    )
    _plot_work_overview(
        tracts=tracts,
        work_points_sample=work_points_overview,
        work_tract_validation=work_tract_lodes,
        commute_bins=commute_bins,
        out_path=args.outdir / "work_overview.png",
    )
    _plot_spatial_product_overview(
        tracts=tracts,
        home_points_sample=home_points_overview,
        work_points_sample=work_points_overview,
        out_path=args.outdir / "spatial_product_overview.png",
    )
    _plot_spatial_validation_overview(
        home_tract_validation=home_tract,
        home_bg_validation=home_bg,
        work_tract_validation=work_tract,
        commute_bins=commute_bins,
        out_path=args.outdir / "spatial_validation_overview.png",
        home_ref_label=str(args.home_ref_label),
        home_ref_y_label=str(args.home_ref_y_label),
    )

    manifest = {
        "tract_zip": str(args.tract_zip),
        "road_zip": str(args.road_zip),
        "metrics_dir": str(args.metrics_dir),
        "sample_dir": str(args.sample_dir),
        "micro_manifest": str(args.micro_manifest),
        "home_focus_bounds": list(home_bounds),
        "artifacts": {
            "home_overview_png": str(args.outdir / "home_overview.png"),
            "home_rank_validation_png": str(args.outdir / "home_rank_validation.png"),
            "work_overview_png": str(args.outdir / "work_overview.png"),
            "spatial_product_overview_png": str(args.outdir / "spatial_product_overview.png"),
            "spatial_validation_overview_png": str(args.outdir / "spatial_validation_overview.png"),
        },
    }
    (args.outdir / "overview_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote {args.outdir / 'home_overview.png'}")
    print(f"[ok] wrote {args.outdir / 'home_rank_validation.png'}")
    print(f"[ok] wrote {args.outdir / 'work_overview.png'}")
    print(f"[ok] wrote {args.outdir / 'spatial_product_overview.png'}")
    print(f"[ok] wrote {args.outdir / 'spatial_validation_overview.png'}")
    print(f"[ok] wrote {args.outdir / 'overview_manifest.json'}")


if __name__ == "__main__":
    main()
