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
from shapely.geometry import box

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.plot_style import paper_style, save_figure


PROJECT_ROOT = Path("/Users/jinlin/Desktop/Project/Synthetic_City")
DEFAULT_OUTDIR = PROJECT_ROOT / "figures" / "phase3_detroit_metro_latest"
DEFAULT_TRACT_ZIP = PROJECT_ROOT / "dataset" / "cache" / "geo" / "tl_2023_26_tract.zip"
DEFAULT_ROAD_ZIP = PROJECT_ROOT / "dataset" / "cache" / "geo" / "MI_road_cleaned.shp.zip"
DEFAULT_HOME_SAMPLE = PROJECT_ROOT / "outputs" / "_sync_phase3_detroit_hcenter_20260330" / "micro_examples" / "home_all_points.csv"
DEFAULT_WORK_SAMPLE = PROJECT_ROOT / "outputs" / "_sync_phase3_detroit_hcenter_20260330" / "micro_examples" / "work_all_points.csv"

# Chosen from the current Detroit best run as representative, not most extreme:
# - a dense residential tract in the Wayne County grid
# - a strong employment tract in the Oakland/Detroit core
DEFAULT_HOME_TRACT = "26163503300"
DEFAULT_WORK_TRACT = "26163533900"


def _panel_label(ax, text: str, x: float = 0.02, y: float = 0.98) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.15},
    )


def _despine(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")


def _read_geodata(path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".zip":
        return gpd.read_file(f"zip://{path}")
    return gpd.read_file(path)


def _study_tracts(path: Path) -> gpd.GeoDataFrame:
    gdf = _read_geodata(path)
    gdf["tract_geoid"] = gdf["GEOID"].astype(str)
    return gdf


def _prepare_roads(path: Path, target_crs) -> gpd.GeoDataFrame:
    roads = _read_geodata(path)
    keep = [c for c in ["MTFCC", "geometry"] if c in roads.columns]
    roads = roads[keep].copy()
    if roads.crs != target_crs:
        roads = roads.to_crs(target_crs)
    roads["MTFCC"] = roads["MTFCC"].astype(str)
    return roads


def _bbox_around_tract(tract: gpd.GeoDataFrame, radius_m: float) -> tuple[float, float, float, float]:
    geom = tract.to_crs(3857).geometry.iloc[0]
    cx, cy = geom.centroid.x, geom.centroid.y
    bb = box(cx - radius_m, cy - radius_m, cx + radius_m, cy + radius_m)
    bb_native = gpd.GeoSeries([bb], crs=3857).to_crs(tract.crs).total_bounds.tolist()
    return tuple(float(v) for v in bb_native)


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


def _subset_points(df: pd.DataFrame, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    xmin, ymin, xmax, ymax = bounds
    return df.loc[
        (df["x"] >= xmin)
        & (df["x"] <= xmax)
        & (df["y"] >= ymin)
        & (df["y"] <= ymax)
    ].copy()


def _plot_locator(ax, tracts: gpd.GeoDataFrame, selected: gpd.GeoDataFrame, bounds: tuple[float, float, float, float], facecolor: str) -> None:
    tracts.plot(ax=ax, color="#F5F5F5", edgecolor="#D8D8D8", linewidth=0.18)
    selected.plot(ax=ax, color=facecolor, edgecolor=facecolor, linewidth=0.8)
    bbox_gdf = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=tracts.crs)
    bbox_gdf.boundary.plot(ax=ax, color="#5A5A5A", linewidth=1.0)
    _despine(ax)


def _plot_zoom(
    ax,
    tracts_zoom: gpd.GeoDataFrame,
    selected: gpd.GeoDataFrame,
    roads_zoom: gpd.GeoDataFrame,
    pts_zoom: pd.DataFrame,
    point_color: str,
    primary_mtfcc: list[str],
    bounds: tuple[float, float, float, float],
) -> None:
    tracts_zoom.plot(ax=ax, color="#FBFBFB", edgecolor="#D5D5D5", linewidth=0.28, zorder=0)
    roads_bg = roads_zoom.loc[~roads_zoom["MTFCC"].isin(primary_mtfcc)]
    roads_fg = roads_zoom.loc[roads_zoom["MTFCC"].isin(primary_mtfcc)]
    if len(roads_bg):
        roads_bg.plot(ax=ax, color="#D9D9D9", linewidth=0.38, alpha=0.75, zorder=1)
    if len(roads_fg):
        roads_fg.plot(ax=ax, color="#7A7A7A", linewidth=0.65, alpha=0.92, zorder=2)
    if len(pts_zoom):
        ax.scatter(
            pts_zoom["x"].to_numpy(),
            pts_zoom["y"].to_numpy(),
            s=7.5,
            c=point_color,
            alpha=0.45,
            linewidths=0.0,
            zorder=3,
        )
    selected.boundary.plot(ax=ax, color=point_color, linewidth=1.2, zorder=4)
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    _despine(ax)


def _occupancy_category(residents: pd.Series) -> pd.Categorical:
    bins = pd.cut(
        residents,
        bins=[0, 1, 2, 4, np.inf],
        labels=["1", "2", "3-4", "5+"],
        right=True,
    )
    return bins


def _plot_home_colocation_map(
    ax,
    tracts_zoom: gpd.GeoDataFrame,
    selected: gpd.GeoDataFrame,
    roads_zoom: gpd.GeoDataFrame,
    occ: pd.DataFrame,
    bounds: tuple[float, float, float, float],
) -> None:
    occ = occ.copy()
    occ["occ_bin"] = _occupancy_category(occ["residents"])

    tracts_zoom.plot(ax=ax, color="#FCFBF8", edgecolor="#E6DED1", linewidth=0.24, zorder=0)
    selected.plot(ax=ax, color="#F3F7FB", edgecolor="none", zorder=0.3)
    roads_bg = roads_zoom.loc[~roads_zoom["MTFCC"].isin(["S1400", "S1740"])]
    roads_fg = roads_zoom.loc[roads_zoom["MTFCC"].isin(["S1400", "S1740"])]
    if len(roads_bg):
        roads_bg.plot(ax=ax, color="#EEE7DC", linewidth=0.28, alpha=0.75, zorder=1)
    if len(roads_fg):
        roads_fg.plot(ax=ax, color="#C8D5E3", linewidth=0.44, alpha=0.80, zorder=1.2)

    ax.scatter(
        occ["x"].to_numpy(),
        occ["y"].to_numpy(),
        s=9,
        c="#C9D9E8",
        alpha=0.42,
        linewidths=0.0,
        zorder=2.0,
    )

    highlight = occ.loc[occ["residents"] >= 5].copy()
    q = highlight["residents"].to_numpy()
    sizes = 16 + 1.55 * np.sqrt(q)
    colors = np.where(
        q >= 12,
        "#0F5F97",
        np.where(q >= 8, "#2F81B7", "#71AED7"),
    )
    ax.scatter(
        highlight["x"].to_numpy(),
        highlight["y"].to_numpy(),
        s=sizes,
        c=colors,
        alpha=0.95,
        linewidths=0.35,
        edgecolors="white",
        zorder=2.6,
    )

    selected.boundary.plot(ax=ax, color="#1E6EA8", linewidth=1.25, zorder=3)
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    _despine(ax)
    ax.text(
        0.03,
        0.06,
        "Background: all used home points\nHighlight: 5+ residents per point",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.9,
        color="#4B5966",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.20},
    )


def _plot_home_occupancy_summary(ax, occ: pd.DataFrame, total_persons: int) -> None:
    cats = ["1", "2", "3-4", "5+"]
    occ = occ.copy()
    occ["occ_bin"] = _occupancy_category(occ["residents"]).astype(str)
    point_share = []
    resident_share = []
    for cat in cats:
        sub = occ.loc[occ["occ_bin"] == cat]
        point_share.append(float(len(sub) / len(occ)) if len(occ) else 0.0)
        resident_share.append(float(sub["residents"].sum() / total_persons) if total_persons > 0 else 0.0)

    x = np.arange(len(cats))
    w = 0.36
    ax.bar(x - w / 2, point_share, width=w, color="#A7CDE8", edgecolor="none", label="Share of used home points")
    ax.bar(x + w / 2, resident_share, width=w, color="#1E6EA8", edgecolor="none", label="Share of residents")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_xlabel("Residents per used home point")
    ax.set_ylabel("Share within selected tract")
    ax.set_ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=8.0,
        handlelength=1.4,
        borderpad=0.2,
        labelspacing=0.35,
    )
    ax.text(
        0.98,
        0.96,
        f"{total_persons:,} residents\n{len(occ):,} used home points",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color="#4D4D4D",
    )


def _make_example(
    *,
    tracts: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    points: pd.DataFrame,
    selected_tract_geoid: str,
    point_color: str,
    primary_mtfcc: list[str],
    radius_m: float,
    out_path: Path,
) -> dict:
    selected = tracts.loc[tracts["tract_geoid"] == str(selected_tract_geoid)].copy()
    if selected.empty:
        raise SystemExit(f"selected tract not found: {selected_tract_geoid}")
    bounds = _bbox_around_tract(selected, radius_m=radius_m)
    tracts_zoom = tracts.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()
    roads_zoom = _clip_to_bbox(roads, bounds)
    pts_zoom = _subset_points(points, bounds)

    with paper_style():
        fig = plt.figure(figsize=(9.2, 4.4))
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[0.9, 1.35],
            left=0.05,
            right=0.985,
            top=0.96,
            bottom=0.07,
            wspace=0.10,
        )
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        _plot_locator(ax_a, tracts, selected, bounds, point_color)
        _plot_zoom(ax_b, tracts_zoom, selected, roads_zoom, pts_zoom, point_color, primary_mtfcc, bounds)
        _panel_label(ax_a, "a")
        _panel_label(ax_b, "b")
        save_figure(fig, out_path)
        plt.close(fig)

    return {
        "selected_tract_geoid": str(selected_tract_geoid),
        "bounds": list(bounds),
        "n_zoom_tracts": int(tracts_zoom.shape[0]),
        "n_zoom_points": int(pts_zoom.shape[0]),
    }


def _make_home_example(
    *,
    tracts: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    points: pd.DataFrame,
    selected_tract_geoid: str,
    radius_m: float,
    out_path: Path,
) -> dict:
    selected = tracts.loc[tracts["tract_geoid"] == str(selected_tract_geoid)].copy()
    if selected.empty:
        raise SystemExit(f"selected tract not found: {selected_tract_geoid}")
    bounds = _bbox_around_tract(selected, radius_m=radius_m)
    tracts_zoom = tracts.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()
    roads_zoom = _clip_to_bbox(roads, bounds)
    pts_zoom = _subset_points(points, bounds)
    pts_zoom_gdf = gpd.GeoDataFrame(
        pts_zoom.copy(),
        geometry=gpd.points_from_xy(pts_zoom["x"], pts_zoom["y"]),
        crs=tracts.crs,
    )
    pts_selected = gpd.sjoin(
        pts_zoom_gdf,
        selected[["tract_geoid", "geometry"]],
        predicate="within",
        how="inner",
    )
    occ = pts_selected.groupby(["x", "y"], as_index=False).size().rename(columns={"size": "residents"})

    with paper_style():
        fig = plt.figure(figsize=(12.0, 4.4))
        gs = fig.add_gridspec(
            1,
            3,
            width_ratios=[0.78, 1.50, 0.95],
            left=0.045,
            right=0.985,
            top=0.955,
            bottom=0.12,
            wspace=0.16,
        )
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])

        _plot_locator(ax_a, tracts, selected, bounds, "#1E6EA8")
        _plot_home_colocation_map(ax_b, tracts_zoom, selected, roads_zoom, occ, bounds)
        _plot_home_occupancy_summary(ax_c, occ, total_persons=len(pts_selected))

        _panel_label(ax_a, "a")
        _panel_label(ax_b, "b")
        _panel_label(ax_c, "c", x=-0.10, y=1.03)
        save_figure(fig, out_path)
        plt.close(fig)

    return {
        "selected_tract_geoid": str(selected_tract_geoid),
        "bounds": list(bounds),
        "n_zoom_tracts": int(tracts_zoom.shape[0]),
        "n_zoom_persons": int(pts_zoom.shape[0]),
        "n_selected_tract_residents": int(len(pts_selected)),
        "n_used_home_points": int(occ.shape[0]),
        "mean_residents_per_used_point": float(occ["residents"].mean()) if len(occ) else 0.0,
        "median_residents_per_used_point": float(occ["residents"].median()) if len(occ) else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="viz_phase3_detroit_micro_examples")
    ap.add_argument("--tract_zip", type=Path, default=DEFAULT_TRACT_ZIP)
    ap.add_argument("--road_zip", type=Path, default=DEFAULT_ROAD_ZIP)
    ap.add_argument("--home_sample", type=Path, default=DEFAULT_HOME_SAMPLE)
    ap.add_argument("--work_sample", type=Path, default=DEFAULT_WORK_SAMPLE)
    ap.add_argument("--home_tract", default=DEFAULT_HOME_TRACT)
    ap.add_argument("--work_tract", default=DEFAULT_WORK_TRACT)
    ap.add_argument("--home_radius_m", type=float, default=1800.0)
    ap.add_argument("--work_radius_m", type=float, default=2200.0)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    tracts = _study_tracts(args.tract_zip)
    roads = _prepare_roads(args.road_zip, tracts.crs)
    home_points = pd.read_csv(args.home_sample)
    work_points = pd.read_csv(args.work_sample)

    home_meta = _make_home_example(
        tracts=tracts,
        roads=roads,
        points=home_points,
        selected_tract_geoid=args.home_tract,
        radius_m=args.home_radius_m,
        out_path=args.outdir / "home_micro_example.png",
    )
    work_meta = _make_example(
        tracts=tracts,
        roads=roads,
        points=work_points,
        selected_tract_geoid=args.work_tract,
        point_color="#C45A00",
        primary_mtfcc=["S1100", "S1200"],
        radius_m=args.work_radius_m,
        out_path=args.outdir / "work_micro_example.png",
    )

    manifest = {
        "tract_zip": str(args.tract_zip),
        "road_zip": str(args.road_zip),
        "home_sample": str(args.home_sample),
        "work_sample": str(args.work_sample),
        "artifacts": {
            "home_micro_example_png": str(args.outdir / "home_micro_example.png"),
            "work_micro_example_png": str(args.outdir / "work_micro_example.png"),
        },
        "home_example": home_meta,
        "work_example": work_meta,
    }
    (args.outdir / "micro_examples_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote {args.outdir / 'home_micro_example.png'}")
    print(f"[ok] wrote {args.outdir / 'work_micro_example.png'}")
    print(f"[ok] wrote {args.outdir / 'micro_examples_manifest.json'}")


if __name__ == "__main__":
    main()
