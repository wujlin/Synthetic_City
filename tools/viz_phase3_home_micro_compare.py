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
from matplotlib.colors import TwoSlopeNorm
from shapely.geometry import box

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.plot_style import FIGSIZE_FULL, add_panel_label, paper_style, save_figure


PROJECT_ROOT = Path("/Users/jinlin/Desktop/Project/Synthetic_City")
DEFAULT_OUTDIR = PROJECT_ROOT / "figures" / "phase3_detroit_metro_latest"
DEFAULT_TRACT_ZIP = PROJECT_ROOT / "dataset" / "cache" / "geo" / "tl_2023_26_tract.zip"
DEFAULT_ROAD_ZIP = PROJECT_ROOT / "dataset" / "cache" / "geo" / "MI_road_cleaned.shp.zip"
DEFAULT_SUBSET_DIR = PROJECT_ROOT / "outputs" / "_sync_phase3_detroit_homecompare_subsets_20260330"
DEFAULT_TRACT = "26163506400"


def _read_geodata(path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".zip":
        return gpd.read_file(f"zip://{path}")
    return gpd.read_file(path)


def _despine_map(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")


def _occupancy_bin(s: pd.Series) -> pd.Categorical:
    return pd.cut(
        s,
        bins=[0, 1, 2, 4, np.inf],
        labels=["1", "2", "3-4", "5+"],
        right=True,
    )


def _plot_locator(ax, tracts: gpd.GeoDataFrame, selected: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> None:
    tracts.plot(ax=ax, color="#F7F7F5", edgecolor="#D9D9D2", linewidth=0.18, zorder=0)
    selected.plot(ax=ax, color="#7AA6D1", edgecolor="#4E82B8", linewidth=0.55, zorder=1)
    gpd.GeoDataFrame(geometry=[box(*bounds)], crs=tracts.crs).boundary.plot(
        ax=ax, color="#2C6BA0", linewidth=0.95, zorder=2
    )
    _despine_map(ax)


def _clip_to_bbox(gdf: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    geom = box(*bounds)
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


def _plot_context(ax, tracts_zoom: gpd.GeoDataFrame, selected: gpd.GeoDataFrame, roads_zoom: gpd.GeoDataFrame) -> None:
    tracts_zoom.plot(ax=ax, color="#FCFBF8", edgecolor="#E4DFD7", linewidth=0.24, zorder=0)
    selected.plot(ax=ax, color="#F4F8FB", edgecolor="none", zorder=0.2)

    local = roads_zoom.loc[roads_zoom["MTFCC"].isin(["S1400", "S1740"])]
    other = roads_zoom.loc[~roads_zoom["MTFCC"].isin(["S1400", "S1740"])]
    if len(other):
        other.plot(ax=ax, color="#E6E0D8", linewidth=0.22, alpha=0.7, zorder=1)
    if len(local):
        local.plot(ax=ax, color="#D4DEE8", linewidth=0.36, alpha=0.82, zorder=1.2)
    selected.boundary.plot(ax=ax, color="#2F79B7", linewidth=1.1, zorder=2)


def _grid_delta(
    *,
    personproxy: pd.DataFrame,
    household: pd.DataFrame,
    selected: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    nx: int = 24,
    ny: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, ymin, xmax, ymax = bounds
    xedges = np.linspace(xmin, xmax, nx + 1)
    yedges = np.linspace(ymin, ymax, ny + 1)

    def _mean_occ(df: pd.DataFrame) -> np.ndarray:
        count, _, _ = np.histogram2d(df["x"], df["y"], bins=[xedges, yedges])
        weight, _, _ = np.histogram2d(df["x"], df["y"], bins=[xedges, yedges], weights=df["n_residents"])
        mean = np.divide(weight, count, out=np.full_like(weight, np.nan, dtype=float), where=count > 0)
        return mean.T

    pp_mean = _mean_occ(personproxy)
    hh_mean = _mean_occ(household)
    delta = hh_mean - pp_mean

    poly = selected.geometry.iloc[0]
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    inside = np.zeros((ny, nx), dtype=bool)
    for iy, y in enumerate(yc):
        for ix, x in enumerate(xc):
            inside[iy, ix] = poly.contains(box(x, y, x, y).centroid) or poly.touches(box(x, y, x, y).centroid)
    delta[~inside] = np.nan
    return xedges, yedges, delta


def _plot_delta_map(
    ax,
    *,
    tracts_zoom: gpd.GeoDataFrame,
    selected: gpd.GeoDataFrame,
    roads_zoom: gpd.GeoDataFrame,
    personproxy: pd.DataFrame,
    household: pd.DataFrame,
    bounds: tuple[float, float, float, float],
):
    _plot_context(ax, tracts_zoom=tracts_zoom, selected=selected, roads_zoom=roads_zoom)
    xedges, yedges, delta = _grid_delta(
        personproxy=personproxy,
        household=household,
        selected=selected,
        bounds=bounds,
    )
    finite = delta[np.isfinite(delta)]
    vmax = float(np.nanquantile(np.abs(finite), 0.95)) if finite.size else 1.0
    vmax = max(vmax, 1.0)
    mesh = ax.pcolormesh(
        xedges,
        yedges,
        delta,
        shading="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        alpha=0.88,
        zorder=2.5,
    )
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    _despine_map(ax)
    return mesh


def _plot_distribution(ax, personproxy: pd.DataFrame, household: pd.DataFrame) -> None:
    bins = ["1", "2", "3-4", "5+"]

    def shares(df: pd.DataFrame) -> list[float]:
        out = []
        cats = _occupancy_bin(df["n_residents"]).astype(str)
        for b in bins:
            out.append(float((cats == b).mean()) if len(df) else 0.0)
        return out

    pp = shares(personproxy)
    hh = shares(household)
    x = np.arange(len(bins))
    w = 0.34
    ax.bar(x - w / 2, pp, width=w, color="#9EC3E3", edgecolor="none", label="Person-proxy")
    ax.bar(x + w / 2, hh, width=w, color="#2E79B6", edgecolor="none", label="Household-aware")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel("Residents per used home point")
    ax.set_ylabel("Share of used home points")
    ax.set_ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, handlelength=1.4, borderpad=0.2)


def main() -> None:
    ap = argparse.ArgumentParser(prog="viz_phase3_home_micro_compare")
    ap.add_argument("--tract_zip", default=str(DEFAULT_TRACT_ZIP))
    ap.add_argument("--road_zip", default=str(DEFAULT_ROAD_ZIP))
    ap.add_argument("--subset_dir", default=str(DEFAULT_SUBSET_DIR))
    ap.add_argument("--tract", default=DEFAULT_TRACT)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = ap.parse_args()

    tract_zip = Path(args.tract_zip).expanduser().resolve()
    road_zip = Path(args.road_zip).expanduser().resolve()
    subset_dir = Path(args.subset_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    tract_geoid = str(args.tract)

    pp_csv = subset_dir / f"personproxy_{tract_geoid}_used_home_points.csv"
    hh_csv = subset_dir / f"household_{tract_geoid}_used_home_points.csv"
    pp_summary_path = subset_dir / f"personproxy_{tract_geoid}_summary.json"
    hh_summary_path = subset_dir / f"household_{tract_geoid}_summary.json"

    personproxy = pd.read_csv(pp_csv)
    household = pd.read_csv(hh_csv)
    pp_summary = json.loads(pp_summary_path.read_text())
    hh_summary = json.loads(hh_summary_path.read_text())
    tracts = _read_geodata(tract_zip)
    tracts["tract_geoid"] = tracts["GEOID"].astype(str)
    selected = tracts.loc[tracts["tract_geoid"] == tract_geoid].copy()
    minx, miny, maxx, maxy = selected.geometry.iloc[0].bounds
    padx = (maxx - minx) * 0.20
    pady = (maxy - miny) * 0.20
    bounds = (minx - padx, miny - pady, maxx + padx, maxy + pady)
    tracts_zoom = _clip_to_bbox(tracts, bounds)
    roads = _read_geodata(road_zip)
    roads["MTFCC"] = roads["MTFCC"].astype(str)
    roads_zoom = _clip_to_bbox(roads, bounds)

    with paper_style():
        fig = plt.figure(figsize=(7.2, 3.2))
        gs = fig.add_gridspec(
            1,
            3,
            width_ratios=[0.72, 1.55, 1.08],
            left=0.06,
            right=0.985,
            bottom=0.20,
            top=0.93,
            wspace=0.22,
        )
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])

        _plot_locator(ax_a, tracts=tracts, selected=selected, bounds=bounds)
        mesh = _plot_delta_map(
            ax_b,
            tracts_zoom=tracts_zoom,
            selected=selected,
            roads_zoom=roads_zoom,
            personproxy=personproxy,
            household=household,
            bounds=bounds,
        )
        _plot_distribution(ax_c, personproxy=personproxy, household=household)

        cbar = fig.colorbar(mesh, ax=ax_b, orientation="horizontal", fraction=0.065, pad=0.08)
        cbar.set_label("Change in residents per used home point\n(household-aware minus person-proxy)", fontsize=8.6)
        cbar.ax.tick_params(labelsize=8.0, length=2.5, colors="#555555")
        cbar.outline.set_linewidth(0.5)

        add_panel_label(ax_a, "a", dx=-26)
        add_panel_label(ax_b, "b", dx=-26)
        add_panel_label(ax_c, "c", dx=-26)

        outdir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, outdir / "home_micro_compare.png")
        plt.close(fig)

    manifest = {
        "selected_tract_geoid": tract_geoid,
        "bounds": list(bounds),
        "personproxy_summary": pp_summary,
        "household_summary": hh_summary,
        "artifact": str(outdir / "home_micro_compare.png"),
    }
    (outdir / "home_micro_compare_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {outdir / 'home_micro_compare.png'}")


if __name__ == "__main__":
    main()
