#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.plot_style import FIGSIZE_FULL, OKABE_ITO, add_panel_label, paper_style, save_figure

_DEFAULT_OUT_DIR = _REPO_ROOT / "figures" / "phase3_spatial_population_latest"


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_geodata(path: pathlib.Path) -> Any:
    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise SystemExit("viz_phase3_household_comparison requires geopandas.") from e

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return gpd.read_parquet(path)
    if suffix == ".zip":
        return gpd.read_file(f"zip://{path}")
    return gpd.read_file(path)


def _ensure_crs(gdf: Any, target_crs: Any) -> Any:
    if target_crs is None:
        return gdf
    if getattr(gdf, "crs", None) is None:
        return gdf.set_crs(target_crs)
    if str(gdf.crs) != str(target_crs):
        return gdf.to_crs(target_crs)
    return gdf


def _clip_to_bounds(gdf: Any, bounds: tuple[float, float, float, float]) -> Any:
    from shapely.geometry import box

    bbox_geom = box(*bounds)
    try:
        idx = list(gdf.sindex.query(bbox_geom, predicate="intersects"))
        clipped = gdf.iloc[idx].copy()
    except Exception:
        clipped = gdf[gdf.geometry.intersects(bbox_geom)].copy()
    if int(clipped.shape[0]) == 0:
        return clipped
    try:
        import geopandas as gpd

        clipper = gpd.GeoDataFrame({"id": [1]}, geometry=[bbox_geom], crs=gdf.crs)
        return gpd.clip(clipped, clipper)
    except Exception:
        return clipped


def _prepare_roads(roads_path: pathlib.Path, target_crs: Any, bounds: tuple[float, float, float, float]) -> Any:
    roads = _read_geodata(roads_path)
    keep_cols = [c for c in ["LINEARID", "MTFCC", "component", "geometry"] if c in roads.columns]
    roads = roads[keep_cols].copy()
    roads = _ensure_crs(roads, target_crs)
    roads["MTFCC"] = roads.get("MTFCC", pd.Series(dtype=str)).astype(str)
    return _clip_to_bounds(roads, bounds)


def _plot_roads(ax: Any, roads: Any) -> None:
    if int(roads.shape[0]) == 0:
        return
    roads = roads.copy()
    roads["mtfcc_class"] = roads["MTFCC"].where(roads["MTFCC"].isin(["S1100", "S1200", "S1400", "S1740"]), "other")
    line_cfg = {
        "S1400": {"color": "#D3D3D3", "linewidth": 0.35, "alpha": 0.9},
        "S1740": {"color": "#E0E0E0", "linewidth": 0.35, "alpha": 0.9},
        "S1100": {"color": "#6C6C6C", "linewidth": 0.6, "alpha": 0.95},
        "S1200": {"color": "#8A8A8A", "linewidth": 0.5, "alpha": 0.95},
        "other": {"color": "#EFEFEF", "linewidth": 0.25, "alpha": 0.7},
    }
    for cls, cfg in line_cfg.items():
        part = roads[roads["mtfcc_class"] == cls]
        if int(part.shape[0]) == 0:
            continue
        part.plot(ax=ax, color=cfg["color"], linewidth=cfg["linewidth"], alpha=cfg["alpha"], zorder=1)


def _despine_map(ax: Any, bounds: tuple[float, float, float, float]) -> None:
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_aspect("equal")


def _load_used_home_points(csv_path: pathlib.Path) -> pd.DataFrame:
    usecols = ["home_candidate_id", "home_x", "home_y"]
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    df = df.dropna(subset=["home_candidate_id", "home_x", "home_y"]).copy()
    df["home_candidate_id"] = df["home_candidate_id"].astype(str)
    used = (
        df.groupby("home_candidate_id", sort=False)
        .agg(x=("home_x", "first"), y=("home_y", "first"), n_persons=("home_candidate_id", "size"))
        .reset_index()
    )
    return used


def _filter_points(points: pd.DataFrame, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    xmin, ymin, xmax, ymax = [float(v) for v in bounds]
    return points[
        (points["x"] >= xmin)
        & (points["x"] <= xmax)
        & (points["y"] >= ymin)
        & (points["y"] <= ymax)
    ].copy()


def _plot_density_panel(
    ax: Any,
    *,
    roads: Any,
    points: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    title: str,
    hist_max: float,
    bins: int = 130,
) -> Any:
    try:
        from matplotlib.colors import LogNorm
    except Exception as e:  # pragma: no cover
        raise SystemExit("matplotlib is required for plotting.") from e

    _plot_roads(ax, roads)
    if not points.empty:
        hist, xedges, yedges = np.histogram2d(
            points["x"].to_numpy(),
            points["y"].to_numpy(),
            bins=[int(bins), int(bins)],
            range=[[bounds[0], bounds[2]], [bounds[1], bounds[3]]],
        )
        hist = hist.T
        hist[hist <= 0] = np.nan
        mesh = ax.pcolormesh(
            xedges,
            yedges,
            hist,
            shading="auto",
            cmap="Blues",
            norm=LogNorm(vmin=1, vmax=max(float(hist_max), 1.0)),
            alpha=0.9,
            zorder=2,
        )
    else:
        mesh = None
    if str(title).strip():
        ax.set_title(title, loc="left", pad=6)
    _despine_map(ax, bounds)
    return mesh


def _draw_sidebar(ax: Any, *, fig: Any, mesh: Any) -> None:
    ax.set_axis_off()
    text_kw = {"transform": ax.transAxes, "ha": "left", "va": "top"}
    line_kw = {"transform": ax.transAxes, "solid_capstyle": "round", "clip_on": False}

    y = 0.96
    ax.text(0.02, y, "Runs", fontweight="bold", fontsize=10.5, **text_kw)
    y -= 0.055
    ax.plot([0.02, 0.22], [y - 0.012, y - 0.012], color=OKABE_ITO["gray"], linewidth=2.1, **line_kw)
    ax.text(0.28, y, "Person-proxy", fontsize=9.5, **text_kw)
    y -= 0.055
    ax.plot([0.02, 0.22], [y - 0.012, y - 0.012], color=OKABE_ITO["blue"], linewidth=2.1, **line_kw)
    ax.text(0.28, y, "Household-aware", fontsize=9.5, **text_kw)

    y -= 0.10
    ax.text(0.02, y, "Road context", fontweight="bold", fontsize=10.5, **text_kw)
    y -= 0.055
    ax.plot([0.02, 0.22], [y - 0.012, y - 0.012], color="#6C6C6C", linewidth=1.2, **line_kw)
    ax.text(0.28, y, "S1100/S1200", fontsize=9.5, **text_kw)
    y -= 0.055
    ax.plot([0.02, 0.22], [y - 0.012, y - 0.012], color="#D3D3D3", linewidth=1.2, **line_kw)
    ax.text(0.28, y, "S1400/S1740", fontsize=9.5, **text_kw)

    if mesh is not None:
        ax.text(0.02, 0.28, "Density", fontweight="bold", fontsize=10.5, **text_kw)
        cax = ax.inset_axes([0.08, 0.05, 0.18, 0.18])
        cbar = fig.colorbar(mesh, cax=cax, orientation="vertical")
        cbar.set_label("Used-home\ndensity", fontsize=9)
        cbar.ax.tick_params(labelsize=8)


def _stats(points: pd.DataFrame) -> dict[str, Any]:
    return {
        "used_home_points": int(points.shape[0]),
        "mean_persons_per_point": float(points["n_persons"].mean()) if int(points.shape[0]) else None,
        "median_persons_per_point": float(points["n_persons"].median()) if int(points.shape[0]) else None,
        "p90_persons_per_point": float(points["n_persons"].quantile(0.9)) if int(points.shape[0]) else None,
        "max_persons_per_point": int(points["n_persons"].max()) if int(points.shape[0]) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="viz_phase3_household_comparison")
    ap.add_argument("--old_run_dir", required=True)
    ap.add_argument("--new_run_dir", required=True)
    ap.add_argument("--roads_path", required=True)
    ap.add_argument("--focus_bounds", nargs=4, type=float, required=True, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    ap.add_argument("--out_dir", default=str(_DEFAULT_OUT_DIR))
    ap.add_argument("--bins", type=int, default=130)
    args = ap.parse_args()

    old_run_dir = pathlib.Path(args.old_run_dir).expanduser().resolve()
    new_run_dir = pathlib.Path(args.new_run_dir).expanduser().resolve()
    roads_path = pathlib.Path(args.roads_path).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    bounds = tuple(float(v) for v in args.focus_bounds)

    old_summary = _read_json(old_run_dir / "run_summary.json")
    new_summary = _read_json(new_run_dir / "run_summary.json")
    old_csv = pathlib.Path(old_summary["summary_json"]).resolve().parent.parent / "synthetic" / "person_locations.csv"
    new_csv = pathlib.Path(new_summary["summary_json"]).resolve().parent.parent / "synthetic" / "person_locations.csv"

    old_points = _load_used_home_points(old_csv)
    new_points = _load_used_home_points(new_csv)
    old_focus = _filter_points(old_points, bounds)
    new_focus = _filter_points(new_points, bounds)

    roads = _prepare_roads(roads_path, 4326, bounds)
    all_focus = pd.concat([old_focus[["x", "y"]], new_focus[["x", "y"]]], ignore_index=True)
    hist_all, _, _ = np.histogram2d(
        all_focus["x"].to_numpy() if not all_focus.empty else np.array([]),
        all_focus["y"].to_numpy() if not all_focus.empty else np.array([]),
        bins=[int(args.bins), int(args.bins)],
        range=[[bounds[0], bounds[2]], [bounds[1], bounds[3]]],
    )
    hist_max = float(np.nanmax(hist_all)) if hist_all.size else 1.0

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("viz_phase3_household_comparison requires matplotlib.") from e

    with paper_style():
        fig = plt.figure(figsize=(FIGSIZE_FULL[0] * 1.95, FIGSIZE_FULL[1] * 1.40))
        gs = fig.add_gridspec(
            2,
            3,
            height_ratios=[1.0, 0.78],
            width_ratios=[1.0, 1.0, 0.42],
            hspace=0.16,
            wspace=0.10,
        )
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax10 = fig.add_subplot(gs[1, :2])
        ax_side = fig.add_subplot(gs[:, 2])

        mesh = _plot_density_panel(
            ax00,
            roads=roads,
            points=old_focus,
            bounds=bounds,
            title="",
            hist_max=hist_max,
            bins=int(args.bins),
        )
        _plot_density_panel(
            ax01,
            roads=roads,
            points=new_focus,
            bounds=bounds,
            title="",
            hist_max=hist_max,
            bins=int(args.bins),
        )
        add_panel_label(ax00, "a", dx=-18, dy=2)
        add_panel_label(ax01, "b", dx=-18, dy=2)

        old_n = old_points["n_persons"].clip(upper=12)
        new_n = new_points["n_persons"].clip(upper=12)
        bins = np.arange(0.5, 12.6, 1.0)
        ax10.hist(old_n, bins=bins, histtype="step", linewidth=2.1, color=OKABE_ITO["gray"], label="Person-proxy", log=True)
        ax10.hist(new_n, bins=bins, histtype="step", linewidth=2.1, color=OKABE_ITO["blue"], label="Household-aware", log=True)
        ax10.set_xticks(range(1, 13))
        ax10.set_xticklabels([str(v) for v in range(1, 12)] + ["12+"])
        ax10.set_xlabel("Residents per used home point")
        ax10.set_ylabel("Count of used home points")
        ax10.spines["top"].set_visible(False)
        ax10.spines["right"].set_visible(False)
        ax10.grid(axis="y", color="#EBEBEB", linewidth=0.8, alpha=0.8)
        add_panel_label(ax10, "c", dx=-40, dy=10)

        old_stats = _stats(old_points)
        new_stats = _stats(new_points)
        _draw_sidebar(ax_side, fig=fig, mesh=mesh)
        fig.subplots_adjust(left=0.055, right=0.985, top=0.96, bottom=0.11)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / "home_assignment_comparison.png"
        save_figure(fig, out_png)
        plt.close(fig)

    summary_payload = {
        "old_run_dir": str(old_run_dir),
        "new_run_dir": str(new_run_dir),
        "focus_bounds": list(bounds),
        "old_stats": old_stats,
        "new_stats": new_stats,
        "old_focus_used_home_points": int(old_focus.shape[0]),
        "new_focus_used_home_points": int(new_focus.shape[0]),
        "roads_path": str(roads_path),
        "output_png": str(out_dir / "home_assignment_comparison.png"),
    }
    _write_json(out_dir / "home_assignment_comparison_summary.json", summary_payload)
    print(f"[ok] wrote: {out_dir / 'home_assignment_comparison.png'}")
    print(f"[ok] wrote: {out_dir / 'home_assignment_comparison_summary.json'}")


if __name__ == "__main__":
    main()
