#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from shapely.geometry import box

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.plot_style import add_panel_label, paper_style, save_figure


PROJECT_ROOT = Path("/Users/jinlin/Desktop/Project/Synthetic_City")
DEFAULT_OUTDIR = PROJECT_ROOT / "figures" / "phase3_detroit_metro_latest"
DEFAULT_TRACT_ZIP = PROJECT_ROOT / "dataset" / "cache" / "geo" / "tl_2023_26_tract.zip"
DEFAULT_ROAD_ZIP = PROJECT_ROOT / "dataset" / "cache" / "geo" / "MI_road_cleaned.shp.zip"
DEFAULT_HOME_DECKGL_CROP = PROJECT_ROOT / "figures" / "phase3_detroit_metro_latest" / "home_micro_deckgl_cropped.png"
DEFAULT_HOME_POINTS = PROJECT_ROOT / "outputs" / "_sync_phase3_detroit_hcenter_20260330" / "micro_examples" / "home_all_points.csv"
DEFAULT_MOBILITY_POINTS = PROJECT_ROOT / "outputs" / "_sync_phase3_detroit_home_micro_validation_20260331" / "_phase3_detroit_home_micro_validation_20260331T000000Z_home_anchors.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "figures" / "phase3_detroit_metro_latest" / "micro_examples_manifest.json"


def _despine_map(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")


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


def _grid_shares(points: pd.DataFrame, bounds: tuple[float, float, float, float], nx: int = 16, ny: int = 12):
    xmin, ymin, xmax, ymax = bounds
    xedges = np.linspace(xmin, xmax, nx + 1)
    yedges = np.linspace(ymin, ymax, ny + 1)
    hist, _, _ = np.histogram2d(points["x"], points["y"], bins=[xedges, yedges])
    hist = hist.T
    share = hist / hist.sum() if hist.sum() > 0 else hist
    share[share <= 0] = np.nan
    return xedges, yedges, share


def _format_share(v: float, _pos=None) -> str:
    if v >= 1e-2:
        return f"{v:.02f}"
    if v >= 1e-3:
        return f"{v:.003f}"
    return f"{v:.0e}"


def _format_signed_share(v: float, _pos=None) -> str:
    av = abs(v)
    if av >= 1e-2:
        s = f"{av:.02f}"
    elif av >= 1e-3:
        s = f"{av:.003f}"
    else:
        s = f"{av:.0e}"
    return f"-{s}" if v < 0 else s


def _plot_locator(ax, tracts: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> None:
    tracts.plot(ax=ax, color="#F7F7F5", edgecolor="#D9D9D2", linewidth=0.18, zorder=0)
    gpd.GeoDataFrame(geometry=[box(*bounds)], crs=tracts.crs).boundary.plot(
        ax=ax, color="#2C6BA0", linewidth=0.95, zorder=2
    )
    _despine_map(ax)


def _add_locator_inset(ax, tracts: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> None:
    iax = ax.inset_axes([0.015, 0.64, 0.22, 0.26])
    iax.set_facecolor("#F4F8FB")
    tracts.plot(ax=iax, color="#EEF4FA", edgecolor="#C9D8E6", linewidth=0.13, zorder=0)
    bbox_geom = box(*bounds)
    gpd.GeoDataFrame(geometry=[bbox_geom], crs=tracts.crs).plot(
        ax=iax,
        facecolor="#9BC3E6",
        edgecolor="#2C6BA0",
        linewidth=0.95,
        alpha=0.48,
        zorder=1.8,
    )
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    iax.scatter(
        [cx],
        [cy],
        s=18,
        c="#1E5E91",
        edgecolors="white",
        linewidths=0.45,
        zorder=2.6,
    )
    _despine_map(iax)
    for spine in iax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_edgecolor("#D6E2EC")


def _plot_share_map(ax, tracts_zoom: gpd.GeoDataFrame, xedges, yedges, share, norm, title: str | None = None):
    tracts_zoom.plot(ax=ax, color="#FCFBF8", edgecolor="#E4DFD7", linewidth=0.22, zorder=0)
    mesh = ax.pcolormesh(
        xedges,
        yedges,
        share,
        shading="auto",
        cmap="YlOrBr",
        norm=norm,
        alpha=0.92,
        zorder=1.5,
    )
    if title:
        ax.set_title(title, pad=4)
    _despine_map(ax)
    return mesh


def _plot_difference_map(ax, tracts_zoom: gpd.GeoDataFrame, xedges, yedges, diff, norm, title: str | None = None):
    tracts_zoom.plot(ax=ax, color="#FCFBF8", edgecolor="#E4DFD7", linewidth=0.22, zorder=0)
    mesh = ax.pcolormesh(
        xedges,
        yedges,
        diff,
        shading="auto",
        cmap="PuOr_r",
        norm=norm,
        alpha=0.92,
        zorder=1.5,
    )
    if title:
        ax.set_title(title, pad=4)
    _despine_map(ax)
    return mesh


def _plot_result_micro(
    ax,
    *,
    tracts_zoom: gpd.GeoDataFrame,
    selected: gpd.GeoDataFrame,
    roads_zoom: gpd.GeoDataFrame,
    occ: pd.DataFrame,
    bounds: tuple[float, float, float, float],
) -> None:
    tracts_zoom.plot(ax=ax, color="#FBFAF7", edgecolor="#E5DFD6", linewidth=0.22, zorder=0)
    selected.plot(ax=ax, color="#F4F8FB", edgecolor="none", zorder=0.15)

    roads_bg = roads_zoom.loc[~roads_zoom["MTFCC"].isin(["S1400", "S1740"])]
    roads_fg = roads_zoom.loc[roads_zoom["MTFCC"].isin(["S1400", "S1740"])]
    if len(roads_bg):
        roads_bg.plot(ax=ax, color="#ECE6DD", linewidth=0.26, alpha=0.62, zorder=0.5)
    if len(roads_fg):
        roads_fg.plot(ax=ax, color="#D4DEE8", linewidth=0.40, alpha=0.75, zorder=0.7)

    if len(occ):
        ax.hexbin(
            occ["x"].to_numpy(),
            occ["y"].to_numpy(),
            C=occ["residents"].to_numpy(),
            reduce_C_function=np.sum,
            gridsize=24,
            cmap="PuBu",
            mincnt=1,
            linewidths=0.0,
            alpha=0.88,
            extent=bounds,
            zorder=1.4,
        )
        high = occ.loc[occ["residents"] >= occ["residents"].quantile(0.9)].copy()
        if len(high):
            ax.scatter(
                high["x"].to_numpy(),
                high["y"].to_numpy(),
                s=18 + 2.6 * np.sqrt(high["residents"].to_numpy()),
                c="#0E5A8A",
                alpha=0.92,
                linewidths=0.35,
                edgecolors="white",
                zorder=2.2,
            )

    selected.boundary.plot(ax=ax, color="#2B76A9", linewidth=1.10, zorder=2.6)
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    _despine_map(ax)


def _plot_result_image(ax, image_path: Path, title: str | None = None) -> None:
    img = mpimg.imread(image_path)
    ax.imshow(img)
    if title:
        ax.set_title(title, pad=4)
    _despine_map(ax)


def _hotspot_categories(
    left: np.ndarray,
    right: np.ndarray,
    quantile: float = 0.8,
) -> tuple[np.ndarray, dict[str, float]]:
    l = np.nan_to_num(left, nan=0.0)
    r = np.nan_to_num(right, nan=0.0)
    lpos = l[l > 0]
    rpos = r[r > 0]
    ql = float(np.quantile(lpos, quantile)) if lpos.size else float("nan")
    qr = float(np.quantile(rpos, quantile)) if rpos.size else float("nan")
    lhot = l >= ql if np.isfinite(ql) else np.zeros_like(l, dtype=bool)
    rhot = r >= qr if np.isfinite(qr) else np.zeros_like(r, dtype=bool)
    cats = np.full(l.shape, np.nan)
    cats[lhot & ~rhot] = 1.0
    cats[lhot & rhot] = 2.0
    cats[~lhot & rhot] = 3.0
    inter = int((lhot & rhot).sum())
    union = int((lhot | rhot).sum())
    metrics = {
        "hotspot_quantile": quantile,
        "synthetic_hotspots": int(lhot.sum()),
        "mobility_hotspots": int(rhot.sum()),
        "shared_hotspots": inter,
        "hotspot_jaccard": float(inter / max(union, 1)),
    }
    return cats, metrics


def _hotspot_mask(arr: np.ndarray, quantile: float = 0.8) -> np.ndarray:
    vals = arr[np.isfinite(arr) & (arr > 0)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=bool)
    threshold = float(np.quantile(vals, quantile))
    return np.nan_to_num(arr, nan=0.0) >= threshold


def _overlay_hotspot_outlines(ax, *, xedges: np.ndarray, yedges: np.ndarray, mask: np.ndarray, color: str) -> None:
    ny, nx = mask.shape
    for iy in range(ny):
        for ix in range(nx):
            if not mask[iy, ix]:
                continue
            ax.add_patch(
                Rectangle(
                    (xedges[ix], yedges[iy]),
                    xedges[ix + 1] - xedges[ix],
                    yedges[iy + 1] - yedges[iy],
                    facecolor="none",
                    edgecolor=color,
                    linewidth=0.8,
                    zorder=2.2,
                )
            )


def _compare_shares(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    l = np.nan_to_num(left, nan=0.0).ravel()
    r = np.nan_to_num(right, nan=0.0).ravel()
    mask = (l > 0) | (r > 0)
    l = l[mask]
    r = r[mask]
    if len(l) < 2 or np.unique(l).size <= 1 or np.unique(r).size <= 1:
        rho = float("nan")
    else:
        rho = float(pd.Series(l).corr(pd.Series(r), method="spearman"))
    denom = float(np.linalg.norm(l) * np.linalg.norm(r))
    cosine = float(np.dot(l, r) / denom) if denom > 0 else float("nan")
    ql = np.quantile(l, 0.9) if len(l) else np.nan
    qr = np.quantile(r, 0.9) if len(r) else np.nan
    top_l = set(np.where(l >= ql)[0].tolist()) if np.isfinite(ql) else set()
    top_r = set(np.where(r >= qr)[0].tolist()) if np.isfinite(qr) else set()
    overlap = float(len(top_l & top_r) / max(len(top_l | top_r), 1))
    return {"rho": rho, "cosine": cosine, "top_decile_jaccard": overlap}


def _plot_hotspot_summary(ax, hotspot_metrics: dict[str, float], metrics: dict[str, float]) -> None:
    shared = float(hotspot_metrics["shared_hotspots"])
    synth_only = float(hotspot_metrics["synthetic_hotspots"] - hotspot_metrics["shared_hotspots"])
    mob_only = float(hotspot_metrics["mobility_hotspots"] - hotspot_metrics["shared_hotspots"])
    union = max(shared + synth_only + mob_only, 1.0)
    vals = np.array([shared, synth_only, mob_only], dtype=float) / union
    cats = ["Shared", "Synthetic only", "Mobility only"]
    cols = ["#A65F2B", "#6EA6D8", "#D6A541"]
    x = np.arange(len(cats))
    ax.bar(x, vals, color=cols, width=0.62, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Share of hotspot union")
    ax.set_title("Hotspot agreement summary", pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.98,
        0.96,
        f"ρ={metrics['rho']:.3f}\nCosine={metrics['cosine']:.3f}\nJaccard={hotspot_metrics['hotspot_jaccard']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color="#4B4B4B",
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="viz_phase3_home_micro_validation")
    ap.add_argument("--tract_zip", default=str(DEFAULT_TRACT_ZIP))
    ap.add_argument("--road_zip", default=str(DEFAULT_ROAD_ZIP))
    ap.add_argument("--result_image", default=str(DEFAULT_HOME_DECKGL_CROP))
    ap.add_argument("--home_points_csv", default=str(DEFAULT_HOME_POINTS))
    ap.add_argument("--mobility_points_csv", default=str(DEFAULT_MOBILITY_POINTS))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = ap.parse_args()

    tract_zip = Path(args.tract_zip).expanduser().resolve()
    road_zip = Path(args.road_zip).expanduser().resolve()
    result_image = Path(args.result_image).expanduser().resolve()
    home_points_csv = Path(args.home_points_csv).expanduser().resolve()
    mobility_points_csv = Path(args.mobility_points_csv).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    manifest = json.loads(manifest_path.read_text())
    bounds = tuple(float(v) for v in manifest["home_example"]["bounds"])
    selected_tract_geoid = str(manifest["home_example"]["selected_tract_geoid"])

    tracts = _read_geodata(tract_zip)
    tracts["tract_geoid"] = tracts["GEOID"].astype(str)
    tracts_zoom = _clip_to_bbox(tracts, bounds)
    selected = tracts.loc[tracts["tract_geoid"] == selected_tract_geoid].copy()
    roads = _prepare_roads(road_zip, tracts.crs)
    roads_zoom = _clip_to_bbox(roads, bounds)

    syn = pd.read_csv(home_points_csv).rename(columns={"x": "x", "y": "y"})
    syn = syn[(syn["x"] >= bounds[0]) & (syn["x"] <= bounds[2]) & (syn["y"] >= bounds[1]) & (syn["y"] <= bounds[3])].copy()
    occ = syn.groupby(["x", "y"], as_index=False).size().rename(columns={"size": "residents"})

    mob = pd.read_csv(mobility_points_csv).rename(columns={"home_longitude": "x", "home_latitude": "y"})
    mob = mob[(mob["x"] >= bounds[0]) & (mob["x"] <= bounds[2]) & (mob["y"] >= bounds[1]) & (mob["y"] <= bounds[3])].copy()

    xedges, yedges, syn_share = _grid_shares(syn, bounds, nx=16, ny=12)
    _, _, mob_share = _grid_shares(mob, bounds, nx=len(xedges) - 1, ny=len(yedges) - 1)
    positive = []
    for arr in [syn_share, mob_share]:
        vals = arr[np.isfinite(arr) & (arr > 0)]
        if vals.size:
            positive.append(vals)
    cat = np.concatenate(positive) if positive else np.array([1e-6, 1e-2])
    norm = LogNorm(vmin=float(cat.min()), vmax=float(cat.max()))
    metrics = _compare_shares(syn_share, mob_share)
    hotspot_quantile = 0.8
    _, hotspot_metrics = _hotspot_categories(syn_share, mob_share, quantile=hotspot_quantile)
    syn_hotmask = _hotspot_mask(syn_share, quantile=hotspot_quantile)
    diff = np.nan_to_num(syn_share, nan=0.0) - np.nan_to_num(mob_share, nan=0.0)
    diff[np.isnan(syn_share) & np.isnan(mob_share)] = np.nan
    diff_vals = diff[np.isfinite(diff)]
    diff_max = float(np.quantile(np.abs(diff_vals), 0.95)) if diff_vals.size else 1e-3
    diff_max = max(diff_max, 1e-4)

    with paper_style():
        fig = plt.figure(figsize=(7.1, 5.25))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.0, 1.0],
            height_ratios=[1.0, 1.0],
            left=0.07,
            right=0.96,
            bottom=0.16,
            top=0.94,
            wspace=0.16,
            hspace=0.23,
        )
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])

        if result_image.exists():
            _plot_result_image(ax_a, result_image, None)
        else:
            _plot_result_micro(
                ax_a,
                tracts_zoom=tracts_zoom,
                selected=selected,
                roads_zoom=roads_zoom,
                occ=occ,
                bounds=bounds,
            )
        _add_locator_inset(ax_a, tracts=tracts, bounds=bounds)

        mesh_b = _plot_share_map(
            ax_b,
            tracts_zoom=tracts_zoom,
            xedges=xedges,
            yedges=yedges,
            share=syn_share,
            norm=norm,
            title=None,
        )
        mesh_c = _plot_share_map(
            ax_c,
            tracts_zoom=tracts_zoom,
            xedges=xedges,
            yedges=yedges,
            share=mob_share,
            norm=norm,
            title=None,
        )
        _overlay_hotspot_outlines(ax_c, xedges=xedges, yedges=yedges, mask=syn_hotmask, color="#2B76A9")
        ax_c.legend(
            handles=[
                Line2D([0], [0], color="#2B76A9", lw=1.2, label="Synthetic top-20% hotspot"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.50, -0.06),
            frameon=False,
            fontsize=7.6,
            handlelength=1.2,
            borderpad=0.10,
            labelspacing=0.18,
        )
        mesh_d = _plot_difference_map(
            ax_d,
            tracts_zoom=tracts_zoom,
            xedges=xedges,
            yedges=yedges,
            diff=diff,
            norm=TwoSlopeNorm(vmin=-diff_max, vcenter=0.0, vmax=diff_max),
            title=None,
        )

        add_panel_label(ax_a, "a", dx=-30)
        add_panel_label(ax_b, "b", dx=-30)
        add_panel_label(ax_c, "c", dx=-30)
        add_panel_label(ax_d, "d", dx=-30)

        cax1 = fig.add_axes([0.165, 0.067, 0.18, 0.018])
        cb = fig.colorbar(mesh_b, cax=cax1, orientation="horizontal")
        cb.set_label("Share within micro window", fontsize=8.6)
        cb.ax.tick_params(labelsize=8.0, length=2.5, colors="#555555")
        cb.outline.set_linewidth(0.45)
        cb.formatter = FuncFormatter(_format_share)
        cb.update_ticks()
        cax2 = fig.add_axes([0.655, 0.067, 0.18, 0.018])
        cb2 = fig.colorbar(mesh_d, cax=cax2, orientation="horizontal")
        cb2.set_label("Synthetic − mobility", fontsize=7.8)
        cb2.ax.tick_params(labelsize=7.2, length=2.2, colors="#555555")
        cb2.outline.set_linewidth(0.45)
        cb2.formatter = FuncFormatter(_format_signed_share)
        cb2.update_ticks()

        outdir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, outdir / "home_micro_validation.png")
        plt.close(fig)

    payload = {
        "bounds": list(bounds),
        "n_synthetic_points": int(len(syn)),
        "n_mobility_anchors": int(len(mob)),
        "metrics": metrics,
        "hotspot_metrics": hotspot_metrics,
        "artifact": str(outdir / "home_micro_validation.png"),
    }
    (outdir / "home_micro_validation_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {outdir / 'home_micro_validation.png'}")


if __name__ == "__main__":
    main()
