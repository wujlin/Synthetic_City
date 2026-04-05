#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from datetime import datetime, timezone
from typing import Any

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.plot_style import FIGSIZE_FULL, OKABE_ITO, paper_style, save_figure

_DEFAULT_OUT_DIR = _REPO_ROOT / "figures" / "phase3_spatial_population_latest"
_EXCEPTION_STAGE = "arterial_missing_exception"


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_geodata(path: pathlib.Path) -> Any:
    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise SystemExit("viz_phase3_spatial_population requires geopandas.") from e

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return gpd.read_parquet(path)
    if suffix == ".zip":
        return gpd.read_file(f"zip://{path}")
    return gpd.read_file(path)


def _sample_points_csv(
    *,
    csv_path: pathlib.Path,
    x_col: str,
    y_col: str,
    sample_n: int,
    chunk_size: int = 200_000,
    seed: int = 0,
    bbox: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    if int(sample_n) <= 0:
        return pd.DataFrame(columns=["x", "y"])
    rnd = random.Random(int(seed))
    reservoir: list[tuple[float, float]] = []
    seen = 0
    usecols = [str(x_col), str(y_col)]
    xmin = ymin = xmax = ymax = None
    if bbox is not None:
        xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=int(chunk_size), low_memory=False):
        chunk = chunk.dropna(subset=[str(x_col), str(y_col)])
        if bbox is not None and not chunk.empty:
            chunk = chunk[
                (chunk[str(x_col)] >= float(xmin))
                & (chunk[str(x_col)] <= float(xmax))
                & (chunk[str(y_col)] >= float(ymin))
                & (chunk[str(y_col)] <= float(ymax))
            ]
        if chunk.empty:
            continue
        for x, y in chunk[[str(x_col), str(y_col)]].itertuples(index=False, name=None):
            seen += 1
            item = (float(x), float(y))
            if len(reservoir) < int(sample_n):
                reservoir.append(item)
                continue
            j = rnd.randrange(seen)
            if j < int(sample_n):
                reservoir[j] = item
    return pd.DataFrame(reservoir, columns=["x", "y"])


def _ensure_crs(gdf: Any, target_crs: Any) -> Any:
    if target_crs is None:
        return gdf
    if getattr(gdf, "crs", None) is None:
        return gdf.set_crs(target_crs)
    if str(gdf.crs) != str(target_crs):
        return gdf.to_crs(target_crs)
    return gdf


def _despine(ax: Any) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_aspect("equal")


def _geo_box(bounds: tuple[float, float, float, float], crs: Any) -> Any:
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except Exception as e:  # pragma: no cover
        raise SystemExit("viz_phase3_spatial_population requires geopandas and shapely.") from e

    return gpd.GeoDataFrame({"id": [1]}, geometry=[box(*bounds)], crs=crs)


def _select_focus_region(
    plot_gdf: Any,
    *,
    pop_col: str = "n_persons",
    search_radius_m: float = 20_000.0,
    focus_radius_m: float = 10_000.0,
    min_focus_groups: int = 10,
    pad_fraction: float = 0.08,
) -> dict[str, Any]:
    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise SystemExit("viz_phase3_spatial_population requires geopandas.") from e

    if int(plot_gdf.shape[0]) == 0:
        raise SystemExit("cannot select focus region from empty GeoDataFrame")

    work = plot_gdf[[c for c in plot_gdf.columns if c != "geometry"]].copy()
    geo = gpd.GeoDataFrame(work, geometry=plot_gdf.geometry, crs=plot_gdf.crs)
    geo = geo[geo.geometry.notna() & ~geo.geometry.is_empty].copy()
    if int(geo.shape[0]) == 0:
        raise SystemExit("all geometries are empty; cannot select focus region")

    geo_proj = geo.to_crs(3857)
    centroids = geo_proj.geometry.centroid
    coords = [(float(pt.x), float(pt.y)) for pt in centroids]
    masses = geo_proj[str(pop_col)].fillna(0).astype(float).tolist()

    best_idx = 0
    best_score = -1.0
    for i, (xi, yi) in enumerate(coords):
        total = 0.0
        for j, (xj, yj) in enumerate(coords):
            if math.hypot(xi - xj, yi - yj) <= float(search_radius_m):
                total += float(masses[j])
        if total > best_score:
            best_score = total
            best_idx = i

    cx, cy = coords[best_idx]
    distances = [math.hypot(cx - xj, cy - yj) for xj, yj in coords]
    focus_mask = [dist <= float(focus_radius_m) for dist in distances]
    if sum(focus_mask) < int(min_focus_groups):
        nearest_idx = sorted(range(len(distances)), key=lambda i: distances[i])[: int(min_focus_groups)]
        focus_mask = [i in nearest_idx for i in range(len(distances))]

    focus_proj = geo_proj.loc[focus_mask].copy()
    xmin, ymin, xmax, ymax = [float(v) for v in focus_proj.total_bounds]
    dx = xmax - xmin
    dy = ymax - ymin
    pad_x = max(dx * float(pad_fraction), 600.0)
    pad_y = max(dy * float(pad_fraction), 600.0)
    padded_proj = (xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y)
    padded_native = _geo_box(padded_proj, 3857).to_crs(plot_gdf.crs).total_bounds.tolist()
    bounds = tuple(float(v) for v in padded_native)

    center_group = str(geo.iloc[best_idx]["tract_geoid"]) if "tract_geoid" in geo.columns else ""
    focus_native = geo[geo.intersects(_geo_box(bounds, plot_gdf.crs).geometry.iloc[0])].copy()
    county_prefixes = sorted({str(v)[:5] for v in focus_native.get("tract_geoid", pd.Series(dtype=str)).astype(str).tolist() if len(str(v)) >= 5})

    return {
        "bounds": bounds,
        "center_group": center_group,
        "search_radius_m": float(search_radius_m),
        "focus_radius_m": float(focus_radius_m),
        "n_focus_groups": int(focus_native.shape[0]),
        "focus_group_ids": focus_native.get("tract_geoid", pd.Series(dtype=str)).astype(str).tolist(),
        "focus_total_persons": int(pd.to_numeric(focus_native.get(pop_col, 0), errors="coerce").fillna(0).sum()),
        "county_prefixes": county_prefixes,
    }


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
        "S1400": {"color": "#BFBFBF", "linewidth": 0.35, "alpha": 0.9},
        "S1740": {"color": "#D9D9D9", "linewidth": 0.35, "alpha": 0.9},
        "S1100": {"color": "#636363", "linewidth": 0.6, "alpha": 0.95},
        "S1200": {"color": "#8C8C8C", "linewidth": 0.5, "alpha": 0.95},
        "other": {"color": "#E6E6E6", "linewidth": 0.25, "alpha": 0.7},
    }
    for cls, cfg in line_cfg.items():
        part = roads[roads["mtfcc_class"] == cls]
        if int(part.shape[0]) == 0:
            continue
        part.plot(ax=ax, color=cfg["color"], linewidth=cfg["linewidth"], alpha=cfg["alpha"], zorder=1)


def _make_overview(
    *,
    plot_gdf: Any,
    home_pts: pd.DataFrame,
    work_pts: pd.DataFrame,
    validation: dict[str, Any],
    out_png: pathlib.Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except Exception as e:  # pragma: no cover
        raise SystemExit("viz_phase3_spatial_population requires matplotlib.") from e

    exception_gdf = plot_gdf[plot_gdf["work_stage_class"].str.contains(_EXCEPTION_STAGE, regex=False)].copy()

    with paper_style():
        fig, axes = plt.subplots(2, 2, figsize=(FIGSIZE_FULL[0] * 1.8, FIGSIZE_FULL[1] * 1.8))
        ax00, ax01, ax10, ax11 = axes.ravel()

        plot_gdf.plot(
            ax=ax00,
            column="n_persons",
            cmap="YlGnBu",
            linewidth=0.15,
            edgecolor="#DDDDDD",
            legend=True,
            legend_kwds={"shrink": 0.75, "label": "Persons per tract"},
        )
        ax00.set_title("Tract Population Mass")
        _despine(ax00)

        plot_gdf.boundary.plot(ax=ax01, linewidth=0.15, color="#BDBDBD")
        if not home_pts.empty:
            hb = ax01.hexbin(
                home_pts["x"].to_numpy(),
                home_pts["y"].to_numpy(),
                gridsize=160,
                mincnt=1,
                linewidths=0.0,
                cmap="Blues",
                norm=LogNorm(),
            )
            fig.colorbar(hb, ax=ax01, shrink=0.75, label="Sampled home-point density")
        ax01.set_title(f"Sampled Home Points (n={len(home_pts):,})")
        _despine(ax01)

        plot_gdf.boundary.plot(ax=ax10, linewidth=0.15, color="#BDBDBD")
        if not work_pts.empty:
            hb = ax10.hexbin(
                work_pts["x"].to_numpy(),
                work_pts["y"].to_numpy(),
                gridsize=160,
                mincnt=1,
                linewidths=0.0,
                cmap="OrRd",
                norm=LogNorm(),
            )
            fig.colorbar(hb, ax=ax10, shrink=0.75, label="Sampled work-point density")
        ax10.set_title(f"Sampled Work Points (n={len(work_pts):,})")
        _despine(ax10)

        plot_gdf.plot(ax=ax11, color="#F2F2F2", linewidth=0.15, edgecolor="#C7C7C7")
        if not exception_gdf.empty:
            exception_gdf.plot(
                ax=ax11,
                color=OKABE_ITO["orange"],
                linewidth=0.35,
                edgecolor=OKABE_ITO["vermillion"],
            )
        plot_gdf.boundary.plot(ax=ax11, linewidth=0.12, color="#BDBDBD")
        ax11.set_title("Work Support Regime")
        ax11.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"Home: {int(validation['coverage']['home_unassigned']):,} unassigned",
                    f"Work: {int(validation['coverage']['work_unassigned']):,} unassigned",
                    f"Primary work tracts: {int(validation['coverage']['work_stage_counts'].get('primary', 0))}",
                    f"Exception tracts: {int(validation['coverage']['work_stage_counts'].get(_EXCEPTION_STAGE, 0))}",
                ]
            ),
            transform=ax11.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D9D9D9", "alpha": 0.95},
        )
        _despine(ax11)

        fig.suptitle("Phase 3 Spatial Population Overview", y=0.98)
        fig.text(
            0.5,
            0.01,
            "Home/work point clouds are sampled from final person-level assignments for rendering.",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#555555",
        )
        fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.97])
        save_figure(fig, out_png)
        plt.close(fig)


def _make_zoom(
    *,
    zoom_gdf: Any,
    roads_zoom: Any,
    home_zoom: pd.DataFrame,
    work_zoom: pd.DataFrame,
    focus_meta: dict[str, Any],
    out_png: pathlib.Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
    except Exception as e:  # pragma: no cover
        raise SystemExit("viz_phase3_spatial_population requires matplotlib.") from e

    exception_zoom = zoom_gdf[zoom_gdf["work_stage_class"].str.contains(_EXCEPTION_STAGE, regex=False)].copy()
    all_bounds = focus_meta["bounds"]

    with paper_style():
        fig = plt.figure(figsize=(FIGSIZE_FULL[0] * 2.05, FIGSIZE_FULL[1] * 1.18))
        gs = GridSpec(
            2,
            2,
            figure=fig,
            height_ratios=[1.0, 0.17],
            width_ratios=[1.0, 1.0],
            hspace=0.08,
            wspace=0.06,
        )
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax_leg = fig.add_subplot(gs[1, 0])
        ax_info = fig.add_subplot(gs[1, 1])

        zoom_gdf.plot(ax=ax0, color="#FBFBFB", edgecolor="#D5D5D5", linewidth=0.35, zorder=0)
        _plot_roads(ax0, roads_zoom)
        if not home_zoom.empty:
            ax0.scatter(
                home_zoom["x"].to_numpy(),
                home_zoom["y"].to_numpy(),
                s=2.2,
                c=OKABE_ITO["blue"],
                alpha=0.25,
                linewidths=0.0,
                zorder=2,
                label="Sampled home points",
            )
        if not work_zoom.empty:
            ax0.scatter(
                work_zoom["x"].to_numpy(),
                work_zoom["y"].to_numpy(),
                s=2.4,
                c=OKABE_ITO["vermillion"],
                alpha=0.25,
                linewidths=0.0,
                zorder=3,
                label="Sampled work points",
            )
        _despine(ax0)

        zoom_gdf.plot(
            ax=ax1,
            column="n_persons",
            cmap="YlGnBu",
            linewidth=0.35,
            edgecolor="#DDDDDD",
            zorder=0,
        )
        if not exception_zoom.empty:
            exception_zoom.plot(
                ax=ax1,
                color=OKABE_ITO["orange"],
                edgecolor=OKABE_ITO["vermillion"],
                linewidth=0.8,
                alpha=0.8,
                zorder=1,
            )
        zoom_gdf.boundary.plot(ax=ax1, linewidth=0.25, color="#BDBDBD", zorder=2)
        _despine(ax1)

        for ax in [ax0, ax1]:
            ax.set_xlim(all_bounds[0], all_bounds[2])
            ax.set_ylim(all_bounds[1], all_bounds[3])
            ax.text(
                0.012,
                0.985,
                "a" if ax is ax0 else "b",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.2},
            )

        ax_leg.axis("off")
        ax_leg.legend(
            handles=[
                Line2D([0], [0], color="#636363", lw=1.2, label="S1100/S1200 roads"),
                Line2D([0], [0], color="#BFBFBF", lw=1.0, label="S1400/S1740 roads"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor=OKABE_ITO["blue"], markersize=6, alpha=0.7, label="Home"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor=OKABE_ITO["vermillion"], markersize=6, alpha=0.7, label="Work"),
            ],
            loc="center left",
            ncol=2,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.4,
            handletextpad=0.6,
            borderaxespad=0.0,
        )

        ax_info.axis("off")
        ax_info.text(
            0.0,
            0.5,
            "\n".join(
                [
                    f"Center tract: {focus_meta['center_group']}",
                    f"Focus tracts: {int(focus_meta['n_focus_groups']):,}",
                    f"Persons in focus: {int(focus_meta['focus_total_persons']):,}",
                    f"Counties: {', '.join(focus_meta['county_prefixes']) or 'n/a'}",
                    f"Exception tracts: {int(exception_zoom.shape[0]):,}",
                ]
            ),
            ha="left",
            va="center",
            fontsize=9.5,
        )

        fig.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.07)
        save_figure(fig, out_png)
        plt.close(fig)


def _folium_color(value: float, vmax: float) -> str:
    vmax = max(float(vmax), 1.0)
    ratio = min(max(float(value) / vmax, 0.0), 1.0)
    if ratio < 0.2:
        return "#f7fbff"
    if ratio < 0.4:
        return "#c6dbef"
    if ratio < 0.6:
        return "#6baed6"
    if ratio < 0.8:
        return "#3182bd"
    return "#08519c"


def _make_interactive_html(
    *,
    zoom_gdf: Any,
    roads_zoom: Any,
    home_zoom: pd.DataFrame,
    work_zoom: pd.DataFrame,
    focus_meta: dict[str, Any],
    out_html: pathlib.Path,
    road_simplify_tol: float = 0.0,
) -> None:
    try:
        import folium
        from folium import FeatureGroup, GeoJson, LayerControl, Map
        from shapely.geometry import mapping
    except Exception as e:  # pragma: no cover
        raise SystemExit("interactive HTML requires folium. Install folium in the active environment.") from e

    zoom_ll = zoom_gdf.to_crs(4326).copy()
    roads_ll = roads_zoom.to_crs(4326).copy()
    if float(road_simplify_tol) > 0.0 and int(roads_ll.shape[0]) > 0:
        roads_ll["geometry"] = roads_ll.geometry.simplify(float(road_simplify_tol), preserve_topology=False)

    xmin, ymin, xmax, ymax = [float(v) for v in zoom_ll.total_bounds]
    center = [(ymin + ymax) / 2.0, (xmin + xmax) / 2.0]

    fmap = Map(location=center, zoom_start=13, tiles="CartoDB positron", control_scale=True)
    vmax = max(float(zoom_ll["n_persons"].fillna(0).max()), 1.0)

    tract_layer = FeatureGroup(name="Tracts", show=True)
    for _, row in zoom_ll.iterrows():
        stage = str(row.get("work_stage_class", ""))
        fill = OKABE_ITO["orange"] if _EXCEPTION_STAGE in stage else _folium_color(float(row.get("n_persons", 0) or 0), vmax)
        GeoJson(
            data=mapping(row.geometry),
            style_function=lambda _feat, fill=fill, stage=stage: {
                "fillColor": fill,
                "color": "#bdbdbd" if _EXCEPTION_STAGE not in stage else OKABE_ITO["vermillion"],
                "weight": 1.0,
                "fillOpacity": 0.55 if _EXCEPTION_STAGE in stage else 0.35,
            },
            tooltip=folium.Tooltip(
                "<br>".join(
                    [
                        f"tract: {row['tract_geoid']}",
                        f"persons: {int(row.get('n_persons', 0) or 0):,}",
                        f"workers: {int(row.get('n_workers', 0) or 0):,}",
                        f"work stage: {stage}",
                    ]
                )
            ),
        ).add_to(tract_layer)
    tract_layer.add_to(fmap)

    road_groups = [
        ("Home-support roads", ["S1400", "S1740"], "#9E9E9E", True),
        ("Work-support roads", ["S1100", "S1200"], "#5F5F5F", True),
    ]
    for layer_name, mtfcc_values, color, show in road_groups:
        layer = FeatureGroup(name=layer_name, show=show)
        part = roads_ll[roads_ll["MTFCC"].isin(mtfcc_values)].copy()
        for _, row in part.iterrows():
            GeoJson(
                data=mapping(row.geometry),
                style_function=lambda _feat, color=color: {
                    "color": color,
                    "weight": 1.0,
                    "opacity": 0.8,
                },
            ).add_to(layer)
        layer.add_to(fmap)

    home_layer = FeatureGroup(name=f"Sampled home points ({len(home_zoom):,})", show=True)
    for row in home_zoom.itertuples(index=False):
        folium.CircleMarker(
            location=[float(row.y), float(row.x)],
            radius=2,
            color=OKABE_ITO["blue"],
            weight=0,
            fill=True,
            fill_color=OKABE_ITO["blue"],
            fill_opacity=0.45,
        ).add_to(home_layer)
    home_layer.add_to(fmap)

    work_layer = FeatureGroup(name=f"Sampled work points ({len(work_zoom):,})", show=True)
    for row in work_zoom.itertuples(index=False):
        folium.CircleMarker(
            location=[float(row.y), float(row.x)],
            radius=2,
            color=OKABE_ITO["vermillion"],
            weight=0,
            fill=True,
            fill_color=OKABE_ITO["vermillion"],
            fill_opacity=0.45,
        ).add_to(work_layer)
    work_layer.add_to(fmap)

    LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[ymin, xmin], [ymax, xmax]])

    title_html = f"""
    <div style="position: fixed; top: 10px; left: 50px; z-index: 9999; background: white;
                padding: 8px 12px; border: 1px solid #d0d0d0; border-radius: 6px;
                font-family: Arial, sans-serif; font-size: 13px;">
      <b>Phase 3 Spatial Population: Auto-selected High-Density Core</b><br>
      Center tract: {focus_meta['center_group']}<br>
      Focus tracts: {int(focus_meta['n_focus_groups']):,}<br>
      Exception tracts: {int((zoom_gdf['work_stage_class'].astype(str).str.contains(_EXCEPTION_STAGE, regex=False)).sum()):,}
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))


def main() -> None:
    ap = argparse.ArgumentParser(prog="viz_phase3_spatial_population")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--areas_path", default="")
    ap.add_argument("--roads_path", default="")
    ap.add_argument("--areas_group_col", default="")
    ap.add_argument("--group_col", default="")
    ap.add_argument("--overview_home_sample_n", type=int, default=120000)
    ap.add_argument("--overview_work_sample_n", type=int, default=120000)
    ap.add_argument("--zoom_home_sample_n", type=int, default=30000)
    ap.add_argument("--zoom_work_sample_n", type=int, default=30000)
    ap.add_argument("--html_home_sample_n", type=int, default=6000)
    ap.add_argument("--html_work_sample_n", type=int, default=6000)
    ap.add_argument("--chunk_size", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search_radius_m", type=float, default=20000.0)
    ap.add_argument("--focus_radius_m", type=float, default=10000.0)
    ap.add_argument("--min_focus_groups", type=int, default=10)
    ap.add_argument("--road_simplify_tol", type=float, default=0.0)
    ap.add_argument("--skip_html", action="store_true")
    ap.add_argument("--out_dir", default=str(_DEFAULT_OUT_DIR))
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    summary = _read_json(run_dir / "metrics" / "summary.json")
    validation = _read_json(run_dir / "metrics" / "roadloc_validation.json")
    group_col = str(args.group_col).strip() or str(summary.get("group_col") or "tract_geoid")
    areas_group_col = str(args.areas_group_col).strip() or str(summary.get("areas_group_col") or group_col)

    input_paths = summary.get("input_paths", {})
    areas_path = pathlib.Path(args.areas_path).expanduser().resolve() if args.areas_path else pathlib.Path(input_paths["areas_path"])
    roads_path = pathlib.Path(args.roads_path).expanduser().resolve() if args.roads_path else pathlib.Path(input_paths["roads_path"])
    persons_csv = pathlib.Path(summary["artifacts"]["person_locations_csv"]).expanduser().resolve()
    group_diag_csv = pathlib.Path(summary["artifacts"]["group_diagnostics_csv"]).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    areas = _read_geodata(areas_path)
    if str(areas_group_col) not in areas.columns:
        raise SystemExit(f"areas missing group column: {areas_group_col}")
    areas = areas[[str(areas_group_col), "geometry"]].copy()
    areas = areas.rename(columns={str(areas_group_col): str(group_col)})
    areas[str(group_col)] = areas[str(group_col)].astype(str)

    group_diag = pd.read_csv(group_diag_csv, low_memory=False)
    group_diag[str(group_col)] = group_diag[str(group_col)].astype(str)
    plot_gdf = areas.merge(group_diag, on=str(group_col), how="inner")
    if int(plot_gdf.shape[0]) == 0:
        raise SystemExit("no groups after merge; cannot plot")
    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise SystemExit("viz_phase3_spatial_population requires geopandas.") from e
    plot_gdf = gpd.GeoDataFrame(plot_gdf, geometry="geometry", crs=areas.crs)
    plot_gdf["work_stage_class"] = plot_gdf["work_source_stage"].fillna("none").astype(str)

    focus_meta = _select_focus_region(
        plot_gdf,
        search_radius_m=float(args.search_radius_m),
        focus_radius_m=float(args.focus_radius_m),
        min_focus_groups=int(args.min_focus_groups),
    )
    zoom_bounds = tuple(float(v) for v in focus_meta["bounds"])
    zoom_gdf = _clip_to_bounds(plot_gdf, zoom_bounds)
    roads_zoom = _prepare_roads(roads_path, plot_gdf.crs, zoom_bounds)

    overview_home = _sample_points_csv(
        csv_path=persons_csv,
        x_col="home_x",
        y_col="home_y",
        sample_n=int(args.overview_home_sample_n),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed),
    )
    overview_work = _sample_points_csv(
        csv_path=persons_csv,
        x_col="work_x",
        y_col="work_y",
        sample_n=int(args.overview_work_sample_n),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed) + 17,
    )
    zoom_home = _sample_points_csv(
        csv_path=persons_csv,
        x_col="home_x",
        y_col="home_y",
        sample_n=int(args.zoom_home_sample_n),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed) + 31,
        bbox=zoom_bounds,
    )
    zoom_work = _sample_points_csv(
        csv_path=persons_csv,
        x_col="work_x",
        y_col="work_y",
        sample_n=int(args.zoom_work_sample_n),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed) + 47,
        bbox=zoom_bounds,
    )
    html_home = _sample_points_csv(
        csv_path=persons_csv,
        x_col="home_x",
        y_col="home_y",
        sample_n=int(args.html_home_sample_n),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed) + 59,
        bbox=zoom_bounds,
    )
    html_work = _sample_points_csv(
        csv_path=persons_csv,
        x_col="work_x",
        y_col="work_y",
        sample_n=int(args.html_work_sample_n),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed) + 71,
        bbox=zoom_bounds,
    )

    overview_png = out_dir / "spatial_population_overview.png"
    zoom_png = out_dir / "high_density_zoom.png"
    html_path = out_dir / "high_density_interactive.html"
    manifest_path = out_dir / "manifest.json"

    _make_overview(
        plot_gdf=plot_gdf,
        home_pts=overview_home,
        work_pts=overview_work,
        validation=validation,
        out_png=overview_png,
    )
    _make_zoom(
        zoom_gdf=zoom_gdf,
        roads_zoom=roads_zoom,
        home_zoom=zoom_home,
        work_zoom=zoom_work,
        focus_meta=focus_meta,
        out_png=zoom_png,
    )
    html_status = "skipped"
    if not bool(args.skip_html):
        _make_interactive_html(
            zoom_gdf=zoom_gdf,
            roads_zoom=roads_zoom,
            home_zoom=html_home,
            work_zoom=html_work,
            focus_meta=focus_meta,
            out_html=html_path,
            road_simplify_tol=float(args.road_simplify_tol),
        )
        html_status = "ok"

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "source_artifacts": {
            "summary_json": str(run_dir / "metrics" / "summary.json"),
            "validation_json": str(run_dir / "metrics" / "roadloc_validation.json"),
            "persons_csv": str(persons_csv),
            "areas_path": str(areas_path),
            "roads_path": str(roads_path),
            "group_diagnostics_csv": str(group_diag_csv),
        },
        "outputs": {
            "overview_png": str(overview_png),
            "zoom_png": str(zoom_png),
            "interactive_html": str(html_path),
        },
        "html_status": html_status,
        "group_col": group_col,
        "overview_sample_sizes": {
            "home": int(len(overview_home)),
            "work": int(len(overview_work)),
        },
        "zoom_sample_sizes": {
            "home": int(len(zoom_home)),
            "work": int(len(zoom_work)),
        },
        "html_sample_sizes": {
            "home": int(len(html_home)),
            "work": int(len(html_work)),
        },
        "focus_meta": focus_meta,
        "coverage": validation.get("coverage", {}),
    }
    _write_json(manifest_path, manifest)

    print(f"[ok] wrote: {overview_png}")
    print(f"[ok] wrote: {zoom_png}")
    print(f"[ok] wrote: {manifest_path}")
    if html_status == "ok":
        print(f"[ok] wrote: {html_path}")
    else:
        print("[info] skipped interactive HTML")


if __name__ == "__main__":
    main()
