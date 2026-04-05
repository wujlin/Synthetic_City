#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pydeck as pdk
from PIL import Image
from playwright.sync_api import sync_playwright
from shapely.geometry import box


PROJECT_ROOT = Path("/Users/jinlin/Desktop/Project/Synthetic_City")
DEFAULT_OUTDIR = PROJECT_ROOT / "figures" / "phase3_detroit_metro_latest"
DEFAULT_MANIFEST = DEFAULT_OUTDIR / "micro_examples_manifest.json"
DEFAULT_ROAD_ZIP = PROJECT_ROOT / "dataset" / "cache" / "geo" / "MI_road_cleaned.shp.zip"
DEFAULT_TRACT_ZIP = PROJECT_ROOT / "dataset" / "cache" / "geo" / "tl_2023_26_tract.zip"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_geodata(path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".zip":
        return gpd.read_file(f"zip://{path}")
    return gpd.read_file(path)


def _clip_to_bounds(gdf: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    geom = box(*bounds)
    try:
        idx = list(gdf.sindex.query(geom, predicate="intersects"))
        subset = gdf.iloc[idx].copy()
    except Exception:
        subset = gdf[gdf.intersects(geom)].copy()
    return subset


def _polygon_records(gdf: gpd.GeoDataFrame) -> list[dict]:
    records: list[dict] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        geoms = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
        for poly in geoms:
            x, y = poly.exterior.xy
            records.append(
                {
                    "tract_geoid": str(row.get("tract_geoid", "")),
                    "polygon": [[float(xx), float(yy)] for xx, yy in zip(x, y)],
                }
            )
    return records


def _path_records(gdf: gpd.GeoDataFrame) -> list[dict]:
    records: list[dict] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        geoms = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
        for part in geoms:
            x, y = part.xy
            records.append(
                {
                    "mtfcc": str(row.get("MTFCC", "")),
                    "path": [[float(xx), float(yy)] for xx, yy in zip(x, y)],
                }
            )
    return records


def _home_columns(points_csv: Path, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    pts = pd.read_csv(points_csv)
    xmin, ymin, xmax, ymax = bounds
    pts = pts.loc[
        (pts["x"] >= xmin) & (pts["x"] <= xmax) & (pts["y"] >= ymin) & (pts["y"] <= ymax)
    ].copy()
    occ = pts.groupby(["x", "y"], as_index=False).size().rename(columns={"size": "residents"})
    occ_gdf = gpd.GeoDataFrame(occ, geometry=gpd.points_from_xy(occ["x"], occ["y"]), crs=4326).to_crs(3857)
    xmin_m, ymin_m, xmax_m, ymax_m = occ_gdf.total_bounds
    cell_size = 70.0
    occ_gdf["ix"] = ((occ_gdf.geometry.x - xmin_m) // cell_size).astype(int)
    occ_gdf["iy"] = ((occ_gdf.geometry.y - ymin_m) // cell_size).astype(int)
    grid = (
        occ_gdf.groupby(["ix", "iy"], as_index=False)
        .agg(
            residents_sum=("residents", "sum"),
            residents_mean=("residents", "mean"),
            used_points=("residents", "size"),
        )
        .copy()
    )
    grid["x_m"] = xmin_m + (grid["ix"] + 0.5) * cell_size
    grid["y_m"] = ymin_m + (grid["iy"] + 0.5) * cell_size
    grid_gdf = gpd.GeoDataFrame(
        grid,
        geometry=gpd.points_from_xy(grid["x_m"], grid["y_m"]),
        crs=3857,
    ).to_crs(4326)
    grid["x"] = grid_gdf.geometry.x
    grid["y"] = grid_gdf.geometry.y
    grid["elevation"] = grid["residents_mean"].clip(lower=1.0, upper=28.0) * 26.0
    grid["fill_color"] = grid["residents_mean"].round().astype(int).map(_warm_color)
    return grid.sort_values(["residents_mean", "residents_sum"], ascending=False).reset_index(drop=True)


def _warm_color(v: int) -> list[int]:
    if v <= 2:
        return [255, 225, 140, 210]
    if v <= 4:
        return [254, 196, 79, 220]
    if v <= 6:
        return [254, 153, 41, 225]
    if v <= 10:
        return [236, 112, 20, 230]
    return [140, 58, 18, 235]


def build_home_deck(
    *,
    tract_zip: Path,
    road_zip: Path,
    manifest_path: Path,
    out_html: Path,
) -> None:
    meta = _read_json(manifest_path)
    home_meta = meta["home_example"]
    bounds = tuple(float(v) for v in home_meta["bounds"])
    selected_tract = str(home_meta["selected_tract_geoid"])

    tracts = _read_geodata(tract_zip)
    tracts["tract_geoid"] = tracts["GEOID"].astype(str)
    tracts = _clip_to_bounds(tracts, bounds).to_crs(4326)
    selected = tracts.loc[tracts["tract_geoid"] == selected_tract].copy()

    roads = _read_geodata(road_zip)[["MTFCC", "geometry"]].copy()
    roads = _clip_to_bounds(roads, bounds).to_crs(4326)
    roads["MTFCC"] = roads["MTFCC"].astype(str)
    roads_bg = roads.loc[~roads["MTFCC"].isin(["S1400", "S1740"])].copy()
    roads_home = roads.loc[roads["MTFCC"].isin(["S1400", "S1740"])].copy()

    columns = _home_columns(Path(meta["home_sample"]), bounds)
    tract_records = _polygon_records(tracts)
    selected_records = _polygon_records(selected)
    road_bg_records = _path_records(roads_bg)
    road_home_records = _path_records(roads_home)

    layers = [
        pdk.Layer(
            "PolygonLayer",
            tract_records,
            get_polygon="polygon",
            get_fill_color=[244, 239, 231, 80],
            get_line_color=[207, 198, 183, 160],
            line_width_min_pixels=1,
            stroked=True,
            filled=True,
            pickable=False,
        ),
        pdk.Layer(
            "PathLayer",
            road_bg_records,
            get_path="path",
            get_color=[211, 204, 191, 160],
            width_min_pixels=1,
            get_width=1.0,
            pickable=False,
        ),
        pdk.Layer(
            "PathLayer",
            road_home_records,
            get_path="path",
            get_color=[187, 164, 126, 210],
            width_min_pixels=1.3,
            get_width=1.6,
            pickable=False,
        ),
        pdk.Layer(
            "PolygonLayer",
            selected_records,
            get_polygon="polygon",
            get_fill_color=[252, 231, 192, 24],
            get_line_color=[201, 122, 29, 240],
            line_width_min_pixels=2.0,
            stroked=True,
            filled=True,
            pickable=False,
        ),
        pdk.Layer(
            "GridCellLayer",
            columns,
            get_position=["x", "y"],
            get_elevation="elevation",
            elevation_scale=1,
            cell_size=70,
            get_fill_color="fill_color",
            extruded=True,
            pickable=True,
            auto_highlight=True,
            opacity=0.92,
        ),
    ]

    xmin, ymin, xmax, ymax = bounds
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            longitude=(xmin + xmax) / 2.0,
            latitude=(ymin + ymax) / 2.0,
            zoom=13.55,
            pitch=42,
            bearing=24,
        ),
        map_provider="carto",
        map_style=pdk.map_styles.CARTO_LIGHT_NO_LABELS,
        tooltip={"text": "Avg residents per home point: {residents_mean}\nUsed home points in cell: {used_points}"},
    )
    deck.to_html(str(out_html), notebook_display=False)


def render_home_deck_png(
    *,
    html_path: Path,
    out_png: Path,
    crop_png: Path | None = None,
    crop_box: tuple[int, int, int, int] = (420, 220, 2840, 1840),
) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-webgl"])
        page = browser.new_page(viewport={"width": 1600, "height": 1100}, device_scale_factor=2)
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        page.wait_for_timeout(2500)
        page.screenshot(path=str(out_png), full_page=True)
        browser.close()
    if crop_png is not None:
        img = Image.open(out_png)
        img.crop(crop_box).save(crop_png)


def main() -> None:
    parser = argparse.ArgumentParser(prog="viz_home_overview_deckgl")
    parser.add_argument("--tract_zip", type=Path, default=DEFAULT_TRACT_ZIP)
    parser.add_argument("--road_zip", type=Path, default=DEFAULT_ROAD_ZIP)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--out_html",
        type=Path,
        default=DEFAULT_OUTDIR / "home_overview_deckgl.html",
    )
    parser.add_argument(
        "--out_png",
        type=Path,
        default=DEFAULT_OUTDIR / "home_overview_deckgl.png",
    )
    parser.add_argument(
        "--crop_png",
        type=Path,
        default=DEFAULT_OUTDIR / "home_overview_deckgl_cropped.png",
    )
    args = parser.parse_args()
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    build_home_deck(
        tract_zip=args.tract_zip,
        road_zip=args.road_zip,
        manifest_path=args.manifest,
        out_html=args.out_html,
    )
    render_home_deck_png(
        html_path=args.out_html,
        out_png=args.out_png,
        crop_png=args.crop_png,
    )
    print(f"[ok] wrote {args.out_html}")
    print(f"[ok] wrote {args.out_png}")
    print(f"[ok] wrote {args.crop_png}")


if __name__ == "__main__":
    main()
