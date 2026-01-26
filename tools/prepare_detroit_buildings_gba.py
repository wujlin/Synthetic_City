#!/usr/bin/env python3
from __future__ import annotations

"""
Prepare a lightweight Detroit building table from GlobalBuildingAtlas (GBA) LoD1 tiles.

Goal (KISS):
- Produce a building feature table that can be used as *spatial condition* in diffusion generation.
- Avoid heavy "validation"; only do the minimum geometry operations needed for features.

Outputs (CSV + metadata JSON):
- bldg_id, puma, tract_geoid (optional), footprint_area_m2, height_m, cap_proxy, dist_cbd_km, centroid_lon, centroid_lat

Notes:
- GBA LoD1 GeoJSON tiles may not carry CRS. Empirically, coordinates look like EPSG:3857.
  We guess CRS by coordinate magnitude; override if needed.
"""

import argparse
import json
import pathlib
from typing import Any


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency: {pkg}. Install it in your conda env.\n"
            "Recommended: conda install -c conda-forge geopandas pyproj shapely"
        ) from e


def _guess_crs_epsg(gdf) -> int:
    # Heuristic: lon/lat would be within [-180,180] / [-90,90]; WebMercator is ~1e7 magnitude.
    minx, miny, maxx, maxy = gdf.total_bounds
    if max(abs(minx), abs(maxx)) > 180 or max(abs(miny), abs(maxy)) > 90:
        return 3857
    return 4326


def _pick_col(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _normalize_puma(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
        return str(int(float(value)))
    except Exception:
        return None


def main() -> None:
    gpd = _require("geopandas")
    pd = _require("pandas")
    pyproj = _require("pyproj")

    p = argparse.ArgumentParser(prog="prepare_detroit_buildings_gba")
    p.add_argument("--gba_tile", required=True, help="Path to a GBA LoD1 tile GeoJSON (5x5 deg).")
    p.add_argument("--tiger_place_zip", required=True, help="Path to TIGER place zip (tl_2023_26_place.zip).")
    p.add_argument("--tiger_puma_zip", required=True, help="Path to TIGER puma20 zip (tl_2023_26_puma20.zip).")
    p.add_argument(
        "--tiger_tract_zip",
        default=None,
        help="Optional: TIGER tract zip (tl_2023_26_tract.zip). If provided, outputs tract_geoid.",
    )
    p.add_argument("--city_name", default="Detroit", help='Place name in TIGER (default: "Detroit").')
    p.add_argument("--out_csv", required=True, help="Output CSV path (recommended under processed/buildings/).")
    p.add_argument("--max_buildings", type=int, default=0, help="Optional cap for PoC (0 = no cap).")
    p.add_argument("--cbd_lon", type=float, default=-83.0458)
    p.add_argument("--cbd_lat", type=float, default=42.3314)
    args = p.parse_args()

    gba_tile = pathlib.Path(args.gba_tile).expanduser().resolve()
    tiger_place_zip = pathlib.Path(args.tiger_place_zip).expanduser().resolve()
    tiger_puma_zip = pathlib.Path(args.tiger_puma_zip).expanduser().resolve()
    tiger_tract_zip = pathlib.Path(args.tiger_tract_zip).expanduser().resolve() if args.tiger_tract_zip else None
    out_csv = pathlib.Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- Place boundary (Detroit city polygon) ---
    place = gpd.read_file(f"zip://{tiger_place_zip}")
    if place.crs is None:
        place = place.set_crs(4269, allow_override=True)  # TIGER/NAD83

    name_col = _pick_col(list(place.columns), ("NAME", "NAMELSAD", "NAME20", "NAME10"))
    if name_col is None:
        raise SystemExit(f"Cannot find name column in TIGER place. Columns: {list(place.columns)}")

    city_name = str(args.city_name).strip().lower()
    mask = place[name_col].astype(str).str.strip().str.lower().eq(city_name)
    if not bool(mask.any()):
        # fallback: contains
        mask = place[name_col].astype(str).str.strip().str.lower().str.contains(city_name, regex=False)
    if not bool(mask.any()):
        raise SystemExit(f'City "{args.city_name}" not found in TIGER place by column {name_col}.')

    detroit_poly = place.loc[mask].geometry.unary_union
    detroit_gdf = gpd.GeoDataFrame({"name": [args.city_name]}, geometry=[detroit_poly], crs=place.crs).to_crs(3857)

    # --- Load GBA tile ---
    b = gpd.read_file(gba_tile)
    if b.crs is None:
        epsg = _guess_crs_epsg(b)
        b = b.set_crs(epsg, allow_override=True)
    if int(b.crs.to_epsg() or 0) != 3857:
        b = b.to_crs(3857)

    # Clip to city polygon
    b = gpd.clip(b, detroit_gdf.geometry.iloc[0])
    if b.empty:
        raise SystemExit("No buildings after clipping. Check tile/city boundary/CRS.")

    # --- Features ---
    id_col = _pick_col(list(b.columns), ("id", "ID", "building_id"))
    if id_col is None:
        b["_bldg_id"] = b.index.astype(str)
        id_col = "_bldg_id"

    height_col = _pick_col(list(b.columns), ("height", "Height", "HEIGHT"))
    if height_col is None:
        raise SystemExit(f"Cannot find height column in GBA tile. Columns: {list(b.columns)}")

    b["height_m"] = pd.to_numeric(b[height_col], errors="coerce")
    b["footprint_area_m2"] = b.geometry.area.astype(float)
    b["floors_proxy"] = (b["height_m"] / 3.0).clip(lower=1.0)
    b["cap_proxy"] = (b["footprint_area_m2"] * b["floors_proxy"]).astype(float)

    cent = b.geometry.centroid
    cent_gdf = gpd.GeoDataFrame(b[[id_col]].copy(), geometry=cent, crs=3857)
    cent_ll = cent_gdf.to_crs(4326)
    b["centroid_lon"] = cent_ll.geometry.x.astype(float)
    b["centroid_lat"] = cent_ll.geometry.y.astype(float)

    # Distance to CBD (in km)
    tr = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
    cbd_x, cbd_y = tr.transform(float(args.cbd_lon), float(args.cbd_lat))
    dx = cent_gdf.geometry.x.to_numpy(dtype=float) - float(cbd_x)
    dy = cent_gdf.geometry.y.to_numpy(dtype=float) - float(cbd_y)
    b["dist_cbd_km"] = (dx * dx + dy * dy) ** 0.5 / 1000.0

    # --- PUMA mapping ---
    puma = gpd.read_file(f"zip://{tiger_puma_zip}")
    if puma.crs is None:
        puma = puma.set_crs(4269, allow_override=True)
    puma = puma.to_crs(3857)
    puma_col = _pick_col(list(puma.columns), ("PUMACE20", "PUMA", "PUMACE10"))
    if puma_col is None:
        raise SystemExit(f"Cannot find PUMA code column. Columns: {list(puma.columns)}")

    joined = gpd.sjoin(cent_gdf, puma[[puma_col, "geometry"]], how="left", predicate="within")
    b["puma"] = joined[puma_col].map(_normalize_puma)

    tract_geoid = None
    if tiger_tract_zip is not None:
        tract = gpd.read_file(f"zip://{tiger_tract_zip}")
        if tract.crs is None:
            tract = tract.set_crs(4269, allow_override=True)
        tract = tract.to_crs(3857)
        geoid_col = _pick_col(list(tract.columns), ("GEOID", "GEOID20", "GEOID10"))
        if geoid_col is None:
            raise SystemExit(f"Cannot find tract GEOID column. Columns: {list(tract.columns)}")
        tract_joined = gpd.sjoin(cent_gdf, tract[[geoid_col, "geometry"]], how="left", predicate="within")
        tract_geoid = tract_joined[geoid_col].astype(str)
        b["tract_geoid"] = tract_geoid

    cols = [id_col, "puma"]
    if tract_geoid is not None:
        cols.append("tract_geoid")
    cols += ["footprint_area_m2", "height_m", "cap_proxy", "dist_cbd_km", "centroid_lon", "centroid_lat"]
    out = b[cols].copy()
    out = out.rename(columns={id_col: "bldg_id"})
    out["bldg_id"] = out["bldg_id"].astype(str)
    required = ["puma", "height_m", "footprint_area_m2"]
    if tract_geoid is not None:
        required.append("tract_geoid")
    out = out.dropna(subset=required).copy()

    if int(args.max_buildings) > 0 and out.shape[0] > int(args.max_buildings):
        # Keep the largest buildings by capacity proxy (better for PoC signal).
        out = out.sort_values("cap_proxy", ascending=False).head(int(args.max_buildings)).copy()

    out.to_csv(out_csv, index=False)

    meta = {
        "dataset": "Detroit buildings from GlobalBuildingAtlas LoD1 (prepared)",
        "gba_tile": str(gba_tile),
        "tiger_place_zip": str(tiger_place_zip),
        "tiger_puma_zip": str(tiger_puma_zip),
        "tiger_tract_zip": str(tiger_tract_zip) if tiger_tract_zip is not None else None,
        "city_name": args.city_name,
        "out_csv": str(out_csv),
        "n_buildings": int(out.shape[0]),
        "features": ["footprint_area_m2", "height_m", "cap_proxy", "dist_cbd_km", "centroid_lon", "centroid_lat", "puma"]
        + (["tract_geoid"] if tiger_tract_zip is not None else []),
        "crs_assumption": "GBA guessed CRS; processing in EPSG:3857; centroid exported in EPSG:4326.",
        "cbd_lonlat": [float(args.cbd_lon), float(args.cbd_lat)],
    }
    out_meta = out_csv.with_suffix(".metadata.json")
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_csv} ({meta['n_buildings']} buildings)")
    print(f"[ok] wrote: {out_meta}")


if __name__ == "__main__":
    main()
