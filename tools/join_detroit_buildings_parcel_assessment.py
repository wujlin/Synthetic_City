#!/usr/bin/env python3
from __future__ import annotations

"""
Join Detroit buildings with Wayne County parcel assessment and derive price tiers.

Intent (PI-aligned):
- Use property assessment as the *core* building-level income proxy.
- Derive a tract-level price tier (Q1-Q5) to serve as the main spatial-economic condition.

Inputs:
- buildings_csv: output of tools/prepare_detroit_buildings_gba.py (must include centroid_lon/lat, footprint_area_m2,
  and preferably tract_geoid).
- parcels_path: Wayne County parcel assessment data (any geopandas-readable format: shp/geojson/gpkg/fgdb layer, etc.)

Outputs:
- buildings_out_csv: buildings_csv + parcel_id + assessed_value + price_per_sqft + price_tier
- buildings_out_csv.metadata.json

KISS notes:
- Spatial join uses building centroid-in-parcel (point-in-polygon). This is usually enough for tiering.
- If multiple buildings share one parcel, we can optionally allocate parcel value by footprint area (or cap_proxy).
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


def _pick_col(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _guess_crs_epsg(gdf) -> int:
    minx, miny, maxx, maxy = gdf.total_bounds
    if max(abs(minx), abs(maxx)) > 180 or max(abs(miny), abs(maxy)) > 90:
        return 3857
    return 4326


def _safe_qcut(values, *, q: int) -> Any:
    """
    pd.qcut can fail with too few unique values; fallback to rank-based binning.
    Returns 1..q integer tiers.
    """
    pd = _require("pandas")
    import numpy as np  # type: ignore

    s = pd.Series(values)
    try:
        bins = pd.qcut(s, q=q, labels=False, duplicates="drop")
        if bins.isna().all():
            raise ValueError("all-NaN bins")
        # qcut may drop some bins; remap to 1..K, then scale to 1..q if needed.
        k = int(bins.max()) + 1
        if k <= 0:
            raise ValueError("invalid bins")
        tiers = (bins.astype(float) + 1.0).to_numpy()
        if k != q:
            # map 1..k to 1..q monotonically
            tiers = np.ceil(tiers / k * q)
        return tiers.astype(int)
    except Exception:
        r = s.rank(method="average", pct=True)
        tiers = (r * q).clip(lower=1.0, upper=float(q))
        return tiers.round().astype(int).to_numpy()


def main() -> None:
    gpd = _require("geopandas")
    pd = _require("pandas")

    p = argparse.ArgumentParser(prog="join_detroit_buildings_parcel_assessment")
    p.add_argument("--buildings_csv", required=True, help="Input buildings CSV (from prepare_detroit_buildings_gba.py).")
    p.add_argument("--parcels_path", required=True, help="Wayne County parcel assessment dataset path.")
    p.add_argument("--parcel_layer", default=None, help="Optional layer name (for GDB/GeoPackage).")
    p.add_argument("--parcel_id_col", default=None, help="Parcel ID column (default: auto-detect).")
    p.add_argument("--assessed_value_col", default=None, help="Assessed value column (default: auto-detect).")
    p.add_argument(
        "--group_for_tier",
        choices=["tract", "puma", "global"],
        default="tract",
        help="Compute price tiers within which group (default: tract).",
    )
    p.add_argument("--n_tiers", type=int, default=5)
    p.add_argument(
        "--allocate_within_parcel",
        choices=["none", "area", "cap_proxy"],
        default="area",
        help="If a parcel contains multiple buildings, allocate assessed value by this weight (default: area).",
    )
    p.add_argument("--out_csv", required=True, help="Output buildings CSV with price tiers.")
    args = p.parse_args()

    buildings_csv = pathlib.Path(args.buildings_csv).expanduser().resolve()
    parcels_path = pathlib.Path(args.parcels_path).expanduser().resolve()
    out_csv = pathlib.Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    b = pd.read_csv(buildings_csv, low_memory=False)
    needed = ["bldg_id", "centroid_lon", "centroid_lat", "footprint_area_m2"]
    missing = [c for c in needed if c not in b.columns]
    if missing:
        raise SystemExit(f"buildings_csv missing columns: {missing}")

    if args.group_for_tier == "tract" and "tract_geoid" not in b.columns:
        raise SystemExit('group_for_tier="tract" requires buildings_csv to have tract_geoid. Re-run prepare_detroit_buildings_gba.py with --tiger_tract_zip.')
    if args.group_for_tier == "puma" and "puma" not in b.columns:
        raise SystemExit('group_for_tier="puma" requires buildings_csv to have puma.')

    b["footprint_area_m2"] = pd.to_numeric(b["footprint_area_m2"], errors="coerce").fillna(0.0).clip(lower=0.0)
    b["cap_proxy"] = pd.to_numeric(b.get("cap_proxy", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)

    # Build centroid points
    b_gdf = gpd.GeoDataFrame(
        b[["bldg_id"]].copy(),
        geometry=gpd.points_from_xy(pd.to_numeric(b["centroid_lon"]), pd.to_numeric(b["centroid_lat"])),
        crs=4326,
    )

    # Load parcels
    if args.parcel_layer:
        parcels = gpd.read_file(parcels_path, layer=args.parcel_layer)
    else:
        parcels = gpd.read_file(parcels_path)
    if parcels.crs is None:
        # Some ArcGIS exports omit CRS. Use a simple heuristic; override by re-exporting with CRS if needed.
        epsg = _guess_crs_epsg(parcels)
        parcels = parcels.set_crs(epsg, allow_override=True)
        print(f"[warn] parcels.crs is None; guessed EPSG:{epsg} from coordinate bounds.")

    # Detect columns
    pid_col = args.parcel_id_col or _pick_col(
        list(parcels.columns),
        ("parcel_id", "PARCEL_ID", "parcelid", "PARCELID", "PARCELNO", "PARCEL_NO", "PID", "OBJECTID"),
    )
    if pid_col is None:
        raise SystemExit(f"Cannot auto-detect parcel_id_col. Columns: {list(parcels.columns)}")

    aval_col = args.assessed_value_col or _pick_col(
        list(parcels.columns),
        (
            "assessed_value",
            "ASSESSED_VALUE",
            "AssessedValue",
            "ASSESSEDVALUE",
            "SEV",
            "STATE_EQUALIZED_VALUE",
            "TOTAL_VALUE",
            "TOTALVAL",
            "TAXABLEVALUE",
        ),
    )
    if aval_col is None:
        raise SystemExit(f"Cannot auto-detect assessed_value_col. Columns: {list(parcels.columns)}")

    # CRS align and join
    parcels = parcels[[pid_col, aval_col, "geometry"]].copy()
    parcels[aval_col] = pd.to_numeric(parcels[aval_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    b_gdf = b_gdf.to_crs(parcels.crs)

    joined = gpd.sjoin(b_gdf, parcels, how="left", predicate="within")
    j = joined.drop(columns=["geometry"]).rename(columns={pid_col: "parcel_id", aval_col: "parcel_assessed_value"})
    b = b.merge(j[["bldg_id", "parcel_id", "parcel_assessed_value"]], on="bldg_id", how="left")
    b["parcel_assessed_value"] = pd.to_numeric(b["parcel_assessed_value"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Allocate parcel assessed value across buildings if requested
    if args.allocate_within_parcel != "none":
        weight_col = "footprint_area_m2" if args.allocate_within_parcel == "area" else "cap_proxy"
        denom = b.groupby("parcel_id", dropna=False)[weight_col].transform("sum").replace(0.0, 1.0)
        frac = (b[weight_col] / denom).clip(lower=0.0, upper=1.0)
        b["assessed_value_alloc"] = b["parcel_assessed_value"] * frac
    else:
        b["assessed_value_alloc"] = b["parcel_assessed_value"]

    # Price per sqft (USD/sqft proxy); convert m2 → sqft
    sqft = b["footprint_area_m2"] * 10.763910416709722
    b["price_per_sqft"] = (b["assessed_value_alloc"] / sqft.replace(0.0, float("nan"))).astype(float)

    # Price tier (Q1..Qn within tract/puma/global)
    if args.group_for_tier == "tract":
        group_key = "tract_geoid"
    elif args.group_for_tier == "puma":
        group_key = "puma"
    else:
        group_key = None

    tiers = []
    if group_key is None:
        tiers = _safe_qcut(b["price_per_sqft"].fillna(0.0), q=int(args.n_tiers))
        b["price_tier"] = tiers
    else:
        b["price_tier"] = 1
        for g, idx in b.groupby(group_key, sort=False).groups.items():
            sub = b.loc[idx, "price_per_sqft"].fillna(0.0)
            b.loc[idx, "price_tier"] = _safe_qcut(sub, q=int(args.n_tiers))

    # Keep output compact and deterministic column order
    cols_out = list(b.columns)
    front = ["bldg_id"]
    for c in ["tract_geoid", "puma", "parcel_id", "parcel_assessed_value", "assessed_value_alloc", "price_per_sqft", "price_tier"]:
        if c in cols_out and c not in front:
            front.append(c)
    rest = [c for c in cols_out if c not in front]
    b = b[front + rest].copy()

    b.to_csv(out_csv, index=False)

    meta = {
        "dataset": "Detroit buildings + Wayne County parcel assessment (joined)",
        "buildings_csv_in": str(buildings_csv),
        "parcels_path": str(parcels_path),
        "parcel_layer": args.parcel_layer,
        "parcel_id_col": str(pid_col),
        "assessed_value_col": str(aval_col),
        "group_for_tier": str(args.group_for_tier),
        "n_tiers": int(args.n_tiers),
        "allocate_within_parcel": str(args.allocate_within_parcel),
        "out_csv": str(out_csv),
        "n_buildings": int(b.shape[0]),
    }
    out_meta = out_csv.with_suffix(".metadata.json")
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_csv} ({meta['n_buildings']} buildings)")
    print(f"[ok] wrote: {out_meta}")


if __name__ == "__main__":
    main()
