#!/usr/bin/env python3
"""Create statewide tract->PUMA mapping CSV from TIGER shapefiles.

Usage:
    python tools/data/create_tract_puma_mapping.py \
        --tract_zip /path/to/tl_2023_26_tract.zip \
        --puma_zip  /path/to/tl_2023_26_puma20.zip \
        --out_csv   /path/to/tract_puma_mapping_state26.csv

Output columns:
    tract_geoid,puma
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    col_upper = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand.upper() in col_upper:
            return col_upper[cand.upper()]
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Create tract->PUMA mapping from TIGER shapefiles.")
    p.add_argument("--tract_zip", required=True, help="TIGER tract shapefile ZIP.")
    p.add_argument("--puma_zip", required=True, help="TIGER PUMA20 shapefile ZIP.")
    p.add_argument("--out_csv", required=True, help="Output CSV path (columns: tract_geoid,puma).")
    p.add_argument("--statefp", default=None, help="Optional state FIPS filter, e.g. 26.")
    args = p.parse_args()

    import geopandas as gpd
    import pandas as pd

    tract_zip = pathlib.Path(args.tract_zip).expanduser().resolve()
    puma_zip = pathlib.Path(args.puma_zip).expanduser().resolve()
    out_csv = pathlib.Path(args.out_csv).expanduser().resolve()

    if not tract_zip.exists():
        raise SystemExit(f"tract ZIP not found: {tract_zip}")
    if not puma_zip.exists():
        raise SystemExit(f"PUMA ZIP not found: {puma_zip}")

    print(f"[info] loading tracts: {tract_zip}", file=sys.stderr)
    tracts = gpd.read_file(f"zip://{tract_zip}")
    print(f"[info] loading pumas:  {puma_zip}", file=sys.stderr)
    pumas = gpd.read_file(f"zip://{puma_zip}")

    tract_geoid_col = _find_col(list(tracts.columns), ["GEOID", "GEOID20"])
    statefp_col = _find_col(list(tracts.columns), ["STATEFP", "STATEFP20"])
    countyfp_col = _find_col(list(tracts.columns), ["COUNTYFP", "COUNTYFP20"])
    tractce_col = _find_col(list(tracts.columns), ["TRACTCE", "TRACTCE20"])

    puma_col = _find_col(list(pumas.columns), ["PUMACE20", "PUMACE", "GEOID20", "GEOID"])

    if puma_col is None:
        raise SystemExit(f"Cannot find PUMA code column in: {list(pumas.columns)}")

    if tract_geoid_col is None:
        # Fallback: build tract GEOID from components.
        if not (statefp_col and countyfp_col and tractce_col):
            raise SystemExit(
                "Cannot build tract_geoid. Need GEOID/GEOID20 or "
                "STATEFP+COUNTYFP+TRACTCE in tract shapefile."
            )
        tracts = tracts.copy()
        tracts["tract_geoid"] = (
            tracts[statefp_col].astype(str).str.zfill(2)
            + tracts[countyfp_col].astype(str).str.zfill(3)
            + tracts[tractce_col].astype(str).str.zfill(6)
        )
        tract_geoid_col = "tract_geoid"

    if args.statefp:
        if statefp_col is None:
            raise SystemExit("state filter requested but no STATEFP column in tract shapefile.")
        before = len(tracts)
        state = str(args.statefp).zfill(2)
        tracts = tracts[tracts[statefp_col].astype(str).str.zfill(2) == state].copy()
        print(f"[info] filtered statefp={state}: {before} -> {len(tracts)} tracts", file=sys.stderr)

    tracts_pts = tracts[[tract_geoid_col, "geometry"]].copy()
    tracts_pts["geometry"] = tracts_pts.geometry.representative_point()

    if tracts_pts.crs != pumas.crs:
        pumas = pumas.to_crs(tracts_pts.crs)

    print(f"[info] spatial join: {len(tracts_pts)} tracts x {len(pumas)} pumas", file=sys.stderr)
    joined = gpd.sjoin(
        tracts_pts,
        pumas[[puma_col, "geometry"]],
        how="left",
        predicate="within",
    )

    result = pd.DataFrame(
        {
            "tract_geoid": joined[tract_geoid_col].astype(str),
            "puma": joined[puma_col].astype(str),
        }
    )
    result = result.drop_duplicates(subset=["tract_geoid"], keep="first")
    result = result.dropna(subset=["puma"])
    result = result[~result["puma"].isin({"", "nan", "None"})].copy()

    def _normalize_puma(v: str) -> str:
        vv = str(v).strip()
        if vv.isdigit():
            return str(int(vv))
        return vv

    result["puma"] = result["puma"].map(_normalize_puma)
    result = result.sort_values("tract_geoid").reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    print(
        f"[ok] wrote {len(result)} rows, {result['puma'].nunique()} unique PUMAs -> {out_csv}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
