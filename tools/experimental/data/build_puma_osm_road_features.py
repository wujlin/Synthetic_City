#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as exc:
        raise RuntimeError(f"Missing dependency: {pkg}. Install it in the WSA dpl environment.") from exc


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


STATE_NAME_TO_FIPS = {
    "alabama": "01",
    "alaska": "02",
    "arizona": "04",
    "arkansas": "05",
    "california": "06",
    "colorado": "08",
    "connecticut": "09",
    "delaware": "10",
    "district-of-columbia": "11",
    "florida": "12",
    "georgia": "13",
    "hawaii": "15",
    "idaho": "16",
    "illinois": "17",
    "indiana": "18",
    "iowa": "19",
    "kansas": "20",
    "kentucky": "21",
    "louisiana": "22",
    "maine": "23",
    "maryland": "24",
    "massachusetts": "25",
    "michigan": "26",
    "minnesota": "27",
    "mississippi": "28",
    "missouri": "29",
    "montana": "30",
    "nebraska": "31",
    "nevada": "32",
    "new-hampshire": "33",
    "new-jersey": "34",
    "new-mexico": "35",
    "new-york": "36",
    "north-carolina": "37",
    "north-dakota": "38",
    "ohio": "39",
    "oklahoma": "40",
    "oregon": "41",
    "pennsylvania": "42",
    "rhode-island": "44",
    "south-carolina": "45",
    "south-dakota": "46",
    "tennessee": "47",
    "texas": "48",
    "utah": "49",
    "vermont": "50",
    "virginia": "51",
    "washington": "53",
    "west-virginia": "54",
    "wisconsin": "55",
    "wyoming": "56",
}

ROAD_GROUPS = ("major", "arterial", "local", "service", "other")


def _canon_statefp(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return str(int(float(s))).zfill(2) if s.replace(".", "", 1).isdigit() else s.zfill(2)


def _canon_puma5(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return str(int(float(s))).zfill(5) if s.replace(".", "", 1).isdigit() else s.zfill(5)


def _pick_col(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _state_from_pbf(path: pathlib.Path) -> str:
    name = path.name.lower().replace("-latest.osm.pbf", "")
    return STATE_NAME_TO_FIPS.get(name, "")


def _road_group(highway: object) -> str:
    if isinstance(highway, (list, tuple, set)):
        text = " ".join(str(x).lower() for x in highway)
    else:
        text = str(highway).lower()
    if any(x in text for x in ("motorway", "trunk", "primary", "secondary")):
        return "major"
    if "tertiary" in text:
        return "arterial"
    if any(x in text for x in ("residential", "living_street", "unclassified")):
        return "local"
    if "service" in text:
        return "service"
    return "other"


def _parse_lanes(x: object) -> float:
    if pd.isna(x):
        return float("nan")
    vals = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", str(x))]
    return float(vals[0]) if vals else float("nan")


def _entropy(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = np.clip(vals, 0.0, None)
    total = float(vals.sum())
    if total <= 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / math.log(len(vals))) if len(vals) > 1 else 0.0


@dataclass
class Acc:
    n_segments: int = 0
    total_km: float = 0.0
    length_by_group: dict[str, float] = field(default_factory=lambda: {g: 0.0 for g in ROAD_GROUPS})
    len_values: list[float] = field(default_factory=list)
    lanes_values: list[float] = field(default_factory=list)
    lit_km: float = 0.0
    bridge_km: float = 0.0
    tunnel_km: float = 0.0


def _base_row(puma_uid: str, statefp: str, puma5: str, geoid: str, area_km2: float, acc: Acc | None) -> dict[str, Any]:
    acc = acc or Acc()
    lengths = np.asarray(acc.len_values, dtype=float)
    lanes = np.asarray(acc.lanes_values, dtype=float)
    lanes = lanes[np.isfinite(lanes)]
    row: dict[str, Any] = {
        "puma_uid": puma_uid,
        "statefp": statefp,
        "puma5": puma5,
        "geoid": geoid,
        "puma_area_km2": float(area_km2),
        "osm_road_segment_count": int(acc.n_segments),
        "osm_road_segment_count_per_km2": float(acc.n_segments / area_km2) if area_km2 > 0 else float("nan"),
        "osm_road_total_km": float(acc.total_km),
        "osm_road_total_km_per_km2": float(acc.total_km / area_km2) if area_km2 > 0 else float("nan"),
        "osm_road_length_mean_km": float(np.mean(lengths)) if lengths.size else 0.0,
        "osm_road_length_p90_km": float(np.percentile(lengths, 90)) if lengths.size else 0.0,
        "osm_road_lanes_mean": float(np.mean(lanes)) if lanes.size else float("nan"),
        "osm_road_lit_share_km": float(acc.lit_km / acc.total_km) if acc.total_km > 0 else 0.0,
        "osm_road_bridge_share_km": float(acc.bridge_km / acc.total_km) if acc.total_km > 0 else 0.0,
        "osm_road_tunnel_share_km": float(acc.tunnel_km / acc.total_km) if acc.total_km > 0 else 0.0,
    }
    group_lengths = np.asarray([acc.length_by_group[g] for g in ROAD_GROUPS], dtype=float)
    row["osm_road_group_entropy"] = _entropy(group_lengths)
    for g in ROAD_GROUPS:
        km = float(acc.length_by_group[g])
        row[f"osm_road_{g}_km"] = km
        row[f"osm_road_{g}_km_per_km2"] = float(km / area_km2) if area_km2 > 0 else float("nan")
        row[f"osm_road_{g}_share_km"] = float(km / acc.total_km) if acc.total_km > 0 else 0.0
    return row


def main() -> int:
    gpd = _require("geopandas")
    pyrosm = _require("pyrosm")

    ap = argparse.ArgumentParser(description="Aggregate local OSM PBF road networks to PUMA-level road/accessibility features.")
    ap.add_argument("--osm_dir", type=pathlib.Path, default=pathlib.Path("/home/jinlin/data/geoexplicit_data/osm"))
    ap.add_argument(
        "--puma_shp",
        type=pathlib.Path,
        default=pathlib.Path(
            "/home/jinlin/projects/Synthetic_City/data/geo_cache/"
            "cb_2020_us_puma20_500k/cb_2020_us_puma20_500k.shp"
        ),
    )
    ap.add_argument("--statefps", default="", help="Optional comma-separated state FIPS subset.")
    ap.add_argument("--out_csv", type=pathlib.Path, required=True)
    ap.add_argument("--out_metadata_json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    osm_dir = args.osm_dir.expanduser().resolve()
    puma_shp = args.puma_shp.expanduser().resolve()
    out_csv = args.out_csv.expanduser().resolve()
    out_meta = (
        args.out_metadata_json.expanduser().resolve()
        if args.out_metadata_json is not None
        else out_csv.with_suffix(".metadata.json")
    )
    requested_statefps = [_canon_statefp(x.strip()) for x in str(args.statefps).split(",") if x.strip()]

    pbf_by_state: dict[str, pathlib.Path] = {}
    for pbf in sorted(osm_dir.glob("*-latest.osm.pbf")):
        statefp = _state_from_pbf(pbf)
        if statefp:
            pbf_by_state[statefp] = pbf
    if requested_statefps:
        pbf_by_state = {s: p for s, p in pbf_by_state.items() if s in set(requested_statefps)}
    if not pbf_by_state:
        raise SystemExit(f"No usable state PBFs found in {osm_dir}")

    pumas = gpd.read_file(puma_shp)
    state_col = _pick_col(list(pumas.columns), ("STATEFP20", "STATEFP", "STATEFP10"))
    puma_col = _pick_col(list(pumas.columns), ("PUMACE20", "PUMA", "PUMACE10"))
    geoid_col = _pick_col(list(pumas.columns), ("GEOID20", "GEOID", "GEOID10"))
    if state_col is None or puma_col is None:
        raise SystemExit(f"Cannot find state/PUMA columns in {puma_shp}: {list(pumas.columns)}")
    pumas["statefp"] = pumas[state_col].map(_canon_statefp)
    pumas["puma5"] = pumas[puma_col].map(_canon_puma5)
    pumas = pumas[pumas["statefp"].isin(pbf_by_state)].copy()
    pumas["puma_uid"] = pumas["statefp"] + pumas["puma5"]
    pumas["geoid"] = pumas[geoid_col].astype(str) if geoid_col is not None else pumas["puma_uid"]
    if pumas.crs is None:
        pumas = pumas.set_crs(4269, allow_override=True)
    pumas_ll = pumas.to_crs(4326)
    pumas_area = pumas.to_crs(5070)
    puma_area = dict(zip(pumas["puma_uid"], (pumas_area.geometry.area / 1_000_000.0).astype(float)))

    accs: dict[str, Acc] = defaultdict(Acc)
    state_records: list[dict[str, Any]] = []
    for statefp, pbf_path in sorted(pbf_by_state.items()):
        state_pumas = pumas_ll[pumas_ll["statefp"] == statefp].copy()
        if state_pumas.empty:
            continue
        print(f"[state] {statefp} {pbf_path}", flush=True)
        try:
            roads = pyrosm.OSM(str(pbf_path)).get_network(network_type="driving")
        except Exception as exc:
            state_records.append({"statefp": statefp, "pbf": str(pbf_path), "status": "failed", "error": str(exc)})
            print(f"[warn] failed {statefp}: {exc}", flush=True)
            continue
        if roads is None or roads.empty:
            state_records.append({"statefp": statefp, "pbf": str(pbf_path), "status": "empty"})
            continue
        roads = roads[roads.geometry.notna()].copy()
        if roads.crs is None:
            roads = roads.set_crs(4326, allow_override=True)
        roads_proj = roads.to_crs(5070)
        lengths = roads_proj.geometry.length.to_numpy(dtype=float) / 1000.0
        midpoints = roads_proj.geometry.interpolate(0.5, normalized=True)
        pts = gpd.GeoDataFrame(
            {
                "row_id": np.arange(len(roads), dtype=np.int64),
                "length_km": lengths,
                "group": roads.get("highway", pd.Series([""] * len(roads), index=roads.index)).map(_road_group).to_numpy(),
                "lanes": roads.get("lanes", pd.Series([np.nan] * len(roads), index=roads.index)).map(_parse_lanes).to_numpy(),
                "lit": roads.get("lit", pd.Series([""] * len(roads), index=roads.index)).astype(str).str.lower().to_numpy(),
                "bridge": roads.get("bridge", pd.Series([""] * len(roads), index=roads.index)).astype(str).str.lower().to_numpy(),
                "tunnel": roads.get("tunnel", pd.Series([""] * len(roads), index=roads.index)).astype(str).str.lower().to_numpy(),
            },
            geometry=midpoints,
            crs=5070,
        ).to_crs(4326)
        joined = gpd.sjoin(pts, state_pumas[["puma_uid", "geometry"]], how="inner", predicate="within")
        if joined.empty:
            state_records.append({"statefp": statefp, "pbf": str(pbf_path), "status": "unassigned", "road_rows": int(len(roads))})
            continue
        for puma_uid, sub in joined.groupby("puma_uid", sort=False):
            acc = accs[str(puma_uid)]
            lk = sub["length_km"].to_numpy(dtype=float)
            acc.n_segments += int(sub.shape[0])
            acc.total_km += float(np.nansum(lk))
            acc.len_values.extend(lk[np.isfinite(lk)].tolist())
            acc.lanes_values.extend(pd.to_numeric(sub["lanes"], errors="coerce").to_numpy(dtype=float).tolist())
            for g, gsub in sub.groupby("group", sort=False):
                if g in acc.length_by_group:
                    acc.length_by_group[g] += float(np.nansum(gsub["length_km"].to_numpy(dtype=float)))
            acc.lit_km += float(np.nansum(sub.loc[sub["lit"].isin(["yes", "true", "1"]), "length_km"].to_numpy(dtype=float)))
            acc.bridge_km += float(np.nansum(sub.loc[~sub["bridge"].isin(["", "no", "none", "nan"]), "length_km"].to_numpy(dtype=float)))
            acc.tunnel_km += float(np.nansum(sub.loc[~sub["tunnel"].isin(["", "no", "none", "nan"]), "length_km"].to_numpy(dtype=float)))
        state_records.append(
            {
                "statefp": statefp,
                "pbf": str(pbf_path),
                "status": "ok",
                "road_rows": int(len(roads)),
                "assigned_rows": int(joined.shape[0]),
                "pumas_with_roads": int(joined["puma_uid"].nunique()),
            }
        )
        print(f"[ok] {statefp} roads={len(roads):,} assigned={joined.shape[0]:,} pumas={joined['puma_uid'].nunique()}", flush=True)

    rows: list[dict[str, Any]] = []
    key_df = pumas[["puma_uid", "statefp", "puma5", "geoid"]].drop_duplicates("puma_uid").sort_values(["statefp", "puma5"])
    for r in key_df.itertuples(index=False):
        uid = str(r.puma_uid)
        rows.append(_base_row(uid, str(r.statefp), str(r.puma5), str(r.geoid), float(puma_area.get(uid, float("nan"))), accs.get(uid)))
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    meta = {
        "dataset": "PUMA-level OSM road/accessibility features from local state PBFs",
        "osm_dir": str(osm_dir),
        "puma_shp": str(puma_shp),
        "out_csv": str(out_csv),
        "out_metadata_json": str(out_meta),
        "statefps": sorted(pbf_by_state),
        "n_pumas": int(out.shape[0]),
        "state_records": state_records,
        "feature_columns": [c for c in out.columns if c not in {"puma_uid", "statefp", "puma5", "geoid"}],
        "notes": [
            "Road segments are assigned to PUMAs by midpoint, not clipped by polygon boundaries.",
            "Lengths are computed in EPSG:5070.",
            "This is a coverage-limited pilot because only local state PBFs are used.",
        ],
    }
    _write_json(out_meta, meta)
    print(f"[done] wrote {out_csv} rows={out.shape[0]} cols={out.shape[1]}")
    print(f"[done] wrote {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
