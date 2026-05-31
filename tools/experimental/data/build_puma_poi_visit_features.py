#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import re
from collections import defaultdict
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _slug(text: object) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^0-9a-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _canon_statefp(x: object) -> str:
    raw = "".join(ch for ch in str(x).replace(".0", "").strip() if ch.isdigit())
    return raw.zfill(2) if raw else ""


def _canon_puma5(x: object) -> str:
    raw = "".join(ch for ch in str(x).replace(".0", "").strip() if ch.isdigit())
    return raw.zfill(5) if raw else ""


def _load_target_uids(path: pathlib.Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    usecols = [c for c in ["statefp", "puma", "puma5", "puma_uid", "puma_uid_key"] if c in header.columns]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    if "puma_uid_key" in df.columns:
        uid = df["puma_uid_key"].map(lambda x: "".join(ch for ch in str(x).replace(".0", "") if ch.isdigit()).zfill(7))
        statefp = uid.str[:2]
        puma5 = uid.str[2:]
    elif "puma_uid" in df.columns:
        uid = df["puma_uid"].map(lambda x: "".join(ch for ch in str(x).replace(".0", "") if ch.isdigit()).zfill(7))
        statefp = uid.str[:2]
        puma5 = uid.str[2:]
    elif {"statefp", "puma"}.issubset(df.columns):
        statefp = df["statefp"].map(_canon_statefp)
        puma5 = df["puma"].map(_canon_puma5)
        uid = statefp + puma5
    elif {"statefp", "puma5"}.issubset(df.columns):
        statefp = df["statefp"].map(_canon_statefp)
        puma5 = df["puma5"].map(_canon_puma5)
        uid = statefp + puma5
    else:
        raise SystemExit("target_wide_csv must contain puma_uid/puma_uid_key or statefp+puma")
    out = pd.DataFrame({"statefp": statefp, "puma5": puma5, "puma_uid_key": uid})
    return out.drop_duplicates("puma_uid_key").sort_values("puma_uid_key").reset_index(drop=True)


def _load_puma(path: pathlib.Path, target: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    state_col = "STATEFP20" if "STATEFP20" in gdf.columns else "STATEFP"
    puma_col = "PUMACE20" if "PUMACE20" in gdf.columns else "PUMACE"
    area_col = "ALAND20" if "ALAND20" in gdf.columns else "ALAND"
    gdf["statefp"] = gdf[state_col].map(_canon_statefp)
    gdf["puma5"] = gdf[puma_col].map(_canon_puma5)
    gdf["puma_uid_key"] = gdf["statefp"] + gdf["puma5"]
    keep = set(target["puma_uid_key"])
    gdf = gdf[gdf["puma_uid_key"].isin(keep)].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(4269, allow_override=True)
    gdf = gdf.to_crs(4326)
    gdf["area_km2"] = pd.to_numeric(gdf.get(area_col, np.nan), errors="coerce") / 1e6
    return gdf[["puma_uid_key", "statefp", "puma5", "area_km2", "geometry"]].copy()


def _entropy(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    total = float(vals.sum())
    if total <= 0:
        return 0.0
    p = vals / total
    return float(-np.sum(p * np.log(np.clip(p, 1e-12, None))))


def _gini(vals: np.ndarray) -> float:
    x = np.sort(np.asarray(vals, dtype=np.float64))
    x = x[np.isfinite(x) & (x > 0)]
    n = int(x.size)
    if n == 0:
        return 0.0
    denom = n * float(x.sum())
    if denom <= 0:
        return 0.0
    return float((2.0 * np.sum(np.arange(1, n + 1) * x) / denom) - (n + 1.0) / n)


def _add_weighted_counter(counter: dict[tuple[str, str], float], df: pd.DataFrame, cat_col: str, weight_col: str) -> None:
    if df.empty:
        return
    grouped = df.groupby(["puma_uid_key", cat_col], dropna=False)[weight_col].sum()
    for (uid, cat), val in grouped.items():
        counter[(str(uid), str(cat))] += float(val)


def _top_categories(counter: dict[tuple[str, str], float], top_n: int) -> list[str]:
    totals: dict[str, float] = defaultdict(float)
    for (_, cat), val in counter.items():
        totals[str(cat)] += float(val)
    return [cat for cat, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[: max(0, int(top_n))]]


def _aggregate(
    *,
    poi_csv: pathlib.Path,
    puma_gdf: gpd.GeoDataFrame,
    chunksize: int,
    max_rows: int,
) -> tuple[dict[str, Any], dict[tuple[str, str], float], dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    usecols = [
        "safegraph_place_id",
        "top_category",
        "sub_category",
        "naics_code",
        "latitude",
        "longitude",
        "raw_visit_counts",
        "raw_visitor_counts",
        "distance_from_home",
        "median_dwell",
    ]
    puma_join = puma_gdf[["puma_uid_key", "geometry"]].copy()
    by_puma: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    top_visit: dict[tuple[str, str], float] = defaultdict(float)
    sub_visit: dict[tuple[str, str], float] = defaultdict(float)
    naics2_visit: dict[tuple[str, str], float] = defaultdict(float)
    meta = {
        "rows_seen": 0,
        "coordinate_valid_rows": 0,
        "joined_rows": 0,
        "visited_joined_rows": 0,
        "raw_visit_counts_joined": 0.0,
        "raw_visitor_counts_joined": 0.0,
        "chunksize": int(chunksize),
        "max_rows": int(max_rows),
    }

    for chunk_idx, chunk in enumerate(pd.read_csv(poi_csv, usecols=lambda c: c in usecols, chunksize=chunksize, low_memory=False), start=1):
        if max_rows > 0:
            remaining = max_rows - int(meta["rows_seen"])
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()
        meta["rows_seen"] += int(len(chunk))
        chunk["latitude"] = pd.to_numeric(chunk.get("latitude"), errors="coerce")
        chunk["longitude"] = pd.to_numeric(chunk.get("longitude"), errors="coerce")
        chunk["raw_visit_counts"] = pd.to_numeric(chunk.get("raw_visit_counts"), errors="coerce").fillna(0.0).clip(lower=0.0)
        chunk["raw_visitor_counts"] = pd.to_numeric(chunk.get("raw_visitor_counts"), errors="coerce").fillna(0.0).clip(lower=0.0)
        chunk["distance_from_home"] = pd.to_numeric(chunk.get("distance_from_home"), errors="coerce")
        chunk["median_dwell"] = pd.to_numeric(chunk.get("median_dwell"), errors="coerce")
        chunk = chunk.dropna(subset=["latitude", "longitude"])
        chunk = chunk[chunk["latitude"].between(18.0, 72.0) & chunk["longitude"].between(-180.0, -60.0)].copy()
        meta["coordinate_valid_rows"] += int(len(chunk))
        if chunk.empty:
            continue

        chunk["top_cat"] = chunk.get("top_category", "unknown").fillna("unknown").map(_slug)
        chunk["sub_cat"] = chunk.get("sub_category", "unknown").fillna("unknown").map(_slug)
        naics = pd.to_numeric(chunk.get("naics_code"), errors="coerce")
        chunk["naics2"] = naics.dropna().astype(int).astype(str).str.zfill(6).str[:2].reindex(chunk.index).fillna("unknown")

        points = gpd.GeoDataFrame(
            chunk,
            geometry=gpd.points_from_xy(chunk["longitude"], chunk["latitude"]),
            crs=4326,
        )
        joined = gpd.sjoin(points, puma_join, how="inner", predicate="within")
        if joined.empty:
            continue
        meta["joined_rows"] += int(len(joined))
        meta["visited_joined_rows"] += int((joined["raw_visit_counts"] > 0).sum())
        meta["raw_visit_counts_joined"] += float(joined["raw_visit_counts"].sum())
        meta["raw_visitor_counts_joined"] += float(joined["raw_visitor_counts"].sum())

        joined["visit_x_distance"] = joined["raw_visit_counts"] * joined["distance_from_home"].fillna(0.0)
        joined["visit_x_dwell"] = joined["raw_visit_counts"] * joined["median_dwell"].fillna(0.0)
        joined["has_distance_weight"] = np.where(joined["distance_from_home"].notna(), joined["raw_visit_counts"], 0.0)
        joined["has_dwell_weight"] = np.where(joined["median_dwell"].notna(), joined["raw_visit_counts"], 0.0)
        joined["is_visited"] = (joined["raw_visit_counts"] > 0).astype(float)

        grouped = joined.groupby("puma_uid_key", dropna=False).agg(
            poi_rows=("safegraph_place_id", "count"),
            visited_poi_rows=("is_visited", "sum"),
            raw_visits=("raw_visit_counts", "sum"),
            raw_visitors=("raw_visitor_counts", "sum"),
            visit_x_distance=("visit_x_distance", "sum"),
            visit_x_dwell=("visit_x_dwell", "sum"),
            distance_weight=("has_distance_weight", "sum"),
            dwell_weight=("has_dwell_weight", "sum"),
        )
        for uid, row in grouped.iterrows():
            rec = by_puma[str(uid)]
            for col, val in row.items():
                rec[col] += float(val)

        _add_weighted_counter(top_visit, joined, "top_cat", "raw_visit_counts")
        _add_weighted_counter(sub_visit, joined, "sub_cat", "raw_visit_counts")
        _add_weighted_counter(naics2_visit, joined, "naics2", "raw_visit_counts")

        print(
            f"[chunk {chunk_idx:05d}] rows={meta['rows_seen']:,} joined={meta['joined_rows']:,} "
            f"visits={meta['raw_visit_counts_joined']:,.0f}",
            flush=True,
        )

    return meta, top_visit, sub_visit, naics2_visit, by_puma


def _make_features(
    *,
    target: pd.DataFrame,
    puma_gdf: gpd.GeoDataFrame,
    by_puma: dict[str, dict[str, float]],
    top_visit: dict[tuple[str, str], float],
    sub_visit: dict[tuple[str, str], float],
    naics2_visit: dict[tuple[str, str], float],
    top_sub: int,
    top_naics2: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    area = puma_gdf.drop_duplicates("puma_uid_key").set_index("puma_uid_key")["area_km2"].to_dict()
    top_cats = _top_categories(top_visit, 9999)
    sub_cats = _top_categories(sub_visit, top_sub)
    naics2_cats = _top_categories(naics2_visit, top_naics2)

    rows: list[dict[str, Any]] = []
    for rec in target.to_dict("records"):
        uid = str(rec["puma_uid_key"])
        agg = by_puma.get(uid, {})
        area_km2 = float(area.get(uid, np.nan))
        poi_rows = float(agg.get("poi_rows", 0.0))
        visited_rows = float(agg.get("visited_poi_rows", 0.0))
        visits = float(agg.get("raw_visits", 0.0))
        visitors = float(agg.get("raw_visitors", 0.0))
        row: dict[str, Any] = {
            "statefp": rec["statefp"],
            "puma5": rec["puma5"],
            "puma_uid_key": uid,
            "poi_visit__poi_count": poi_rows,
            "poi_visit__visited_poi_count": visited_rows,
            "poi_visit__visited_poi_share": float(visited_rows / poi_rows) if poi_rows > 0 else 0.0,
            "poi_visit__raw_visit_sum": visits,
            "poi_visit__raw_visitor_sum": visitors,
            "poi_visit__log1p_raw_visit_sum": float(np.log1p(visits)),
            "poi_visit__log1p_raw_visitor_sum": float(np.log1p(visitors)),
            "poi_visit__raw_visit_density_per_km2": float(visits / area_km2) if area_km2 > 0 else 0.0,
            "poi_visit__raw_visitor_density_per_km2": float(visitors / area_km2) if area_km2 > 0 else 0.0,
            "poi_visit__visits_per_poi": float(visits / poi_rows) if poi_rows > 0 else 0.0,
            "poi_visit__visitors_per_poi": float(visitors / poi_rows) if poi_rows > 0 else 0.0,
            "poi_visit__visits_per_visitor": float(visits / visitors) if visitors > 0 else 0.0,
            "poi_visit__visit_weighted_distance_from_home": float(agg.get("visit_x_distance", 0.0) / agg.get("distance_weight", 0.0))
            if agg.get("distance_weight", 0.0) > 0
            else 0.0,
            "poi_visit__visit_weighted_median_dwell": float(agg.get("visit_x_dwell", 0.0) / agg.get("dwell_weight", 0.0))
            if agg.get("dwell_weight", 0.0) > 0
            else 0.0,
        }

        top_vals = np.array([top_visit.get((uid, cat), 0.0) for cat in top_cats], dtype=np.float64)
        sub_vals = np.array([sub_visit.get((uid, cat), 0.0) for cat in sub_cats], dtype=np.float64)
        naics_vals = np.array([naics2_visit.get((uid, cat), 0.0) for cat in naics2_cats], dtype=np.float64)
        row["poi_visit__top_category_entropy"] = _entropy(top_vals)
        row["poi_visit__top_category_gini"] = _gini(top_vals)
        row["poi_visit__top_category_n_nonzero"] = int(np.sum(top_vals > 0))
        row["poi_visit__top_category_top_share"] = float(top_vals.max() / visits) if visits > 0 and top_vals.size else 0.0
        row["poi_visit__sub_category_entropy"] = _entropy(sub_vals)
        row["poi_visit__naics2_entropy"] = _entropy(naics_vals)

        for cat, val in zip(top_cats, top_vals, strict=True):
            row[f"poi_visit__top_share__{cat}"] = float(val / visits) if visits > 0 else 0.0
            row[f"poi_visit__top_log1p_visit__{cat}"] = float(np.log1p(val))
        for cat, val in zip(sub_cats, sub_vals, strict=True):
            row[f"poi_visit__sub_share__{cat}"] = float(val / visits) if visits > 0 else 0.0
        for cat, val in zip(naics2_cats, naics_vals, strict=True):
            row[f"poi_visit__naics2_share__{cat}"] = float(val / visits) if visits > 0 else 0.0
        rows.append(row)

    out = pd.DataFrame(rows)
    meta = {
        "n_pumas": int(out.shape[0]),
        "pumas_with_visit": int((out["poi_visit__raw_visit_sum"] > 0).sum()),
        "n_top_categories": int(len(top_cats)),
        "n_sub_categories_kept": int(len(sub_cats)),
        "n_naics2_categories_kept": int(len(naics2_cats)),
        "top_categories": top_cats,
        "sub_categories": sub_cats,
        "naics2_categories": naics2_cats,
        "n_feature_columns": int(out.shape[1] - 3),
    }
    return out, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate SafeGraph monthly POI visits into PUMA-level visit-weighted functional features.")
    ap.add_argument("--poi_csv", type=pathlib.Path, default=pathlib.Path("/home/jinlin/data/Mobility_Data/merged_poi_201902.csv"))
    ap.add_argument(
        "--puma_shp",
        type=pathlib.Path,
        default=pathlib.Path("data/geo_cache/cb_2020_us_puma20_500k/cb_2020_us_puma20_500k.shp"),
    )
    ap.add_argument(
        "--target_wide_csv",
        type=pathlib.Path,
        default=pathlib.Path(
            "/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/"
            "exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv"
        ),
    )
    ap.add_argument("--out_csv", type=pathlib.Path, default=pathlib.Path("data/us/processed/features/puma_poi_visit_safegraph_201902_us_v1.csv"))
    ap.add_argument("--chunksize", type=int, default=250_000)
    ap.add_argument("--top_sub", type=int, default=80)
    ap.add_argument("--top_naics2", type=int, default=40)
    ap.add_argument("--max_rows", type=int, default=0, help="Debug limit; 0 means full file.")
    args = ap.parse_args()

    target = _load_target_uids(args.target_wide_csv)
    puma_gdf = _load_puma(args.puma_shp, target)
    if puma_gdf.empty:
        raise SystemExit("no target PUMA polygons loaded")

    agg_meta, top_visit, sub_visit, naics2_visit, by_puma = _aggregate(
        poi_csv=args.poi_csv,
        puma_gdf=puma_gdf,
        chunksize=int(args.chunksize),
        max_rows=int(args.max_rows),
    )
    features, feature_meta = _make_features(
        target=target,
        puma_gdf=puma_gdf,
        by_puma=by_puma,
        top_visit=top_visit,
        sub_visit=sub_visit,
        naics2_visit=naics2_visit,
        top_sub=int(args.top_sub),
        top_naics2=int(args.top_naics2),
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out_csv, index=False)
    meta = {
        "created_utc": _utc_ts(),
        "poi_csv": str(args.poi_csv),
        "puma_shp": str(args.puma_shp),
        "target_wide_csv": str(args.target_wide_csv),
        "out_csv": str(args.out_csv),
        "definition": (
            "PUMA-level SafeGraph POI visit-weighted functional features. "
            "Uses raw_visit_counts/raw_visitor_counts and POI categories; does not parse visitor_home_cbgs or visitor_work_cbgs."
        ),
        "aggregation": agg_meta,
        "features": feature_meta,
    }
    _write_json(args.out_csv.with_suffix(args.out_csv.suffix + ".metadata.json"), meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
