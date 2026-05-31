#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import re
import sys
from collections import Counter
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


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(np.clip(p, 1e-12, None))))


def _gini_from_counts(counts: np.ndarray) -> float:
    x = np.sort(np.asarray(counts, dtype=np.float64))
    x = x[x > 0]
    n = x.size
    if n == 0:
        return 0.0
    denom = n * np.sum(x)
    if denom <= 0.0:
        return 0.0
    return float((2.0 * np.sum(np.arange(1, n + 1) * x) / denom) - (n + 1.0) / n)


def _aggregate_counts(
    *,
    poi_dir: pathlib.Path,
    puma_gdf: gpd.GeoDataFrame,
    chunksize: int,
    active_only: bool,
    drop_duplicates: bool,
    max_files: int,
) -> tuple[Counter[str], Counter[tuple[str, str]], Counter[tuple[str, str]], Counter[tuple[str, str]], dict[str, Any]]:
    files = sorted(poi_dir.glob("Global_POI_Data-*.csv"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise SystemExit(f"no Global_POI_Data-*.csv files found under {poi_dir}")

    usecols = [
        "COUNTRY_CODE",
        "LATITUDE",
        "LONGITUDE",
        "MAIN_CATEGORY",
        "SUB_CATEGORY",
        "BUSINESS_CATEGORY",
        "DATAPLOR_STATUS",
        "DUPLICATE_ID",
    ]
    puma_join = puma_gdf[["puma_uid_key", "geometry"]].copy()
    total_by_puma: Counter[str] = Counter()
    main_by_puma: Counter[tuple[str, str]] = Counter()
    sub_by_puma: Counter[tuple[str, str]] = Counter()
    business_by_puma: Counter[tuple[str, str]] = Counter()
    total_rows = 0
    us_rows = 0
    joined_rows = 0

    for file_idx, path in enumerate(files, start=1):
        for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=chunksize, low_memory=False):
            total_rows += int(len(chunk))
            if "COUNTRY_CODE" in chunk.columns:
                chunk = chunk[chunk["COUNTRY_CODE"].astype(str).str.lower().eq("us")].copy()
            if active_only and "DATAPLOR_STATUS" in chunk.columns:
                chunk = chunk[chunk["DATAPLOR_STATUS"].astype(str).str.lower().eq("active")].copy()
            if drop_duplicates and "DUPLICATE_ID" in chunk.columns:
                chunk = chunk[chunk["DUPLICATE_ID"].isna()].copy()
            chunk["LATITUDE"] = pd.to_numeric(chunk.get("LATITUDE"), errors="coerce")
            chunk["LONGITUDE"] = pd.to_numeric(chunk.get("LONGITUDE"), errors="coerce")
            chunk = chunk.dropna(subset=["LATITUDE", "LONGITUDE"])
            chunk = chunk[
                chunk["LATITUDE"].between(18.0, 72.0) & chunk["LONGITUDE"].between(-180.0, -60.0)
            ].copy()
            us_rows += int(len(chunk))
            if chunk.empty:
                continue

            points = gpd.GeoDataFrame(
                chunk,
                geometry=gpd.points_from_xy(chunk["LONGITUDE"], chunk["LATITUDE"]),
                crs=4326,
            )
            joined = gpd.sjoin(points, puma_join, how="inner", predicate="within")
            if joined.empty:
                continue
            joined_rows += int(len(joined))
            joined["main"] = joined.get("MAIN_CATEGORY", "unknown").fillna("unknown").map(_slug)
            joined["sub"] = joined.get("SUB_CATEGORY", "unknown").fillna("unknown").map(_slug)
            joined["business"] = joined.get("BUSINESS_CATEGORY", "unknown").fillna("unknown").map(_slug)

            total_by_puma.update(joined["puma_uid_key"].astype(str).tolist())
            main_by_puma.update(map(tuple, joined[["puma_uid_key", "main"]].astype(str).to_numpy()))
            sub_by_puma.update(map(tuple, joined[["puma_uid_key", "sub"]].astype(str).to_numpy()))
            business_by_puma.update(map(tuple, joined[["puma_uid_key", "business"]].astype(str).to_numpy()))

        print(f"[file {file_idx:03d}/{len(files):03d}] {path.name} total_rows={total_rows} us_rows={us_rows} joined={joined_rows}", flush=True)

    meta = {
        "n_files": len(files),
        "total_rows_seen": total_rows,
        "us_candidate_rows": us_rows,
        "joined_rows": joined_rows,
        "active_only": active_only,
        "drop_duplicates": drop_duplicates,
        "chunksize": chunksize,
    }
    return total_by_puma, main_by_puma, sub_by_puma, business_by_puma, meta


def _top_categories(counter: Counter[tuple[str, str]], top_n: int) -> list[str]:
    c: Counter[str] = Counter()
    for (_, cat), n in counter.items():
        c[cat] += int(n)
    return [cat for cat, _ in c.most_common(max(0, int(top_n)))]


def _make_features(
    target: pd.DataFrame,
    puma_gdf: gpd.GeoDataFrame,
    total_by_puma: Counter[str],
    main_by_puma: Counter[tuple[str, str]],
    sub_by_puma: Counter[tuple[str, str]],
    business_by_puma: Counter[tuple[str, str]],
    *,
    top_sub: int,
    top_business: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    area = puma_gdf.drop_duplicates("puma_uid_key").set_index("puma_uid_key")["area_km2"].to_dict()
    main_cats = _top_categories(main_by_puma, 9999)
    sub_cats = _top_categories(sub_by_puma, top_sub)
    business_cats = _top_categories(business_by_puma, top_business)

    rows: list[dict[str, Any]] = []
    for rec in target.to_dict("records"):
        uid = str(rec["puma_uid_key"])
        total = float(total_by_puma.get(uid, 0))
        area_km2 = float(area.get(uid, np.nan))
        row: dict[str, Any] = {
            "statefp": rec["statefp"],
            "puma5": rec["puma5"],
            "puma_uid_key": uid,
            "poi__total_count": total,
            "poi__log1p_total_count": float(np.log1p(total)),
            "poi__count_density_per_km2": float(total / area_km2) if area_km2 and area_km2 > 0 else 0.0,
            "poi__log1p_count_density_per_km2": float(np.log1p(total / area_km2)) if area_km2 and area_km2 > 0 else 0.0,
        }
        main_counts = np.array([main_by_puma.get((uid, cat), 0) for cat in main_cats], dtype=np.float64)
        row["poi__main_entropy"] = _entropy_from_counts(main_counts)
        row["poi__main_gini"] = _gini_from_counts(main_counts)
        row["poi__main_n_nonzero"] = int(np.sum(main_counts > 0))
        row["poi__main_top_share"] = float(main_counts.max() / total) if total > 0 and main_counts.size else 0.0
        for cat, count in zip(main_cats, main_counts, strict=True):
            row[f"poi__main_count__{cat}"] = float(count)
            row[f"poi__main_share__{cat}"] = float(count / total) if total > 0 else 0.0
        for cat in sub_cats:
            count = float(sub_by_puma.get((uid, cat), 0))
            row[f"poi__sub_share__{cat}"] = float(count / total) if total > 0 else 0.0
            row[f"poi__sub_log1p_count__{cat}"] = float(np.log1p(count))
        for cat in business_cats:
            count = float(business_by_puma.get((uid, cat), 0))
            row[f"poi__business_share__{cat}"] = float(count / total) if total > 0 else 0.0
        rows.append(row)

    feature_df = pd.DataFrame(rows)
    meta = {
        "n_pumas": int(feature_df.shape[0]),
        "n_main_categories": int(len(main_cats)),
        "n_sub_categories_kept": int(len(sub_cats)),
        "n_business_categories_kept": int(len(business_cats)),
        "main_categories": main_cats,
        "sub_categories": sub_cats,
        "business_categories": business_cats,
        "n_feature_columns": int(feature_df.shape[1] - 3),
        "pumas_with_poi": int((feature_df["poi__total_count"] > 0).sum()),
    }
    return feature_df, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate Dataplor POI category composition into PUMA-level land-use features.")
    ap.add_argument("--poi_dir", type=pathlib.Path, default=pathlib.Path("/home/jinlin/data/geoexplicit_data/dataplor_unzip"))
    ap.add_argument("--puma_shp", type=pathlib.Path, default=pathlib.Path("data/geo_cache/cb_2020_us_puma20_500k/cb_2020_us_puma20_500k.shp"))
    ap.add_argument("--target_wide_csv", type=pathlib.Path, default=pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv"))
    ap.add_argument("--out_csv", type=pathlib.Path, default=pathlib.Path("data/us/processed/features/puma_poi_landuse_dataplor_us_v1.csv"))
    ap.add_argument("--chunksize", type=int, default=250_000)
    ap.add_argument("--top_sub", type=int, default=80)
    ap.add_argument("--top_business", type=int, default=80)
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--include_inactive", action="store_true")
    ap.add_argument("--keep_duplicates", action="store_true")
    args = ap.parse_args()

    target = _load_target_uids(args.target_wide_csv)
    puma_gdf = _load_puma(args.puma_shp, target)
    if puma_gdf.empty:
        raise SystemExit("no target PUMA polygons loaded")

    total, main, sub, business, agg_meta = _aggregate_counts(
        poi_dir=args.poi_dir,
        puma_gdf=puma_gdf,
        chunksize=int(args.chunksize),
        active_only=not bool(args.include_inactive),
        drop_duplicates=not bool(args.keep_duplicates),
        max_files=int(args.max_files),
    )
    features, feature_meta = _make_features(
        target,
        puma_gdf,
        total,
        main,
        sub,
        business,
        top_sub=int(args.top_sub),
        top_business=int(args.top_business),
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out_csv, index=False)
    meta = {
        "created_utc": _utc_ts(),
        "poi_dir": str(args.poi_dir),
        "puma_shp": str(args.puma_shp),
        "target_wide_csv": str(args.target_wide_csv),
        "out_csv": str(args.out_csv),
        "definition": "PUMA-level POI land-use composition from Dataplor categories; no visit/popularity data.",
        "aggregation": agg_meta,
        "features": feature_meta,
    }
    _write_json(args.out_csv.with_suffix(args.out_csv.suffix + ".metadata.json"), meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
