#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/jinlin/Desktop/Project/Synthetic_City")
DEFAULT_TRACT_ZIP = PROJECT_ROOT / "dataset" / "cache" / "geo" / "tl_2023_26_tract.zip"
DEFAULT_HOME_SAMPLE = PROJECT_ROOT / "outputs" / "_sync_phase3_detroit_hcenter_20260330" / "overview_samples" / "home_sample_160k.csv"
DEFAULT_BG_METRICS = PROJECT_ROOT / "outputs" / "_sync_phase3_detroit_hcenter_20260330" / "home_bg_spearman_by_tract.csv"
DEFAULT_TRACT_COMPARE = PROJECT_ROOT / "outputs" / "_sync_phase3_detroit_hcenter_20260330" / "home_tract_comparison.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "figures" / "phase3_detroit_metro_latest"

DETROIT_CORE_PREFIXES = ("26099", "26125", "26163")


def _read_geodata(path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".zip":
        return gpd.read_file(f"zip://{path}")
    return gpd.read_file(path)


def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    std = float(s.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - float(s.mean())) / std


def _tract_geometry_metrics(tracts: gpd.GeoDataFrame) -> pd.DataFrame:
    g = tracts.to_crs(3857).copy()
    area = g.geometry.area
    perim = g.geometry.length
    compact = (4.0 * np.pi * area) / np.maximum(perim**2, 1.0)
    out = pd.DataFrame(
        {
            "tract_geoid": tracts["tract_geoid"].astype(str).to_numpy(),
            "area_m2": area.to_numpy(),
            "compactness": np.clip(compact.to_numpy(), 0.0, 1.0),
        }
    )
    return out


def _sample_home_metrics(points_csv: Path, tracts: gpd.GeoDataFrame) -> pd.DataFrame:
    pts = pd.read_csv(points_csv)
    pts = pts.dropna(subset=["x", "y"]).copy()
    gdf = gpd.GeoDataFrame(
        pts,
        geometry=gpd.points_from_xy(pts["x"], pts["y"]),
        crs=tracts.crs,
    )
    joined = gpd.sjoin(
        gdf,
        tracts[["tract_geoid", "geometry"]],
        how="inner",
        predicate="within",
    )
    person_counts = joined.groupby("tract_geoid", as_index=False).size().rename(columns={"size": "sampled_person_points"})
    used = joined.groupby(["tract_geoid", "x", "y"], as_index=False).size().rename(columns={"size": "sampled_residents_per_point"})
    used_stats = (
        used.groupby("tract_geoid", as_index=False)
        .agg(
            sampled_used_home_points=("sampled_residents_per_point", "size"),
            sampled_mean_occ=("sampled_residents_per_point", "mean"),
            sampled_median_occ=("sampled_residents_per_point", "median"),
            sampled_residents_in_5plus=("sampled_residents_per_point", lambda s: float(s[s >= 5].sum())),
        )
    )
    out = person_counts.merge(used_stats, on="tract_geoid", how="left")
    out["sampled_share_residents_5plus"] = (
        out["sampled_residents_in_5plus"] / np.maximum(out["sampled_person_points"], 1.0)
    )
    out = out.drop(columns=["sampled_residents_in_5plus"])
    return out


def _prepare_base_table(
    *,
    tract_zip: Path,
    bg_metrics_csv: Path,
    tract_compare_csv: Path,
    home_sample_csv: Path,
) -> pd.DataFrame:
    tracts = _read_geodata(tract_zip)
    tracts["tract_geoid"] = tracts["GEOID"].astype(str)
    tracts = tracts.loc[tracts["tract_geoid"].str.startswith(DETROIT_CORE_PREFIXES)].copy()

    bg = pd.read_csv(bg_metrics_csv)
    ht = pd.read_csv(tract_compare_csv)
    bg["tract_geoid"] = bg["tract_geoid"].astype(str)
    ht["tract_geoid"] = ht["tract_geoid"].astype(str)
    sample = _sample_home_metrics(home_sample_csv, tracts)
    geom = _tract_geometry_metrics(tracts)

    merged = (
        tracts[["tract_geoid"]]
        .merge(bg, on="tract_geoid", how="left")
        .merge(ht, on="tract_geoid", how="left")
        .merge(sample, on="tract_geoid", how="left")
        .merge(geom, on="tract_geoid", how="left")
    )
    merged["share_abs_diff"] = (merged["left_share"] - merged["right_share"]).abs()
    merged["county_fips"] = merged["tract_geoid"].str[:5]
    return merged


def _rank_validation(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.copy()
    eligible_bg = valid["eligible"].astype("boolean").fillna(False).astype(bool)
    valid["eligible_validation"] = (
        eligible_bg
        & (valid["mobility_total"].fillna(0) >= 50)
        & (valid["synthetic_total"].fillna(0) >= 2000)
        & (valid["n_bg_units"].fillna(0) >= 4)
    )
    sub = valid.loc[valid["eligible_validation"]].copy()
    sub["validation_score"] = (
        1.25 * _zscore(sub["spearman_bg"])
        + 0.90 * _zscore(-sub["share_abs_diff"])
        + 0.60 * _zscore(np.log1p(sub["mobility_total"]))
        + 0.25 * _zscore(sub["compactness"])
    )
    sub = sub.sort_values(["validation_score", "spearman_bg", "mobility_total"], ascending=[False, False, False])
    return sub.reset_index(drop=True)


def _rank_result(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["eligible_result"] = (
        (result["synthetic_total"].fillna(0) >= 2000)
        & (result["sampled_person_points"].fillna(0) >= 60)
        & (result["sampled_used_home_points"].fillna(0) >= 20)
    )
    sub = result.loc[result["eligible_result"]].copy()
    sub["result_score"] = (
        0.95 * _zscore(np.log1p(sub["synthetic_total"]))
        + 1.00 * _zscore(sub["sampled_mean_occ"])
        + 1.05 * _zscore(sub["sampled_share_residents_5plus"])
        + 0.35 * _zscore(sub["compactness"])
    )
    sub = sub.sort_values(
        ["result_score", "sampled_share_residents_5plus", "sampled_mean_occ"],
        ascending=[False, False, False],
    )
    return sub.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="select_phase3_micro_candidates")
    ap.add_argument("--tract_zip", type=Path, default=DEFAULT_TRACT_ZIP)
    ap.add_argument("--home_sample_csv", type=Path, default=DEFAULT_HOME_SAMPLE)
    ap.add_argument("--bg_metrics_csv", type=Path, default=DEFAULT_BG_METRICS)
    ap.add_argument("--tract_compare_csv", type=Path, default=DEFAULT_TRACT_COMPARE)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    base = _prepare_base_table(
        tract_zip=args.tract_zip,
        bg_metrics_csv=args.bg_metrics_csv,
        tract_compare_csv=args.tract_compare_csv,
        home_sample_csv=args.home_sample_csv,
    )
    validation = _rank_validation(base)
    result = _rank_result(base)

    base_path = args.outdir / "home_micro_candidate_base.csv"
    result_path = args.outdir / "home_micro_candidate_result.csv"
    validation_path = args.outdir / "home_micro_candidate_validation.csv"
    manifest_path = args.outdir / "home_micro_candidate_manifest.json"

    base.to_csv(base_path, index=False)
    result.to_csv(result_path, index=False)
    validation.to_csv(validation_path, index=False)

    manifest = {
        "detroit_core_prefixes": list(DETROIT_CORE_PREFIXES),
        "selection_rules": {
            "result_micro": {
                "eligibility": {
                    "synthetic_total_min": 2000,
                    "sampled_person_points_min": 60,
                    "sampled_used_home_points_min": 20,
                },
                "score_terms": [
                    "log(synthetic_total)",
                    "sampled_mean_occ",
                    "sampled_share_residents_5plus",
                    "compactness",
                ],
            },
            "validation_micro": {
                "eligibility": {
                    "eligible_bg": True,
                    "mobility_total_min": 50,
                    "synthetic_total_min": 2000,
                    "n_bg_units_min": 4,
                },
                "score_terms": [
                    "spearman_bg",
                    "-share_abs_diff",
                    "log(mobility_total)",
                    "compactness",
                ],
            },
        },
        "artifacts": {
            "base_csv": str(base_path),
            "result_csv": str(result_path),
            "validation_csv": str(validation_path),
        },
        "top_result_candidates": result.head(10)[
            [
                "tract_geoid",
                "county_fips",
                "result_score",
                "synthetic_total",
                "sampled_mean_occ",
                "sampled_share_residents_5plus",
                "compactness",
            ]
        ].to_dict(orient="records"),
        "top_validation_candidates": validation.head(10)[
            [
                "tract_geoid",
                "county_fips",
                "validation_score",
                "spearman_bg",
                "share_abs_diff",
                "mobility_total",
                "synthetic_total",
                "compactness",
            ]
        ].to_dict(orient="records"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"[ok] wrote {base_path}")
    print(f"[ok] wrote {result_path}")
    print(f"[ok] wrote {validation_path}")
    print(f"[ok] wrote {manifest_path}")
    print("[top result]")
    print(
        result.head(10)[
            ["tract_geoid", "result_score", "synthetic_total", "sampled_mean_occ", "sampled_share_residents_5plus"]
        ].to_string(index=False)
    )
    print("[top validation]")
    print(
        validation.head(10)[
            ["tract_geoid", "validation_score", "spearman_bg", "share_abs_diff", "mobility_total"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
