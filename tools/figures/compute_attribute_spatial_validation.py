#!/usr/bin/env python3
from __future__ import annotations

"""
Compute tract-level ACS consistency metrics for selected residential subgroups.

Why:
- Figure 4 now shows synthetic residential composition maps for several resident groups.
- We want a reproducible quantitative check against tract-level ACS tables aligned to
  the same Detroit study area.

Important scope note:
- These metrics compare the synthetic residential product against tract-level ACS
  subgroup references.
- They are best interpreted as ACS consistency checks for attribute-conditioned
  spatial patterns, not as fully external validation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/jinlin/Desktop/Project/Synthetic_City")
DEFAULT_SYNTHETIC = (
    PROJECT_ROOT / "outputs" / "_sync_phase3_detroit_hcenter_20260330" / "attribute_spatial_home_shares.csv"
)
DEFAULT_CENSUS_DIR = PROJECT_ROOT / "dataset" / "census"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "_sync_phase3_detroit_hcenter_20260330"


def _cosine(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 0.0 else float("nan")


def _tvd_from_counts(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left_share = left / max(float(left.sum()), 1.0)
    right_share = right / max(float(right.sum()), 1.0)
    tvd = float(0.5 * np.abs(left_share - right_share).sum())
    cosine = _cosine(left_share, right_share)
    return tvd, cosine


def _sum_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = None
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
        out = vals if out is None else (out + vals)
    return out if out is not None else pd.Series(np.zeros(len(df), dtype=float), index=df.index)


def _load_reference_tables(census_dir: Path, tract_geoids: set[str]) -> pd.DataFrame:
    b01001 = pd.read_csv(
        census_dir / "acs5_2022_B01001_tract_michigan.csv.gz",
        compression="gzip",
        dtype={"GEOID": str},
        low_memory=False,
    )
    b15003 = pd.read_csv(
        census_dir / "acs5_2022_B15003_tract_michigan.csv.gz",
        compression="gzip",
        dtype={"GEOID": str},
        low_memory=False,
    )
    b23025 = pd.read_csv(
        census_dir / "acs5_2022_B23025_tract_michigan.csv.gz",
        compression="gzip",
        dtype={"GEOID": str},
        low_memory=False,
    )
    b20001 = pd.read_csv(
        census_dir / "acs5_2022_B20001_tract_michigan.csv.gz",
        compression="gzip",
        dtype={"GEOID": str},
        low_memory=False,
    )

    b01001 = b01001.loc[b01001["GEOID"].isin(tract_geoids)].copy()
    b15003 = b15003.loc[b15003["GEOID"].isin(tract_geoids)].copy()
    b23025 = b23025.loc[b23025["GEOID"].isin(tract_geoids)].copy()
    b20001 = b20001.loc[b20001["GEOID"].isin(tract_geoids)].copy()

    ref = (
        b01001.loc[:, ["GEOID", "B01001_001E"]]
        .rename(columns={"GEOID": "tract_geoid", "B01001_001E": "acs_total_pop"})
        .copy()
    )
    ref["acs_total_pop"] = pd.to_numeric(ref["acs_total_pop"], errors="coerce").fillna(0.0).astype(float)

    child_cols = [
        "B01001_003E",
        "B01001_004E",
        "B01001_005E",
        "B01001_006E",
        "B01001_027E",
        "B01001_028E",
        "B01001_029E",
        "B01001_030E",
    ]
    senior_cols = [
        "B01001_020E",
        "B01001_021E",
        "B01001_022E",
        "B01001_023E",
        "B01001_024E",
        "B01001_025E",
        "B01001_044E",
        "B01001_045E",
        "B01001_046E",
        "B01001_047E",
        "B01001_048E",
        "B01001_049E",
    ]
    ref["child_count_ref"] = _sum_cols(b01001, child_cols).to_numpy(dtype=float)
    ref["female_count_ref"] = pd.to_numeric(b01001["B01001_026E"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ref["senior_count_ref"] = _sum_cols(b01001, senior_cols).to_numpy(dtype=float)
    employed = pd.to_numeric(b23025["B23025_004E"], errors="coerce").fillna(0.0).astype(float)

    bachelor_plus = _sum_cols(b15003, [f"B15003_{idx:03d}E" for idx in range(22, 26)])
    high_income = _sum_cols(b20001, ["B20001_022E", "B20001_043E"])

    ref = ref.merge(
        b15003.loc[:, ["GEOID"]]
        .assign(bachelor_plus_count_ref=bachelor_plus.to_numpy(dtype=float))
        .rename(columns={"GEOID": "tract_geoid"}),
        on="tract_geoid",
        how="left",
    )
    ref = ref.merge(
        b23025.loc[:, ["GEOID"]]
        .assign(employed_count_ref=employed.to_numpy(dtype=float))
        .rename(columns={"GEOID": "tract_geoid"}),
        on="tract_geoid",
        how="left",
    )
    ref = ref.merge(
        b20001.loc[:, ["GEOID"]]
        .assign(high_income_count_ref=high_income.to_numpy(dtype=float))
        .rename(columns={"GEOID": "tract_geoid"}),
        on="tract_geoid",
        how="left",
    )

    for col in ["bachelor_plus_count_ref", "employed_count_ref", "high_income_count_ref"]:
        ref[col] = pd.to_numeric(ref[col], errors="coerce").fillna(0.0).astype(float)

    denom = ref["acs_total_pop"].replace(0.0, np.nan)
    ref["child_share_ref"] = ref["child_count_ref"] / denom
    ref["female_share_ref"] = ref["female_count_ref"] / denom
    ref["senior_share_ref"] = ref["senior_count_ref"] / denom
    ref["bachelor_plus_share_ref"] = ref["bachelor_plus_count_ref"] / denom
    ref["employed_share_ref"] = ref["employed_count_ref"] / denom
    ref["high_income_share_ref"] = ref["high_income_count_ref"] / denom
    return ref


def compute_metrics(synthetic_csv: Path, census_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    synthetic = pd.read_csv(synthetic_csv, dtype={"tract_geoid": str})
    synthetic["total_residents"] = pd.to_numeric(synthetic["total_residents"], errors="coerce").fillna(0.0)

    ref = _load_reference_tables(census_dir, set(synthetic["tract_geoid"]))
    merged = synthetic.merge(ref, on="tract_geoid", how="inner")

    specs = [
        ("children_0_17", "child_share", "child_share_ref", "child_count_ref"),
        ("female", "female_share", "female_share_ref", "female_count_ref"),
        ("bachelor_plus", "bachelor_plus_share", "bachelor_plus_share_ref", "bachelor_plus_count_ref"),
        ("employed", "employed_share", "employed_share_ref", "employed_count_ref"),
        ("income_100k_plus", "high_income_share", "high_income_share_ref", "high_income_count_ref"),
    ]

    rows: list[dict[str, float | int | str]] = []
    for subgroup, syn_share_col, ref_share_col, ref_count_col in specs:
        cols = [
            "tract_geoid",
            "total_residents",
            "acs_total_pop",
            syn_share_col,
            ref_share_col,
            ref_count_col,
        ]
        valid = merged.loc[:, cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
        valid["synthetic_count"] = (
            pd.to_numeric(valid[syn_share_col], errors="coerce").fillna(0.0).astype(float)
            * pd.to_numeric(valid["total_residents"], errors="coerce").fillna(0.0).astype(float)
        )

        syn_share = valid[syn_share_col].to_numpy(dtype=float)
        ref_share = valid[ref_share_col].to_numpy(dtype=float)
        weights = valid["acs_total_pop"].to_numpy(dtype=float)
        ref_count = valid[ref_count_col].to_numpy(dtype=float)
        syn_count = valid["synthetic_count"].to_numpy(dtype=float)

        share_spearman = float(valid[syn_share_col].corr(valid[ref_share_col], method="spearman"))
        share_cosine = _cosine(syn_share, ref_share)
        share_mae_weighted = float(np.average(np.abs(syn_share - ref_share), weights=weights))
        share_rmse_weighted = float(np.sqrt(np.average((syn_share - ref_share) ** 2, weights=weights)))
        distribution_tvd, distribution_cosine = _tvd_from_counts(syn_count, ref_count)

        rows.append(
            {
                "subgroup": subgroup,
                "n_tracts": int(len(valid)),
                "share_spearman": share_spearman,
                "share_cosine": share_cosine,
                "share_weighted_mae": share_mae_weighted,
                "share_weighted_rmse": share_rmse_weighted,
                "distribution_tvd": distribution_tvd,
                "distribution_cosine": distribution_cosine,
                "synthetic_total": float(syn_count.sum()),
                "reference_total": float(ref_count.sum()),
            }
        )

    summary = pd.DataFrame(rows)
    return summary, merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_csv", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--census_dir", type=Path, default=DEFAULT_CENSUS_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary, merged = compute_metrics(args.synthetic_csv, args.census_dir)

    summary_path = args.out_dir / "attribute_spatial_validation.csv"
    detailed_path = args.out_dir / "attribute_spatial_validation_detailed.csv"

    summary.to_csv(summary_path, index=False)
    merged.to_csv(detailed_path, index=False)

    pd.set_option("display.max_columns", None)
    print(summary.to_string(index=False))
    print(f"\n[wrote] {summary_path}")
    print(f"[wrote] {detailed_path}")


if __name__ == "__main__":
    main()
