#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VARIABLES = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop", "EARN_16p_bin"]

TRACT_GROUPS = {
    "children_0_17": ("AGEP_bin", ["[0.0, 5.0)", "[5.0, 18.0)"]),
    "female": ("SEX", ["2"]),
    "employed": ("ESR_allpop", ["employed"]),
    "bachelor_plus": ("SCHL_allpop", ["bachelor_plus"]),
    "income_100k_plus": ("EARN_16p_bin", ["ge_100k"]),
}


def _canon_uid(value: object) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def export_puma_condition_consistency(
    *,
    persons_parquet: Path,
    condition_csv: Path,
    out_csv: Path,
    statefp: str,
) -> None:
    statefp = str(statefp).zfill(2)
    persons = pd.read_parquet(
        persons_parquet,
        columns=["puma_uid", *VARIABLES],
    )
    persons["puma_uid"] = persons["puma_uid"].map(_canon_uid)
    persons = persons[persons["puma_uid"].str.startswith(statefp)].copy()
    if persons.empty:
        raise SystemExit(f"No synthetic persons found for statefp={statefp}")

    condition = pd.read_csv(condition_csv, dtype={"puma_uid": str, "variable": str, "category": str})
    condition["puma_uid"] = condition["puma_uid"].map(_canon_uid)
    condition = condition[condition["puma_uid"].isin(set(persons["puma_uid"]))].copy()
    condition = condition[condition["variable"].isin(VARIABLES)].copy()
    if condition.empty:
        raise SystemExit("No matching condition rows found.")

    # Normalize category strings so numeric categories align between ACS rows and sampled persons.
    condition["category"] = condition["category"].astype(str)
    persons_long: list[pd.DataFrame] = []
    for variable in VARIABLES:
        tmp = (
            persons.groupby(["puma_uid", variable], observed=True)
            .size()
            .reset_index(name="synthetic_count")
            .rename(columns={variable: "category"})
        )
        tmp["variable"] = variable
        tmp["category"] = tmp["category"].astype(str)
        if variable == "SEX":
            tmp["category"] = tmp["category"].str.replace(r"\.0$", "", regex=True)
            condition.loc[condition["variable"] == variable, "category"] = (
                condition.loc[condition["variable"] == variable, "category"]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)
            )
        persons_long.append(tmp)

    synthetic = pd.concat(persons_long, ignore_index=True)
    synthetic["synthetic_total"] = synthetic.groupby(["puma_uid", "variable"])["synthetic_count"].transform("sum")
    synthetic["synthetic_share"] = synthetic["synthetic_count"] / synthetic["synthetic_total"]

    condition = condition.rename(columns={"target": "condition_count"})
    condition["condition_count"] = pd.to_numeric(condition["condition_count"], errors="coerce")
    condition["condition_total"] = condition.groupby(["puma_uid", "variable"])["condition_count"].transform("sum")
    condition["condition_share"] = condition["condition_count"] / condition["condition_total"]

    merged = condition[
        ["puma_uid", "variable", "category", "condition_count", "condition_total", "condition_share"]
    ].merge(
        synthetic[["puma_uid", "variable", "category", "synthetic_count", "synthetic_total", "synthetic_share"]],
        on=["puma_uid", "variable", "category"],
        how="left",
    )
    merged["synthetic_count"] = merged["synthetic_count"].fillna(0.0)
    merged["synthetic_total"] = merged["synthetic_total"].fillna(
        merged.groupby(["puma_uid", "variable"])["synthetic_count"].transform("sum")
    )
    merged["synthetic_share"] = merged["synthetic_share"].fillna(0.0)
    merged["share_gap"] = merged["synthetic_share"] - merged["condition_share"]
    merged["abs_share_gap"] = merged["share_gap"].abs()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values(["puma_uid", "variable", "category"]).to_csv(out_csv, index=False)


def export_tract_condition_consistency(
    *,
    persons_parquet: Path,
    tract_condition_csv: Path,
    out_detail_csv: Path,
    out_summary_csv: Path,
    statefp: str,
) -> None:
    statefp = str(statefp).zfill(2)
    persons = pd.read_parquet(
        persons_parquet,
        columns=["tract_geoid", "statefp", *VARIABLES],
    )
    persons["statefp"] = persons["statefp"].astype(str).str.zfill(2)
    persons = persons[persons["statefp"] == statefp].copy()
    persons["tract_geoid"] = persons["tract_geoid"].map(_canon_uid)
    if persons.empty:
        raise SystemExit(f"No synthetic persons found for statefp={statefp}")

    condition = pd.read_csv(tract_condition_csv, dtype={"tract_geoid": str, "variable": str, "category": str})
    condition["tract_geoid"] = condition["tract_geoid"].map(_canon_uid)
    condition["category"] = condition["category"].astype(str).str.replace(r"\.0$", "", regex=True)
    condition["target"] = pd.to_numeric(condition["target"], errors="coerce").fillna(0.0)
    condition = condition[condition["tract_geoid"].isin(set(persons["tract_geoid"]))].copy()

    rows: list[dict[str, object]] = []
    for name, (variable, categories) in TRACT_GROUPS.items():
        cats = [str(c).replace(".0", "") if variable == "SEX" else str(c) for c in categories]
        ref_var = condition[condition["variable"] == variable].copy()
        ref_total = ref_var.groupby("tract_geoid", observed=True)["target"].sum().rename("reference_total")
        ref_count = (
            ref_var[ref_var["category"].isin(cats)]
            .groupby("tract_geoid", observed=True)["target"]
            .sum()
            .rename("reference_count")
        )

        person_var = persons[["tract_geoid", variable]].copy()
        person_var[variable] = person_var[variable].astype(str).str.replace(r"\.0$", "", regex=True)
        syn_total = person_var.groupby("tract_geoid", observed=True).size().rename("synthetic_total")
        syn_count = (
            person_var[person_var[variable].isin(cats)]
            .groupby("tract_geoid", observed=True)
            .size()
            .rename("synthetic_count")
        )

        df = pd.concat([ref_total, ref_count, syn_total, syn_count], axis=1).fillna(0.0).reset_index()
        df["attribute_group"] = name
        df["reference_share"] = np.divide(
            df["reference_count"],
            df["reference_total"],
            out=np.zeros(len(df), dtype=float),
            where=df["reference_total"].to_numpy(dtype=float) > 0,
        )
        df["synthetic_share"] = np.divide(
            df["synthetic_count"],
            df["synthetic_total"],
            out=np.zeros(len(df), dtype=float),
            where=df["synthetic_total"].to_numpy(dtype=float) > 0,
        )
        df["share_gap"] = df["synthetic_share"] - df["reference_share"]
        df["abs_share_gap"] = df["share_gap"].abs()
        rows.extend(df.to_dict("records"))

    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for name, df in detail.groupby("attribute_group", observed=True):
        ref = df["reference_share"].to_numpy(dtype=float)
        syn = df["synthetic_share"].to_numpy(dtype=float)
        weights = df["reference_total"].to_numpy(dtype=float)
        valid = np.isfinite(ref) & np.isfinite(syn) & (weights > 0)
        ref = ref[valid]
        syn = syn[valid]
        weights = weights[valid]
        if len(ref) == 0:
            continue
        ref_dist = df.loc[valid, "reference_count"].to_numpy(dtype=float)
        syn_dist = df.loc[valid, "synthetic_count"].to_numpy(dtype=float)
        ref_dist = ref_dist / max(float(ref_dist.sum()), 1e-12)
        syn_dist = syn_dist / max(float(syn_dist.sum()), 1e-12)
        denom = max(float(np.linalg.norm(ref) * np.linalg.norm(syn)), 1e-12)
        summary_rows.append(
            {
                "attribute_group": name,
                "n_tracts": int(valid.sum()),
                "share_spearman": float(pd.Series(ref).corr(pd.Series(syn), method="spearman")),
                "share_cosine": float(np.dot(ref, syn) / denom),
                "share_weighted_mae": float(np.average(np.abs(syn - ref), weights=weights)),
                "share_weighted_rmse": float(np.sqrt(np.average((syn - ref) ** 2, weights=weights))),
                "distribution_tvd": float(0.5 * np.abs(syn_dist - ref_dist).sum()),
                "synthetic_total": float(df.loc[valid, "synthetic_count"].sum()),
                "reference_total": float(df.loc[valid, "reference_count"].sum()),
            }
        )

    out_detail_csv.parent.mkdir(parents=True, exist_ok=True)
    detail.sort_values(["attribute_group", "tract_geoid"]).to_csv(out_detail_csv, index=False)
    pd.DataFrame(summary_rows).sort_values("attribute_group").to_csv(out_summary_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persons-parquet", type=Path, required=True)
    parser.add_argument("--condition-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--tract-condition-csv", type=Path)
    parser.add_argument("--tract-out-detail-csv", type=Path)
    parser.add_argument("--tract-out-summary-csv", type=Path)
    parser.add_argument("--statefp", default="26")
    args = parser.parse_args()
    export_puma_condition_consistency(
        persons_parquet=args.persons_parquet,
        condition_csv=args.condition_csv,
        out_csv=args.out_csv,
        statefp=args.statefp,
    )
    if args.tract_condition_csv and args.tract_out_detail_csv and args.tract_out_summary_csv:
        export_tract_condition_consistency(
            persons_parquet=args.persons_parquet,
            tract_condition_csv=args.tract_condition_csv,
            out_detail_csv=args.tract_out_detail_csv,
            out_summary_csv=args.tract_out_summary_csv,
            statefp=args.statefp,
        )


if __name__ == "__main__":
    main()
