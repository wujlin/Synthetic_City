#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.paths import ensure_dir, project_root
from src.synthpop.validation.mobility_anchor import compare_share_frames


def _utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _first_existing(paths: list[pathlib.Path]) -> pathlib.Path:
    for path in paths:
        if path.exists():
            return path
    raise SystemExit(f"none of these paths exists: {[str(p) for p in paths]}")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = v.notna() & w.gt(0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(v.loc[mask].to_numpy(dtype=float), weights=w.loc[mask].to_numpy(dtype=float)))


def _resolve_work_center_map(tract_od: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    use = tract_od.loc[:, ["work_tract_geoid", "work_center_geoid"]].dropna().copy()
    use["work_tract_geoid"] = use["work_tract_geoid"].astype(str)
    use["work_center_geoid"] = use["work_center_geoid"].astype(str).str.replace(r"\.0$", "", regex=True)
    counts = (
        use.groupby(["work_tract_geoid", "work_center_geoid"], as_index=False, sort=False)
        .size()
        .rename(columns={"size": "n"})
    )
    counts = counts.sort_values(
        by=["work_tract_geoid", "n", "work_center_geoid"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    mapping = counts.drop_duplicates("work_tract_geoid", keep="first").loc[:, ["work_tract_geoid", "work_center_geoid"]].copy()
    ambiguity = counts.groupby("work_tract_geoid", sort=False)["work_center_geoid"].nunique()
    meta = {
        "n_work_tracts": int(mapping["work_tract_geoid"].nunique()),
        "n_ambiguous_work_tracts": int((ambiguity > 1).sum()),
    }
    return mapping, meta


def _conditional_group_metrics(
    *,
    od: pd.DataFrame,
    group_cols: list[str],
    item_col: str,
    syn_col: str,
    mob_col: str,
    min_mobility_total: int,
    min_item_units: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    syn_arr = pd.to_numeric(od[syn_col], errors="coerce").fillna(0.0)
    mob_arr = pd.to_numeric(od[mob_col], errors="coerce").fillna(0.0)
    work = od.copy()
    work[syn_col] = syn_arr
    work[mob_col] = mob_arr

    for keys, grp in work.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: str(val) for col, val in zip(group_cols, keys)}
        syn_total = float(grp[syn_col].sum())
        mob_total = float(grp[mob_col].sum())
        n_items = int(grp[item_col].nunique())
        row["synthetic_total"] = syn_total
        row["mobility_total"] = mob_total
        row["n_item_units"] = n_items
        eligible = (mob_total >= float(min_mobility_total)) and (n_items >= int(min_item_units))
        row["eligible"] = bool(eligible)
        if not eligible:
            row["spearman_conditional"] = float("nan")
            row["cosine_conditional"] = float("nan")
            row["tvd_conditional"] = float("nan")
            rows.append(row)
            continue

        syn_share = grp[syn_col].to_numpy(dtype=float) / max(syn_total, 1e-12)
        mob_share = grp[mob_col].to_numpy(dtype=float) / max(mob_total, 1e-12)
        denom = float(np.linalg.norm(syn_share) * np.linalg.norm(mob_share))
        cosine = float(np.dot(syn_share, mob_share) / denom) if denom > 0.0 else float("nan")
        tvd = float(0.5 * np.abs(syn_share - mob_share).sum())
        syn_series = pd.Series(syn_share)
        mob_series = pd.Series(mob_share)
        if syn_series.nunique() > 1 and mob_series.nunique() > 1:
            spearman = float(syn_series.corr(mob_series, method="spearman"))
        else:
            spearman = float("nan")
        row["spearman_conditional"] = spearman
        row["cosine_conditional"] = cosine
        row["tvd_conditional"] = tvd
        rows.append(row)

    out = pd.DataFrame(rows)
    eligible = out[out["eligible"]].copy()
    summary = {
        "n_groups_total": int(len(out)),
        "n_groups_eligible": int(len(eligible)),
        "mobility_mass_eligible_share": float(eligible["mobility_total"].sum() / max(float(out["mobility_total"].sum()), 1e-12)),
        "weighted_mean_spearman": _weighted_mean(eligible["spearman_conditional"], eligible["mobility_total"]),
        "weighted_mean_cosine": _weighted_mean(eligible["cosine_conditional"], eligible["mobility_total"]),
        "weighted_mean_tvd": _weighted_mean(eligible["tvd_conditional"], eligible["mobility_total"]),
        "median_spearman": float(pd.to_numeric(eligible["spearman_conditional"], errors="coerce").median()) if not eligible.empty else float("nan"),
        "share_spearman_ge_0_3": float((pd.to_numeric(eligible["spearman_conditional"], errors="coerce") >= 0.3).mean()) if not eligible.empty else float("nan"),
        "share_spearman_ge_0_5": float((pd.to_numeric(eligible["spearman_conditional"], errors="coerce") >= 0.5).mean()) if not eligible.empty else float("nan"),
    }
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp_validate_destination_layers")
    ap.add_argument("--validate_run_dir", required=True)
    ap.add_argument("--tract_od_path", required=True)
    ap.add_argument("--label", default="validate_destination_layers")
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--min_mobility_total", type=int, default=20)
    ap.add_argument("--min_item_units", type=int, default=2)
    args = ap.parse_args()

    validate_run_dir = pathlib.Path(args.validate_run_dir).expanduser().resolve()
    tract_od_path = pathlib.Path(args.tract_od_path).expanduser().resolve()
    if not validate_run_dir.exists():
        raise SystemExit(f"validate_run_dir not found: {validate_run_dir}")
    if not tract_od_path.exists():
        raise SystemExit(f"tract_od_path not found: {tract_od_path}")

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    metrics_dir = ensure_dir(run_dir / "metrics")

    summary_path = _first_existing([validate_run_dir / "run_summary.json", validate_run_dir / "metrics" / "summary.json"])
    validate_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    od_path = _first_existing([validate_run_dir / "metrics" / "work_od_comparison.csv"])
    od = pd.read_csv(od_path, low_memory=False)
    required = ["home_tract_geoid", "work_tract_geoid", "synthetic_count", "mobility_count"]
    missing = [c for c in required if c not in od.columns]
    if missing:
        raise SystemExit(f"work_od_comparison missing columns: {missing}")
    od["home_tract_geoid"] = od["home_tract_geoid"].astype(str)
    od["work_tract_geoid"] = od["work_tract_geoid"].astype(str)
    od["synthetic_count"] = pd.to_numeric(od["synthetic_count"], errors="coerce").fillna(0.0)
    od["mobility_count"] = pd.to_numeric(od["mobility_count"], errors="coerce").fillna(0.0)
    od["work_county_geoid"] = od["work_tract_geoid"].str.slice(0, 5)

    tract_od = pd.read_csv(tract_od_path, usecols=["work_tract_geoid", "work_center_geoid"], low_memory=False)
    center_map, center_meta = _resolve_work_center_map(tract_od)
    od = od.merge(center_map, on="work_tract_geoid", how="left")
    od["work_center_geoid"] = od["work_center_geoid"].fillna("missing_center").astype(str)

    county = (
        od.groupby(["home_tract_geoid", "work_county_geoid"], as_index=False, sort=False)[["synthetic_count", "mobility_count"]]
        .sum()
    )
    county_comp, county_summary = compare_share_frames(
        left=county.loc[:, ["home_tract_geoid", "work_county_geoid", "synthetic_count"]],
        right=county.loc[:, ["home_tract_geoid", "work_county_geoid", "mobility_count"]],
        key_cols=["home_tract_geoid", "work_county_geoid"],
        left_value_col="synthetic_count",
        right_value_col="mobility_count",
    )

    center = (
        od.groupby(["home_tract_geoid", "work_center_geoid"], as_index=False, sort=False)[["synthetic_count", "mobility_count"]]
        .sum()
    )
    center_comp, center_summary = compare_share_frames(
        left=center.loc[:, ["home_tract_geoid", "work_center_geoid", "synthetic_count"]],
        right=center.loc[:, ["home_tract_geoid", "work_center_geoid", "mobility_count"]],
        key_cols=["home_tract_geoid", "work_center_geoid"],
        left_value_col="synthetic_count",
        right_value_col="mobility_count",
    )

    within_county, within_county_summary = _conditional_group_metrics(
        od=od,
        group_cols=["home_tract_geoid", "work_county_geoid"],
        item_col="work_tract_geoid",
        syn_col="synthetic_count",
        mob_col="mobility_count",
        min_mobility_total=int(args.min_mobility_total),
        min_item_units=int(args.min_item_units),
    )
    within_center, within_center_summary = _conditional_group_metrics(
        od=od,
        group_cols=["home_tract_geoid", "work_center_geoid"],
        item_col="work_tract_geoid",
        syn_col="synthetic_count",
        mob_col="mobility_count",
        min_mobility_total=int(args.min_mobility_total),
        min_item_units=int(args.min_item_units),
    )

    total_od_tvd = float(validate_summary["work_od_validation"]["tvd_share"])
    county_between_tvd = float(county_summary["tvd_share"])
    center_between_tvd = float(center_summary["tvd_share"])
    county_within_residual_tvd = max(total_od_tvd - county_between_tvd, 0.0)
    center_within_residual_tvd = max(total_od_tvd - center_between_tvd, 0.0)

    summary = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "timestamp_utc": _utc_now_compact(),
        "validate_run_dir": str(validate_run_dir),
        "tract_od_path": str(tract_od_path),
        "validate_summary_path": str(summary_path),
        "work_center_mapping_meta": center_meta,
        "base_work_od_validation": validate_summary["work_od_validation"],
        "county_od_validation": county_summary,
        "center_od_validation": center_summary,
        "within_county_tract_validation": within_county_summary,
        "within_center_tract_validation": within_center_summary,
        "error_decomposition": {
            "total_work_od_tvd": total_od_tvd,
            "between_county_tvd": county_between_tvd,
            "within_county_residual_tvd": county_within_residual_tvd,
            "between_center_tvd": center_between_tvd,
            "within_center_residual_tvd": center_within_residual_tvd,
        },
    }

    _write_json(run_dir / "run_summary.json", summary)
    _write_json(metrics_dir / "summary.json", summary)
    county_comp.to_csv(metrics_dir / "county_od_comparison.csv", index=False)
    center_comp.to_csv(metrics_dir / "center_od_comparison.csv", index=False)
    within_county.to_csv(metrics_dir / "within_county_group_metrics.csv", index=False)
    within_center.to_csv(metrics_dir / "within_center_group_metrics.csv", index=False)


if __name__ == "__main__":
    main()
