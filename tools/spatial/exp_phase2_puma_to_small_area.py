#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Any

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from synthpop.data.poi_safegraph import aggregate_home_origin_profiles
from synthpop.paths import ensure_dir, project_root
from synthpop.spatial.puma_to_small_area import (
    allocate_joint_wide_to_small_areas,
    blend_prior_targets_long,
    compare_targets_long,
    low_rank_project_targets_long,
    low_rank_plus_sparse_project_targets_long,
    low_rank_plus_smooth_project_targets_long,
    predict_targets_from_group_features,
    summarize_type_allocation_against_targets,
)
from tools.data.build_external_condition_v1_michigan import _build_tract_puma_map, _canon_puma5, _read_tract_puma_csv


def _utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _target_variables(targets: pd.DataFrame) -> set[str]:
    return set(targets["variable"].astype(str).unique().tolist())


def _require_target_variables(
    *,
    targets: pd.DataFrame,
    variables: list[str],
    purpose: str,
    strict: bool,
) -> list[str]:
    requested = [str(v) for v in variables if str(v)]
    available = _target_variables(targets)
    missing = [v for v in requested if v not in available]
    if missing and bool(strict):
        hint = ""
        if "AGEP_SEX_cross" in missing:
            hint = " Rebuild tract targets with --include_age_sex_cross."
        raise SystemExit(
            f"targets_long missing requested {purpose} variable(s): {missing}. "
            f"Available variables: {sorted(available)}.{hint}"
        )
    if missing:
        print(
            f"[warn] targets_long missing requested {purpose} variable(s): {missing}; "
            f"available={sorted(available)}",
            file=sys.stderr,
        )
    return missing


def _canon_puma_uid(statefp: str, puma: Any) -> str:
    puma5 = _canon_puma5(puma)
    return f"{str(statefp).zfill(2)}{puma5}" if puma5 else ""


def _canon_region_series(series: pd.Series, *, region_col: str) -> pd.Series:
    out = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    if str(region_col) == "puma_uid":
        out = out.str.zfill(7)
    return out


def _load_targets(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    need = {"variable", "category", "target"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"targets_long missing columns: {miss}")
    for col in ["tract_geoid", "cbg_geoid", "county_geoid", "puma_uid", "puma", "statefp"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def _drop_empty_region_col(df: pd.DataFrame, *, region_col: str) -> pd.DataFrame:
    if str(region_col) not in df.columns:
        return df
    s = df[str(region_col)].astype(str).str.strip()
    s = s.where(~s.isin({"nan", "None", "null"}), "")
    if bool((s != "").any()):
        df[str(region_col)] = s
        return df
    return df.drop(columns=[str(region_col)]).copy()


def _build_group_to_region(
    *,
    group_col: str,
    region_col: str,
    statefp: str,
    group_to_region_csv: str,
    tract_puma_csv: str,
    tract_zip: str,
    puma_zip: str,
) -> pd.DataFrame:
    if group_to_region_csv:
        path = pathlib.Path(group_to_region_csv).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"group_to_region_csv not found: {path}")
        df = pd.read_csv(path, low_memory=False)
        if group_col not in df.columns or region_col not in df.columns:
            raise SystemExit(f"group_to_region_csv must contain {group_col} and {region_col}")
        out = df[[group_col, region_col]].drop_duplicates().copy()
        out[group_col] = out[group_col].astype(str)
        out[region_col] = out[region_col].astype(str)
        return out

    if group_col != "tract_geoid":
        raise SystemExit("automatic crosswalk construction currently supports group_col=tract_geoid only")

    tract_to_puma: dict[str, str]
    if tract_puma_csv:
        tract_to_puma = _read_tract_puma_csv(pathlib.Path(tract_puma_csv).expanduser().resolve())
    else:
        tract_zip_path = pathlib.Path(tract_zip).expanduser().resolve()
        puma_zip_path = pathlib.Path(puma_zip).expanduser().resolve()
        if not tract_zip_path.exists() or not puma_zip_path.exists():
            raise SystemExit("need either --group_to_region_csv or --tract_puma_csv or both --tract_zip and --puma_zip")
        tract_to_puma = _build_tract_puma_map(
            tract_zip=tract_zip_path,
            puma_zip=puma_zip_path,
            statefp=str(statefp).zfill(2),
        )

    rows = []
    for tract, puma in tract_to_puma.items():
        uid = _canon_puma_uid(str(statefp), puma)
        if not uid:
            continue
        rows.append({group_col: str(tract), region_col: uid})
    return pd.DataFrame(rows).drop_duplicates()


def _allocation_entropy(allocation: pd.DataFrame, *, type_idx_col: str, count_col: str, group_col: str) -> dict[str, float]:
    import numpy as np

    weighted_entropy = 0.0
    total_mass = 0.0
    max_share_vals: list[float] = []
    for _, g in allocation.groupby(str(type_idx_col), sort=False):
        w = g[str(count_col)].to_numpy(dtype=float)
        mass = float(w.sum())
        if mass <= 0.0:
            continue
        p = w / mass
        ent = -float((p * np.log(np.clip(p, 1e-12, None))).sum())
        weighted_entropy += mass * ent
        total_mass += mass
        max_share_vals.append(float(p.max()))
    return {
        "weighted_mean_entropy": (weighted_entropy / total_mass) if total_mass > 0 else 0.0,
        "mean_max_group_share": (float(np.mean(max_share_vals)) if max_share_vals else 0.0),
        "n_groups": int(allocation[str(group_col)].astype(str).nunique()),
    }


def _group_mass_summary(
    *,
    allocation: pd.DataFrame,
    group_col: str,
    count_col: str,
    group_weights: pd.DataFrame | None,
    weight_col: str,
) -> dict[str, Any]:
    import numpy as np

    pred = (
        allocation.groupby(str(group_col), as_index=False, sort=False)[str(count_col)]
        .sum()
        .rename(columns={str(count_col): "pred_count"})
    )
    total = float(pred["pred_count"].sum())
    pred["pred_share"] = pred["pred_count"] / max(total, 1e-12)
    out: dict[str, Any] = {
        "n_groups": int(pred.shape[0]),
        "pred_total": total,
    }
    if group_weights is None or group_weights.empty or str(weight_col) not in group_weights.columns:
        return out

    ref = group_weights[[str(group_col), str(weight_col)]].copy()
    ref[str(group_col)] = ref[str(group_col)].astype(str)
    ref[str(weight_col)] = pd.to_numeric(ref[str(weight_col)], errors="coerce").fillna(0.0).clip(lower=0.0)
    s = float(ref[str(weight_col)].sum())
    ref["ref_share"] = ref[str(weight_col)] / max(s, 1e-12)
    merged = pred.merge(ref[[str(group_col), "ref_share"]], on=str(group_col), how="left")
    merged["ref_share"] = pd.to_numeric(merged["ref_share"], errors="coerce").fillna(0.0)
    out["tvd_vs_group_weights"] = 0.5 * float((merged["pred_share"] - merged["ref_share"]).abs().sum())
    if merged.shape[0] >= 2:
        out["corr_vs_group_weights"] = float(np.corrcoef(merged["pred_share"], merged["ref_share"])[0, 1])
    else:
        out["corr_vs_group_weights"] = None
    return out


def _shuffle_feature_rows_within_region(
    df: pd.DataFrame,
    *,
    group_col: str,
    region_col: str,
    seed: int,
) -> pd.DataFrame:
    import numpy as np

    if df.empty:
        return df.copy()
    out = df.copy()
    feature_cols = [c for c in out.columns if c not in {str(group_col), str(region_col)}]
    if not feature_cols:
        return out
    rng = np.random.default_rng(int(seed))
    blocks: list[pd.DataFrame] = []
    for _, gg in out.groupby(str(region_col), sort=False):
        block = gg.copy()
        if block.shape[0] <= 1:
            blocks.append(block)
            continue
        perm = rng.permutation(block.shape[0])
        feat = block[feature_cols].iloc[perm].reset_index(drop=True)
        keep = block[[str(group_col), str(region_col)]].reset_index(drop=True)
        blocks.append(pd.concat([keep, feat], axis=1))
    return pd.concat(blocks, axis=0, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp_phase2_puma_to_small_area")
    ap.add_argument("--joint_wide_csv", required=True)
    ap.add_argument("--schema_json", required=True)
    ap.add_argument("--targets_long_csv", required=True)
    ap.add_argument("--prior_targets_csv", default="")
    ap.add_argument("--group_col", default="tract_geoid")
    ap.add_argument("--region_col", default="puma_uid")
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--group_to_region_csv", default="")
    ap.add_argument("--tract_puma_csv", default="")
    ap.add_argument("--tract_zip", default="")
    ap.add_argument("--puma_zip", default="")
    ap.add_argument("--hard_variables", default="AGEP_bin,SEX")
    ap.add_argument("--prior_variables", default="AGEP_bin,SEX,SCHL_allpop,ESR_allpop,EARN_16p_bin")
    ap.add_argument("--prior_variable_weights", default="")
    ap.add_argument("--strict_prior_variables", action="store_true")
    ap.add_argument("--mobility_poi_csv", default="")
    ap.add_argument("--mobility_region_filter", default="MI")
    ap.add_argument("--mobility_group_level", default="tract")
    ap.add_argument("--mobility_top_n_categories", type=int, default=24)
    ap.add_argument("--mobility_weight_col", default="home_origin_share")
    ap.add_argument("--mobility_use_group_weight", action="store_true")
    ap.add_argument("--mobility_use_type_prior", action="store_true")
    ap.add_argument("--mobility_type_prior_variables", default="SCHL_allpop,ESR_allpop,EARN_16p_bin")
    ap.add_argument("--mobility_use_residual_prior", action="store_true")
    ap.add_argument("--mobility_residual_variables", default="SCHL_allpop,ESR_allpop,EARN_16p_bin")
    ap.add_argument("--mobility_residual_variable_weights", default="")
    ap.add_argument("--mobility_residual_ratio_clip", type=float, default=3.0)
    ap.add_argument("--mobility_residual_use_low_rank", action="store_true")
    ap.add_argument("--mobility_residual_use_low_rank_sparse", action="store_true")
    ap.add_argument("--mobility_residual_use_low_rank_smooth", action="store_true")
    ap.add_argument("--mobility_residual_low_rank", type=int, default=2)
    ap.add_argument("--mobility_residual_sparse_weight", type=float, default=0.5)
    ap.add_argument("--mobility_residual_sparse_threshold", type=float, default=0.0)
    ap.add_argument("--mobility_residual_smooth_weight", type=float, default=0.5)
    ap.add_argument("--mobility_residual_smooth_knn", type=int, default=8)
    ap.add_argument("--mobility_residual_smooth_bandwidth", type=float, default=0.0)
    ap.add_argument("--mobility_feature_prefixes", default="cat__,home_origin_")
    ap.add_argument("--mobility_shuffle_within_region", action="store_true")
    ap.add_argument("--mobility_shuffle_seed", type=int, default=0)
    ap.add_argument("--mobility_prior_base_weight", type=float, default=1.0)
    ap.add_argument("--mobility_prior_extra_weight", type=float, default=1.0)
    ap.add_argument("--mobility_ridge_alpha", type=float, default=1.0)
    ap.add_argument("--mobility_min_train_groups", type=int, default=64)
    ap.add_argument("--puma_uids", default="")
    ap.add_argument("--max_regions", type=int, default=0)
    ap.add_argument("--integerize", action="store_true")
    ap.add_argument("--max_iters", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--label", default="phase2_puma_to_small_area")
    args = ap.parse_args()

    joint_wide_csv = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    schema_json = pathlib.Path(args.schema_json).expanduser().resolve()
    targets_long_csv = pathlib.Path(args.targets_long_csv).expanduser().resolve()
    prior_targets_csv = pathlib.Path(args.prior_targets_csv).expanduser().resolve() if args.prior_targets_csv else None
    for p in [joint_wide_csv, schema_json, targets_long_csv]:
        if not p.exists():
            raise SystemExit(f"required input not found: {p}")
    if prior_targets_csv is not None and not prior_targets_csv.exists():
        raise SystemExit(f"prior_targets_csv not found: {prior_targets_csv}")

    if args.run_dir:
        run_dir = pathlib.Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    metrics_dir = ensure_dir(run_dir / "metrics")
    synthetic_dir = ensure_dir(run_dir / "synthetic")

    joint = pd.read_csv(joint_wide_csv, low_memory=False)
    joint[str(args.region_col)] = _canon_region_series(joint[str(args.region_col)], region_col=str(args.region_col))
    targets = _load_targets(targets_long_csv)
    prior_targets = _load_targets(prior_targets_csv) if prior_targets_csv is not None else targets.copy()
    targets = _drop_empty_region_col(targets, region_col=str(args.region_col))
    prior_targets = _drop_empty_region_col(prior_targets, region_col=str(args.region_col))

    group_to_region = _build_group_to_region(
        group_col=str(args.group_col),
        region_col=str(args.region_col),
        statefp=str(args.statefp),
        group_to_region_csv=str(args.group_to_region_csv),
        tract_puma_csv=str(args.tract_puma_csv),
        tract_zip=str(args.tract_zip),
        puma_zip=str(args.puma_zip),
    )
    group_to_region[str(args.group_col)] = group_to_region[str(args.group_col)].astype(str)
    group_to_region[str(args.region_col)] = _canon_region_series(
        group_to_region[str(args.region_col)], region_col=str(args.region_col)
    )

    targets = targets.merge(group_to_region, on=str(args.group_col), how="inner")
    prior_targets = prior_targets.merge(group_to_region, on=str(args.group_col), how="inner")
    group_to_region_all = group_to_region.copy()
    targets_all = targets.copy()
    prior_targets_all = prior_targets.copy()

    requested_regions = _parse_csv_list(str(args.puma_uids))
    if requested_regions:
        selected = requested_regions
    else:
        selected = sorted(set(targets_all[str(args.region_col)].tolist()) & set(joint[str(args.region_col)].tolist()))
    if int(args.max_regions) > 0:
        selected = selected[: int(args.max_regions)]

    joint = joint[joint[str(args.region_col)].astype(str).isin(selected)].copy()
    targets = targets_all[targets_all[str(args.region_col)].astype(str).isin(selected)].copy()
    prior_targets = prior_targets_all[prior_targets_all[str(args.region_col)].astype(str).isin(selected)].copy()
    group_to_region = group_to_region_all[group_to_region_all[str(args.region_col)].astype(str).isin(selected)].copy()

    if joint.empty or targets.empty:
        raise SystemExit("empty experiment slice after region filtering")

    hard_variables_cfg = _parse_csv_list(str(args.hard_variables))
    prior_variables_cfg = _parse_csv_list(str(args.prior_variables))
    type_prior_variables_cfg = _parse_csv_list(str(args.mobility_type_prior_variables))
    residual_variables_cfg = _parse_csv_list(str(args.mobility_residual_variables))
    feature_prefixes_cfg = tuple(_parse_csv_list(str(args.mobility_feature_prefixes)))
    _require_target_variables(
        targets=targets,
        variables=hard_variables_cfg,
        purpose="hard",
        strict=True,
    )
    _require_target_variables(
        targets=prior_targets,
        variables=prior_variables_cfg,
        purpose="prior",
        strict=bool(args.strict_prior_variables),
    )
    residual_projection_modes = sum(
        [
            bool(args.mobility_residual_use_low_rank),
            bool(args.mobility_residual_use_low_rank_sparse),
            bool(args.mobility_residual_use_low_rank_smooth),
        ]
    )
    if residual_projection_modes > 1:
        raise SystemExit(
            "choose at most one residual projection mode among "
            "--mobility_residual_use_low_rank, "
            "--mobility_residual_use_low_rank_sparse, "
            "--mobility_residual_use_low_rank_smooth"
        )
    group_weights = None
    mobility_features = None
    mobility_profile_path = None
    mobility_pred_targets_path = None
    mobility_blended_prior_path = None
    mobility_residual_targets_path = None
    mobility_residual_lowrank_targets_path = None
    mobility_residual_lowrank_sparse_targets_path = None
    mobility_residual_lowrank_smooth_targets_path = None
    mobility_fit_path = None
    mobility_compare_path = None
    mobility_fit_meta: dict[str, Any] | None = None
    mobility_compare: dict[str, Any] | None = None
    mobility_lowrank_meta: dict[str, Any] | None = None
    mobility_lowrank_sparse_meta: dict[str, Any] | None = None
    mobility_lowrank_smooth_meta: dict[str, Any] | None = None
    mobility_features_all = None
    residual_targets_for_alloc = None
    if args.mobility_poi_csv:
        mobility_profile_path = synthetic_dir / f"mobility_profile_{args.mobility_group_level}.csv"
        mobility_features_all = aggregate_home_origin_profiles(
            merged_poi=pathlib.Path(args.mobility_poi_csv).expanduser().resolve(),
            group_level=str(args.mobility_group_level),
            region_filter=(str(args.mobility_region_filter) if args.mobility_region_filter else None),
            top_n_categories=int(args.mobility_top_n_categories),
        )
        if not mobility_features_all.empty:
            mobility_features_all.to_csv(mobility_profile_path, index=False)
            if str(args.group_col) != f"{str(args.mobility_group_level)}_geoid":
                raise SystemExit(
                    f"mobility_group_level={args.mobility_group_level} does not match group_col={args.group_col}"
                )
            mobility_features_all = mobility_features_all.merge(group_to_region_all, on=str(args.group_col), how="inner")
            if bool(args.mobility_shuffle_within_region):
                mobility_features_all = _shuffle_feature_rows_within_region(
                    mobility_features_all,
                    group_col=str(args.group_col),
                    region_col=str(args.region_col),
                    seed=int(args.mobility_shuffle_seed),
                )
            mobility_features = mobility_features_all[
                mobility_features_all[str(args.region_col)].astype(str).isin(selected)
            ].copy()
            if bool(args.mobility_use_group_weight):
                group_weights = mobility_features.copy()

    if bool(args.mobility_use_type_prior) or bool(args.mobility_use_residual_prior):
        if not args.mobility_poi_csv:
            raise SystemExit("mobility prior modes require --mobility_poi_csv")
        if mobility_features_all is None or mobility_features_all.empty:
            raise SystemExit("mobility features are empty; cannot build type-conditioned prior")
        type_prior_variables = list(type_prior_variables_cfg)
        if not type_prior_variables:
            type_prior_variables = [
                v
                for v in prior_variables_cfg
                if v not in set(hard_variables_cfg)
            ]
            type_prior_variables_cfg = list(type_prior_variables)
        residual_variables = list(residual_variables_cfg)
        if not residual_variables:
            residual_variables = list(type_prior_variables)
            residual_variables_cfg = list(residual_variables)
        pred_variables = sorted(
            set(type_prior_variables if bool(args.mobility_use_type_prior) else []).union(
                set(residual_variables if bool(args.mobility_use_residual_prior) else [])
            )
        )
        pred_targets_long_all, mobility_fit_meta = predict_targets_from_group_features(
            group_features=mobility_features_all,
            reference_targets_long=prior_targets_all,
            group_col=str(args.group_col),
            region_col=str(args.region_col),
            variable_col="variable",
            category_col="category",
            target_col="target",
            variables=pred_variables,
            feature_prefixes=feature_prefixes_cfg,
            ridge_alpha=float(args.mobility_ridge_alpha),
            min_train_groups=int(args.mobility_min_train_groups),
        )
        pred_targets_long = pred_targets_long_all.merge(
            group_to_region_all[[str(args.group_col), str(args.region_col)]].drop_duplicates(),
            on=str(args.group_col),
            how="inner",
        )
        pred_targets_long = pred_targets_long[
            pred_targets_long[str(args.region_col)].astype(str).isin(selected)
        ].drop(columns=[str(args.region_col)]).copy()
        if pred_targets_long.empty:
            raise SystemExit("mobility type-conditioned prior produced empty predictions")
        mobility_pred_targets_path = synthetic_dir / "mobility_pred_targets_long.csv"
        pred_targets_long.to_csv(mobility_pred_targets_path, index=False)
        base_prior_targets = prior_targets.copy()
        mobility_compare = {
            "pred_vs_base": compare_targets_long(
                predicted_targets_long=pred_targets_long,
                reference_targets_long=base_prior_targets,
                group_col=str(args.group_col),
                variable_col="variable",
                category_col="category",
                target_col="target",
            ),
        }
        if bool(args.mobility_use_type_prior):
            pred_targets_for_typeprior = pred_targets_long[
                pred_targets_long["variable"].astype(str).isin(type_prior_variables)
            ].copy()
            prior_targets = blend_prior_targets_long(
                base_targets_long=base_prior_targets,
                extra_targets_long=pred_targets_for_typeprior,
                group_col=str(args.group_col),
                variable_col="variable",
                category_col="category",
                target_col="target",
                variables=type_prior_variables,
                base_weight=float(args.mobility_prior_base_weight),
                extra_weight=float(args.mobility_prior_extra_weight),
            )
            mobility_blended_prior_path = synthetic_dir / "mobility_blended_prior_targets_long.csv"
            prior_targets.to_csv(mobility_blended_prior_path, index=False)
            mobility_compare["blend_vs_base"] = compare_targets_long(
                predicted_targets_long=prior_targets,
                reference_targets_long=base_prior_targets,
                group_col=str(args.group_col),
                variable_col="variable",
                category_col="category",
                target_col="target",
            )
        if bool(args.mobility_use_residual_prior):
            residual_targets_raw = pred_targets_long[
                pred_targets_long["variable"].astype(str).isin(residual_variables)
            ].copy()
            if bool(args.mobility_residual_use_low_rank_smooth):
                residual_targets_for_alloc, mobility_lowrank_smooth_meta = low_rank_plus_smooth_project_targets_long(
                    targets_long=residual_targets_raw,
                    group_features=mobility_features,
                    reference_targets_long=base_prior_targets,
                    group_col=str(args.group_col),
                    region_col=str(args.region_col),
                    variable_col="variable",
                    category_col="category",
                    target_col="target",
                    variables=residual_variables,
                    feature_prefixes=feature_prefixes_cfg,
                    rank=int(args.mobility_residual_low_rank),
                    smooth_weight=float(args.mobility_residual_smooth_weight),
                    smooth_knn=int(args.mobility_residual_smooth_knn),
                    smooth_bandwidth=float(args.mobility_residual_smooth_bandwidth),
                )
                mobility_residual_targets_path = synthetic_dir / "mobility_residual_targets_raw.csv"
                residual_targets_raw.to_csv(mobility_residual_targets_path, index=False)
                mobility_residual_lowrank_smooth_targets_path = synthetic_dir / "mobility_residual_targets_lowrank_smooth.csv"
                residual_targets_for_alloc.to_csv(mobility_residual_lowrank_smooth_targets_path, index=False)
                mobility_compare["lowrank_smooth_vs_raw"] = compare_targets_long(
                    predicted_targets_long=residual_targets_for_alloc,
                    reference_targets_long=residual_targets_raw,
                    group_col=str(args.group_col),
                    variable_col="variable",
                    category_col="category",
                    target_col="target",
                )
            elif bool(args.mobility_residual_use_low_rank_sparse):
                residual_targets_for_alloc, mobility_lowrank_sparse_meta = low_rank_plus_sparse_project_targets_long(
                    targets_long=residual_targets_raw,
                    reference_targets_long=base_prior_targets,
                    group_col=str(args.group_col),
                    variable_col="variable",
                    category_col="category",
                    target_col="target",
                    variables=residual_variables,
                    rank=int(args.mobility_residual_low_rank),
                    sparse_weight=float(args.mobility_residual_sparse_weight),
                    sparse_threshold=float(args.mobility_residual_sparse_threshold),
                )
                mobility_residual_targets_path = synthetic_dir / "mobility_residual_targets_raw.csv"
                residual_targets_raw.to_csv(mobility_residual_targets_path, index=False)
                mobility_residual_lowrank_sparse_targets_path = synthetic_dir / "mobility_residual_targets_lowrank_sparse.csv"
                residual_targets_for_alloc.to_csv(mobility_residual_lowrank_sparse_targets_path, index=False)
                mobility_compare["lowrank_sparse_vs_raw"] = compare_targets_long(
                    predicted_targets_long=residual_targets_for_alloc,
                    reference_targets_long=residual_targets_raw,
                    group_col=str(args.group_col),
                    variable_col="variable",
                    category_col="category",
                    target_col="target",
                )
            elif bool(args.mobility_residual_use_low_rank):
                residual_targets_for_alloc, mobility_lowrank_meta = low_rank_project_targets_long(
                    targets_long=residual_targets_raw,
                    reference_targets_long=base_prior_targets,
                    group_col=str(args.group_col),
                    variable_col="variable",
                    category_col="category",
                    target_col="target",
                    variables=residual_variables,
                    rank=int(args.mobility_residual_low_rank),
                )
                mobility_residual_targets_path = synthetic_dir / "mobility_residual_targets_raw.csv"
                residual_targets_raw.to_csv(mobility_residual_targets_path, index=False)
                mobility_residual_lowrank_targets_path = synthetic_dir / "mobility_residual_targets_lowrank.csv"
                residual_targets_for_alloc.to_csv(mobility_residual_lowrank_targets_path, index=False)
                mobility_compare["lowrank_vs_raw"] = compare_targets_long(
                    predicted_targets_long=residual_targets_for_alloc,
                    reference_targets_long=residual_targets_raw,
                    group_col=str(args.group_col),
                    variable_col="variable",
                    category_col="category",
                    target_col="target",
                )
            else:
                residual_targets_for_alloc = residual_targets_raw
                mobility_residual_targets_path = synthetic_dir / "mobility_residual_targets_long.csv"
                residual_targets_for_alloc.to_csv(mobility_residual_targets_path, index=False)
        mobility_fit_path = metrics_dir / "mobility_prior_fit.json"
        mobility_fit_path.write_text(json.dumps(mobility_fit_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        mobility_compare_path = metrics_dir / "mobility_prior_compare.json"
        mobility_compare_path.write_text(json.dumps(mobility_compare, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    prior_variable_weights: dict[str, float] = {}
    for item in _parse_csv_list(str(args.prior_variable_weights)):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            prior_variable_weights[str(key).strip()] = float(value)
        except Exception:
            continue
    residual_variable_weights: dict[str, float] = {}
    for item in _parse_csv_list(str(args.mobility_residual_variable_weights)):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            residual_variable_weights[str(key).strip()] = float(value)
        except Exception:
            continue

    alloc, meta = allocate_joint_wide_to_small_areas(
        joint_wide=joint,
        schema=schema_json,
        hard_targets_long=targets,
        prior_targets_long=prior_targets,
        group_to_region=group_to_region,
        group_col=str(args.group_col),
        region_col=str(args.region_col),
        hard_variables=hard_variables_cfg,
        prior_variables=prior_variables_cfg,
        prior_variable_weights=prior_variable_weights,
        residual_targets_long=residual_targets_for_alloc,
        residual_variables=residual_variables_cfg,
        residual_variable_weights=residual_variable_weights,
        residual_ratio_clip=float(args.mobility_residual_ratio_clip),
        group_weights=group_weights,
        group_weight_col=str(args.mobility_weight_col),
        integerize=bool(args.integerize),
        max_iters=int(args.max_iters),
        tol=float(args.tol),
    )

    summary = summarize_type_allocation_against_targets(
        allocation_long=alloc,
        targets_long=targets,
        group_col=str(args.group_col),
    )
    entropy = _allocation_entropy(
        allocation=alloc,
        type_idx_col="type_idx",
        count_col="count",
        group_col=str(args.group_col),
    )
    group_mass = _group_mass_summary(
        allocation=alloc,
        group_col=str(args.group_col),
        count_col="count",
        group_weights=group_weights,
        weight_col=str(args.mobility_weight_col),
    )

    alloc_path = synthetic_dir / "type_assignment_long.csv"
    group_totals_path = synthetic_dir / "group_total_counts.csv"
    alloc.to_csv(alloc_path, index=False)
    (
        alloc.groupby([str(args.region_col), str(args.group_col)], as_index=False, sort=False)["count"]
        .sum()
        .rename(columns={"count": "assigned_count"})
        .to_csv(group_totals_path, index=False)
    )

    run_summary = {
        "created_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "label": str(args.label),
        "joint_wide_csv": str(joint_wide_csv),
        "schema_json": str(schema_json),
        "targets_long_csv": str(targets_long_csv),
        "prior_targets_csv": (str(prior_targets_csv) if prior_targets_csv is not None else str(targets_long_csv)),
        "group_col": str(args.group_col),
        "region_col": str(args.region_col),
        "n_regions": int(len(selected)),
        "regions": selected,
        "hard_variables": hard_variables_cfg,
        "prior_variables": prior_variables_cfg,
        "strict_prior_variables": bool(args.strict_prior_variables),
        "prior_variable_weights": prior_variable_weights,
        "mobility_poi_csv": (str(args.mobility_poi_csv) if args.mobility_poi_csv else None),
        "mobility_profile_csv": (str(mobility_profile_path) if mobility_profile_path is not None else None),
        "mobility_use_group_weight": bool(args.mobility_use_group_weight),
        "mobility_use_type_prior": bool(args.mobility_use_type_prior),
        "mobility_use_residual_prior": bool(args.mobility_use_residual_prior),
        "mobility_type_prior_variables": type_prior_variables_cfg,
        "mobility_residual_variables": residual_variables_cfg,
        "mobility_residual_variable_weights": residual_variable_weights,
        "mobility_residual_ratio_clip": float(args.mobility_residual_ratio_clip),
        "mobility_residual_use_low_rank": bool(args.mobility_residual_use_low_rank),
        "mobility_residual_use_low_rank_sparse": bool(args.mobility_residual_use_low_rank_sparse),
        "mobility_residual_use_low_rank_smooth": bool(args.mobility_residual_use_low_rank_smooth),
        "mobility_residual_low_rank": int(args.mobility_residual_low_rank),
        "mobility_residual_sparse_weight": float(args.mobility_residual_sparse_weight),
        "mobility_residual_sparse_threshold": float(args.mobility_residual_sparse_threshold),
        "mobility_residual_smooth_weight": float(args.mobility_residual_smooth_weight),
        "mobility_residual_smooth_knn": int(args.mobility_residual_smooth_knn),
        "mobility_residual_smooth_bandwidth": float(args.mobility_residual_smooth_bandwidth),
        "mobility_feature_prefixes": list(feature_prefixes_cfg),
        "mobility_shuffle_within_region": bool(args.mobility_shuffle_within_region),
        "mobility_shuffle_seed": int(args.mobility_shuffle_seed),
        "mobility_prior_base_weight": float(args.mobility_prior_base_weight),
        "mobility_prior_extra_weight": float(args.mobility_prior_extra_weight),
        "mobility_ridge_alpha": float(args.mobility_ridge_alpha),
        "mobility_min_train_groups": int(args.mobility_min_train_groups),
        "integerize": bool(args.integerize),
        "max_iters": int(args.max_iters),
        "tol": float(args.tol),
        "artifacts": {
            "allocation_long_csv": str(alloc_path),
            "group_total_counts_csv": str(group_totals_path),
            "metrics_summary_json": str(metrics_dir / "summary.json"),
            "mobility_pred_targets_long_csv": (str(mobility_pred_targets_path) if mobility_pred_targets_path is not None else None),
            "mobility_blended_prior_targets_long_csv": (
                str(mobility_blended_prior_path) if mobility_blended_prior_path is not None else None
            ),
            "mobility_residual_targets_long_csv": (
                str(mobility_residual_targets_path) if mobility_residual_targets_path is not None else None
            ),
            "mobility_residual_targets_lowrank_csv": (
                str(mobility_residual_lowrank_targets_path) if mobility_residual_lowrank_targets_path is not None else None
            ),
            "mobility_residual_targets_lowrank_sparse_csv": (
                str(mobility_residual_lowrank_sparse_targets_path)
                if mobility_residual_lowrank_sparse_targets_path is not None
                else None
            ),
            "mobility_residual_targets_lowrank_smooth_csv": (
                str(mobility_residual_lowrank_smooth_targets_path)
                if mobility_residual_lowrank_smooth_targets_path is not None
                else None
            ),
            "mobility_prior_fit_json": (str(mobility_fit_path) if mobility_fit_path is not None else None),
            "mobility_prior_compare_json": (str(mobility_compare_path) if mobility_compare_path is not None else None),
        },
        "metrics": {
            "targets": summary,
            "allocation_entropy": entropy,
            "group_mass": group_mass,
            "mobility_prior_fit": mobility_fit_meta,
            "mobility_prior_compare": mobility_compare,
            "mobility_lowrank": mobility_lowrank_meta,
            "mobility_lowrank_sparse": mobility_lowrank_sparse_meta,
            "mobility_lowrank_smooth": mobility_lowrank_smooth_meta,
        },
        "solver_meta": meta,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(run_summary["metrics"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] run_dir={run_dir}")


if __name__ == "__main__":
    main()
