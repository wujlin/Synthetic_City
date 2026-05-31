#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.experimental.representation.ssl_copula_residual_probe import (  # noqa: E402
    FULL_VARIABLE_ORDER,
    _aggregate_full_to_coarse,
    _fit_pc_scores,
    _load_acs_conditions,
    _load_spatial,
    _load_target,
    _outer_joint,
    _parse_csv_floats,
    _parse_csv_ints,
    _residual_arrays,
    _write_json,
)


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _score_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    explained_ratio: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    weighted_nonneg = 0.0
    weighted_raw = 0.0
    positive_share = 0.0
    top5: list[float] = []
    for pc_idx in range(y_true.shape[1]):
        r2 = float(r2_score(y_true[:, pc_idx], y_pred[:, pc_idx]))
        evr = float(explained_ratio[pc_idx])
        weighted_raw += evr * r2
        if r2 > 0.0:
            weighted_nonneg += evr * r2
            positive_share += evr
        if pc_idx < 5:
            top5.append(r2)
        rows.append(
            {
                "pc_index": int(pc_idx + 1),
                "explained_ratio": evr,
                "r2": r2,
            }
        )
    return (
        {
            "weighted_nonnegative_r2": float(weighted_nonneg),
            "weighted_raw_r2": float(weighted_raw),
            "positive_pc_explained_share": float(positive_share),
            "mean_r2_top5": float(np.mean(top5)) if top5 else float("nan"),
        },
        rows,
    )


def _fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alphas: list[float],
) -> np.ndarray:
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.asarray(alphas, dtype=float)))
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=np.float64)


def _cluster_labels(
    x: np.ndarray,
    train_mask: np.ndarray,
    n_types: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_mask])
    x_all = scaler.transform(x)
    k_eff = max(2, min(int(n_types), int(train_mask.sum()) - 1))
    km = KMeans(n_clusters=k_eff, random_state=int(seed), n_init=20)
    labels_train = km.fit_predict(x_train)
    labels_all = km.predict(x_all)
    sizes = np.bincount(labels_train, minlength=k_eff)
    meta = {
        "n_types": int(k_eff),
        "min_train_type_size": int(sizes.min()),
        "max_train_type_size": int(sizes.max()),
        "median_train_type_size": float(np.median(sizes)),
    }
    return labels_all, labels_train, meta


def _type_mean_predict(
    scores: np.ndarray,
    train_mask: np.ndarray,
    labels_all: np.ndarray,
) -> np.ndarray:
    y_train = scores[train_mask]
    labels_train = labels_all[train_mask]
    labels_test = labels_all[~train_mask]
    global_mean = y_train.mean(axis=0)
    means: dict[int, np.ndarray] = {}
    for label in np.unique(labels_train):
        means[int(label)] = y_train[labels_train == label].mean(axis=0)
    pred = np.vstack([means.get(int(label), global_mean) for label in labels_test])
    return pred.astype(np.float64)


def _type_specific_ridge_predict(
    x: np.ndarray,
    scores: np.ndarray,
    train_mask: np.ndarray,
    labels_all: np.ndarray,
    pooled_pred: np.ndarray,
    alphas: list[float],
    min_type_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    pred = pooled_pred.copy()
    labels_train = labels_all[train_mask]
    labels_test = labels_all[~train_mask]
    x_train_all = x[train_mask]
    y_train_all = scores[train_mask]
    x_test_all = x[~train_mask]
    fitted = 0
    fallback = 0
    for label in np.unique(labels_test):
        tr = labels_train == label
        te = labels_test == label
        if int(tr.sum()) < int(min_type_size):
            fallback += int(te.sum())
            continue
        pred[te] = _fit_predict_ridge(
            x_train=x_train_all[tr],
            y_train=y_train_all[tr],
            x_test=x_test_all[te],
            alphas=alphas,
        )
        fitted += int(te.sum())
    return pred, {"type_specific_test_rows": int(fitted), "pooled_fallback_test_rows": int(fallback)}


def _load_optional_view(path: pathlib.Path | None, target_keys: pd.DataFrame) -> np.ndarray | None:
    if path is None:
        return None
    return _load_spatial(path, target_keys)[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Test whether region-type regimes improve copula-residual prediction beyond a pooled predictor."
        )
    )
    data_root = pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data")
    ap.add_argument(
        "--target_wide_csv",
        type=pathlib.Path,
        default=data_root / "us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv",
    )
    ap.add_argument(
        "--condition_csv",
        type=pathlib.Path,
        default=data_root / "us/processed/external_conditions/extcond_v1_agesex_earn_v1_acs5_2022_puma_us.csv",
    )
    ap.add_argument(
        "--spatial_csv",
        type=pathlib.Path,
        default=data_root / "us/processed/features/puma_spatial_features_5var_knn6.csv",
    )
    ap.add_argument("--uncertainty_csv", type=pathlib.Path, default=None)
    ap.add_argument("--heldout_statefps", default="26,06,12,48,55")
    ap.add_argument("--reference_mode", choices=["target_marginals", "acs_marginals"], default="acs_marginals")
    ap.add_argument("--joint_space", choices=["full", "coarse"], default="full")
    ap.add_argument("--feature_sets", default="acs_all_scale,acs_all_scale_uncertainty,acs_all_spatial_scale,acs_all_spatial_scale_uncertainty")
    ap.add_argument("--n_types", default="4,8,12,16")
    ap.add_argument("--min_type_size", type=int, default=40)
    ap.add_argument("--max_pcs", type=int, default=40)
    ap.add_argument("--ridge_alphas", default="0.01,0.1,1,10,100,1000")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    out_dir = args.output_dir or pathlib.Path(f"outputs/_ssl_region_type_probe_{_utc_ts()}")
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    target_keys, p_true, p_eq_target, target_marginals = _load_target(args.target_wide_csv)
    _, acs_marginals, x_acs_1d, x_acs_all, x_scale, _, _, _ = _load_acs_conditions(args.condition_csv, target_keys)
    x_spatial = _load_spatial(args.spatial_csv, target_keys)[0]
    x_uncertainty = _load_optional_view(args.uncertainty_csv, target_keys)

    p_eq_acs = _outer_joint([acs_marginals[v] for v in FULL_VARIABLE_ORDER])
    if args.joint_space == "coarse":
        p_true = _aggregate_full_to_coarse(p_true)
        p_eq_target = _aggregate_full_to_coarse(p_eq_target)
        p_eq_acs = _aggregate_full_to_coarse(p_eq_acs)
    reference = p_eq_target if args.reference_mode == "target_marginals" else p_eq_acs
    _, _, log_ratio = _residual_arrays(p_true, reference, eps=float(args.eps))

    feature_bank: dict[str, np.ndarray] = {
        "acs_1d_scale": np.concatenate([x_acs_1d, x_scale], axis=1),
        "acs_all_scale": np.concatenate([x_acs_all, x_scale], axis=1),
        "acs_all_spatial_scale": np.concatenate([x_acs_all, x_spatial, x_scale], axis=1),
    }
    if x_uncertainty is not None:
        feature_bank["acs_all_scale_uncertainty"] = np.concatenate([x_acs_all, x_scale, x_uncertainty], axis=1)
        feature_bank["acs_all_spatial_scale_uncertainty"] = np.concatenate(
            [x_acs_all, x_spatial, x_scale, x_uncertainty],
            axis=1,
        )

    requested = [x.strip() for x in str(args.feature_sets).split(",") if x.strip()]
    feature_sets = {name: feature_bank[name] for name in requested if name in feature_bank}
    if not feature_sets:
        raise SystemExit("no requested feature sets are available")

    heldout_statefps = [str(x).zfill(2) for x in _parse_csv_ints(args.heldout_statefps)]
    n_types_list = _parse_csv_ints(args.n_types)
    ridge_alphas = _parse_csv_floats(args.ridge_alphas)
    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()

    summary_rows: list[dict[str, Any]] = []
    pc_rows: list[dict[str, Any]] = []
    type_size_rows: list[dict[str, Any]] = []

    for heldout in heldout_statefps:
        train_mask = statefp != heldout
        if int((~train_mask).sum()) == 0:
            print(f"[warn] heldout state {heldout} has no rows; skipped", file=sys.stderr)
            continue
        scores, explained_ratio, cumulative = _fit_pc_scores(
            log_ratio,
            train_mask=train_mask,
            n_components=int(args.max_pcs),
            seed=int(args.seed),
        )
        y_test = scores[~train_mask]
        for feature_name, x in feature_sets.items():
            pooled_pred = _fit_predict_ridge(
                x_train=x[train_mask],
                y_train=scores[train_mask],
                x_test=x[~train_mask],
                alphas=ridge_alphas,
            )
            pooled_summary, pooled_pc_rows = _score_prediction(y_test, pooled_pred, explained_ratio)
            summary_rows.append(
                {
                    "heldout_statefp": heldout,
                    "feature_set": feature_name,
                    "method": "pooled_ridge",
                    "n_types": 0,
                    "n_train": int(train_mask.sum()),
                    "n_test": int((~train_mask).sum()),
                    "n_features": int(x.shape[1]),
                    "n_pcs": int(scores.shape[1]),
                    "pc_cumulative_explained": float(cumulative),
                    "type_specific_test_rows": 0,
                    "pooled_fallback_test_rows": 0,
                    **pooled_summary,
                }
            )
            for row in pooled_pc_rows:
                row.update({"heldout_statefp": heldout, "feature_set": feature_name, "method": "pooled_ridge", "n_types": 0})
                pc_rows.append(row)

            for n_types in n_types_list:
                labels_all, labels_train, cluster_meta = _cluster_labels(
                    x=x,
                    train_mask=train_mask,
                    n_types=int(n_types),
                    seed=int(args.seed),
                )
                sizes = np.bincount(labels_train, minlength=int(cluster_meta["n_types"]))
                for label, size in enumerate(sizes.tolist()):
                    type_size_rows.append(
                        {
                            "heldout_statefp": heldout,
                            "feature_set": feature_name,
                            "n_types": int(cluster_meta["n_types"]),
                            "type_id": int(label),
                            "train_size": int(size),
                            "test_size": int(np.sum(labels_all[~train_mask] == label)),
                        }
                    )

                mean_pred = _type_mean_predict(scores=scores, train_mask=train_mask, labels_all=labels_all)
                mean_summary, mean_pc_rows = _score_prediction(y_test, mean_pred, explained_ratio)
                summary_rows.append(
                    {
                        "heldout_statefp": heldout,
                        "feature_set": feature_name,
                        "method": "type_mean",
                        "n_types": int(cluster_meta["n_types"]),
                        "n_train": int(train_mask.sum()),
                        "n_test": int((~train_mask).sum()),
                        "n_features": int(x.shape[1]),
                        "n_pcs": int(scores.shape[1]),
                        "pc_cumulative_explained": float(cumulative),
                        "type_specific_test_rows": 0,
                        "pooled_fallback_test_rows": 0,
                        **cluster_meta,
                        **mean_summary,
                    }
                )
                for row in mean_pc_rows:
                    row.update({"heldout_statefp": heldout, "feature_set": feature_name, "method": "type_mean", "n_types": int(cluster_meta["n_types"])})
                    pc_rows.append(row)

                type_pred, type_meta = _type_specific_ridge_predict(
                    x=x,
                    scores=scores,
                    train_mask=train_mask,
                    labels_all=labels_all,
                    pooled_pred=pooled_pred,
                    alphas=ridge_alphas,
                    min_type_size=int(args.min_type_size),
                )
                type_summary, type_pc_rows = _score_prediction(y_test, type_pred, explained_ratio)
                summary_rows.append(
                    {
                        "heldout_statefp": heldout,
                        "feature_set": feature_name,
                        "method": "type_specific_ridge",
                        "n_types": int(cluster_meta["n_types"]),
                        "n_train": int(train_mask.sum()),
                        "n_test": int((~train_mask).sum()),
                        "n_features": int(x.shape[1]),
                        "n_pcs": int(scores.shape[1]),
                        "pc_cumulative_explained": float(cumulative),
                        **cluster_meta,
                        **type_meta,
                        **type_summary,
                    }
                )
                for row in type_pc_rows:
                    row.update({"heldout_statefp": heldout, "feature_set": feature_name, "method": "type_specific_ridge", "n_types": int(cluster_meta["n_types"])})
                    pc_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(metrics_dir / "region_type_probe_summary.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(metrics_dir / "region_type_probe_pc_long.csv", index=False)
    pd.DataFrame(type_size_rows).to_csv(metrics_dir / "region_type_sizes.csv", index=False)

    if not summary.empty:
        mean_summary = (
            summary.groupby(["feature_set", "method", "n_types"], as_index=False)
            .agg(
                mean_weighted_nonnegative_r2=("weighted_nonnegative_r2", "mean"),
                mean_weighted_raw_r2=("weighted_raw_r2", "mean"),
                mean_r2_top5=("mean_r2_top5", "mean"),
                min_train_type_size=("min_train_type_size", "min"),
            )
            .sort_values("mean_weighted_nonnegative_r2", ascending=False)
        )
        mean_summary.to_csv(metrics_dir / "region_type_probe_mean_summary.csv", index=False)
        print(mean_summary.head(30).to_string(index=False))

    _write_json(
        out_dir / "run_summary.json",
        {
            "target_wide_csv": str(args.target_wide_csv),
            "condition_csv": str(args.condition_csv),
            "spatial_csv": str(args.spatial_csv),
            "uncertainty_csv": str(args.uncertainty_csv) if args.uncertainty_csv is not None else "",
            "heldout_statefps": heldout_statefps,
            "reference_mode": args.reference_mode,
            "joint_space": args.joint_space,
            "feature_sets": list(feature_sets.keys()),
            "n_types": n_types_list,
            "min_type_size": int(args.min_type_size),
            "max_pcs": int(args.max_pcs),
            "metrics": {
                "summary": str(metrics_dir / "region_type_probe_summary.csv"),
                "mean_summary": str(metrics_dir / "region_type_probe_mean_summary.csv"),
                "pc_long": str(metrics_dir / "region_type_probe_pc_long.csv"),
                "type_sizes": str(metrics_dir / "region_type_sizes.csv"),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
