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
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeCV
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
    _parse_csv_ints,
    _residual_arrays,
    _write_json,
)
from tools.experimental.representation.ssl_residual_retrieval_probe import _retrieval_metrics  # noqa: E402


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_label(text: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "external"


def _make_eval_splits(
    statefp: np.ndarray,
    heldouts: list[str],
    *,
    split_mode: str,
    test_fraction: float,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    splits: list[tuple[str, np.ndarray]] = []
    rng = np.random.default_rng(int(seed))
    for heldout in heldouts:
        state_mask = statefp == heldout
        if not np.any(state_mask):
            continue
        if split_mode == "state_holdout":
            splits.append((heldout, ~state_mask))
            continue
        idx = np.where(state_mask)[0]
        if idx.size < 2:
            continue
        n_test = int(np.ceil(float(test_fraction) * idx.size))
        n_test = min(max(n_test, 1), idx.size - 1)
        test_idx = rng.choice(idx, size=n_test, replace=False)
        train_mask = np.ones(statefp.shape[0], dtype=bool)
        train_mask[test_idx] = False
        splits.append((heldout, train_mask))
    return splits


def _weighted_r2_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_mask: np.ndarray,
    explained_ratio: np.ndarray,
) -> dict[str, float]:
    r2s = []
    top5 = []
    weighted_nonneg = 0.0
    weighted_raw = 0.0
    positive_share = 0.0
    for j in range(y_true.shape[1]):
        r2 = float(r2_score(y_true[~train_mask, j], y_pred[~train_mask, j]))
        evr = float(explained_ratio[j])
        weighted_nonneg += max(0.0, r2) * evr
        weighted_raw += r2 * evr
        if r2 > 0.0:
            positive_share += evr
        if j < 5:
            top5.append(r2)
        r2s.append(r2)
    return {
        "weighted_nonnegative_r2": float(weighted_nonneg),
        "weighted_raw_r2": float(weighted_raw),
        "positive_pc_explained_share": float(positive_share),
        "mean_r2_top5": float(np.mean(top5)) if top5 else float("nan"),
        "median_r2_all": float(np.median(r2s)) if r2s else float("nan"),
    }


def _standardize_train(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    out = np.empty_like(np.asarray(x, dtype=np.float64), dtype=np.float64)
    out[train_mask] = scaler.fit_transform(x[train_mask])
    out[~train_mask] = scaler.transform(x[~train_mask])
    return out, scaler


def _residualize_external(
    x_context_z: np.ndarray,
    x_external_z: np.ndarray,
    train_mask: np.ndarray,
    alphas: list[float],
    fixed_alpha: float | None,
) -> tuple[np.ndarray, dict[str, float]]:
    model = Ridge(alpha=float(fixed_alpha)) if fixed_alpha is not None else RidgeCV(alphas=np.asarray(alphas, dtype=np.float64))
    model.fit(x_context_z[train_mask], x_external_z[train_mask])
    pred = model.predict(x_context_z)
    residual = x_external_z - pred
    train_r2 = float(r2_score(x_external_z[train_mask], pred[train_mask], multioutput="variance_weighted"))
    test_r2 = float(r2_score(x_external_z[~train_mask], pred[~train_mask], multioutput="variance_weighted"))
    alpha = float(fixed_alpha) if fixed_alpha is not None else getattr(model, "alpha_", float("nan"))
    if np.ndim(alpha) > 0:
        alpha = float(np.asarray(alpha).ravel()[0])
    return residual, {"acs_to_external_train_r2": train_r2, "acs_to_external_test_r2": test_r2, "acs_to_external_alpha": float(alpha)}


def _fit_aligned_feature_sets(
    *,
    x_context: np.ndarray,
    x_external: np.ndarray,
    scores: np.ndarray,
    train_mask: np.ndarray,
    align_dims: list[int],
    alphas: list[float],
    fixed_alpha: float | None,
    label: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    x_context_z, _ = _standardize_train(x_context, train_mask)
    x_external_z, _ = _standardize_train(x_external, train_mask)
    x_external_resid, resid_diag = _residualize_external(x_context_z, x_external_z, train_mask, alphas, fixed_alpha)

    feature_sets: dict[str, np.ndarray] = {
        "acs_all_scale": x_context,
        label: x_external,
        f"acs_all_scale_{label}_raw": np.concatenate([x_context, x_external], axis=1),
        f"{label}_acs_residualized": x_external_resid,
        f"acs_all_scale_{label}_residualized": np.concatenate([x_context, x_external_resid], axis=1),
    }
    pred_rows: list[dict[str, Any]] = []

    ridge = Ridge(alpha=float(fixed_alpha)) if fixed_alpha is not None else RidgeCV(alphas=np.asarray(alphas, dtype=np.float64))
    ridge.fit(x_external_resid[train_mask], scores[train_mask])
    yhat_ridge = ridge.predict(x_external_resid)
    feature_sets[f"{label}_aligned_ridge_yhat"] = yhat_ridge
    feature_sets[f"acs_all_scale_{label}_aligned_ridge_yhat"] = np.concatenate([x_context, yhat_ridge], axis=1)
    pred_rows.append({"alignment": f"{label}_aligned_ridge_yhat", **resid_diag})

    max_dim = max(1, min(x_external_resid.shape[1], scores.shape[1], int(train_mask.sum()) - 1))
    for dim in align_dims:
        d = max(1, min(int(dim), max_dim))
        pls = PLSRegression(n_components=d, scale=True)
        pls.fit(x_external_resid[train_mask], scores[train_mask])
        h = pls.transform(x_external_resid)
        yhat = pls.predict(x_external_resid)
        feature_sets[f"{label}_aligned_pls{d}_latent"] = h
        feature_sets[f"acs_all_scale_{label}_aligned_pls{d}_latent"] = np.concatenate([x_context, h], axis=1)
        feature_sets[f"{label}_aligned_pls{d}_yhat"] = yhat
        feature_sets[f"acs_all_scale_{label}_aligned_pls{d}_yhat"] = np.concatenate([x_context, yhat], axis=1)
        pred_rows.append({"alignment": f"{label}_aligned_pls{d}_yhat", "n_components": int(d), **resid_diag})
    return feature_sets, pred_rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Target-align an external view to copula-residual PC scores before analog retrieval. "
            "This tests whether view-target geometry mismatch can be solved by supervised low-dimensional alignment."
        )
    )
    data_root = pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data")
    ap.add_argument("--target_wide_csv", type=pathlib.Path, default=data_root / "us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv")
    ap.add_argument("--condition_csv", type=pathlib.Path, default=data_root / "us/processed/external_conditions/extcond_v1_agesex_earn_v1_acs5_2022_puma_us.csv")
    ap.add_argument("--external_csv", type=pathlib.Path, required=True)
    ap.add_argument("--external_label", default="external")
    ap.add_argument("--heldout_statefps", default="26,12,48,55,06")
    ap.add_argument("--split_mode", choices=["state_holdout", "within_state_random"], default="state_holdout")
    ap.add_argument("--test_fraction", type=float, default=0.25)
    ap.add_argument("--reference_mode", choices=["target_marginals", "acs_marginals"], default="acs_marginals")
    ap.add_argument("--joint_space", choices=["full", "coarse"], default="full")
    ap.add_argument("--max_pcs", type=int, default=40)
    ap.add_argument("--align_dims", default="2,4,8,16,32")
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--ridge_alphas", default="0.01,0.1,1,10,100,1000")
    ap.add_argument("--fixed_alpha", type=float, default=None, help="Use fixed-alpha Ridge instead of RidgeCV for faster gate runs.")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    label = _safe_label(str(args.external_label))
    out_dir = args.output_dir or pathlib.Path(f"outputs/_external_view_target_alignment_{label}_{args.joint_space}_{args.split_mode}_{_utc_ts()}")
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    target_keys, p_true, p_eq_target, _ = _load_target(args.target_wide_csv)
    _, acs_marginals, _, x_acs_all, x_scale, *_ = _load_acs_conditions(args.condition_csv, target_keys)
    x_context = np.concatenate([x_acs_all, x_scale], axis=1)
    x_external = _load_spatial(args.external_csv, target_keys)[0]

    p_eq_acs = _outer_joint([acs_marginals[v] for v in FULL_VARIABLE_ORDER])
    if args.joint_space == "coarse":
        p_true = _aggregate_full_to_coarse(p_true)
        p_eq_target = _aggregate_full_to_coarse(p_eq_target)
        p_eq_acs = _aggregate_full_to_coarse(p_eq_acs)
    reference = p_eq_target if args.reference_mode == "target_marginals" else p_eq_acs
    _, _, log_ratio = _residual_arrays(p_true, reference, eps=float(args.eps))

    heldouts = [str(x).zfill(2) for x in _parse_csv_ints(args.heldout_statefps)]
    align_dims = [int(x) for x in _parse_csv_ints(args.align_dims)]
    ks = [int(x) for x in _parse_csv_ints(args.ks)]
    alphas = [float(x) for x in str(args.ridge_alphas).split(",") if x.strip()]
    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()
    splits = _make_eval_splits(
        statefp,
        heldouts,
        split_mode=str(args.split_mode),
        test_fraction=float(args.test_fraction),
        seed=int(args.seed),
    )

    retrieval_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for heldout, train_mask in splits:
        if int((~train_mask).sum()) == 0:
            continue
        scores, explained_ratio, cumulative = _fit_pc_scores(
            log_ratio,
            train_mask=train_mask,
            n_components=int(args.max_pcs),
            seed=int(args.seed),
        )
        feature_sets, pred_diag_rows = _fit_aligned_feature_sets(
            x_context=x_context,
            x_external=x_external,
            scores=scores,
            train_mask=train_mask,
            align_dims=align_dims,
            alphas=alphas,
            fixed_alpha=args.fixed_alpha,
            label=label,
        )
        for row in pred_diag_rows:
            feature_name = str(row["alignment"])
            yhat = feature_sets[feature_name]
            prediction_rows.append(
                {
                    "heldout_statefp": heldout,
                    "split_mode": str(args.split_mode),
                    "joint_space": str(args.joint_space),
                    "reference_mode": str(args.reference_mode),
                    "feature_set": feature_name,
                    "n_features": int(yhat.shape[1]),
                    "n_pcs": int(scores.shape[1]),
                    "pc_cumulative_explained": float(cumulative),
                    **_weighted_r2_summary(scores, yhat, train_mask, explained_ratio),
                    **row,
                }
            )
        for feature_name, x in feature_sets.items():
            for k in ks:
                summ, _ = _retrieval_metrics(
                    x=x,
                    scores=scores,
                    train_mask=train_mask,
                    explained_ratio=explained_ratio,
                    k=int(k),
                )
                retrieval_rows.append(
                    {
                        "heldout_statefp": heldout,
                        "split_mode": str(args.split_mode),
                        "joint_space": str(args.joint_space),
                        "reference_mode": str(args.reference_mode),
                        "feature_set": feature_name,
                        "n_features": int(x.shape[1]),
                        "n_pcs": int(scores.shape[1]),
                        "pc_cumulative_explained": float(cumulative),
                        **summ,
                    }
                )

    retrieval_df = pd.DataFrame(retrieval_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    retrieval_df.to_csv(metrics_dir / "target_aligned_retrieval_summary.csv", index=False)
    prediction_df.to_csv(metrics_dir / "target_aligned_prediction_summary.csv", index=False)
    if not retrieval_df.empty:
        mean_retrieval = (
            retrieval_df.groupby(["k", "feature_set"], as_index=False)
            .agg(
                weighted_norm_mse=("weighted_norm_mse", "mean"),
                top5_weighted_norm_mse=("top5_weighted_norm_mse", "mean"),
                n_features=("n_features", "first"),
            )
            .sort_values(["k", "weighted_norm_mse"])
        )
        mean_retrieval.to_csv(metrics_dir / "target_aligned_retrieval_mean_summary.csv", index=False)
        print("[retrieval mean]")
        print(mean_retrieval.head(80).to_string(index=False))
    if not prediction_df.empty:
        mean_prediction = (
            prediction_df.groupby(["feature_set"], as_index=False)
            .agg(
                mean_weighted_nonnegative_r2=("weighted_nonnegative_r2", "mean"),
                mean_weighted_raw_r2=("weighted_raw_r2", "mean"),
                mean_r2_top5=("mean_r2_top5", "mean"),
                n_features=("n_features", "first"),
            )
            .sort_values("mean_weighted_nonnegative_r2", ascending=False)
        )
        mean_prediction.to_csv(metrics_dir / "target_aligned_prediction_mean_summary.csv", index=False)
        print("[prediction mean]")
        print(mean_prediction.head(40).to_string(index=False))

    _write_json(
        out_dir / "run_summary.json",
        {
            "target_wide_csv": str(args.target_wide_csv),
            "condition_csv": str(args.condition_csv),
            "external_csv": str(args.external_csv),
            "external_label": label,
            "heldout_statefps": heldouts,
            "split_mode": str(args.split_mode),
            "test_fraction": float(args.test_fraction),
            "reference_mode": str(args.reference_mode),
            "joint_space": str(args.joint_space),
            "max_pcs": int(args.max_pcs),
            "align_dims": align_dims,
            "ks": ks,
            "ridge_alphas": alphas,
            "fixed_alpha": None if args.fixed_alpha is None else float(args.fixed_alpha),
            "metrics": {
                "retrieval_summary": str(metrics_dir / "target_aligned_retrieval_summary.csv"),
                "retrieval_mean_summary": str(metrics_dir / "target_aligned_retrieval_mean_summary.csv"),
                "prediction_summary": str(metrics_dir / "target_aligned_prediction_summary.csv"),
                "prediction_mean_summary": str(metrics_dir / "target_aligned_prediction_mean_summary.csv"),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
