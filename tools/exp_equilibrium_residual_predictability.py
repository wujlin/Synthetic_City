#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import pathlib
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summary(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "median": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _ridge_fit_predict(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    x_eval = np.asarray(x_eval, dtype=float)

    x_mean = x_train.mean(axis=0, dtype=np.float64)
    x_std = x_train.std(axis=0, dtype=np.float64)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    xtr = (x_train - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)
    xev = (x_eval - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)

    y_mean = float(np.mean(y_train))
    y_ctr = y_train - y_mean

    xtx = xtr.T @ xtr
    reg = alpha * np.eye(xtx.shape[0], dtype=float)
    beta = np.linalg.solve(xtx + reg, xtr.T @ y_ctr)

    pred_train = y_mean + xtr @ beta
    pred_eval = y_mean + xev @ beta
    return pred_train, pred_eval


def _fold_indices(n: int, *, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return [order[i::n_folds] for i in range(n_folds)]


def _select_ridge_alpha(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    alphas: list[float],
    n_folds: int,
    seed: int,
) -> float:
    idx_folds = _fold_indices(x_train.shape[0], n_folds=n_folds, seed=seed)
    best_alpha = float(alphas[0])
    best_score = -math.inf
    for alpha in alphas:
        scores: list[float] = []
        for i in range(n_folds):
            val_idx = idx_folds[i]
            tr_idx = np.concatenate([idx_folds[j] for j in range(n_folds) if j != i], axis=0)
            _, pred_val = _ridge_fit_predict(
                x_train=x_train[tr_idx],
                y_train=y_train[tr_idx],
                x_eval=x_train[val_idx],
                alpha=float(alpha),
            )
            scores.append(_r2_score(y_train[val_idx], pred_val))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = float(alpha)
    return best_alpha


def _collapse_axis(arr: np.ndarray, axis: int, groups: list[list[int]]) -> np.ndarray:
    out = []
    for g in groups:
        out.append(np.take(arr, indices=g, axis=axis).sum(axis=axis))
    return np.stack(out, axis=axis)


def _aggregate_shape(p512: np.ndarray, k_mode: int) -> tuple[np.ndarray, tuple[int, ...], dict[str, list[str]]]:
    arr = p512.reshape((-1, 4, 2, 4, 4, 4))
    labels_512 = {
        "age": ["0-24", "25-44", "45-64", "65+"],
        "sex": ["male", "female"],
        "income": ["<25k", "25k-50k", "50k-100k", "100k+"],
        "schl": ["<HS", "HS/GED", "SomeCollege", "BA+"],
        "esr": ["Employed", "Unemployed", "Armed", "NILF"],
    }
    if int(k_mode) == 512:
        return arr.reshape((arr.shape[0], -1)), (4, 2, 4, 4, 4), labels_512

    if int(k_mode) == 128:
        arr = _collapse_axis(arr, 4, [[0, 1], [2, 3]])
        arr = _collapse_axis(arr, 5, [[0, 1, 2], [3]])
        labels = {
            "age": labels_512["age"],
            "sex": labels_512["sex"],
            "income": labels_512["income"],
            "schl": ["<SomeCollege", "SomeCollege+"],
            "esr": ["LaborForce", "NILF"],
        }
        return arr.reshape((arr.shape[0], -1)), (4, 2, 4, 2, 2), labels

    if int(k_mode) == 32:
        arr = _collapse_axis(arr, 1, [[0, 1], [2, 3]])
        arr = _collapse_axis(arr, 3, [[0, 1], [2, 3]])
        arr = _collapse_axis(arr, 4, [[0, 1], [2, 3]])
        arr = _collapse_axis(arr, 5, [[0, 1, 2], [3]])
        labels = {
            "age": ["0-44", "45+"],
            "sex": labels_512["sex"],
            "income": ["<50k", "50k+"],
            "schl": ["<SomeCollege", "SomeCollege+"],
            "esr": ["LaborForce", "NILF"],
        }
        return arr.reshape((arr.shape[0], -1)), (2, 2, 2, 2, 2), labels

    raise ValueError(f"unsupported k_mode={k_mode}")


def _compute_marginals(p: np.ndarray, shape: tuple[int, ...]) -> list[np.ndarray]:
    arr = p.reshape((-1, *shape))
    out: list[np.ndarray] = []
    for axis in range(1, arr.ndim):
        reduce_axes = tuple(ax for ax in range(1, arr.ndim) if ax != axis)
        out.append(arr.sum(axis=reduce_axes))
    return out


def _independence_joint(marginals: list[np.ndarray]) -> np.ndarray:
    n = marginals[0].shape[0]
    out = marginals[0][:, :, None, None, None]
    out = out * marginals[1][:, None, :, None, None]
    out = out * marginals[2][:, None, None, :, None]
    out = out * marginals[3][:, None, None, None, :]
    out = out[..., None] * marginals[4][:, None, None, None, None, :]
    return out.reshape((n, -1))


def _load_joint_wide(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _pca_from_train(z_train: np.ndarray, z_test: np.ndarray, explained_var_threshold: float) -> dict[str, Any]:
    mu = z_train.mean(axis=0, dtype=np.float64)
    ztr = z_train - mu.reshape(1, -1)
    zte = z_test - mu.reshape(1, -1)

    u, s, vt = np.linalg.svd(ztr, full_matrices=False)
    denom = max(ztr.shape[0] - 1, 1)
    explained_var = (s**2) / denom
    explained_ratio = explained_var / max(float(explained_var.sum()), 1e-12)
    cum = np.cumsum(explained_ratio)
    n_comp = int(np.searchsorted(cum, float(explained_var_threshold), side="left") + 1)
    n_comp = max(1, min(n_comp, vt.shape[0]))
    components = vt[:n_comp]
    train_scores = ztr @ components.T
    test_scores = zte @ components.T
    return {
        "mean": mu,
        "components": components,
        "explained_var": explained_var[:n_comp],
        "explained_ratio": explained_ratio[:n_comp],
        "cumulative_explained_ratio": float(cum[n_comp - 1]),
        "train_scores": train_scores,
        "test_scores": test_scores,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure pure MaxEnt residual magnitude and predictability on PUMA joint distributions.")
    ap.add_argument("--joint_wide_csv", type=pathlib.Path, default=pathlib.Path("outputs/_tmp_puma5var_us_smoke/puma_5var_joint_wide.csv"))
    ap.add_argument("--k_modes", type=str, default="32,128,512")
    ap.add_argument("--test_statefp", type=str, default="26")
    ap.add_argument("--explained_var_threshold", type=float, default=0.90)
    ap.add_argument("--ridge_alphas", type=str, default="0.01,0.1,1,10,100")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--predictable_r2_threshold", type=float, default=0.10)
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    out_dir = args.output_dir or pathlib.Path(f"outputs/_exp_equilibrium_residual_predictability_{_utc_ts()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_joint_wide(args.joint_wide_csv)
    joint_cols = [c for c in df.columns if c.startswith("p_joint_")]
    if len(joint_cols) != 512:
        raise SystemExit(f"expected 512 p_joint columns, got {len(joint_cols)}")

    p512 = df[joint_cols].to_numpy(dtype=float)
    p512 = p512 / np.clip(p512.sum(axis=1, keepdims=True), 1e-12, None)

    statefp = df["statefp"].astype(str).str.zfill(2).to_numpy()
    puma_uid = df["puma_uid"].astype(str).to_numpy()
    train_mask = statefp != str(args.test_statefp).zfill(2)
    test_mask = ~train_mask

    k_modes = [int(x.strip()) for x in str(args.k_modes).split(",") if x.strip()]
    ridge_alphas = [float(x.strip()) for x in str(args.ridge_alphas).split(",") if x.strip()]

    run_summary: dict[str, Any] = {
        "created_utc": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "joint_wide_csv": str(args.joint_wide_csv.resolve()),
        "n_regions_total": int(df.shape[0]),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "test_statefp": str(args.test_statefp).zfill(2),
        "k_modes": k_modes,
        "explained_var_threshold": float(args.explained_var_threshold),
        "ridge_alphas": ridge_alphas,
        "cv_folds": int(args.cv_folds),
        "predictable_r2_threshold": float(args.predictable_r2_threshold),
        "results": {},
    }

    region_rows: list[pd.DataFrame] = []

    for k_mode in k_modes:
        p_true, shape, labels = _aggregate_shape(p512, k_mode=int(k_mode))
        marginals = _compute_marginals(p_true, shape)
        x = np.concatenate(marginals, axis=1)
        p_eq = _independence_joint(marginals)

        delta_prob = p_true - p_eq
        tvd = 0.5 * np.abs(delta_prob).sum(axis=1)
        kl = np.sum(p_true * (np.log(p_true + float(args.eps)) - np.log(p_eq + float(args.eps))), axis=1)
        log_ratio = np.log(p_true + float(args.eps)) - np.log(p_eq + float(args.eps))

        pca = _pca_from_train(
            z_train=log_ratio[train_mask],
            z_test=log_ratio[test_mask],
            explained_var_threshold=float(args.explained_var_threshold),
        )

        pc_metrics: list[dict[str, Any]] = []
        explained_ratio = np.asarray(pca["explained_ratio"], dtype=float)
        train_scores = np.asarray(pca["train_scores"], dtype=float)
        test_scores = np.asarray(pca["test_scores"], dtype=float)

        weighted_predictable_share = 0.0
        threshold_predictable_share = 0.0

        for i in range(train_scores.shape[1]):
            y_train = train_scores[:, i]
            y_test = test_scores[:, i]
            alpha = _select_ridge_alpha(
                x_train=x[train_mask],
                y_train=y_train,
                alphas=ridge_alphas,
                n_folds=min(int(args.cv_folds), max(2, train_scores.shape[0] // 20)),
                seed=int(args.seed) + i,
            )
            pred_train, pred_test = _ridge_fit_predict(
                x_train=x[train_mask],
                y_train=y_train,
                x_eval=x[test_mask],
                alpha=float(alpha),
            )
            r2_train = _r2_score(y_train, pred_train)
            r2_test = _r2_score(y_test, pred_test)
            weighted_predictable_share += max(0.0, r2_test) * float(explained_ratio[i])
            if r2_test >= float(args.predictable_r2_threshold):
                threshold_predictable_share += float(explained_ratio[i])
            pc_metrics.append(
                {
                    "pc_index": int(i + 1),
                    "explained_variance_ratio": float(explained_ratio[i]),
                    "ridge_alpha": float(alpha),
                    "r2_train": float(r2_train),
                    "r2_test": float(r2_test),
                }
            )

        result = {
            "shape": {
                "age": int(shape[0]),
                "sex": int(shape[1]),
                "income": int(shape[2]),
                "schl": int(shape[3]),
                "esr": int(shape[4]),
            },
            "constraint_dim": int(x.shape[1]),
            "equilibrium_deviation": {
                "tvd_all": _summary(tvd),
                "tvd_train": _summary(tvd[train_mask]),
                "tvd_test": _summary(tvd[test_mask]),
                "kl_all": _summary(kl),
                "kl_train": _summary(kl[train_mask]),
                "kl_test": _summary(kl[test_mask]),
            },
            "predictability": {
                "transform": "log_ratio",
                "n_pcs": int(train_scores.shape[1]),
                "cumulative_explained_ratio": float(pca["cumulative_explained_ratio"]),
                "weighted_predictable_share_nonneg_r2": float(weighted_predictable_share),
                "threshold_predictable_share": float(threshold_predictable_share),
                "pc_metrics": pc_metrics,
            },
            "labels": labels,
        }
        run_summary["results"][str(k_mode)] = result

        score_data: dict[str, Any] = {
            "statefp": statefp,
            "puma_uid": puma_uid,
            "split": np.where(train_mask, "train", "test"),
            "k_mode": np.full(statefp.shape[0], int(k_mode), dtype=int),
            "tvd_eq": tvd,
            "kl_eq": kl,
        }
        for i in range(train_scores.shape[1]):
            vals = np.empty(statefp.shape[0], dtype=float)
            vals[train_mask] = train_scores[:, i]
            vals[test_mask] = test_scores[:, i]
            score_data[f"pc{i+1}_score"] = vals
        region_rows.append(pd.DataFrame(score_data))

    region_df = pd.concat(region_rows, axis=0, ignore_index=True)
    region_df.to_csv(out_dir / "region_residual_metrics.csv", index=False)
    _write_json(out_dir / "run_summary.json", run_summary)

    concise = {}
    for k, v in run_summary["results"].items():
        concise[k] = {
            "tvd_test_mean": v["equilibrium_deviation"]["tvd_test"]["mean"],
            "kl_test_mean": v["equilibrium_deviation"]["kl_test"]["mean"],
            "n_pcs": v["predictability"]["n_pcs"],
            "weighted_predictable_share_nonneg_r2": v["predictability"]["weighted_predictable_share_nonneg_r2"],
            "threshold_predictable_share": v["predictability"]["threshold_predictable_share"],
        }
    _write_json(out_dir / "metrics" / "concise_summary.json", concise)
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
