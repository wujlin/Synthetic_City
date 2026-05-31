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
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.model.external_c2f_full_earn_schema import FULL_VARIABLE_ORDER
from tools.model.external_c2f_full_earn_schema import (
    AGE_FINE_TO_COARSE,
    COARSE_SHAPE,
    EARN_FINE_TO_COARSE,
    ESR_FINE_TO_COARSE,
    FULL_SHAPE,
    SCHL_FINE_TO_COARSE,
)
from tools.experimental.representation.ssl_copula_residual_probe import (
    _aggregate_full_to_coarse,
    _load_acs_conditions,
    _load_spatial,
    _load_target,
    _outer_joint,
)


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)


def _tvd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum(axis=1)


def _cosine(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    num = np.sum(p * q, axis=1)
    den = np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1)
    return num / np.clip(den, 1e-12, None)


def _exp_tilt(p_eq: np.ndarray, z_hat: np.ndarray, lam: float, clip: float) -> np.ndarray:
    z = np.clip(float(lam) * np.asarray(z_hat, dtype=np.float64), -float(clip), float(clip))
    # subtract row max for numerical stability; normalization removes this constant
    z = z - z.max(axis=1, keepdims=True)
    q = np.asarray(p_eq, dtype=np.float64) * np.exp(z)
    return _normalize(q)


def _aggregate_marginal(values: np.ndarray, mapping: np.ndarray, out_dim: int) -> np.ndarray:
    values = _normalize(values)
    out = np.zeros((values.shape[0], int(out_dim)), dtype=np.float64)
    for fine_idx, coarse_idx in enumerate(np.asarray(mapping, dtype=int).tolist()):
        out[:, int(coarse_idx)] += values[:, int(fine_idx)]
    return _normalize(out)


def _constraint_marginals_for_space(
    acs_marginals: dict[str, np.ndarray],
    *,
    joint_space: str,
) -> tuple[list[np.ndarray], tuple[int, ...]]:
    if joint_space == "full":
        return [acs_marginals[v] for v in FULL_VARIABLE_ORDER], tuple(FULL_SHAPE)
    if joint_space == "coarse":
        constraints = [
            _aggregate_marginal(acs_marginals["AGEP_bin"], AGE_FINE_TO_COARSE, COARSE_SHAPE[0]),
            acs_marginals["SEX"],
            _aggregate_marginal(acs_marginals["SCHL_allpop"], SCHL_FINE_TO_COARSE, COARSE_SHAPE[2]),
            _aggregate_marginal(acs_marginals["ESR_allpop"], ESR_FINE_TO_COARSE, COARSE_SHAPE[3]),
            _aggregate_marginal(acs_marginals["EARN_16p_bin"], EARN_FINE_TO_COARSE, COARSE_SHAPE[4]),
        ]
        return constraints, tuple(COARSE_SHAPE)
    raise ValueError(f"unsupported joint_space={joint_space}")


def _mean_marginal_gap(p: np.ndarray, targets: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    tab = np.asarray(p, dtype=np.float64).reshape((p.shape[0], *shape))
    axes = tuple(range(1, len(shape) + 1))
    gaps: list[np.ndarray] = []
    for var_axis, target in enumerate(targets, start=1):
        cur = tab.sum(axis=tuple(a for a in axes if a != var_axis))
        cur = _normalize(cur)
        target = _normalize(target)
        gaps.append(_tvd(cur, target))
    return np.mean(np.vstack(gaps), axis=0)


def _ipf_project_rows(
    seed: np.ndarray,
    targets: list[np.ndarray],
    shape: tuple[int, ...],
    *,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    p = _normalize(seed).reshape((seed.shape[0], *shape))
    targets = [_normalize(x) for x in targets]
    axes = tuple(range(1, len(shape) + 1))
    for _ in range(int(max_iter)):
        max_gap = 0.0
        for var_axis, target in enumerate(targets, start=1):
            cur = p.sum(axis=tuple(a for a in axes if a != var_axis))
            factor = target / np.clip(cur, 1e-12, None)
            reshape = [p.shape[0]] + [1] * len(shape)
            reshape[var_axis] = factor.shape[1]
            p *= factor.reshape(reshape)
            max_gap = max(max_gap, float(np.max(np.abs(cur - target))))
        p /= np.clip(p.sum(axis=axes, keepdims=True), 1e-12, None)
        if max_gap <= float(tol):
            break
    return _normalize(p.reshape((seed.shape[0], -1)))


def _ridge_predict_scores(
    x: np.ndarray,
    scores: np.ndarray,
    train_mask: np.ndarray,
    alphas: list[float],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    pred = np.zeros_like(scores, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for j in range(scores.shape[1]):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.asarray(alphas, dtype=float)))
        model.fit(x[train_mask], scores[train_mask, j])
        pred[:, j] = model.predict(x)
        rows.append(
            {
                "pc_index": int(j + 1),
                "ridge_alpha": float(model.named_steps["ridgecv"].alpha_),
                "r2_train": float(r2_score(scores[train_mask, j], pred[train_mask, j])),
                "r2_test": float(r2_score(scores[~train_mask, j], pred[~train_mask, j])),
            }
        )
    return pred, rows


def _mlp_predict_scores(
    x: np.ndarray,
    scores: np.ndarray,
    train_mask: np.ndarray,
    explained_ratio: np.ndarray,
    *,
    seed: int,
    epochs: int,
    hidden_dim: int,
    embed_dim: int,
    lr: float,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self, d_in: int, hidden: int, embed: int, d_out: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, embed),
                nn.ReLU(),
                nn.Linear(embed, d_out),
            )

        def forward(self, x_in: torch.Tensor) -> torch.Tensor:
            return self.net(x_in)

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(x[train_mask]).astype(np.float32)
    x_all = scaler_x.transform(x).astype(np.float32)
    y_train_raw = scores[train_mask].astype(np.float64)
    y_mean = y_train_raw.mean(axis=0, keepdims=True)
    y_std = y_train_raw.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)
    y_train = ((y_train_raw - y_mean) / y_std).astype(np.float32)

    dev = torch.device(device)
    model = Model(x.shape[1], int(hidden_dim), int(embed_dim), scores.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=dev)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=dev)
    x_all_t = torch.tensor(x_all, dtype=torch.float32, device=dev)
    weights = np.asarray(explained_ratio, dtype=np.float64)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    weights_t = torch.tensor(weights.astype(np.float32), dtype=torch.float32, device=dev).reshape(1, -1)
    n_train = int(x_train.shape[0])
    batch_size = min(int(batch_size), n_train)
    for _ in range(int(epochs)):
        order = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            pred = model(x_train_t[idx])
            loss = torch.mean(((pred - y_train_t[idx]) ** 2) * weights_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    with torch.no_grad():
        pred_all = model(x_all_t).cpu().numpy().astype(np.float64) * y_std + y_mean
    info = {
        "seed": int(seed),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "embed_dim": int(embed_dim),
        "lr": float(lr),
    }
    return pred_all, info


def _summarize_metric(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.quantile(values, 0.5)),
    }


def _append_distribution_rows(
    *,
    rows: list[dict[str, Any]],
    by_puma_rows: list[dict[str, Any]],
    puma_meta: pd.DataFrame,
    heldout_mask: np.ndarray,
    heldout: str,
    feature_set: str,
    method: str,
    seed: int,
    lam: float,
    projection_mode: str,
    n_train: int,
    p_true_test: np.ndarray,
    p_reference_test: np.ndarray,
    q: np.ndarray,
    constraint_targets_test: list[np.ndarray],
    constraint_shape: tuple[int, ...],
    pc_weighted_nonnegative_r2: float,
    pc_weighted_raw_r2: float,
) -> None:
    tvd = _tvd(p_true_test, q)
    cosine = _cosine(p_true_test, q)
    baseline_tvd = _tvd(p_true_test, p_reference_test)
    delta = baseline_tvd - tvd
    marginal_gap = _mean_marginal_gap(q, constraint_targets_test, constraint_shape)
    rows.append(
        {
            "heldout_statefp": heldout,
            "feature_set": feature_set,
            "method": method,
            "seed": int(seed),
            "lambda": float(lam),
            "projection_mode": projection_mode,
            "n_train": int(n_train),
            "n_test": int(p_true_test.shape[0]),
            "tvd_mean": float(np.mean(tvd)),
            "tvd_std": float(np.std(tvd)),
            "cosine_mean": float(np.mean(cosine)),
            "mean_marginal_gap": float(np.mean(marginal_gap)),
            "tvd_delta_vs_reference": float(np.mean(delta)),
            "tvd_relative_improvement": float(np.mean(delta) / max(float(np.mean(baseline_tvd)), 1e-12)),
            "pc_weighted_nonnegative_r2": float(pc_weighted_nonnegative_r2),
            "pc_weighted_raw_r2": float(pc_weighted_raw_r2),
        }
    )
    meta = puma_meta.loc[heldout_mask, ["puma_uid_key", "statefp", "puma5"]].reset_index(drop=True)
    for i in range(p_true_test.shape[0]):
        by_puma_rows.append(
            {
                "puma_uid": str(meta.loc[i, "puma_uid_key"]).zfill(7),
                "statefp": str(meta.loc[i, "statefp"]).zfill(2),
                "puma5": str(meta.loc[i, "puma5"]).zfill(5),
                "heldout_statefp": heldout,
                "feature_set": feature_set,
                "method": method,
                "seed": int(seed),
                "lambda": float(lam),
                "projection_mode": projection_mode,
                "tvd": float(tvd[i]),
                "cosine": float(cosine[i]),
                "mean_marginal_gap": float(marginal_gap[i]),
                "reference_tvd": float(baseline_tvd[i]),
                "delta_vs_reference": float(delta[i]),
            }
        )


def _write_baseline_comparison(
    *,
    correction_by_puma: pd.DataFrame,
    baseline_csv: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    if not baseline_csv.exists():
        raise SystemExit(f"baseline_by_puma_csv not found: {baseline_csv}")
    base = pd.read_csv(baseline_csv)
    base["puma_uid"] = base["puma_uid"].map(lambda x: str(int(float(x))).zfill(7) if str(x).replace(".", "", 1).isdigit() else str(x).zfill(7))
    comp = correction_by_puma.merge(base, on="puma_uid", how="left", suffixes=("", "_baseline"))
    comp.to_csv(out_path.parent / "baseline_comparison_by_puma.csv", index=False)

    baseline_cols = [
        "pipeline_tvd_mean",
        "ipf_tvd",
        "one_shot_tvd_mean",
        "tvd_co_national",
        "tvd_co_local",
        "tvd_independence",
    ]
    rows: list[dict[str, Any]] = []
    group_cols = ["feature_set", "method", "seed", "lambda", "projection_mode"]
    for key, sub in comp.groupby(group_cols, dropna=False):
        row: dict[str, Any] = dict(zip(group_cols, key))
        row["n"] = int(sub["tvd"].notna().sum())
        row["tvd_mean"] = float(pd.to_numeric(sub["tvd"], errors="coerce").mean())
        row["tvd_std"] = float(pd.to_numeric(sub["tvd"], errors="coerce").std(ddof=0))
        row["mean_marginal_gap"] = float(pd.to_numeric(sub["mean_marginal_gap"], errors="coerce").mean())
        for col in baseline_cols:
            if col not in sub.columns:
                continue
            a = pd.to_numeric(sub["tvd"], errors="coerce")
            b = pd.to_numeric(sub[col], errors="coerce")
            ok = a.notna() & b.notna()
            if not bool(ok.any()):
                continue
            diff = a[ok].astype(float) - b[ok].astype(float)
            row[f"delta_vs_{col}"] = float(diff.mean())
            row[f"win_rate_vs_{col}"] = float((diff < 0.0).mean())
            row[f"{col}_mean"] = float(b[ok].mean())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["projection_mode", "tvd_mean"], ascending=[True, True])
    summary.to_csv(out_path, index=False)


def main() -> int:
    data_root = pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data")
    ap = argparse.ArgumentParser(description="Convert predicted log-ratio copula residuals into corrected distributions.")
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
    ap.add_argument("--joint_space", choices=["coarse", "full"], default="coarse")
    ap.add_argument("--reference_mode", choices=["target_marginals", "acs_marginals"], default="target_marginals")
    ap.add_argument("--heldout_statefps", default="26,06,12,48,55")
    ap.add_argument("--feature_sets", default="acs_1d,acs_all")
    ap.add_argument("--methods", default="ridge,mlp")
    ap.add_argument("--n_pcs", type=int, default=40)
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--ridge_alphas", default="0.01,0.1,1,10,100,1000")
    ap.add_argument("--mlp_seeds", default="0,1,2")
    ap.add_argument("--mlp_epochs", type=int, default=250)
    ap.add_argument("--mlp_hidden_dim", type=int, default=128)
    ap.add_argument("--mlp_embed_dim", type=int, default=32)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--clip_log_ratio", type=float, default=8.0)
    ap.add_argument("--projection_modes", default="none", help="Comma-separated projection modes: none,acs_ipf.")
    ap.add_argument("--ipf_iters", type=int, default=500)
    ap.add_argument("--ipf_tol", type=float, default=1e-9)
    ap.add_argument("--baseline_by_puma_csv", type=pathlib.Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    out_dir = args.output_dir or pathlib.Path(f"outputs/_residual_aware_correction_{_utc_ts()}")
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    target_keys, p_true, p_eq_target, target_marginals = _load_target(args.target_wide_csv)
    acs_feature_df, acs_marginals, x_acs_1d, x_acs_all, _, _ = _load_acs_conditions(args.condition_csv, target_keys)
    x_spatial, _ = _load_spatial(args.spatial_csv, target_keys)
    p_eq_acs = _outer_joint([acs_marginals[v] for v in FULL_VARIABLE_ORDER])

    if args.joint_space == "coarse":
        p_true = _aggregate_full_to_coarse(p_true)
        p_eq_target = _aggregate_full_to_coarse(p_eq_target)
        p_eq_acs = _aggregate_full_to_coarse(p_eq_acs)

    p_true = _normalize(p_true)
    p_eq = _normalize(p_eq_target if args.reference_mode == "target_marginals" else p_eq_acs)
    log_ratio = np.log(p_true + float(args.eps)) - np.log(p_eq + float(args.eps))

    feature_map = {
        "acs_1d": x_acs_1d,
        "acs_all": x_acs_all,
        "acs_all_spatial": np.concatenate([x_acs_all, x_spatial], axis=1),
    }
    feature_sets = [x.strip() for x in str(args.feature_sets).split(",") if x.strip()]
    methods = [x.strip() for x in str(args.methods).split(",") if x.strip()]
    projection_modes = [x.strip() for x in str(args.projection_modes).split(",") if x.strip()]
    bad_projection_modes = sorted(set(projection_modes) - {"none", "acs_ipf"})
    if bad_projection_modes:
        raise SystemExit(f"unknown projection_modes: {bad_projection_modes}")
    lambdas = _parse_floats(args.lambdas)
    heldout_statefps = [str(x).zfill(2) for x in _parse_ints(args.heldout_statefps)]
    ridge_alphas = _parse_floats(args.ridge_alphas)
    mlp_seeds = _parse_ints(args.mlp_seeds)
    constraint_marginals, constraint_shape = _constraint_marginals_for_space(
        acs_marginals,
        joint_space=str(args.joint_space),
    )

    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()
    rows: list[dict[str, Any]] = []
    by_puma_rows: list[dict[str, Any]] = []
    pc_rows: list[dict[str, Any]] = []

    for heldout in heldout_statefps:
        train_mask = statefp != heldout
        if int((~train_mask).sum()) == 0:
            continue
        n_comp = min(int(args.n_pcs), int(train_mask.sum()) - 1, log_ratio.shape[1])
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=0)
        scores_train = pca.fit_transform(log_ratio[train_mask])
        scores_all = pca.transform(log_ratio)
        evr = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
        heldout_mask = ~train_mask
        constraint_targets_test = [m[heldout_mask] for m in constraint_marginals]
        p_reference_test = p_eq[heldout_mask]
        p_true_test = p_true[heldout_mask]

        _append_distribution_rows(
            rows=rows,
            by_puma_rows=by_puma_rows,
            puma_meta=target_keys,
            heldout_mask=heldout_mask,
            heldout=heldout,
            feature_set="reference",
            method="reference",
            seed=-1,
            lam=0.0,
            projection_mode="none",
            n_train=int(train_mask.sum()),
            p_true_test=p_true_test,
            p_reference_test=p_reference_test,
            q=p_reference_test,
            constraint_targets_test=constraint_targets_test,
            constraint_shape=constraint_shape,
            pc_weighted_nonnegative_r2=0.0,
            pc_weighted_raw_r2=0.0,
        )
        if "acs_ipf" in projection_modes:
            q_ref_proj = _ipf_project_rows(
                p_reference_test,
                constraint_targets_test,
                constraint_shape,
                max_iter=int(args.ipf_iters),
                tol=float(args.ipf_tol),
            )
            _append_distribution_rows(
                rows=rows,
                by_puma_rows=by_puma_rows,
                puma_meta=target_keys,
                heldout_mask=heldout_mask,
                heldout=heldout,
                feature_set="reference",
                method="reference",
                seed=-1,
                lam=0.0,
                projection_mode="acs_ipf",
                n_train=int(train_mask.sum()),
                p_true_test=p_true_test,
                p_reference_test=p_reference_test,
                q=q_ref_proj,
                constraint_targets_test=constraint_targets_test,
                constraint_shape=constraint_shape,
                pc_weighted_nonnegative_r2=0.0,
                pc_weighted_raw_r2=0.0,
            )

        for feature_name in feature_sets:
            if feature_name not in feature_map:
                raise SystemExit(f"unknown feature_set: {feature_name}")
            x = feature_map[feature_name]

            predictions: list[tuple[str, int, np.ndarray, list[dict[str, Any]]]] = []
            if "ridge" in methods:
                pred_scores, ridge_rows = _ridge_predict_scores(x, scores_all, train_mask, ridge_alphas)
                predictions.append(("ridge", -1, pred_scores, ridge_rows))

            if "mlp" in methods:
                for seed in mlp_seeds:
                    pred_scores, info = _mlp_predict_scores(
                        x,
                        scores_all,
                        train_mask,
                        evr,
                        seed=int(seed),
                        epochs=int(args.mlp_epochs),
                        hidden_dim=int(args.mlp_hidden_dim),
                        embed_dim=int(args.mlp_embed_dim),
                        lr=float(args.mlp_lr),
                        batch_size=int(args.batch_size),
                        device=str(args.device),
                    )
                    predictions.append(("mlp", int(seed), pred_scores, []))

            for method, seed, pred_scores, score_rows in predictions:
                pc_r2 = []
                for j in range(n_comp):
                    r2_test = float(r2_score(scores_all[~train_mask, j], pred_scores[~train_mask, j]))
                    pc_r2.append(r2_test)
                    pc_rows.append(
                        {
                            "heldout_statefp": heldout,
                            "feature_set": feature_name,
                            "method": method,
                            "seed": int(seed),
                            "pc_index": int(j + 1),
                            "explained_variance_ratio": float(evr[j]),
                            "r2_test": r2_test,
                        }
                    )
                pc_r2_arr = np.asarray(pc_r2, dtype=np.float64)
                weighted_nonneg = float(np.sum(np.maximum(pc_r2_arr, 0.0) * evr))
                weighted_raw = float(np.sum(pc_r2_arr * evr))
                z_hat_all = pca.inverse_transform(pred_scores)
                z_hat_test = z_hat_all[heldout_mask]
                for lam in lambdas:
                    q_raw = _exp_tilt(p_reference_test, z_hat_test, lam=float(lam), clip=float(args.clip_log_ratio))
                    if "none" in projection_modes:
                        _append_distribution_rows(
                            rows=rows,
                            by_puma_rows=by_puma_rows,
                            puma_meta=target_keys,
                            heldout_mask=heldout_mask,
                            heldout=heldout,
                            feature_set=feature_name,
                            method=method,
                            seed=int(seed),
                            lam=float(lam),
                            projection_mode="none",
                            n_train=int(train_mask.sum()),
                            p_true_test=p_true_test,
                            p_reference_test=p_reference_test,
                            q=q_raw,
                            constraint_targets_test=constraint_targets_test,
                            constraint_shape=constraint_shape,
                            pc_weighted_nonnegative_r2=weighted_nonneg,
                            pc_weighted_raw_r2=weighted_raw,
                        )
                    if "acs_ipf" in projection_modes:
                        q_projected = _ipf_project_rows(
                            q_raw,
                            constraint_targets_test,
                            constraint_shape,
                            max_iter=int(args.ipf_iters),
                            tol=float(args.ipf_tol),
                        )
                        _append_distribution_rows(
                            rows=rows,
                            by_puma_rows=by_puma_rows,
                            puma_meta=target_keys,
                            heldout_mask=heldout_mask,
                            heldout=heldout,
                            feature_set=feature_name,
                            method=method,
                            seed=int(seed),
                            lam=float(lam),
                            projection_mode="acs_ipf",
                            n_train=int(train_mask.sum()),
                            p_true_test=p_true_test,
                            p_reference_test=p_reference_test,
                            q=q_projected,
                            constraint_targets_test=constraint_targets_test,
                            constraint_shape=constraint_shape,
                            pc_weighted_nonnegative_r2=weighted_nonneg,
                            pc_weighted_raw_r2=weighted_raw,
                        )

        zero_scores = np.zeros_like(scores_all, dtype=np.float64)
        for method, seed, pred_scores in [
            ("mean_residual", -1, zero_scores),
            ("oracle_pc", -1, scores_all),
        ]:
            pc_r2 = []
            for j in range(n_comp):
                if method == "oracle_pc":
                    r2_test = 1.0
                else:
                    r2_test = float(r2_score(scores_all[~train_mask, j], pred_scores[~train_mask, j]))
                pc_r2.append(r2_test)
                pc_rows.append(
                    {
                        "heldout_statefp": heldout,
                        "feature_set": "pca_prior",
                        "method": method,
                        "seed": int(seed),
                        "pc_index": int(j + 1),
                        "explained_variance_ratio": float(evr[j]),
                        "r2_test": r2_test,
                    }
                )
            pc_r2_arr = np.asarray(pc_r2, dtype=np.float64)
            weighted_nonneg = float(np.sum(np.maximum(pc_r2_arr, 0.0) * evr))
            weighted_raw = float(np.sum(pc_r2_arr * evr))
            z_hat_test = pca.inverse_transform(pred_scores)[heldout_mask]
            for lam in lambdas:
                q_raw = _exp_tilt(p_reference_test, z_hat_test, lam=float(lam), clip=float(args.clip_log_ratio))
                if "none" in projection_modes:
                    _append_distribution_rows(
                        rows=rows,
                        by_puma_rows=by_puma_rows,
                        puma_meta=target_keys,
                        heldout_mask=heldout_mask,
                        heldout=heldout,
                        feature_set="pca_prior",
                        method=method,
                        seed=int(seed),
                        lam=float(lam),
                        projection_mode="none",
                        n_train=int(train_mask.sum()),
                        p_true_test=p_true_test,
                        p_reference_test=p_reference_test,
                        q=q_raw,
                        constraint_targets_test=constraint_targets_test,
                        constraint_shape=constraint_shape,
                        pc_weighted_nonnegative_r2=weighted_nonneg,
                        pc_weighted_raw_r2=weighted_raw,
                    )
                if "acs_ipf" in projection_modes:
                    q_projected = _ipf_project_rows(
                        q_raw,
                        constraint_targets_test,
                        constraint_shape,
                        max_iter=int(args.ipf_iters),
                        tol=float(args.ipf_tol),
                    )
                    _append_distribution_rows(
                        rows=rows,
                        by_puma_rows=by_puma_rows,
                        puma_meta=target_keys,
                        heldout_mask=heldout_mask,
                        heldout=heldout,
                        feature_set="pca_prior",
                        method=method,
                        seed=int(seed),
                        lam=float(lam),
                        projection_mode="acs_ipf",
                        n_train=int(train_mask.sum()),
                        p_true_test=p_true_test,
                        p_reference_test=p_reference_test,
                        q=q_projected,
                        constraint_targets_test=constraint_targets_test,
                        constraint_shape=constraint_shape,
                        pc_weighted_nonnegative_r2=weighted_nonneg,
                        pc_weighted_raw_r2=weighted_raw,
                    )

    metrics = pd.DataFrame(rows)
    by_puma_metrics = pd.DataFrame(by_puma_rows)
    pc_metrics = pd.DataFrame(pc_rows)
    metrics.to_csv(out_dir / "metrics" / "correction_tvd_long.csv", index=False)
    by_puma_metrics.to_csv(out_dir / "metrics" / "correction_tvd_by_puma.csv", index=False)
    pc_metrics.to_csv(out_dir / "metrics" / "score_prediction_pc_long.csv", index=False)

    summary = (
        metrics[metrics["method"] != "reference"]
        .groupby(["feature_set", "method", "seed", "lambda", "projection_mode"], as_index=False)
        .agg(
            tvd_mean=("tvd_mean", "mean"),
            tvd_std=("tvd_std", "mean"),
            cosine_mean=("cosine_mean", "mean"),
            mean_marginal_gap=("mean_marginal_gap", "mean"),
            tvd_delta_vs_reference=("tvd_delta_vs_reference", "mean"),
            tvd_relative_improvement=("tvd_relative_improvement", "mean"),
            pc_weighted_nonnegative_r2=("pc_weighted_nonnegative_r2", "mean"),
            pc_weighted_raw_r2=("pc_weighted_raw_r2", "mean"),
        )
        .sort_values(["tvd_delta_vs_reference"], ascending=False)
    )
    summary.to_csv(out_dir / "metrics" / "correction_tvd_summary.csv", index=False)
    if args.baseline_by_puma_csv is not None:
        _write_baseline_comparison(
            correction_by_puma=by_puma_metrics,
            baseline_csv=args.baseline_by_puma_csv,
            out_path=out_dir / "metrics" / "baseline_comparison_summary.csv",
        )

    run_summary = {
        "created_utc": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "question": "Can predicted log-ratio copula residuals reduce held-out TVD after exponential tilting?",
        "target_wide_csv": str(args.target_wide_csv),
        "condition_csv": str(args.condition_csv),
        "spatial_csv": str(args.spatial_csv),
        "joint_space": str(args.joint_space),
        "reference_mode": str(args.reference_mode),
        "n_regions": int(p_true.shape[0]),
        "joint_k": int(p_true.shape[1]),
        "heldout_statefps": heldout_statefps,
        "feature_sets": feature_sets,
        "methods": methods,
        "n_pcs": int(args.n_pcs),
        "lambdas": lambdas,
        "clip_log_ratio": float(args.clip_log_ratio),
        "projection_modes": projection_modes,
        "ipf_iters": int(args.ipf_iters),
        "ipf_tol": float(args.ipf_tol),
        "baseline_by_puma_csv": str(args.baseline_by_puma_csv) if args.baseline_by_puma_csv else None,
        "mlp": {
            "seeds": mlp_seeds,
            "epochs": int(args.mlp_epochs),
            "hidden_dim": int(args.mlp_hidden_dim),
            "embed_dim": int(args.mlp_embed_dim),
            "lr": float(args.mlp_lr),
            "device": str(args.device),
        },
        "outputs": {
            "correction_tvd_long": str(out_dir / "metrics" / "correction_tvd_long.csv"),
            "correction_tvd_by_puma": str(out_dir / "metrics" / "correction_tvd_by_puma.csv"),
            "correction_tvd_summary": str(out_dir / "metrics" / "correction_tvd_summary.csv"),
            "score_prediction_pc_long": str(out_dir / "metrics" / "score_prediction_pc_long.csv"),
            "baseline_comparison_summary": str(out_dir / "metrics" / "baseline_comparison_summary.csv")
            if args.baseline_by_puma_csv
            else None,
        },
    }
    _write_json(out_dir / "run_summary.json", run_summary)
    print(summary.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
