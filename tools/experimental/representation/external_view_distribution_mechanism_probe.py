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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.model.external_c2f_full_earn_schema import COARSE_SHAPE, FULL_SHAPE, FULL_VARIABLE_ORDER
from tools.experimental.representation.ssl_copula_residual_probe import (  # noqa: E402
    _aggregate_full_to_coarse,
    _add_puma_uid,
    _load_acs_conditions,
    _load_spatial,
    _load_target,
    _normalize_rows,
    _outer_joint,
    _parse_csv_ints,
    _residual_arrays,
    _write_json,
)


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_label(text: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "external"


def _canon_statefps(raw: str) -> list[str]:
    return [str(x).zfill(2) for x in _parse_csv_ints(raw)]


def _row_tvd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).sum(axis=1)


def _row_l1(a: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(a, dtype=np.float64)).sum(axis=1)


def _row_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    num = (aa * bb).sum(axis=1)
    den = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
    return num / np.clip(den, eps, None)


def _normalize_positive_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    sums = arr.sum(axis=1, keepdims=True)
    fallback = np.full_like(arr, 1.0 / max(arr.shape[1], 1))
    return np.where(sums > eps, arr / np.clip(sums, eps, None), fallback)


def _ipf_project_rows(
    seed_joint: np.ndarray,
    target_marginals: list[np.ndarray],
    *,
    shape: tuple[int, ...],
    max_iter: int,
    eps: float,
) -> np.ndarray:
    """Vectorized IPF projection for many rows with the same joint shape."""
    n = int(seed_joint.shape[0])
    x = _normalize_positive_rows(seed_joint, eps=eps).reshape((n,) + tuple(shape))
    targets = [_normalize_positive_rows(t, eps=eps) for t in target_marginals]
    for axis, target in enumerate(targets):
        if target.shape != (n, int(shape[axis])):
            raise ValueError(f"target marginal shape mismatch at axis={axis}: {target.shape}")

    d = len(shape)
    for _ in range(int(max_iter)):
        for axis, target in enumerate(targets):
            sum_axes = tuple(j + 1 for j in range(d) if j != axis)
            current = x.sum(axis=sum_axes)
            factor = target / np.clip(current, eps, None)
            reshape = [n] + [1] * d
            reshape[axis + 1] = int(shape[axis])
            x *= factor.reshape(reshape)
        flat = x.reshape(n, -1)
        flat = _normalize_positive_rows(flat, eps=eps)
        x = flat.reshape((n,) + tuple(shape))
    return x.reshape(n, -1)


def _marginals_from_rows(p_joint: np.ndarray, *, shape: tuple[int, ...]) -> list[np.ndarray]:
    n = int(p_joint.shape[0])
    tab = np.asarray(p_joint, dtype=np.float64).reshape((n,) + tuple(shape))
    out: list[np.ndarray] = []
    for axis in range(len(shape)):
        sum_axes = tuple(j + 1 for j in range(len(shape)) if j != axis)
        out.append(_normalize_positive_rows(tab.sum(axis=sum_axes)))
    return out


def _fit_residual_pca(
    log_ratio: np.ndarray,
    train_mask: np.ndarray,
    *,
    max_pcs: int,
    seed: int,
) -> tuple[np.ndarray, PCA, StandardScaler, np.ndarray]:
    n_components = max(1, min(int(max_pcs), int(train_mask.sum()) - 1, int(log_ratio.shape[1])))
    scaler = StandardScaler(with_mean=True, with_std=False)
    z_train = scaler.fit_transform(log_ratio[train_mask])
    z_all = scaler.transform(log_ratio)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(seed))
    scores = pca.fit_transform(z_train)
    all_scores = pca.transform(z_all)
    return all_scores, pca, scaler, np.asarray(pca.explained_variance_ratio_, dtype=np.float64)


def _standardize_by_train(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    out = np.empty_like(np.asarray(x, dtype=np.float64), dtype=np.float64)
    out[train_mask] = scaler.fit_transform(x[train_mask])
    out[~train_mask] = scaler.transform(x[~train_mask])
    return out


def _ridge_predict_scores(
    x: np.ndarray,
    scores: np.ndarray,
    train_mask: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    xz = _standardize_by_train(x, train_mask)
    model = Ridge(alpha=float(alpha))
    model.fit(xz[train_mask], scores[train_mask])
    return np.asarray(model.predict(xz), dtype=np.float64)


def _residualize_external(
    x_context: np.ndarray,
    x_external: np.ndarray,
    train_mask: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    context_z = _standardize_by_train(x_context, train_mask)
    external_z = _standardize_by_train(x_external, train_mask)
    model = Ridge(alpha=float(alpha))
    model.fit(context_z[train_mask], external_z[train_mask])
    pred = model.predict(context_z)
    residual = external_z - pred
    diag = {
        "acs_to_external_train_r2": float(r2_score(external_z[train_mask], pred[train_mask], multioutput="variance_weighted")),
        "acs_to_external_test_r2": float(r2_score(external_z[~train_mask], pred[~train_mask], multioutput="variance_weighted")),
    }
    return residual, diag


def _fit_feature_predictions(
    *,
    x_context: np.ndarray,
    x_external: np.ndarray,
    scores: np.ndarray,
    train_mask: np.ndarray,
    alpha: float,
    align_dims: list[int],
    label: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    pred_scores: dict[str, np.ndarray] = {
        "acs_ridge": _ridge_predict_scores(x_context, scores, train_mask, alpha=alpha),
        f"{label}_only_ridge": _ridge_predict_scores(x_external, scores, train_mask, alpha=alpha),
        f"acs_{label}_raw_ridge": _ridge_predict_scores(np.concatenate([x_context, x_external], axis=1), scores, train_mask, alpha=alpha),
    }
    feature_spaces: dict[str, np.ndarray] = {
        "acs_ridge": x_context,
        f"{label}_only_ridge": x_external,
        f"acs_{label}_raw_ridge": np.concatenate([x_context, x_external], axis=1),
    }
    residual_ext, diag = _residualize_external(x_context, x_external, train_mask, alpha=alpha)

    max_dim = max(1, min(int(train_mask.sum()) - 1, residual_ext.shape[1], scores.shape[1]))
    rows: list[dict[str, Any]] = []
    for dim in align_dims:
        d = max(1, min(int(dim), max_dim))
        pls = PLSRegression(n_components=d, scale=True)
        pls.fit(residual_ext[train_mask], scores[train_mask])
        h = np.asarray(pls.transform(residual_ext), dtype=np.float64)
        x_aligned = np.concatenate([x_context, h], axis=1)
        model_name = f"acs_{label}_pls{d}_ridge"
        pred_scores[model_name] = _ridge_predict_scores(x_aligned, scores, train_mask, alpha=alpha)
        feature_spaces[model_name] = x_aligned
        yhat_pls = np.asarray(pls.predict(residual_ext), dtype=np.float64)
        rows.append(
            {
                "model": model_name,
                "pls_components": int(d),
                "pls_train_r2": float(r2_score(scores[train_mask], yhat_pls[train_mask], multioutput="variance_weighted")),
                "pls_test_r2": float(r2_score(scores[~train_mask], yhat_pls[~train_mask], multioutput="variance_weighted")),
                **diag,
            }
        )
    return pred_scores, feature_spaces, rows


def _scores_to_distribution(
    *,
    p_eq: np.ndarray,
    scores_pred: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,
    top_pcs: int,
    clip_log_ratio: float,
) -> np.ndarray:
    n_pcs = int(pca.components_.shape[0])
    scores_use = np.zeros((scores_pred.shape[0], n_pcs), dtype=np.float64)
    n_use = max(0, min(int(top_pcs), n_pcs, scores_pred.shape[1]))
    if n_use > 0:
        scores_use[:, :n_use] = scores_pred[:, :n_use]
    centered = scores_use @ np.asarray(pca.components_, dtype=np.float64)
    log_ratio = centered + np.asarray(scaler.mean_, dtype=np.float64).reshape(1, -1)
    log_ratio = np.clip(log_ratio, -float(clip_log_ratio), float(clip_log_ratio))
    q = np.asarray(p_eq, dtype=np.float64) * np.exp(log_ratio)
    return _normalize_positive_rows(q)


def _nearest_train_distance(x: np.ndarray, train_mask: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    xz = _standardize_by_train(x, train_mask)
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(xz[train_mask])
    dist, _ = nn.kneighbors(xz[test_idx])
    return dist[:, 0].astype(np.float64)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.quantile(x, 0.5)),
        "p10": float(np.quantile(x, 0.1)),
        "p90": float(np.quantile(x, 0.9)),
    }


def _confidence_gate_summary(by_puma: pd.DataFrame, *, external_label: str) -> pd.DataFrame:
    baseline = by_puma[by_puma["model"] == "acs_ridge"][
        ["heldout_statefp", "puma_uid_key", "top_pcs", "tvd_post"]
    ].rename(columns={"tvd_post": "acs_tvd_post"})
    ext = by_puma[by_puma["model"] != "acs_ridge"].merge(
        baseline,
        on=["heldout_statefp", "puma_uid_key", "top_pcs"],
        how="left",
    )
    ext["delta_vs_acs_post"] = ext["acs_tvd_post"] - ext["tvd_post"]

    gate_specs = [
        ("high_poi_density", "poi__log1p_count_density_per_km2", "high"),
        ("high_poi_count", "poi__log1p_total_count", "high"),
        ("high_poi_entropy", "poi__main_entropy", "high"),
        ("low_poi_top_share", "poi__main_top_share", "low"),
        ("low_feature_ood", "feature_nn_dist", "low"),
    ]
    rows: list[dict[str, Any]] = []
    for (heldout, model, top_pcs), grp in ext.groupby(["heldout_statefp", "model", "top_pcs"], sort=False):
        for gate_name, col, direction in gate_specs:
            if col not in grp.columns or grp[col].notna().sum() < 4:
                continue
            values = grp[col].astype(float).to_numpy()
            thr = float(np.nanmedian(values))
            if direction == "high":
                mask = values >= thr
            else:
                mask = values <= thr
            for label, m in [("selected", mask), ("other", ~mask)]:
                sub = grp[m]
                rows.append(
                    {
                        "heldout_statefp": heldout,
                        "model": model,
                        "top_pcs": int(top_pcs),
                        "gate": gate_name,
                        "group": label,
                        "threshold": thr,
                        **{f"delta_{k}": v for k, v in _summary(sub["delta_vs_acs_post"].to_numpy()).items()},
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Distribution-level mechanism probe for external-view representation learning. "
            "It tests whether external-view residual predictions survive ACS constraint projection."
        )
    )
    data_root = pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data")
    ap.add_argument("--target_wide_csv", type=pathlib.Path, default=data_root / "us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv")
    ap.add_argument("--condition_csv", type=pathlib.Path, default=data_root / "us/processed/external_conditions/extcond_v1_agesex_earn_v1_acs5_2022_puma_us.csv")
    ap.add_argument("--external_csv", type=pathlib.Path, required=True)
    ap.add_argument("--external_label", default="external")
    ap.add_argument("--heldout_statefps", default="26,12,48,55,06")
    ap.add_argument("--joint_space", choices=["full", "coarse"], default="full")
    ap.add_argument("--max_pcs", type=int, default=40)
    ap.add_argument("--top_pcs", default="5,10,20,40")
    ap.add_argument("--align_dims", default="4,8,16")
    ap.add_argument("--ridge_alpha", type=float, default=10.0)
    ap.add_argument("--ipf_iters", type=int, default=80)
    ap.add_argument("--clip_log_ratio", type=float, default=8.0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    label = _safe_label(args.external_label)
    out_dir = args.output_dir or pathlib.Path(f"outputs/_external_view_distribution_mechanism_{label}_{args.joint_space}_{_utc_ts()}")
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    target_keys, p_true_full, _, _ = _load_target(args.target_wide_csv)
    _, acs_marginals_full, _, x_acs_all, x_scale, *_ = _load_acs_conditions(args.condition_csv, target_keys)
    x_context = np.concatenate([x_acs_all, x_scale], axis=1)
    x_external, external_cols = _load_spatial(args.external_csv, target_keys)

    p_eq_acs_full = _outer_joint([acs_marginals_full[v] for v in FULL_VARIABLE_ORDER])
    if args.joint_space == "full":
        shape = tuple(FULL_SHAPE)
        p_true = p_true_full
        p_eq_acs = p_eq_acs_full
        target_marginals_all = [acs_marginals_full[v] for v in FULL_VARIABLE_ORDER]
    else:
        shape = tuple(COARSE_SHAPE)
        p_true = _aggregate_full_to_coarse(p_true_full)
        p_eq_acs = _aggregate_full_to_coarse(p_eq_acs_full)
        target_marginals_all = _marginals_from_rows(p_eq_acs, shape=shape)

    _, _, log_ratio = _residual_arrays(p_true, p_eq_acs, eps=float(args.eps))
    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()
    heldouts = _canon_statefps(args.heldout_statefps)
    top_pcs_list = [int(x) for x in _parse_csv_ints(args.top_pcs)]
    align_dims = [int(x) for x in _parse_csv_ints(args.align_dims)]

    poi_cols = [
        "poi__log1p_total_count",
        "poi__log1p_count_density_per_km2",
        "poi__main_entropy",
        "poi__main_top_share",
    ]
    poi_meta = _add_puma_uid(pd.read_csv(args.external_csv, low_memory=False))
    poi_meta = poi_meta.drop_duplicates("puma_uid_key").set_index("puma_uid_key")
    poi_meta = poi_meta.reindex(target_keys["puma_uid_key"])[[c for c in poi_cols if c in poi_meta.columns]].reset_index(drop=True)

    by_puma_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []

    for heldout in heldouts:
        test_mask = statefp == heldout
        if not np.any(test_mask):
            continue
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        scores, pca, scaler, evr = _fit_residual_pca(log_ratio, train_mask, max_pcs=args.max_pcs, seed=args.seed)
        pred_scores, feature_spaces, align_diag = _fit_feature_predictions(
            x_context=x_context,
            x_external=x_external,
            scores=scores,
            train_mask=train_mask,
            alpha=float(args.ridge_alpha),
            align_dims=align_dims,
            label=label,
        )
        for row in align_diag:
            alignment_rows.append({"heldout_statefp": heldout, **row})

        oracle_scores = scores.copy()
        model_scores = {"acs_ridge": pred_scores["acs_ridge"], **pred_scores, "oracle_residual_pc": oracle_scores}
        model_feature_spaces = {"acs_ridge": feature_spaces["acs_ridge"], **feature_spaces}

        p_eq_test = p_eq_acs[test_idx]
        p_true_test = p_true[test_idx]
        target_marginals_test = [m[test_idx] for m in target_marginals_all]
        maxent_tvd = _row_tvd(p_eq_test, p_true_test)

        acs_by_top: dict[int, dict[str, np.ndarray]] = {}
        for top_pcs in top_pcs_list:
            acs_pre = _scores_to_distribution(
                p_eq=p_eq_test,
                scores_pred=model_scores["acs_ridge"][test_idx],
                pca=pca,
                scaler=scaler,
                top_pcs=top_pcs,
                clip_log_ratio=float(args.clip_log_ratio),
            )
            acs_post = _ipf_project_rows(
                acs_pre,
                target_marginals_test,
                shape=shape,
                max_iter=int(args.ipf_iters),
                eps=float(args.eps),
            )
            acs_by_top[int(top_pcs)] = {"pre": acs_pre, "post": acs_post}

        for model, score_pred in model_scores.items():
            feature_nn = None
            if model in model_feature_spaces:
                feature_nn = _nearest_train_distance(model_feature_spaces[model], train_mask, test_idx)
            for top_pcs in top_pcs_list:
                if model == "maxent":
                    continue
                q_pre = _scores_to_distribution(
                    p_eq=p_eq_test,
                    scores_pred=score_pred[test_idx],
                    pca=pca,
                    scaler=scaler,
                    top_pcs=top_pcs,
                    clip_log_ratio=float(args.clip_log_ratio),
                )
                q_post = _ipf_project_rows(
                    q_pre,
                    target_marginals_test,
                    shape=shape,
                    max_iter=int(args.ipf_iters),
                    eps=float(args.eps),
                )
                tvd_pre = _row_tvd(q_pre, p_true_test)
                tvd_post = _row_tvd(q_post, p_true_test)
                summary_rows.append(
                    {
                        "heldout_statefp": heldout,
                        "model": model,
                        "top_pcs": int(top_pcs),
                        "n_test": int(test_idx.size),
                        "maxent_tvd_mean": float(np.mean(maxent_tvd)),
                        "tvd_pre_mean": float(np.mean(tvd_pre)),
                        "tvd_post_mean": float(np.mean(tvd_post)),
                        "delta_post_vs_maxent_mean": float(np.mean(maxent_tvd - tvd_post)),
                        "pca_explained_used": float(np.sum(evr[: min(int(top_pcs), len(evr))])),
                    }
                )
                if model != "acs_ridge":
                    acs_pre = acs_by_top[int(top_pcs)]["pre"]
                    acs_post = acs_by_top[int(top_pcs)]["post"]
                    delta_pre = q_pre - acs_pre
                    delta_post = q_post - acs_post
                    survival = _row_l1(delta_post) / np.clip(_row_l1(delta_pre), float(args.eps), None)
                    projection_rows.append(
                        {
                            "heldout_statefp": heldout,
                            "model": model,
                            "top_pcs": int(top_pcs),
                            "n_test": int(test_idx.size),
                            "delta_pre_l1_mean": float(np.mean(_row_l1(delta_pre))),
                            "delta_post_l1_mean": float(np.mean(_row_l1(delta_post))),
                            "projection_survival_mean": float(np.mean(survival)),
                            "projection_cosine_mean": float(np.mean(_row_cosine(delta_pre, delta_post))),
                            "delta_post_vs_acs_mean": float(np.mean(_row_tvd(acs_post, p_true_test) - tvd_post)),
                        }
                    )

                for pos, idx in enumerate(test_idx):
                    row: dict[str, Any] = {
                        "heldout_statefp": heldout,
                        "statefp": str(target_keys.iloc[idx]["statefp"]).zfill(2),
                        "puma5": str(target_keys.iloc[idx]["puma5"]).zfill(5),
                        "puma_uid_key": str(target_keys.iloc[idx]["puma_uid_key"]),
                        "model": model,
                        "top_pcs": int(top_pcs),
                        "maxent_tvd": float(maxent_tvd[pos]),
                        "tvd_pre": float(tvd_pre[pos]),
                        "tvd_post": float(tvd_post[pos]),
                        "feature_nn_dist": float(feature_nn[pos]) if feature_nn is not None else float("nan"),
                    }
                    for col in poi_meta.columns:
                        row[col] = float(poi_meta.iloc[idx][col]) if pd.notna(poi_meta.iloc[idx][col]) else float("nan")
                    by_puma_rows.append(row)

        # Add a direct max-ent row for reference without duplicating per-PC projection work.
        summary_rows.append(
            {
                "heldout_statefp": heldout,
                "model": "maxent_acs_marginals",
                "top_pcs": 0,
                "n_test": int(test_idx.size),
                "maxent_tvd_mean": float(np.mean(maxent_tvd)),
                "tvd_pre_mean": float(np.mean(maxent_tvd)),
                "tvd_post_mean": float(np.mean(maxent_tvd)),
                "delta_post_vs_maxent_mean": 0.0,
                "pca_explained_used": 0.0,
            }
        )

    summary = pd.DataFrame(summary_rows)
    by_puma = pd.DataFrame(by_puma_rows)
    projection = pd.DataFrame(projection_rows)
    alignment = pd.DataFrame(alignment_rows)
    confidence = _confidence_gate_summary(by_puma, external_label=label) if not by_puma.empty else pd.DataFrame()

    summary.to_csv(metrics_dir / "distribution_correction_summary.csv", index=False)
    by_puma.to_csv(metrics_dir / "distribution_correction_by_puma.csv", index=False)
    projection.to_csv(metrics_dir / "projection_survival_summary.csv", index=False)
    alignment.to_csv(metrics_dir / "alignment_diagnostics.csv", index=False)
    confidence.to_csv(metrics_dir / "confidence_gate_summary.csv", index=False)

    run_summary = {
        "output_dir": str(out_dir),
        "joint_space": args.joint_space,
        "heldout_statefps": heldouts,
        "external_label": label,
        "external_csv": str(args.external_csv),
        "external_feature_count": int(len(external_cols)),
        "max_pcs": int(args.max_pcs),
        "top_pcs": top_pcs_list,
        "align_dims": align_dims,
        "ridge_alpha": float(args.ridge_alpha),
        "ipf_iters": int(args.ipf_iters),
        "clip_log_ratio": float(args.clip_log_ratio),
        "summary_csv": str(metrics_dir / "distribution_correction_summary.csv"),
        "projection_csv": str(metrics_dir / "projection_survival_summary.csv"),
        "confidence_csv": str(metrics_dir / "confidence_gate_summary.csv"),
    }
    _write_json(out_dir / "run_summary.json", run_summary)

    if not summary.empty:
        view = (
            summary.groupby(["model", "top_pcs"], as_index=False)["tvd_post_mean"]
            .mean()
            .sort_values(["top_pcs", "tvd_post_mean"])
            .head(40)
        )
        print(view.to_string(index=False))
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
