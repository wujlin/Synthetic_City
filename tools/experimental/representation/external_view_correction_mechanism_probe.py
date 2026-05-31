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

from tools.experimental.representation.ssl_copula_residual_probe import (  # noqa: E402
    FULL_VARIABLE_ORDER,
    _load_acs_conditions,
    _load_spatial,
    _load_target,
    _outer_joint,
    _parse_csv_ints,
    _residual_arrays,
    _write_json,
)
from tools.model.train_us_puma_5var_diffusion import _ipf_nd  # noqa: E402
from tools.model.external_c2f_full_earn_schema import FULL_SHAPE  # noqa: E402


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_label(text: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "external"


def _tvd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(np.abs(np.asarray(a) - np.asarray(b)), axis=1)


def _l1_rows(a: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.asarray(a)), axis=1)


def _make_eval_splits(statefp: np.ndarray, heldouts: list[str]) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for h in heldouts:
        mask = statefp != h
        if int((~mask).sum()) > 0:
            out.append((h, mask))
    return out


def _fit_pc_model(log_ratio: np.ndarray, train_mask: np.ndarray, n_components: int, seed: int) -> tuple[PCA, StandardScaler, np.ndarray, np.ndarray]:
    scaler = StandardScaler(with_mean=True, with_std=False)
    z_train = scaler.fit_transform(log_ratio[train_mask])
    z_all = scaler.transform(log_ratio)
    n_components = max(1, min(int(n_components), int(train_mask.sum()) - 1, z_all.shape[1]))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(seed))
    scores = pca.fit_transform(z_train)
    scores_all = pca.transform(z_all)
    return pca, scaler, scores_all, np.asarray(pca.explained_variance_ratio_, dtype=np.float64)


def _standardize_train(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    out = np.empty_like(np.asarray(x, dtype=np.float64), dtype=np.float64)
    out[train_mask] = scaler.fit_transform(x[train_mask])
    out[~train_mask] = scaler.transform(x[~train_mask])
    return out


def _residualize_external(x_context: np.ndarray, x_external: np.ndarray, train_mask: np.ndarray, alpha: float) -> tuple[np.ndarray, dict[str, float]]:
    x_context_z = _standardize_train(x_context, train_mask)
    x_external_z = _standardize_train(x_external, train_mask)
    model = Ridge(alpha=float(alpha))
    model.fit(x_context_z[train_mask], x_external_z[train_mask])
    pred = model.predict(x_context_z)
    resid = x_external_z - pred
    return resid, {
        "acs_to_external_train_r2": float(r2_score(x_external_z[train_mask], pred[train_mask], multioutput="variance_weighted")),
        "acs_to_external_test_r2": float(r2_score(x_external_z[~train_mask], pred[~train_mask], multioutput="variance_weighted")),
    }


def _fit_feature_sets(
    x_context: np.ndarray,
    x_external: np.ndarray,
    scores: np.ndarray,
    train_mask: np.ndarray,
    *,
    pls_dim: int,
    alpha: float,
    label: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ext_resid, diag = _residualize_external(x_context, x_external, train_mask, alpha)
    d = max(1, min(int(pls_dim), ext_resid.shape[1], scores.shape[1], int(train_mask.sum()) - 1))
    pls = PLSRegression(n_components=d, scale=True)
    pls.fit(ext_resid[train_mask], scores[train_mask])
    h = pls.transform(ext_resid)
    yhat = pls.predict(ext_resid)
    return {
        "acs": x_context,
        f"acs_{label}_raw": np.concatenate([x_context, x_external], axis=1),
        f"acs_{label}_pls{d}_latent": np.concatenate([x_context, h], axis=1),
        f"acs_{label}_pls{d}_yhat": np.concatenate([x_context, yhat], axis=1),
    }, {**diag, "pls_dim": int(d)}


def _predict_scores(feature_sets: dict[str, np.ndarray], scores: np.ndarray, train_mask: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, x in feature_sets.items():
        xz = _standardize_train(x, train_mask)
        model = Ridge(alpha=float(alpha))
        model.fit(xz[train_mask], scores[train_mask])
        out[name] = model.predict(xz)
    return out


def _scores_to_distribution(
    *,
    pred_scores: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,
    reference: np.ndarray,
    keep_pcs: int,
    eps: float,
) -> np.ndarray:
    s = np.zeros_like(pred_scores)
    k = max(1, min(int(keep_pcs), s.shape[1]))
    s[:, :k] = pred_scores[:, :k]
    centered = pca.inverse_transform(s)
    log_ratio = scaler.inverse_transform(centered)
    logits = np.log(np.clip(reference, eps, None)) + log_ratio
    logits = logits - np.max(logits, axis=1, keepdims=True)
    q = np.exp(logits)
    q = q / np.clip(q.sum(axis=1, keepdims=True), eps, None)
    return q


def _project_rows(q: np.ndarray, targets: list[np.ndarray], max_iter: int) -> np.ndarray:
    rows = []
    for i in range(q.shape[0]):
        t = [m[i] for m in targets]
        rows.append(_ipf_nd(seed_joint=q[i], target_marginals=t, shape=tuple(FULL_SHAPE), max_iter=int(max_iter)))
    return np.asarray(rows, dtype=np.float64)


def _ood_distance(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    xz = _standardize_train(x, train_mask)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(xz[train_mask])
    dist = np.full(x.shape[0], np.nan, dtype=np.float64)
    d, _ = nn.kneighbors(xz[~train_mask])
    dist[~train_mask] = d[:, 0]
    return dist


def main() -> int:
    ap = argparse.ArgumentParser(description="Mechanism diagnostics for external-view residual correction and projection survival.")
    data_root = pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data")
    ap.add_argument("--target_wide_csv", type=pathlib.Path, default=data_root / "us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv")
    ap.add_argument("--condition_csv", type=pathlib.Path, default=data_root / "us/processed/external_conditions/extcond_v1_agesex_earn_v1_acs5_2022_puma_us.csv")
    ap.add_argument("--external_csv", type=pathlib.Path, required=True)
    ap.add_argument("--external_label", default="external")
    ap.add_argument("--heldout_statefps", default="26,12,48,55,06")
    ap.add_argument("--max_pcs", type=int, default=40)
    ap.add_argument("--keep_pcs", default="5,10,20,40")
    ap.add_argument("--pls_dim", type=int, default=8)
    ap.add_argument("--ridge_alpha", type=float, default=10.0)
    ap.add_argument("--ipf_iters", type=int, default=80)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    label = _safe_label(str(args.external_label))
    out_dir = args.output_dir or pathlib.Path(f"outputs/_external_view_correction_mechanism_{label}_{_utc_ts()}")
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    target_keys, p_true, _p_eq_target, _target_marginals = _load_target(args.target_wide_csv)
    _, acs_marginals, _x_acs_1d, x_acs_all, x_scale, *_ = _load_acs_conditions(args.condition_csv, target_keys)
    x_context = np.concatenate([x_acs_all, x_scale], axis=1)
    x_external = _load_spatial(args.external_csv, target_keys)[0]
    p_eq_acs = _outer_joint([acs_marginals[v] for v in FULL_VARIABLE_ORDER])
    _, _, log_ratio = _residual_arrays(p_true, p_eq_acs, eps=float(args.eps))
    targets = [acs_marginals[v] for v in FULL_VARIABLE_ORDER]

    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()
    heldouts = [str(x).zfill(2) for x in _parse_csv_ints(args.heldout_statefps)]
    keep_pcs = [int(x) for x in _parse_csv_ints(args.keep_pcs)]
    rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    survival_rows: list[dict[str, Any]] = []

    for heldout, train_mask in _make_eval_splits(statefp, heldouts):
        test_mask = ~train_mask
        pca, scaler, scores, explained = _fit_pc_model(log_ratio, train_mask, int(args.max_pcs), int(args.seed))
        feature_sets, align_diag = _fit_feature_sets(
            x_context,
            x_external,
            scores,
            train_mask,
            pls_dim=int(args.pls_dim),
            alpha=float(args.ridge_alpha),
            label=label,
        )
        pred_scores = _predict_scores(feature_sets, scores, train_mask, float(args.ridge_alpha))
        ood = _ood_distance(feature_sets[f"acs_{label}_pls{align_diag['pls_dim']}_latent"], train_mask)

        for k in keep_pcs:
            q_pre: dict[str, np.ndarray] = {}
            q_post: dict[str, np.ndarray] = {}
            for name, pred in pred_scores.items():
                q = _scores_to_distribution(
                    pred_scores=pred,
                    pca=pca,
                    scaler=scaler,
                    reference=p_eq_acs,
                    keep_pcs=int(k),
                    eps=float(args.eps),
                )
                q_pre[name] = q
                q_post[name] = _project_rows(q[test_mask], [m[test_mask] for m in targets], max_iter=int(args.ipf_iters))
                tvd_pre = _tvd(q[test_mask], p_true[test_mask])
                tvd_post = _tvd(q_post[name], p_true[test_mask])
                rows.append(
                    {
                        "heldout_statefp": heldout,
                        "keep_pcs": int(k),
                        "feature_set": name,
                        "n_features": int(feature_sets[name].shape[1]),
                        "tvd_pre_mean": float(np.mean(tvd_pre)),
                        "tvd_post_mean": float(np.mean(tvd_post)),
                        "tvd_pre_median": float(np.median(tvd_pre)),
                        "tvd_post_median": float(np.median(tvd_post)),
                        "pc_explained_kept": float(np.sum(explained[: min(int(k), explained.shape[0])])),
                        **align_diag,
                    }
                )
                for local_idx, global_idx in enumerate(np.where(test_mask)[0]):
                    region_rows.append(
                        {
                            "heldout_statefp": heldout,
                            "puma_uid_key": str(target_keys.iloc[global_idx]["puma_uid_key"]).zfill(7),
                            "keep_pcs": int(k),
                            "feature_set": name,
                            "tvd_pre": float(tvd_pre[local_idx]),
                            "tvd_post": float(tvd_post[local_idx]),
                            "aligned_ood_distance": float(ood[global_idx]),
                        }
                    )

            base = "acs"
            for name in sorted(q_pre):
                if name == base:
                    continue
                pre_delta = q_pre[name][test_mask] - q_pre[base][test_mask]
                post_delta = q_post[name] - q_post[base]
                pre_l1 = _l1_rows(pre_delta)
                post_l1 = _l1_rows(post_delta)
                survival = post_l1 / np.clip(pre_l1, 1e-12, None)
                base_post_tvd = _tvd(q_post[base], p_true[test_mask])
                ext_post_tvd = _tvd(q_post[name], p_true[test_mask])
                improvement = base_post_tvd - ext_post_tvd
                survival_rows.append(
                    {
                        "heldout_statefp": heldout,
                        "keep_pcs": int(k),
                        "feature_set": name,
                        "mean_pre_delta_l1": float(np.mean(pre_l1)),
                        "mean_post_delta_l1": float(np.mean(post_l1)),
                        "mean_projection_survival": float(np.mean(survival)),
                        "median_projection_survival": float(np.median(survival)),
                        "mean_post_tvd_improvement_vs_acs": float(np.mean(improvement)),
                        "median_post_tvd_improvement_vs_acs": float(np.median(improvement)),
                        "share_regions_improved_vs_acs": float(np.mean(improvement > 0)),
                    }
                )

    summary = pd.DataFrame(rows)
    by_region = pd.DataFrame(region_rows)
    survival = pd.DataFrame(survival_rows)
    summary.to_csv(metrics_dir / "mode_specific_tvd_summary.csv", index=False)
    by_region.to_csv(metrics_dir / "mode_specific_tvd_by_puma.csv", index=False)
    survival.to_csv(metrics_dir / "projection_survival_summary.csv", index=False)

    if not summary.empty:
        mean_summary = (
            summary.groupby(["keep_pcs", "feature_set"], as_index=False)
            .agg(tvd_post_mean=("tvd_post_mean", "mean"), tvd_pre_mean=("tvd_pre_mean", "mean"), n_features=("n_features", "first"))
            .sort_values(["keep_pcs", "tvd_post_mean"])
        )
        mean_summary.to_csv(metrics_dir / "mode_specific_tvd_mean_summary.csv", index=False)
        print("[tvd mean]")
        print(mean_summary.to_string(index=False))
    if not survival.empty:
        mean_survival = (
            survival.groupby(["keep_pcs", "feature_set"], as_index=False)
            .agg(
                mean_projection_survival=("mean_projection_survival", "mean"),
                mean_post_tvd_improvement_vs_acs=("mean_post_tvd_improvement_vs_acs", "mean"),
                share_regions_improved_vs_acs=("share_regions_improved_vs_acs", "mean"),
            )
            .sort_values(["keep_pcs", "mean_post_tvd_improvement_vs_acs"], ascending=[True, False])
        )
        mean_survival.to_csv(metrics_dir / "projection_survival_mean_summary.csv", index=False)
        print("[projection survival]")
        print(mean_survival.to_string(index=False))

    if not by_region.empty:
        acs = by_region[by_region["feature_set"] == "acs"].rename(columns={"tvd_post": "acs_tvd_post"})[
            ["heldout_statefp", "puma_uid_key", "keep_pcs", "acs_tvd_post", "aligned_ood_distance"]
        ]
        merged = by_region.merge(acs, on=["heldout_statefp", "puma_uid_key", "keep_pcs"], suffixes=("", "_acs"))
        merged = merged[merged["feature_set"] != "acs"].copy()
        merged["post_tvd_improvement_vs_acs"] = merged["acs_tvd_post"] - merged["tvd_post"]
        conf_rows = []
        for (k, feature), sub in merged.groupby(["keep_pcs", "feature_set"], sort=False):
            q = sub["aligned_ood_distance"].quantile([0.25, 0.5, 0.75]).to_dict()
            for label_q, mask in [
                ("low_ood_q25", sub["aligned_ood_distance"] <= q[0.25]),
                ("low_ood_q50", sub["aligned_ood_distance"] <= q[0.5]),
                ("high_ood_q50", sub["aligned_ood_distance"] > q[0.5]),
                ("high_ood_q75", sub["aligned_ood_distance"] > q[0.75]),
            ]:
                part = sub[mask]
                if part.empty:
                    continue
                conf_rows.append(
                    {
                        "keep_pcs": int(k),
                        "feature_set": feature,
                        "confidence_subset": label_q,
                        "n_regions": int(part.shape[0]),
                        "mean_post_tvd_improvement_vs_acs": float(part["post_tvd_improvement_vs_acs"].mean()),
                        "share_regions_improved_vs_acs": float((part["post_tvd_improvement_vs_acs"] > 0).mean()),
                    }
                )
        conf = pd.DataFrame(conf_rows)
        conf.to_csv(metrics_dir / "confidence_gated_summary.csv", index=False)

    _write_json(
        out_dir / "run_summary.json",
        {
            "target_wide_csv": str(args.target_wide_csv),
            "condition_csv": str(args.condition_csv),
            "external_csv": str(args.external_csv),
            "external_label": label,
            "heldout_statefps": heldouts,
            "max_pcs": int(args.max_pcs),
            "keep_pcs": keep_pcs,
            "pls_dim": int(args.pls_dim),
            "ridge_alpha": float(args.ridge_alpha),
            "ipf_iters": int(args.ipf_iters),
            "metrics": {
                "mode_specific_tvd_summary": str(metrics_dir / "mode_specific_tvd_summary.csv"),
                "mode_specific_tvd_mean_summary": str(metrics_dir / "mode_specific_tvd_mean_summary.csv"),
                "projection_survival_summary": str(metrics_dir / "projection_survival_summary.csv"),
                "projection_survival_mean_summary": str(metrics_dir / "projection_survival_mean_summary.csv"),
                "confidence_gated_summary": str(metrics_dir / "confidence_gated_summary.csv"),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
