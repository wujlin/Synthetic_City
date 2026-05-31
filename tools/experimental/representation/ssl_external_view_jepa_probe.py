#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline

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
    _ridge_probe,
    _write_json,
)


def _ridge_probe_multioutput(
    x: np.ndarray,
    scores: np.ndarray,
    train_mask: np.ndarray,
    explained_ratio: np.ndarray,
    alphas: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.asarray(alphas, dtype=float)))
    model.fit(x[train_mask], scores[train_mask])
    pred_train = model.predict(x[train_mask])
    pred_test = model.predict(x[~train_mask])
    rows: list[dict[str, Any]] = []
    weighted_nonneg_r2 = 0.0
    weighted_raw_r2 = 0.0
    positive_pc_share = 0.0
    top5_r2: list[float] = []
    ridge_step = model.named_steps["ridgecv"]
    alpha = getattr(ridge_step, "alpha_", float("nan"))
    if np.ndim(alpha) > 0:
        alpha = float(np.asarray(alpha).ravel()[0])
    for pc_idx in range(scores.shape[1]):
        y_train = scores[train_mask, pc_idx]
        y_test = scores[~train_mask, pc_idx]
        r2_train = float(r2_score(y_train, pred_train[:, pc_idx]))
        r2_test = float(r2_score(y_test, pred_test[:, pc_idx]))
        evr = float(explained_ratio[pc_idx])
        weighted_nonneg_r2 += max(0.0, r2_test) * evr
        weighted_raw_r2 += r2_test * evr
        if r2_test > 0.0:
            positive_pc_share += evr
        if pc_idx < 5:
            top5_r2.append(r2_test)
        rows.append(
            {
                "pc_index": int(pc_idx + 1),
                "explained_variance_ratio": evr,
                "ridge_alpha": float(alpha),
                "r2_train": r2_train,
                "r2_test": r2_test,
            }
        )
    summary = {
        "n_pcs": int(scores.shape[1]),
        "weighted_nonnegative_r2": float(weighted_nonneg_r2),
        "weighted_raw_r2": float(weighted_raw_r2),
        "positive_pc_explained_share": float(positive_pc_share),
        "mean_r2_top5": float(np.mean(top5_r2)) if top5_r2 else float("nan"),
        "median_r2_all": float(np.median([r["r2_test"] for r in rows])) if rows else float("nan"),
    }
    return summary, rows


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_label(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(text).strip()) or "view"


def _effective_rank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return float("nan")
    s = np.linalg.svd(x - x.mean(axis=0, keepdims=True), compute_uv=False)
    if not np.any(s > 0):
        return 0.0
    p = s / np.clip(s.sum(), 1e-12, None)
    return float(np.exp(-np.sum(p * np.log(np.clip(p, 1e-12, None)))))


def _embedding_diag(emb: np.ndarray, train_mask: np.ndarray) -> dict[str, float]:
    z = np.asarray(emb[train_mask], dtype=np.float64)
    std = z.std(axis=0)
    return {
        "embedding_std_mean": float(np.mean(std)),
        "embedding_std_min": float(np.min(std)),
        "embedding_effective_rank": _effective_rank(z),
    }


def _load_wide_numeric(path: pathlib.Path, target_keys: pd.DataFrame, prefix: str) -> tuple[np.ndarray, list[str]]:
    df = _load_spatial(path, target_keys)[0]
    # _load_spatial already returns only numeric non-key columns. Re-load names for audit.
    raw = pd.read_csv(path, nrows=5)
    numeric_cols = [
        c for c in raw.columns
        if c not in {"statefp", "puma", "puma5", "puma_uid", "puma_uid_key"}
        and (prefix == "" or c.startswith(prefix))
    ]
    if prefix:
        full = pd.read_csv(path, low_memory=False)
        from tools.experimental.representation.ssl_copula_residual_probe import _add_puma_uid

        full = _add_puma_uid(full).drop_duplicates("puma_uid_key").set_index("puma_uid_key")
        aligned = full.reindex(index=target_keys["puma_uid_key"])[numeric_cols]
        aligned = aligned.replace([np.inf, -np.inf], np.nan)
        med = aligned.median(numeric_only=True)
        aligned = aligned.fillna(med).fillna(0.0)
        return aligned.to_numpy(dtype=np.float64), numeric_cols
    return df, numeric_cols


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


def _train_external_jepa(
    x_context: np.ndarray,
    y_target: np.ndarray,
    train_mask: np.ndarray,
    *,
    seeds: list[int],
    epochs: int,
    hidden_dim: int,
    embed_dim: int,
    predictor_hidden_dim: int,
    context_mask_rate: float,
    ema_tau: float,
    lr: float,
    batch_size: int,
    var_weight: float,
    cov_weight: float,
    device: str,
    label: str,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Encoder(nn.Module):
        def __init__(self, d_in: int, hidden: int, embed: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, embed),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class Predictor(nn.Module):
        def __init__(self, embed: int, hidden: int, d_out: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embed, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_out),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z)

    def make_context(xb: torch.Tensor) -> torch.Tensor:
        mask = (torch.rand_like(xb) < float(context_mask_rate)).float()
        return xb * (1.0 - mask)

    def variance_loss(z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1.0 - std))

    def covariance_loss(z: torch.Tensor) -> torch.Tensor:
        if z.shape[0] < 2:
            return torch.zeros((), dtype=z.dtype, device=z.device)
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / max(z.shape[0] - 1, 1)
        off = cov[~torch.eye(cov.shape[0], dtype=torch.bool, device=z.device)]
        return torch.mean(off.pow(2))

    @torch.no_grad()
    def update_ema(src: nn.Module, dst: nn.Module, tau: float) -> None:
        for ps, pdst in zip(src.parameters(), dst.parameters(), strict=True):
            pdst.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_context[train_mask]).astype(np.float32)
    x_all = x_scaler.transform(x_context).astype(np.float32)
    y_train = y_scaler.fit_transform(y_target[train_mask]).astype(np.float32)
    y_all_scaled = y_scaler.transform(y_target).astype(np.float32)

    dev = torch.device(device)
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=dev)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=dev)
    x_all_t = torch.tensor(x_all, dtype=torch.float32, device=dev)
    n_train = int(x_train.shape[0])
    batch_size = min(int(batch_size), n_train)
    label = _safe_label(label)

    out: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        online = Encoder(x_train.shape[1], int(hidden_dim), int(embed_dim)).to(dev)
        target = Encoder(x_train.shape[1], int(hidden_dim), int(embed_dim)).to(dev)
        target.load_state_dict(online.state_dict())
        for p in target.parameters():
            p.requires_grad_(False)
        predictor = Predictor(int(embed_dim), int(predictor_hidden_dim), y_train.shape[1]).to(dev)
        opt = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()), lr=float(lr), weight_decay=1e-4)
        last_loss = last_pred = last_var = last_cov = float("nan")
        for _ in range(int(epochs)):
            order = torch.randperm(n_train, device=dev)
            for start in range(0, n_train, batch_size):
                idx = order[start : start + batch_size]
                if idx.numel() < 4:
                    continue
                z_online = online(make_context(x_train_t[idx]))
                pred = predictor(z_online)
                with torch.no_grad():
                    z_target = target(x_train_t[idx])
                pred_loss = F.mse_loss(pred, y_train_t[idx])
                var_loss = variance_loss(z_online) + variance_loss(z_target)
                cov_loss = covariance_loss(z_online) + covariance_loss(z_target)
                loss = pred_loss + float(var_weight) * var_loss + float(cov_weight) * cov_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                update_ema(online, target, float(ema_tau))
                last_loss = float(loss.detach().cpu().item())
                last_pred = float(pred_loss.detach().cpu().item())
                last_var = float(var_loss.detach().cpu().item())
                last_cov = float(cov_loss.detach().cpu().item())
        with torch.no_grad():
            h = online(x_all_t).cpu().numpy().astype(np.float64)
            pred_scaled = predictor(online(x_all_t)).cpu().numpy().astype(np.float64)
            pred_target = y_scaler.inverse_transform(pred_scaled)
        name = f"external_jepa_{label}_seed{seed}"
        out[name] = {"h": h, "pred_target": pred_target, "target_scaled": y_all_scaled}
        rows.append(
            {
                "representation": name,
                "seed": int(seed),
                "final_loss": last_loss,
                "final_pred_loss": last_pred,
                "final_var_loss": last_var,
                "final_cov_loss": last_cov,
                "target_mse_scaled_all": float(np.mean((pred_scaled - y_all_scaled) ** 2)),
                **_embedding_diag(h, train_mask),
            }
        )
    return out, rows


def main() -> int:
    ap = argparse.ArgumentParser(description="External-view JEPA probe: predict external latent from ACS context, then test residual-PC utility.")
    data_root = pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data")
    ap.add_argument("--target_wide_csv", type=pathlib.Path, default=data_root / "us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv")
    ap.add_argument("--condition_csv", type=pathlib.Path, default=data_root / "us/processed/external_conditions/extcond_v1_agesex_earn_v1_acs5_2022_puma_us.csv")
    ap.add_argument("--external_view_csv", type=pathlib.Path, required=True)
    ap.add_argument("--external_prefix", default="func__")
    ap.add_argument("--context_extra_csv", type=pathlib.Path, default=None)
    ap.add_argument("--context_extra_label", default="extra")
    ap.add_argument("--heldout_statefps", default="26,12,48,55")
    ap.add_argument("--split_mode", choices=["state_holdout", "within_state_random"], default="state_holdout")
    ap.add_argument("--test_fraction", type=float, default=0.25)
    ap.add_argument("--reference_mode", choices=["target_marginals", "acs_marginals"], default="acs_marginals")
    ap.add_argument("--joint_space", choices=["full", "coarse"], default="full")
    ap.add_argument("--context_feature_sets", default="acs_all_scale")
    ap.add_argument("--max_pcs", type=int, default=40)
    ap.add_argument("--ridge_alphas", default="0.01,0.1,1,10,100,1000")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jepa_seeds", default="0,1,2")
    ap.add_argument("--jepa_epochs", type=int, default=200)
    ap.add_argument("--jepa_embed_dim", type=int, default=32)
    ap.add_argument("--jepa_hidden_dim", type=int, default=128)
    ap.add_argument("--jepa_predictor_hidden_dim", type=int, default=64)
    ap.add_argument("--context_mask_rate", type=float, default=0.35)
    ap.add_argument("--ema_tau", type=float, default=0.99)
    ap.add_argument("--var_weight", type=float, default=1.0)
    ap.add_argument("--cov_weight", type=float, default=0.05)
    ap.add_argument("--jepa_lr", type=float, default=1e-3)
    ap.add_argument("--jepa_batch_size", type=int, default=256)
    ap.add_argument("--skip_jepa", action="store_true", help="Only run raw-context and oracle external-view probes.")
    ap.add_argument("--fast_multioutput_ridge", action="store_true", help="Fit one multi-output RidgeCV per feature set instead of one RidgeCV per PC.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    out_dir = args.output_dir or pathlib.Path(f"outputs/_ssl_external_view_jepa_probe_{_utc_ts()}")
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    target_keys, p_true, p_eq_target, target_marginals = _load_target(args.target_wide_csv)
    _, acs_marginals, x_acs_1d, x_acs_all, x_scale, _, _, _ = _load_acs_conditions(args.condition_csv, target_keys)
    y_ext, ext_cols = _load_wide_numeric(args.external_view_csv, target_keys, str(args.external_prefix))
    x_context_extra = None
    context_extra_label = _safe_label(args.context_extra_label)
    if args.context_extra_csv is not None:
        x_context_extra, _ = _load_wide_numeric(args.context_extra_csv, target_keys, "")
    p_eq_acs = _outer_joint([acs_marginals[v] for v in FULL_VARIABLE_ORDER])
    if args.joint_space == "coarse":
        p_true = _aggregate_full_to_coarse(p_true)
        p_eq_target = _aggregate_full_to_coarse(p_eq_target)
        p_eq_acs = _aggregate_full_to_coarse(p_eq_acs)
    reference = p_eq_target if args.reference_mode == "target_marginals" else p_eq_acs
    _, _, log_ratio = _residual_arrays(p_true, reference, eps=float(args.eps))

    feature_bank = {
        "acs_1d_scale": np.concatenate([x_acs_1d, x_scale], axis=1),
        "acs_all_scale": np.concatenate([x_acs_all, x_scale], axis=1),
    }
    if x_context_extra is not None:
        feature_bank[f"acs_all_scale_{context_extra_label}"] = np.concatenate([x_acs_all, x_scale, x_context_extra], axis=1)
    context_names = [x.strip() for x in str(args.context_feature_sets).split(",") if x.strip()]
    contexts = {name: feature_bank[name] for name in context_names if name in feature_bank}
    if not contexts:
        raise SystemExit("no available context feature sets requested")
    heldout_statefps = [str(x).zfill(2) for x in _parse_csv_ints(args.heldout_statefps)]
    jepa_seeds = _parse_csv_ints(args.jepa_seeds)
    alphas = [float(x) for x in str(args.ridge_alphas).split(",") if x.strip()]
    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()

    summary_rows: list[dict[str, Any]] = []
    pc_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []

    splits = _make_eval_splits(
        statefp,
        heldout_statefps,
        split_mode=str(args.split_mode),
        test_fraction=float(args.test_fraction),
        seed=int(args.seed),
    )
    for heldout, train_mask in splits:
        if int((~train_mask).sum()) == 0:
            print(f"[warn] split {heldout} has no held-out rows; skipped", file=sys.stderr)
            continue
        scores, explained_ratio, cumulative = _fit_pc_scores(log_ratio, train_mask=train_mask, n_components=int(args.max_pcs), seed=int(args.seed))

        def add_probe(label: str, x: np.ndarray, context_name: str, representation: str) -> None:
            if bool(args.fast_multioutput_ridge):
                summ, rows = _ridge_probe_multioutput(x=x, scores=scores, train_mask=train_mask, explained_ratio=explained_ratio, alphas=alphas)
            else:
                summ, rows = _ridge_probe(x=x, scores=scores, train_mask=train_mask, explained_ratio=explained_ratio, alphas=alphas)
            summary_rows.append(
                {
                    "heldout_statefp": heldout,
                    "split_mode": str(args.split_mode),
                    "context_feature_set": context_name,
                    "representation": representation,
                    "feature_set": label,
                    "n_features": int(x.shape[1]),
                    "n_pcs": int(scores.shape[1]),
                    "pc_cumulative_explained": float(cumulative),
                    **summ,
                }
            )
            for row in rows:
                row.update({"heldout_statefp": heldout, "context_feature_set": context_name, "representation": representation, "feature_set": label})
                pc_rows.append(row)

        add_probe("external_only", y_ext, "external", "external_oracle")
        for context_name, x_context in contexts.items():
            add_probe(context_name, x_context, context_name, "raw_context")
            add_probe(f"{context_name}_plus_external", np.concatenate([x_context, y_ext], axis=1), context_name, "raw_context_plus_external_oracle")
            if bool(args.skip_jepa):
                continue
            reps, rows = _train_external_jepa(
                x_context=x_context,
                y_target=y_ext,
                train_mask=train_mask,
                seeds=jepa_seeds,
                epochs=int(args.jepa_epochs),
                hidden_dim=int(args.jepa_hidden_dim),
                embed_dim=int(args.jepa_embed_dim),
                predictor_hidden_dim=int(args.jepa_predictor_hidden_dim),
                context_mask_rate=float(args.context_mask_rate),
                ema_tau=float(args.ema_tau),
                lr=float(args.jepa_lr),
                batch_size=int(args.jepa_batch_size),
                var_weight=float(args.var_weight),
                cov_weight=float(args.cov_weight),
                device=str(args.device),
                label=context_name,
            )
            for row in rows:
                row.update({"heldout_statefp": heldout, "split_mode": str(args.split_mode), "context_feature_set": context_name})
                train_rows.append(row)
            for rep_name, rep in reps.items():
                h = rep["h"]
                pred_ext = rep["pred_target"]
                add_probe(rep_name + "_h", h, context_name, "jepa_context_embedding")
                add_probe(rep_name + "_pred_external", pred_ext, context_name, "jepa_predicted_external")
                add_probe(rep_name + "_context_plus_pred_external", np.concatenate([x_context, pred_ext], axis=1), context_name, "raw_context_plus_jepa_predicted_external")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(metrics_dir / "external_view_jepa_probe_summary.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(metrics_dir / "external_view_jepa_probe_pc_long.csv", index=False)
    pd.DataFrame(train_rows).to_csv(metrics_dir / "external_view_jepa_training_diagnostics.csv", index=False)
    if not summary.empty:
        mean_summary = (
            summary.groupby(["context_feature_set", "representation", "feature_set"], as_index=False)
            .agg(
                mean_weighted_nonnegative_r2=("weighted_nonnegative_r2", "mean"),
                mean_weighted_raw_r2=("weighted_raw_r2", "mean"),
                mean_r2_top5=("mean_r2_top5", "mean"),
            )
            .sort_values("mean_weighted_nonnegative_r2", ascending=False)
        )
        mean_summary.to_csv(metrics_dir / "external_view_jepa_probe_mean_summary.csv", index=False)
        print(mean_summary.head(40).to_string(index=False))

    _write_json(
        out_dir / "run_summary.json",
        {
            "target_wide_csv": str(args.target_wide_csv),
            "condition_csv": str(args.condition_csv),
            "external_view_csv": str(args.external_view_csv),
            "external_prefix": str(args.external_prefix),
            "context_extra_csv": str(args.context_extra_csv) if args.context_extra_csv is not None else "",
            "context_extra_label": context_extra_label if args.context_extra_csv is not None else "",
            "external_feature_count": int(len(ext_cols)),
            "heldout_statefps": heldout_statefps,
            "split_mode": str(args.split_mode),
            "test_fraction": float(args.test_fraction),
            "reference_mode": args.reference_mode,
            "joint_space": args.joint_space,
            "context_feature_sets": list(contexts.keys()),
            "jepa_seeds": jepa_seeds,
            "jepa_epochs": int(args.jepa_epochs),
            "skip_jepa": bool(args.skip_jepa),
            "fast_multioutput_ridge": bool(args.fast_multioutput_ridge),
            "metrics": {
                "summary": str(metrics_dir / "external_view_jepa_probe_summary.csv"),
                "mean_summary": str(metrics_dir / "external_view_jepa_probe_mean_summary.csv"),
                "pc_long": str(metrics_dir / "external_view_jepa_probe_pc_long.csv"),
                "training_diagnostics": str(metrics_dir / "external_view_jepa_training_diagnostics.csv"),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
