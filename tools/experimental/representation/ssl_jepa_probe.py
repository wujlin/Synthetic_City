#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
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
    _ridge_probe,
    _write_json,
)


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_label(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(text).strip()) or "feature"


def _effective_rank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return float("nan")
    centered = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    if not np.any(s > 0):
        return 0.0
    p = s / np.clip(s.sum(), 1e-12, None)
    return float(np.exp(-np.sum(p * np.log(np.clip(p, 1e-12, None)))))


def _embedding_diagnostics(emb: np.ndarray, train_mask: np.ndarray) -> dict[str, float]:
    z = np.asarray(emb[train_mask], dtype=np.float64)
    std = z.std(axis=0)
    corr = np.corrcoef(z, rowvar=False)
    if corr.ndim == 0:
        offdiag_abs = float("nan")
    else:
        off = corr[~np.eye(corr.shape[0], dtype=bool)]
        offdiag_abs = float(np.nanmean(np.abs(off))) if off.size else 0.0
    return {
        "embedding_std_mean": float(np.mean(std)),
        "embedding_std_min": float(np.min(std)),
        "embedding_std_max": float(np.max(std)),
        "embedding_effective_rank": _effective_rank(z),
        "embedding_abs_corr_offdiag_mean": offdiag_abs,
    }


def _jepa_embeddings(
    x: np.ndarray,
    train_mask: np.ndarray,
    *,
    seeds: list[int],
    epochs: int,
    hidden_dim: int,
    embed_dim: int,
    predictor_hidden_dim: int,
    context_mask_rate: float,
    target_mask_rate: float,
    ema_tau: float,
    lr: float,
    batch_size: int,
    var_weight: float,
    cov_weight: float,
    device: str,
    label: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
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

        def forward(self, x_in: torch.Tensor) -> torch.Tensor:
            return self.net(x_in)

    class Predictor(nn.Module):
        def __init__(self, embed: int, hidden: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embed, hidden),
                nn.GELU(),
                nn.Linear(hidden, embed),
            )

        def forward(self, z_in: torch.Tensor) -> torch.Tensor:
            return self.net(z_in)

    def make_aug(xb: torch.Tensor, mask_rate: float) -> torch.Tensor:
        if mask_rate <= 0:
            return xb
        mask = (torch.rand_like(xb) < float(mask_rate)).float()
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
        for p_src, p_dst in zip(src.parameters(), dst.parameters(), strict=True):
            p_dst.data.mul_(float(tau)).add_(p_src.data, alpha=1.0 - float(tau))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_mask]).astype(np.float32)
    x_all = scaler.transform(x).astype(np.float32)
    n_train, d_in = x_train.shape
    batch_size = min(int(batch_size), n_train)
    dev = torch.device(device)
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=dev)
    x_all_t = torch.tensor(x_all, dtype=torch.float32, device=dev)

    out: dict[str, np.ndarray] = {}
    train_rows: list[dict[str, Any]] = []
    label = _safe_label(label)
    for seed in seeds:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        online = Encoder(d_in=d_in, hidden=int(hidden_dim), embed=int(embed_dim)).to(dev)
        target = Encoder(d_in=d_in, hidden=int(hidden_dim), embed=int(embed_dim)).to(dev)
        target.load_state_dict(online.state_dict())
        for p in target.parameters():
            p.requires_grad_(False)
        predictor = Predictor(embed=int(embed_dim), hidden=int(predictor_hidden_dim)).to(dev)
        opt = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()), lr=float(lr), weight_decay=1e-4)

        last_loss = float("nan")
        last_pred = float("nan")
        last_var = float("nan")
        last_cov = float("nan")
        for _ in range(int(epochs)):
            order = torch.randperm(n_train, device=dev)
            for start in range(0, n_train, batch_size):
                idx = order[start : start + batch_size]
                if idx.numel() < 4:
                    continue
                xb = x_train_t[idx]
                xc = make_aug(xb, float(context_mask_rate))
                xt = make_aug(xb, float(target_mask_rate))
                z_online = online(xc)
                pred = predictor(z_online)
                with torch.no_grad():
                    z_target = target(xt)
                pred_loss = F.mse_loss(F.normalize(pred, dim=1), F.normalize(z_target, dim=1))
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
            emb = online(x_all_t).cpu().numpy().astype(np.float64)
        name = f"jepa_{label}_seed{seed}"
        out[name] = emb
        row = {
            "embedding_name": name,
            "seed": int(seed),
            "feature_label": label,
            "final_loss": last_loss,
            "final_pred_loss": last_pred,
            "final_var_loss": last_var,
            "final_cov_loss": last_cov,
            **_embedding_diagnostics(emb, train_mask),
        }
        train_rows.append(row)
    return out, train_rows


def _load_optional_view(path: pathlib.Path | None, target_keys: pd.DataFrame) -> np.ndarray | None:
    if path is None:
        return None
    return _load_spatial(path, target_keys)[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Tabular JEPA probe for census-condition representations against copula-residual PCs."
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
    ap.add_argument("--uncertainty_csv", type=pathlib.Path, default=None)
    ap.add_argument("--heldout_statefps", default="26,06,12,48,55")
    ap.add_argument("--reference_mode", choices=["target_marginals", "acs_marginals"], default="acs_marginals")
    ap.add_argument("--joint_space", choices=["full", "coarse"], default="full")
    ap.add_argument("--feature_sets", default="acs_all_scale,acs_all_scale_uncertainty")
    ap.add_argument("--max_pcs", type=int, default=40)
    ap.add_argument("--ridge_alphas", default="0.01,0.1,1,10,100,1000")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jepa_seeds", default="0,1,2")
    ap.add_argument("--jepa_epochs", type=int, default=300)
    ap.add_argument("--jepa_embed_dim", type=int, default=32)
    ap.add_argument("--jepa_hidden_dim", type=int, default=128)
    ap.add_argument("--jepa_predictor_hidden_dim", type=int, default=64)
    ap.add_argument("--context_mask_rate", type=float, default=0.45)
    ap.add_argument("--target_mask_rate", type=float, default=0.10)
    ap.add_argument("--ema_tau", type=float, default=0.99)
    ap.add_argument("--var_weight", type=float, default=1.0)
    ap.add_argument("--cov_weight", type=float, default=0.05)
    ap.add_argument("--jepa_lr", type=float, default=1e-3)
    ap.add_argument("--jepa_batch_size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    out_dir = args.output_dir or pathlib.Path(f"outputs/_ssl_jepa_probe_{_utc_ts()}")
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    target_keys, p_true, p_eq_target, target_marginals = _load_target(args.target_wide_csv)
    _, acs_marginals, x_acs_1d, x_acs_all, x_scale, _, _, _ = _load_acs_conditions(args.condition_csv, target_keys)
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
    }
    if x_uncertainty is not None:
        feature_bank["acs_all_scale_uncertainty"] = np.concatenate([x_acs_all, x_scale, x_uncertainty], axis=1)
    requested = [x.strip() for x in str(args.feature_sets).split(",") if x.strip()]
    feature_sets = {name: feature_bank[name] for name in requested if name in feature_bank}
    if not feature_sets:
        raise SystemExit("no requested feature sets are available")

    heldout_statefps = [str(x).zfill(2) for x in _parse_csv_ints(args.heldout_statefps)]
    jepa_seeds = _parse_csv_ints(args.jepa_seeds)
    ridge_alphas = [float(x) for x in str(args.ridge_alphas).split(",") if x.strip()]
    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()

    summary_rows: list[dict[str, Any]] = []
    pc_rows: list[dict[str, Any]] = []
    jepa_rows: list[dict[str, Any]] = []

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
        for feature_name, x in feature_sets.items():
            raw_summary, raw_pc_rows = _ridge_probe(
                x=x,
                scores=scores,
                train_mask=train_mask,
                explained_ratio=explained_ratio,
                alphas=ridge_alphas,
            )
            summary_rows.append(
                {
                    "heldout_statefp": heldout,
                    "feature_set": feature_name,
                    "representation": "raw",
                    "n_features": int(x.shape[1]),
                    "n_pcs": int(scores.shape[1]),
                    "pc_cumulative_explained": float(cumulative),
                    **raw_summary,
                }
            )
            for row in raw_pc_rows:
                row.update({"heldout_statefp": heldout, "feature_set": feature_name, "representation": "raw"})
                pc_rows.append(row)

            embeds, train_meta = _jepa_embeddings(
                x=x,
                train_mask=train_mask,
                seeds=jepa_seeds,
                epochs=int(args.jepa_epochs),
                hidden_dim=int(args.jepa_hidden_dim),
                embed_dim=int(args.jepa_embed_dim),
                predictor_hidden_dim=int(args.jepa_predictor_hidden_dim),
                context_mask_rate=float(args.context_mask_rate),
                target_mask_rate=float(args.target_mask_rate),
                ema_tau=float(args.ema_tau),
                lr=float(args.jepa_lr),
                batch_size=int(args.jepa_batch_size),
                var_weight=float(args.var_weight),
                cov_weight=float(args.cov_weight),
                device=str(args.device),
                label=feature_name,
            )
            for row in train_meta:
                row.update({"heldout_statefp": heldout})
                jepa_rows.append(row)
            for emb_name, emb in embeds.items():
                emb_summary, emb_pc_rows = _ridge_probe(
                    x=emb,
                    scores=scores,
                    train_mask=train_mask,
                    explained_ratio=explained_ratio,
                    alphas=ridge_alphas,
                )
                summary_rows.append(
                    {
                        "heldout_statefp": heldout,
                        "feature_set": feature_name,
                        "representation": emb_name,
                        "n_features": int(emb.shape[1]),
                        "n_pcs": int(scores.shape[1]),
                        "pc_cumulative_explained": float(cumulative),
                        **_embedding_diagnostics(emb, train_mask),
                        **emb_summary,
                    }
                )
                for row in emb_pc_rows:
                    row.update({"heldout_statefp": heldout, "feature_set": feature_name, "representation": emb_name})
                    pc_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(metrics_dir / "jepa_probe_summary.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(metrics_dir / "jepa_probe_pc_long.csv", index=False)
    pd.DataFrame(jepa_rows).to_csv(metrics_dir / "jepa_training_diagnostics.csv", index=False)
    if not summary.empty:
        mean_summary = (
            summary.groupby(["feature_set", "representation"], as_index=False)
            .agg(
                mean_weighted_nonnegative_r2=("weighted_nonnegative_r2", "mean"),
                mean_weighted_raw_r2=("weighted_raw_r2", "mean"),
                mean_r2_top5=("mean_r2_top5", "mean"),
                mean_embedding_std_min=("embedding_std_min", "mean"),
                mean_embedding_effective_rank=("embedding_effective_rank", "mean"),
            )
            .sort_values("mean_weighted_nonnegative_r2", ascending=False)
        )
        mean_summary.to_csv(metrics_dir / "jepa_probe_mean_summary.csv", index=False)
        print(mean_summary.head(30).to_string(index=False))

    _write_json(
        out_dir / "run_summary.json",
        {
            "target_wide_csv": str(args.target_wide_csv),
            "condition_csv": str(args.condition_csv),
            "uncertainty_csv": str(args.uncertainty_csv) if args.uncertainty_csv is not None else "",
            "heldout_statefps": heldout_statefps,
            "reference_mode": args.reference_mode,
            "joint_space": args.joint_space,
            "feature_sets": list(feature_sets.keys()),
            "jepa_seeds": jepa_seeds,
            "jepa_epochs": int(args.jepa_epochs),
            "jepa_embed_dim": int(args.jepa_embed_dim),
            "context_mask_rate": float(args.context_mask_rate),
            "target_mask_rate": float(args.target_mask_rate),
            "ema_tau": float(args.ema_tau),
            "metrics": {
                "summary": str(metrics_dir / "jepa_probe_summary.csv"),
                "mean_summary": str(metrics_dir / "jepa_probe_mean_summary.csv"),
                "pc_long": str(metrics_dir / "jepa_probe_pc_long.csv"),
                "training_diagnostics": str(metrics_dir / "jepa_training_diagnostics.csv"),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
