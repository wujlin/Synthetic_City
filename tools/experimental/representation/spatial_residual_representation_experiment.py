#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.model.external_c2f_full_earn_schema import FULL_VARIABLE_ORDER
from tools.experimental.representation.residual_aware_correction_experiment import _exp_tilt, _normalize, _parse_floats, _parse_ints, _tvd
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


def _load_spatial_frame(path: pathlib.Path, target_keys: pd.DataFrame) -> pd.DataFrame:
    spatial = pd.read_csv(path, low_memory=False)
    spatial["statefp"] = spatial["statefp"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    spatial["puma5"] = spatial["puma"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    spatial["puma_uid_key"] = spatial["statefp"] + spatial["puma5"]
    spatial = spatial.drop_duplicates("puma_uid_key").set_index("puma_uid_key")
    aligned = spatial.reindex(index=target_keys["puma_uid_key"].astype(str))
    return aligned.reset_index(drop=True)


def _context_targets(spatial_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n1_cols = sorted([c for c in spatial_df.columns if c.startswith("neigh1_marg_")])
    n2_cols = sorted([c for c in spatial_df.columns if c.startswith("neigh2_marg_")])
    if not n1_cols or not n2_cols:
        raise SystemExit("spatial_df missing neigh1/neigh2 marginal columns")
    n1 = spatial_df[n1_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    n2 = spatial_df[n2_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    return n1, n2


def _coords(spatial_df: pd.DataFrame) -> np.ndarray:
    cols = ["centroid_x", "centroid_y"]
    if not all(c in spatial_df.columns for c in cols):
        raise SystemExit("spatial_df missing centroid columns")
    arr = spatial_df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    med = np.nanmedian(arr, axis=0)
    arr = np.where(np.isfinite(arr), arr, med.reshape(1, -1))
    return arr


def _knn_edges(coords: np.ndarray, train_mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx = np.where(train_mask)[0]
    coords_train = coords[train_idx]
    k_eff = min(int(k) + 1, coords_train.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(coords_train)
    dist, ind = nn.kneighbors(coords_train, return_distance=True)
    src: list[int] = []
    dst: list[int] = []
    dd: list[float] = []
    for local_i in range(coords_train.shape[0]):
        for jj in range(1, k_eff):
            src.append(int(local_i))
            dst.append(int(ind[local_i, jj]))
            dd.append(float(dist[local_i, jj]))
    d = np.asarray(dd, dtype=np.float64)
    tau = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    w = np.exp(-d / max(tau, 1e-12))
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), w.astype(np.float64)


def _standardize_train_all(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_mask]).astype(np.float32)
    x_all = scaler.transform(x).astype(np.float32)
    return x_train, x_all


@dataclass(frozen=True)
class RouteConfig:
    name: str
    smooth_alpha: float = 0.0
    ctx1_beta: float = 0.0
    ctx2_gamma: float = 0.0


def _parse_routes(route_spec: str) -> list[RouteConfig]:
    routes: list[RouteConfig] = []
    for token in [x.strip() for x in route_spec.split(",") if x.strip()]:
        if token == "residual":
            routes.append(RouteConfig("residual"))
        elif token.startswith("smooth"):
            val = float(token.split(":", 1)[1]) if ":" in token else 0.01
            routes.append(RouteConfig(f"smooth_{val:g}", smooth_alpha=val))
        elif token.startswith("context"):
            val = float(token.split(":", 1)[1]) if ":" in token else 0.1
            routes.append(RouteConfig(f"context_{val:g}", ctx1_beta=val))
        elif token.startswith("multiscale"):
            if ":" in token:
                vals = [float(v) for v in token.split(":", 1)[1].split("+")]
                beta = vals[0]
                gamma = vals[1] if len(vals) > 1 else vals[0]
            else:
                beta = gamma = 0.1
            routes.append(RouteConfig(f"multiscale_{beta:g}_{gamma:g}", ctx1_beta=beta, ctx2_gamma=gamma))
        else:
            raise SystemExit(f"unknown route: {token}")
    return routes


def _train_spatial_model(
    *,
    x: np.ndarray,
    scores: np.ndarray,
    explained_ratio: np.ndarray,
    context1: np.ndarray,
    context2: np.ndarray,
    coords: np.ndarray,
    train_mask: np.ndarray,
    route: RouteConfig,
    seed: int,
    epochs: int,
    hidden_dim: int,
    embed_dim: int,
    lr: float,
    batch_size: int,
    knn_k: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self, d_in: int, hidden: int, embed: int, d_out: int, d_ctx1: int, d_ctx2: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, embed), nn.ReLU())
            self.res_head = nn.Linear(embed, d_out)
            self.ctx1_head = nn.Linear(embed, d_ctx1)
            self.ctx2_head = nn.Linear(embed, d_ctx2)

        def forward(self, x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            h = self.encoder(x_in)
            return h, self.res_head(h), self.ctx1_head(h), self.ctx2_head(h)

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    x_train, x_all = _standardize_train_all(x, train_mask)
    y_train_raw = scores[train_mask].astype(np.float64)
    y_mean = y_train_raw.mean(axis=0, keepdims=True)
    y_std = y_train_raw.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)
    y_train = ((y_train_raw - y_mean) / y_std).astype(np.float32)

    c1_train, _ = _standardize_train_all(context1, train_mask)
    c2_train, _ = _standardize_train_all(context2, train_mask)
    edge_src, edge_dst, edge_w = _knn_edges(coords, train_mask, k=int(knn_k))

    dev = torch.device(device)
    model = Model(
        d_in=x_train.shape[1],
        hidden=int(hidden_dim),
        embed=int(embed_dim),
        d_out=scores.shape[1],
        d_ctx1=c1_train.shape[1],
        d_ctx2=c2_train.shape[1],
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=dev)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=dev)
    c1_train_t = torch.tensor(c1_train, dtype=torch.float32, device=dev)
    c2_train_t = torch.tensor(c2_train, dtype=torch.float32, device=dev)
    x_all_t = torch.tensor(x_all, dtype=torch.float32, device=dev)
    weights = np.asarray(explained_ratio, dtype=np.float64)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    weights_t = torch.tensor(weights.astype(np.float32), dtype=torch.float32, device=dev).reshape(1, -1)
    edge_src_t = torch.tensor(edge_src, dtype=torch.long, device=dev)
    edge_dst_t = torch.tensor(edge_dst, dtype=torch.long, device=dev)
    edge_w_t = torch.tensor(edge_w.astype(np.float32), dtype=torch.float32, device=dev)

    n_train = int(x_train.shape[0])
    batch_size = min(int(batch_size), n_train)
    last_losses: dict[str, float] = {}
    for _ in range(int(epochs)):
        order = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            _, pred, ctx1_pred, ctx2_pred = model(x_train_t[idx])
            res_loss = torch.mean(((pred - y_train_t[idx]) ** 2) * weights_t)
            ctx1_loss = torch.mean((ctx1_pred - c1_train_t[idx]) ** 2)
            ctx2_loss = torch.mean((ctx2_pred - c2_train_t[idx]) ** 2)
            loss = res_loss + float(route.ctx1_beta) * ctx1_loss + float(route.ctx2_gamma) * ctx2_loss
            if float(route.smooth_alpha) > 0.0 and edge_src_t.numel() > 0:
                h_all, _, _, _ = model(x_train_t)
                diff = h_all[edge_src_t] - h_all[edge_dst_t]
                smooth_loss = torch.sum(edge_w_t * torch.mean(diff * diff, dim=1)) / torch.sum(edge_w_t).clamp_min(1e-8)
                loss = loss + float(route.smooth_alpha) * smooth_loss
            else:
                smooth_loss = torch.zeros((), dtype=torch.float32, device=dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_losses = {
                "res_loss": float(res_loss.detach().cpu()),
                "ctx1_loss": float(ctx1_loss.detach().cpu()),
                "ctx2_loss": float(ctx2_loss.detach().cpu()),
                "smooth_loss": float(smooth_loss.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
            }

    with torch.no_grad():
        _, pred_all, _, _ = model(x_all_t)
        pred_scores = pred_all.cpu().numpy().astype(np.float64) * y_std + y_mean
    return pred_scores, {
        "route": route.name,
        "seed": int(seed),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "embed_dim": int(embed_dim),
        "smooth_alpha": float(route.smooth_alpha),
        "ctx1_beta": float(route.ctx1_beta),
        "ctx2_gamma": float(route.ctx2_gamma),
        "last_losses": last_losses,
    }


def main() -> int:
    data_root = pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data")
    ap = argparse.ArgumentParser(description="Evaluate spatial inductive biases for residual-aware representation learning.")
    ap.add_argument("--target_wide_csv", type=pathlib.Path, default=data_root / "us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv")
    ap.add_argument("--condition_csv", type=pathlib.Path, default=data_root / "us/processed/external_conditions/extcond_v1_agesex_earn_v1_acs5_2022_puma_us.csv")
    ap.add_argument("--spatial_csv", type=pathlib.Path, default=data_root / "us/processed/features/puma_spatial_features_5var_knn6.csv")
    ap.add_argument("--joint_space", choices=["coarse", "full"], default="coarse")
    ap.add_argument("--reference_mode", choices=["acs_marginals", "target_marginals"], default="acs_marginals")
    ap.add_argument("--heldout_statefps", default="26,06,12,48,55")
    ap.add_argument("--routes", default="residual,smooth:0.01,context:0.1,multiscale:0.1+0.1")
    ap.add_argument("--input_mode", choices=["acs_1d", "acs_all"], default="acs_1d")
    ap.add_argument("--n_pcs", type=int, default=40)
    ap.add_argument("--lambdas", default="1.0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--embed_dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--knn_k", type=int, default=6)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--clip_log_ratio", type=float, default=8.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    out_dir = args.output_dir or pathlib.Path(f"outputs/_spatial_residual_representation_{_utc_ts()}")
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    target_keys, p_true, p_eq_target, target_marginals = _load_target(args.target_wide_csv)
    _, acs_marginals, x_acs_1d, x_acs_all, _, _ = _load_acs_conditions(args.condition_csv, target_keys)
    spatial_df = _load_spatial_frame(args.spatial_csv, target_keys)
    context1, context2 = _context_targets(spatial_df)
    coords = _coords(spatial_df)
    p_eq_acs = _outer_joint([acs_marginals[v] for v in FULL_VARIABLE_ORDER])

    if args.joint_space == "coarse":
        p_true = _aggregate_full_to_coarse(p_true)
        p_eq_target = _aggregate_full_to_coarse(p_eq_target)
        p_eq_acs = _aggregate_full_to_coarse(p_eq_acs)

    p_true = _normalize(p_true)
    p_eq = _normalize(p_eq_acs if args.reference_mode == "acs_marginals" else p_eq_target)
    log_ratio = np.log(p_true + float(args.eps)) - np.log(p_eq + float(args.eps))
    x = x_acs_1d if args.input_mode == "acs_1d" else x_acs_all
    heldout_statefps = [str(x).zfill(2) for x in _parse_ints(args.heldout_statefps)]
    routes = _parse_routes(args.routes)
    seeds = _parse_ints(args.seeds)
    lambdas = _parse_floats(args.lambdas)
    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()

    metric_rows: list[dict[str, Any]] = []
    pc_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []

    for heldout in heldout_statefps:
        train_mask = statefp != heldout
        if int((~train_mask).sum()) == 0:
            continue
        n_comp = min(int(args.n_pcs), int(train_mask.sum()) - 1, log_ratio.shape[1])
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=0)
        pca.fit(log_ratio[train_mask])
        scores_all = pca.transform(log_ratio)
        evr = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
        baseline_tvd = _tvd(p_true[~train_mask], p_eq[~train_mask])
        metric_rows.append(
            {
                "heldout_statefp": heldout,
                "route": "reference",
                "seed": -1,
                "lambda": 0.0,
                "tvd_mean": float(np.mean(baseline_tvd)),
                "tvd_delta_vs_reference": 0.0,
                "tvd_relative_improvement": 0.0,
                "pc_weighted_nonnegative_r2": 0.0,
                "pc_weighted_raw_r2": 0.0,
            }
        )

        for route in routes:
            for seed in seeds:
                pred_scores, info = _train_spatial_model(
                    x=x,
                    scores=scores_all,
                    explained_ratio=evr,
                    context1=context1,
                    context2=context2,
                    coords=coords,
                    train_mask=train_mask,
                    route=route,
                    seed=int(seed),
                    epochs=int(args.epochs),
                    hidden_dim=int(args.hidden_dim),
                    embed_dim=int(args.embed_dim),
                    lr=float(args.lr),
                    batch_size=int(args.batch_size),
                    knn_k=int(args.knn_k),
                    device=str(args.device),
                )
                train_rows.append({"heldout_statefp": heldout, **info})
                r2_vals = []
                for j in range(n_comp):
                    r2 = float(r2_score(scores_all[~train_mask, j], pred_scores[~train_mask, j]))
                    r2_vals.append(r2)
                    pc_rows.append(
                        {
                            "heldout_statefp": heldout,
                            "route": route.name,
                            "seed": int(seed),
                            "pc_index": int(j + 1),
                            "explained_variance_ratio": float(evr[j]),
                            "r2_test": r2,
                        }
                    )
                r2_arr = np.asarray(r2_vals, dtype=np.float64)
                weighted_nonneg = float(np.sum(np.maximum(r2_arr, 0.0) * evr))
                weighted_raw = float(np.sum(r2_arr * evr))
                z_hat = pca.inverse_transform(pred_scores)[~train_mask]
                for lam in lambdas:
                    q = _exp_tilt(p_eq[~train_mask], z_hat, lam=float(lam), clip=float(args.clip_log_ratio))
                    tvd = _tvd(p_true[~train_mask], q)
                    delta = baseline_tvd - tvd
                    metric_rows.append(
                        {
                            "heldout_statefp": heldout,
                            "route": route.name,
                            "seed": int(seed),
                            "lambda": float(lam),
                            "tvd_mean": float(np.mean(tvd)),
                            "tvd_delta_vs_reference": float(np.mean(delta)),
                            "tvd_relative_improvement": float(np.mean(delta) / max(float(np.mean(baseline_tvd)), 1e-12)),
                            "pc_weighted_nonnegative_r2": weighted_nonneg,
                            "pc_weighted_raw_r2": weighted_raw,
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    pc = pd.DataFrame(pc_rows)
    train = pd.DataFrame(train_rows)
    metrics.to_csv(out_dir / "metrics" / "spatial_route_tvd_long.csv", index=False)
    pc.to_csv(out_dir / "metrics" / "spatial_route_pc_long.csv", index=False)
    train.to_csv(out_dir / "metrics" / "spatial_route_training.csv", index=False)

    summary = (
        metrics[metrics["route"] != "reference"]
        .groupby(["route", "lambda"], as_index=False)
        .agg(
            tvd_mean=("tvd_mean", "mean"),
            tvd_std=("tvd_mean", "std"),
            tvd_delta_vs_reference=("tvd_delta_vs_reference", "mean"),
            tvd_relative_improvement=("tvd_relative_improvement", "mean"),
            pc_weighted_nonnegative_r2=("pc_weighted_nonnegative_r2", "mean"),
            pc_weighted_raw_r2=("pc_weighted_raw_r2", "mean"),
        )
        .sort_values("tvd_mean")
    )
    summary.to_csv(out_dir / "metrics" / "spatial_route_summary.csv", index=False)

    run_summary = {
        "created_utc": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "question": "Do spatial smoothness, neighbor context, or multi-scale context improve residual-aware correction?",
        "joint_space": str(args.joint_space),
        "reference_mode": str(args.reference_mode),
        "input_mode": str(args.input_mode),
        "heldout_statefps": heldout_statefps,
        "routes": [r.__dict__ for r in routes],
        "seeds": seeds,
        "n_pcs": int(args.n_pcs),
        "lambdas": lambdas,
        "outputs": {
            "spatial_route_tvd_long": str(out_dir / "metrics" / "spatial_route_tvd_long.csv"),
            "spatial_route_pc_long": str(out_dir / "metrics" / "spatial_route_pc_long.csv"),
            "spatial_route_summary": str(out_dir / "metrics" / "spatial_route_summary.csv"),
            "spatial_route_training": str(out_dir / "metrics" / "spatial_route_training.csv"),
        },
    }
    _write_json(out_dir / "run_summary.json", run_summary)
    print(summary.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
