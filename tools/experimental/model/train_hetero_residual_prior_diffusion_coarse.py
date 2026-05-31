#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from synthpop.model.diffusion_tabular import DiffusionTabularModel, TabDDPMConfig, _require_torch
from tools.model.external_c2f_full_earn_schema import COARSE_K, COARSE_SHAPE
from tools.experimental.representation.residual_aware_correction_experiment import (
    _cosine,
    _exp_tilt,
    _ipf_project_rows,
    _load_spatial,
    _mean_marginal_gap,
    _normalize,
    _tvd,
)
from tools.experimental.representation.ssl_copula_residual_probe import (
    _aggregate_full_to_coarse,
    _load_acs_conditions,
    _load_target,
    _outer_joint,
    _write_json,
)
from tools.experimental.model.train_residual_hierarchical_coarse_diffusion import (
    _append_eval_rows,
    _average_projected_draws,
    _coarse_constraint_marginals,
)


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_ints(spec: str) -> list[int]:
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


def _parse_floats(spec: str) -> list[float]:
    return [float(x.strip()) for x in str(spec).split(",") if x.strip()]


def _parse_hidden_dims(spec: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in str(spec).split(",") if x.strip())
    if not vals:
        raise SystemExit("--hidden_dims cannot be empty")
    return vals


@dataclass
class HeteroFit:
    mean_std_all: np.ndarray
    logvar_std_all: np.ndarray
    score_mean: np.ndarray
    score_std: np.ndarray
    train_weighted_nll: float
    test_weighted_nll: float
    coverage_68: float
    coverage_90: float
    coverage_95: float
    mean_pred_var_abs_error_corr: float
    final_loss: float


def _train_hetero_prior(
    *,
    x: np.ndarray,
    scores: np.ndarray,
    train_mask: np.ndarray,
    explained_ratio: np.ndarray,
    seed: int,
    epochs: int,
    hidden_dim: int,
    embed_dim: int,
    lr: float,
    batch_size: int,
    device: str,
) -> HeteroFit:
    torch = _require_torch()
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self, d_in: int, hidden: int, embed: int, d_out: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.SiLU(),
                nn.Linear(hidden, embed),
                nn.SiLU(),
            )
            self.mean_head = nn.Linear(embed, d_out)
            self.logvar_head = nn.Linear(embed, d_out)

        def forward(self, x_in: Any) -> tuple[Any, Any]:
            h = self.encoder(x_in)
            return self.mean_head(h), self.logvar_head(h)

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))

    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x[train_mask]).astype(np.float32)
    x_all = x_scaler.transform(x).astype(np.float32)
    y_train_raw = scores[train_mask].astype(np.float64)
    y_all_raw = scores.astype(np.float64)
    y_mean = y_train_raw.mean(axis=0, keepdims=True)
    y_std = y_train_raw.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)
    y_train = ((y_train_raw - y_mean) / y_std).astype(np.float32)
    y_all = ((y_all_raw - y_mean) / y_std).astype(np.float32)

    dev = torch.device(device)
    model = Model(x_train.shape[1], int(hidden_dim), int(embed_dim), scores.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=dev)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=dev)
    x_all_t = torch.tensor(x_all, dtype=torch.float32, device=dev)
    weights = np.asarray(explained_ratio, dtype=np.float64)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    weights_t = torch.tensor(weights.astype(np.float32), dtype=torch.float32, device=dev).reshape(1, -1)
    n_train = int(x_train.shape[0])
    batch_size = min(int(batch_size), n_train)
    final_loss = float("nan")

    for _ in range(int(epochs)):
        order = torch.randperm(n_train, device=dev)
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            pred_mean, pred_logvar = model(x_train_t[idx])
            pred_logvar = torch.clamp(pred_logvar, min=-6.0, max=4.0)
            nll = 0.5 * (((pred_mean - y_train_t[idx]) ** 2) * torch.exp(-pred_logvar) + pred_logvar)
            loss = torch.mean(nll * weights_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            final_loss = float(loss.detach().cpu().item())

    with torch.no_grad():
        mean_all_t, logvar_all_t = model(x_all_t)
        logvar_all_t = torch.clamp(logvar_all_t, min=-6.0, max=4.0)
        mean_all = mean_all_t.cpu().numpy().astype(np.float64)
        logvar_all = logvar_all_t.cpu().numpy().astype(np.float64)

    test_mask = ~train_mask
    y_all_f = y_all.astype(np.float64)
    pred_var = np.exp(logvar_all)
    nll_all = 0.5 * (((mean_all - y_all_f) ** 2) / np.clip(pred_var, 1e-12, None) + logvar_all)
    train_weighted_nll = float(np.mean(nll_all[train_mask] * weights.reshape(1, -1)))
    test_weighted_nll = float(np.mean(nll_all[test_mask] * weights.reshape(1, -1)))

    std = np.sqrt(np.clip(pred_var[test_mask], 1e-12, None))
    err = np.abs(y_all_f[test_mask] - mean_all[test_mask])
    coverage_68 = float(np.mean(err <= 1.0 * std))
    coverage_90 = float(np.mean(err <= 1.6448536269514722 * std))
    coverage_95 = float(np.mean(err <= 1.959963984540054 * std))
    corrs: list[float] = []
    for j in range(scores.shape[1]):
        pv = pred_var[test_mask, j]
        ae = err[:, j]
        if np.std(pv) > 1e-12 and np.std(ae) > 1e-12:
            corr = float(np.corrcoef(pv, ae)[0, 1])
            if np.isfinite(corr):
                corrs.append(corr)

    return HeteroFit(
        mean_std_all=mean_all,
        logvar_std_all=logvar_all,
        score_mean=y_mean.astype(np.float64),
        score_std=y_std.astype(np.float64),
        train_weighted_nll=train_weighted_nll,
        test_weighted_nll=test_weighted_nll,
        coverage_68=coverage_68,
        coverage_90=coverage_90,
        coverage_95=coverage_95,
        mean_pred_var_abs_error_corr=float(np.mean(corrs)) if corrs else float("nan"),
        final_loss=float(final_loss),
    )


def _scores_std_to_z_draws(
    *,
    pca: PCA,
    score_mean: np.ndarray,
    score_std: np.ndarray,
    score_draws_std: np.ndarray,
) -> np.ndarray:
    n_test, n_draws, n_pcs = score_draws_std.shape
    flat_std = score_draws_std.reshape((n_test * n_draws, n_pcs))
    flat_scores = flat_std * score_std.reshape(1, -1) + score_mean.reshape(1, -1)
    flat_z = pca.inverse_transform(flat_scores)
    return flat_z.reshape((n_test, n_draws, -1))


def _append_prior_calibration_row(
    rows: list[dict[str, Any]],
    *,
    heldout: str,
    feature_set: str,
    seed: int,
    fit: HeteroFit,
    n_pcs: int,
    pca_explained: np.ndarray,
) -> None:
    rows.append(
        {
            "heldout_statefp": heldout,
            "feature_set": feature_set,
            "seed": int(seed),
            "n_pcs": int(n_pcs),
            "pca_explained_sum": float(np.sum(pca_explained)),
            "hetero_train_weighted_nll": float(fit.train_weighted_nll),
            "hetero_test_weighted_nll": float(fit.test_weighted_nll),
            "coverage_68": float(fit.coverage_68),
            "coverage_90": float(fit.coverage_90),
            "coverage_95": float(fit.coverage_95),
            "mean_pred_var_abs_error_corr": float(fit.mean_pred_var_abs_error_corr),
            "hetero_final_loss": float(fit.final_loss),
        }
    )


def main() -> int:
    data_root = pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data")
    default_uncertainty = pathlib.Path("/home/jinlin/projects/Synthetic_City/data/us/processed/features/acs5_2022_puma_uncertainty_us.csv")
    ap = argparse.ArgumentParser(
        description="Train a heteroskedastic residual-PC prior plus diffusion over standardized residual noise."
    )
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
    ap.add_argument("--uncertainty_csv", type=pathlib.Path, default=default_uncertainty)
    ap.add_argument("--heldout_statefps", default="26,06,12,48,55")
    ap.add_argument("--feature_sets", default="acs_all_scale_survey_uncertainty,acs_all_scale")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lambdas", default="1.0")
    ap.add_argument("--projection_modes", default="acs_ipf")
    ap.add_argument("--n_pcs", type=int, default=40)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--clip_log_ratio", type=float, default=8.0)
    ap.add_argument("--hetero_epochs", type=int, default=400)
    ap.add_argument("--hetero_hidden_dim", type=int, default=128)
    ap.add_argument("--hetero_embed_dim", type=int, default=32)
    ap.add_argument("--hetero_lr", type=float, default=1e-3)
    ap.add_argument("--diffusion_epochs", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--n_draws", type=int, default=64)
    ap.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    ap.add_argument("--hidden_dims", default="256,256")
    ap.add_argument("--condition_injection", choices=["concat", "film"], default="film")
    ap.add_argument("--film_hidden_dim", type=int, default=128)
    ap.add_argument("--diffusion_lr", type=float, default=1e-3)
    ap.add_argument("--diffusion_weight_decay", type=float, default=1e-4)
    ap.add_argument("--noise_clip", type=float, default=6.0)
    ap.add_argument("--ipf_iters", type=int, default=300)
    ap.add_argument("--ipf_tol", type=float, default=1e-9)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    torch = _require_torch()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir or pathlib.Path(f"outputs/_hetero_residual_prior_diffusion_coarse_{_utc_ts()}")
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    target_keys, p_true_full, _, _target_marginals = _load_target(args.target_wide_csv)
    _acs_feature_df, acs_marginals, x_acs_1d, x_acs_all, x_acs_scale, _, _, _ = _load_acs_conditions(
        args.condition_csv,
        target_keys,
    )
    x_spatial, _spatial_cols = _load_spatial(args.spatial_csv, target_keys)
    x_uncertainty: np.ndarray | None = None
    if args.uncertainty_csv is not None and pathlib.Path(args.uncertainty_csv).exists():
        x_uncertainty, _uncertainty_cols = _load_spatial(args.uncertainty_csv, target_keys)

    p_true = _aggregate_full_to_coarse(p_true_full)
    coarse_targets, _coarse_targets_named = _coarse_constraint_marginals(acs_marginals)
    p_eq = _normalize(_outer_joint(coarse_targets))
    log_ratio = np.log(p_true + float(args.eps)) - np.log(p_eq + float(args.eps))

    feature_map: dict[str, np.ndarray] = {
        "acs_1d": x_acs_1d,
        "acs_all": x_acs_all,
        "acs_1d_scale": np.concatenate([x_acs_1d, x_acs_scale], axis=1),
        "acs_all_scale": np.concatenate([x_acs_all, x_acs_scale], axis=1),
        "acs_all_spatial_scale": np.concatenate([x_acs_all, x_spatial, x_acs_scale], axis=1),
    }
    if x_uncertainty is not None:
        feature_map["survey_uncertainty"] = x_uncertainty
        feature_map["acs_all_survey_uncertainty"] = np.concatenate([x_acs_all, x_uncertainty], axis=1)
        feature_map["acs_all_scale_survey_uncertainty"] = np.concatenate([x_acs_all, x_acs_scale, x_uncertainty], axis=1)
        feature_map["acs_all_spatial_scale_survey_uncertainty"] = np.concatenate(
            [x_acs_all, x_spatial, x_acs_scale, x_uncertainty],
            axis=1,
        )

    feature_sets = [x.strip() for x in str(args.feature_sets).split(",") if x.strip()]
    projection_modes = [x.strip() for x in str(args.projection_modes).split(",") if x.strip()]
    heldout_statefps = [str(x).zfill(2) for x in _parse_ints(args.heldout_statefps)]
    seeds = _parse_ints(args.seeds)
    lambdas = _parse_floats(args.lambdas)
    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()

    rows: list[dict[str, Any]] = []
    by_puma_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for heldout in heldout_statefps:
        train_mask = statefp != heldout
        heldout_mask = ~train_mask
        if int(heldout_mask.sum()) == 0:
            continue
        n_pcs = min(int(args.n_pcs), int(train_mask.sum()) - 1, int(COARSE_K))
        pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=0)
        scores_train = pca.fit_transform(log_ratio[train_mask])
        scores_all = pca.transform(log_ratio)
        explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
        p_true_test = p_true[heldout_mask]
        p_eq_test = p_eq[heldout_mask]
        targets_test = [m[heldout_mask] for m in coarse_targets]

        _append_eval_rows(
            rows=rows,
            by_puma_rows=by_puma_rows,
            target_keys=target_keys,
            heldout_mask=heldout_mask,
            heldout=heldout,
            model="reference_independence",
            feature_set="none",
            seed=-1,
            lam=0.0,
            projection_mode="none",
            p_true=p_true_test,
            q=p_eq_test,
            p_reference=p_eq_test,
            target_marginals=targets_test,
            n_train=int(train_mask.sum()),
            train_loss=None,
        )
        train_seed = _normalize(p_true[train_mask].mean(axis=0, keepdims=True))[0]
        ipf_rows = []
        for row_idx in range(int(heldout_mask.sum())):
            ipf_rows.append(
                _ipf_project_rows(
                    train_seed.reshape(1, -1),
                    [m[row_idx : row_idx + 1] for m in targets_test],
                    tuple(COARSE_SHAPE),
                    max_iter=int(args.ipf_iters),
                    tol=float(args.ipf_tol),
                )[0]
            )
        q_ipf = _normalize(np.vstack(ipf_rows))
        _append_eval_rows(
            rows=rows,
            by_puma_rows=by_puma_rows,
            target_keys=target_keys,
            heldout_mask=heldout_mask,
            heldout=heldout,
            model="ipf_train_seed",
            feature_set="none",
            seed=-1,
            lam=1.0,
            projection_mode="acs_ipf",
            p_true=p_true_test,
            q=q_ipf,
            p_reference=p_eq_test,
            target_marginals=targets_test,
            n_train=int(train_mask.sum()),
            train_loss=None,
        )

        for feature_set in feature_sets:
            if feature_set not in feature_map:
                raise SystemExit(f"unknown feature_set={feature_set}; available={sorted(feature_map)}")
            x = np.asarray(feature_map[feature_set], dtype=np.float64)
            x_scaler = StandardScaler()
            x_train_scaled = x_scaler.fit_transform(x[train_mask]).astype(np.float32)
            x_test_scaled = x_scaler.transform(x[heldout_mask]).astype(np.float32)

            for seed in seeds:
                fit = _train_hetero_prior(
                    x=x,
                    scores=scores_all,
                    train_mask=train_mask,
                    explained_ratio=explained,
                    seed=int(seed),
                    epochs=int(args.hetero_epochs),
                    hidden_dim=int(args.hetero_hidden_dim),
                    embed_dim=int(args.hetero_embed_dim),
                    lr=float(args.hetero_lr),
                    batch_size=int(args.batch_size),
                    device=str(device),
                )
                _append_prior_calibration_row(
                    calibration_rows,
                    heldout=heldout,
                    feature_set=feature_set,
                    seed=int(seed),
                    fit=fit,
                    n_pcs=n_pcs,
                    pca_explained=explained,
                )

                mean_test = fit.mean_std_all[heldout_mask]
                logvar_test = fit.logvar_std_all[heldout_mask]
                std_test = np.sqrt(np.exp(logvar_test))
                mean_train = fit.mean_std_all[train_mask]
                logvar_train = fit.logvar_std_all[train_mask]
                std_train = np.sqrt(np.exp(logvar_train))
                scores_std_all = ((scores_all - fit.score_mean.reshape(1, -1)) / fit.score_std.reshape(1, -1)).astype(np.float64)
                noise_train = (scores_std_all[train_mask] - mean_train) / np.clip(std_train, 1e-6, None)
                noise_train = np.clip(noise_train, -float(args.noise_clip), float(args.noise_clip)).astype(np.float32)

                cond_train = np.concatenate([x_train_scaled, mean_train.astype(np.float32), logvar_train.astype(np.float32)], axis=1)
                cond_test = np.concatenate([x_test_scaled, mean_test.astype(np.float32), logvar_test.astype(np.float32)], axis=1)

                model = DiffusionTabularModel(
                    input_dim=int(n_pcs),
                    cond_dim=int(cond_train.shape[1]),
                    seed=int(seed),
                    config=TabDDPMConfig(
                        timesteps=int(args.timesteps),
                        hidden_dims=_parse_hidden_dims(args.hidden_dims),
                        condition_injection=str(args.condition_injection),
                        film_hidden_dim=int(args.film_hidden_dim),
                        lr=float(args.diffusion_lr),
                        weight_decay=float(args.diffusion_weight_decay),
                        grad_clip=1.0,
                    ),
                )
                fit_summary = model.fit(
                    x=torch.tensor(noise_train, dtype=torch.float32),
                    cond=torch.tensor(cond_train, dtype=torch.float32),
                    epochs=int(args.diffusion_epochs),
                    batch_size=int(args.batch_size),
                    device=str(device),
                    log_every=0,
                )
                ckpt = out_dir / "checkpoints" / f"noise_diffusion_heldout{heldout}_{feature_set}_seed{seed}.pt"
                model.save(ckpt)

                # Mean-only residual correction.
                mean_draws_std = mean_test[:, None, :]
                mean_z_draws = _scores_std_to_z_draws(
                    pca=pca,
                    score_mean=fit.score_mean,
                    score_std=fit.score_std,
                    score_draws_std=mean_draws_std,
                )

                rng = np.random.default_rng(int(seed) + 7919)
                gaussian_noise = rng.normal(size=(int(heldout_mask.sum()), int(args.n_draws), int(n_pcs)))
                gaussian_draws_std = mean_test[:, None, :] + std_test[:, None, :] * gaussian_noise
                gaussian_z_draws = _scores_std_to_z_draws(
                    pca=pca,
                    score_mean=fit.score_mean,
                    score_std=fit.score_std,
                    score_draws_std=gaussian_draws_std,
                )

                cond_rep = np.repeat(cond_test, int(args.n_draws), axis=0)
                noise_draws = model.sample(
                    n=int(cond_rep.shape[0]),
                    cond=torch.tensor(cond_rep, dtype=torch.float32),
                    device=str(device),
                    sampler=str(args.sampler),
                    num_steps=int(args.sample_steps),
                    eta=0.0,
                ).numpy()
                noise_draws = noise_draws.reshape((int(heldout_mask.sum()), int(args.n_draws), int(n_pcs)))
                diffusion_draws_std = mean_test[:, None, :] + std_test[:, None, :] * noise_draws
                diffusion_z_draws = _scores_std_to_z_draws(
                    pca=pca,
                    score_mean=fit.score_mean,
                    score_std=fit.score_std,
                    score_draws_std=diffusion_draws_std,
                )

                draw_sets = [
                    ("hetero_prior_mean", mean_z_draws, fit.final_loss),
                    ("hetero_gaussian_prior_avg", gaussian_z_draws, fit.final_loss),
                    ("hetero_residual_noise_diffusion_avg", diffusion_z_draws, float(fit_summary["loss"])),
                ]
                for model_name, z_draws, train_loss in draw_sets:
                    for lam in lambdas:
                        for projection_mode in projection_modes:
                            q = _average_projected_draws(
                                p_eq=p_eq_test,
                                z_draws=z_draws,
                                lam=float(lam),
                                projection_mode=projection_mode,
                                target_marginals=targets_test,
                                ipf_iters=int(args.ipf_iters),
                                ipf_tol=float(args.ipf_tol),
                                clip_log_ratio=float(args.clip_log_ratio),
                            )
                            _append_eval_rows(
                                rows=rows,
                                by_puma_rows=by_puma_rows,
                                target_keys=target_keys,
                                heldout_mask=heldout_mask,
                                heldout=heldout,
                                model=model_name,
                                feature_set=feature_set,
                                seed=int(seed),
                                lam=float(lam),
                                projection_mode=projection_mode,
                                p_true=p_true_test,
                                q=q,
                                p_reference=p_eq_test,
                                target_marginals=targets_test,
                                n_train=int(train_mask.sum()),
                                train_loss=float(train_loss),
                            )

    metrics = pd.DataFrame(rows)
    by_puma = pd.DataFrame(by_puma_rows)
    calibration = pd.DataFrame(calibration_rows)
    metrics.to_csv(out_dir / "metrics" / "hetero_residual_prior_diffusion_tvd_long.csv", index=False)
    by_puma.to_csv(out_dir / "metrics" / "hetero_residual_prior_diffusion_by_puma.csv", index=False)
    calibration.to_csv(out_dir / "metrics" / "hetero_residual_prior_calibration.csv", index=False)
    summary = (
        metrics.groupby(["model", "feature_set", "seed", "lambda", "projection_mode"], as_index=False)
        .agg(
            tvd_mean=("tvd_mean", "mean"),
            tvd_std=("tvd_std", "mean"),
            cosine_mean=("cosine_mean", "mean"),
            mean_marginal_gap=("mean_marginal_gap", "mean"),
            delta_vs_reference=("delta_vs_reference", "mean"),
            relative_improvement_vs_reference=("relative_improvement_vs_reference", "mean"),
            train_loss=("train_loss", "mean"),
        )
        .sort_values("tvd_mean")
    )
    summary.to_csv(out_dir / "metrics" / "hetero_residual_prior_diffusion_summary.csv", index=False)

    run_summary = {
        "created_utc": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "question": "Can a heteroskedastic residual-PC prior precondition diffusion over coarse copula residual noise?",
        "target_wide_csv": str(args.target_wide_csv),
        "condition_csv": str(args.condition_csv),
        "spatial_csv": str(args.spatial_csv),
        "uncertainty_csv": str(args.uncertainty_csv),
        "uncertainty_loaded": bool(x_uncertainty is not None),
        "heldout_statefps": heldout_statefps,
        "feature_sets": feature_sets,
        "seeds": seeds,
        "lambdas": lambdas,
        "projection_modes": projection_modes,
        "joint_space": "coarse",
        "joint_k": int(COARSE_K),
        "coarse_shape": list(COARSE_SHAPE),
        "n_pcs": int(args.n_pcs),
        "hetero": {
            "epochs": int(args.hetero_epochs),
            "hidden_dim": int(args.hetero_hidden_dim),
            "embed_dim": int(args.hetero_embed_dim),
            "lr": float(args.hetero_lr),
        },
        "diffusion": {
            "epochs": int(args.diffusion_epochs),
            "timesteps": int(args.timesteps),
            "sample_steps": int(args.sample_steps),
            "n_draws": int(args.n_draws),
            "sampler": str(args.sampler),
            "hidden_dims": list(_parse_hidden_dims(args.hidden_dims)),
            "condition_injection": str(args.condition_injection),
            "lr": float(args.diffusion_lr),
            "weight_decay": float(args.diffusion_weight_decay),
            "noise_clip": float(args.noise_clip),
        },
        "device": str(device),
        "outputs": {
            "summary": str(out_dir / "metrics" / "hetero_residual_prior_diffusion_summary.csv"),
            "long": str(out_dir / "metrics" / "hetero_residual_prior_diffusion_tvd_long.csv"),
            "by_puma": str(out_dir / "metrics" / "hetero_residual_prior_diffusion_by_puma.csv"),
            "calibration": str(out_dir / "metrics" / "hetero_residual_prior_calibration.csv"),
            "checkpoints": str(out_dir / "checkpoints"),
        },
    }
    _write_json(out_dir / "run_summary.json", run_summary)
    print(summary.head(40).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
