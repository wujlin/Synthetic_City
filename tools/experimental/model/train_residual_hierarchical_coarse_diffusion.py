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
    _load_acs_conditions,
    _load_spatial,
    _load_target,
    _mean_marginal_gap,
    _normalize,
    _tvd,
    _write_json,
)
from tools.experimental.representation.ssl_copula_residual_probe import _aggregate_full_to_coarse, _outer_joint


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


def _aggregate_marginal(values: np.ndarray, mapping: np.ndarray, out_dim: int) -> np.ndarray:
    values = _normalize(values)
    out = np.zeros((values.shape[0], int(out_dim)), dtype=np.float64)
    for fine_idx, coarse_idx in enumerate(np.asarray(mapping, dtype=int).tolist()):
        out[:, int(coarse_idx)] += values[:, int(fine_idx)]
    return _normalize(out)


def _coarse_constraint_marginals(acs_marginals: dict[str, np.ndarray]) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    from tools.model.external_c2f_full_earn_schema import (
        AGE_FINE_TO_COARSE,
        EARN_FINE_TO_COARSE,
        ESR_FINE_TO_COARSE,
        SCHL_FINE_TO_COARSE,
    )

    ordered = [
        _aggregate_marginal(acs_marginals["AGEP_bin"], AGE_FINE_TO_COARSE, COARSE_SHAPE[0]),
        acs_marginals["SEX"],
        _aggregate_marginal(acs_marginals["SCHL_allpop"], SCHL_FINE_TO_COARSE, COARSE_SHAPE[2]),
        _aggregate_marginal(acs_marginals["ESR_allpop"], ESR_FINE_TO_COARSE, COARSE_SHAPE[3]),
        _aggregate_marginal(acs_marginals["EARN_16p_bin"], EARN_FINE_TO_COARSE, COARSE_SHAPE[4]),
    ]
    named = {
        "AGEP_bin": ordered[0],
        "SEX": ordered[1],
        "SCHL_allpop": ordered[2],
        "ESR_allpop": ordered[3],
        "EARN_16p_bin": ordered[4],
    }
    return ordered, named


def _average_projected_draws(
    *,
    p_eq: np.ndarray,
    z_draws: np.ndarray,
    lam: float,
    projection_mode: str,
    target_marginals: list[np.ndarray],
    ipf_iters: int,
    ipf_tol: float,
    clip_log_ratio: float,
) -> np.ndarray:
    n_test, n_draws, k = z_draws.shape
    flat_eq = np.repeat(np.asarray(p_eq, dtype=np.float64), int(n_draws), axis=0)
    flat_z = z_draws.reshape((n_test * n_draws, k))
    q = _exp_tilt(flat_eq, flat_z, lam=float(lam), clip=float(clip_log_ratio))
    if projection_mode == "acs_ipf":
        flat_targets = [np.repeat(np.asarray(m, dtype=np.float64), int(n_draws), axis=0) for m in target_marginals]
        q = _ipf_project_rows(
            q,
            flat_targets,
            tuple(COARSE_SHAPE),
            max_iter=int(ipf_iters),
            tol=float(ipf_tol),
        )
    elif projection_mode != "none":
        raise ValueError(f"unsupported projection_mode={projection_mode}")
    q = q.reshape((n_test, n_draws, k)).mean(axis=1)
    return _normalize(q)


def _append_eval_rows(
    *,
    rows: list[dict[str, Any]],
    by_puma_rows: list[dict[str, Any]],
    target_keys: pd.DataFrame,
    heldout_mask: np.ndarray,
    heldout: str,
    model: str,
    feature_set: str,
    seed: int,
    lam: float,
    projection_mode: str,
    p_true: np.ndarray,
    q: np.ndarray,
    p_reference: np.ndarray,
    target_marginals: list[np.ndarray],
    n_train: int,
    train_loss: float | None,
) -> None:
    tvd = _tvd(p_true, q)
    ref_tvd = _tvd(p_true, p_reference)
    cosine = _cosine(p_true, q)
    gap = _mean_marginal_gap(q, target_marginals, tuple(COARSE_SHAPE))
    delta = ref_tvd - tvd
    rows.append(
        {
            "heldout_statefp": heldout,
            "model": model,
            "feature_set": feature_set,
            "seed": int(seed),
            "lambda": float(lam),
            "projection_mode": projection_mode,
            "n_train": int(n_train),
            "n_test": int(p_true.shape[0]),
            "train_loss": None if train_loss is None else float(train_loss),
            "tvd_mean": float(np.mean(tvd)),
            "tvd_std": float(np.std(tvd)),
            "cosine_mean": float(np.mean(cosine)),
            "mean_marginal_gap": float(np.mean(gap)),
            "delta_vs_reference": float(np.mean(delta)),
            "relative_improvement_vs_reference": float(np.mean(delta) / max(float(np.mean(ref_tvd)), 1e-12)),
        }
    )
    meta = target_keys.loc[heldout_mask, ["puma_uid_key", "statefp", "puma5"]].reset_index(drop=True)
    for i in range(p_true.shape[0]):
        by_puma_rows.append(
            {
                "puma_uid": str(meta.loc[i, "puma_uid_key"]).zfill(7),
                "statefp": str(meta.loc[i, "statefp"]).zfill(2),
                "puma5": str(meta.loc[i, "puma5"]).zfill(5),
                "heldout_statefp": heldout,
                "model": model,
                "feature_set": feature_set,
                "seed": int(seed),
                "lambda": float(lam),
                "projection_mode": projection_mode,
                "tvd": float(tvd[i]),
                "reference_tvd": float(ref_tvd[i]),
                "delta_vs_reference": float(delta[i]),
                "cosine": float(cosine[i]),
                "mean_marginal_gap": float(gap[i]),
            }
        )


def main() -> int:
    data_root = pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data")
    ap = argparse.ArgumentParser(description="Train coarse residual diffusion over log-ratio residuals.")
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
    ap.add_argument("--heldout_statefps", default="26")
    ap.add_argument("--feature_sets", default="acs_1d")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lambdas", default="1.0")
    ap.add_argument("--projection_modes", default="acs_ipf")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--clip_log_ratio", type=float, default=8.0)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--n_draws", type=int, default=64)
    ap.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    ap.add_argument("--hidden_dims", default="512,512")
    ap.add_argument("--condition_injection", choices=["concat", "film"], default="film")
    ap.add_argument("--film_hidden_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--ipf_iters", type=int, default=300)
    ap.add_argument("--ipf_tol", type=float, default=1e-9)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    torch = _require_torch()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir or pathlib.Path(f"outputs/_residual_hier_coarse_diffusion_{_utc_ts()}")
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    target_keys, p_true_full, _, _target_marginals = _load_target(args.target_wide_csv)
    _acs_feature_df, acs_marginals, x_acs_1d, x_acs_all, _, _ = _load_acs_conditions(args.condition_csv, target_keys)
    x_spatial, _spatial_cols = _load_spatial(args.spatial_csv, target_keys)
    p_true = _aggregate_full_to_coarse(p_true_full)
    coarse_targets, _coarse_targets_named = _coarse_constraint_marginals(acs_marginals)
    p_eq = _normalize(_outer_joint(coarse_targets))
    z = np.log(p_true + float(args.eps)) - np.log(p_eq + float(args.eps))

    feature_map = {
        "acs_1d": x_acs_1d,
        "acs_all": x_acs_all,
        "acs_all_spatial": np.concatenate([x_acs_all, x_spatial], axis=1),
    }
    feature_sets = [x.strip() for x in str(args.feature_sets).split(",") if x.strip()]
    projection_modes = [x.strip() for x in str(args.projection_modes).split(",") if x.strip()]
    heldout_statefps = [str(x).zfill(2) for x in _parse_ints(args.heldout_statefps)]
    seeds = _parse_ints(args.seeds)
    lambdas = _parse_floats(args.lambdas)
    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()

    rows: list[dict[str, Any]] = []
    by_puma_rows: list[dict[str, Any]] = []

    for heldout in heldout_statefps:
        train_mask = statefp != heldout
        heldout_mask = ~train_mask
        if int(heldout_mask.sum()) == 0:
            continue
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
                raise SystemExit(f"unknown feature_set: {feature_set}")
            x = np.asarray(feature_map[feature_set], dtype=np.float64)
            x_scaler = StandardScaler()
            cond_train = x_scaler.fit_transform(x[train_mask]).astype(np.float32)
            cond_test = x_scaler.transform(x[heldout_mask]).astype(np.float32)

            z_scaler = StandardScaler()
            z_train = z_scaler.fit_transform(z[train_mask]).astype(np.float32)

            for seed in seeds:
                model = DiffusionTabularModel(
                    input_dim=int(COARSE_K),
                    cond_dim=int(cond_train.shape[1]),
                    seed=int(seed),
                    config=TabDDPMConfig(
                        timesteps=int(args.timesteps),
                        hidden_dims=_parse_hidden_dims(args.hidden_dims),
                        condition_injection=str(args.condition_injection),
                        film_hidden_dim=int(args.film_hidden_dim),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                        grad_clip=1.0,
                    ),
                )
                fit_summary = model.fit(
                    x=torch.tensor(z_train, dtype=torch.float32),
                    cond=torch.tensor(cond_train, dtype=torch.float32),
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    device=str(device),
                    log_every=0,
                )
                cond_rep = np.repeat(cond_test, int(args.n_draws), axis=0)
                z_draws_std = model.sample(
                    n=int(cond_rep.shape[0]),
                    cond=torch.tensor(cond_rep, dtype=torch.float32),
                    device=str(device),
                    sampler=str(args.sampler),
                    num_steps=int(args.sample_steps),
                    eta=0.0,
                ).numpy()
                z_draws = z_scaler.inverse_transform(z_draws_std).reshape(
                    (int(heldout_mask.sum()), int(args.n_draws), int(COARSE_K))
                )
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
                            model="coarse_residual_diffusion",
                            feature_set=feature_set,
                            seed=int(seed),
                            lam=float(lam),
                            projection_mode=projection_mode,
                            p_true=p_true_test,
                            q=q,
                            p_reference=p_eq_test,
                            target_marginals=targets_test,
                            n_train=int(train_mask.sum()),
                            train_loss=float(fit_summary["loss"]),
                        )

    metrics = pd.DataFrame(rows)
    by_puma = pd.DataFrame(by_puma_rows)
    metrics.to_csv(out_dir / "metrics" / "coarse_residual_diffusion_tvd_long.csv", index=False)
    by_puma.to_csv(out_dir / "metrics" / "coarse_residual_diffusion_by_puma.csv", index=False)
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
    summary.to_csv(out_dir / "metrics" / "coarse_residual_diffusion_summary.csv", index=False)

    run_summary = {
        "created_utc": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "question": "Does diffusion over coarse log-ratio residuals improve projected coarse-joint TVD?",
        "target_wide_csv": str(args.target_wide_csv),
        "condition_csv": str(args.condition_csv),
        "spatial_csv": str(args.spatial_csv),
        "heldout_statefps": heldout_statefps,
        "feature_sets": feature_sets,
        "seeds": seeds,
        "lambdas": lambdas,
        "projection_modes": projection_modes,
        "joint_space": "coarse",
        "joint_k": int(COARSE_K),
        "epochs": int(args.epochs),
        "timesteps": int(args.timesteps),
        "sample_steps": int(args.sample_steps),
        "n_draws": int(args.n_draws),
        "sampler": str(args.sampler),
        "hidden_dims": list(_parse_hidden_dims(args.hidden_dims)),
        "condition_injection": str(args.condition_injection),
        "device": str(device),
        "outputs": {
            "summary": str(out_dir / "metrics" / "coarse_residual_diffusion_summary.csv"),
            "long": str(out_dir / "metrics" / "coarse_residual_diffusion_tvd_long.csv"),
            "by_puma": str(out_dir / "metrics" / "coarse_residual_diffusion_by_puma.csv"),
        },
    }
    _write_json(out_dir / "run_summary.json", run_summary)
    print(summary.head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
