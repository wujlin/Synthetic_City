#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import shutil
import sys
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.model.train_external_joint_hier_diffusion_full as one_shot_base
import tools.model.train_external_joint_hier_diffusion_full_earn as _full_earn  # noqa: F401
import tools.model.train_external_joint_hier_diffusion_full as stage1_base
from tools.model.eval_external_c2f_full_earn_pipeline import (
    _coarse_marginals_from_full_ext,
    _combine_from_coarse,
    _compute_stage2_scaler,
    _load_full_joint_wide,
    _load_stage1_model,
    _run_full_ipf,
)
from tools.model.train_us_puma_5var_diffusion import (
    _canon_puma5,
    _canon_statefp,
    _canon_uid,
    _ipf_nd,
    _require_torch,
    _softmax_rows,
    _tvd,
)
from tools.model.train_us_puma_external_v1_diffusion import _load_condition_specs_from_schema, _load_external_condition_matrix


FULL_VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop", "EARN_16p_bin"]


@dataclass(frozen=True)
class PipelineConfig:
    label: str
    stage1_joint_wide_csv: str
    stage1_schema_json: str
    stage1_condition_csv: str
    stage1_condition_schema_json: str
    stage1_checkpoint: str
    stage2_wide_csv: str
    stage2_schema_json: str
    stage2_checkpoint: str
    stage1_timesteps: int
    stage2_n_eval_joint_samples: int
    ipf_iters: int
    seed: int


@dataclass(frozen=True)
class OneShotConfig:
    label: str
    joint_wide_csv: str
    schema_json: str
    condition_csv: str
    condition_schema_json: str
    checkpoint: str
    timesteps: int
    n_eval_joint_samples: int
    ipf_iters: int
    logp_clip_quantile_low: float
    logp_clip_quantile_high: float
    support_mask_mode: str
    support_mask_eps: float
    seed: int


PIPELINE_CONFIGS: list[PipelineConfig] = [
    PipelineConfig(
        label="seed0",
        stage1_joint_wide_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv",
        stage1_schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json",
        stage1_condition_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_conditions/extcond_v1_earn_v1_acs5_2022_puma_us.csv",
        stage1_condition_schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json",
        stage1_checkpoint="/home/jinlin/projects/Synthetic_City/outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep3000_seed0_20260326T162906Z/checkpoints/leave_mi_out/best.pt",
        stage2_wide_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_c2f/mainline_gate50_selcoarse_ep3000_seed0_stage1ipfcondmix/extc2f_full_earn_stage1ipfcondmix_pums_2023_puma_us_wide.csv",
        stage2_schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_c2f/mainline_gate50_selcoarse_ep3000_seed0_stage1ipfcondmix/extc2f_full_earn_stage1ipfcondmix_pums_2023_puma_us.schema.json",
        stage2_checkpoint="/home/jinlin/projects/Synthetic_City/outputs/_us_puma_external_c2f_full_earn_teacher_stage1ipfcondmix_mainline_gate50_selcoarse_ep3000_seed0_maskw_a05_cleanheadw1_cons05_gate50_blend25_bestsel50_20260327T023511Z/checkpoints/external_c2f_full_earn_teacher/leave_mi_out/best.pt",
        stage1_timesteps=200,
        stage2_n_eval_joint_samples=64,
        ipf_iters=200,
        seed=0,
    ),
    PipelineConfig(
        label="seed1",
        stage1_joint_wide_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv",
        stage1_schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json",
        stage1_condition_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_conditions/extcond_v1_earn_v1_acs5_2022_puma_us.csv",
        stage1_condition_schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json",
        stage1_checkpoint="/home/jinlin/projects/Synthetic_City/outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep3000_seed1_20260326T163214Z/checkpoints/leave_mi_out/best.pt",
        stage2_wide_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_c2f/mainline_gate50_selcoarse_ep3000_seed1_stage1ipfcondmix/extc2f_full_earn_stage1ipfcondmix_pums_2023_puma_us_wide.csv",
        stage2_schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_c2f/mainline_gate50_selcoarse_ep3000_seed1_stage1ipfcondmix/extc2f_full_earn_stage1ipfcondmix_pums_2023_puma_us.schema.json",
        stage2_checkpoint="/home/jinlin/projects/Synthetic_City/outputs/_us_puma_external_c2f_full_earn_teacher_stage1ipfcondmix_mainline_gate50_selcoarse_ep3000_seed1_maskw_a05_cleanheadw1_cons05_gate50_blend25_bestsel50_20260327T040359Z/checkpoints/external_c2f_full_earn_teacher/leave_mi_out/best.pt",
        stage1_timesteps=200,
        stage2_n_eval_joint_samples=64,
        ipf_iters=200,
        seed=1,
    ),
    PipelineConfig(
        label="seed2",
        stage1_joint_wide_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv",
        stage1_schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json",
        stage1_condition_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_conditions/extcond_v1_earn_v1_acs5_2022_puma_us.csv",
        stage1_condition_schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json",
        stage1_checkpoint="/home/jinlin/projects/Synthetic_City/outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep3600snap200_seed2_20260327T003137Z/checkpoints/leave_mi_out/epoch_3200.pt",
        stage2_wide_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_c2f/extc2f_full_earn_stage1ipfcondmix_pums_2023_puma_us_wide.csv",
        stage2_schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_c2f/extc2f_full_earn_stage1ipfcondmix_pums_2023_puma_us.schema.json",
        stage2_checkpoint="/home/jinlin/projects/Synthetic_City/outputs/_us_puma_external_c2f_full_earn_teacher_stage1ipfcondmix_ep3200_seed2_maskw_a05_cleanheadw1_cons05_gate50_blend25_bestsel50_20260327T040359Z/checkpoints/external_c2f_full_earn_teacher/leave_mi_out/best.pt",
        stage1_timesteps=200,
        stage2_n_eval_joint_samples=64,
        ipf_iters=200,
        seed=2,
    ),
]


ONE_SHOT_CONFIGS: list[OneShotConfig] = [
    OneShotConfig(
        label="seed0",
        joint_wide_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv",
        schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json",
        condition_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_conditions/extcond_v1_earn_v1_acs5_2022_puma_us.csv",
        condition_schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json",
        checkpoint="/home/jinlin/projects/Synthetic_City/outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep3000_seed0_20260326T162906Z/checkpoints/leave_mi_out/best.pt",
        timesteps=200,
        n_eval_joint_samples=32,
        ipf_iters=200,
        logp_clip_quantile_low=0.001,
        logp_clip_quantile_high=0.999,
        support_mask_mode="none",
        support_mask_eps=1e-12,
        seed=0,
    ),
    OneShotConfig(
        label="seed1",
        joint_wide_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv",
        schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json",
        condition_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_conditions/extcond_v1_earn_v1_acs5_2022_puma_us.csv",
        condition_schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json",
        checkpoint="/home/jinlin/projects/Synthetic_City/outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep3000_seed1_20260326T163214Z/checkpoints/leave_mi_out/best.pt",
        timesteps=200,
        n_eval_joint_samples=32,
        ipf_iters=200,
        logp_clip_quantile_low=0.001,
        logp_clip_quantile_high=0.999,
        support_mask_mode="none",
        support_mask_eps=1e-12,
        seed=1,
    ),
    OneShotConfig(
        label="seed2",
        joint_wide_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv",
        schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json",
        condition_csv="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_conditions/extcond_v1_earn_v1_acs5_2022_puma_us.csv",
        condition_schema_json="/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us.schema.json",
        checkpoint="/home/jinlin/projects/Synthetic_City/outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep3000_seed2_20260326T163526Z/checkpoints/leave_mi_out/best.pt",
        timesteps=200,
        n_eval_joint_samples=32,
        ipf_iters=200,
        logp_clip_quantile_low=0.001,
        logp_clip_quantile_high=0.999,
        support_mask_mode="none",
        support_mask_eps=1e-12,
        seed=2,
    ),
]


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch = _require_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_ext_inputs(*, condition_csv: pathlib.Path, condition_schema_json: pathlib.Path, ids: list[str], stage1_schema_json: pathlib.Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    stage1_var_specs = stage1_base._load_var_specs_from_schema(schema_json=stage1_schema_json)
    cond_specs = _load_condition_specs_from_schema(
        condition_schema_json=condition_schema_json,
        fallback_var_specs=stage1_var_specs,
    )
    cond_raw, block_slices, _ = _load_external_condition_matrix(
        condition_csv=condition_csv,
        ids=ids,
        var_specs=cond_specs,
    )
    ext_marg = {var: cond_raw[:, sl].copy() for var, sl in block_slices.items()}
    ext_marg = stage1_base._augment_ext_marginals_from_cross(cond_raw=cond_raw, block_slices=block_slices, ext_marg=ext_marg)
    return cond_raw, ext_marg


def _build_ipf_baseline(*, p_true_all: np.ndarray, ext_marg: dict[str, np.ndarray], mi_idx: np.ndarray, train_idx: np.ndarray, ipf_iters: int) -> np.ndarray:
    train_seed = np.asarray(p_true_all[train_idx], dtype=np.float64).mean(axis=0)
    train_seed = train_seed / max(float(train_seed.sum()), 1e-12)
    out = np.zeros((mi_idx.size,), dtype=np.float64)
    for j, idx in enumerate(mi_idx.tolist()):
        marginals_ext = [np.asarray(ext_marg[var][idx], dtype=np.float64) for var in FULL_VARIABLE_ORDER]
        p_ipf = _ipf_nd(
            seed_joint=train_seed.reshape(one_shot_base.FINE_SHAPE),
            target_marginals=marginals_ext,
            shape=one_shot_base.FINE_SHAPE,
            max_iter=int(ipf_iters),
        ).reshape(-1)
        p_ipf = p_ipf / max(float(p_ipf.sum()), 1e-12)
        out[j] = _tvd(p_ipf, np.asarray(p_true_all[idx], dtype=np.float64))
    return out


def _compute_one_shot_seed_metrics(cfg: OneShotConfig) -> pd.DataFrame:
    _set_all_seeds(cfg.seed)
    joint_wide_csv = pathlib.Path(cfg.joint_wide_csv).expanduser().resolve()
    schema_json = pathlib.Path(cfg.schema_json).expanduser().resolve()
    condition_csv = pathlib.Path(cfg.condition_csv).expanduser().resolve()
    condition_schema_json = pathlib.Path(cfg.condition_schema_json).expanduser().resolve()
    checkpoint = pathlib.Path(cfg.checkpoint).expanduser().resolve()

    df, p_true_all, ids = _load_full_joint_wide(joint_wide_csv=joint_wide_csv, schema_json=schema_json)
    cond_raw, ext_marg = _load_ext_inputs(
        condition_csv=condition_csv,
        condition_schema_json=condition_schema_json,
        ids=ids,
        stage1_schema_json=schema_json,
    )

    is_mi = (df["statefp"] == "26").to_numpy(dtype=bool)
    mi_idx = np.where(is_mi)[0]
    train_idx = np.where(~is_mi)[0]

    if str(cfg.support_mask_mode).lower().strip() == "dataset_nonzero":
        active_cols = np.where((p_true_all > float(cfg.support_mask_eps)).any(axis=0))[0].astype(np.int64)
    else:
        active_cols = np.arange(p_true_all.shape[1], dtype=np.int64)

    p_fine = p_true_all[:, active_cols].astype(np.float32)
    p_fine = p_fine / np.maximum(p_fine.sum(axis=1, keepdims=True), 1e-12)
    x_log_all = np.log(np.clip(p_fine, 0.0, None) + 1e-6).astype(np.float32)
    x_train_log = x_log_all[train_idx]
    x_mean = x_train_log.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = x_train_log.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)

    if 0.0 <= float(cfg.logp_clip_quantile_low) < float(cfg.logp_clip_quantile_high) <= 1.0:
        logp_clip_lo = np.quantile(x_train_log, float(cfg.logp_clip_quantile_low), axis=0).astype(np.float32)
        logp_clip_hi = np.quantile(x_train_log, float(cfg.logp_clip_quantile_high), axis=0).astype(np.float32)
    else:
        logp_clip_lo = None
        logp_clip_hi = None

    torch = _require_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = _load_stage1_model(checkpoint_path=checkpoint, timesteps=int(cfg.timesteps), seed=int(cfg.seed))
    model.to(device)

    cond_eval_t = torch.from_numpy(cond_raw[mi_idx]).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        z_eval = model.encoder(cond_eval_t)
    x_samples = model.sample_latent_conditioned(
        z_cond=z_eval,
        n_draws=int(cfg.n_eval_joint_samples),
        device=device,
    ).numpy()

    logp = x_samples.astype(np.float64) * x_std.reshape(1, 1, -1).astype(np.float64) + x_mean.reshape(1, 1, -1).astype(np.float64)
    if logp_clip_lo is not None and logp_clip_hi is not None:
        lo = logp_clip_lo.reshape(1, 1, -1).astype(np.float64)
        hi = logp_clip_hi.reshape(1, 1, -1).astype(np.float64)
        logp = np.clip(logp, lo, hi)
    p_draws = np.asarray([_softmax_rows(logp[i]) for i in range(logp.shape[0])], dtype=np.float64)
    p_hat_raw = p_draws.mean(axis=1)
    if active_cols.size != p_true_all.shape[1]:
        p_hat_full = one_shot_base._expand_active_prob_np(
            p_active=p_hat_raw,
            active_cols=active_cols,
            full_dim=int(p_true_all.shape[1]),
        )
    else:
        p_hat_full = p_hat_raw
    p_hat_full = p_hat_full / np.maximum(p_hat_full.sum(axis=1, keepdims=True), 1e-12)

    ipf_vals = _build_ipf_baseline(
        p_true_all=p_true_all,
        ext_marg=ext_marg,
        mi_idx=mi_idx,
        train_idx=train_idx,
        ipf_iters=int(cfg.ipf_iters),
    )

    rows: list[dict[str, Any]] = []
    for j, idx in enumerate(mi_idx.tolist()):
        ext_row = {var: np.asarray(ext_marg[var][idx], dtype=np.float64) for var in FULL_VARIABLE_ORDER}
        p_eval = _run_full_ipf(seed_joint=p_hat_full[j], ext_row=ext_row, ipf_iters=int(cfg.ipf_iters))
        rows.append(
            {
                "puma_uid": str(df.iloc[idx]["puma_uid"]),
                "statefp": str(df.iloc[idx]["statefp"]),
                "puma5": str(df.iloc[idx]["puma5"]),
                f"one_shot_tvd_{cfg.label}": float(_tvd(p_eval, np.asarray(p_true_all[idx], dtype=np.float64))),
                "ipf_tvd": float(ipf_vals[j]),
            }
        )
    return pd.DataFrame(rows)


def _compute_pipeline_seed_metrics(cfg: PipelineConfig) -> pd.DataFrame:
    _set_all_seeds(cfg.seed)
    stage1_joint_wide_csv = pathlib.Path(cfg.stage1_joint_wide_csv).expanduser().resolve()
    stage1_schema_json = pathlib.Path(cfg.stage1_schema_json).expanduser().resolve()
    stage1_condition_csv = pathlib.Path(cfg.stage1_condition_csv).expanduser().resolve()
    stage1_condition_schema_json = pathlib.Path(cfg.stage1_condition_schema_json).expanduser().resolve()
    stage1_checkpoint = pathlib.Path(cfg.stage1_checkpoint).expanduser().resolve()
    stage2_wide_csv = pathlib.Path(cfg.stage2_wide_csv).expanduser().resolve()
    stage2_schema_json = pathlib.Path(cfg.stage2_schema_json).expanduser().resolve()
    stage2_checkpoint = pathlib.Path(cfg.stage2_checkpoint).expanduser().resolve()

    df, p_true_all, ids = _load_full_joint_wide(joint_wide_csv=stage1_joint_wide_csv, schema_json=stage1_schema_json)
    cond_raw, ext_marg = _load_ext_inputs(
        condition_csv=stage1_condition_csv,
        condition_schema_json=stage1_condition_schema_json,
        ids=ids,
        stage1_schema_json=stage1_schema_json,
    )
    is_mi = (df["statefp"] == "26").to_numpy(dtype=bool)
    mi_idx = np.where(is_mi)[0]

    torch = _require_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stage1_model, _ = _load_stage1_model(
        checkpoint_path=stage1_checkpoint,
        timesteps=int(cfg.stage1_timesteps),
        seed=int(cfg.seed),
    )
    stage1_model.to(device)

    stage2_x_mean, stage2_x_std = _compute_stage2_scaler(
        wide_csv=stage2_wide_csv,
        schema_json=stage2_schema_json,
    )
    from tools.model.external_c2f_full_earn_stage2_model import load_stage2_model

    stage2_model, _ = load_stage2_model(checkpoint_path=stage2_checkpoint)

    cond_mi_t = torch.from_numpy(cond_raw[mi_idx]).to(device=device, dtype=torch.float32)
    coarse_pred_raw = stage1_model.predict_coarse(cond_raw=cond_mi_t).detach().cpu().numpy().astype(np.float64)

    rows: list[dict[str, Any]] = []
    for local_pos, idx in enumerate(mi_idx.tolist()):
        p_true = np.asarray(p_true_all[idx], dtype=np.float64)
        ext_row = {var: np.asarray(ext_marg[var][idx], dtype=np.float64) for var in FULL_VARIABLE_ORDER}
        coarse_targets = _coarse_marginals_from_full_ext(ext_row)

        p_coarse_raw = coarse_pred_raw[local_pos]
        p_coarse_raw = p_coarse_raw / max(float(p_coarse_raw.sum()), 1e-12)
        p_coarse_proj = _ipf_nd(
            seed_joint=p_coarse_raw.reshape(_full_earn.base.COARSE_SHAPE),
            target_marginals=coarse_targets,
            shape=_full_earn.base.COARSE_SHAPE,
            max_iter=int(cfg.ipf_iters),
        )
        p_coarse_proj = p_coarse_proj / max(float(p_coarse_proj.sum()), 1e-12)

        p_full_from_proj, _ = _combine_from_coarse(
            stage2_model=stage2_model,
            coarse_prob=p_coarse_proj,
            x_mean=stage2_x_mean,
            x_std=stage2_x_std,
            n_draws=int(cfg.stage2_n_eval_joint_samples),
            device=device,
        )
        p_full_from_proj_ipf = _run_full_ipf(
            seed_joint=p_full_from_proj,
            ext_row=ext_row,
            ipf_iters=int(cfg.ipf_iters),
        )
        rows.append(
            {
                "puma_uid": str(df.iloc[idx]["puma_uid"]),
                "statefp": str(df.iloc[idx]["statefp"]),
                "puma5": str(df.iloc[idx]["puma5"]),
                f"pipeline_tvd_{cfg.label}": float(_tvd(p_full_from_proj_ipf, p_true)),
            }
        )
    return pd.DataFrame(rows)


def _load_heterogeneity(path: pathlib.Path) -> pd.DataFrame:
    obj = json.loads(path.read_text(encoding="utf-8"))
    mapping = obj["tvd_to_global"]
    rows = []
    for puma, val in mapping.items():
        rows.append(
            {
                "puma_uid": _canon_uid("26", puma),
                "statefp": "26",
                "puma5": _canon_puma5(puma),
                "heterogeneity_tvd": float(val),
            }
        )
    return pd.DataFrame(rows)


def _build_metric_table(*, heterogeneity_json: pathlib.Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    hetero_df = _load_heterogeneity(heterogeneity_json)

    pipeline_df = hetero_df.copy()
    for cfg in PIPELINE_CONFIGS:
        pipeline_seed_df = _compute_pipeline_seed_metrics(cfg)
        pipeline_df = pipeline_df.merge(pipeline_seed_df, on=["puma_uid", "statefp", "puma5"], how="left")

    one_shot_df = hetero_df[["puma_uid", "statefp", "puma5"]].copy()
    ipf_added = False
    for cfg in ONE_SHOT_CONFIGS:
        one_seed_df = _compute_one_shot_seed_metrics(cfg)
        cols = ["puma_uid", "statefp", "puma5", f"one_shot_tvd_{cfg.label}"]
        if not ipf_added:
            cols.append("ipf_tvd")
            ipf_added = True
        one_shot_df = one_shot_df.merge(one_seed_df[cols], on=["puma_uid", "statefp", "puma5"], how="left")

    df = pipeline_df.merge(one_shot_df, on=["puma_uid", "statefp", "puma5"], how="left")
    pipeline_cols = [f"pipeline_tvd_{cfg.label}" for cfg in PIPELINE_CONFIGS]
    one_shot_cols = [f"one_shot_tvd_{cfg.label}" for cfg in ONE_SHOT_CONFIGS]
    df["pipeline_tvd_mean"] = df[pipeline_cols].mean(axis=1)
    df["pipeline_tvd_std"] = df[pipeline_cols].std(axis=1, ddof=0)
    df["one_shot_tvd_mean"] = df[one_shot_cols].mean(axis=1)
    df["one_shot_tvd_std"] = df[one_shot_cols].std(axis=1, ddof=0)
    df["delta_tvd_ipf_minus_pipeline"] = df["ipf_tvd"] - df["pipeline_tvd_mean"]
    df["delta_tvd_ipf_minus_oneshot"] = df["ipf_tvd"] - df["one_shot_tvd_mean"]

    spearman = pd.DataFrame(
        {
            "heterogeneity": df["heterogeneity_tvd"],
            "gain": df["delta_tvd_ipf_minus_pipeline"],
        }
    ).corr(method="spearman").iloc[0, 1]
    payload = {
        "n_pumas": int(df.shape[0]),
        "pipeline_mean": float(df["pipeline_tvd_mean"].mean()),
        "one_shot_mean": float(df["one_shot_tvd_mean"].mean()),
        "ipf_mean": float(df["ipf_tvd"].mean()),
        "pipeline_beats_ipf_n": int((df["delta_tvd_ipf_minus_pipeline"] > 0).sum()),
        "one_shot_beats_ipf_n": int((df["delta_tvd_ipf_minus_oneshot"] > 0).sum()),
        "spearman_gain_vs_heterogeneity": float(spearman),
    }
    return df.sort_values("puma_uid").reset_index(drop=True), payload


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return x, y
    slope, intercept = np.polyfit(x, y, deg=1)
    xx = np.linspace(float(x.min()), float(x.max()), 200)
    yy = slope * xx + intercept
    return xx, yy


def _plot_panels(*, df: pd.DataFrame, shapefile: pathlib.Path, out_png: pathlib.Path) -> None:
    import geopandas as gpd
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    df = df.copy()
    df["puma_uid"] = df["puma_uid"].astype(str)
    df["statefp"] = df["statefp"].astype(str)
    df["puma5"] = df["puma5"].astype(str).map(_canon_puma5)

    gdf = gpd.read_file(shapefile)
    gdf["statefp"] = gdf["STATEFP20"].map(_canon_statefp)
    puma_col = "PUMACE20" if "PUMACE20" in gdf.columns else "PUMA20"
    gdf["puma5"] = gdf[puma_col].map(_canon_puma5)
    gdf["puma_uid"] = gdf.apply(lambda r: _canon_uid(r["statefp"], r["puma5"]), axis=1)
    gdf = gdf.loc[gdf["statefp"] == "26"].copy()
    gdf = gdf.merge(df, on=["puma_uid", "statefp", "puma5"], how="inner")

    clay = "#b07243"
    blue = "#5b88b2"
    orange = "#d99b5d"
    grey = "#8e8e8e"

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.0))
    ax_map, ax_scatter, ax_rank, ax_gain = axes.ravel()

    vals = gdf["delta_tvd_ipf_minus_pipeline"].to_numpy(dtype=float)
    vmax = max(abs(float(np.nanmin(vals))), abs(float(np.nanmax(vals))), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    gdf.plot(
        column="delta_tvd_ipf_minus_pipeline",
        cmap="BrBG",
        linewidth=0.35,
        edgecolor="#f4f4f4",
        legend=True,
        legend_kwds={"label": "TVD gain over IPF", "shrink": 0.82},
        ax=ax_map,
        norm=norm,
    )
    cbar_ax = next((ax for ax in fig.axes if ax not in {ax_map, ax_scatter, ax_rank, ax_gain}), None)
    if cbar_ax is not None:
        cbar_ax.tick_params(labelsize=10)
        cbar_ax.set_ylabel("TVD gain over IPF", fontsize=11)
    ax_map.set_axis_off()

    x_ipf = df["ipf_tvd"].to_numpy(dtype=float)
    y_pipe = df["pipeline_tvd_mean"].to_numpy(dtype=float)
    y_one = df["one_shot_tvd_mean"].to_numpy(dtype=float)
    lo = min(float(np.min(x_ipf)), float(np.min(y_pipe)), float(np.min(y_one)))
    hi = max(float(np.max(x_ipf)), float(np.max(y_pipe)), float(np.max(y_one)))
    ax_scatter.scatter(x_ipf, y_one, s=28, facecolors="none", edgecolors=orange, linewidths=1.0, alpha=0.9, label="One-shot")
    ax_scatter.scatter(x_ipf, y_pipe, s=28, color=blue, alpha=0.85, label="Pipeline")
    ax_scatter.plot([lo, hi], [lo, hi], linestyle="--", color=grey, linewidth=1.0)
    ax_scatter.set_xlabel("IPF TVD")
    ax_scatter.set_ylabel("Model TVD")
    ax_scatter.legend(frameon=False, loc="upper left")
    rank = np.arange(1, df.shape[0] + 1)
    ax_rank.plot(rank, np.sort(df["pipeline_tvd_mean"].to_numpy(dtype=float)), color=blue, linewidth=2.0, label="Pipeline")
    ax_rank.plot(
        rank,
        np.sort(df["one_shot_tvd_mean"].to_numpy(dtype=float)),
        color=orange,
        linewidth=2.0,
        linestyle="--",
        marker="o",
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.0,
        markevery=6,
        label="One-shot",
    )
    ax_rank.plot(rank, np.sort(df["ipf_tvd"].to_numpy(dtype=float)), color=clay, linewidth=2.0, label="IPF")
    ax_rank.set_xlabel("PUMA rank")
    ax_rank.set_ylabel("TVD")
    ax_rank.legend(frameon=False, loc="upper left")

    pairwise_csv = out_png.parent / "michigan_pairwise_tvd_summary.csv"
    pair_df = pd.read_csv(pairwise_csv)
    pair_labels = {
        "AGEP_bin__SEX": "Age-Sex",
        "AGEP_bin__SCHL_allpop": "Age-Edu",
        "AGEP_bin__ESR_allpop": "Age-Emp",
        "AGEP_bin__EARN_16p_bin": "Age-Inc",
        "SEX__SCHL_allpop": "Sex-Edu",
        "SEX__ESR_allpop": "Sex-Emp",
        "SEX__EARN_16p_bin": "Sex-Inc",
        "SCHL_allpop__ESR_allpop": "Edu-Emp",
        "SCHL_allpop__EARN_16p_bin": "Edu-Inc",
        "ESR_allpop__EARN_16p_bin": "Emp-Inc",
    }
    x = np.arange(pair_df.shape[0], dtype=float)
    width = 0.38
    ax_gain.bar(x - width / 2, pair_df["pipeline_tvd_mean"], width=width, color=blue, alpha=0.9, label="Pipeline")
    ax_gain.bar(x + width / 2, pair_df["ipf_tvd_mean"], width=width, color=clay, alpha=0.85, label="IPF")
    ax_gain.set_xticks(x)
    ax_gain.set_xticklabels([pair_labels.get(v, v) for v in pair_df["pair"]], rotation=35, ha="right")
    ax_gain.set_ylabel("Mean pairwise TVD")
    ax_gain.legend(frameon=False, loc="upper right")
    ymax = float(max(pair_df["pipeline_tvd_mean"].max(), pair_df["ipf_tvd_mean"].max()))
    ax_gain.set_ylim(0.0, ymax + 0.004)

    fig.tight_layout()
    pos_map = ax_map.get_position()
    pos_scatter = ax_scatter.get_position()
    pos_rank = ax_rank.get_position()
    pos_gain = ax_gain.get_position()
    x_left = min(pos_map.x0, pos_rank.x0) - 0.018
    x_right = min(pos_scatter.x0, pos_gain.x0) - 0.018
    fig.text(x_left, pos_map.y1 + 0.008, "a", fontsize=17, fontweight="bold", ha="left", va="bottom")
    fig.text(x_right, pos_scatter.y1 + 0.008, "b", fontsize=17, fontweight="bold", ha="left", va="bottom")
    fig.text(x_left, pos_rank.y1 + 0.008, "c", fontsize=17, fontweight="bold", ha="left", va="bottom")
    fig.text(x_right, pos_gain.y1 + 0.008, "d", fontsize=17, fontweight="bold", ha="left", va="bottom")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heterogeneity_json", default=str(REPO_ROOT / "outputs" / "_exp0_copula_mi_20260209T151039Z" / "tvd_to_global.json"))
    ap.add_argument("--shapefile", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cached_dir", default="")
    args = ap.parse_args()

    heterogeneity_json = pathlib.Path(args.heterogeneity_json).expanduser().resolve()
    shapefile = pathlib.Path(args.shapefile).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(args.cached_dir).strip():
        cached_dir = pathlib.Path(args.cached_dir).expanduser().resolve()
        df = pd.read_csv(cached_dir / "michigan_regional_validation_by_puma.csv")
        summary = json.loads((cached_dir / "michigan_regional_validation_summary.json").read_text(encoding="utf-8"))
        shutil.copy2(cached_dir / "michigan_pairwise_tvd_summary.csv", out_dir / "michigan_pairwise_tvd_summary.csv")
        (out_dir / "michigan_regional_validation_by_puma.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        (out_dir / "michigan_regional_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    else:
        df, summary = _build_metric_table(heterogeneity_json=heterogeneity_json)
        (out_dir / "michigan_regional_validation_by_puma.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        (out_dir / "michigan_regional_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _plot_panels(
        df=df,
        shapefile=shapefile,
        out_png=out_dir / "fig_04_michigan_regional_validation.png",
    )


if __name__ == "__main__":
    main()
