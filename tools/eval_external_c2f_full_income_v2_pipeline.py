#!/usr/bin/env python3
from __future__ import annotations

"""
Evaluate a two-stage coarse-to-fine pipeline for the 5-way full-income v2 task.

Stage 1:
  shared-latent hierarchical diffusion model
  produces a 432-cell coarse distribution

Stage 2:
  teacher-forced diffusion model
  refines each coarse parent into a local fine split

This evaluator composes the two stages back into a full 5000-cell joint and
reports both raw and post-IPF performance on held-out Michigan PUMAs.
"""

import argparse
import datetime as _dt
import json
import pathlib
import random
import sys
from typing import Any

import numpy as np
import pandas as pd


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.train_external_joint_hier_diffusion_full_income as _full_income  # noqa: F401
import tools.train_external_joint_hier_diffusion_full as stage1_base
from tools.external_c2f_full_earn_stage2_model import load_stage2_model, sample_stage2_local_raw_batch
from tools.external_c2f_full_income_v2_schema import (
    AGE_FINE_TO_COARSE,
    COARSE_K,
    COARSE_SHAPE,
    FULL_K,
    FULL_SHAPE,
    FULL_VARIABLE_ORDER,
    INCOME_FINE_TO_COARSE,
    MAX_CHILDREN,
    PADDED_PARENT_CHILD_FULL,
    PADDED_PARENT_CHILD_INCOME_REGIME,
    PARENT_CHILD_SLOT_MASK,
    SCHL_FINE_TO_COARSE,
    ESR_FINE_TO_COARSE,
    coarse_from_full_flat,
)
from tools.train_external_c2f_full_earn_teacher import _project_to_child_mask, _uniform_from_child_mask
from tools.train_us_puma_5var_diffusion import (
    _canon_puma5,
    _canon_statefp,
    _canon_uid,
    _cosine,
    _ipf_nd,
    _require_torch,
    _summ,
    _tvd,
    _utc_now_iso,
    _write_json,
)
from tools.train_us_puma_external_v1_diffusion import (
    _load_condition_specs_from_schema,
    _load_external_condition_matrix,
)


def _load_stage1_model(*, checkpoint_path: pathlib.Path, timesteps: int, seed: int) -> tuple[Any, dict[str, Any]]:
    torch = _require_torch()
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit(f"Unsupported stage1 checkpoint format: {checkpoint_path}")
    ckpt_format = str(payload.get("format", ""))
    if ckpt_format == "synthpop.external_joint_hier_diffusion_full.v0":
        fine_shape = tuple(int(x) for x in payload.get("fine_shape", FULL_SHAPE))
        input_dim = int(payload.get("active_fine_dim", int(np.prod(fine_shape))))
        diffusion_cfg = stage1_base.TabDDPMConfig(
            timesteps=int(timesteps),
            hidden_dims=tuple(int(x) for x in payload["diffusion_hidden_dims"]),
            condition_injection=str(payload.get("condition_injection", "concat")),
            film_hidden_dim=int(payload.get("film_hidden_dim", 128)),
        )
        model = stage1_base.SharedLatentHierarchicalDiffusion(
            input_dim=int(input_dim),
            cond_raw_dim=int(payload["cond_raw_dim"]),
            latent_dim=int(payload["latent_dim"]),
            encoder_hidden_dims=tuple(int(x) for x in payload["encoder_hidden_dims"]),
            coarse_hidden_dims=tuple(int(x) for x in payload["coarse_hidden_dims"]),
            diffusion_config=diffusion_cfg,
            seed=int(seed),
        )
        model._modules.load_state_dict(payload["state_dict"], strict=True)
        return model, payload
    raise SystemExit(f"Unsupported stage1 checkpoint format: {checkpoint_path}")


def _load_full_joint_wide(*, joint_wide_csv: pathlib.Path, schema_json: pathlib.Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    schema = json.loads(schema_json.read_text(encoding="utf-8"))
    shape = tuple(int(x) for x in schema["shape"])
    if tuple(shape) != FULL_SHAPE:
        raise SystemExit(f"Unexpected full target shape: got {shape}, expected {FULL_SHAPE}")

    df = pd.read_csv(joint_wide_csv, low_memory=False)
    req = {"statefp", "puma", "puma_uid"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"joint_wide_csv missing columns: {miss}")

    p_joint_cols = [f"p_joint_{i:03d}" for i in range(FULL_K)]
    miss_joint = [c for c in p_joint_cols if c not in df.columns]
    if miss_joint:
        raise SystemExit(f"joint_wide_csv missing joint columns: {miss_joint[:5]}")

    df["statefp"] = df["statefp"].map(_canon_statefp)
    df["puma5"] = df["puma"].map(_canon_puma5)
    df["puma_uid"] = df.apply(lambda r: _canon_uid(r["statefp"], r["puma5"]), axis=1)
    p_joint = df[p_joint_cols].to_numpy(dtype=np.float32)
    p_joint = np.clip(p_joint, 0.0, None)
    p_joint = p_joint / np.maximum(p_joint.sum(axis=1, keepdims=True), 1e-12)
    ids = df["puma_uid"].astype(str).tolist()
    return df, p_joint, ids


def _compute_stage2_scaler(*, wide_csv: pathlib.Path, schema_json: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    schema = json.loads(schema_json.read_text(encoding="utf-8"))
    target_dim = int(schema["target_dim"])
    p_joint_cols = [f"p_joint_{i:03d}" for i in range(target_dim)]

    df = pd.read_csv(wide_csv, low_memory=False)
    df["statefp"] = df["statefp"].map(_canon_statefp)
    is_mi = (df["statefp"] == "26").to_numpy(dtype=bool)
    train_idx = np.where(~is_mi)[0]
    if train_idx.size == 0:
        raise SystemExit("No non-Michigan rows found for stage2 scaler.")

    p_joint = df[p_joint_cols].to_numpy(dtype=np.float32)
    p_joint = np.clip(p_joint, 0.0, None)
    p_joint = p_joint / np.maximum(p_joint.sum(axis=1, keepdims=True), 1e-12)
    x_log = np.log(np.clip(p_joint, 0.0, None) + 1e-6).astype(np.float32)
    x_train = x_log[train_idx]
    x_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
    return x_mean, x_std


def _load_stage2_structure(*, schema_json: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    schema = json.loads(schema_json.read_text(encoding="utf-8"))
    padded = schema.get("parent_to_child_full_indices_padded")
    mask = schema.get("parent_child_slot_mask")
    regime = schema.get("parent_child_income_regime_padded")
    if padded is None or mask is None:
        return (
            PADDED_PARENT_CHILD_FULL.astype(np.int32),
            PARENT_CHILD_SLOT_MASK.astype(np.float32),
            PADDED_PARENT_CHILD_INCOME_REGIME.astype(np.int16)
            if regime is None
            else np.asarray(regime, dtype=np.int16),
        )

    padded_arr = np.asarray(padded, dtype=np.int32)
    mask_arr = np.asarray(mask, dtype=np.float32)
    if padded_arr.shape != PADDED_PARENT_CHILD_FULL.shape:
        raise SystemExit(
            f"Unexpected parent_to_child_full_indices_padded shape: got {tuple(padded_arr.shape)}, expected {tuple(PADDED_PARENT_CHILD_FULL.shape)}"
        )
    if mask_arr.shape != PARENT_CHILD_SLOT_MASK.shape:
        raise SystemExit(
            f"Unexpected parent_child_slot_mask shape: got {tuple(mask_arr.shape)}, expected {tuple(PARENT_CHILD_SLOT_MASK.shape)}"
        )
    if regime is None:
        regime_arr = PADDED_PARENT_CHILD_INCOME_REGIME.astype(np.int16)
    else:
        regime_arr = np.asarray(regime, dtype=np.int16)
        if regime_arr.shape != PADDED_PARENT_CHILD_INCOME_REGIME.shape:
            raise SystemExit(
                "Unexpected parent_child_income_regime_padded shape: "
                f"got {tuple(regime_arr.shape)}, expected {tuple(PADDED_PARENT_CHILD_INCOME_REGIME.shape)}"
            )
    return padded_arr, mask_arr, regime_arr


def _sample_stage2_local_batch(
    *,
    model: Any,
    cond_rows: np.ndarray,
    child_mask_rows: np.ndarray,
    regime_index_rows: np.ndarray | None,
    n_draws: int,
    device: str | None,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> np.ndarray:
    return sample_stage2_local_raw_batch(
        model=model,
        cond_rows=cond_rows,
        child_mask_rows=child_mask_rows,
        regime_index_rows=regime_index_rows,
        n_draws=int(n_draws),
        device=device,
        x_mean=x_mean,
        x_std=x_std,
    ).astype(np.float64)


def _aggregate_full_marginal_to_coarse(*, p_full: np.ndarray, map_idx: np.ndarray, coarse_dim: int) -> np.ndarray:
    out = np.bincount(np.asarray(map_idx, dtype=np.int64), weights=np.asarray(p_full, dtype=np.float64), minlength=int(coarse_dim)).astype(np.float64)
    out = out / max(float(out.sum()), 1e-12)
    return out


def _coarse_marginals_from_full_ext(ext_row: dict[str, np.ndarray]) -> list[np.ndarray]:
    return [
        _aggregate_full_marginal_to_coarse(
            p_full=np.asarray(ext_row["AGEP_bin"], dtype=np.float64),
            map_idx=AGE_FINE_TO_COARSE,
            coarse_dim=COARSE_SHAPE[0],
        ),
        np.asarray(ext_row["SEX"], dtype=np.float64),
        _aggregate_full_marginal_to_coarse(
            p_full=np.asarray(ext_row["SCHL_allpop"], dtype=np.float64),
            map_idx=SCHL_FINE_TO_COARSE,
            coarse_dim=COARSE_SHAPE[2],
        ),
        _aggregate_full_marginal_to_coarse(
            p_full=np.asarray(ext_row["ESR_allpop"], dtype=np.float64),
            map_idx=ESR_FINE_TO_COARSE,
            coarse_dim=COARSE_SHAPE[3],
        ),
        _aggregate_full_marginal_to_coarse(
            p_full=np.asarray(ext_row["PINCP_allpop_bin"], dtype=np.float64),
            map_idx=INCOME_FINE_TO_COARSE,
            coarse_dim=COARSE_SHAPE[4],
        ),
    ]


def _compose_stage2_conditions(*, coarse_prob: np.ndarray, parent_child_slot_mask: np.ndarray) -> np.ndarray:
    coarse_prob = np.asarray(coarse_prob, dtype=np.float64).reshape(-1)
    if coarse_prob.shape[0] != COARSE_K:
        raise ValueError(f"Unexpected coarse length={coarse_prob.shape[0]}, expected={COARSE_K}")

    parent_eye = np.eye(COARSE_K, dtype=np.float64)
    coarse_rep = np.repeat(coarse_prob.reshape(1, -1), repeats=COARSE_K, axis=0)
    child_mask = np.asarray(parent_child_slot_mask, dtype=np.float64)
    parent_mass = coarse_prob.reshape((-1, 1)).astype(np.float64)
    cond_rows = np.concatenate([coarse_rep, parent_eye, child_mask, parent_mass], axis=1)
    return cond_rows.astype(np.float32)


def _combine_from_coarse(
    *,
    stage2_model: Any,
    coarse_prob: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    n_draws: int,
    device: str | None,
    parent_child_slot_mask: np.ndarray,
    padded_parent_child_full: np.ndarray,
    parent_child_income_regime: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    coarse_prob = np.asarray(coarse_prob, dtype=np.float64).reshape(-1)
    cond_rows = _compose_stage2_conditions(
        coarse_prob=coarse_prob,
        parent_child_slot_mask=parent_child_slot_mask,
    )
    local_raw = _sample_stage2_local_batch(
        model=stage2_model,
        cond_rows=cond_rows,
        child_mask_rows=np.asarray(parent_child_slot_mask, dtype=np.float32),
        regime_index_rows=None if parent_child_income_regime is None else np.asarray(parent_child_income_regime, dtype=np.int16),
        n_draws=int(n_draws),
        device=device,
        x_mean=x_mean,
        x_std=x_std,
    )

    out_diff = np.zeros((FULL_K,), dtype=np.float64)
    out_uniform = np.zeros((FULL_K,), dtype=np.float64)
    for parent_idx in range(COARSE_K):
        parent_mass = float(coarse_prob[parent_idx])
        if parent_mass <= 0.0:
            continue
        child_slots = np.asarray(padded_parent_child_full[parent_idx], dtype=np.int32)
        child_slots = child_slots[child_slots >= 0]
        mask = np.asarray(parent_child_slot_mask[parent_idx], dtype=np.float64)
        p_local = _project_to_child_mask(local_raw[parent_idx], mask)
        p_uniform = _uniform_from_child_mask(mask)
        out_diff[child_slots] = p_local[: child_slots.shape[0]] * parent_mass
        out_uniform[child_slots] = p_uniform[: child_slots.shape[0]] * parent_mass

    out_diff = out_diff / max(float(out_diff.sum()), 1e-12)
    out_uniform = out_uniform / max(float(out_uniform.sum()), 1e-12)
    return out_diff, out_uniform


def _run_full_ipf(*, seed_joint: np.ndarray, ext_row: dict[str, np.ndarray], ipf_iters: int) -> np.ndarray:
    target_marginals = [np.asarray(ext_row[var], dtype=np.float64) for var in FULL_VARIABLE_ORDER]
    out = _ipf_nd(
        seed_joint=np.asarray(seed_joint, dtype=np.float64).reshape(FULL_SHAPE),
        target_marginals=target_marginals,
        shape=FULL_SHAPE,
        max_iter=int(ipf_iters),
    )
    out = out / max(float(out.sum()), 1e-12)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="eval_external_c2f_full_income_v2_pipeline")
    ap.add_argument("--stage1_joint_wide_csv", required=True)
    ap.add_argument("--stage1_schema_json", required=True)
    ap.add_argument("--stage1_condition_csv", required=True)
    ap.add_argument("--stage1_condition_schema_json", default=None)
    ap.add_argument("--stage1_checkpoint", required=True)
    ap.add_argument("--stage1_timesteps", type=int, default=200)
    ap.add_argument("--stage2_wide_csv", required=True)
    ap.add_argument("--stage2_schema_json", required=True)
    ap.add_argument("--stage2_checkpoint", required=True)
    ap.add_argument("--one_shot_summary_json", default=None)
    ap.add_argument("--stage2_n_eval_joint_samples", type=int, default=64)
    ap.add_argument("--ipf_iters", type=int, default=200)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    stage1_joint_csv = pathlib.Path(args.stage1_joint_wide_csv).expanduser().resolve()
    stage1_schema_json = pathlib.Path(args.stage1_schema_json).expanduser().resolve()
    stage1_condition_csv = pathlib.Path(args.stage1_condition_csv).expanduser().resolve()
    stage1_condition_schema_json = pathlib.Path(args.stage1_condition_schema_json).expanduser().resolve() if args.stage1_condition_schema_json else None
    stage1_checkpoint = pathlib.Path(args.stage1_checkpoint).expanduser().resolve()
    stage2_wide_csv = pathlib.Path(args.stage2_wide_csv).expanduser().resolve()
    stage2_schema_json = pathlib.Path(args.stage2_schema_json).expanduser().resolve()
    stage2_checkpoint = pathlib.Path(args.stage2_checkpoint).expanduser().resolve()
    one_shot_summary = pathlib.Path(args.one_shot_summary_json).expanduser().resolve() if args.one_shot_summary_json else None

    for p in [
        stage1_joint_csv,
        stage1_schema_json,
        stage1_condition_csv,
        stage1_checkpoint,
        stage2_wide_csv,
        stage2_schema_json,
        stage2_checkpoint,
    ]:
        if not p.exists():
            raise SystemExit(f"Required path not found: {p}")
    if stage1_condition_schema_json is not None and not stage1_condition_schema_json.exists():
        raise SystemExit(f"Required path not found: {stage1_condition_schema_json}")
    if one_shot_summary is not None and not one_shot_summary.exists():
        raise SystemExit(f"one_shot_summary_json not found: {one_shot_summary}")

    run_id = f"_us_puma_external_c2f_full_income_v2_eval_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else (_REPO_ROOT / "outputs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    df, p_true_all, ids = _load_full_joint_wide(joint_wide_csv=stage1_joint_csv, schema_json=stage1_schema_json)
    is_mi = (df["statefp"] == "26").to_numpy(dtype=bool)
    mi_idx = np.where(is_mi)[0]
    if mi_idx.size == 0:
        raise SystemExit("No Michigan PUMAs found in stage1_joint_wide_csv.")

    stage1_var_specs = stage1_base._load_var_specs_from_schema(schema_json=stage1_schema_json)
    cond_specs = _load_condition_specs_from_schema(
        condition_schema_json=stage1_condition_schema_json,
        fallback_var_specs=stage1_var_specs,
    )
    cond_raw, block_slices, _ = _load_external_condition_matrix(
        condition_csv=stage1_condition_csv,
        ids=ids,
        var_specs=cond_specs,
    )
    ext_marg = {var: cond_raw[:, sl].copy() for var, sl in block_slices.items()}
    ext_marg = stage1_base._augment_ext_marginals_from_cross(cond_raw=cond_raw, block_slices=block_slices, ext_marg=ext_marg)

    stage1_model, stage1_payload = _load_stage1_model(
        checkpoint_path=stage1_checkpoint,
        timesteps=int(args.stage1_timesteps),
        seed=int(args.seed),
    )

    torch = _require_torch()
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)
    stage1_model.to(device)

    stage2_x_mean, stage2_x_std = _compute_stage2_scaler(wide_csv=stage2_wide_csv, schema_json=stage2_schema_json)
    padded_parent_child_full, parent_child_slot_mask, parent_child_income_regime = _load_stage2_structure(schema_json=stage2_schema_json)
    stage2_model, stage2_payload = load_stage2_model(checkpoint_path=stage2_checkpoint)

    cond_mi_t = torch.from_numpy(cond_raw[mi_idx]).to(device=device, dtype=torch.float32)
    coarse_pred_raw = stage1_model.predict_coarse(cond_raw=cond_mi_t).detach().cpu().numpy().astype(np.float64)

    tvd_coarse_raw: list[float] = []
    tvd_coarse_ipf: list[float] = []
    tvd_pipeline_raw_raw: list[float] = []
    tvd_pipeline_raw_ipf: list[float] = []
    tvd_pipeline_proj_raw: list[float] = []
    tvd_pipeline_proj_ipf: list[float] = []
    tvd_uniform_proj_raw: list[float] = []
    tvd_uniform_proj_ipf: list[float] = []
    tvd_oracle_raw: list[float] = []
    tvd_oracle_ipf: list[float] = []
    cosine_pipeline_proj_ipf: list[float] = []
    cosine_oracle_ipf: list[float] = []

    for local_pos, idx in enumerate(mi_idx.tolist()):
        p_true = np.asarray(p_true_all[idx], dtype=np.float64)
        p_coarse_true = coarse_from_full_flat(p_true)
        ext_row = {var: np.asarray(ext_marg[var][idx], dtype=np.float64) for var in FULL_VARIABLE_ORDER}
        coarse_targets = _coarse_marginals_from_full_ext(ext_row)

        p_coarse_raw = coarse_pred_raw[local_pos]
        p_coarse_raw = p_coarse_raw / max(float(p_coarse_raw.sum()), 1e-12)
        p_coarse_proj = _ipf_nd(
            seed_joint=p_coarse_raw.reshape(COARSE_SHAPE),
            target_marginals=coarse_targets,
            shape=COARSE_SHAPE,
            max_iter=int(args.ipf_iters),
        )
        p_coarse_proj = p_coarse_proj / max(float(p_coarse_proj.sum()), 1e-12)

        p_full_from_raw, p_uniform_from_raw = _combine_from_coarse(
            stage2_model=stage2_model,
            coarse_prob=p_coarse_raw,
            x_mean=stage2_x_mean,
            x_std=stage2_x_std,
            n_draws=int(args.stage2_n_eval_joint_samples),
            device=device,
            parent_child_slot_mask=parent_child_slot_mask,
            padded_parent_child_full=padded_parent_child_full,
            parent_child_income_regime=parent_child_income_regime,
        )
        p_full_from_proj, p_uniform_from_proj = _combine_from_coarse(
            stage2_model=stage2_model,
            coarse_prob=p_coarse_proj,
            x_mean=stage2_x_mean,
            x_std=stage2_x_std,
            n_draws=int(args.stage2_n_eval_joint_samples),
            device=device,
            parent_child_slot_mask=parent_child_slot_mask,
            padded_parent_child_full=padded_parent_child_full,
            parent_child_income_regime=parent_child_income_regime,
        )
        p_full_oracle, _ = _combine_from_coarse(
            stage2_model=stage2_model,
            coarse_prob=p_coarse_true,
            x_mean=stage2_x_mean,
            x_std=stage2_x_std,
            n_draws=int(args.stage2_n_eval_joint_samples),
            device=device,
            parent_child_slot_mask=parent_child_slot_mask,
            padded_parent_child_full=padded_parent_child_full,
            parent_child_income_regime=parent_child_income_regime,
        )

        p_full_from_raw_ipf = _run_full_ipf(seed_joint=p_full_from_raw, ext_row=ext_row, ipf_iters=int(args.ipf_iters))
        p_full_from_proj_ipf = _run_full_ipf(seed_joint=p_full_from_proj, ext_row=ext_row, ipf_iters=int(args.ipf_iters))
        p_uniform_from_proj_ipf = _run_full_ipf(seed_joint=p_uniform_from_proj, ext_row=ext_row, ipf_iters=int(args.ipf_iters))
        p_full_oracle_ipf = _run_full_ipf(seed_joint=p_full_oracle, ext_row=ext_row, ipf_iters=int(args.ipf_iters))

        tvd_coarse_raw.append(_tvd(p_coarse_raw, p_coarse_true))
        tvd_coarse_ipf.append(_tvd(p_coarse_proj, p_coarse_true))
        tvd_pipeline_raw_raw.append(_tvd(p_full_from_raw, p_true))
        tvd_pipeline_raw_ipf.append(_tvd(p_full_from_raw_ipf, p_true))
        tvd_pipeline_proj_raw.append(_tvd(p_full_from_proj, p_true))
        tvd_pipeline_proj_ipf.append(_tvd(p_full_from_proj_ipf, p_true))
        tvd_uniform_proj_raw.append(_tvd(p_uniform_from_proj, p_true))
        tvd_uniform_proj_ipf.append(_tvd(p_uniform_from_proj_ipf, p_true))
        tvd_oracle_raw.append(_tvd(p_full_oracle, p_true))
        tvd_oracle_ipf.append(_tvd(p_full_oracle_ipf, p_true))
        cosine_pipeline_proj_ipf.append(_cosine(p_full_from_proj_ipf, p_true))
        cosine_oracle_ipf.append(_cosine(p_full_oracle_ipf, p_true))

    results: dict[str, Any] = {
        "stage1_coarse": {
            "tvd_raw": _summ(tvd_coarse_raw),
            "tvd_ipf": _summ(tvd_coarse_ipf),
        },
        "pipeline_stage1_raw": {
            "tvd_joint_raw": _summ(tvd_pipeline_raw_raw),
            "tvd_joint": _summ(tvd_pipeline_raw_ipf),
        },
        "pipeline_stage1_coarse_ipf": {
            "tvd_joint_raw": _summ(tvd_pipeline_proj_raw),
            "tvd_joint": _summ(tvd_pipeline_proj_ipf),
            "cosine_joint": _summ(cosine_pipeline_proj_ipf),
        },
        "uniform_refine_with_stage1_coarse_ipf": {
            "tvd_joint_raw": _summ(tvd_uniform_proj_raw),
            "tvd_joint": _summ(tvd_uniform_proj_ipf),
        },
        "oracle_stage2_true_coarse": {
            "tvd_joint_raw": _summ(tvd_oracle_raw),
            "tvd_joint": _summ(tvd_oracle_ipf),
            "cosine_joint": _summ(cosine_oracle_ipf),
        },
    }
    if one_shot_summary is not None:
        results["references"] = json.loads(one_shot_summary.read_text(encoding="utf-8"))

    run_summary = {
        "created_utc": _utc_now_iso(),
        "stage1_joint_wide_csv": str(stage1_joint_csv),
        "stage1_schema_json": str(stage1_schema_json),
        "stage1_condition_csv": str(stage1_condition_csv),
        "stage1_condition_schema_json": str(stage1_condition_schema_json) if stage1_condition_schema_json else None,
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage1_checkpoint_meta": {
            "format": str(stage1_payload.get("format", "")),
            "cond_raw_dim": int(stage1_payload["cond_raw_dim"]),
            "latent_dim": int(stage1_payload["latent_dim"]),
            "condition_injection": str(stage1_payload["condition_injection"]),
            "support_mask_mode": str(stage1_payload.get("support_mask_mode", "none")),
            "input_dim": int(
                stage1_payload.get(
                    "input_dim",
                    stage1_payload.get(
                        "active_fine_dim",
                        int(np.prod(tuple(int(x) for x in stage1_payload.get("fine_shape", FULL_SHAPE)))),
                    ),
                )
            ),
        },
        "stage2_wide_csv": str(stage2_wide_csv),
        "stage2_schema_json": str(stage2_schema_json),
        "stage2_checkpoint": str(stage2_checkpoint),
        "stage2_checkpoint_meta": {
            "format": str(stage2_payload.get("format", "")),
            "predict_mode": str(stage2_payload.get("predict_mode", "diffusion")),
            "blend_alpha": float(stage2_payload.get("blend_alpha", 0.0)),
        },
        "n_mi_pumas": int(mi_idx.size),
        "stage2_n_eval_joint_samples": int(args.stage2_n_eval_joint_samples),
        "ipf_iters": int(args.ipf_iters),
        "device": str(device),
        "seed": int(args.seed),
        "results": results,
    }

    _write_json(out_dir / "run_summary.json", run_summary)
    _write_json(out_dir / "metrics" / "coarse_to_fine_summary.json", results)
    print(f"[ok] wrote: {out_dir}")


if __name__ == "__main__":
    main()
