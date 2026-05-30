#!/usr/bin/env python3
from __future__ import annotations

"""
Evaluate a two-stage coarse-to-fine pipeline for the age x education refinement task.

Stage 1:
  external v1-lite diffusion on
    AGEP_lite(4) x SEX(2) x SCHL_lite(3) x ESR_lite(3)

Stage 2:
  teacher-forced diffusion refining
    AGEP_fine(10) x SCHL_fine(5)
  within each (SEX, ESR_lite) subgroup.

This evaluator composes the two stages into a full K=300 prediction:
  AGEP_fine(10) x SEX(2) x SCHL_fine(5) x ESR_lite(3)

It is intended as the first end-to-end probe of the coarse-to-fine idea.
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


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.model.diffusion_tabular import DiffusionTabularModel, TabDDPMConfig
from tools.model.build_external_c2f_age_schl_teacher import (
    COARSE_SHAPE,
    ESR_LITE_LABELS,
    FINE_SHAPE,
    SEX_LABELS,
    _aggregate_coarse_age_schl,
)
from tools.data.external_v1_variant_presets import AGE_LITE_LABELS, SCHL_LITE_LABELS
from tools.model.train_external_c2f_age_schl_teacher import _project_to_parent_table, _uniform_from_parent_table
from tools.model.train_us_puma_5var_diffusion import (
    _canon_puma5,
    _canon_statefp,
    _canon_uid,
    _cosine,
    _ipf_nd,
    _parse_hidden_dims,
    _require_torch,
    _softmax_rows,
    _summ,
    _tvd,
    _utc_now_iso,
    _write_json,
)
from tools.model.train_us_puma_external_v1_diffusion import _load_external_condition_matrix, _load_var_specs_from_schema


STAGE1_SHAPE = (len(AGE_LITE_LABELS), len(SEX_LABELS), len(SCHL_LITE_LABELS), len(ESR_LITE_LABELS))
STAGE1_K = int(np.prod(STAGE1_SHAPE))
STAGE2_K = int(np.prod(FINE_SHAPE))
FINAL_SHAPE = (FINE_SHAPE[0], len(SEX_LABELS), FINE_SHAPE[1], len(ESR_LITE_LABELS))
FINAL_K = int(np.prod(FINAL_SHAPE))


def _load_joint_wide(
    *,
    joint_wide_csv: pathlib.Path,
    schema_json: pathlib.Path,
) -> tuple[pd.DataFrame, np.ndarray, list[str], tuple[int, ...]]:
    schema = json.loads(schema_json.read_text(encoding="utf-8"))
    shape = tuple(int(x) for x in schema["shape"])
    K = int(np.prod(shape))

    df = pd.read_csv(joint_wide_csv, low_memory=False)
    req = {"statefp", "puma", "puma_uid"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"joint_wide_csv missing columns: {miss}")

    df["statefp"] = df["statefp"].map(_canon_statefp)
    df["puma5"] = df["puma"].map(_canon_puma5)
    df["puma_uid"] = df.apply(lambda r: _canon_uid(r["statefp"], r["puma5"]), axis=1)
    p_joint_cols = [f"p_joint_{i:03d}" for i in range(K)]
    miss_joint = [c for c in p_joint_cols if c not in df.columns]
    if miss_joint:
        raise SystemExit(f"joint_wide_csv missing joint columns: {miss_joint[:5]}")

    p_joint = df[p_joint_cols].to_numpy(dtype=np.float32)
    p_joint = np.clip(p_joint, 0.0, None)
    p_joint = p_joint / np.maximum(p_joint.sum(axis=1, keepdims=True), 1e-12)
    ids = df["puma_uid"].astype(str).tolist()
    return df, p_joint, ids, shape


def _compute_train_scaler(x_log_all: np.ndarray, *, is_mi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.where(~is_mi)[0]
    if train_idx.size == 0:
        raise SystemExit("No non-Michigan rows found for training scaler.")
    x_train_log = x_log_all[train_idx]
    x_mean = x_train_log.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = x_train_log.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
    return x_mean, x_std


def _sample_mean_prob(
    *,
    model: DiffusionTabularModel,
    cond_row: np.ndarray | None,
    n_draws: int,
    device: str | None,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> np.ndarray:
    torch = _require_torch()
    if cond_row is None:
        z = model.sample(n=int(n_draws), cond=None, device=device).numpy()
    else:
        c = np.repeat(cond_row.reshape(1, -1), repeats=int(n_draws), axis=0).astype(np.float32)
        z = model.sample(n=int(n_draws), cond=torch.from_numpy(c), device=device).numpy()
    logp = z.astype(np.float64) * x_std.reshape(1, -1).astype(np.float64) + x_mean.reshape(1, -1).astype(np.float64)
    p_draws = _softmax_rows(logp)
    p_hat = np.mean(p_draws, axis=0)
    p_hat = p_hat / max(float(p_hat.sum()), 1e-12)
    return p_hat.astype(np.float64)


def _compose_stage2_condition(*, parent_cond: np.ndarray, sex_idx: int, esr_idx: int, subgroup_mass: float) -> np.ndarray:
    sex = np.zeros((len(SEX_LABELS),), dtype=np.float32)
    sex[sex_idx] = 1.0
    esr = np.zeros((len(ESR_LITE_LABELS),), dtype=np.float32)
    esr[esr_idx] = 1.0
    vec = np.concatenate(
        [
            np.asarray(parent_cond, dtype=np.float32).reshape(-1),
            sex,
            esr,
            np.asarray([float(subgroup_mass)], dtype=np.float32),
        ],
        axis=0,
    )
    return vec.astype(np.float32)


def _combine_c2f_prediction(
    *,
    stage1_joint: np.ndarray,
    stage2_model: DiffusionTabularModel,
    stage2_x_mean: np.ndarray,
    stage2_x_std: np.ndarray,
    stage2_n_draws: int,
    device: str | None,
    child_parent_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coarse = np.asarray(stage1_joint, dtype=np.float64).reshape(STAGE1_SHAPE)
    out_diff = np.zeros(FINAL_SHAPE, dtype=np.float64)
    out_uniform = np.zeros(FINAL_SHAPE, dtype=np.float64)

    for sex_idx in range(len(SEX_LABELS)):
        for esr_idx in range(len(ESR_LITE_LABELS)):
            coarse_block = coarse[:, sex_idx, :, esr_idx]
            subgroup_mass = float(coarse_block.sum())
            if subgroup_mass <= 0:
                continue
            parent_cond = coarse_block / subgroup_mass
            cond_vec = _compose_stage2_condition(
                parent_cond=parent_cond,
                sex_idx=sex_idx,
                esr_idx=esr_idx,
                subgroup_mass=subgroup_mass,
            )
            fine_raw = _sample_mean_prob(
                model=stage2_model,
                cond_row=cond_vec,
                n_draws=int(stage2_n_draws),
                device=device,
                x_mean=stage2_x_mean,
                x_std=stage2_x_std,
            )
            fine_proj = _project_to_parent_table(
                fine_raw,
                parent_table=np.asarray(parent_cond, dtype=np.float64).reshape(-1),
                child_parent_index=child_parent_index,
            ).reshape(FINE_SHAPE)
            fine_uniform = _uniform_from_parent_table(
                parent_table=np.asarray(parent_cond, dtype=np.float64).reshape(-1),
                child_parent_index=child_parent_index,
            ).reshape(FINE_SHAPE)

            out_diff[:, sex_idx, :, esr_idx] = fine_proj * subgroup_mass
            out_uniform[:, sex_idx, :, esr_idx] = fine_uniform * subgroup_mass

    out_diff = out_diff / max(float(out_diff.sum()), 1e-12)
    out_uniform = out_uniform / max(float(out_uniform.sum()), 1e-12)
    return out_diff.reshape(-1), out_uniform.reshape(-1)


def _combine_oracle_stage2_prediction(
    *,
    p_true: np.ndarray,
    stage2_model: DiffusionTabularModel,
    stage2_x_mean: np.ndarray,
    stage2_x_std: np.ndarray,
    stage2_n_draws: int,
    device: str | None,
    child_parent_index: np.ndarray,
) -> np.ndarray:
    true_tab = np.asarray(p_true, dtype=np.float64).reshape(FINAL_SHAPE)
    out_oracle = np.zeros(FINAL_SHAPE, dtype=np.float64)

    for sex_idx in range(len(SEX_LABELS)):
        for esr_idx in range(len(ESR_LITE_LABELS)):
            fine_true = true_tab[:, sex_idx, :, esr_idx]
            subgroup_mass = float(fine_true.sum())
            if subgroup_mass <= 0:
                continue
            fine_cond_true = fine_true / subgroup_mass
            parent_cond_true = _aggregate_coarse_age_schl(fine_cond_true)
            cond_vec = _compose_stage2_condition(
                parent_cond=parent_cond_true,
                sex_idx=sex_idx,
                esr_idx=esr_idx,
                subgroup_mass=subgroup_mass,
            )
            fine_raw = _sample_mean_prob(
                model=stage2_model,
                cond_row=cond_vec,
                n_draws=int(stage2_n_draws),
                device=device,
                x_mean=stage2_x_mean,
                x_std=stage2_x_std,
            )
            fine_proj = _project_to_parent_table(
                fine_raw,
                parent_table=np.asarray(parent_cond_true, dtype=np.float64).reshape(-1),
                child_parent_index=child_parent_index,
            ).reshape(FINE_SHAPE)
            out_oracle[:, sex_idx, :, esr_idx] = fine_proj * subgroup_mass

    out_oracle = out_oracle / max(float(out_oracle.sum()), 1e-12)
    return out_oracle.reshape(-1)


def _marginal_final(p: np.ndarray, *, axis_name: str) -> np.ndarray:
    tab = np.asarray(p, dtype=np.float64).reshape(FINAL_SHAPE)
    if axis_name == "AGEP_bin":
        return tab.sum(axis=(1, 2, 3))
    if axis_name == "SEX":
        return tab.sum(axis=(0, 2, 3))
    if axis_name == "SCHL_allpop":
        return tab.sum(axis=(0, 1, 3))
    if axis_name == "ESR_allpop":
        return tab.sum(axis=(0, 1, 2))
    raise ValueError(f"Unsupported axis_name={axis_name}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="eval_external_c2f_age_schl_pipeline")
    ap.add_argument("--stage1_joint_wide_csv", required=True)
    ap.add_argument("--stage1_schema_json", required=True)
    ap.add_argument("--stage1_condition_csv", required=True)
    ap.add_argument("--stage1_checkpoint", required=True)
    ap.add_argument("--stage2_wide_csv", required=True)
    ap.add_argument("--stage2_schema_json", required=True)
    ap.add_argument("--stage2_checkpoint", required=True)
    ap.add_argument("--final_target_wide_csv", required=True)
    ap.add_argument("--final_target_schema_json", required=True)
    ap.add_argument("--one_shot_summary_json", default=None)
    ap.add_argument("--stage1_n_eval_joint_samples", type=int, default=64)
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
    stage1_checkpoint = pathlib.Path(args.stage1_checkpoint).expanduser().resolve()
    stage2_wide_csv = pathlib.Path(args.stage2_wide_csv).expanduser().resolve()
    stage2_schema_json = pathlib.Path(args.stage2_schema_json).expanduser().resolve()
    stage2_checkpoint = pathlib.Path(args.stage2_checkpoint).expanduser().resolve()
    final_target_csv = pathlib.Path(args.final_target_wide_csv).expanduser().resolve()
    final_target_schema_json = pathlib.Path(args.final_target_schema_json).expanduser().resolve()
    one_shot_summary = pathlib.Path(args.one_shot_summary_json).expanduser().resolve() if args.one_shot_summary_json else None

    for p in [
        stage1_joint_csv,
        stage1_schema_json,
        stage1_condition_csv,
        stage1_checkpoint,
        stage2_wide_csv,
        stage2_schema_json,
        stage2_checkpoint,
        final_target_csv,
        final_target_schema_json,
    ]:
        if not p.exists():
            raise SystemExit(f"Required path not found: {p}")
    if one_shot_summary is not None and not one_shot_summary.exists():
        raise SystemExit(f"one_shot_summary_json not found: {one_shot_summary}")

    run_id = f"_us_puma_external_c2f_age_schl_eval_{_dt.datetime.now(_dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else (_REPO_ROOT / "outputs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    stage1_df, stage1_p_joint, stage1_ids, stage1_shape = _load_joint_wide(
        joint_wide_csv=stage1_joint_csv,
        schema_json=stage1_schema_json,
    )
    if tuple(stage1_shape) != STAGE1_SHAPE:
        raise SystemExit(f"Unexpected stage1 shape: got {stage1_shape}, expected {STAGE1_SHAPE}")
    stage1_is_mi = (stage1_df["statefp"] == "26").to_numpy(dtype=bool)
    stage1_x_log = np.log(np.clip(stage1_p_joint, 0.0, None) + 1e-6).astype(np.float32)
    stage1_x_mean, stage1_x_std = _compute_train_scaler(stage1_x_log, is_mi=stage1_is_mi)
    stage1_var_specs = _load_var_specs_from_schema(schema_json=stage1_schema_json)
    stage1_cond, stage1_block_slices, _ = _load_external_condition_matrix(
        condition_csv=stage1_condition_csv,
        ids=stage1_ids,
        var_specs=stage1_var_specs,
    )
    stage1_ext_marg = {var: stage1_cond[:, s].copy() for var, s in stage1_block_slices.items()}

    stage1_model = DiffusionTabularModel(
        input_dim=STAGE1_K,
        cond_dim=int(stage1_cond.shape[1]),
        seed=int(args.seed),
        config=TabDDPMConfig(timesteps=1000, hidden_dims=_parse_hidden_dims("256,256")),
    )
    stage1_model.load(stage1_checkpoint)

    stage2_schema = json.loads(stage2_schema_json.read_text(encoding="utf-8"))
    child_parent_index = np.asarray(stage2_schema["child_parent_index"], dtype=np.int16)
    if child_parent_index.shape[0] != STAGE2_K:
        raise SystemExit("Unexpected child_parent_index length in stage2 schema.")

    stage2_df = pd.read_csv(stage2_wide_csv, low_memory=False)
    stage2_df["statefp"] = stage2_df["statefp"].map(_canon_statefp)
    stage2_is_mi = (stage2_df["statefp"] == "26").to_numpy(dtype=bool)
    stage2_joint_cols = [f"p_joint_{i:03d}" for i in range(STAGE2_K)]
    stage2_p_joint = stage2_df[stage2_joint_cols].to_numpy(dtype=np.float32)
    stage2_p_joint = np.clip(stage2_p_joint, 0.0, None)
    stage2_p_joint = stage2_p_joint / np.maximum(stage2_p_joint.sum(axis=1, keepdims=True), 1e-12)
    stage2_x_log = np.log(np.clip(stage2_p_joint, 0.0, None) + 1e-6).astype(np.float32)
    stage2_x_mean, stage2_x_std = _compute_train_scaler(stage2_x_log, is_mi=stage2_is_mi)

    stage2_model = DiffusionTabularModel(
        input_dim=STAGE2_K,
        cond_dim=18,
        seed=int(args.seed),
        config=TabDDPMConfig(timesteps=1000, hidden_dims=_parse_hidden_dims("256,256")),
    )
    stage2_model.load(stage2_checkpoint)

    final_df, final_p_joint, final_ids, final_shape = _load_joint_wide(
        joint_wide_csv=final_target_csv,
        schema_json=final_target_schema_json,
    )
    if tuple(final_shape) != FINAL_SHAPE:
        raise SystemExit(f"Unexpected final target shape: got {final_shape}, expected {FINAL_SHAPE}")
    final_is_mi = (final_df["statefp"] == "26").to_numpy(dtype=bool)
    mi_ids = [uid for uid, flag in zip(final_ids, final_is_mi.tolist()) if flag]
    if not mi_ids:
        raise SystemExit("No Michigan PUMAs found in final target.")

    stage1_idx = {uid: i for i, uid in enumerate(stage1_ids)}
    final_idx = {uid: i for i, uid in enumerate(final_ids)}

    tvd_c2f: list[float] = []
    tvd_uniform: list[float] = []
    tvd_oracle: list[float] = []
    cosine_c2f: list[float] = []
    cosine_uniform: list[float] = []
    cosine_oracle: list[float] = []
    tvd_age_c2f: list[float] = []
    tvd_schl_c2f: list[float] = []
    tvd_age_uniform: list[float] = []
    tvd_schl_uniform: list[float] = []
    tvd_age_oracle: list[float] = []
    tvd_schl_oracle: list[float] = []

    for uid in mi_ids:
        i1 = stage1_idx[uid]
        i3 = final_idx[uid]

        p1_raw = _sample_mean_prob(
            model=stage1_model,
            cond_row=stage1_cond[i1],
            n_draws=int(args.stage1_n_eval_joint_samples),
            device=args.device,
            x_mean=stage1_x_mean,
            x_std=stage1_x_std,
        )
        marginals_ext = [stage1_ext_marg[var][i1] for var, _, _ in stage1_var_specs]
        p1_proj = _ipf_nd(
            seed_joint=p1_raw.reshape(STAGE1_SHAPE),
            target_marginals=[np.asarray(m, dtype=float) for m in marginals_ext],
            shape=STAGE1_SHAPE,
            max_iter=int(args.ipf_iters),
        ).reshape(-1)
        p1_proj = p1_proj / max(float(p1_proj.sum()), 1e-12)

        p_final, p_uniform = _combine_c2f_prediction(
            stage1_joint=p1_proj,
            stage2_model=stage2_model,
            stage2_x_mean=stage2_x_mean,
            stage2_x_std=stage2_x_std,
            stage2_n_draws=int(args.stage2_n_eval_joint_samples),
            device=args.device,
            child_parent_index=child_parent_index,
        )
        p_true = final_p_joint[i3]
        p_oracle = _combine_oracle_stage2_prediction(
            p_true=p_true,
            stage2_model=stage2_model,
            stage2_x_mean=stage2_x_mean,
            stage2_x_std=stage2_x_std,
            stage2_n_draws=int(args.stage2_n_eval_joint_samples),
            device=args.device,
            child_parent_index=child_parent_index,
        )

        tvd_c2f.append(_tvd(p_final, p_true))
        tvd_uniform.append(_tvd(p_uniform, p_true))
        tvd_oracle.append(_tvd(p_oracle, p_true))
        cosine_c2f.append(_cosine(p_final, p_true))
        cosine_uniform.append(_cosine(p_uniform, p_true))
        cosine_oracle.append(_cosine(p_oracle, p_true))
        tvd_age_c2f.append(_tvd(_marginal_final(p_final, axis_name="AGEP_bin"), _marginal_final(p_true, axis_name="AGEP_bin")))
        tvd_schl_c2f.append(_tvd(_marginal_final(p_final, axis_name="SCHL_allpop"), _marginal_final(p_true, axis_name="SCHL_allpop")))
        tvd_age_uniform.append(_tvd(_marginal_final(p_uniform, axis_name="AGEP_bin"), _marginal_final(p_true, axis_name="AGEP_bin")))
        tvd_schl_uniform.append(_tvd(_marginal_final(p_uniform, axis_name="SCHL_allpop"), _marginal_final(p_true, axis_name="SCHL_allpop")))
        tvd_age_oracle.append(_tvd(_marginal_final(p_oracle, axis_name="AGEP_bin"), _marginal_final(p_true, axis_name="AGEP_bin")))
        tvd_schl_oracle.append(_tvd(_marginal_final(p_oracle, axis_name="SCHL_allpop"), _marginal_final(p_true, axis_name="SCHL_allpop")))

    results: dict[str, Any] = {
        "coarse_to_fine": {
            "tvd_joint": _summ(tvd_c2f),
            "cosine_joint": _summ(cosine_c2f),
            "tvd_AGEP_bin": _summ(tvd_age_c2f),
            "tvd_SCHL_allpop": _summ(tvd_schl_c2f),
        },
        "uniform_refine_baseline": {
            "tvd_joint": _summ(tvd_uniform),
            "cosine_joint": _summ(cosine_uniform),
            "tvd_AGEP_bin": _summ(tvd_age_uniform),
            "tvd_SCHL_allpop": _summ(tvd_schl_uniform),
        },
        "oracle_stage2": {
            "tvd_joint": _summ(tvd_oracle),
            "cosine_joint": _summ(cosine_oracle),
            "tvd_AGEP_bin": _summ(tvd_age_oracle),
            "tvd_SCHL_allpop": _summ(tvd_schl_oracle),
        },
    }

    if one_shot_summary is not None:
        ref = json.loads(one_shot_summary.read_text(encoding="utf-8"))
        results["references"] = ref

    run_summary = {
        "created_utc": _utc_now_iso(),
        "stage1_joint_wide_csv": str(stage1_joint_csv),
        "stage1_schema_json": str(stage1_schema_json),
        "stage1_condition_csv": str(stage1_condition_csv),
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage2_wide_csv": str(stage2_wide_csv),
        "stage2_schema_json": str(stage2_schema_json),
        "stage2_checkpoint": str(stage2_checkpoint),
        "final_target_wide_csv": str(final_target_csv),
        "final_target_schema_json": str(final_target_schema_json),
        "stage1_shape": list(STAGE1_SHAPE),
        "stage2_shape": list(FINE_SHAPE),
        "final_shape": list(FINAL_SHAPE),
        "n_mi_pumas": int(len(mi_ids)),
        "stage1_n_eval_joint_samples": int(args.stage1_n_eval_joint_samples),
        "stage2_n_eval_joint_samples": int(args.stage2_n_eval_joint_samples),
        "ipf_iters": int(args.ipf_iters),
        "device": str(args.device),
        "seed": int(args.seed),
        "results": results,
    }

    _write_json(out_dir / "run_summary.json", run_summary)
    _write_json(out_dir / "metrics" / "coarse_to_fine_summary.json", results)
    print(f"[ok] wrote: {out_dir}")


if __name__ == "__main__":
    main()
