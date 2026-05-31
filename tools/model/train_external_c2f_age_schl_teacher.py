#!/usr/bin/env python3
from __future__ import annotations

"""
Train the teacher-forced stage-2 coarse-to-fine diffusion model for age x education refinement.

Each row is a (PUMA, SEX, ESR_lite) subgroup.
Target: fine AGEP(10) x SCHL(5) conditional distribution.
Condition:
  - coarse AGEP_lite(4) x SCHL_lite(3) table within the subgroup
  - subgroup SEX one-hot
  - subgroup ESR_lite one-hot
  - subgroup parent mass

This script is a stage-2 learnability probe, not an end-to-end pipeline.
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

from synthpop.model.diffusion_tabular import DiffusionTabularModel, TabDDPMConfig
from tools.model.train_us_puma_5var_diffusion import (
    _canon_statefp,
    _cosine,
    _parse_hidden_dims,
    _require_torch,
    _softmax_rows,
    _summ,
    _tvd,
    _utc_now_iso,
    _write_json,
)


AGE_DIM = 10
SCHL_DIM = 5
COARSE_PARENT_DIM = 12
FINE_K = AGE_DIM * SCHL_DIM


def _marginal_from_joint_2d(p: np.ndarray, *, axis: int) -> np.ndarray:
    tab = np.asarray(p, dtype=float).reshape(AGE_DIM, SCHL_DIM)
    if axis == 0:
        return tab.sum(axis=1)
    if axis == 1:
        return tab.sum(axis=0)
    raise ValueError(f"Unsupported axis={axis}")


def _project_to_parent_table(p_raw: np.ndarray, parent_table: np.ndarray, child_parent_index: np.ndarray) -> np.ndarray:
    p_raw = np.asarray(p_raw, dtype=float).reshape(-1)
    parent_table = np.asarray(parent_table, dtype=float).reshape(-1)
    out = np.zeros_like(p_raw, dtype=float)
    for pid in range(int(parent_table.shape[0])):
        mask = child_parent_index == pid
        target_mass = float(parent_table[pid])
        if target_mass <= 0:
            continue
        raw_mass = float(p_raw[mask].sum())
        if raw_mass > 0:
            out[mask] = p_raw[mask] / raw_mass * target_mass
        else:
            out[mask] = target_mass / max(int(mask.sum()), 1)
    out = out / max(float(out.sum()), 1e-12)
    return out


def _uniform_from_parent_table(parent_table: np.ndarray, child_parent_index: np.ndarray) -> np.ndarray:
    out = np.zeros((child_parent_index.shape[0],), dtype=float)
    for pid in range(int(np.max(child_parent_index)) + 1):
        mask = child_parent_index == pid
        target_mass = float(parent_table[pid])
        if target_mass <= 0:
            continue
        out[mask] = target_mass / max(int(mask.sum()), 1)
    out = out / max(float(out.sum()), 1e-12)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="train_external_c2f_age_schl_teacher")
    ap.add_argument("--wide_csv", required=True)
    ap.add_argument("--schema_json", required=True)
    ap.add_argument("--eval_mode", choices=["leave_mi_out"], default="leave_mi_out")
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--hidden_dims", default="256,256")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--condition_injection", choices=["concat", "film"], default="concat")
    ap.add_argument("--film_hidden_dim", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--n_eval_joint_samples", type=int, default=128)
    ap.add_argument("--save_final_model", action="store_true")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    torch = _require_torch()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    in_path = pathlib.Path(args.wide_csv).expanduser().resolve()
    schema_path = pathlib.Path(args.schema_json).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"wide_csv not found: {in_path}")
    if not schema_path.exists():
        raise SystemExit(f"schema_json not found: {schema_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    child_parent_index = np.asarray(schema["child_parent_index"], dtype=np.int16)
    if child_parent_index.shape[0] != FINE_K:
        raise SystemExit("schema_json child_parent_index has unexpected length")

    run_id = f"_us_puma_external_c2f_age_schl_teacher_{_dt.datetime.now(_dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else (_REPO_ROOT / "outputs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, low_memory=False)
    req = {"statefp", "puma_uid", "subgroup_uid"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"wide_csv missing columns: {miss}")

    parent_cols = [f"c_parent_{i:02d}" for i in range(COARSE_PARENT_DIM)]
    sex_cols = [f"c_sex_{i:02d}" for i in range(2)]
    esr_cols = [f"c_esr_{i:02d}" for i in range(3)]
    cond_cols = parent_cols + sex_cols + esr_cols + ["c_parent_mass"]
    p_joint_cols = [f"p_joint_{i:03d}" for i in range(FINE_K)]
    p_age_cols = [f"p_age_{i:02d}" for i in range(AGE_DIM)]
    p_schl_cols = [f"p_schl_{i:02d}" for i in range(SCHL_DIM)]
    for cols in [cond_cols, p_joint_cols, p_age_cols, p_schl_cols]:
        miss_cols = [c for c in cols if c not in df.columns]
        if miss_cols:
            raise SystemExit(f"wide_csv missing columns: {miss_cols[:5]}")

    df["statefp"] = df["statefp"].map(_canon_statefp)
    is_mi = df["statefp"] == "26"
    if int(is_mi.sum()) == 0:
        raise SystemExit("No Michigan subgroup rows found (statefp==26).")

    p_joint = df[p_joint_cols].to_numpy(dtype=np.float32)
    p_joint = np.clip(p_joint, 0.0, None)
    p_joint = p_joint / np.maximum(p_joint.sum(axis=1, keepdims=True), 1e-12)
    p_age = df[p_age_cols].to_numpy(dtype=np.float32)
    p_schl = df[p_schl_cols].to_numpy(dtype=np.float32)
    cond = df[cond_cols].to_numpy(dtype=np.float32)
    ids = df["subgroup_uid"].astype(str).tolist()

    x_log_all = np.log(np.clip(p_joint, 0.0, None) + 1e-6).astype(np.float32)
    train_idx = np.where(~is_mi.to_numpy(dtype=bool))[0]
    test_idx = np.where(is_mi.to_numpy(dtype=bool))[0]
    if train_idx.size == 0 or test_idx.size == 0:
        raise SystemExit("Invalid leave_mi_out split.")

    x_train_log = x_log_all[train_idx]
    x_mean = x_train_log.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = x_train_log.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
    x_train = ((x_train_log - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)).astype(np.float32)

    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    model = DiffusionTabularModel(
        input_dim=FINE_K,
        cond_dim=int(cond.shape[1]),
        seed=int(args.seed),
        config=TabDDPMConfig(
            timesteps=int(args.timesteps),
            hidden_dims=hidden_dims,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            condition_injection=str(args.condition_injection),
            film_hidden_dim=int(args.film_hidden_dim),
        ),
    )
    fit_kwargs: dict[str, Any] = {
        "x": torch.from_numpy(x_train),
        "cond": torch.from_numpy(cond[train_idx]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "device": args.device,
        "log_every": int(args.log_every),
    }
    model.fit(**fit_kwargs)

    saved_checkpoints: list[str] = []
    if bool(args.save_final_model):
        ckpt = out_dir / "checkpoints" / "external_c2f_age_schl_teacher" / "leave_mi_out" / "final.pt"
        model.save(ckpt)
        saved_checkpoints.append(str(ckpt))

    tvd_raw: list[float] = []
    tvd_projected: list[float] = []
    cosine_raw: list[float] = []
    cosine_projected: list[float] = []
    tvd_age_raw: list[float] = []
    tvd_age_projected: list[float] = []
    tvd_schl_raw: list[float] = []
    tvd_schl_projected: list[float] = []
    tvd_uniform: list[float] = []

    for j, idx in enumerate(test_idx):
        c_row = cond[idx]
        parent_table = c_row[:COARSE_PARENT_DIM]
        p_true = p_joint[idx]

        c = np.repeat(c_row.reshape(1, -1), repeats=int(args.n_eval_joint_samples), axis=0).astype(np.float32)
        z = model.sample(n=int(args.n_eval_joint_samples), cond=torch.from_numpy(c), device=args.device).numpy()
        logp = z.astype(np.float64) * x_std.reshape(1, -1).astype(np.float64) + x_mean.reshape(1, -1).astype(np.float64)
        p_draws = _softmax_rows(logp)
        p_hat_raw = np.mean(p_draws, axis=0)
        p_hat_raw = p_hat_raw / max(float(p_hat_raw.sum()), 1e-12)
        p_hat_projected = _project_to_parent_table(p_hat_raw, parent_table=parent_table, child_parent_index=child_parent_index)
        p_uniform = _uniform_from_parent_table(parent_table=parent_table, child_parent_index=child_parent_index)

        tvd_raw.append(_tvd(p_hat_raw, p_true))
        tvd_projected.append(_tvd(p_hat_projected, p_true))
        cosine_raw.append(_cosine(p_hat_raw, p_true))
        cosine_projected.append(_cosine(p_hat_projected, p_true))
        tvd_uniform.append(_tvd(p_uniform, p_true))

        mt_age = p_age[idx]
        mt_schl = p_schl[idx]
        tvd_age_raw.append(_tvd(_marginal_from_joint_2d(p_hat_raw, axis=0), mt_age))
        tvd_age_projected.append(_tvd(_marginal_from_joint_2d(p_hat_projected, axis=0), mt_age))
        tvd_schl_raw.append(_tvd(_marginal_from_joint_2d(p_hat_raw, axis=1), mt_schl))
        tvd_schl_projected.append(_tvd(_marginal_from_joint_2d(p_hat_projected, axis=1), mt_schl))

    ablation_summary = {
        "teacher_forced_stage2": {
            "tvd_joint_raw": _summ(tvd_raw),
            "tvd_joint_projected": _summ(tvd_projected),
            "cosine_joint_raw": _summ(cosine_raw),
            "cosine_joint_projected": _summ(cosine_projected),
            "tvd_AGEP_bin_raw": _summ(tvd_age_raw),
            "tvd_AGEP_bin_projected": _summ(tvd_age_projected),
            "tvd_SCHL_allpop_raw": _summ(tvd_schl_raw),
            "tvd_SCHL_allpop_projected": _summ(tvd_schl_projected),
        },
        "baseline_uniform_parent": {
            "tvd_joint": _summ(tvd_uniform),
        },
    }

    run_summary = {
        "created_utc": _utc_now_iso(),
        "wide_csv": str(in_path),
        "schema_json": str(schema_path),
        "n_rows_total": int(df.shape[0]),
        "n_mi_rows": int(is_mi.sum()),
        "n_non_mi_rows": int((~is_mi).sum()),
        "target_shape": {"age": AGE_DIM, "schl": SCHL_DIM},
        "coarse_parent_dim": int(COARSE_PARENT_DIM),
        "cond_dim": int(cond.shape[1]),
        "eval_mode": "leave_mi_out",
        "timesteps": int(args.timesteps),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_dims": list(hidden_dims),
        "condition_injection": str(args.condition_injection),
        "film_hidden_dim": int(args.film_hidden_dim),
        "n_eval_joint_samples": int(args.n_eval_joint_samples),
        "seed": int(args.seed),
        "device": args.device,
        "save_final_model": bool(args.save_final_model),
        "saved_checkpoints": saved_checkpoints,
        "results": ablation_summary,
    }

    _write_json(out_dir / "run_summary.json", run_summary)
    _write_json(out_dir / "metrics" / "ablation_summary.json", ablation_summary)
    print(f"[ok] wrote: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
