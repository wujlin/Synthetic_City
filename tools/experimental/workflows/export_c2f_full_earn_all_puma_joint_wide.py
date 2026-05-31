#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import random
import sys
from typing import Any

import numpy as np
import pandas as pd


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


FULL_VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop", "EARN_16p_bin"]


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _joint_cols(df: pd.DataFrame) -> list[str]:
    cols = [str(c) for c in df.columns if str(c).startswith("p_joint_")]
    if not cols:
        raise SystemExit("reference joint_wide has no p_joint_* columns")
    return sorted(cols, key=lambda x: int(x.rsplit("_", 1)[1]))


def _select_device(device: str) -> str:
    from tools.model.train_us_puma_5var_diffusion import _require_torch

    torch = _require_torch()
    if str(device).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="export_c2f_full_earn_all_puma_joint_wide",
        description="Export all-PUMA predicted full joint distributions from a coarse-to-fine full-earn checkpoint pair.",
    )
    ap.add_argument("--coarse_preset", default="fine_1440")
    ap.add_argument("--stage1_joint_wide_csv", required=True, type=pathlib.Path)
    ap.add_argument("--stage1_schema_json", required=True, type=pathlib.Path)
    ap.add_argument("--stage1_condition_csv", required=True, type=pathlib.Path)
    ap.add_argument("--stage1_condition_schema_json", required=True, type=pathlib.Path)
    ap.add_argument("--stage1_condition_scale_mode", choices=["none", "log10_total", "log10_total_unit"], default="none")
    ap.add_argument("--stage1_condition_extra_csv", default=None, type=pathlib.Path)
    ap.add_argument("--stage1_condition_extra_standardize", choices=["none", "zscore"], default="none")
    ap.add_argument("--stage1_condition_extra_missing_policy", choices=["require", "zero"], default="require")
    ap.add_argument("--stage1_checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--stage1_timesteps", type=int, default=200)
    ap.add_argument("--stage1_batch_size", type=int, default=512)
    ap.add_argument("--stage2_wide_csv", required=True, type=pathlib.Path)
    ap.add_argument("--stage2_schema_json", required=True, type=pathlib.Path)
    ap.add_argument("--stage2_checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--stage2_n_eval_joint_samples", type=int, default=64)
    ap.add_argument("--ipf_iters", type=int, default=200)
    ap.add_argument("--heldout_statefp_for_scaler", default="26")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out_csv", required=True, type=pathlib.Path)
    ap.add_argument("--out_npz", default=None, type=pathlib.Path)
    ap.add_argument("--out_summary_json", required=True, type=pathlib.Path)
    ap.add_argument("--progress_every", type=int, default=50)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ["SYNTHETIC_CITY_C2F_COARSE_PRESET"] = str(args.coarse_preset)

    from tools.model.eval_external_c2f_full_earn_pipeline import (
        _coarse_marginals_from_full_ext,
        _combine_from_coarse,
        _compute_stage2_scaler,
        _load_full_joint_wide,
        _load_stage1_model,
        _run_full_ipf,
    )
    from tools.model.external_c2f_full_earn_schema import COARSE_K, COARSE_PRESET, COARSE_SHAPE
    from tools.model.external_c2f_full_earn_stage2_model import load_stage2_model
    import tools.model.train_external_joint_hier_diffusion_full as stage1_base
    from tools.model.train_us_puma_5var_diffusion import _ipf_nd, _require_torch, _tvd
    from tools.model.train_us_puma_external_v1_diffusion import (
        _append_condition_extra_matrix,
        _load_condition_specs_from_schema,
        _load_external_condition_matrix,
    )

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    paths = {
        "stage1_joint_wide_csv": args.stage1_joint_wide_csv.expanduser().resolve(),
        "stage1_schema_json": args.stage1_schema_json.expanduser().resolve(),
        "stage1_condition_csv": args.stage1_condition_csv.expanduser().resolve(),
        "stage1_condition_schema_json": args.stage1_condition_schema_json.expanduser().resolve(),
        "stage1_checkpoint": args.stage1_checkpoint.expanduser().resolve(),
        "stage2_wide_csv": args.stage2_wide_csv.expanduser().resolve(),
        "stage2_schema_json": args.stage2_schema_json.expanduser().resolve(),
        "stage2_checkpoint": args.stage2_checkpoint.expanduser().resolve(),
        "out_csv": args.out_csv.expanduser().resolve(),
        "out_summary_json": args.out_summary_json.expanduser().resolve(),
    }
    stage1_condition_extra_csv = args.stage1_condition_extra_csv.expanduser().resolve() if args.stage1_condition_extra_csv else None
    out_npz = args.out_npz.expanduser().resolve() if args.out_npz else None
    for key, path in paths.items():
        if key.startswith("out_"):
            continue
        if not path.exists():
            raise SystemExit(f"Required path not found: {path}")
    if stage1_condition_extra_csv is not None and not stage1_condition_extra_csv.exists():
        raise SystemExit(f"Required path not found: {stage1_condition_extra_csv}")

    df, p_true_all, ids = _load_full_joint_wide(
        joint_wide_csv=paths["stage1_joint_wide_csv"],
        schema_json=paths["stage1_schema_json"],
    )
    joint_cols = _joint_cols(df)
    meta_cols = [c for c in df.columns if c not in set(joint_cols)]

    stage1_var_specs = stage1_base._load_var_specs_from_schema(schema_json=paths["stage1_schema_json"])
    cond_specs = _load_condition_specs_from_schema(
        condition_schema_json=paths["stage1_condition_schema_json"],
        fallback_var_specs=stage1_var_specs,
    )
    cond_raw, block_slices, cond_meta = _load_external_condition_matrix(
        condition_csv=paths["stage1_condition_csv"],
        ids=ids,
        var_specs=cond_specs,
        condition_scale_mode=str(args.stage1_condition_scale_mode),
    )
    cond_raw, cond_meta = _append_condition_extra_matrix(
        cond_raw=cond_raw,
        cond_meta=cond_meta,
        extra_csv=stage1_condition_extra_csv,
        ids=ids,
        standardize=str(args.stage1_condition_extra_standardize),
        missing_policy=str(args.stage1_condition_extra_missing_policy),
    )
    ext_marg = {var: cond_raw[:, sl].copy() for var, sl in block_slices.items()}
    ext_marg = stage1_base._augment_ext_marginals_from_cross(cond_raw=cond_raw, block_slices=block_slices, ext_marg=ext_marg)

    torch = _require_torch()
    device = _select_device(str(args.device))
    stage1_model, stage1_payload = _load_stage1_model(
        checkpoint_path=paths["stage1_checkpoint"],
        timesteps=int(args.stage1_timesteps),
        seed=int(args.seed),
    )
    expected_cond_dim = int(stage1_payload["cond_raw_dim"])
    if int(cond_raw.shape[1]) != expected_cond_dim:
        raise SystemExit(f"condition matrix dim={cond_raw.shape[1]} but stage1 checkpoint expects {expected_cond_dim}")
    stage1_model.to(device)

    stage2_x_mean, stage2_x_std = _compute_stage2_scaler(
        wide_csv=paths["stage2_wide_csv"],
        schema_json=paths["stage2_schema_json"],
        heldout_statefp=str(args.heldout_statefp_for_scaler).zfill(2),
    )
    stage2_model, stage2_payload = load_stage2_model(checkpoint_path=paths["stage2_checkpoint"])

    n = int(p_true_all.shape[0])
    batch = max(1, int(args.stage1_batch_size))
    coarse_pred_raw = np.zeros((n, int(COARSE_K)), dtype=np.float64)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        cond_t = torch.from_numpy(cond_raw[start:end]).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            pred = stage1_model.predict_coarse(cond_raw=cond_t).detach().cpu().numpy().astype(np.float64)
        coarse_pred_raw[start:end] = pred
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_full = np.zeros_like(p_true_all, dtype=np.float32)
    tvd_rows: list[float] = []
    progress_every = max(1, int(args.progress_every))
    for idx in range(n):
        ext_row = {var: np.asarray(ext_marg[var][idx], dtype=np.float64) for var in FULL_VARIABLE_ORDER}
        coarse_targets = _coarse_marginals_from_full_ext(ext_row)
        p_coarse_raw = coarse_pred_raw[idx]
        p_coarse_raw = p_coarse_raw / max(float(p_coarse_raw.sum()), 1e-12)
        p_coarse_proj = _ipf_nd(
            seed_joint=p_coarse_raw.reshape(COARSE_SHAPE),
            target_marginals=coarse_targets,
            shape=COARSE_SHAPE,
            max_iter=int(args.ipf_iters),
        )
        p_coarse_proj = p_coarse_proj / max(float(p_coarse_proj.sum()), 1e-12)
        p_full, _ = _combine_from_coarse(
            stage2_model=stage2_model,
            coarse_prob=p_coarse_proj,
            x_mean=stage2_x_mean,
            x_std=stage2_x_std,
            n_draws=int(args.stage2_n_eval_joint_samples),
            device=device,
        )
        p_full = _run_full_ipf(seed_joint=p_full, ext_row=ext_row, ipf_iters=int(args.ipf_iters))
        pred_full[idx] = p_full.astype(np.float32)
        tvd_rows.append(float(_tvd(p_full, p_true_all[idx])))
        if (idx + 1) % progress_every == 0 or (idx + 1) == n:
            print(f"[progress] exported {idx + 1}/{n} pumas; tvd_mean_so_far={float(np.mean(tvd_rows)):.6f}", flush=True)

    out = pd.concat([df[meta_cols].reset_index(drop=True), pd.DataFrame(pred_full, columns=joint_cols)], axis=1)
    paths["out_csv"].parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths["out_csv"], index=False)
    if out_npz is not None:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            p_true=p_true_all.astype(np.float32),
            predicted=pred_full.astype(np.float32),
            puma_uid=df["puma_uid"].astype(str).to_numpy(),
            statefp=df["statefp"].astype(str).to_numpy(),
            puma5=df["puma5"].astype(str).to_numpy(),
        )

    statefp = df["statefp"].astype(str).str.zfill(2)
    tvd_arr = np.asarray(tvd_rows, dtype=np.float64)
    heldout_mask = statefp.to_numpy(dtype=object) == str(args.heldout_statefp_for_scaler).zfill(2)
    row_sums = pred_full.astype(np.float64).sum(axis=1)
    summary = {
        "created_utc": _utc_now(),
        "coarse_preset": str(COARSE_PRESET),
        "coarse_shape": list(COARSE_SHAPE),
        "coarse_k": int(COARSE_K),
        "stage1_checkpoint": str(paths["stage1_checkpoint"]),
        "stage1_checkpoint_meta": {
            "format": str(stage1_payload.get("format", "")),
            "coarse_preset": str(stage1_payload.get("coarse_preset", "")),
            "coarse_shape": list(stage1_payload.get("coarse_shape", [])),
            "cond_raw_dim": int(stage1_payload.get("cond_raw_dim", 0)),
        },
        "stage2_checkpoint": str(paths["stage2_checkpoint"]),
        "stage2_checkpoint_meta": {
            "format": str(stage2_payload.get("format", "")),
            "predict_mode": str(stage2_payload.get("predict_mode", "diffusion")),
            "blend_alpha": float(stage2_payload.get("blend_alpha", 0.0)),
        },
        "stage1_joint_wide_csv": str(paths["stage1_joint_wide_csv"]),
        "stage1_condition_csv": str(paths["stage1_condition_csv"]),
        "stage1_condition_scale_mode": str(args.stage1_condition_scale_mode),
        "stage1_condition_extra_csv": str(stage1_condition_extra_csv) if stage1_condition_extra_csv is not None else None,
        "stage1_condition_extra_standardize": str(args.stage1_condition_extra_standardize),
        "stage1_condition_extra_missing_policy": str(args.stage1_condition_extra_missing_policy),
        "stage2_wide_csv": str(paths["stage2_wide_csv"]),
        "stage2_schema_json": str(paths["stage2_schema_json"]),
        "out_csv": str(paths["out_csv"]),
        "out_npz": str(out_npz) if out_npz is not None else None,
        "n_pumas": int(n),
        "statefp_count": int(statefp.nunique()),
        "total_person_weight": float(pd.to_numeric(out.get("total_person_weight"), errors="coerce").fillna(0.0).sum())
        if "total_person_weight" in out.columns
        else None,
        "heldout_statefp_for_scaler": str(args.heldout_statefp_for_scaler).zfill(2),
        "stage2_n_eval_joint_samples": int(args.stage2_n_eval_joint_samples),
        "ipf_iters": int(args.ipf_iters),
        "stage1_batch_size": int(batch),
        "seed": int(args.seed),
        "device": str(device),
        "condition_meta": cond_meta,
        "row_sum_min": float(np.min(row_sums)),
        "row_sum_max": float(np.max(row_sums)),
        "tvd_mean_all_pumas": float(np.mean(tvd_arr)),
        "tvd_median_all_pumas": float(np.median(tvd_arr)),
        "tvd_mean_heldout_state": float(np.mean(tvd_arr[heldout_mask])) if bool(np.any(heldout_mask)) else None,
    }
    _write_json(paths["out_summary_json"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
