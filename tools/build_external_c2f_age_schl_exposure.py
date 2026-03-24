#!/usr/bin/env python3
from __future__ import annotations

"""
Build an exposure-matched stage-2 dataset for age x education refinement.

Unlike the teacher-forced builder, this script does not use the true coarse
AGEP_lite x SCHL_lite parent table as the stage-2 condition. Instead, it first
runs the trained stage-1 lite diffusion model and uses its projected coarse
prediction for each PUMA. The fine target remains the true conditional
AGEP_fine x SCHL_fine table within each (SEX, ESR_lite) subgroup.

This isolates a clean scientific question:

  Does the current stage-2 gap mainly come from exposure mismatch between
  training on true parents and inferring on predicted parents?
"""

import argparse
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

from src.synthpop.model.diffusion_tabular import DiffusionTabularModel, TabDDPMConfig
from tools.build_external_c2f_age_schl_teacher import (
    COARSE_SHAPE,
    ESR_LITE_LABELS,
    ESR_FULL_TO_LITE,
    FINE_SHAPE,
    SEX_LABELS,
    _aggregate_coarse_age_schl,
    _child_parent_index,
)
from tools.eval_external_c2f_age_schl_pipeline import (
    STAGE1_K,
    STAGE1_SHAPE,
    _compute_train_scaler,
    _load_joint_wide,
    _sample_mean_prob,
)
from tools.train_us_puma_5var_diffusion import (
    _canon_puma5,
    _canon_statefp,
    _canon_uid,
    _parse_hidden_dims,
    _require_torch,
)
from tools.train_us_puma_external_v1_diffusion import _load_external_condition_matrix, _load_var_specs_from_schema


def _utc_now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_external_c2f_age_schl_exposure")
    ap.add_argument("--stage1_joint_wide_csv", required=True)
    ap.add_argument("--stage1_schema_json", required=True)
    ap.add_argument("--stage1_condition_csv", required=True)
    ap.add_argument("--stage1_checkpoint", required=True)
    ap.add_argument("--final_target_wide_csv", required=True)
    ap.add_argument("--final_target_schema_json", required=True)
    ap.add_argument("--n_eval_joint_samples", type=int, default=64)
    ap.add_argument("--ipf_iters", type=int, default=200)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    _require_torch()

    stage1_joint_csv = pathlib.Path(args.stage1_joint_wide_csv).expanduser().resolve()
    stage1_schema_json = pathlib.Path(args.stage1_schema_json).expanduser().resolve()
    stage1_condition_csv = pathlib.Path(args.stage1_condition_csv).expanduser().resolve()
    stage1_checkpoint = pathlib.Path(args.stage1_checkpoint).expanduser().resolve()
    final_target_csv = pathlib.Path(args.final_target_wide_csv).expanduser().resolve()
    final_target_schema_json = pathlib.Path(args.final_target_schema_json).expanduser().resolve()
    for p in [
        stage1_joint_csv,
        stage1_schema_json,
        stage1_condition_csv,
        stage1_checkpoint,
        final_target_csv,
        final_target_schema_json,
    ]:
        if not p.exists():
            raise SystemExit(f"path not found: {p}")

    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = final_target_csv.parent.parent / "external_c2f"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = "extc2f_age_schl_exposure_pums_2023_puma_us"
    wide_csv = out_dir / f"{stem}_wide.csv"
    schema_json = out_dir / f"{stem}.schema.json"
    metadata_json = out_dir / f"{stem}.metadata.json"
    if any(p.exists() for p in [wide_csv, schema_json, metadata_json]) and not args.overwrite:
        raise SystemExit(f"output exists under {out_dir} (use --overwrite)")

    stage1_df, stage1_p_joint, stage1_ids, stage1_shape = _load_joint_wide(
        joint_wide_csv=stage1_joint_csv,
        schema_json=stage1_schema_json,
    )
    if tuple(stage1_shape) != STAGE1_SHAPE:
        raise SystemExit(f"unexpected stage1 shape: {stage1_shape}")
    stage1_is_mi = (stage1_df["statefp"] == "26").to_numpy(dtype=bool)
    stage1_x_log = np.log(np.clip(stage1_p_joint, 0.0, None) + 1e-6).astype(np.float32)
    stage1_x_mean, stage1_x_std = _compute_train_scaler(stage1_x_log, is_mi=stage1_is_mi)
    stage1_var_specs = _load_var_specs_from_schema(schema_json=stage1_schema_json)
    stage1_cond, _, _ = _load_external_condition_matrix(
        condition_csv=stage1_condition_csv,
        ids=stage1_ids,
        var_specs=stage1_var_specs,
    )

    stage1_model = DiffusionTabularModel(
        input_dim=STAGE1_K,
        cond_dim=int(stage1_cond.shape[1]),
        seed=int(args.seed),
        config=TabDDPMConfig(timesteps=1000, hidden_dims=_parse_hidden_dims("256,256")),
    )
    stage1_model.load(stage1_checkpoint)

    final_df, _, _, final_shape = _load_joint_wide(
        joint_wide_csv=final_target_csv,
        schema_json=final_target_schema_json,
    )
    expected_final_shape = (FINE_SHAPE[0], len(SEX_LABELS), FINE_SHAPE[1], len(ESR_LITE_LABELS))
    if tuple(final_shape) != expected_final_shape:
        raise SystemExit(f"unexpected final target shape: {final_shape}, expected {expected_final_shape}")

    stage1_idx = {uid: i for i, uid in enumerate(stage1_ids)}
    child_parent_idx = _child_parent_index()
    rows: list[dict[str, Any]] = []
    parent_tvd_list: list[float] = []
    mass_abs_err_list: list[float] = []
    zero_pred_mass_rows = 0

    p_joint_cols = [f"p_joint_{i:03d}" for i in range(int(np.prod(final_shape)))]

    for r in final_df.to_dict(orient="records"):
        statefp = _canon_statefp(r["statefp"])
        puma5 = _canon_puma5(r.get("puma5", r["puma"]))
        puma_uid = _canon_uid(statefp, puma5)
        if puma_uid not in stage1_idx:
            raise SystemExit(f"missing stage1 row for puma_uid={puma_uid}")

        i1 = stage1_idx[puma_uid]
        p1_raw = _sample_mean_prob(
            model=stage1_model,
            cond_row=stage1_cond[i1],
            n_draws=int(args.n_eval_joint_samples),
            device=args.device,
            x_mean=stage1_x_mean,
            x_std=stage1_x_std,
        )
        from tools.train_us_puma_5var_diffusion import _ipf_nd, _tvd  # local import to keep file self-contained

        marginals_ext = []
        start = 0
        for _, _, cats in stage1_var_specs:
            stop = start + len(cats)
            marginals_ext.append(np.asarray(stage1_cond[i1, start:stop], dtype=float))
            start = stop
        p1_proj = _ipf_nd(
            seed_joint=p1_raw.reshape(STAGE1_SHAPE),
            target_marginals=[np.asarray(m, dtype=float) for m in marginals_ext],
            shape=STAGE1_SHAPE,
            max_iter=int(args.ipf_iters),
        ).reshape(STAGE1_SHAPE)
        p_old = np.asarray([float(r[c]) for c in p_joint_cols], dtype=np.float64).reshape(final_shape)

        for si, sex_lab in enumerate(SEX_LABELS):
            for esr_lite_idx, esr_lite_lab in enumerate(ESR_LITE_LABELS):
                fine_true = p_old[:, si, :, esr_lite_idx]
                true_mass = float(fine_true.sum())
                if true_mass <= 0:
                    continue

                pred_parent = p1_proj[:, si, :, esr_lite_idx]
                pred_mass = float(pred_parent.sum())
                if pred_mass <= 0:
                    zero_pred_mass_rows += 1
                    pred_parent_cond = np.zeros(COARSE_SHAPE, dtype=np.float64)
                else:
                    pred_parent_cond = pred_parent / pred_mass

                fine_cond_true = fine_true / true_mass
                true_parent_cond = _aggregate_coarse_age_schl(fine_cond_true)

                parent_tvd_list.append(_tvd(pred_parent_cond.reshape(-1), true_parent_cond.reshape(-1)))
                mass_abs_err_list.append(abs(pred_mass - true_mass))

                row: dict[str, Any] = {
                    "statefp": statefp,
                    "puma": str(int(puma5)) if puma5 else "",
                    "puma5": puma5,
                    "puma_uid": puma_uid,
                    "subgroup_sex": sex_lab,
                    "subgroup_esr": esr_lite_lab,
                    "subgroup_uid": f"{puma_uid}__sex{sex_lab}__esr{esr_lite_lab}",
                    "parent_mass": pred_mass,
                    "parent_mass_true": true_mass,
                    "parent_tvd_pred_to_true": _tvd(pred_parent_cond.reshape(-1), true_parent_cond.reshape(-1)),
                    "parent_mass_abs_err": abs(pred_mass - true_mass),
                    "total_person_weight": float(r["total_person_weight"]),
                    "n_persons_unweighted": int(r["n_persons_unweighted"]),
                }

                sex_onehot = np.zeros((len(SEX_LABELS),), dtype=np.float64)
                sex_onehot[si] = 1.0
                esr_onehot = np.zeros((len(ESR_LITE_LABELS),), dtype=np.float64)
                esr_onehot[esr_lite_idx] = 1.0

                for i, v in enumerate(pred_parent_cond.reshape(-1)):
                    row[f"c_parent_{i:02d}"] = float(v)
                for i, v in enumerate(sex_onehot):
                    row[f"c_sex_{i:02d}"] = float(v)
                for i, v in enumerate(esr_onehot):
                    row[f"c_esr_{i:02d}"] = float(v)
                row["c_parent_mass"] = float(pred_mass)

                p_age = fine_cond_true.sum(axis=1)
                p_schl = fine_cond_true.sum(axis=0)
                for i, v in enumerate(p_age):
                    row[f"p_age_{i:02d}"] = float(v)
                for i, v in enumerate(p_schl):
                    row[f"p_schl_{i:02d}"] = float(v)
                for i, v in enumerate(fine_cond_true.reshape(-1)):
                    row[f"p_joint_{i:03d}"] = float(v)
                rows.append(row)

    wide = pd.DataFrame(rows)
    wide.to_csv(wide_csv, index=False)

    schema = {
        "schema": "external_c2f_age_schl_exposure",
        "created_at": _utc_now_iso(),
        "target_variable_order": ["AGEP_bin", "SCHL_allpop"],
        "target_shape": list(FINE_SHAPE),
        "target_K": int(np.prod(FINE_SHAPE)),
        "coarse_variable_order": ["AGEP_bin_lite", "SCHL_allpop_lite"],
        "coarse_shape": list(COARSE_SHAPE),
        "coarse_K": int(np.prod(COARSE_SHAPE)),
        "subgroup_variables": {"SEX": SEX_LABELS, "ESR_allpop_lite": ESR_LITE_LABELS},
        "condition_blocks": {
            "parent_table": [f"c_parent_{i:02d}" for i in range(int(np.prod(COARSE_SHAPE)))],
            "sex": [f"c_sex_{i:02d}" for i in range(len(SEX_LABELS))],
            "esr": [f"c_esr_{i:02d}" for i in range(len(ESR_LITE_LABELS))],
            "parent_mass": ["c_parent_mass"],
        },
        "child_parent_index": child_parent_idx.astype(int).tolist(),
        "stage1_source": {
            "joint_wide_csv": str(stage1_joint_csv),
            "schema_json": str(stage1_schema_json),
            "condition_csv": str(stage1_condition_csv),
            "checkpoint": str(stage1_checkpoint),
            "n_eval_joint_samples": int(args.n_eval_joint_samples),
            "ipf_iters": int(args.ipf_iters),
        },
    }
    schema_json.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta = {
        "schema": "external_c2f_age_schl_exposure",
        "created_at": _utc_now_iso(),
        "source_final_target_csv": str(final_target_csv),
        "outputs": {"wide_csv": str(wide_csv), "schema_json": str(schema_json)},
        "n_rows": int(wide.shape[0]),
        "n_unique_pumas": int(wide["puma_uid"].nunique()),
        "n_zero_pred_mass_rows": int(zero_pred_mass_rows),
        "target_shape": list(FINE_SHAPE),
        "coarse_shape": list(COARSE_SHAPE),
        "mean_parent_tvd_pred_to_true": float(np.mean(parent_tvd_list)) if parent_tvd_list else None,
        "mean_parent_mass_abs_err": float(np.mean(mass_abs_err_list)) if mass_abs_err_list else None,
    }
    metadata_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {wide_csv}")


if __name__ == "__main__":
    main()
