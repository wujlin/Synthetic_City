#!/usr/bin/env python3
from __future__ import annotations

"""
Build a teacher-forced stage-2 coarse-to-fine dataset for the 5-way full-income v2 target.

Stage-1 coarse schema:
  AGEP_lite(4) x SEX(2) x SCHL_lite(3) x ESR_lite(3) x INCOME_lite(6) = 432

Stage-2 target:
  For each (PUMA, coarse parent cell), predict the local fine split among the
  child cells that map into that parent. The local target is represented in a
  padded child-slot coordinate system of width MAX_CHILDREN.

This builder uses either ground-truth coarse tables or stage-1 predicted coarse
tables projected with coarse-level IPF.
"""

import argparse
import csv
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.build_external_target_v1_michigan import _utc_now_iso
from tools.eval_external_c2f_full_income_v2_pipeline import _coarse_marginals_from_full_ext, _load_stage1_model
from tools.external_c2f_full_income_v2_schema import (
    CHILD_INCOME_AUX_FULL,
    COARSE_CATEGORIES,
    COARSE_K,
    COARSE_SHAPE,
    COARSE_VARIABLE_ORDER,
    FULL_K,
    FULL_SHAPE,
    FULL_VARIABLE_ORDER,
    MAX_CHILDREN,
    PADDED_PARENT_CHILD_FULL,
    PADDED_PARENT_CHILD_INCOME_REGIME,
    PARENT_CHILD_SLOT_MASK,
    PARENT_TO_CHILD_FULL,
    coarse_from_full_flat,
    parent_index_labels,
)
from tools.external_income_v1_schema import INCOME_LABELS
from tools.train_external_joint_hier_diffusion_full import _augment_ext_marginals_from_cross
from tools.train_us_puma_5var_diffusion import _canon_puma5, _canon_statefp, _canon_uid, _ipf_nd, _require_torch
from tools.train_us_puma_external_v1_diffusion import (
    _load_condition_specs_from_schema,
    _load_external_condition_matrix,
    _load_var_specs_from_schema,
)


def _build_stage1_ipf_conditioned_coarse(
    *,
    ids: list[str],
    condition_csv: pathlib.Path,
    stage1_schema_json: pathlib.Path,
    condition_schema_json: pathlib.Path | None,
    stage1_checkpoint: pathlib.Path,
    stage1_timesteps: int,
    stage1_ipf_iters: int,
    stage1_seed: int,
    stage1_device: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch = _require_torch()

    var_specs = _load_var_specs_from_schema(schema_json=stage1_schema_json)
    cond_specs = _load_condition_specs_from_schema(
        condition_schema_json=condition_schema_json,
        fallback_var_specs=var_specs,
    )
    cond_raw, block_slices, _ = _load_external_condition_matrix(
        condition_csv=condition_csv,
        ids=ids,
        var_specs=cond_specs,
    )
    ext_marg = {var: cond_raw[:, sl].copy() for var, sl in block_slices.items()}
    ext_marg = _augment_ext_marginals_from_cross(
        cond_raw=cond_raw,
        block_slices=block_slices,
        ext_marg=ext_marg,
    )

    stage1_model, _ = _load_stage1_model(
        checkpoint_path=stage1_checkpoint,
        timesteps=int(stage1_timesteps),
        seed=int(stage1_seed),
    )
    if stage1_device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(stage1_device)
    stage1_model.to(device)

    cond_t = torch.from_numpy(cond_raw).to(device=device, dtype=torch.float32)
    coarse_raw = stage1_model.predict_coarse(cond_raw=cond_t).detach().cpu().numpy().astype(np.float64)
    coarse_raw = coarse_raw / np.maximum(coarse_raw.sum(axis=1, keepdims=True), 1e-12)

    coarse_ipf = np.zeros_like(coarse_raw, dtype=np.float64)
    for row_idx in range(cond_raw.shape[0]):
        ext_row = {var: np.asarray(ext_marg[var][row_idx], dtype=np.float64) for var in FULL_VARIABLE_ORDER}
        coarse_targets = _coarse_marginals_from_full_ext(ext_row)
        coarse_proj = _ipf_nd(
            seed_joint=coarse_raw[row_idx].reshape(COARSE_SHAPE),
            target_marginals=coarse_targets,
            shape=COARSE_SHAPE,
            max_iter=int(stage1_ipf_iters),
        )
        coarse_proj = coarse_proj / max(float(coarse_proj.sum()), 1e-12)
        coarse_ipf[row_idx] = coarse_proj.reshape(-1)

    summary = {
        "condition_source": "stage1_coarse_ipf",
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage1_schema_json": str(stage1_schema_json),
        "condition_csv": str(condition_csv),
        "condition_schema_json": str(condition_schema_json) if condition_schema_json is not None else None,
        "stage1_timesteps": int(stage1_timesteps),
        "stage1_ipf_iters": int(stage1_ipf_iters),
        "stage1_seed": int(stage1_seed),
        "stage1_device": str(device),
        "coarse_raw_mean_entropy": float(np.mean(-np.sum(coarse_raw * np.log(np.clip(coarse_raw, 1e-12, None)), axis=1))),
        "coarse_ipf_mean_entropy": float(np.mean(-np.sum(coarse_ipf * np.log(np.clip(coarse_ipf, 1e-12, None)), axis=1))),
    }
    return coarse_ipf, summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_external_c2f_full_income_v2_teacher")
    ap.add_argument("--joint_wide_csv", required=True, help="Full 5-way full-income v2 target joint_wide csv (K=5000).")
    ap.add_argument("--out_dir", default=None, help="Default: sibling processed/external_c2f directory.")
    ap.add_argument(
        "--child_mask_mode",
        choices=["parent_all", "global_nonzero"],
        default="parent_all",
        help="How to define active child slots inside each coarse parent.",
    )
    ap.add_argument(
        "--child_support_eps",
        type=float,
        default=0.0,
        help="Minimum global support mass for a fine cell to remain active when child_mask_mode=global_nonzero.",
    )
    ap.add_argument("--use_stage1_coarse_ipf_for_condition", action="store_true")
    ap.add_argument("--append_true_coarse_rows", action="store_true")
    ap.add_argument("--stage1_checkpoint", default=None)
    ap.add_argument("--stage1_schema_json", default=None)
    ap.add_argument("--stage1_condition_csv", default=None)
    ap.add_argument("--stage1_condition_schema_json", default=None)
    ap.add_argument("--stage1_timesteps", type=int, default=200)
    ap.add_argument("--stage1_ipf_iters", type=int, default=200)
    ap.add_argument("--stage1_seed", type=int, default=0)
    ap.add_argument("--stage1_device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_path = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"joint_wide_csv not found: {in_path}")

    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = in_path.parent.parent / "external_c2f"
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.use_stage1_coarse_ipf_for_condition) and bool(args.append_true_coarse_rows):
        stem = "extc2f_full_income_v2_stage1ipfcondmix_pums_2023_puma_us"
    elif bool(args.use_stage1_coarse_ipf_for_condition):
        stem = "extc2f_full_income_v2_stage1ipfcond_pums_2023_puma_us"
    else:
        stem = "extc2f_full_income_v2_teacher_pums_2023_puma_us"
    wide_csv = out_dir / f"{stem}_wide.csv"
    schema_json = out_dir / f"{stem}.schema.json"
    metadata_json = out_dir / f"{stem}.metadata.json"
    if any(p.exists() for p in [wide_csv, schema_json, metadata_json]) and not args.overwrite:
        raise SystemExit(f"output exists under {out_dir} (use --overwrite)")

    df = pd.read_csv(in_path, low_memory=False)
    req = {"statefp", "puma", "puma_uid", "total_person_weight"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"joint_wide_csv missing columns: {miss}")
    df["statefp"] = df["statefp"].map(_canon_statefp)
    df["puma5"] = df["puma"].map(_canon_puma5)
    df["puma_uid"] = df.apply(lambda row: _canon_uid(row["statefp"], row["puma5"]), axis=1)

    p_joint_cols = [f"p_joint_{i:03d}" for i in range(FULL_K)]
    miss_joint = [c for c in p_joint_cols if c not in df.columns]
    if miss_joint:
        raise SystemExit(f"joint_wide_csv missing joint columns: {miss_joint[:5]}")

    p_joint_all = df[p_joint_cols].to_numpy(dtype=np.float64)
    p_joint_all = np.clip(p_joint_all, 0.0, None)
    global_support_mass = p_joint_all.sum(axis=0, dtype=np.float64)
    global_support_mask = global_support_mass > float(args.child_support_eps)

    if str(args.child_mask_mode) == "global_nonzero":
        effective_parent_child_mask = np.zeros_like(PARENT_CHILD_SLOT_MASK, dtype=np.float64)
        for parent_idx, children in enumerate(PARENT_TO_CHILD_FULL):
            effective_parent_child_mask[parent_idx, : int(children.shape[0])] = global_support_mask[children].astype(np.float64)
    else:
        effective_parent_child_mask = PARENT_CHILD_SLOT_MASK.astype(np.float64)

    condition_summary: dict[str, Any]
    coarse_condition_by_puma: np.ndarray | None = None
    if bool(args.use_stage1_coarse_ipf_for_condition):
        stage1_checkpoint = pathlib.Path(str(args.stage1_checkpoint)).expanduser().resolve() if args.stage1_checkpoint else None
        stage1_schema_json = pathlib.Path(str(args.stage1_schema_json)).expanduser().resolve() if args.stage1_schema_json else None
        stage1_condition_csv = pathlib.Path(str(args.stage1_condition_csv)).expanduser().resolve() if args.stage1_condition_csv else None
        stage1_condition_schema_json = pathlib.Path(str(args.stage1_condition_schema_json)).expanduser().resolve() if args.stage1_condition_schema_json else None
        req_paths = {
            "stage1_checkpoint": stage1_checkpoint,
            "stage1_schema_json": stage1_schema_json,
            "stage1_condition_csv": stage1_condition_csv,
        }
        missing = [name for name, path in req_paths.items() if path is None or not path.exists()]
        if missing:
            raise SystemExit(f"missing required stage1 path(s) for conditioned coarse: {missing}")
        if stage1_condition_schema_json is not None and not stage1_condition_schema_json.exists():
            raise SystemExit(f"stage1_condition_schema_json not found: {stage1_condition_schema_json}")

        ids = df["puma_uid"].astype(str).tolist()
        coarse_condition_by_puma, condition_summary = _build_stage1_ipf_conditioned_coarse(
            ids=ids,
            condition_csv=stage1_condition_csv,
            stage1_schema_json=stage1_schema_json,
            condition_schema_json=stage1_condition_schema_json,
            stage1_checkpoint=stage1_checkpoint,
            stage1_timesteps=int(args.stage1_timesteps),
            stage1_ipf_iters=int(args.stage1_ipf_iters),
            stage1_seed=int(args.stage1_seed),
            stage1_device=args.stage1_device,
        )
    else:
        condition_summary = {"condition_source": "true_coarse"}

    n_zero_parent = 0
    parent_nonzero_counts: list[int] = []
    condition_source_counts: dict[str, int] = {}
    regime_active_counts: list[int] = []
    n_rows_written = 0

    base_fieldnames = [
        "statefp",
        "puma",
        "puma_uid",
        "parent_idx",
        "parent_uid",
        "parent_mass",
        "total_person_weight",
        "child_count",
        "AGEP_bin_lite",
        "SEX",
        "SCHL_allpop_lite",
        "ESR_allpop_lite",
        "PINCP_allpop_bin_lite",
        "condition_source",
    ]
    fieldnames = (
        base_fieldnames
        + [f"c_coarse_{i:03d}" for i in range(COARSE_K)]
        + [f"c_parent_{i:03d}" for i in range(COARSE_K)]
        + [f"c_child_mask_{i:03d}" for i in range(MAX_CHILDREN)]
        + ["c_parent_mass"]
        + [f"aux_income_regime_target_{i:02d}" for i in range(len(INCOME_LABELS))]
        + [f"aux_income_regime_mask_{i:02d}" for i in range(len(INCOME_LABELS))]
        + [f"p_joint_{i:03d}" for i in range(MAX_CHILDREN)]
    )

    with wide_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row_idx in range(len(df)):
            r = df.iloc[row_idx]
            p_full = np.asarray(p_joint_all[row_idx], dtype=np.float64)
            puma_uid = str(r["puma_uid"])
            total_person_weight = float(r["total_person_weight"])
            statefp = str(r["statefp"])
            puma = str(r["puma"])

            p_full = np.clip(p_full, 0.0, None)
            p_full = p_full / max(float(p_full.sum()), 1e-12)
            p_coarse_true = coarse_from_full_flat(p_full)
            if coarse_condition_by_puma is not None:
                p_coarse_cond = np.asarray(coarse_condition_by_puma[row_idx], dtype=np.float64)
            else:
                p_coarse_cond = p_coarse_true

            if bool(args.use_stage1_coarse_ipf_for_condition) and bool(args.append_true_coarse_rows):
                condition_variants = [
                    ("true_coarse", p_coarse_true),
                    ("stage1_coarse_ipf", p_coarse_cond),
                ]
            elif bool(args.use_stage1_coarse_ipf_for_condition):
                condition_variants = [("stage1_coarse_ipf", p_coarse_cond)]
            else:
                condition_variants = [("true_coarse", p_coarse_true)]

            n_nonzero_this = 0
            for parent_idx in range(COARSE_K):
                children = PARENT_TO_CHILD_FULL[parent_idx]
                parent_mass = float(p_full[children].sum())
                if parent_mass <= 0.0:
                    n_zero_parent += 1
                    continue

                parent_meta = parent_index_labels(parent_idx)
                local = p_full[children] / parent_mass
                child_mask = effective_parent_child_mask[parent_idx].astype(np.float64)
                active_child_count = int((child_mask > 0.5).sum())
                if active_child_count <= 0:
                    continue
                n_nonzero_this += 1
                regime_target = np.zeros((len(INCOME_LABELS),), dtype=np.float64)
                regime_mask = np.zeros((len(INCOME_LABELS),), dtype=np.float64)
                for slot_idx, child_idx in enumerate(children.tolist()):
                    regime_idx = int(CHILD_INCOME_AUX_FULL[int(child_idx)])
                    regime_target[regime_idx] += float(local[slot_idx])
                    if slot_idx < child_mask.shape[0] and child_mask[slot_idx] > 0.5:
                        regime_mask[regime_idx] = 1.0
                regime_target = regime_target / max(float(regime_target.sum()), 1e-12)
                regime_active_counts.append(int((regime_mask > 0.5).sum()))

                row: dict[str, Any] = {
                    "statefp": statefp,
                    "puma": puma,
                    "puma_uid": puma_uid,
                    "parent_idx": int(parent_idx),
                    "parent_uid": f"{puma_uid}__parent{int(parent_idx):03d}",
                    "parent_mass": float(parent_mass),
                    "total_person_weight": total_person_weight,
                    "child_count": int(active_child_count),
                }
                row.update(parent_meta)

                parent_onehot = np.zeros((COARSE_K,), dtype=np.float64)
                parent_onehot[parent_idx] = 1.0
                padded = np.zeros((MAX_CHILDREN,), dtype=np.float64)
                padded[: int(children.shape[0])] = local

                cond_vals_by_source = {name: arr.tolist() for name, arr in condition_variants}
                parent_onehot_vals = parent_onehot.tolist()
                child_mask_vals = child_mask.tolist()
                regime_target_vals = regime_target.tolist()
                regime_mask_vals = regime_mask.tolist()
                padded_vals = padded.tolist()

                for condition_source, p_coarse_variant in condition_variants:
                    row_out = dict(row)
                    if len(condition_variants) > 1:
                        row_out["parent_uid"] = f"{row_out['parent_uid']}__{condition_source}"
                    row_out["condition_source"] = str(condition_source)
                    for i, v in enumerate(cond_vals_by_source[condition_source]):
                        row_out[f"c_coarse_{i:03d}"] = float(v)
                    for i, v in enumerate(parent_onehot_vals):
                        row_out[f"c_parent_{i:03d}"] = float(v)
                    for i, v in enumerate(child_mask_vals):
                        row_out[f"c_child_mask_{i:03d}"] = float(v)
                    row_out["c_parent_mass"] = float(p_coarse_variant[parent_idx])
                    for i, v in enumerate(regime_target_vals):
                        row_out[f"aux_income_regime_target_{i:02d}"] = float(v)
                    for i, v in enumerate(regime_mask_vals):
                        row_out[f"aux_income_regime_mask_{i:02d}"] = float(v)
                    for i, v in enumerate(padded_vals):
                        row_out[f"p_joint_{i:03d}"] = float(v)
                    writer.writerow(row_out)
                    n_rows_written += 1
                    condition_source_counts[condition_source] = condition_source_counts.get(condition_source, 0) + 1
            parent_nonzero_counts.append(int(n_nonzero_this))

    schema = {
        "schema": "external_c2f_full_income_v2_teacher",
        "created_at": _utc_now_iso(),
        "full_variable_order": FULL_VARIABLE_ORDER,
        "full_shape": list(FULL_SHAPE),
        "full_K": int(FULL_K),
        "coarse_variable_order": COARSE_VARIABLE_ORDER,
        "coarse_shape": list(COARSE_SHAPE),
        "coarse_K": int(COARSE_K),
        "coarse_categories": COARSE_CATEGORIES,
        "target_dim": int(MAX_CHILDREN),
        "condition_blocks": {
            "coarse_table": [f"c_coarse_{i:03d}" for i in range(COARSE_K)],
            "parent_onehot": [f"c_parent_{i:03d}" for i in range(COARSE_K)],
            "child_mask": [f"c_child_mask_{i:03d}" for i in range(MAX_CHILDREN)],
            "parent_mass": ["c_parent_mass"],
        },
        "auxiliary_blocks": {
            "income_regime_target": [f"aux_income_regime_target_{i:02d}" for i in range(len(INCOME_LABELS))],
            "income_regime_mask": [f"aux_income_regime_mask_{i:02d}" for i in range(len(INCOME_LABELS))],
        },
        "income_regime_labels": list(INCOME_LABELS),
        "income_regime_semantics": "fine_income_bin_within_coarse_income_lite",
        "target_block": [f"p_joint_{i:03d}" for i in range(MAX_CHILDREN)],
        "parent_to_child_full_indices_padded": PADDED_PARENT_CHILD_FULL.astype(int).tolist(),
        "parent_child_income_regime_padded": PADDED_PARENT_CHILD_INCOME_REGIME.astype(int).tolist(),
        "parent_child_slot_mask": effective_parent_child_mask.astype(int).tolist(),
    }
    parent_size_hist: dict[str, int] = {}
    for parent_idx in range(COARSE_K):
        cnt = int((effective_parent_child_mask[parent_idx] > 0.5).sum())
        parent_size_hist[str(cnt)] = int(parent_size_hist.get(str(cnt), 0) + 1)
    schema["parent_size_histogram"] = parent_size_hist
    schema_json.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta = {
        "schema": "external_c2f_full_income_v2_teacher",
        "created_at": _utc_now_iso(),
        "source_joint_wide_csv": str(in_path),
        "outputs": {
            "wide_csv": str(wide_csv),
            "schema_json": str(schema_json),
        },
        "condition_summary": condition_summary,
        "condition_source_counts": {str(k): int(v) for k, v in sorted(condition_source_counts.items())},
        "n_rows": int(n_rows_written),
        "n_unique_pumas": int(df["puma_uid"].nunique()),
        "n_zero_parent_skipped": int(n_zero_parent),
        "mean_nonzero_parents_per_puma": float(np.mean(parent_nonzero_counts)) if parent_nonzero_counts else None,
        "target_dim": int(MAX_CHILDREN),
        "coarse_K": int(COARSE_K),
        "child_mask_mode": str(args.child_mask_mode),
        "child_support_eps": float(args.child_support_eps),
        "global_support_nonzero_cells": int(global_support_mask.sum()),
        "global_support_zero_cells": int((~global_support_mask).sum()),
        "mean_active_income_regimes_per_parent": float(np.mean(regime_active_counts)) if regime_active_counts else None,
    }
    metadata_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {wide_csv}")


if __name__ == "__main__":
    main()
