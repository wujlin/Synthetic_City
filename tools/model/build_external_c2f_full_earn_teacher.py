#!/usr/bin/env python3
from __future__ import annotations

"""
Build a teacher-forced stage-2 coarse-to-fine dataset for the 5-way full-earn target.

Stage-1 coarse schema:
  AGEP_lite(4) x SEX(2) x SCHL_lite(3) x ESR_lite(3) x EARN_lite(4) = 288

Stage-2 target:
  For each (PUMA, coarse parent cell), predict the local fine split among the
  child cells that map into that parent. The local target is represented in a
  padded child-slot coordinate system of width MAX_CHILDREN.

This builder uses ground-truth parent mass and ground-truth coarse tables
derived from the full 5-way target, so it is strictly a teacher-forced
learnability probe.
"""

import argparse
import json
import pathlib
import re
import sys
from typing import Any

import numpy as np
import pandas as pd


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.build_external_target_v1_michigan import _utc_now_iso
from tools.model.eval_external_c2f_full_earn_pipeline import _coarse_marginals_from_full_ext, _load_stage1_model
from tools.model.external_c2f_full_earn_schema import (
    COARSE_CATEGORIES,
    COARSE_K,
    COARSE_PRESET,
    COARSE_SHAPE,
    COARSE_VARIABLE_ORDER,
    FULL_K,
    FULL_SHAPE,
    FULL_VARIABLE_ORDER,
    MAX_CHILDREN,
    PADDED_PARENT_CHILD_FULL,
    PARENT_CHILD_SLOT_MASK,
    PARENT_TO_CHILD_FULL,
    coarse_from_full_flat,
    parent_count_histogram,
    parent_index_labels,
)
from tools.model.train_external_joint_hier_diffusion_full import _augment_ext_marginals_from_cross
from tools.model.train_us_puma_5var_diffusion import _canon_puma5, _canon_statefp, _canon_uid, _ipf_nd, _require_torch
from tools.model.train_us_puma_external_v1_diffusion import (
    _append_condition_extra_matrix,
    _load_condition_specs_from_schema,
    _load_external_condition_matrix,
    _load_var_specs_from_schema,
)


def _infer_year_scope(path: pathlib.Path) -> tuple[int, str]:
    match = re.search(r"exttarget_v1_full_earn_pums_(\d{4})_puma_(.+)_joint_wide\.csv$", path.name)
    if not match:
        return 2023, "us"
    return int(match.group(1)), str(match.group(2))


def _build_stage1_ipf_conditioned_coarse(
    *,
    ids: list[str],
    condition_csv: pathlib.Path,
    condition_extra_csv: pathlib.Path | None,
    condition_extra_standardize: str,
    condition_extra_missing_policy: str,
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
    cond_raw, block_slices, cond_meta = _load_external_condition_matrix(
        condition_csv=condition_csv,
        ids=ids,
        var_specs=cond_specs,
    )
    cond_raw, cond_meta = _append_condition_extra_matrix(
        cond_raw=cond_raw,
        cond_meta=cond_meta,
        extra_csv=condition_extra_csv,
        ids=ids,
        standardize=condition_extra_standardize,
        missing_policy=condition_extra_missing_policy,
    )
    ext_marg = {var: cond_raw[:, sl].copy() for var, sl in block_slices.items()}
    ext_marg = _augment_ext_marginals_from_cross(
        cond_raw=cond_raw,
        block_slices=block_slices,
        ext_marg=ext_marg,
    )

    stage1_model, stage1_payload = _load_stage1_model(
        checkpoint_path=stage1_checkpoint,
        timesteps=int(stage1_timesteps),
        seed=int(stage1_seed),
    )
    expected_cond_dim = int(stage1_payload["cond_raw_dim"])
    if int(cond_raw.shape[1]) != expected_cond_dim:
        raise SystemExit(
            "Stage-1 condition dimension mismatch while building C2F teacher data: "
            f"condition matrix has {cond_raw.shape[1]}, checkpoint expects {expected_cond_dim}. "
            "Use the same stage1_condition_extra_* settings used for training."
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
    coarse_targets_by_row: list[list[np.ndarray]] = []
    for row_idx in range(cond_raw.shape[0]):
        ext_row = {var: np.asarray(ext_marg[var][row_idx], dtype=np.float64) for var in FULL_VARIABLE_ORDER}
        coarse_targets = _coarse_marginals_from_full_ext(ext_row)
        coarse_targets_by_row.append(coarse_targets)
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
        "condition_meta": cond_meta,
        "condition_extra_csv": str(condition_extra_csv) if condition_extra_csv is not None else None,
        "condition_extra_standardize": str(condition_extra_standardize),
        "condition_extra_missing_policy": str(condition_extra_missing_policy),
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
    ap = argparse.ArgumentParser(prog="build_external_c2f_full_earn_teacher")
    ap.add_argument("--joint_wide_csv", required=True, help="Full 5-way external target joint_wide csv (K=3000).")
    ap.add_argument("--out_dir", default=None, help="Default: sibling processed/external_c2f directory.")
    ap.add_argument("--use_stage1_coarse_ipf_for_condition", action="store_true")
    ap.add_argument("--append_true_coarse_rows", action="store_true")
    ap.add_argument("--stage1_checkpoint", default=None)
    ap.add_argument("--stage1_schema_json", default=None)
    ap.add_argument("--stage1_condition_csv", default=None)
    ap.add_argument("--stage1_condition_schema_json", default=None)
    ap.add_argument("--stage1_condition_extra_csv", default=None)
    ap.add_argument("--stage1_condition_extra_standardize", choices=["none", "zscore"], default="none")
    ap.add_argument("--stage1_condition_extra_missing_policy", choices=["require", "zero"], default="require")
    ap.add_argument("--stage1_timesteps", type=int, default=200)
    ap.add_argument("--stage1_ipf_iters", type=int, default=200)
    ap.add_argument("--stage1_seed", type=int, default=0)
    ap.add_argument("--stage1_device", default=None)
    ap.add_argument("--output_stem", default=None)
    ap.add_argument(
        "--flush_rows",
        type=int,
        default=0,
        help=(
            "If positive, stream rows to CSV every N rows instead of materializing "
            "the whole stage-2 table in memory. This is useful for wide fine "
            "coarse presets."
        ),
    )
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

    inferred_year, inferred_scope = _infer_year_scope(in_path)
    if args.output_stem:
        stem = str(args.output_stem)
    elif bool(args.use_stage1_coarse_ipf_for_condition) and bool(args.append_true_coarse_rows):
        stem = f"extc2f_full_earn_stage1ipfcondmix_pums_{inferred_year}_puma_{inferred_scope}"
    elif bool(args.use_stage1_coarse_ipf_for_condition):
        stem = f"extc2f_full_earn_stage1ipfcond_pums_{inferred_year}_puma_{inferred_scope}"
    else:
        stem = f"extc2f_full_earn_teacher_pums_{inferred_year}_puma_{inferred_scope}"
    wide_csv = out_dir / f"{stem}_wide.csv"
    schema_json = out_dir / f"{stem}.schema.json"
    metadata_json = out_dir / f"{stem}.metadata.json"
    if any(p.exists() for p in [wide_csv, schema_json, metadata_json]) and not args.overwrite:
        raise SystemExit(f"output exists under {out_dir} (use --overwrite)")
    if bool(args.overwrite):
        for path in [wide_csv, schema_json, metadata_json]:
            if path.exists():
                path.unlink()

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

    condition_summary: dict[str, Any]
    coarse_condition_by_puma: np.ndarray | None = None
    if bool(args.use_stage1_coarse_ipf_for_condition):
        stage1_checkpoint = pathlib.Path(str(args.stage1_checkpoint)).expanduser().resolve() if args.stage1_checkpoint else None
        stage1_schema_json = pathlib.Path(str(args.stage1_schema_json)).expanduser().resolve() if args.stage1_schema_json else None
        stage1_condition_csv = pathlib.Path(str(args.stage1_condition_csv)).expanduser().resolve() if args.stage1_condition_csv else None
        stage1_condition_schema_json = pathlib.Path(str(args.stage1_condition_schema_json)).expanduser().resolve() if args.stage1_condition_schema_json else None
        stage1_condition_extra_csv = pathlib.Path(str(args.stage1_condition_extra_csv)).expanduser().resolve() if args.stage1_condition_extra_csv else None
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
        if stage1_condition_extra_csv is not None and not stage1_condition_extra_csv.exists():
            raise SystemExit(f"stage1_condition_extra_csv not found: {stage1_condition_extra_csv}")

        ids = df["puma_uid"].astype(str).tolist()
        coarse_condition_by_puma, condition_summary = _build_stage1_ipf_conditioned_coarse(
            ids=ids,
            condition_csv=stage1_condition_csv,
            condition_extra_csv=stage1_condition_extra_csv,
            condition_extra_standardize=str(args.stage1_condition_extra_standardize),
            condition_extra_missing_policy=str(args.stage1_condition_extra_missing_policy),
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

    rows: list[dict[str, Any]] = []
    flush_rows = int(args.flush_rows)
    wrote_header = False
    n_rows_written = 0
    n_zero_parent = 0
    parent_nonzero_counts: list[int] = []
    condition_source_counts: dict[str, int] = {}

    def _flush_rows() -> None:
        nonlocal rows, wrote_header, n_rows_written
        if not rows:
            return
        chunk = pd.DataFrame(rows)
        chunk.to_csv(
            wide_csv,
            mode="a" if wrote_header else "w",
            index=False,
            header=not wrote_header,
        )
        wrote_header = True
        n_rows_written += int(chunk.shape[0])
        rows = []

    for row_idx, r in enumerate(df.to_dict(orient="records")):
        p_full = np.asarray([float(r[c]) for c in p_joint_cols], dtype=np.float64)
        p_full = np.clip(p_full, 0.0, None)
        p_full = p_full / max(float(p_full.sum()), 1e-12)
        p_coarse_true = coarse_from_full_flat(p_full)
        if coarse_condition_by_puma is not None:
            p_coarse_cond = np.asarray(coarse_condition_by_puma[row_idx], dtype=np.float64)
        else:
            p_coarse_cond = p_coarse_true
        puma_uid = str(r["puma_uid"])

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

            n_nonzero_this += 1
            parent_meta = parent_index_labels(parent_idx)
            local = p_full[children] / parent_mass
            row: dict[str, Any] = {
                "statefp": str(r["statefp"]),
                "puma": str(r["puma"]),
                "puma_uid": puma_uid,
                "parent_idx": int(parent_idx),
                "parent_uid": f"{puma_uid}__parent{int(parent_idx):03d}",
                "parent_mass": float(parent_mass),
                "total_person_weight": float(r["total_person_weight"]),
                "child_count": int(children.shape[0]),
            }
            row.update(parent_meta)

            parent_onehot = np.zeros((COARSE_K,), dtype=np.float64)
            parent_onehot[parent_idx] = 1.0
            child_mask = PARENT_CHILD_SLOT_MASK[parent_idx].astype(np.float64)
            padded = np.zeros((MAX_CHILDREN,), dtype=np.float64)
            padded[: int(children.shape[0])] = local

            for condition_source, p_coarse_variant in condition_variants:
                row_out = dict(row)
                if len(condition_variants) > 1:
                    row_out["parent_uid"] = f"{row_out['parent_uid']}__{condition_source}"
                row_out["condition_source"] = str(condition_source)
                for i, v in enumerate(p_coarse_variant.tolist()):
                    row_out[f"c_coarse_{i:03d}"] = float(v)
                for i, v in enumerate(parent_onehot.tolist()):
                    row_out[f"c_parent_{i:03d}"] = float(v)
                for i, v in enumerate(child_mask.tolist()):
                    row_out[f"c_child_mask_{i:03d}"] = float(v)
                row_out["c_parent_mass"] = float(p_coarse_variant[parent_idx])
                for i, v in enumerate(padded.tolist()):
                    row_out[f"p_joint_{i:03d}"] = float(v)
                rows.append(row_out)
                condition_source_counts[condition_source] = condition_source_counts.get(condition_source, 0) + 1
                if flush_rows > 0 and len(rows) >= flush_rows:
                    _flush_rows()
        parent_nonzero_counts.append(int(n_nonzero_this))

    if flush_rows > 0:
        _flush_rows()
        n_rows = int(n_rows_written)
    else:
        wide = pd.DataFrame(rows)
        wide.to_csv(wide_csv, index=False)
        n_rows = int(wide.shape[0])

    schema = {
        "schema": "external_c2f_full_earn_teacher",
        "created_at": _utc_now_iso(),
        "full_variable_order": FULL_VARIABLE_ORDER,
        "full_shape": list(FULL_SHAPE),
        "full_K": int(FULL_K),
        "coarse_variable_order": COARSE_VARIABLE_ORDER,
        "coarse_preset": str(COARSE_PRESET),
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
        "target_block": [f"p_joint_{i:03d}" for i in range(MAX_CHILDREN)],
        "parent_to_child_full_indices_padded": PADDED_PARENT_CHILD_FULL.astype(int).tolist(),
        "parent_child_slot_mask": PARENT_CHILD_SLOT_MASK.astype(int).tolist(),
        "parent_size_histogram": {str(k): int(v) for k, v in parent_count_histogram().items()},
    }
    schema_json.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta = {
        "schema": "external_c2f_full_earn_teacher",
        "created_at": _utc_now_iso(),
        "pums_year": int(inferred_year),
        "scope": str(inferred_scope),
        "source_joint_wide_csv": str(in_path),
        "outputs": {
            "wide_csv": str(wide_csv),
            "schema_json": str(schema_json),
        },
        "condition_summary": condition_summary,
        "condition_source_counts": {str(k): int(v) for k, v in sorted(condition_source_counts.items())},
        "n_rows": int(n_rows),
        "n_unique_pumas": int(df["puma_uid"].nunique()),
        "n_zero_parent_skipped": int(n_zero_parent),
        "mean_nonzero_parents_per_puma": float(np.mean(parent_nonzero_counts)) if parent_nonzero_counts else None,
        "target_dim": int(MAX_CHILDREN),
        "coarse_preset": str(COARSE_PRESET),
        "coarse_K": int(COARSE_K),
        "flush_rows": int(flush_rows),
    }
    metadata_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {wide_csv}")


if __name__ == "__main__":
    main()
