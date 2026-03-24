#!/usr/bin/env python3
from __future__ import annotations

"""
Build a teacher-forced coarse-to-fine stage-2 dataset for age x education refinement.

Stage-1 coarse schema:
  AGEP_lite(4) x SEX(2) x SCHL_lite(3) x ESR_lite(3)

Stage-2 target:
  For each (PUMA, SEX, ESR_lite) subgroup, predict the fine conditional table
  AGEP_fine(10) x SCHL_fine(5), given the coarse conditional table
  AGEP_lite(4) x SCHL_lite(3) within that subgroup.

This builder uses ground-truth coarse tables derived from the full target, so the
resulting dataset is explicitly teacher-forced and should not be interpreted as an
end-to-end evaluation artifact.
"""

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.build_external_target_v1_michigan import AGE_LABELS, ESR_LABELS, SCHL_LABELS, SEX_LABELS, _utc_now_iso
from tools.external_v1_variant_presets import AGE_LITE_LABELS, AGE_TO_LITE, ESR_LITE_LABELS, ESR_TO_LITE, SCHL_LITE_LABELS, SCHL_TO_LITE
from tools.train_us_puma_5var_diffusion import _canon_puma5, _canon_statefp, _canon_uid


FULL_SHAPE = (len(AGE_LABELS), len(SEX_LABELS), len(SCHL_LABELS), len(ESR_LABELS))
FINE_SHAPE = (len(AGE_LABELS), len(SCHL_LABELS))
COARSE_SHAPE = (len(AGE_LITE_LABELS), len(SCHL_LITE_LABELS))

AGE_FINE_TO_COARSE = np.asarray([AGE_LITE_LABELS.index(AGE_TO_LITE[x]) for x in AGE_LABELS], dtype=np.int16)
SCHL_FINE_TO_COARSE = np.asarray([SCHL_LITE_LABELS.index(SCHL_TO_LITE[x]) for x in SCHL_LABELS], dtype=np.int16)
ESR_FULL_TO_LITE = np.asarray([ESR_LITE_LABELS.index(ESR_TO_LITE[x]) for x in ESR_LABELS], dtype=np.int16)


def _aggregate_coarse_age_schl(fine: np.ndarray) -> np.ndarray:
    coarse = np.zeros(COARSE_SHAPE, dtype=np.float64)
    for ai in range(FINE_SHAPE[0]):
        ac = int(AGE_FINE_TO_COARSE[ai])
        for qi in range(FINE_SHAPE[1]):
            qc = int(SCHL_FINE_TO_COARSE[qi])
            coarse[ac, qc] += float(fine[ai, qi])
    return coarse


def _child_parent_index() -> np.ndarray:
    out = np.zeros((FINE_SHAPE[0] * FINE_SHAPE[1],), dtype=np.int16)
    k = 0
    for ai in range(FINE_SHAPE[0]):
        ac = int(AGE_FINE_TO_COARSE[ai])
        for qi in range(FINE_SHAPE[1]):
            qc = int(SCHL_FINE_TO_COARSE[qi])
            out[k] = int(ac * COARSE_SHAPE[1] + qc)
            k += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_external_c2f_age_schl_teacher")
    ap.add_argument("--joint_wide_csv", required=True, help="Full external target v1 joint_wide csv (K=500).")
    ap.add_argument("--out_dir", default=None, help="Default: sibling processed/external_c2f directory.")
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

    stem = "extc2f_age_schl_teacher_pums_2023_puma_us"
    wide_csv = out_dir / f"{stem}_wide.csv"
    schema_json = out_dir / f"{stem}.schema.json"
    metadata_json = out_dir / f"{stem}.metadata.json"
    if any(p.exists() for p in [wide_csv, schema_json, metadata_json]) and not args.overwrite:
        raise SystemExit(f"output exists under {out_dir} (use --overwrite)")

    df = pd.read_csv(in_path, low_memory=False)
    req = {"statefp", "puma", "puma_uid", "total_person_weight", "n_persons_unweighted"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"joint_wide_csv missing columns: {miss}")

    p_joint_cols = [f"p_joint_{i:03d}" for i in range(int(np.prod(FULL_SHAPE)))]
    missing_joint = [c for c in p_joint_cols if c not in df.columns]
    if missing_joint:
        raise SystemExit(f"joint_wide_csv missing joint columns: {missing_joint[:5]}")

    child_parent_idx = _child_parent_index()
    n_rows_zero_mass = 0
    rows: list[dict[str, Any]] = []

    for r in df.to_dict(orient="records"):
        p_old = np.asarray([float(r[c]) for c in p_joint_cols], dtype=np.float64).reshape(FULL_SHAPE)
        statefp = _canon_statefp(r["statefp"])
        puma5 = _canon_puma5(r.get("puma5", r["puma"]))
        puma_uid = _canon_uid(statefp, puma5)

        for si, sex_lab in enumerate(SEX_LABELS):
            for esr_lite_idx, esr_lite_lab in enumerate(ESR_LITE_LABELS):
                full_esr_idx = np.where(ESR_FULL_TO_LITE == esr_lite_idx)[0]
                fine = np.take(p_old[:, si, :, :], indices=full_esr_idx, axis=2).sum(axis=2)
                subgroup_mass = float(fine.sum())
                if subgroup_mass <= 0:
                    n_rows_zero_mass += 1
                    continue

                fine_cond = fine / subgroup_mass
                coarse_cond = _aggregate_coarse_age_schl(fine_cond)

                row: dict[str, Any] = {
                    "statefp": statefp,
                    "puma": str(int(puma5)) if puma5 else "",
                    "puma5": puma5,
                    "puma_uid": puma_uid,
                    "subgroup_sex": sex_lab,
                    "subgroup_esr": esr_lite_lab,
                    "subgroup_uid": f"{puma_uid}__sex{sex_lab}__esr{esr_lite_lab}",
                    "parent_mass": subgroup_mass,
                    "total_person_weight": float(r["total_person_weight"]),
                    "n_persons_unweighted": int(r["n_persons_unweighted"]),
                }

                sex_onehot = np.zeros((len(SEX_LABELS),), dtype=np.float64)
                sex_onehot[si] = 1.0
                esr_onehot = np.zeros((len(ESR_LITE_LABELS),), dtype=np.float64)
                esr_onehot[esr_lite_idx] = 1.0

                for i, v in enumerate(coarse_cond.reshape(-1)):
                    row[f"c_parent_{i:02d}"] = float(v)
                for i, v in enumerate(sex_onehot):
                    row[f"c_sex_{i:02d}"] = float(v)
                for i, v in enumerate(esr_onehot):
                    row[f"c_esr_{i:02d}"] = float(v)
                row["c_parent_mass"] = float(subgroup_mass)

                p_age = fine_cond.sum(axis=1)
                p_schl = fine_cond.sum(axis=0)
                for i, v in enumerate(p_age):
                    row[f"p_age_{i:02d}"] = float(v)
                for i, v in enumerate(p_schl):
                    row[f"p_schl_{i:02d}"] = float(v)
                for i, v in enumerate(fine_cond.reshape(-1)):
                    row[f"p_joint_{i:03d}"] = float(v)
                rows.append(row)

    wide = pd.DataFrame(rows)
    wide.to_csv(wide_csv, index=False)

    schema = {
        "schema": "external_c2f_age_schl_teacher",
        "created_at": _utc_now_iso(),
        "target_variable_order": ["AGEP_bin", "SCHL_allpop"],
        "target_shape": list(FINE_SHAPE),
        "target_K": int(np.prod(FINE_SHAPE)),
        "target_categories": {
            "AGEP_bin": AGE_LABELS,
            "SCHL_allpop": SCHL_LABELS,
        },
        "coarse_variable_order": ["AGEP_bin_lite", "SCHL_allpop_lite"],
        "coarse_shape": list(COARSE_SHAPE),
        "coarse_K": int(np.prod(COARSE_SHAPE)),
        "coarse_categories": {
            "AGEP_bin_lite": AGE_LITE_LABELS,
            "SCHL_allpop_lite": SCHL_LITE_LABELS,
        },
        "subgroup_variables": {
            "SEX": SEX_LABELS,
            "ESR_allpop_lite": ESR_LITE_LABELS,
        },
        "condition_blocks": {
            "parent_table": [f"c_parent_{i:02d}" for i in range(int(np.prod(COARSE_SHAPE)))],
            "sex": [f"c_sex_{i:02d}" for i in range(len(SEX_LABELS))],
            "esr": [f"c_esr_{i:02d}" for i in range(len(ESR_LITE_LABELS))],
            "parent_mass": ["c_parent_mass"],
        },
        "child_parent_index": child_parent_idx.astype(int).tolist(),
    }
    schema_json.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta = {
        "schema": "external_c2f_age_schl_teacher",
        "created_at": _utc_now_iso(),
        "source_joint_wide_csv": str(in_path),
        "outputs": {
            "wide_csv": str(wide_csv),
            "schema_json": str(schema_json),
        },
        "n_rows": int(wide.shape[0]),
        "n_unique_pumas": int(wide["puma_uid"].nunique()),
        "n_rows_zero_mass_skipped": int(n_rows_zero_mass),
        "target_shape": list(FINE_SHAPE),
        "target_K": int(np.prod(FINE_SHAPE)),
        "coarse_shape": list(COARSE_SHAPE),
        "coarse_K": int(np.prod(COARSE_SHAPE)),
        "subgroup_count_per_puma_max": int(len(SEX_LABELS) * len(ESR_LITE_LABELS)),
    }
    metadata_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {wide_csv}")


if __name__ == "__main__":
    main()
