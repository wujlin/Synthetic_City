#!/usr/bin/env python3
from __future__ import annotations

"""
Export compact dataset examples for selected PUMAs.

Outputs:
- a compact aggregate snapshot for each selected PUMA
- a few sampled synthetic rows from the generated joint distribution
- a JSON metadata file with basic diagnostics
"""

import argparse
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(pathlib.Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.synthpop.model.diffusion_tabular import DiffusionTabularModel
from _eval_5var_common import _marginal_from_joint, _tvd, infer_one_region, load_eval_data, write_json


DEFAULT_PUMA_UIDS = ["2602903", "2601100"]
DEFAULT_REGION_LABELS = {
    "2602903": "PUMA 2602903",
    "2601100": "PUMA 2601100",
}

AGE_LABELS = ["Young", "Old"]
SEX_LABELS = ["Male", "Female"]
INCOME_LABELS = ["Low", "High"]
EDU_LABELS = ["Low", "High"]
EMPLOY_LABELS = ["Employed", "Not employed"]


def _parse_csv_list(spec: str) -> list[str]:
    return [x.strip() for x in str(spec).split(",") if x.strip()]


def _select_fixed_rows(*, data: Any, puma_uids: list[str]) -> list[int]:
    id_to_idx = {str(pid): int(i) for i, pid in enumerate(data.ids)}
    missing = [pid for pid in puma_uids if pid not in id_to_idx]
    if missing:
        raise SystemExit(f"Requested PUMA(s) not found in eval data: {missing}")
    return [id_to_idx[pid] for pid in puma_uids]


def _decode_joint_cell(idx: int, *, shape: tuple[int, ...]) -> dict[str, str]:
    age, sex, income, edu, employ = np.unravel_index(int(idx), shape)
    return {
        "AGEP": AGE_LABELS[int(age)],
        "SEX": SEX_LABELS[int(sex)],
        "PINCP": INCOME_LABELS[int(income)],
        "SCHL": EDU_LABELS[int(edu)],
        "ESR": EMPLOY_LABELS[int(employ)],
    }


def _counts_from_marginal(p: np.ndarray, total: float) -> np.ndarray:
    out = np.asarray(p, dtype=float) * float(total)
    return np.rint(out).astype(int)


def _build_constraint_row(*, data: Any, row_idx: int, region_label: str) -> dict[str, Any]:
    total = float(data.totals[row_idx])
    p_true = data.p_joint[row_idx]
    age = _counts_from_marginal(_marginal_from_joint(p_true, shape=data.shape, axis=0), total)
    sex = _counts_from_marginal(_marginal_from_joint(p_true, shape=data.shape, axis=1), total)
    income = _counts_from_marginal(_marginal_from_joint(p_true, shape=data.shape, axis=2), total)
    edu = _counts_from_marginal(_marginal_from_joint(p_true, shape=data.shape, axis=3), total)
    employ = _counts_from_marginal(_marginal_from_joint(p_true, shape=data.shape, axis=4), total)

    statefp = str(data.df.iloc[row_idx]["statefp"]).zfill(2)
    puma = str(int(data.df.iloc[row_idx]["puma"]))
    return {
        "puma_uid": str(data.ids[row_idx]),
        "statefp": statefp,
        "puma": puma,
        "region_label": region_label,
        "total_population": int(round(total)),
        "age_young": int(age[0]),
        "age_old": int(age[1]),
        "sex_male": int(sex[0]),
        "sex_female": int(sex[1]),
        "income_low": int(income[0]),
        "income_high": int(income[1]),
        "education_low": int(edu[0]),
        "education_high": int(edu[1]),
        "employment_employed": int(employ[0]),
        "employment_not_employed": int(employ[1]),
    }


def _sample_synthetic_rows(
    *,
    p_hat: np.ndarray,
    shape: tuple[int, ...],
    n_rows: int,
    rng: np.random.Generator,
    puma_uid: str,
    region_label: str,
    statefp: str,
    puma: str,
) -> list[dict[str, Any]]:
    probs = np.asarray(p_hat, dtype=float).reshape(-1)
    probs = probs / max(float(probs.sum()), 1e-12)
    cell_idx = rng.choice(probs.size, size=int(n_rows), replace=True, p=probs)
    rows: list[dict[str, Any]] = []
    for i, idx in enumerate(cell_idx.tolist(), start=1):
        decoded = _decode_joint_cell(int(idx), shape=shape)
        rows.append(
            {
                "puma_uid": puma_uid,
                "statefp": statefp,
                "puma": puma,
                "region_label": region_label,
                "row_id": int(i),
                "AGEP": decoded["AGEP"],
                "SEX": decoded["SEX"],
                "PINCP": decoded["PINCP"],
                "SCHL": decoded["SCHL"],
                "ESR": decoded["ESR"],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(prog="export_dataset_examples")
    ap.add_argument("--joint_wide_csv", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--condition", choices=["none", "marginal", "pairwise", "marginal_pairwise"], default="pairwise")
    ap.add_argument("--eval_mode", choices=["leave_mi_out", "mi_kfold"], default="leave_mi_out")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold_index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_eval_joint_samples", type=int, default=128)
    ap.add_argument("--posthoc_ipf", action="store_true")
    ap.add_argument("--ipf_iters", type=int, default=200)
    ap.add_argument("--puma_uids", default="2602903,2601100")
    ap.add_argument("--region_labels", default="PUMA 2602903,PUMA 2601100")
    ap.add_argument("--rows_per_puma", type=int, default=6)
    ap.add_argument("--out_constraints_csv", required=True)
    ap.add_argument("--out_samples_csv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    joint_csv = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    ckpt = pathlib.Path(args.checkpoint).expanduser().resolve()
    out_constraints_csv = pathlib.Path(args.out_constraints_csv).expanduser().resolve()
    out_samples_csv = pathlib.Path(args.out_samples_csv).expanduser().resolve()
    out_json = pathlib.Path(args.out_json).expanduser().resolve()
    if not joint_csv.exists():
        raise SystemExit(f"joint_wide_csv not found: {joint_csv}")
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    data = load_eval_data(
        joint_wide_csv=joint_csv,
        condition_names=[str(args.condition)],
        eval_mode=str(args.eval_mode),
        n_folds=int(args.n_folds),
        fold_index=int(args.fold_index),
        seed=int(args.seed),
    )
    model = DiffusionTabularModel(input_dim=1, cond_dim=0, seed=int(args.seed))
    model.load(ckpt)

    puma_uids = _parse_csv_list(str(args.puma_uids)) or list(DEFAULT_PUMA_UIDS)
    region_labels = _parse_csv_list(str(args.region_labels))
    if not region_labels:
        region_labels = [DEFAULT_REGION_LABELS.get(pid, f"PUMA {pid}") for pid in puma_uids]
    if len(region_labels) != len(puma_uids):
        raise SystemExit("--region_labels must have the same length as --puma_uids")

    row_indices = _select_fixed_rows(data=data, puma_uids=puma_uids)
    rng = np.random.default_rng(int(args.seed))

    constraint_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    for row_idx, label in zip(row_indices, region_labels):
        p_true = data.p_joint[row_idx]
        p_hat_raw, p_hat_eval = infer_one_region(
            model=model,
            data=data,
            row_idx=int(row_idx),
            condition=str(args.condition),
            n_eval_joint_samples=int(args.n_eval_joint_samples),
            device=None,
            posthoc_ipf=bool(args.posthoc_ipf),
            ipf_iters=int(args.ipf_iters),
        )
        p_used = p_hat_eval if bool(args.posthoc_ipf) else p_hat_raw
        puma_uid = str(data.ids[row_idx])
        statefp = str(data.df.iloc[row_idx]["statefp"]).zfill(2)
        puma = str(int(data.df.iloc[row_idx]["puma"]))

        constraint_rows.append(_build_constraint_row(data=data, row_idx=int(row_idx), region_label=label))
        sample_rows.extend(
            _sample_synthetic_rows(
                p_hat=p_used,
                shape=data.shape,
                n_rows=int(args.rows_per_puma),
                rng=rng,
                puma_uid=puma_uid,
                region_label=label,
                statefp=statefp,
                puma=puma,
            )
        )
        meta_rows.append(
            {
                "puma_uid": puma_uid,
                "region_label": label,
                "tvd_raw": float(_tvd(p_hat_raw, p_true)),
                "tvd_eval": float(_tvd(p_hat_eval, p_true)),
                "total_population": int(round(float(data.totals[row_idx]))),
            }
        )

    constraints_df = pd.DataFrame(constraint_rows)
    samples_df = pd.DataFrame(sample_rows)
    out_constraints_csv.parent.mkdir(parents=True, exist_ok=True)
    out_samples_csv.parent.mkdir(parents=True, exist_ok=True)
    constraints_df.to_csv(out_constraints_csv, index=False)
    samples_df.to_csv(out_samples_csv, index=False)

    write_json(
        out_json,
        {
            "joint_wide_csv": str(joint_csv),
            "checkpoint": str(ckpt),
            "condition": str(args.condition),
            "eval_mode": str(args.eval_mode),
            "n_eval_joint_samples": int(args.n_eval_joint_samples),
            "posthoc_ipf": bool(args.posthoc_ipf),
            "rows_per_puma": int(args.rows_per_puma),
            "pumas": meta_rows,
        },
    )

    print(f"[ok] wrote: {out_constraints_csv}")
    print(f"[ok] wrote: {out_samples_csv}")
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()
