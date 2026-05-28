#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

import numpy as np
import pandas as pd


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _joint_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns.astype(str).tolist() if c.startswith("p_joint_")]
    if not cols:
        raise SystemExit("reference_joint_wide_csv has no p_joint_* columns")
    return sorted(cols, key=lambda x: int(x.rsplit("_", 1)[1]))


def _standard_puma_uid_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"statefp", "puma5"} <= set(out.columns):
        state = out["statefp"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
        puma5 = out["puma5"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
        out["puma_uid"] = state + puma5
    elif {"statefp", "puma"} <= set(out.columns):
        state = out["statefp"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
        puma5 = out["puma"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
        out["puma_uid"] = state + puma5
    elif "puma_uid" in out.columns:
        out["puma_uid"] = out["puma_uid"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)
    else:
        raise SystemExit("metadata requires either puma_uid or statefp+puma/puma5")
    if "puma5" in out.columns:
        out["puma5"] = out["puma5"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    if "statefp" in out.columns:
        out["statefp"] = out["statefp"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    return out


def _id_frame_from_npz(npz: np.lib.npyio.NpzFile) -> pd.DataFrame:
    need = ["puma_uid", "statefp", "puma5"]
    missing = [k for k in need if k not in npz.files]
    if missing:
        raise SystemExit(f"npz missing id arrays: {missing}")
    return _standard_puma_uid_frame(pd.DataFrame(
        {
            "puma_uid": npz["puma_uid"].astype(str),
            "statefp_npz": pd.Series(npz["statefp"].astype(str)).astype(str).str.zfill(2).to_numpy(),
            "puma5_npz": pd.Series(npz["puma5"].astype(str)).astype(str).str.zfill(5).to_numpy(),
        }
    ))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Export a predicted all-PUMA joint vector NPZ into the joint_wide CSV format "
            "expected by Phase 2 tract allocation."
        )
    )
    ap.add_argument("--npz", required=True, type=pathlib.Path)
    ap.add_argument("--key", required=True)
    ap.add_argument("--reference_joint_wide_csv", required=True, type=pathlib.Path)
    ap.add_argument("--schema_json", required=True, type=pathlib.Path)
    ap.add_argument("--out_csv", required=True, type=pathlib.Path)
    ap.add_argument("--out_summary_json", required=True, type=pathlib.Path)
    ap.add_argument("--normalize", action="store_true", help="Renormalize rows to sum to one before export.")
    args = ap.parse_args()

    npz_path = args.npz.expanduser().resolve()
    reference_path = args.reference_joint_wide_csv.expanduser().resolve()
    schema_path = args.schema_json.expanduser().resolve()
    out_csv = args.out_csv.expanduser().resolve()
    out_summary = args.out_summary_json.expanduser().resolve()

    for p in [npz_path, reference_path, schema_path]:
        if not p.exists():
            raise SystemExit(f"input not found: {p}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    expected_k = int(schema.get("K") or np.prod(schema.get("shape", [])))
    if expected_k <= 0:
        raise SystemExit(f"cannot infer K from schema: {schema_path}")

    # The exported model file stores string IDs as numpy object arrays.
    # This NPZ is produced by our own pipeline, so enabling pickle is needed
    # only to recover those IDs; prediction arrays remain numeric.
    with np.load(npz_path, allow_pickle=True) as npz:
        if str(args.key) not in npz.files:
            raise SystemExit(f"npz missing prediction key {args.key!r}; available={npz.files}")
        pred = np.asarray(npz[str(args.key)], dtype=np.float64)
        ids = _id_frame_from_npz(npz)

    if pred.ndim != 2 or pred.shape[1] != expected_k:
        raise SystemExit(f"prediction shape {pred.shape} does not match schema K={expected_k}")
    if pred.shape[0] != ids.shape[0]:
        raise SystemExit(f"prediction rows {pred.shape[0]} do not match id rows {ids.shape[0]}")
    pred = np.clip(pred, 0.0, None)
    row_sums_before = pred.sum(axis=1)
    if bool(args.normalize):
        pred = pred / np.maximum(row_sums_before[:, None], 1e-12)
    row_sums_after = pred.sum(axis=1)

    ref = pd.read_csv(reference_path, low_memory=False)
    jcols = _joint_cols(ref)
    if len(jcols) != expected_k:
        raise SystemExit(f"reference has {len(jcols)} joint cols, expected K={expected_k}")
    if "puma_uid" not in ref.columns:
        raise SystemExit("reference_joint_wide_csv missing puma_uid")

    ref = _standard_puma_uid_frame(ref)
    ids = _standard_puma_uid_frame(ids)
    meta_cols = [c for c in ref.columns if c not in jcols]
    out = ref[meta_cols].merge(ids, on="puma_uid", how="inner", validate="one_to_one")
    if out.shape[0] != pred.shape[0]:
        raise SystemExit(f"merged metadata rows {out.shape[0]} do not match prediction rows {pred.shape[0]}")

    out = out.sort_values(["statefp", "puma5", "puma_uid"], kind="stable").reset_index(drop=True)
    order = ids.reset_index().rename(columns={"index": "_pred_idx"}).merge(out[["puma_uid"]], on="puma_uid", how="right")["_pred_idx"].to_numpy()
    pred_ordered = pred[order]
    joint_df = pd.DataFrame(pred_ordered.astype(np.float32), columns=jcols)
    out = pd.concat([out.drop(columns=["statefp_npz", "puma5_npz"], errors="ignore"), joint_df], axis=1)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    summary = {
        "created_utc": _utc_now(),
        "npz": str(npz_path),
        "key": str(args.key),
        "reference_joint_wide_csv": str(reference_path),
        "schema_json": str(schema_path),
        "out_csv": str(out_csv),
        "n_pumas": int(out.shape[0]),
        "K": int(expected_k),
        "normalize": bool(args.normalize),
        "row_sum_before_min": float(np.min(row_sums_before)),
        "row_sum_before_max": float(np.max(row_sums_before)),
        "row_sum_after_min": float(np.min(row_sums_after)),
        "row_sum_after_max": float(np.max(row_sums_after)),
        "statefp_count": int(out["statefp"].astype(str).str.zfill(2).nunique()) if "statefp" in out.columns else None,
        "total_person_weight": float(pd.to_numeric(out.get("total_person_weight"), errors="coerce").fillna(0.0).sum())
        if "total_person_weight" in out.columns
        else None,
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
