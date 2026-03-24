#!/usr/bin/env python3
from __future__ import annotations

"""
Aggregate external-target v1 PUMS-derived joints into a lower-dimensional external-target v1-lite schema.
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

from tools.build_external_target_v1_michigan import _utc_now_iso
from tools.train_us_puma_5var_diffusion import _canon_puma5, _canon_statefp, _canon_uid, _canon_uid_loose


OLD_SHAPE = (10, 2, 5, 5)
NEW_AGE_LABELS = ["[0.0, 18.0)", "[18.0, 35.0)", "[35.0, 65.0)", "[65.0, 1000.0)"]
NEW_SEX_LABELS = ["1", "2"]
NEW_SCHL_LABELS = ["not_25p", "non_bachelor", "bachelor_plus"]
NEW_ESR_LABELS = ["not_16p", "employed", "not_employed"]
NEW_SHAPE = (len(NEW_AGE_LABELS), len(NEW_SEX_LABELS), len(NEW_SCHL_LABELS), len(NEW_ESR_LABELS))

AGE_INDEX_MAP = np.asarray([0, 0, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int16)
SCHL_INDEX_MAP = np.asarray([0, 1, 1, 1, 2], dtype=np.int16)
ESR_INDEX_MAP = np.asarray([0, 1, 2, 2, 2], dtype=np.int16)


def _aggregate_joint(p_old: np.ndarray) -> np.ndarray:
    tab_old = np.asarray(p_old, dtype=np.float64).reshape(OLD_SHAPE)
    tab_new = np.zeros(NEW_SHAPE, dtype=np.float64)
    for ai in range(OLD_SHAPE[0]):
        an = int(AGE_INDEX_MAP[ai])
        for si in range(OLD_SHAPE[1]):
            for qi in range(OLD_SHAPE[2]):
                qn = int(SCHL_INDEX_MAP[qi])
                for ei in range(OLD_SHAPE[3]):
                    en = int(ESR_INDEX_MAP[ei])
                    tab_new[an, si, qn, en] += tab_old[ai, si, qi, ei]
    tab_new = tab_new / max(float(tab_new.sum()), 1e-12)
    return tab_new.reshape(-1)


def _condition_alignment(*, wide: pd.DataFrame, condition_csv: pathlib.Path) -> dict[str, Any]:
    cond = pd.read_csv(condition_csv, low_memory=False)
    if "puma_uid" not in cond.columns:
        raise SystemExit(f"condition_csv missing puma_uid: {condition_csv}")
    cond["puma_uid"] = cond["puma_uid"].map(_canon_uid_loose)
    cond["target"] = pd.to_numeric(cond["target"], errors="coerce").fillna(0.0)

    target_records: list[dict[str, Any]] = []
    for r in wide.to_dict(orient="records"):
        uid = _canon_uid(r["statefp"], r.get("puma5", r["puma"]))
        for i, cat in enumerate(NEW_AGE_LABELS):
            target_records.append({"puma_uid": uid, "variable": "AGEP_bin", "category": cat, "prob": float(r[f"p_age_{i:02d}"])})
        for i, cat in enumerate(NEW_SEX_LABELS):
            target_records.append({"puma_uid": uid, "variable": "SEX", "category": cat, "prob": float(r[f"p_sex_{i:02d}"])})
        for i, cat in enumerate(NEW_SCHL_LABELS):
            target_records.append({"puma_uid": uid, "variable": "SCHL_allpop", "category": cat, "prob": float(r[f"p_schl_{i:02d}"])})
        for i, cat in enumerate(NEW_ESR_LABELS):
            target_records.append({"puma_uid": uid, "variable": "ESR_allpop", "category": cat, "prob": float(r[f"p_esr_{i:02d}"])})
    tgt = pd.DataFrame(target_records)

    out: dict[str, Any] = {"condition_csv": str(condition_csv), "geography_key": "puma_uid", "variables": {}}
    for var in ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop"]:
        t_var = tgt[tgt["variable"] == var].copy()
        c_var = cond[cond["variable"] == var].copy()
        groups = sorted(set(t_var["puma_uid"]) & set(c_var["puma_uid"]))
        vals: list[float] = []
        by_group: dict[str, float] = {}
        for g in groups:
            tt = t_var[t_var["puma_uid"] == g].groupby("category", sort=False)["prob"].sum()
            cc = c_var[c_var["puma_uid"] == g].groupby("category", sort=False)["target"].sum()
            cc = cc / float(cc.sum()) if float(cc.sum()) > 0 else cc
            cats = sorted(set(tt.index.tolist()) | set(cc.index.tolist()))
            p = np.asarray([float(tt.get(k, 0.0)) for k in cats], dtype=float)
            q = np.asarray([float(cc.get(k, 0.0)) for k in cats], dtype=float)
            tvd = 0.5 * float(np.abs(p - q).sum())
            vals.append(tvd)
            by_group[g] = tvd
        out["variables"][var] = {
            "mean_tvd": float(np.mean(vals)) if vals else None,
            "max_tvd": float(np.max(vals)) if vals else None,
            "n_groups": int(len(vals)),
            "by_group": by_group,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_external_target_v1_lite")
    ap.add_argument("--joint_wide_csv", required=True)
    ap.add_argument("--condition_csv", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_path = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"joint_wide_csv not found: {in_path}")

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = in_path.name.replace("exttarget_v1_", "exttarget_v1_lite_").replace("_joint_wide.csv", "")
    wide_csv = out_dir / f"{stem}_joint_wide.csv"
    long_csv = out_dir / f"{stem}_joint_long.csv"
    schema_json = out_dir / f"{stem}.schema.json"
    metadata_json = out_dir / f"{stem}.metadata.json"
    if (wide_csv.exists() or long_csv.exists() or schema_json.exists() or metadata_json.exists()) and not args.overwrite:
        raise SystemExit(f"output exists under {out_dir} (use --overwrite)")

    df = pd.read_csv(in_path, low_memory=False)
    req = {"statefp", "puma", "puma_uid", "total_person_weight", "n_persons_unweighted"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"joint_wide_csv missing columns: {miss}")

    p_joint_cols = [f"p_joint_{i:03d}" for i in range(int(np.prod(OLD_SHAPE)))]
    missing_joint = [c for c in p_joint_cols if c not in df.columns]
    if missing_joint:
        raise SystemExit(f"joint_wide_csv missing joint columns: {missing_joint[:5]}")

    out_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for r in df.to_dict(orient="records"):
        p_old = np.asarray([float(r[c]) for c in p_joint_cols], dtype=np.float64)
        p_new = _aggregate_joint(p_old)
        tab = p_new.reshape(NEW_SHAPE)
        statefp = _canon_statefp(r["statefp"])
        puma5 = _canon_puma5(r.get("puma5", r["puma"]))
        puma_uid = _canon_uid(statefp, puma5)
        row = {
            "statefp": statefp,
            "puma": str(int(puma5)) if puma5 else "",
            "puma5": puma5,
            "puma_uid": puma_uid,
            "total_person_weight": float(r["total_person_weight"]),
            "n_persons_unweighted": int(r["n_persons_unweighted"]),
        }
        p_age = tab.sum(axis=(1, 2, 3))
        p_sex = tab.sum(axis=(0, 2, 3))
        p_schl = tab.sum(axis=(0, 1, 3))
        p_esr = tab.sum(axis=(0, 1, 2))
        for i, v in enumerate(p_age):
            row[f"p_age_{i:02d}"] = float(v)
        for i, v in enumerate(p_sex):
            row[f"p_sex_{i:02d}"] = float(v)
        for i, v in enumerate(p_schl):
            row[f"p_schl_{i:02d}"] = float(v)
        for i, v in enumerate(p_esr):
            row[f"p_esr_{i:02d}"] = float(v)
        for i, v in enumerate(p_new):
            row[f"p_joint_{i:03d}"] = float(v)
        out_rows.append(row)

        for ai, age_cat in enumerate(NEW_AGE_LABELS):
            for si, sex_cat in enumerate(NEW_SEX_LABELS):
                for qi, schl_cat in enumerate(NEW_SCHL_LABELS):
                    for ei, esr_cat in enumerate(NEW_ESR_LABELS):
                        long_rows.append(
                            {
                                "statefp": statefp,
                                "puma": str(int(puma5)) if puma5 else "",
                                "puma_uid": puma_uid,
                                "AGEP_bin": age_cat,
                                "SEX": sex_cat,
                                "SCHL_allpop": schl_cat,
                                "ESR_allpop": esr_cat,
                                "prob": float(tab[ai, si, qi, ei]),
                            }
                        )

    wide = pd.DataFrame(out_rows)
    long = pd.DataFrame(long_rows)
    wide.to_csv(wide_csv, index=False)
    long.to_csv(long_csv, index=False)

    schema = {
        "schema": "external_target_v1_lite",
        "variable_order": ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop"],
        "shape": list(NEW_SHAPE),
        "K": int(np.prod(NEW_SHAPE)),
        "categories": {
            "AGEP_bin": NEW_AGE_LABELS,
            "SEX": NEW_SEX_LABELS,
            "SCHL_allpop": NEW_SCHL_LABELS,
            "ESR_allpop": NEW_ESR_LABELS,
        },
    }
    schema_json.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta: dict[str, Any] = {
        "schema": "external_target_v1_lite",
        "created_at": _utc_now_iso(),
        "source_joint_wide_csv": str(in_path),
        "outputs": {
            "joint_wide_csv": str(wide_csv),
            "joint_long_csv": str(long_csv),
            "schema_json": str(schema_json),
        },
        "shape": list(NEW_SHAPE),
        "K": int(np.prod(NEW_SHAPE)),
        "variables": schema["categories"],
    }
    if args.condition_csv:
        cond_path = pathlib.Path(args.condition_csv).expanduser().resolve()
        if not cond_path.exists():
            raise SystemExit(f"condition_csv not found: {cond_path}")
        meta["condition_alignment"] = _condition_alignment(wide=wide, condition_csv=cond_path)
    metadata_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {wide_csv}")


if __name__ == "__main__":
    main()
