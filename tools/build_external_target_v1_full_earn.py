#!/usr/bin/env python3
from __future__ import annotations

"""
Build a 5-variable external full-joint target by expanding the validated
conditional earnings target:

  p(age, sex, education, employment, earnings | region)
  = p(age, sex, education, employment | region)
    * p(earnings | age, sex, education, employment, region)

This avoids inventing a new income-side approximation at target construction
time. The 5-way joint is derived exactly from the existing conditional target
artifact built from the same PUMS source.
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

from src.synthpop.paths import data_root
from tools.build_external_target_v1_michigan import AGE_LABELS, ESR_LABELS, SCHL_LABELS, SEX_LABELS, _utc_now_iso
from tools.external_earn_v1_schema import EARN_LABELS
from tools.train_us_puma_5var_diffusion import _canon_puma5, _canon_statefp, _canon_uid, _canon_uid_loose


VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop", "EARN_16p_bin"]
CATEGORIES = {
    "AGEP_bin": AGE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LABELS,
    "ESR_allpop": ESR_LABELS,
    "EARN_16p_bin": EARN_LABELS,
}
SHAPE = tuple(len(CATEGORIES[v]) for v in VARIABLE_ORDER)
K = int(np.prod(SHAPE))


def _load_conditional(cond_target_csv: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(cond_target_csv, low_memory=False)
    req = {
        "statefp",
        "puma",
        "puma_uid",
        "cell_idx",
        "age_idx",
        "sex_idx",
        "schl_idx",
        "esr_idx",
        "cell_prob",
        "total_person_weight",
    }
    p_cols = [f"p_earn_{i:02d}" for i in range(len(EARN_LABELS))]
    req |= set(p_cols)
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"conditional target missing columns: {miss}")
    df["statefp"] = df["statefp"].map(_canon_statefp)
    df["puma5"] = df["puma"].map(_canon_puma5)
    if "puma_uid" in df.columns:
        df["puma_uid"] = df["puma_uid"].map(_canon_uid_loose)
    else:
        df["puma_uid"] = df.apply(lambda r: _canon_uid(r["statefp"], r["puma5"]), axis=1)
    df["cell_prob"] = pd.to_numeric(df["cell_prob"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["total_person_weight"] = pd.to_numeric(df["total_person_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    for c in p_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    return df


def _condition_alignment(*, wide: pd.DataFrame, condition_csv: pathlib.Path) -> dict[str, Any]:
    cond = pd.read_csv(condition_csv, low_memory=False)
    if "puma_uid" not in cond.columns:
        raise SystemExit(f"condition_csv missing puma_uid: {condition_csv}")
    cond["puma_uid"] = cond["puma_uid"].map(_canon_uid_loose)
    cond["target"] = pd.to_numeric(cond["target"], errors="coerce").fillna(0.0)

    target_records: list[dict[str, Any]] = []
    prefix_map = {
        "AGEP_bin": "p_age",
        "SEX": "p_sex",
        "SCHL_allpop": "p_schl",
        "ESR_allpop": "p_esr",
        "EARN_16p_bin": "p_earn",
    }
    for r in wide.to_dict(orient="records"):
        uid = str(r["puma_uid"])
        for var in VARIABLE_ORDER:
            labels = CATEGORIES[var]
            prefix = prefix_map[var]
            for i, cat in enumerate(labels):
                target_records.append({"puma_uid": uid, "variable": var, "category": cat, "prob": float(r[f"{prefix}_{i:02d}"])})
    tgt = pd.DataFrame(target_records)

    out: dict[str, Any] = {"condition_csv": str(condition_csv), "geography_key": "puma_uid", "variables": {}}
    for var in VARIABLE_ORDER:
        t_var = tgt[tgt["variable"] == var].copy()
        c_var = cond[cond["variable"] == var].copy()
        groups = sorted(set(t_var["puma_uid"]) & set(c_var["puma_uid"]))
        vals: list[float] = []
        for g in groups:
            tt = t_var[t_var["puma_uid"] == g].groupby("category", sort=False)["prob"].sum()
            cc = c_var[c_var["puma_uid"] == g].groupby("category", sort=False)["target"].sum()
            cc = cc / float(cc.sum()) if float(cc.sum()) > 0 else cc
            cats = sorted(set(tt.index.tolist()) | set(cc.index.tolist()))
            p = np.asarray([float(tt.get(k, 0.0)) for k in cats], dtype=float)
            q = np.asarray([float(cc.get(k, 0.0)) for k in cats], dtype=float)
            vals.append(0.5 * float(np.abs(p - q).sum()))
        out["variables"][var] = {
            "mean_tvd": float(np.mean(vals)) if vals else None,
            "max_tvd": float(np.max(vals)) if vals else None,
            "n_groups": int(len(vals)),
        }
    return out


def main() -> None:
    default_root = data_root()
    ap = argparse.ArgumentParser(prog="build_external_target_v1_full_earn")
    ap.add_argument(
        "--conditional_target_csv",
        default=str(default_root / "us" / "processed" / "external_targets" / "exttarget_earn_cond_v1_pums_2023_puma_us.csv"),
    )
    ap.add_argument("--condition_csv", default=None)
    ap.add_argument(
        "--out_dir",
        default=str(default_root / "us" / "processed" / "external_targets"),
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cond_target_csv = pathlib.Path(args.conditional_target_csv).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cond_target_csv.exists():
        raise SystemExit(f"conditional_target_csv not found: {cond_target_csv}")

    stem = "exttarget_v1_full_earn_pums_2023_puma_us"
    wide_csv = out_dir / f"{stem}_joint_wide.csv"
    long_csv = out_dir / f"{stem}_joint_long.csv"
    schema_json = out_dir / f"{stem}.schema.json"
    metadata_json = out_dir / f"{stem}.metadata.json"
    if any(p.exists() for p in [wide_csv, long_csv, schema_json, metadata_json]) and not args.overwrite:
        raise SystemExit(f"output exists under {out_dir} (use --overwrite)")

    df = _load_conditional(cond_target_csv)
    p_cols = [f"p_earn_{i:02d}" for i in range(len(EARN_LABELS))]

    wide_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for uid, g in df.groupby("puma_uid", sort=True):
        statefp = _canon_statefp(str(g["statefp"].iloc[0]))
        puma5 = _canon_puma5(str(g["puma5"].iloc[0]))
        total_person_weight = float(pd.to_numeric(g["total_person_weight"], errors="coerce").fillna(0.0).max())

        tab = np.zeros(SHAPE, dtype=np.float64)
        for _, r in g.iterrows():
            ai = int(r["age_idx"])
            si = int(r["sex_idx"])
            qi = int(r["schl_idx"])
            ei = int(r["esr_idx"])
            cell_prob = float(r["cell_prob"])
            p_earn = np.asarray([float(r[c]) for c in p_cols], dtype=np.float64)
            p_earn = p_earn / max(float(p_earn.sum()), 1e-12)
            tab[ai, si, qi, ei, :] += cell_prob * p_earn

        tab = tab / max(float(tab.sum()), 1e-12)
        row = {
            "statefp": statefp,
            "puma": str(int(puma5)) if puma5 else "",
            "puma5": puma5,
            "puma_uid": str(uid),
            "total_person_weight": total_person_weight,
        }
        marginals = {
            "AGEP_bin": tab.sum(axis=(1, 2, 3, 4)),
            "SEX": tab.sum(axis=(0, 2, 3, 4)),
            "SCHL_allpop": tab.sum(axis=(0, 1, 3, 4)),
            "ESR_allpop": tab.sum(axis=(0, 1, 2, 4)),
            "EARN_16p_bin": tab.sum(axis=(0, 1, 2, 3)),
        }
        prefix_map = {
            "AGEP_bin": "p_age",
            "SEX": "p_sex",
            "SCHL_allpop": "p_schl",
            "ESR_allpop": "p_esr",
            "EARN_16p_bin": "p_earn",
        }
        for var in VARIABLE_ORDER:
            prefix = prefix_map[var]
            for i, v in enumerate(marginals[var].tolist()):
                row[f"{prefix}_{i:02d}"] = float(v)
        flat = tab.reshape(-1)
        for i, v in enumerate(flat.tolist()):
            row[f"p_joint_{i:03d}"] = float(v)
        wide_rows.append(row)

        for ai, age_cat in enumerate(CATEGORIES["AGEP_bin"]):
            for si, sex_cat in enumerate(CATEGORIES["SEX"]):
                for qi, schl_cat in enumerate(CATEGORIES["SCHL_allpop"]):
                    for ei, esr_cat in enumerate(CATEGORIES["ESR_allpop"]):
                        for wi, earn_cat in enumerate(CATEGORIES["EARN_16p_bin"]):
                            long_rows.append(
                                {
                                    "statefp": statefp,
                                    "puma": str(int(puma5)) if puma5 else "",
                                    "puma_uid": str(uid),
                                    "AGEP_bin": age_cat,
                                    "SEX": sex_cat,
                                    "SCHL_allpop": schl_cat,
                                    "ESR_allpop": esr_cat,
                                    "EARN_16p_bin": earn_cat,
                                    "prob": float(tab[ai, si, qi, ei, wi]),
                                }
                            )

    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)
    wide.to_csv(wide_csv, index=False)
    long.to_csv(long_csv, index=False)

    schema = {
        "schema": "external_target_v1_full_earn",
        "variant": "full_earn",
        "variable_order": VARIABLE_ORDER,
        "shape": list(SHAPE),
        "K": K,
        "categories": CATEGORIES,
    }
    schema_json.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta: dict[str, Any] = {
        "schema": "external_target_v1_full_earn",
        "variant": "full_earn",
        "created_at": _utc_now_iso(),
        "source_conditional_target_csv": str(cond_target_csv),
        "outputs": {
            "joint_wide_csv": str(wide_csv),
            "joint_long_csv": str(long_csv),
            "schema_json": str(schema_json),
        },
        "shape": list(SHAPE),
        "K": K,
        "variables": CATEGORIES,
        "note": "Derived exactly from p(4-attr cell|region) and p(earn|4-attr cell, region) built from the same PUMS source.",
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
