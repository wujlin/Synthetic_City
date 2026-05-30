#!/usr/bin/env python3
from __future__ import annotations

"""
Build a 5-variable external full-joint target with person income (v2 10-bin schema):

  AGEP_bin x SEX x SCHL_allpop x ESR_allpop x PINCP_allpop_bin
"""

import argparse
import json
import pathlib
import sys
import zipfile
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.paths import data_root
from tools.data.build_external_condition_v1_acs_puma import _parse_states, _scope_tag
from tools.data.build_external_target_v1_michigan import (
    AGE_LABELS,
    ESR_LABELS,
    SCHL_LABELS,
    SEX_LABELS,
    _bin_age,
    _bin_esr_allpop,
    _bin_schl_allpop,
    _normalize_puma,
    _resolve_person_zip,
    _utc_now_iso,
)
from src.synthpop.data.state_codes import STATEFP_TO_POSTAL as _STATEFP_TO_POSTAL_50
from tools.data.external_income_v1_schema import INCOME_LABELS, bin_income_allpop
from tools.model.train_us_puma_5var_diffusion import _canon_puma5, _canon_statefp, _canon_uid, _canon_uid_loose


VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop", "PINCP_allpop_bin"]
CATEGORIES = {
    "AGEP_bin": AGE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LABELS,
    "ESR_allpop": ESR_LABELS,
    "PINCP_allpop_bin": INCOME_LABELS,
}
SHAPE = tuple(len(CATEGORIES[v]) for v in VARIABLE_ORDER)
K = int(np.prod(SHAPE))


def _find_csv_member(path: pathlib.Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise SystemExit(f"No CSV member found in zip: {path}")
        names = sorted(names, key=lambda n: zf.getinfo(n).file_size, reverse=True)
        return names[0]


def _load_person_df(person_path: pathlib.Path) -> pd.DataFrame:
    usecols = ["PUMA", "PUMA20", "PWGTP", "AGEP", "SEX", "PINCP", "SCHL", "ESR"]
    if person_path.suffix.lower() == ".zip":
        member = _find_csv_member(person_path)
        with zipfile.ZipFile(person_path) as zf, zf.open(member) as f:
            return pd.read_csv(f, usecols=lambda c: c in set(usecols), low_memory=False)
    return pd.read_csv(person_path, usecols=lambda c: c in set(usecols), low_memory=False)


def _bin_sex(sex: np.ndarray) -> np.ndarray:
    out = np.full(sex.shape, -1, dtype=np.int16)
    out[sex == 1] = 0
    out[sex == 2] = 1
    return out


def _aggregate_state(*, statefp: str, person_path: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = _load_person_df(person_path)
    required = ["PWGTP", "AGEP", "SEX", "PINCP"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"person file missing required columns {missing}: {person_path}")

    puma_col = "PUMA20" if "PUMA20" in df.columns else "PUMA" if "PUMA" in df.columns else None
    if puma_col is None:
        raise SystemExit(f"person file missing PUMA/PUMA20: {person_path}")

    puma = df[puma_col].map(_normalize_puma).astype(object)
    w = pd.to_numeric(df["PWGTP"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    age = pd.to_numeric(df["AGEP"], errors="coerce").to_numpy(dtype=float)
    sex = pd.to_numeric(df["SEX"], errors="coerce").to_numpy(dtype=float)
    income = pd.to_numeric(df["PINCP"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    schl = pd.to_numeric(df["SCHL"], errors="coerce").to_numpy(dtype=float) if "SCHL" in df.columns else np.full(age.shape, np.nan)
    esr = pd.to_numeric(df["ESR"], errors="coerce").to_numpy(dtype=float) if "ESR" in df.columns else np.full(age.shape, np.nan)

    age_b = _bin_age(age)
    sex_b = _bin_sex(sex)
    income_b = bin_income_allpop(age, income)
    schl_b = _bin_schl_allpop(age, schl)
    esr_b = _bin_esr_allpop(age, esr)

    valid = (
        puma.notna().to_numpy(dtype=bool)
        & np.isfinite(w)
        & (w > 0)
        & np.isfinite(age)
        & (age_b >= 0)
        & (age_b < SHAPE[0])
        & (sex_b >= 0)
        & (sex_b < SHAPE[1])
        & (schl_b >= 0)
        & (schl_b < SHAPE[2])
        & (esr_b >= 0)
        & (esr_b < SHAPE[3])
        & (income_b >= 0)
        & (income_b < SHAPE[4])
    )

    puma_v = puma.to_numpy(dtype=object)[valid].astype(str)
    w_v = w[valid]
    age_v = age_b[valid]
    sex_v = sex_b[valid]
    schl_v = schl_b[valid]
    esr_v = esr_b[valid]
    income_v = income_b[valid]

    rows: list[dict[str, Any]] = []
    for pu in sorted(set(puma_v.tolist())):
        mask = puma_v == pu
        idx = np.ravel_multi_index((age_v[mask], sex_v[mask], schl_v[mask], esr_v[mask], income_v[mask]), dims=SHAPE)
        counts = np.zeros((K,), dtype=float)
        np.add.at(counts, idx, w_v[mask])
        total = float(counts.sum())
        if total <= 0:
            continue
        probs = counts / total
        tab = probs.reshape(SHAPE)
        puma5 = str(int(pu)).zfill(5)
        puma_uid = _canon_uid(statefp, puma5)
        rows.append(
            {
                "statefp": _canon_statefp(statefp),
                "puma": str(int(pu)),
                "puma5": puma5,
                "puma_uid": puma_uid,
                "total_person_weight": total,
                "n_persons_unweighted": int(mask.sum()),
                "p_joint": probs.astype(float),
                "p_age": tab.sum(axis=(1, 2, 3, 4)).astype(float),
                "p_sex": tab.sum(axis=(0, 2, 3, 4)).astype(float),
                "p_schl": tab.sum(axis=(0, 1, 3, 4)).astype(float),
                "p_esr": tab.sum(axis=(0, 1, 2, 4)).astype(float),
                "p_income": tab.sum(axis=(0, 1, 2, 3)).astype(float),
            }
        )

    info = {
        "statefp": _canon_statefp(statefp),
        "person_path": str(person_path),
        "n_rows_raw": int(df.shape[0]),
        "n_rows_valid": int(valid.sum()),
        "n_pumas": int(len(rows)),
    }
    return rows, info


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
        "PINCP_allpop_bin": "p_income",
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
    ap = argparse.ArgumentParser(prog="build_external_target_v1_full_income")
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--statefps", default="")
    ap.add_argument("--all_states", action="store_true")
    ap.add_argument("--pums_year", type=int, default=2023)
    ap.add_argument("--pums_period", default="5-Year")
    ap.add_argument(
        "--pums_dir",
        default=str(default_root / "us" / "raw" / "pums" / "pums_2023_5-Year"),
        help="Directory containing state-level PUMS person zips.",
    )
    ap.add_argument("--condition_csv", default=None)
    ap.add_argument("--out_dir", default=str(default_root / "us" / "processed" / "external_targets"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    states = _parse_states(statefp=args.statefp, statefps=args.statefps, all_states=bool(args.all_states))
    pums_dir = pathlib.Path(args.pums_dir).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pums_dir.exists():
        raise SystemExit(f"pums_dir not found: {pums_dir}")

    rows: list[dict[str, Any]] = []
    state_infos: list[dict[str, Any]] = []
    for statefp in states:
        if statefp not in _STATEFP_TO_POSTAL_50:
            raise SystemExit(f"Unsupported statefp={statefp}")
        person_zip = _resolve_person_zip(pums_dir=pums_dir, statefp=statefp)
        st_rows, st_info = _aggregate_state(statefp=statefp, person_path=person_zip)
        rows.extend(st_rows)
        state_infos.append(st_info)
        print(f"[ok] state={statefp} n_pumas={st_info['n_pumas']} n_rows_valid={st_info['n_rows_valid']}", file=sys.stderr)

    if not rows:
        raise SystemExit("No PUMA rows were produced.")

    scope_tag = _scope_tag(states)
    stem = f"exttarget_v1_full_income_v2_pums_{int(args.pums_year)}_puma_{scope_tag}"
    wide_csv = out_dir / f"{stem}_joint_wide.csv"
    long_csv = out_dir / f"{stem}_joint_long.csv"
    schema_json = out_dir / f"{stem}.schema.json"
    metadata_json = out_dir / f"{stem}.metadata.json"
    if any(p.exists() for p in [wide_csv, long_csv, schema_json, metadata_json]) and not args.overwrite:
        raise SystemExit(f"output exists under {out_dir} (use --overwrite)")

    wide_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    for r in rows:
        row = {
            "statefp": r["statefp"],
            "puma": r["puma"],
            "puma5": r["puma5"],
            "puma_uid": r["puma_uid"],
            "total_person_weight": float(r["total_person_weight"]),
            "n_persons_unweighted": int(r["n_persons_unweighted"]),
        }
        for i, v in enumerate(np.asarray(r["p_age"], dtype=float).reshape(-1)):
            row[f"p_age_{i:02d}"] = float(v)
        for i, v in enumerate(np.asarray(r["p_sex"], dtype=float).reshape(-1)):
            row[f"p_sex_{i:02d}"] = float(v)
        for i, v in enumerate(np.asarray(r["p_schl"], dtype=float).reshape(-1)):
            row[f"p_schl_{i:02d}"] = float(v)
        for i, v in enumerate(np.asarray(r["p_esr"], dtype=float).reshape(-1)):
            row[f"p_esr_{i:02d}"] = float(v)
        for i, v in enumerate(np.asarray(r["p_income"], dtype=float).reshape(-1)):
            row[f"p_income_{i:02d}"] = float(v)
        for i, v in enumerate(np.asarray(r["p_joint"], dtype=float).reshape(-1)):
            row[f"p_joint_{i:03d}"] = float(v)
        wide_rows.append(row)

        tab = np.asarray(r["p_joint"], dtype=float).reshape(SHAPE)
        for ai, age_cat in enumerate(CATEGORIES["AGEP_bin"]):
            for si, sex_cat in enumerate(CATEGORIES["SEX"]):
                for qi, schl_cat in enumerate(CATEGORIES["SCHL_allpop"]):
                    for ei, esr_cat in enumerate(CATEGORIES["ESR_allpop"]):
                        for wi, income_cat in enumerate(CATEGORIES["PINCP_allpop_bin"]):
                            long_rows.append(
                                {
                                    "statefp": r["statefp"],
                                    "puma": r["puma"],
                                    "puma_uid": r["puma_uid"],
                                    "AGEP_bin": age_cat,
                                    "SEX": sex_cat,
                                    "SCHL_allpop": schl_cat,
                                    "ESR_allpop": esr_cat,
                                    "PINCP_allpop_bin": income_cat,
                                    "prob": float(tab[ai, si, qi, ei, wi]),
                                }
                            )

    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)
    wide.to_csv(wide_csv, index=False)
    long.to_csv(long_csv, index=False)

    schema = {
        "schema": "external_target_v1_full_income_v2",
        "variant": "full_income_v2",
        "variable_order": VARIABLE_ORDER,
        "shape": list(SHAPE),
        "K": K,
        "categories": CATEGORIES,
    }
    schema_json.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta: dict[str, Any] = {
        "schema": "external_target_v1_full_income_v2",
        "variant": "full_income_v2",
        "created_at": _utc_now_iso(),
        "scope": scope_tag,
        "statefps": states,
        "n_states": int(len(states)),
        "pums_year": int(args.pums_year),
        "pums_period": str(args.pums_period),
        "pums_dir": str(pums_dir),
        "outputs": {
            "joint_wide_csv": str(wide_csv),
            "joint_long_csv": str(long_csv),
            "schema_json": str(schema_json),
        },
        "shape": list(SHAPE),
        "K": K,
        "variables": CATEGORIES,
        "state_summaries": state_infos,
        "note": "Built directly from PUMS PINCP under the v2 all-population income schema aligned to B06010 with separate under-15 and 15+ no-income buckets.",
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
