#!/usr/bin/env python3
from __future__ import annotations

"""
Build PUMS-derived Michigan PUMA-level joint targets under the external-condition v1 schema.

Schema v1 (all-population variables):
- SEX: {1, 2}
- AGEP_bin: 10 coarse bins aligned with ACS B01001
- SCHL_allpop: {not_25p, less_than_high_school, high_school_or_ged,
                some_college_or_assoc, bachelor_plus}
- ESR_allpop: {not_16p, employed, unemployed, armed_forces, not_in_labor_force}

Design goal:
- keep the target PUMS-derived
- but redefine it under the same observable schema used by external ACS conditions
- produce a stable, trainable PUMA-level joint target for the first external-condition experiment
"""

import argparse
import datetime as _dt
import json
import pathlib
import sys
import zipfile
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.paths import data_root
from tools.detroit_fetch_public_data import _STATEFP_TO_POSTAL_50


AGE_EDGES = np.asarray([0.0, 5.0, 18.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 1000.0], dtype=float)
AGE_LABELS = [str(pd.Interval(float(AGE_EDGES[i]), float(AGE_EDGES[i + 1]), closed="left")) for i in range(len(AGE_EDGES) - 1)]
SEX_LABELS = ["1", "2"]
SCHL_LABELS = ["not_25p", "less_than_high_school", "high_school_or_ged", "some_college_or_assoc", "bachelor_plus"]
ESR_LABELS = ["not_16p", "employed", "unemployed", "armed_forces", "not_in_labor_force"]
SHAPE = (len(AGE_LABELS), len(SEX_LABELS), len(SCHL_LABELS), len(ESR_LABELS))


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _find_csv_member(path: pathlib.Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise SystemExit(f"No CSV member found in zip: {path}")
        names = sorted(names, key=lambda n: zf.getinfo(n).file_size, reverse=True)
        return names[0]


def _resolve_person_zip(*, pums_dir: pathlib.Path, statefp: str) -> pathlib.Path:
    if statefp not in _STATEFP_TO_POSTAL_50:
        raise SystemExit(f"Unsupported statefp={statefp}")
    postal = _STATEFP_TO_POSTAL_50[statefp]
    candidates = [
        pums_dir / f"csv_p{postal}.zip",
        pums_dir / f"psam_p{statefp}.zip",
    ]
    out = next((p for p in candidates if p.exists()), None)
    if out is not None:
        return out
    recursive = list(pums_dir.rglob(f"csv_p{postal}.zip")) + list(pums_dir.rglob(f"psam_p{statefp}.zip"))
    if recursive:
        return recursive[0]
    raise SystemExit(f"Missing person zip for state={statefp}. tried={candidates} under {pums_dir}")


def _normalize_puma(value: Any) -> str | None:
    if value is None:
        return None
    try:
        s = str(value).strip()
        if not s:
            return None
        v = int(float(s))
        if v <= 0:
            return None
        return str(v)
    except Exception:
        return None


def _bin_age(age: np.ndarray) -> np.ndarray:
    return np.searchsorted(AGE_EDGES, age, side="right").astype(np.int16) - 1


def _bin_sex(sex: np.ndarray) -> np.ndarray:
    out = np.full(sex.shape, -1, dtype=np.int16)
    out[sex == 1] = 0
    out[sex == 2] = 1
    return out


def _bin_schl_allpop(age: np.ndarray, schl: np.ndarray) -> np.ndarray:
    out = np.full(age.shape, 0, dtype=np.int16)  # default not_25p
    mask25 = np.isfinite(age) & (age >= 25)
    out[mask25] = 1  # default less_than_high_school for 25+
    # ACS PUMS SCHL coarse mapping aligned with B15003:
    # <=15: less than high school
    # 16-17: regular high school diploma / GED
    # 18-20: some college / associate
    # >=21: bachelor's and above
    out[mask25 & np.isin(schl, [16, 17])] = 2
    out[mask25 & np.isin(schl, [18, 19, 20])] = 3
    out[mask25 & (schl >= 21)] = 4
    out[~np.isfinite(age)] = -1
    return out


def _bin_esr_allpop(age: np.ndarray, esr: np.ndarray) -> np.ndarray:
    out = np.full(age.shape, 0, dtype=np.int16)  # default not_16p
    mask16 = np.isfinite(age) & (age >= 16)
    out[mask16] = 4  # default not_in_labor_force
    out[mask16 & np.isin(esr, [1, 2])] = 1
    out[mask16 & np.isin(esr, [3])] = 2
    out[mask16 & np.isin(esr, [4, 5])] = 3
    out[~np.isfinite(age)] = -1
    return out


def _tvd(p: Any, q: Any) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    return 0.5 * float(np.abs(p - q).sum())


def _condition_alignment(*, rows: list[dict[str, Any]], condition_csv: pathlib.Path) -> dict[str, Any]:
    cond = pd.read_csv(condition_csv, low_memory=False)
    cond_cols = set(cond.columns.astype(str).tolist())
    needed = {"variable", "category", "target"}
    if not needed <= cond_cols:
        raise SystemExit(f"condition_csv missing required columns {needed}: {condition_csv}")
    if "puma_uid" in cond_cols:
        geo_col = "puma_uid"
    elif "puma" in cond_cols:
        geo_col = "puma"
    else:
        raise SystemExit(f"condition_csv missing geography column 'puma_uid' or 'puma': {condition_csv}")

    def _normalize_geo(value: Any) -> str:
        s = str(value).strip()
        if not s or s.lower() == "nan":
            return ""
        try:
            s = str(int(float(s)))
        except Exception:
            pass
        return s.zfill(7) if geo_col == "puma_uid" else s.lstrip("0") or "0"

    cond[geo_col] = cond[geo_col].map(_normalize_geo)

    target_records: list[dict[str, Any]] = []
    for r in rows:
        tab = np.asarray(r["p_joint"], dtype=float).reshape(SHAPE)
        puma = str(r["puma"])
        puma_uid = str(r.get("puma_uid") or f"{str(r.get('statefp', '')).zfill(2)}{puma}")
        for i, cat in enumerate(SEX_LABELS):
            target_records.append({"puma": puma, "puma_uid": puma_uid, "variable": "SEX", "category": cat, "prob": float(tab[:, i, :, :].sum())})
        for i, cat in enumerate(AGE_LABELS):
            target_records.append({"puma": puma, "puma_uid": puma_uid, "variable": "AGEP_bin", "category": cat, "prob": float(tab[i, :, :, :].sum())})
        for i, cat in enumerate(SCHL_LABELS):
            target_records.append({"puma": puma, "puma_uid": puma_uid, "variable": "SCHL_allpop", "category": cat, "prob": float(tab[:, :, i, :].sum())})
        for i, cat in enumerate(ESR_LABELS):
            target_records.append({"puma": puma, "puma_uid": puma_uid, "variable": "ESR_allpop", "category": cat, "prob": float(tab[:, :, :, i].sum())})

    tgt = pd.DataFrame(target_records)
    tgt[geo_col] = tgt[geo_col].map(_normalize_geo)
    out: dict[str, Any] = {
        "condition_csv": str(condition_csv),
        "geography_key": geo_col,
        "variables": {},
    }
    for var in sorted(set(tgt["variable"].tolist())):
        t_var = tgt[tgt["variable"] == var].copy()
        c_var = cond[cond["variable"] == var].copy()
        groups = sorted(set(t_var[geo_col].astype(str).tolist()) & set(c_var[geo_col].astype(str).tolist()))
        by_group: dict[str, float] = {}
        for g in groups:
            tt = t_var[t_var[geo_col].astype(str) == g].groupby("category", sort=False)["prob"].sum().astype(float)
            cc = c_var[c_var[geo_col].astype(str) == g].groupby("category", sort=False)["target"].sum().astype(float)
            cc = cc / float(cc.sum()) if float(cc.sum()) > 0 else cc
            cats = sorted(set(tt.index.tolist()) | set(cc.index.tolist()))
            p = np.asarray([float(tt.get(k, 0.0)) for k in cats], dtype=float)
            q = np.asarray([float(cc.get(k, 0.0)) for k in cats], dtype=float)
            by_group[g] = _tvd(p, q)
        vals = list(by_group.values())
        out["variables"][var] = {
            "mean_tvd": float(np.mean(vals)) if vals else None,
            "max_tvd": float(np.max(vals)) if vals else None,
            "n_groups": int(len(vals)),
            "by_group": by_group,
        }
    return out


def _aggregate_state(*, statefp: str, person_zip: pathlib.Path, alpha: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    member = _find_csv_member(person_zip)
    usecols = ["PUMA", "PUMA20", "PWGTP", "AGEP", "SEX", "SCHL", "ESR"]
    with zipfile.ZipFile(person_zip) as zf, zf.open(member) as f:
        df = pd.read_csv(f, usecols=lambda c: c in set(usecols), low_memory=False)

    required = ["PWGTP", "AGEP", "SEX"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"person zip missing required columns {missing}: {person_zip}")

    puma_col = "PUMA20" if "PUMA20" in df.columns else "PUMA" if "PUMA" in df.columns else None
    if puma_col is None:
        raise SystemExit(f"person zip missing PUMA/PUMA20: {person_zip}")

    puma = df[puma_col].map(_normalize_puma).astype(object)
    w = pd.to_numeric(df["PWGTP"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    age = pd.to_numeric(df["AGEP"], errors="coerce").to_numpy(dtype=float)
    sex = pd.to_numeric(df["SEX"], errors="coerce").to_numpy(dtype=float)
    schl = pd.to_numeric(df["SCHL"], errors="coerce").to_numpy(dtype=float) if "SCHL" in df.columns else np.full(age.shape, np.nan)
    esr = pd.to_numeric(df["ESR"], errors="coerce").to_numpy(dtype=float) if "ESR" in df.columns else np.full(age.shape, np.nan)

    age_b = _bin_age(age)
    sex_b = _bin_sex(sex)
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
    )

    puma_v = puma.to_numpy(dtype=object)[valid].astype(str)
    w_v = w[valid]
    age_v = age_b[valid]
    sex_v = sex_b[valid]
    schl_v = schl_b[valid]
    esr_v = esr_b[valid]

    rows: list[dict[str, Any]] = []
    for pu in sorted(set(puma_v.tolist())):
        mask = puma_v == pu
        if not bool(mask.any()):
            continue
        idx = np.ravel_multi_index((age_v[mask], sex_v[mask], schl_v[mask], esr_v[mask]), dims=SHAPE)
        counts = np.zeros((int(np.prod(SHAPE)),), dtype=float)
        np.add.at(counts, idx, w_v[mask])
        total = float(counts.sum())
        if total <= 0:
            continue
        sm = counts + float(alpha)
        p_joint = sm / float(sm.sum())
        tab = p_joint.reshape(SHAPE)
        p_age = tab.sum(axis=(1, 2, 3))
        p_sex = tab.sum(axis=(0, 2, 3))
        p_schl = tab.sum(axis=(0, 1, 3))
        p_esr = tab.sum(axis=(0, 1, 2))

        puma5 = str(int(pu)).zfill(5)
        puma_uid = f"{str(statefp).zfill(2)}{puma5}"
        rows.append(
            {
                "statefp": str(statefp).zfill(2),
                "puma": str(int(pu)),
                "puma5": puma5,
                "puma_uid": puma_uid,
                "total_person_weight": total,
                "n_persons_unweighted": int(mask.sum()),
                "p_joint": p_joint.astype(float),
                "p_age": p_age.astype(float),
                "p_sex": p_sex.astype(float),
                "p_schl": p_schl.astype(float),
                "p_esr": p_esr.astype(float),
            }
        )

    info = {
        "statefp": str(statefp).zfill(2),
        "person_zip": str(person_zip),
        "n_rows_raw": int(df.shape[0]),
        "n_rows_valid": int(valid.sum()),
        "n_pumas": int(len(rows)),
    }
    return rows, info


def _to_wide_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    k = int(np.prod(SHAPE))
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
        pj = np.asarray(r["p_joint"], dtype=float).reshape(-1)
        if pj.size != k:
            raise ValueError(f"p_joint size mismatch: {pj.size} vs {k}")
        for i, v in enumerate(pj):
            row[f"p_joint_{i:03d}"] = float(v)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _to_long_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for r in rows:
        probs = np.asarray(r["p_joint"], dtype=float).reshape(SHAPE)
        counts = probs * float(r["total_person_weight"])
        for ai, age_cat in enumerate(AGE_LABELS):
            for si, sex_cat in enumerate(SEX_LABELS):
                for ci, schl_cat in enumerate(SCHL_LABELS):
                    for ei, esr_cat in enumerate(ESR_LABELS):
                        out.append(
                            {
                                "statefp": r["statefp"],
                                "puma": r["puma"],
                                "puma_uid": r["puma_uid"],
                                "age_bin_idx": ai,
                                "age_category": age_cat,
                                "sex_bin_idx": si,
                                "sex_category": sex_cat,
                                "schl_idx": ci,
                                "schl_category": schl_cat,
                                "esr_idx": ei,
                                "esr_category": esr_cat,
                                "prob_joint": float(probs[ai, si, ci, ei]),
                                "count_weighted": float(counts[ai, si, ci, ei]),
                            }
                        )
    return pd.DataFrame(out)


def main() -> None:
    default_data_root = data_root()
    ap = argparse.ArgumentParser(prog="build_external_target_v1_michigan")
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--pums_year", type=int, default=2023)
    ap.add_argument("--pums_period", default="5-Year")
    ap.add_argument(
        "--pums_dir",
        default=str(default_data_root / "detroit" / "raw" / "pums" / "pums_2023_5-Year"),
        help="Directory containing Michigan PUMS person zip (csv_pmi.zip or psam_p26.zip).",
    )
    ap.add_argument("--alpha", type=float, default=0.0, help="Optional Laplace smoothing per joint cell.")
    ap.add_argument("--condition_csv", default=None, help="Optional external-condition CSV for marginal alignment diagnostics.")
    ap.add_argument(
        "--out_dir",
        default=str(default_data_root / "detroit" / "processed" / "external_targets"),
        help="Output directory.",
    )
    args = ap.parse_args()

    if float(args.alpha) < 0:
        raise SystemExit("--alpha must be >= 0")

    pums_dir = pathlib.Path(args.pums_dir).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pums_dir.exists():
        raise SystemExit(f"pums_dir not found: {pums_dir}")

    person_zip = _resolve_person_zip(pums_dir=pums_dir, statefp=str(args.statefp).zfill(2))
    rows, info = _aggregate_state(statefp=str(args.statefp).zfill(2), person_zip=person_zip, alpha=float(args.alpha))
    if not rows:
        raise SystemExit("No PUMA rows were produced.")

    wide = _to_wide_df(rows)
    long = _to_long_df(rows)

    stem = f"exttarget_v1_pums_{int(args.pums_year)}_puma_state{str(args.statefp).zfill(2)}_michigan"
    wide_csv = out_dir / f"{stem}_joint_wide.csv"
    long_csv = out_dir / f"{stem}_joint_long.csv"
    schema_json = out_dir / f"{stem}.schema.json"
    metadata_json = out_dir / f"{stem}.metadata.json"

    wide.to_csv(wide_csv, index=False)
    long.to_csv(long_csv, index=False)
    _write_json(
        schema_json,
        {
            "schema": "external_target_v1",
            "variable_order": ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop"],
            "shape": list(SHAPE),
            "K": int(np.prod(SHAPE)),
            "categories": {
                "AGEP_bin": AGE_LABELS,
                "SEX": SEX_LABELS,
                "SCHL_allpop": SCHL_LABELS,
                "ESR_allpop": ESR_LABELS,
            },
        },
    )

    meta: dict[str, Any] = {
        "schema": "external_target_v1",
        "created_at": _utc_now_iso(),
        "statefp": str(args.statefp).zfill(2),
        "pums_year": int(args.pums_year),
        "pums_period": str(args.pums_period),
        "pums_dir": str(pums_dir),
        "person_zip": str(person_zip),
        "alpha": float(args.alpha),
        "shape": list(SHAPE),
        "K": int(np.prod(SHAPE)),
        "outputs": {
            "joint_wide_csv": str(wide_csv),
            "joint_long_csv": str(long_csv),
            "schema_json": str(schema_json),
        },
        "info": info,
        "design_notes": [
            "Schema follows external_condition_v1 exactly.",
            "The target source can use a different ACS/PUMS release year from the external condition file.",
            "Michigan PUMS 2023 is the preferred default because the available 2022 file has poor PUMA20 coverage.",
        ],
    }

    if args.condition_csv:
        condition_csv = pathlib.Path(args.condition_csv).expanduser().resolve()
        if not condition_csv.exists():
            raise SystemExit(f"condition_csv not found: {condition_csv}")
        meta["condition_alignment"] = _condition_alignment(rows=rows, condition_csv=condition_csv)

    _write_json(metadata_json, meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
