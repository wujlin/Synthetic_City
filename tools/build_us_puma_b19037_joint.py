#!/usr/bin/env python3
from __future__ import annotations

"""
Build US-wide PUMA-level B19037-style joint distributions from PUMS.

Goal:
- Input: US PUMS person + household ZIPs (50 states).
- Output: per-(state,PUMA) joint distribution of:
    householder_age_bin x household_income_bin
  aligned to ACS B19037 binning.

Primary outputs:
- puma_b19037_joint_wide.csv
- puma_b19037_joint_long.csv
- b19037_schema.json
- run.metadata.json
"""

import argparse
import datetime as _dt
import json
import math
import pathlib
import sys
import zipfile
from typing import Any

import numpy as np
import pandas as pd


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.detroit_fetch_public_data import _STATEFP_TO_POSTAL_50
from tools.poc_tabddpm_acs_supervised_b19037 import (
    _parse_age_bounds,
    _parse_b19037_schema,
    _parse_income_edges,
)


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_puma(value: Any) -> str | None:
    if value is None:
        return None
    try:
        s = str(value).strip()
        if not s:
            return None
        return str(int(float(s)))
    except Exception:
        return None


def _find_first_csv_in_zip(path: pathlib.Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise SystemExit(f"No CSV member found in zip: {path}")
        names = sorted(names, key=lambda n: zf.getinfo(n).file_size, reverse=True)
        return names[0]


def _resolve_state_pair(*, pums_dir: pathlib.Path, statefp: str) -> tuple[pathlib.Path, pathlib.Path]:
    if statefp not in _STATEFP_TO_POSTAL_50:
        raise SystemExit(f"Unsupported statefp={statefp}.")
    postal = _STATEFP_TO_POSTAL_50[statefp]
    person_candidates = [
        pums_dir / f"csv_p{postal}.zip",
        pums_dir / f"psam_p{statefp}.zip",
    ]
    hh_candidates = [
        pums_dir / f"csv_h{postal}.zip",
        pums_dir / f"psam_h{statefp}.zip",
    ]
    p = next((x for x in person_candidates if x.exists()), None)
    h = next((x for x in hh_candidates if x.exists()), None)
    if p is None or h is None:
        raise SystemExit(
            f"Missing PUMS files for state={statefp}. "
            f"person tried={person_candidates}, household tried={hh_candidates}"
        )
    return p, h


def _load_householder_age(*, person_zip: pathlib.Path) -> tuple[pd.DataFrame, str]:
    member = _find_first_csv_in_zip(person_zip)
    usecols = ["SERIALNO", "SPORDER", "RELP", "AGEP"]
    with zipfile.ZipFile(person_zip) as zf, zf.open(member) as f:
        d = pd.read_csv(f, usecols=lambda c: c in set(usecols), low_memory=False)

    if "SERIALNO" not in d.columns or "AGEP" not in d.columns:
        raise SystemExit(f"person zip missing required cols SERIALNO/AGEP: {person_zip}")

    method = "SPORDER==1"
    if "RELP" in d.columns:
        relp = pd.to_numeric(d["RELP"], errors="coerce")
        if (relp == 0).any():
            d = d[relp == 0].copy()
            method = "RELP==0"
        elif "SPORDER" in d.columns:
            sp = pd.to_numeric(d["SPORDER"], errors="coerce")
            d = d[sp == 1].copy()
            method = "SPORDER==1(fallback)"
        else:
            raise SystemExit(f"Neither RELP==0 nor SPORDER available in person zip: {person_zip}")
    elif "SPORDER" in d.columns:
        sp = pd.to_numeric(d["SPORDER"], errors="coerce")
        d = d[sp == 1].copy()
    else:
        raise SystemExit(f"person zip missing RELP and SPORDER: {person_zip}")

    d["SERIALNO"] = d["SERIALNO"].astype(str)
    d["AGEP"] = pd.to_numeric(d["AGEP"], errors="coerce")
    d = d.dropna(subset=["SERIALNO", "AGEP"]).copy()
    d = d.drop_duplicates(subset=["SERIALNO"], keep="first")
    return d[["SERIALNO", "AGEP"]], method


def _load_household_income(*, hh_zip: pathlib.Path) -> pd.DataFrame:
    member = _find_first_csv_in_zip(hh_zip)
    usecols = ["SERIALNO", "PUMA", "PUMA20", "WGTP", "HINCP"]
    with zipfile.ZipFile(hh_zip) as zf, zf.open(member) as f:
        d = pd.read_csv(f, usecols=lambda c: c in set(usecols), low_memory=False)
    required = {"SERIALNO", "WGTP", "HINCP"}
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise SystemExit(f"household zip missing required cols {missing}: {hh_zip}")

    puma_col = "PUMA20" if "PUMA20" in d.columns else "PUMA" if "PUMA" in d.columns else None
    if puma_col is None:
        raise SystemExit(f"household zip missing PUMA/PUMA20: {hh_zip}")

    d["SERIALNO"] = d["SERIALNO"].astype(str)
    d["PUMA_STR"] = d[puma_col].map(_normalize_puma)
    d["WGTP"] = pd.to_numeric(d["WGTP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    d["HINCP"] = pd.to_numeric(d["HINCP"], errors="coerce")
    d = d.dropna(subset=["SERIALNO", "PUMA_STR", "HINCP"]).copy()
    return d[["SERIALNO", "PUMA_STR", "WGTP", "HINCP"]]


def _aggregate_state(
    *,
    statefp: str,
    person_zip: pathlib.Path,
    hh_zip: pathlib.Path,
    n_row: int,
    n_col: int,
    age_bounds: list[tuple[float, float]],
    income_edges: list[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    holder, method = _load_householder_age(person_zip=person_zip)
    hh = _load_household_income(hh_zip=hh_zip)
    df = hh.merge(holder, on="SERIALNO", how="inner")
    if df.empty:
        return [], {
            "statefp": statefp,
            "person_zip": str(person_zip),
            "household_zip": str(hh_zip),
            "n_joined_rows": 0,
            "n_pumas": 0,
            "householder_method": method,
        }

    age = pd.to_numeric(df["AGEP"], errors="coerce").to_numpy(dtype=float)
    inc = pd.to_numeric(df["HINCP"], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df["WGTP"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    puma = df["PUMA_STR"].astype(str).to_numpy(dtype=str)

    age_idx = np.full(age.shape, -1, dtype=int)
    for i, (lo, hi) in enumerate(age_bounds):
        m = (age >= float(lo)) & (age < float(hi))
        age_idx[m] = int(i)
    inc_idx = np.searchsorted(np.asarray(income_edges, dtype=float), inc, side="left").astype(int)

    valid = (
        (age_idx >= 0)
        & (age_idx < n_row)
        & (inc_idx >= 0)
        & (inc_idx < n_col)
        & np.isfinite(w)
        & (w > 0)
        & np.isfinite(inc)
    )
    age_idx = age_idx[valid]
    inc_idx = inc_idx[valid]
    w = w[valid]
    puma = puma[valid]

    rows: list[dict[str, Any]] = []
    for pu in sorted(set(puma.tolist())):
        m = puma == pu
        if not bool(m.any()):
            continue
        a = age_idx[m]
        c = inc_idx[m]
        ww = w[m]
        flat = a.astype(int) * int(n_col) + c.astype(int)
        counts = np.zeros((int(n_row) * int(n_col),), dtype=float)
        np.add.at(counts, flat, ww)
        total = float(counts.sum())
        if total <= 0 or not math.isfinite(total):
            continue
        p_joint = counts / total
        p_age = counts.reshape(int(n_row), int(n_col)).sum(axis=1) / total
        p_inc = counts.reshape(int(n_row), int(n_col)).sum(axis=0) / total
        puma5 = str(int(pu)).zfill(5)
        puma_uid = f"{str(statefp).zfill(2)}{puma5}"
        rows.append(
            {
                "statefp": str(statefp).zfill(2),
                "puma": str(int(pu)),
                "puma5": puma5,
                "puma_uid": puma_uid,
                "total_households": total,
                "n_households_unweighted": int(m.sum()),
                "p_joint": p_joint.astype(float),
                "p_age": p_age.astype(float),
                "p_income": p_inc.astype(float),
            }
        )

    info = {
        "statefp": str(statefp).zfill(2),
        "person_zip": str(person_zip),
        "household_zip": str(hh_zip),
        "householder_method": method,
        "n_joined_rows": int(df.shape[0]),
        "n_valid_rows": int(valid.sum()),
        "n_pumas": int(len(rows)),
    }
    return rows, info


def _to_wide_df(rows: list[dict[str, Any]], n_row: int, n_col: int) -> pd.DataFrame:
    K = int(n_row) * int(n_col)
    out_rows: list[dict[str, Any]] = []
    for r in rows:
        row = {
            "statefp": r["statefp"],
            "puma": r["puma"],
            "puma5": r["puma5"],
            "puma_uid": r["puma_uid"],
            "total_households": float(r["total_households"]),
            "n_households_unweighted": int(r["n_households_unweighted"]),
        }
        pj = np.asarray(r["p_joint"], dtype=float).reshape(-1)
        pa = np.asarray(r["p_age"], dtype=float).reshape(-1)
        pi = np.asarray(r["p_income"], dtype=float).reshape(-1)
        for i in range(int(n_row)):
            row[f"p_age_{i:02d}"] = float(pa[i])
        for j in range(int(n_col)):
            row[f"p_income_{j:02d}"] = float(pi[j])
        for k in range(K):
            row[f"p_joint_{k:03d}"] = float(pj[k])
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _to_long_df(rows: list[dict[str, Any]], n_row: int, n_col: int) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    for r in rows:
        counts = np.asarray(r["p_joint"], dtype=float).reshape(int(n_row), int(n_col)) * float(r["total_households"])
        probs = np.asarray(r["p_joint"], dtype=float).reshape(int(n_row), int(n_col))
        for i in range(int(n_row)):
            for j in range(int(n_col)):
                out_rows.append(
                    {
                        "statefp": r["statefp"],
                        "puma": r["puma"],
                        "puma_uid": r["puma_uid"],
                        "age_bin_idx": int(i),
                        "income_bin_idx": int(j),
                        "count_weighted": float(counts[i, j]),
                        "prob_joint": float(probs[i, j]),
                    }
                )
    return pd.DataFrame(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_us_puma_b19037_joint")
    ap.add_argument(
        "--pums_dir",
        default="dataset/wsA_staging/us/raw/pums/pums_2023_5-Year",
        help="Directory containing US PUMS ZIPs (csv_p??.zip + csv_h??.zip).",
    )
    ap.add_argument(
        "--b19037_variables_csv",
        default="dataset/wsA_staging/detroit/raw/census/acs/acs5_2023/acs5_2023_B19037_variables.csv",
        help="ACS B19037 variables CSV for schema parsing.",
    )
    ap.add_argument(
        "--statefps",
        default="all",
        help='Comma-separated state FIPS to process, or "all".',
    )
    ap.add_argument(
        "--out_dir",
        default="dataset/wsA_staging/us/processed/pums/puma_b19037_joint_2023_5-Year",
        help="Output directory.",
    )
    args = ap.parse_args()

    pums_dir = pathlib.Path(args.pums_dir).expanduser().resolve()
    vars_csv = pathlib.Path(args.b19037_variables_csv).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pums_dir.exists():
        raise SystemExit(f"pums_dir not found: {pums_dir}")
    if not vars_csv.exists():
        raise SystemExit(f"b19037_variables_csv not found: {vars_csv}")

    schema = _parse_b19037_schema(variables_csv=vars_csv)
    age_bins = list(schema["age_bins"])
    income_bins = list(schema["income_bins"])
    n_row = int(len(age_bins))
    n_col = int(len(income_bins))
    age_bounds = _parse_age_bounds(age_bins)
    income_edges = _parse_income_edges(income_bins)

    if str(args.statefps).lower() == "all":
        statefps = sorted(_STATEFP_TO_POSTAL_50.keys())
    else:
        raw = [x.strip() for x in str(args.statefps).split(",") if x.strip()]
        statefps = [str(x).zfill(2) for x in raw]
    bad = [s for s in statefps if s not in _STATEFP_TO_POSTAL_50]
    if bad:
        raise SystemExit(f"Unsupported statefps: {bad}")

    all_rows: list[dict[str, Any]] = []
    by_state: list[dict[str, Any]] = []

    for sf in statefps:
        person_zip, hh_zip = _resolve_state_pair(pums_dir=pums_dir, statefp=sf)
        rows, info = _aggregate_state(
            statefp=sf,
            person_zip=person_zip,
            hh_zip=hh_zip,
            n_row=n_row,
            n_col=n_col,
            age_bounds=age_bounds,
            income_edges=income_edges,
        )
        all_rows.extend(rows)
        by_state.append(info)
        print(
            f"[ok] state={sf} pumas={info['n_pumas']} joined={info['n_joined_rows']} valid={info['n_valid_rows']} method={info['householder_method']}",
            file=sys.stderr,
        )

    if not all_rows:
        raise SystemExit("No PUMA distributions produced. Check inputs.")

    wide = _to_wide_df(all_rows, n_row=n_row, n_col=n_col).sort_values(["statefp", "puma5"]).reset_index(drop=True)
    long = _to_long_df(all_rows, n_row=n_row, n_col=n_col).sort_values(["statefp", "puma_uid", "age_bin_idx", "income_bin_idx"]).reset_index(drop=True)

    wide_path = out_dir / "puma_b19037_joint_wide.csv"
    long_path = out_dir / "puma_b19037_joint_long.csv"
    schema_path = out_dir / "b19037_schema.json"
    meta_path = out_dir / "run.metadata.json"

    wide.to_csv(wide_path, index=False)
    long.to_csv(long_path, index=False)
    _write_json(
        schema_path,
        {
            "table_id": "B19037",
            "n_age_bins": int(n_row),
            "n_income_bins": int(n_col),
            "age_bins": age_bins,
            "income_bins": income_bins,
            "income_edges_for_searchsorted_left": income_edges,
        },
    )
    _write_json(
        meta_path,
        {
            "created_utc": _utc_now_iso(),
            "pums_dir": str(pums_dir),
            "variables_csv": str(vars_csv),
            "statefps": statefps,
            "n_states": int(len(statefps)),
            "n_puma_units": int(wide.shape[0]),
            "n_long_rows": int(long.shape[0]),
            "outputs": {
                "wide_csv": str(wide_path),
                "long_csv": str(long_path),
                "schema_json": str(schema_path),
            },
            "by_state": by_state,
            "notes": [
                "puma_uid = statefp(2) + puma5(5) to avoid cross-state PUMA code collisions.",
                "Householder age uses RELP==0 when available, else SPORDER==1 fallback.",
                "Bins follow ACS B19037 variables.csv parsing.",
            ],
        },
    )

    print(f"[ok] wrote: {wide_path}", file=sys.stderr)
    print(f"[ok] wrote: {long_path}", file=sys.stderr)
    print(f"[ok] wrote: {schema_path}", file=sys.stderr)
    print(f"[ok] wrote: {meta_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
