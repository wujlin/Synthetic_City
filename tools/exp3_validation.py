#!/usr/bin/env python3
"""
Exp 3: Multi-level validation for Exp1+Exp2 synthetic population.

This script is intentionally "thin": it wires together existing validation functions and
writes small JSON artifacts suitable for PI review.

Supported validation layers (v0):
- L2 Tract marginals vs ACS targets_long (compute_stats_metrics_against_targets_long)
- L3 PUMA marginals/associations vs PUMS microdata (compute_stats_metrics)

Optional (future / not in v0):
- L1 BG marginals vs DHC 2020 (needs DHC tables + BG synthetic)
- L4 LODES work-location validation
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import sys
import zipfile
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Missing dependency: {pkg}. Install it in your conda env.") from e


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _find_first_csv_in_zip(zip_path: pathlib.Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No .csv found inside: {zip_path}")
        return names[0]


def _age_idx_to_midpoint(age_idx_series: Any) -> Any:
    pd = _require("pandas")
    # 23-bin midpoints used across Exp2/Exp4 for AGEP approximation.
    mids = [2.0, 7.0, 12.0, 16.0, 19.0, 20.0, 21.0, 23.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 61.0, 64.0, 66.0, 69.0, 74.0, 79.0, 84.0, 90.0]
    age_idx = pd.to_numeric(age_idx_series, errors="coerce").fillna(0).astype(int).clip(lower=0, upper=22)
    return age_idx.map(lambda x: float(mids[int(x)]))


def _harmonize_synthetic_columns(syn: Any, *, puma_group_col: str) -> Any:
    pd = _require("pandas")

    out = syn.copy()
    cols = {str(c).lower(): str(c) for c in out.columns}

    def _has(c: str) -> bool:
        return c in out.columns

    def _pick(*cands: str) -> str | None:
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    # AGEP
    if not _has("AGEP"):
        age_idx_col = _pick("age_idx", "agebin", "age_bin", "age_group_idx")
        if age_idx_col is not None:
            out["AGEP"] = _age_idx_to_midpoint(out[age_idx_col])

    # SEX
    if not _has("SEX"):
        sex_col = _pick("sex")
        if sex_col is not None:
            out["SEX"] = pd.to_numeric(out[sex_col], errors="coerce").fillna(1).astype(int).clip(lower=1, upper=2)

    # PINCP
    if not _has("PINCP"):
        inc_col = _pick("income", "pincp")
        if inc_col is not None:
            out["PINCP"] = pd.to_numeric(out[inc_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    # ESR
    if not _has("ESR"):
        esr_col = _pick("esr")
        if esr_col is not None:
            out["ESR"] = out[esr_col].astype(str)

    # puma grouping
    if not _has("puma"):
        if str(puma_group_col) in out.columns:
            out["puma"] = out[str(puma_group_col)].astype(str)
        else:
            puma_col = _pick("puma", "puma20")
            if puma_col is not None:
                out["puma"] = out[puma_col].astype(str)

    return out


def _resolve_pums_person_zip(*, data_root: pathlib.Path, pums_year: int, pums_period: str, statefp: str) -> pathlib.Path:
    statefp = str(statefp).zfill(2)
    state_postal_lower = "mi" if statefp == "26" else None
    raw_dir = data_root / "detroit" / "raw" / "pums" / f"pums_{pums_year}_{pums_period}"
    candidates: list[pathlib.Path] = [raw_dir / f"psam_p{statefp}.zip"]
    if state_postal_lower is not None:
        candidates.append(raw_dir / f"csv_p{state_postal_lower}i.zip")  # csv_pmi.zip
        candidates.append(raw_dir / f"csv_p{state_postal_lower}.zip")
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit(f"PUMS person zip not found. Tried: {candidates}")


def main() -> None:
    pd = _require("pandas")

    from src.synthpop.pipeline.detroit_v0 import make_run_id
    from src.synthpop.paths import data_root as default_data_root
    from src.synthpop.validation.stats import compute_stats_metrics, compute_stats_metrics_against_targets_long

    ap = argparse.ArgumentParser(prog="exp3_validation")
    ap.add_argument("--synthetic_path", required=True, help="Synthetic microdata (csv/parquet).")
    ap.add_argument("--data_root", default=str(default_data_root()))
    ap.add_argument("--pums_year", type=int, default=2023)
    ap.add_argument("--pums_period", default="5-Year")
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--pums_person_zip", default=None, help="Override PUMS person zip path.")
    ap.add_argument("--acs_targets_long", default=None, help="Path to ACS targets_long (csv/parquet).")
    ap.add_argument("--tract_group_col", default="tract_geoid")
    ap.add_argument("--puma_group_col", default="puma")
    ap.add_argument("--out_dir", default=None, help="Default: outputs/<run_id> under repo.")
    args = ap.parse_args()

    out_dir = (
        pathlib.Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (_REPO_ROOT / "outputs" / make_run_id(prefix="exp3_validation"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        out_dir / "run.metadata.json",
        {
            "created_utc": _utc_now_iso(),
            "argv": sys.argv,
            "script": pathlib.Path(__file__).name,
            "env": {"RAW_ROOT": os.environ.get("RAW_ROOT"), "SYNTHCITY_DATA_ROOT": os.environ.get("SYNTHCITY_DATA_ROOT")},
            "args": vars(args),
        },
    )

    syn_path = pathlib.Path(args.synthetic_path).expanduser().resolve()
    if not syn_path.exists():
        raise SystemExit(f"synthetic_path not found: {syn_path}")
    if syn_path.suffix.lower() == ".parquet":
        syn = pd.read_parquet(syn_path)
    else:
        syn = pd.read_csv(syn_path, low_memory=False)
    syn = _harmonize_synthetic_columns(syn, puma_group_col=str(args.puma_group_col))

    # --- L2: Tract marginals vs ACS targets_long ---
    l2 = None
    if args.acs_targets_long:
        tgt_path = pathlib.Path(args.acs_targets_long).expanduser().resolve()
        if not tgt_path.exists():
            raise SystemExit(f"acs_targets_long not found: {tgt_path}")
        if tgt_path.suffix.lower() == ".parquet":
            tgt = pd.read_parquet(tgt_path)
        else:
            tgt = pd.read_csv(tgt_path, low_memory=False)

        # Try to evaluate variables present in targets_long, but only those present in syn.
        variables = sorted(set(tgt["variable"].astype(str).unique().tolist()))
        l2 = compute_stats_metrics_against_targets_long(
            synthetic=syn,
            targets_long=tgt,
            group_col=str(args.tract_group_col),
            variables=variables,
        )
        _write_json(out_dir / "L2_tract_validation.json", l2)

    # --- L3: PUMA validation vs PUMS (note: this is not "holdout" unless you pass a holdout slice) ---
    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    if args.pums_person_zip:
        pums_zip = pathlib.Path(args.pums_person_zip).expanduser().resolve()
    else:
        pums_zip = _resolve_pums_person_zip(
            data_root=data_root, pums_year=int(args.pums_year), pums_period=str(args.pums_period), statefp=str(args.statefp)
        )
    member = _find_first_csv_in_zip(pums_zip)
    usecols = ["PUMA", "PWGTP", "AGEP", "SEX", "PINCP", "ESR"]
    with zipfile.ZipFile(pums_zip) as zf, zf.open(member) as f:
        ref = pd.read_csv(f, usecols=lambda c: c in set(usecols), low_memory=False)

    if "PUMA" not in ref.columns:
        raise SystemExit("PUMS reference missing PUMA column.")
    ref["puma"] = ref["PUMA"].astype(str)
    if "puma" not in syn.columns and str(args.puma_group_col) in syn.columns:
        syn["puma"] = syn[str(args.puma_group_col)].astype(str)

    # Ensure required columns exist.
    for col in ["puma", "AGEP", "SEX", "PINCP"]:
        if col not in syn.columns:
            raise SystemExit(f"synthetic missing column required for L3: {col}")
    for col in ["puma", "AGEP", "SEX", "PINCP"]:
        if col not in ref.columns:
            raise SystemExit(f"reference missing column required for L3: {col}")

    # Standardize columns to match compute_stats_metrics defaults.
    syn_l3 = syn.copy()
    ref_l3 = ref.copy()
    syn_l3["puma"] = syn_l3["puma"].astype(str)
    ref_l3["puma"] = ref_l3["puma"].astype(str)

    if "ESR" not in syn_l3.columns:
        syn_l3["ESR"] = "NA"
    if "ESR" not in ref_l3.columns:
        ref_l3["ESR"] = "NA"

    l3 = compute_stats_metrics(
        synthetic=syn_l3,
        reference=ref_l3.rename(columns={"puma": "puma"}),
        group_col="puma",
        continuous_cols=["AGEP", "PINCP"],
        categorical_cols=["SEX", "ESR"],
    )
    _write_json(out_dir / "L3_puma_validation.json", l3)

    summary = {
        "created_utc": _utc_now_iso(),
        "paths": {
            "synthetic": str(syn_path),
            "acs_targets_long": (str(args.acs_targets_long) if args.acs_targets_long else None),
            "pums_zip": str(pums_zip),
        },
        "layers": {
            "L2_tract": ("written" if l2 is not None else "skipped"),
            "L3_puma": "written",
        },
    }
    _write_json(out_dir / "validation_summary.json", summary)
    print(f"[ok] wrote: {out_dir}")


if __name__ == "__main__":
    main()
