#!/usr/bin/env python3
from __future__ import annotations

"""
Diagnose ACS B01001 (tract-level) coverage against the study area inferred from buildings.

Motivation:
- TVD=0.5 in our reports usually indicates one side is an all-zero vector. In this pipeline,
  the most common root cause is missing/zero ACS counts for a subset of tracts (often due to
  tract_geoid mismatch or incomplete ACS download scope).

This script is intentionally KISS:
- Inputs: (1) ACS B01001 tract CSV.gz, (2) buildings CSV (must include tract_geoid and puma).
- Output: a small JSON with per-PUMA tract coverage stats and example missing tracts.

Typical usage (workstation):
  python tools/data/diagnose_acs_b01001_coverage.py \\
    --acs_b01001_csv_gz "$DATA_ROOT/detroit/raw/census/acs/acs5_2023/acs5_2023_B01001_tract_state26_county163.csv.gz" \\
    --buildings_csv "$DATA_ROOT/detroit/processed/buildings/buildings_detroit_features_price.csv" \\
    --out_json "$OUT_DIR/metrics/acs_b01001_coverage.json"
"""

import argparse
import json
import pathlib
from typing import Any


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as e:
        raise RuntimeError(f"Missing dependency: {pkg}. Install it in your conda env.") from e


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _norm_digits(value: Any, *, width: int) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    if "." in s:
        try:
            s = str(int(float(s)))
        except Exception:
            pass
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(int(width))


def _norm_geoid11(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    if "." in s:
        try:
            s = str(int(float(s)))
        except Exception:
            pass
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    if len(digits) == 11:
        return digits
    if len(digits) > 11:
        return digits[-11:]
    return digits.zfill(11)


def _norm_puma(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    try:
        return str(int(float(s)))
    except Exception:
        digits = "".join(ch for ch in s if ch.isdigit())
        return digits if digits else None


def main() -> None:
    pd = _require("pandas")

    p = argparse.ArgumentParser(prog="diagnose_acs_b01001_coverage")
    p.add_argument("--acs_b01001_csv_gz", required=True, help="ACS B01001 tract CSV.gz.")
    p.add_argument("--buildings_csv", required=True, help="Buildings CSV used for the study area (must include tract_geoid and puma).")
    p.add_argument("--out_json", default=None, help="Output JSON path.")
    p.add_argument("--max_examples", type=int, default=10, help="Max missing tracts to list per PUMA (default: 10).")
    args = p.parse_args()

    acs_path = pathlib.Path(args.acs_b01001_csv_gz).expanduser().resolve()
    buildings_csv = pathlib.Path(args.buildings_csv).expanduser().resolve()
    if not acs_path.exists():
        raise SystemExit(f"acs_b01001_csv_gz not found: {acs_path}")
    if not buildings_csv.exists():
        raise SystemExit(f"buildings_csv not found: {buildings_csv}")

    out_json = (
        pathlib.Path(args.out_json).expanduser().resolve()
        if args.out_json
        else pathlib.Path("outputs") / "acs_b01001_coverage.json"
    )

    # --- Load buildings (tract->puma) ---
    b = pd.read_csv(buildings_csv, usecols=lambda c: c in {"tract_geoid", "puma"}, low_memory=False)
    if "tract_geoid" not in b.columns or "puma" not in b.columns:
        raise SystemExit("buildings_csv must contain tract_geoid and puma columns.")

    b["tract_geoid"] = b["tract_geoid"].map(_norm_geoid11)
    b["puma"] = b["puma"].map(_norm_puma)
    b = b.dropna(subset=["tract_geoid", "puma"]).drop_duplicates().copy()
    if b.empty:
        raise SystemExit("After normalization, no (tract_geoid, puma) pairs remain in buildings_csv.")

    # --- Load ACS B01001 ---
    acs = pd.read_csv(acs_path, compression="gzip", low_memory=False)
    needed = {"state", "county", "tract", "B01001_001E"}
    missing = sorted(needed - set(acs.columns))
    if missing:
        raise SystemExit(f"ACS B01001 missing columns: {missing}. Columns: {list(acs.columns)[:30]}")

    state = acs["state"].map(lambda v: _norm_digits(v, width=2))
    county = acs["county"].map(lambda v: _norm_digits(v, width=3))
    tract = acs["tract"].map(lambda v: _norm_digits(v, width=6))
    acs["tract_geoid"] = (state.fillna("") + county.fillna("") + tract.fillna("")).astype(str)
    acs = acs[acs["tract_geoid"].str.len() == 11].copy()

    acs["total_pop"] = pd.to_numeric(acs["B01001_001E"], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    acs_pop = acs.set_index("tract_geoid")["total_pop"].to_dict()

    # --- Coverage stats per PUMA ---
    by_puma: dict[str, Any] = {}
    for puma, sub in b.groupby("puma", sort=True):
        tracts = sorted(set(sub["tract_geoid"].astype(str).tolist()))
        pops = [float(acs_pop.get(t, 0.0)) for t in tracts]
        in_acs = [t for t in tracts if t in acs_pop]
        missing_tracts = [t for t in tracts if t not in acs_pop]
        pop0_tracts = [t for t in tracts if (t in acs_pop and float(acs_pop.get(t, 0.0)) <= 0.0)]
        by_puma[str(puma)] = {
            "n_tracts_buildings": int(len(tracts)),
            "n_tracts_in_acs": int(len(in_acs)),
            "n_tracts_missing_in_acs": int(len(missing_tracts)),
            "n_tracts_pop0_in_acs": int(len(pop0_tracts)),
            "total_pop_sum": float(sum(pops)),
            "missing_tracts_example": missing_tracts[: max(0, int(args.max_examples))],
            "pop0_tracts_example": pop0_tracts[: max(0, int(args.max_examples))],
        }

    bad_pumas_missing = sorted([p for p, v in by_puma.items() if int(v["n_tracts_in_acs"]) == 0])
    bad_pumas_pop0 = sorted([p for p, v in by_puma.items() if float(v["total_pop_sum"]) <= 0.0])

    report = {
        "meta": {
            "acs_b01001_csv_gz": str(acs_path),
            "buildings_csv": str(buildings_csv),
            "n_building_tract_pairs": int(b.shape[0]),
            "n_unique_tracts": int(b["tract_geoid"].nunique(dropna=True)),
            "n_unique_pumas": int(b["puma"].nunique(dropna=True)),
        },
        "by_puma": by_puma,
        "alerts": {
            "pumas_with_zero_tracts_in_acs": bad_pumas_missing,
            "pumas_with_zero_total_pop": bad_pumas_pop0,
        },
    }

    _write_json(out_json, report)
    print(f"[ok] wrote: {out_json}")
    if bad_pumas_missing or bad_pumas_pop0:
        print("[warn] detected suspicious PUMAs:")
        if bad_pumas_missing:
            print(f"  - zero tracts in ACS: {bad_pumas_missing}")
        if bad_pumas_pop0:
            print(f"  - zero total ACS pop: {bad_pumas_pop0}")


if __name__ == "__main__":
    main()

