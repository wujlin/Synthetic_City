#!/usr/bin/env python3
from __future__ import annotations

"""
Aggregate external-condition v1 marginals into a lower-dimensional external-condition v1-lite schema.
"""

import argparse
import json
import pathlib
import sys
from typing import Any

import pandas as pd


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.build_external_condition_v1_michigan import _utc_now_iso
from tools.train_us_puma_5var_diffusion import _canon_puma5, _canon_statefp, _canon_uid_loose


AGE_MAP = {
    "[0.0, 5.0)": "[0.0, 18.0)",
    "[5.0, 18.0)": "[0.0, 18.0)",
    "[18.0, 25.0)": "[18.0, 35.0)",
    "[25.0, 35.0)": "[18.0, 35.0)",
    "[35.0, 45.0)": "[35.0, 65.0)",
    "[45.0, 55.0)": "[35.0, 65.0)",
    "[55.0, 65.0)": "[35.0, 65.0)",
    "[65.0, 75.0)": "[65.0, 1000.0)",
    "[75.0, 85.0)": "[65.0, 1000.0)",
    "[85.0, 1000.0)": "[65.0, 1000.0)",
}
SCHL_MAP = {
    "not_25p": "not_25p",
    "less_than_high_school": "non_bachelor",
    "high_school_or_ged": "non_bachelor",
    "some_college_or_assoc": "non_bachelor",
    "bachelor_plus": "bachelor_plus",
}
ESR_MAP = {
    "not_16p": "not_16p",
    "employed": "employed",
    "unemployed": "not_employed",
    "armed_forces": "not_employed",
    "not_in_labor_force": "not_employed",
}

LITE_CATEGORIES = {
    "AGEP_bin": ["[0.0, 18.0)", "[18.0, 35.0)", "[35.0, 65.0)", "[65.0, 1000.0)"],
    "SEX": ["1", "2"],
    "SCHL_allpop": ["not_25p", "non_bachelor", "bachelor_plus"],
    "ESR_allpop": ["not_16p", "employed", "not_employed"],
}


def _normalize_geo_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "statefp" in out.columns:
        out["statefp"] = out["statefp"].map(_canon_statefp)
    if "puma" in out.columns:
        out["puma"] = out["puma"].map(_canon_puma5)
    if "puma_uid" in out.columns:
        out["puma_uid"] = out["puma_uid"].map(_canon_uid_loose)
    return out


def _map_category(var: str, cat: str) -> str:
    if var == "AGEP_bin":
        return AGE_MAP[cat]
    if var == "SEX":
        return cat
    if var == "SCHL_allpop":
        return SCHL_MAP[cat]
    if var == "ESR_allpop":
        return ESR_MAP[cat]
    raise SystemExit(f"Unsupported variable in condition CSV: {var}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_external_condition_v1_lite")
    ap.add_argument("--condition_csv", required=True)
    ap.add_argument("--out_path", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_path = pathlib.Path(args.condition_csv).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"condition_csv not found: {in_path}")

    default_out = in_path.with_name(in_path.name.replace("extcond_v1_", "extcond_v1_lite_"))
    out_path = pathlib.Path(args.out_path).expanduser().resolve() if args.out_path else default_out
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"out_path exists: {out_path} (use --overwrite)")

    dtype_map = {"statefp": str, "puma": str, "puma_uid": str, "variable": str, "category": str}
    df = pd.read_csv(in_path, dtype={k: v for k, v in dtype_map.items() if k in pd.read_csv(in_path, nrows=0).columns}, low_memory=False)
    df = _normalize_geo_cols(df)
    required = {"variable", "category", "target"}
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise SystemExit(f"condition_csv missing columns: {miss}")

    geo_cols = [c for c in ["statefp", "puma", "puma_uid"] if c in df.columns]
    df["category"] = df.apply(lambda r: _map_category(str(r["variable"]), str(r["category"])), axis=1)
    group_cols = geo_cols + ["variable", "category"]
    keep_cols = [c for c in ["table_id", "universe", "source", "acs_year", "geo_level"] if c in df.columns]

    agg = df.groupby(group_cols, as_index=False, sort=False)["target"].sum()
    for c in keep_cols:
        agg[c] = df.groupby(group_cols, as_index=False, sort=False)[c].first()[c]
    agg["schema"] = "external_condition_v1_lite"

    ordered_cols = geo_cols + ["variable", "category", "target"] + keep_cols + ["schema"]
    agg = agg[ordered_cols]
    agg.to_csv(out_path, index=False)

    meta = {
        "dataset": "External condition v1-lite",
        "schema": "external_condition_v1_lite",
        "source_condition_csv": str(in_path),
        "out_path": str(out_path),
        "variables": LITE_CATEGORIES,
        "created_utc": _utc_now_iso(),
    }
    out_path.with_suffix(out_path.suffix + ".metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()
