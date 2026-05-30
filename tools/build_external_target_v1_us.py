#!/usr/bin/env python3
from __future__ import annotations

"""
Build US-wide PUMS-derived PUMA-level joint targets under the external-condition v1 schema.

Design goal:
- keep the target PUMS-derived
- but redefine it under the same observable schema used by external ACS conditions
- produce a stable nationwide PUMA-level target for the first external-condition experiment
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
from src.synthpop.data.state_codes import STATEFP_TO_POSTAL as _STATEFP_TO_POSTAL_50
from tools.build_external_target_v1_michigan import (
    AGE_LABELS,
    ESR_LABELS,
    SCHL_LABELS,
    SEX_LABELS,
    SHAPE,
    _aggregate_state,
    _condition_alignment,
    _resolve_person_zip,
    _to_long_df,
    _to_wide_df,
    _utc_now_iso,
    _write_json,
)


def _parse_states(*, statefp: str, statefps: str, all_states: bool) -> list[str]:
    if bool(all_states):
        return sorted(_STATEFP_TO_POSTAL_50.keys())
    if str(statefps).strip():
        vals = [str(x).strip().zfill(2) for x in str(statefps).split(",") if str(x).strip()]
        if not vals:
            raise SystemExit("--statefps provided but empty.")
        bad = [s for s in vals if s not in _STATEFP_TO_POSTAL_50]
        if bad:
            raise SystemExit(f"Unsupported statefps: {bad}")
        return vals
    s = str(statefp).zfill(2)
    if s not in _STATEFP_TO_POSTAL_50:
        raise SystemExit(f"Unsupported statefp={s}")
    return [s]


def _scope_tag(states: list[str]) -> str:
    if len(states) == len(_STATEFP_TO_POSTAL_50):
        return "us"
    return "state" + "_".join(states)


def main() -> None:
    default_root = data_root()
    ap = argparse.ArgumentParser(prog="build_external_target_v1_us")
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
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--condition_csv", default=None)
    ap.add_argument(
        "--out_dir",
        default=str(default_root / "us" / "processed" / "external_targets"),
        help="Output directory.",
    )
    args = ap.parse_args()

    if float(args.alpha) < 0:
        raise SystemExit("--alpha must be >= 0")

    states = _parse_states(statefp=args.statefp, statefps=args.statefps, all_states=bool(args.all_states))
    pums_dir = pathlib.Path(args.pums_dir).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pums_dir.exists():
        raise SystemExit(f"pums_dir not found: {pums_dir}")

    rows: list[dict[str, Any]] = []
    state_infos: list[dict[str, Any]] = []
    for statefp in states:
        person_zip = _resolve_person_zip(pums_dir=pums_dir, statefp=statefp)
        st_rows, st_info = _aggregate_state(statefp=statefp, person_zip=person_zip, alpha=float(args.alpha))
        rows.extend(st_rows)
        state_infos.append(st_info)
        print(f"[ok] state={statefp} n_pumas={st_info['n_pumas']} n_rows_valid={st_info['n_rows_valid']}", file=sys.stderr)

    if not rows:
        raise SystemExit("No PUMA rows were produced.")

    wide = _to_wide_df(rows)
    long = _to_long_df(rows)

    scope_tag = _scope_tag(states)
    stem = f"exttarget_v1_pums_{int(args.pums_year)}_puma_{scope_tag}"
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
        "scope": scope_tag,
        "statefps": states,
        "n_states": int(len(states)),
        "pums_year": int(args.pums_year),
        "pums_period": str(args.pums_period),
        "pums_dir": str(pums_dir),
        "alpha": float(args.alpha),
        "shape": list(SHAPE),
        "K": int(np.prod(SHAPE)),
        "outputs": {
            "joint_wide_csv": str(wide_csv),
            "joint_long_csv": str(long_csv),
            "schema_json": str(schema_json),
        },
        "info": {
            "n_pumas": int(len(rows)),
            "state_summaries": state_infos,
        },
        "design_notes": [
            "Schema follows external_condition_v1 exactly.",
            "The target source can use a different ACS/PUMS release year from the external condition file.",
            "Nationwide build uses the US-level PUMS directory and aggregates one state zip at a time.",
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
