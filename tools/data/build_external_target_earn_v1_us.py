#!/usr/bin/env python3
from __future__ import annotations

"""
Build US-wide PUMS-derived PUMA-level earnings-proxy targets.

Target variable:
- EARN_16p_bin derived from PERNP

Question answered by this artifact:
- Is there a stable nationwide PUMA-level earnings proxy target that can be paired
  with the existing external full-condition pipeline?
"""

import argparse
import json
import pathlib
import sys
from typing import Any

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.paths import data_root
from tools.data.build_external_target_earn_v1_michigan import _aggregate_state, _utc_now_iso
from src.synthpop.data.state_codes import STATEFP_TO_POSTAL as _STATEFP_TO_POSTAL_50
from tools.model.train_us_puma_5var_diffusion import _canon_statefp


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
    ap = argparse.ArgumentParser(prog="build_external_target_earn_v1_us")
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
    ap.add_argument(
        "--out_dir",
        default=str(default_root / "us" / "processed" / "external_targets"),
        help="Output directory.",
    )
    args = ap.parse_args()

    states = _parse_states(statefp=args.statefp, statefps=args.statefps, all_states=bool(args.all_states))
    pums_dir = pathlib.Path(args.pums_dir).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pums_dir.exists():
        raise SystemExit(f"pums_dir not found: {pums_dir}")

    from tools.data.build_external_target_v1_michigan import _resolve_person_zip

    rows: list[dict[str, Any]] = []
    state_infos: list[dict[str, Any]] = []
    for statefp in states:
        person_zip = _resolve_person_zip(pums_dir=pums_dir, statefp=statefp)
        st_rows, st_info = _aggregate_state(statefp=statefp, person_path=person_zip)
        rows.extend(st_rows)
        state_infos.append(st_info)
        print(f"[ok] state={statefp} n_pumas={st_info['n_pumas']} n_rows_valid={st_info['n_rows_valid']}", file=sys.stderr)

    if not rows:
        raise SystemExit("No PUMA rows were produced.")

    wide_rows = []
    long_rows = []
    categories = [
        "not_in_earnings_universe",
        "lt_25k",
        "25k_50k",
        "50k_75k",
        "75k_100k",
        "ge_100k",
    ]
    for r in rows:
        row = {
            "statefp": _canon_statefp(r["statefp"]),
            "puma": r["puma"],
            "puma5": r["puma5"],
            "puma_uid": r["puma_uid"],
            "total_person_weight": float(r["total_person_weight"]),
            "n_persons_unweighted": int(r["n_persons_unweighted"]),
        }
        for i, v in enumerate(r["p_earn"]):
            row[f"p_earn_{i:02d}"] = float(v)
        for i, v in enumerate(r["count_earn"]):
            row[f"count_earn_{i:02d}"] = float(v)
            long_rows.append(
                {
                    "statefp": _canon_statefp(r["statefp"]),
                    "puma": r["puma"],
                    "puma_uid": r["puma_uid"],
                    "variable": "EARN_16p_bin",
                    "category": categories[i],
                    "prob": float(r["p_earn"][i]),
                    "count_weighted": float(v),
                }
            )
        wide_rows.append(row)

    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)

    scope_tag = _scope_tag(states)
    stem = f"exttarget_earn_v1_pums_{int(args.pums_year)}_puma_{scope_tag}"
    wide_csv = out_dir / f"{stem}.csv"
    long_csv = out_dir / f"{stem}_long.csv"
    schema_json = out_dir / f"{stem}.schema.json"
    metadata_json = out_dir / f"{stem}.metadata.json"

    wide.to_csv(wide_csv, index=False)
    long.to_csv(long_csv, index=False)
    schema_json.write_text(
        json.dumps(
            {
                "schema": "external_target_earn_v1",
                "variable_order": ["EARN_16p_bin"],
                "shape": [len(categories)],
                "K": len(categories),
                "categories": {"EARN_16p_bin": categories},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    meta = {
        "schema": "external_target_earn_v1",
        "created_at": _utc_now_iso(),
        "scope": scope_tag,
        "statefps": states,
        "n_states": int(len(states)),
        "pums_year": int(args.pums_year),
        "pums_period": str(args.pums_period),
        "pums_dir": str(pums_dir),
        "outputs": {
            "wide_csv": str(wide_csv),
            "long_csv": str(long_csv),
            "schema_json": str(schema_json),
        },
        "variable": {
            "EARN_16p_bin": {
                "source_variable": "PERNP",
                "categories": categories,
                "note": "All-population earnings proxy: age<16 or PERNP<=0 map to not_in_earnings_universe.",
            }
        },
        "info": {
            "n_pumas": int(len(rows)),
            "state_summaries": state_infos,
        },
    }
    metadata_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
