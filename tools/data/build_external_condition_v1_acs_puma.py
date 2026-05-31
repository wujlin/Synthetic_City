#!/usr/bin/env python3
from __future__ import annotations

"""
Build external-condition v1 directly from ACS 5-year detailed tables at PUMA geography.

Design goal:
- condition-led schema for the first realistic external-condition experiment
- avoid tract->PUMA aggregation when ACS can provide PUMA-level tables directly
- support a clean nationwide build for diffusion training

Schema v1 (all-population variables):
- SEX: categories {1, 2}
- AGEP_bin: 10 coarse bins from B01001
- SCHL_allpop: {not_25p, less_than_high_school, high_school_or_ged,
                some_college_or_assoc, bachelor_plus}
- ESR_allpop: {not_16p, employed, unemployed, armed_forces, not_in_labor_force}

Optional cross block:
- AGEP_SEX_cross: 20 age-by-sex categories kept directly from B01001
"""

import argparse
import csv
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.build_external_condition_v1_michigan import (
    AGE_SEX_CROSS_VAR,
    _b01001_records,
    _b15003_records,
    _b23025_records,
    _condition_schema_bundle,
    _utc_now_iso,
)
from synthpop.data.state_codes import STATEFP_TO_POSTAL as _STATEFP_TO_POSTAL_50
from synthpop.paths import data_root


TABLE_VAR_COUNTS = {
    "B01001": 49,
    "B15003": 25,
    "B23025": 7,
}


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Missing dependency: {pkg}. Install it in your conda env.") from e


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


def _var_names(table_id: str) -> list[str]:
    n = int(TABLE_VAR_COUNTS[table_id])
    return [f"{table_id}_{i:03d}E" for i in range(1, n + 1)]


def _fetch_acs_puma_table(*, acs_year: int, table_id: str, statefp: str, api_key: str | None) -> Any:
    pd = _require("pandas")

    base = f"https://api.census.gov/data/{int(acs_year)}/acs/acs5"
    params = {
        "get": ",".join(["NAME"] + _var_names(table_id)),
        "for": "public use microdata area:*",
        "in": f"state:{str(statefp).zfill(2)}",
    }
    if api_key:
        params["key"] = api_key
    url = base + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as resp:
        rows = json.loads(resp.read().decode("utf-8"))
    if not rows or len(rows) < 2:
        raise SystemExit(f"Empty ACS response for {table_id}, state={statefp}")
    df = pd.DataFrame(rows[1:], columns=rows[0])
    puma_col = "public use microdata area"
    if puma_col not in df.columns:
        raise SystemExit(f"ACS response missing '{puma_col}' for {table_id}, state={statefp}")
    df["statefp"] = str(statefp).zfill(2)
    df["puma"] = df[puma_col].astype(str).str.zfill(5)
    df["puma_uid"] = df["statefp"] + df["puma"]
    return df


def _build_state_records(*, acs_year: int, statefp: str, api_key: str | None, include_age_sex_cross: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pd = _require("pandas")

    dfs = {
        "B01001": _fetch_acs_puma_table(acs_year=acs_year, table_id="B01001", statefp=statefp, api_key=api_key),
        "B15003": _fetch_acs_puma_table(acs_year=acs_year, table_id="B15003", statefp=statefp, api_key=api_key),
        "B23025": _fetch_acs_puma_table(acs_year=acs_year, table_id="B23025", statefp=statefp, api_key=api_key),
    }

    pop_map = pd.to_numeric(dfs["B01001"]["B01001_001E"], errors="coerce").fillna(0.0)
    pop_map.index = dfs["B01001"]["puma_uid"].astype(str)
    total_pop_b15003 = dfs["B15003"]["puma_uid"].astype(str).map(pop_map).fillna(0.0)
    total_pop_b23025 = dfs["B23025"]["puma_uid"].astype(str).map(pop_map).fillna(0.0)

    records: list[dict[str, Any]] = []
    records.extend(_b01001_records(dfs["B01001"], group_col="puma_uid", include_age_sex_cross=bool(include_age_sex_cross)))
    records.extend(_b15003_records(dfs["B15003"], group_col="puma_uid", total_pop=total_pop_b15003))
    records.extend(_b23025_records(dfs["B23025"], group_col="puma_uid", total_pop=total_pop_b23025))

    for r in records:
        uid = str(r["puma_uid"])
        r["statefp"] = uid[:2]
        r["puma"] = uid[2:]

    info = {
        "statefp": str(statefp).zfill(2),
        "n_pumas": int(dfs["B01001"].shape[0]),
        "tables": {
            "B01001_rows": int(dfs["B01001"].shape[0]),
            "B15003_rows": int(dfs["B15003"].shape[0]),
            "B23025_rows": int(dfs["B23025"].shape[0]),
        },
    }
    return records, info


def main() -> None:
    default_root = data_root()
    ap = argparse.ArgumentParser(prog="build_external_condition_v1_acs_puma")
    ap.add_argument("--acs_year", type=int, default=2022)
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--statefps", default="")
    ap.add_argument("--all_states", action="store_true")
    ap.add_argument("--api_key", default=None, help="Optional Census API key or set CENSUS_API_KEY.")
    ap.add_argument("--include_age_sex_cross", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    states = _parse_states(statefp=args.statefp, statefps=args.statefps, all_states=bool(args.all_states))
    scope_tag = _scope_tag(states)
    out_path = (
        pathlib.Path(args.out_path).expanduser().resolve()
        if args.out_path
        else (
            default_root
            / "us"
            / "processed"
            / "external_conditions"
            / f"extcond_{'v1_agesex' if bool(args.include_age_sex_cross) else 'v1'}_acs5_{int(args.acs_year)}_puma_{scope_tag}.csv"
        ).resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"out_path exists: {out_path} (use --overwrite)")

    api_key = args.api_key or os.environ.get("CENSUS_API_KEY")

    records: list[dict[str, Any]] = []
    state_infos: list[dict[str, Any]] = []
    for statefp in states:
        st_records, st_info = _build_state_records(
            acs_year=int(args.acs_year),
            statefp=statefp,
            api_key=api_key,
            include_age_sex_cross=bool(args.include_age_sex_cross),
        )
        records.extend(st_records)
        state_infos.append(st_info)
        print(f"[ok] state={statefp} n_pumas={st_info['n_pumas']}", file=sys.stderr)

    records = sorted(
        records,
        key=lambda r: (
            str(r.get("statefp", "")),
            str(r.get("puma", "")),
            str(r.get("variable", "")),
            str(r.get("category", "")),
        ),
    )

    schema_name = "external_condition_v1_agesex" if bool(args.include_age_sex_cross) else "external_condition_v1"
    cols = [
        "statefp",
        "puma",
        "puma_uid",
        "variable",
        "category",
        "target",
        "table_id",
        "universe",
        "source",
        "acs_year",
        "geo_level",
        "schema",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow(
                {
                    "statefp": r.get("statefp"),
                    "puma": r.get("puma"),
                    "puma_uid": r.get("puma_uid"),
                    "variable": r.get("variable"),
                    "category": r.get("category"),
                    "target": r.get("target"),
                    "table_id": r.get("table_id"),
                    "universe": r.get("universe"),
                    "source": "acs5_api",
                    "acs_year": int(args.acs_year),
                    "geo_level": "puma",
                    "schema": schema_name,
                }
            )

    schema_bundle = _condition_schema_bundle(include_age_sex_cross=bool(args.include_age_sex_cross))
    meta = {
        "dataset": "US ACS external condition v1 at PUMA geography",
        "schema": schema_name,
        "acs_year": int(args.acs_year),
        "geo_level": "puma",
        "scope": scope_tag,
        "statefps": states,
        "n_states": int(len(states)),
        "n_rows": int(len(records)),
        "n_pumas": int(len(sorted({str(r["puma_uid"]) for r in records}))),
        "api_key_used": bool(api_key),
        "include_age_sex_cross": bool(args.include_age_sex_cross),
        "variable_order": schema_bundle["variable_order"],
        "variables": {
            **{
                var: {
                    "source_table": "B01001" if var in {"SEX", "AGEP_bin", AGE_SEX_CROSS_VAR} else ("B15003" if var == "SCHL_allpop" else "B23025"),
                    "categories": cats,
                    "universe": "all_persons",
                }
                for var, cats in schema_bundle["categories"].items()
            },
            **(
                {
                    AGE_SEX_CROSS_VAR: {
                        "source_table": "B01001",
                        "categories": schema_bundle["categories"][AGE_SEX_CROSS_VAR],
                        "universe": "all_persons",
                        "note": "Age-by-sex cross block kept directly from B01001 instead of being split into separate age and sex marginals only.",
                    }
                }
                if bool(args.include_age_sex_cross)
                else {}
            ),
        },
        "state_summaries": state_infos,
        "created_utc": _utc_now_iso(),
        "out_path": str(out_path),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".metadata.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    schema_path = out_path.with_suffix(out_path.suffix + ".schema.json")
    schema_path.write_text(
        json.dumps(
            {
                "schema": schema_name,
                "variable_order": schema_bundle["variable_order"],
                "categories": schema_bundle["categories"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()
