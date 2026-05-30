#!/usr/bin/env python3
from __future__ import annotations

"""
Build a nationwide PUMA-level external-condition file with person-income added.

Schema:
- AGEP_bin
- SEX
- SCHL_allpop
- ESR_allpop
- PINCP_allpop_bin
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

from src.synthpop.paths import data_root
from tools.data.build_external_condition_v1_acs_puma import _parse_states, _scope_tag
from tools.data.build_external_condition_v1_michigan import (
    AGE_LABELS,
    ESR_ALLPOP_LABELS,
    SCHL_ALLPOP_LABELS,
    SEX_LABELS,
    _b01001_records,
    _b15003_records,
    _b23025_records,
    _utc_now_iso,
)
from src.synthpop.data.state_codes import STATEFP_TO_POSTAL as _STATEFP_TO_POSTAL_50
from tools.data.external_income_v1_schema import B06010_INCOME_COLS, B06010_TOTAL_COL, INCOME_LABELS, b06010_schema_present


TABLE_VAR_COUNTS = {
    "B01001": 49,
    "B06010": 55,
    "B15003": 25,
    "B23025": 7,
}


def _var_names(table_id: str) -> list[str]:
    n = int(TABLE_VAR_COUNTS[table_id])
    return [f"{table_id}_{i:03d}E" for i in range(1, n + 1)]


def _fetch_acs_puma_table(*, acs_year: int, table_id: str, statefp: str, api_key: str | None, var_names: list[str] | None = None) -> Any:
    pd = __import__("pandas")

    base = f"https://api.census.gov/data/{int(acs_year)}/acs/acs5"
    use_vars = list(var_names) if var_names else _var_names(table_id)
    params = {
        "get": ",".join(["NAME"] + use_vars),
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


def _b06010_records(df: Any, *, group_col: str, total_pop: Any) -> list[dict[str, Any]]:
    pd = __import__("pandas")

    def num(col: str) -> Any:
        return pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    if not b06010_schema_present(df):
        raise SystemExit("B06010 schema not recognized; expected total-level individual income bins.")

    total_15p = num(B06010_TOTAL_COL)
    not_15p = (total_pop - total_15p).clip(lower=0.0)

    out: list[dict[str, Any]] = []
    for idx, g in df[group_col].astype(str).items():
        g = str(g)
        if not g or g == "nan":
            continue
        out.append(
            {
                group_col: g,
                "variable": "PINCP_allpop_bin",
                "category": INCOME_LABELS[0],
                "target": float(not_15p.loc[idx]),
                "table_id": "B06010+B01001",
                "universe": "all_persons",
            }
        )
        for cat in INCOME_LABELS[1:]:
            cols = B06010_INCOME_COLS[cat]
            val = sum(float(num(c).loc[idx]) for c in cols)
            out.append(
                {
                    group_col: g,
                    "variable": "PINCP_allpop_bin",
                    "category": cat,
                    "target": float(val),
                    "table_id": "B06010",
                    "universe": "all_persons",
                }
            )
    return out


def _build_state_records(*, acs_year: int, statefp: str, api_key: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pd = __import__("pandas")

    dfs = {
        "B01001": _fetch_acs_puma_table(acs_year=acs_year, table_id="B01001", statefp=statefp, api_key=api_key),
        "B06010": _fetch_acs_puma_table(
            acs_year=acs_year,
            table_id="B06010",
            statefp=statefp,
            api_key=api_key,
            var_names=[f"B06010_{i:03d}E" for i in range(1, 12)],
        ),
        "B15003": _fetch_acs_puma_table(acs_year=acs_year, table_id="B15003", statefp=statefp, api_key=api_key),
        "B23025": _fetch_acs_puma_table(acs_year=acs_year, table_id="B23025", statefp=statefp, api_key=api_key),
    }

    pop_map = pd.to_numeric(dfs["B01001"]["B01001_001E"], errors="coerce").fillna(0.0)
    pop_map.index = dfs["B01001"]["puma_uid"].astype(str)
    total_pop_b15003 = dfs["B15003"]["puma_uid"].astype(str).map(pop_map).fillna(0.0)
    total_pop_b23025 = dfs["B23025"]["puma_uid"].astype(str).map(pop_map).fillna(0.0)
    total_pop_b06010 = dfs["B06010"]["puma_uid"].astype(str).map(pop_map).fillna(0.0)

    records: list[dict[str, Any]] = []
    records.extend(_b01001_records(dfs["B01001"], group_col="puma_uid", include_age_sex_cross=False))
    records.extend(_b15003_records(dfs["B15003"], group_col="puma_uid", total_pop=total_pop_b15003))
    records.extend(_b23025_records(dfs["B23025"], group_col="puma_uid", total_pop=total_pop_b23025))
    records.extend(_b06010_records(dfs["B06010"], group_col="puma_uid", total_pop=total_pop_b06010))

    for r in records:
        uid = str(r["puma_uid"])
        r["statefp"] = uid[:2]
        r["puma"] = uid[2:]

    info = {
        "statefp": str(statefp).zfill(2),
        "n_pumas": int(dfs["B01001"].shape[0]),
        "tables": {
            "B01001_rows": int(dfs["B01001"].shape[0]),
            "B06010_rows": int(dfs["B06010"].shape[0]),
            "B15003_rows": int(dfs["B15003"].shape[0]),
            "B23025_rows": int(dfs["B23025"].shape[0]),
        },
    }
    return records, info


def main() -> None:
    default_root = data_root()
    ap = argparse.ArgumentParser(prog="build_external_condition_v1_income_acs_puma")
    ap.add_argument("--acs_year", type=int, default=2022)
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--statefps", default="")
    ap.add_argument("--all_states", action="store_true")
    ap.add_argument("--api_key", default=None, help="Optional Census API key or set CENSUS_API_KEY.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    states = _parse_states(statefp=args.statefp, statefps=args.statefps, all_states=bool(args.all_states))
    bad = [s for s in states if s not in _STATEFP_TO_POSTAL_50]
    if bad:
        raise SystemExit(f"Unsupported statefps: {bad}")

    scope_tag = _scope_tag(states)
    out_path = (
        pathlib.Path(args.out_path).expanduser().resolve()
        if args.out_path
        else (
            default_root
            / "us"
            / "processed"
            / "external_conditions"
            / f"extcond_v1_income_v2_acs5_{int(args.acs_year)}_puma_{scope_tag}.csv"
        ).resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"out_path exists: {out_path} (use --overwrite)")

    api_key = args.api_key or os.environ.get("CENSUS_API_KEY")
    records: list[dict[str, Any]] = []
    state_infos: list[dict[str, Any]] = []
    for statefp in states:
        st_records, st_info = _build_state_records(acs_year=int(args.acs_year), statefp=statefp, api_key=api_key)
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
                    "schema": "external_condition_v1_plus_income_v2",
                }
            )

    variable_order = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop", "PINCP_allpop_bin"]
    categories = {
        "AGEP_bin": list(AGE_LABELS),
        "SEX": list(SEX_LABELS),
        "SCHL_allpop": list(SCHL_ALLPOP_LABELS),
        "ESR_allpop": list(ESR_ALLPOP_LABELS),
        "PINCP_allpop_bin": list(INCOME_LABELS),
    }

    meta = {
        "dataset": "US ACS external condition v1 with person income at PUMA geography",
        "schema": "external_condition_v1_plus_income_v2",
        "acs_year": int(args.acs_year),
        "geo_level": "puma",
        "scope": scope_tag,
        "statefps": states,
        "n_states": int(len(states)),
        "n_rows": int(len(records)),
        "n_pumas": int(len(sorted({str(r['puma_uid']) for r in records}))),
        "api_key_used": bool(api_key),
        "variable_order": variable_order,
        "variables": {
            "AGEP_bin": {"source_table": "B01001", "categories": list(AGE_LABELS), "universe": "all_persons"},
            "SEX": {"source_table": "B01001", "categories": list(SEX_LABELS), "universe": "all_persons"},
            "SCHL_allpop": {"source_table": "B15003+B01001", "categories": list(SCHL_ALLPOP_LABELS), "universe": "all_persons"},
            "ESR_allpop": {"source_table": "B23025+B01001", "categories": list(ESR_ALLPOP_LABELS), "universe": "all_persons"},
            "PINCP_allpop_bin": {
                "source_tables": ["B06010", "B01001"],
                "categories": list(INCOME_LABELS),
                "universe": "all_persons",
                "note": "All-population person-income proxy built by combining B06010 15+ income bins with B01001 total population. Bucket 0 is under-15/not-in-universe; bucket 1 is 15+ with no income.",
            },
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
                "schema": "external_condition_v1_plus_income_v2",
                "variable_order": variable_order,
                "categories": categories,
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
