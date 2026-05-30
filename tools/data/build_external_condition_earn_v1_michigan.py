#!/usr/bin/env python3
from __future__ import annotations

"""
Build a Michigan PUMA-level external earnings proxy from ACS B20001.

Question answered by this artifact:
- Can tract-level ACS B20001 be aggregated into a stable PUMA-level all-population
  earnings proxy that aligns with a PUMS-derived PERNP target?
"""

import argparse
import csv
import json
import pathlib
import sys
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.build_external_condition_v1_michigan import (
    _build_tract_puma_map,
    _default_table_path,
    _ensure_dir,
    _load_csv_gz,
    _make_tract_geoid,
    _read_tract_puma_csv,
    _utc_now_iso,
)
from tools.data.external_earn_v1_schema import (
    B20001_FEMALE_COLS,
    B20001_MALE_COLS,
    EARN_LABELS,
    b20001_schema_present,
    coarse_b20001_groups,
)
from tools.model.train_us_puma_5var_diffusion import _canon_puma5


def _b20001_records(df: Any, *, group_col: str, total_pop: Any) -> list[dict[str, Any]]:
    pd = __import__("pandas")

    def num(col: str) -> Any:
        return pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    if not b20001_schema_present(df):
        raise SystemExit("B20001 schema not recognized; expected male/female detailed earnings bins.")

    total_earners = num("B20001_001E")
    not_in_universe = (total_pop - total_earners).clip(lower=0.0)
    coarse = coarse_b20001_groups()

    detailed = [num(B20001_MALE_COLS[i]) + num(B20001_FEMALE_COLS[i]) for i in range(20)]

    out: list[dict[str, Any]] = []
    for idx, g in df[group_col].astype(str).items():
        g = str(g)
        if not g or g == "nan":
            continue
        out.append(
            {
                group_col: g,
                "variable": "EARN_16p_bin",
                "category": "not_in_earnings_universe",
                "target": float(not_in_universe.loc[idx]),
                "table_id": "B20001+B01001",
                "universe": "all_persons",
            }
        )
        for cat, idxs in coarse.items():
            v = 0.0
            for k in idxs:
                v += float(detailed[k].loc[idx])
            out.append(
                {
                    group_col: g,
                    "variable": "EARN_16p_bin",
                    "category": cat,
                    "target": float(v),
                    "table_id": "B20001",
                    "universe": "all_persons",
                }
            )
    return out


def _aggregate_to_puma(records: list[dict[str, Any]], tract_to_puma: dict[str, str], *, group_col: str) -> list[dict[str, Any]]:
    pd = __import__("pandas")
    df = pd.DataFrame(records)
    df["puma"] = df[group_col].map(tract_to_puma)
    df = df[df["puma"].notna()].copy()
    grouped = (
        df.groupby(["puma", "variable", "category", "table_id", "universe"], sort=False, as_index=False)["target"].sum()
    )
    return grouped.to_dict(orient="records")


def main() -> None:
    from src.synthpop.paths import data_root as default_data_root

    ap = argparse.ArgumentParser(prog="build_external_condition_earn_v1_michigan")
    ap.add_argument("--data_root", default=str(default_data_root()))
    ap.add_argument("--acs_year", type=int, default=2022)
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--aggregate_to", choices=["tract", "puma"], default="puma")
    ap.add_argument("--tract_puma_csv", default="")
    ap.add_argument("--tract_zip", default="")
    ap.add_argument("--puma_zip", default="")
    ap.add_argument("--out_path", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    statefp = str(args.statefp).zfill(2)
    if statefp != "26":
        raise SystemExit("earn_v1 script is Michigan-only (statefp=26).")

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    acs_dir = data_root / "detroit" / "raw" / "census" / "acs" / f"acs5_{int(args.acs_year)}"
    if not acs_dir.exists():
        raise SystemExit(f"ACS dir not found: {acs_dir}")

    table_paths = {
        "B01001": _default_table_path(acs_dir=acs_dir, acs_year=int(args.acs_year), table_id="B01001"),
        "B20001": _default_table_path(acs_dir=acs_dir, acs_year=int(args.acs_year), table_id="B20001"),
    }
    for k, p in table_paths.items():
        if not p.exists():
            raise SystemExit(f"ACS table not found: {k} -> {p}")

    dfs = {k: _load_csv_gz(p) for k, p in table_paths.items()}
    for k, df in dfs.items():
        df["tract_geoid"] = _make_tract_geoid(df).astype(str)
        dfs[k] = df

    pd = __import__("pandas")
    total_pop = pd.to_numeric(dfs["B01001"]["B01001_001E"], errors="coerce").fillna(0.0)
    records = _b20001_records(dfs["B20001"], group_col="tract_geoid", total_pop=total_pop)

    group_col = "tract_geoid"
    tract_puma_source = None
    if args.aggregate_to == "puma":
        if args.tract_puma_csv:
            tract_to_puma = _read_tract_puma_csv(pathlib.Path(args.tract_puma_csv).expanduser().resolve())
            tract_puma_source = str(pathlib.Path(args.tract_puma_csv).expanduser().resolve())
        else:
            tract_zip = pathlib.Path(args.tract_zip).expanduser().resolve() if args.tract_zip else None
            puma_zip = pathlib.Path(args.puma_zip).expanduser().resolve() if args.puma_zip else None
            if tract_zip is None or puma_zip is None or not tract_zip.exists() or not puma_zip.exists():
                raise SystemExit("aggregate_to=puma requires either --tract_puma_csv or both --tract_zip and --puma_zip.")
            tract_to_puma = _build_tract_puma_map(tract_zip=tract_zip, puma_zip=puma_zip, statefp=statefp)
            tract_puma_source = f"spatial_join:{tract_zip}|{puma_zip}"
        records = _aggregate_to_puma(records, tract_to_puma, group_col="tract_geoid")
        group_col = "puma"

    out_dir = data_root / "detroit" / "processed" / "external_conditions"
    _ensure_dir(out_dir)
    default_name = f"extcond_earn_v1_acs5_{int(args.acs_year)}_{group_col}_state{statefp}_michigan.csv"
    out_path = pathlib.Path(args.out_path).expanduser().resolve() if args.out_path else (out_dir / default_name)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"out_path exists: {out_path} (use --overwrite)")

    cols = ["statefp", "puma", "puma_uid", group_col, "variable", "category", "target", "table_id", "universe", "source", "acs_year", "geo_level", "schema"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            puma5 = _canon_puma5(r.get("puma")) if group_col == "puma" else ""
            w.writerow(
                {
                    "statefp": statefp if group_col == "puma" else "",
                    "puma": puma5 if group_col == "puma" else "",
                    "puma_uid": (statefp + puma5) if group_col == "puma" and puma5 else "",
                    group_col: r.get(group_col),
                    "variable": r.get("variable"),
                    "category": r.get("category"),
                    "target": r.get("target"),
                    "table_id": r.get("table_id"),
                    "universe": r.get("universe"),
                    "source": "acs5",
                    "acs_year": int(args.acs_year),
                    "geo_level": group_col,
                    "schema": "external_condition_earn_v1",
                }
            )

    meta = {
        "dataset": "Michigan ACS external earnings proxy v1",
        "schema": "external_condition_earn_v1",
        "acs_year": int(args.acs_year),
        "statefp": statefp,
        "group_col": group_col,
        "tables": list(table_paths.keys()),
        "table_paths": {k: str(v) for k, v in table_paths.items()},
        "tract_puma_source": tract_puma_source,
        "variable": {
            "EARN_16p_bin": {
                "source_tables": ["B20001", "B01001"],
                "categories": EARN_LABELS,
                "universe": "all_persons",
                "note": "B20001 earnings histogram converted to an all-population variable by adding not_in_earnings_universe = total population - 16+ with earnings.",
            }
        },
        "created_utc": _utc_now_iso(),
        "out_path": str(out_path),
    }
    out_path.with_suffix(out_path.suffix + ".metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    out_path.with_suffix(out_path.suffix + ".schema.json").write_text(
        json.dumps(
            {
                "schema": "external_condition_earn_v1",
                "variable_order": ["EARN_16p_bin"],
                "categories": {"EARN_16p_bin": EARN_LABELS},
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
