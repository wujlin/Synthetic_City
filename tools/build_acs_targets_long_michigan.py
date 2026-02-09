#!/usr/bin/env python3
from __future__ import annotations

"""
Build ACS "targets_long" (marginals_long) for Michigan (statewide) at tract level.

Why:
- compute_stats_metrics_against_targets_long expects a normalized format:
    (tract_geoid, variable, category, target)
- Our workstation downloads sometimes use custom filenames like:
    acs5_2022_B01001_tract_michigan.csv.gz
  instead of the detroit_fetch_public_data.py naming convention.

Scope (KISS, v0):
- B01001 -> SEX + AGEP_bin (coarse bins aligned with src/synthpop/validation/stats.py)
- B23025 -> ESR_16p (coarse 16+ categories aligned with stats.py derived ESR_16p)
- Optionally include B19001 (HHINCP_bin) for completeness (usually not directly comparable yet).

Output:
  <data_root>/detroit/processed/marginals/acs5_<year>_marginals_long_tract_state26_michigan.csv

Notes:
- This script is intentionally "schema-tolerant": it tries multiple ways to locate tract GEOID.
- It does not require geopandas.
"""

import argparse
import csv
import json
import pathlib
import sys
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
    import datetime as _dt

    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_dir(p: pathlib.Path) -> pathlib.Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_csv_gz(path: pathlib.Path) -> Any:
    pd = _require("pandas")
    return pd.read_csv(path, compression="gzip", low_memory=False)


def _interval_labels(edges: list[float]) -> list[str]:
    pd = _require("pandas")

    labels = []
    for i in range(len(edges) - 1):
        labels.append(str(pd.Interval(float(edges[i]), float(edges[i + 1]), closed="left")))
    return labels


def _make_tract_geoid(df: Any) -> Any:
    pd = _require("pandas")

    cols = set(df.columns.astype(str).tolist())
    if "tract_geoid" in cols:
        return df["tract_geoid"].astype(str)

    if {"state", "county", "tract"} <= cols:
        state = df["state"].astype(str).str.zfill(2)
        county = df["county"].astype(str).str.zfill(3)
        tract = df["tract"].astype(str).str.zfill(6)
        return (state + county + tract).astype(str)

    # Common alt columns in some exports.
    for c in ("GEOID", "GEOID20", "geoid", "tract_geoid20"):
        if c in cols:
            s = df[c].astype(str).str.replace(r"[^0-9]", "", regex=True)
            # Prefer the last 11 digits for tract.
            return s.str[-11:].astype(str)

    # GEO_ID looks like "1400000US26163500100"
    if "GEO_ID" in cols:
        s = df["GEO_ID"].astype(str)
        s = s.str.replace("US", "", regex=False).str.replace(r"[^0-9]", "", regex=True)
        return s.str[-11:].astype(str)

    raise SystemExit(f"Cannot derive tract_geoid; columns={list(df.columns)[:40]}")


def _b01001_records(df: Any, *, group_col: str) -> list[dict[str, Any]]:
    pd = _require("pandas")

    def num(col: str) -> Any:
        return pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    # Coarse bins aligned with stats.py defaults.
    age_edges = [0.0, 5.0, 18.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 1000.0]
    age_labels = _interval_labels(age_edges)

    # Sex totals.
    sex_male = num("B01001_002E")
    sex_female = num("B01001_026E")

    def s(cols: list[str]) -> Any:
        out = None
        for c in cols:
            v = num(c)
            out = v if out is None else (out + v)
        return out if out is not None else 0.0

    age_bin_cols = {
        age_labels[0]: ["B01001_003E", "B01001_027E"],  # <5
        age_labels[1]: ["B01001_004E", "B01001_005E", "B01001_006E", "B01001_028E", "B01001_029E", "B01001_030E"],  # 5-17
        age_labels[2]: [
            "B01001_007E",
            "B01001_008E",
            "B01001_009E",
            "B01001_010E",
            "B01001_031E",
            "B01001_032E",
            "B01001_033E",
            "B01001_034E",
        ],  # 18-24
        age_labels[3]: ["B01001_011E", "B01001_012E", "B01001_035E", "B01001_036E"],  # 25-34
        age_labels[4]: ["B01001_013E", "B01001_014E", "B01001_037E", "B01001_038E"],  # 35-44
        age_labels[5]: ["B01001_015E", "B01001_016E", "B01001_039E", "B01001_040E"],  # 45-54
        age_labels[6]: ["B01001_017E", "B01001_018E", "B01001_019E", "B01001_041E", "B01001_042E", "B01001_043E"],  # 55-64
        age_labels[7]: ["B01001_020E", "B01001_021E", "B01001_022E", "B01001_044E", "B01001_045E", "B01001_046E"],  # 65-74
        age_labels[8]: ["B01001_023E", "B01001_024E", "B01001_047E", "B01001_048E"],  # 75-84
        age_labels[9]: ["B01001_025E", "B01001_049E"],  # 85+
    }

    out: list[dict[str, Any]] = []
    for idx, g in df[group_col].astype(str).items():
        g = str(g)
        if not g or g == "nan":
            continue
        out.append({group_col: g, "variable": "SEX", "category": "1", "target": float(sex_male.loc[idx])})
        out.append({group_col: g, "variable": "SEX", "category": "2", "target": float(sex_female.loc[idx])})
        for label, cols in age_bin_cols.items():
            out.append({group_col: g, "variable": "AGEP_bin", "category": str(label), "target": float(s(cols).loc[idx])})
    return out


def _b23025_records(df: Any, *, group_col: str) -> list[dict[str, Any]]:
    pd = _require("pandas")

    def num(col: str) -> Any:
        return pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    employed = num("B23025_004E")
    unemployed = num("B23025_005E")
    armed = num("B23025_006E")
    not_in_lf = num("B23025_007E")

    out: list[dict[str, Any]] = []
    for idx, g in df[group_col].astype(str).items():
        g = str(g)
        if not g or g == "nan":
            continue
        out.append({group_col: g, "variable": "ESR_16p", "category": "employed", "target": float(employed.loc[idx])})
        out.append({group_col: g, "variable": "ESR_16p", "category": "unemployed", "target": float(unemployed.loc[idx])})
        out.append({group_col: g, "variable": "ESR_16p", "category": "armed_forces", "target": float(armed.loc[idx])})
        out.append({group_col: g, "variable": "ESR_16p", "category": "not_in_labor_force", "target": float(not_in_lf.loc[idx])})
    return out


def _b19001_records(df: Any, *, group_col: str) -> list[dict[str, Any]]:
    pd = _require("pandas")

    def num(col: str) -> Any:
        return pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    bins = [
        (0.0, 10_000.0, "B19001_002E"),
        (10_000.0, 15_000.0, "B19001_003E"),
        (15_000.0, 20_000.0, "B19001_004E"),
        (20_000.0, 25_000.0, "B19001_005E"),
        (25_000.0, 30_000.0, "B19001_006E"),
        (30_000.0, 35_000.0, "B19001_007E"),
        (35_000.0, 40_000.0, "B19001_008E"),
        (40_000.0, 45_000.0, "B19001_009E"),
        (45_000.0, 50_000.0, "B19001_010E"),
        (50_000.0, 60_000.0, "B19001_011E"),
        (60_000.0, 75_000.0, "B19001_012E"),
        (75_000.0, 100_000.0, "B19001_013E"),
        (100_000.0, 125_000.0, "B19001_014E"),
        (125_000.0, 150_000.0, "B19001_015E"),
        (150_000.0, 200_000.0, "B19001_016E"),
        (200_000.0, float("inf"), "B19001_017E"),
    ]

    out: list[dict[str, Any]] = []
    for idx, g in df[group_col].astype(str).items():
        g = str(g)
        if not g or g == "nan":
            continue
        for lo, hi, col in bins:
            cat = f"[{lo}, inf)" if hi == float("inf") else f"[{lo}, {hi})"
            out.append({group_col: g, "variable": "HHINCP_bin", "category": cat, "target": float(num(col).loc[idx])})
    return out


def _default_table_path(*, acs_dir: pathlib.Path, acs_year: int, table_id: str) -> pathlib.Path:
    # Prefer detroit_fetch_public_data.py naming if present; else fall back to "michigan" tag.
    cand = [
        acs_dir / f"acs5_{acs_year}_{table_id}_tract_state26_countyall.csv.gz",
        acs_dir / f"acs5_{acs_year}_{table_id}_tract_michigan.csv.gz",
    ]
    for p in cand:
        if p.exists():
            return p
    return cand[0]


def main() -> None:
    from src.synthpop.paths import data_root as default_data_root

    ap = argparse.ArgumentParser(prog="build_acs_targets_long_michigan")
    ap.add_argument("--data_root", default=str(default_data_root()))
    ap.add_argument("--acs_year", type=int, default=2022)
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--geo_level", choices=["tract"], default="tract")
    ap.add_argument("--tables", default="B01001,B23025")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    acs_year = int(args.acs_year)
    statefp = str(args.statefp).zfill(2)
    if statefp != "26":
        raise SystemExit("v0 script is Michigan-only (statefp=26).")

    tables = [t.strip() for t in str(args.tables).split(",") if t.strip()]
    if not tables:
        raise SystemExit("--tables cannot be empty")

    acs_dir = data_root / "detroit" / "raw" / "census" / "acs" / f"acs5_{acs_year}"
    if not acs_dir.exists():
        raise SystemExit(f"ACS dir not found: {acs_dir}")

    # Load requested tables.
    dfs: dict[str, Any] = {}
    for t in tables:
        path = _default_table_path(acs_dir=acs_dir, acs_year=acs_year, table_id=t)
        if not path.exists():
            raise SystemExit(f"ACS table not found: {path}")
        dfs[t] = _load_csv_gz(path)

    # Normalize tract geoid.
    for t, df in dfs.items():
        df["tract_geoid"] = _make_tract_geoid(df).astype(str)
        dfs[t] = df

    records: list[dict[str, Any]] = []
    group_col = "tract_geoid"
    if "B01001" in dfs:
        records.extend(_b01001_records(dfs["B01001"], group_col=group_col))
    if "B23025" in dfs:
        records.extend(_b23025_records(dfs["B23025"], group_col=group_col))
    if "B19001" in dfs:
        records.extend(_b19001_records(dfs["B19001"], group_col=group_col))

    if not records:
        raise SystemExit("No records produced; check --tables.")

    out_dir = data_root / "detroit" / "processed" / "marginals"
    _ensure_dir(out_dir)
    out_path = (
        pathlib.Path(args.out_path).expanduser().resolve()
        if args.out_path
        else (out_dir / f"acs5_{acs_year}_marginals_long_tract_state{statefp}_michigan.csv")
    )
    if out_path.exists() and not args.overwrite:
        print(f"[skip] exists: {out_path}")
        return

    cols = [group_col, "variable", "category", "target", "table_id", "source", "acs_year", "geo_level"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            var = r.get("variable")
            if var in {"SEX", "AGEP_bin"}:
                table_id = "B01001"
            elif var == "ESR_16p":
                table_id = "B23025"
            elif var == "HHINCP_bin":
                table_id = "B19001"
            else:
                table_id = None
            w.writerow(
                {
                    group_col: r.get(group_col),
                    "variable": var,
                    "category": r.get("category"),
                    "target": r.get("target"),
                    "table_id": table_id,
                    "source": "acs5",
                    "acs_year": int(acs_year),
                    "geo_level": "tract",
                }
            )

    meta = {
        "dataset": "ACS 5-year targets_long (derived)",
        "acs_year": int(acs_year),
        "statefp": statefp,
        "geo_level": "tract",
        "tables": tables,
        "created_utc": _utc_now_iso(),
        "out_path": str(out_path),
        "acs_dir": str(acs_dir),
    }
    (out_path.with_suffix(out_path.suffix + ".metadata.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()

