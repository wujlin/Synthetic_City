#!/usr/bin/env python3
from __future__ import annotations

"""
Build a lightweight "marginals_long" table from downloaded ACS detailed tables.

Why:
- detroit_fetch_public_data.py can download ACS 5-year tables at tract/BG.
- For validation and controlled generation we need a normalized, reviewable marginals format:
  (group, variable, category, target_count).

Scope (KISS, v0):
- Implements B01001 (age×sex) -> SEX + AGEP_bin (coarse bins aligned with stats.py defaults).
- Implements B19001 (household income bins) -> HHINCP_bin (ACS-native bins).
- Implements B23025 (employment status, 16+) -> ESR_16p (coarse categories aligned with stats external validation).
- Supports optional aggregation from tract -> puma via TIGER spatial join (centroid-in-polygon).
"""

import argparse
import csv
import json
import pathlib
from typing import Any


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency: {pkg}. Install it in your conda env.\n"
            "Recommended: conda install -c conda-forge pandas geopandas pyproj shapely"
        ) from e


def _utc_now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_dir(p: pathlib.Path) -> pathlib.Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _normalize_puma(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
        return str(int(float(value)))
    except Exception:
        return None


def _pick_col(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _interval_labels(edges: list[float]) -> list[str]:
    pd = _require("pandas")

    labels = []
    for i in range(len(edges) - 1):
        labels.append(str(pd.Interval(float(edges[i]), float(edges[i + 1]), closed="left")))
    return labels


def _load_acs_csv_gz(path: pathlib.Path) -> Any:
    pd = _require("pandas")
    return pd.read_csv(path, compression="gzip", low_memory=False)


def _make_tract_geoid(df) -> Any:
    pd = _require("pandas")

    state = df["state"].astype(str).str.zfill(2)
    county = df["county"].astype(str).str.zfill(3)
    tract = df["tract"].astype(str).str.zfill(6)
    return (state + county + tract).astype(str)


def _tract_to_puma_map(*, tiger_tract_zip: pathlib.Path, tiger_puma_zip: pathlib.Path) -> dict[str, str]:
    gpd = _require("geopandas")

    tract = gpd.read_file(f"zip://{tiger_tract_zip}")
    puma = gpd.read_file(f"zip://{tiger_puma_zip}")

    if tract.crs is None:
        tract = tract.set_crs(4269, allow_override=True)
    if puma.crs is None:
        puma = puma.set_crs(4269, allow_override=True)

    tract = tract.to_crs(3857)
    puma = puma.to_crs(3857)

    tract_geoid_col = _pick_col(list(tract.columns), ("GEOID", "GEOID20", "GEOID10"))
    puma_col = _pick_col(list(puma.columns), ("PUMACE20", "PUMA", "PUMACE10"))
    if tract_geoid_col is None:
        raise SystemExit(f"Cannot find tract GEOID column in: {tiger_tract_zip}")
    if puma_col is None:
        raise SystemExit(f"Cannot find PUMA code column in: {tiger_puma_zip}")

    cent = tract.geometry.centroid
    tract_cent = tract[[tract_geoid_col]].copy()
    tract_cent = tract_cent.rename(columns={tract_geoid_col: "tract_geoid"})
    tract_cent["geometry"] = cent

    joined = gpd.sjoin(tract_cent, puma[[puma_col, "geometry"]], how="left", predicate="within")
    out = {}
    for r in joined[["tract_geoid", puma_col]].itertuples(index=False):
        tg = str(r.tract_geoid)
        pc = _normalize_puma(getattr(r, puma_col))
        if tg and pc:
            out[tg] = pc
    return out


def _b01001_to_marginals(df, *, group_col: str) -> list[dict[str, Any]]:
    pd = _require("pandas")

    def num(col: str) -> Any:
        return pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    # Coarse bins aligned with src/synthpop/validation/stats.py defaults.
    age_edges = [0.0, 5.0, 18.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 1000.0]
    age_labels = _interval_labels(age_edges)

    # SEX totals: match ACS PUMS coding (1=male,2=female)
    sex_male = num("B01001_002E")
    sex_female = num("B01001_026E")

    # Age bins = sum of male+female detailed cells
    # Mapping indices follow ACS B01001 canonical layout.
    def s(cols: list[str]) -> Any:
        out = None
        for c in cols:
            v = num(c)
            out = v if out is None else (out + v)
        return out if out is not None else 0.0

    # Detailed age cells for each coarse bin
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

        out.append({"variable": "SEX", "category": "1", "target": float(sex_male.loc[idx]), group_col: g})
        out.append({"variable": "SEX", "category": "2", "target": float(sex_female.loc[idx]), group_col: g})

        for label, cols in age_bin_cols.items():
            out.append({"variable": "AGEP_bin", "category": str(label), "target": float(s(cols).loc[idx]), group_col: g})

    return out


def _b19001_to_marginals(df, *, group_col: str) -> list[dict[str, Any]]:
    pd = _require("pandas")

    def num(col: str) -> Any:
        return pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    # ACS-native bins (household income, past 12 months)
    # Left-closed bins; last bin is open-ended.
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
            if hi == float("inf"):
                cat = f"[{lo}, inf)"
            else:
                cat = f"[{lo}, {hi})"
            out.append({"variable": "HHINCP_bin", "category": cat, "target": float(num(col).loc[idx]), group_col: g})
    return out


def _b23025_to_marginals(df, *, group_col: str) -> list[dict[str, Any]]:
    pd = _require("pandas")

    def num(col: str) -> Any:
        return pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    # B23025: Employment status for the population 16 years and over.
    # Coarse categories (do NOT try to match PUMS ESR 1..6 directly):
    # - employed: employed (civilian) among 16+
    # - unemployed: unemployed (civilian) among 16+
    # - armed_forces: armed forces among 16+
    # - not_in_labor_force: not in labor force among 16+
    employed = num("B23025_004E")
    unemployed = num("B23025_005E")
    armed = num("B23025_006E")
    not_in_lf = num("B23025_007E")

    out: list[dict[str, Any]] = []
    for idx, g in df[group_col].astype(str).items():
        g = str(g)
        if not g or g == "nan":
            continue
        out.append({"variable": "ESR_16p", "category": "employed", "target": float(employed.loc[idx]), group_col: g})
        out.append({"variable": "ESR_16p", "category": "unemployed", "target": float(unemployed.loc[idx]), group_col: g})
        out.append({"variable": "ESR_16p", "category": "armed_forces", "target": float(armed.loc[idx]), group_col: g})
        out.append({"variable": "ESR_16p", "category": "not_in_labor_force", "target": float(not_in_lf.loc[idx]), group_col: g})
    return out


def main() -> None:
    from src.synthpop.paths import data_root as default_data_root

    p = argparse.ArgumentParser(prog="build_acs_marginals_long")
    p.add_argument("--out_root", default=str(default_data_root()))
    p.add_argument("--acs_year", type=int, default=2023)
    p.add_argument("--statefp", default="26")
    p.add_argument("--countyfp", default="163")
    p.add_argument("--geo_level", choices=["tract", "bg"], default="tract")
    p.add_argument("--tables", default="B01001,B19001")
    p.add_argument("--tiger_year", type=int, default=2023)
    p.add_argument(
        "--aggregate_to",
        choices=["none", "puma"],
        default="puma",
        help='Aggregate tract-level ACS to "puma" using TIGER puma20 (centroid join).',
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_root = pathlib.Path(args.out_root).expanduser().resolve()
    acs_year = int(args.acs_year)
    statefp = str(args.statefp).zfill(2)
    countyfp = str(args.countyfp).zfill(3)
    geo_level = str(args.geo_level)
    aggregate_to = str(args.aggregate_to)

    tables = [t.strip() for t in str(args.tables).split(",") if t.strip()]
    if not tables:
        raise SystemExit("--tables cannot be empty")

    acs_dir = out_root / "detroit" / "raw" / "census" / "acs" / f"acs5_{acs_year}"
    if not acs_dir.exists():
        raise SystemExit(f"ACS dir not found (run fetch first): {acs_dir}")

    # Currently only tract->puma aggregation is supported.
    if aggregate_to == "puma" and geo_level != "tract":
        raise SystemExit('--aggregate_to="puma" currently requires --geo_level=tract (KISS).')

    # Load tables
    table_dfs: dict[str, Any] = {}
    for table_id in tables:
        in_path = acs_dir / f"acs5_{acs_year}_{table_id}_{geo_level}_state{statefp}_county{countyfp}.csv.gz"
        if not in_path.exists():
            raise SystemExit(f"ACS table not found: {in_path} (run detroit_fetch_public_data.py acs first)")
        table_dfs[table_id] = _load_acs_csv_gz(in_path)

    # Build tract geoid
    for table_id, df in table_dfs.items():
        if geo_level == "tract":
            for col in ["state", "county", "tract"]:
                if col not in df.columns:
                    raise SystemExit(f"{table_id} missing expected column: {col}")
            df["tract_geoid"] = _make_tract_geoid(df)
        else:
            raise SystemExit("bg not implemented in v0 build script yet (use tract).")

    # Optional aggregation to PUMA
    group_col = "tract_geoid"
    if aggregate_to == "puma":
        tiger_dir = out_root / "detroit" / "raw" / "geo" / "tiger" / f"TIGER{int(args.tiger_year)}"
        tiger_tract_zip = tiger_dir / f"tl_{int(args.tiger_year)}_{statefp}_tract.zip"
        tiger_puma_zip = tiger_dir / f"tl_{int(args.tiger_year)}_{statefp}_puma20.zip"
        if not tiger_tract_zip.exists() or not tiger_puma_zip.exists():
            raise SystemExit(
                "TIGER tract/puma zip not found. Run detroit_fetch_public_data.py tiger first.\n"
                f"Expected:\n  - {tiger_tract_zip}\n  - {tiger_puma_zip}"
            )
        tract_to_puma = _tract_to_puma_map(tiger_tract_zip=tiger_tract_zip, tiger_puma_zip=tiger_puma_zip)
        if not tract_to_puma:
            raise SystemExit("Failed to build tract->puma map (empty).")
        for _table_id, df in table_dfs.items():
            df["puma"] = df["tract_geoid"].map(lambda x: tract_to_puma.get(str(x)))
            df["puma"] = df["puma"].astype(str)
        group_col = "puma"

    # Build marginals records
    records: list[dict[str, Any]] = []
    if "B01001" in table_dfs:
        df = table_dfs["B01001"]
        if aggregate_to == "puma":
            agg_cols = [c for c in df.columns if c.startswith("B01001_") and c.endswith("E")]
            df = df.groupby("puma", dropna=False, sort=False)[agg_cols].sum().reset_index()
        records.extend(_b01001_to_marginals(df, group_col=group_col))

    if "B19001" in table_dfs:
        df = table_dfs["B19001"]
        if aggregate_to == "puma":
            agg_cols = [c for c in df.columns if c.startswith("B19001_") and c.endswith("E")]
            df = df.groupby("puma", dropna=False, sort=False)[agg_cols].sum().reset_index()
        records.extend(_b19001_to_marginals(df, group_col=group_col))

    if "B23025" in table_dfs:
        df = table_dfs["B23025"]
        if aggregate_to == "puma":
            agg_cols = [c for c in df.columns if c.startswith("B23025_") and c.endswith("E")]
            df = df.groupby("puma", dropna=False, sort=False)[agg_cols].sum().reset_index()
        records.extend(_b23025_to_marginals(df, group_col=group_col))

    if not records:
        raise SystemExit("No marginals records produced. Check --tables.")

    # Write output
    out_dir = out_root / "detroit" / "processed" / "marginals"
    _ensure_dir(out_dir)
    out_name = f"acs5_{acs_year}_marginals_long_{group_col}_state{statefp}_county{countyfp}.csv"
    out_path = out_dir / out_name
    if out_path.exists() and not args.overwrite:
        print(f"[skip] exists: {out_path}")
        return

    # Stable column order: group_col first
    cols = [group_col, "variable", "category", "target", "table_id", "source", "acs_year", "geo_level"]
    geo_level_out = "puma" if group_col == "puma" else ("tract" if group_col == "tract_geoid" else group_col)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            # Best-effort table_id inference from variable name.
            # (We keep it explicit to make review/auditing easy.)
            if r.get("variable") in {"SEX", "AGEP_bin"}:
                table_id = "B01001"
            elif r.get("variable") == "HHINCP_bin":
                table_id = "B19001"
            elif r.get("variable") == "ESR_16p":
                table_id = "B23025"
            else:
                table_id = None
            r2 = {
                group_col: r.get(group_col),
                "variable": r.get("variable"),
                "category": r.get("category"),
                "target": r.get("target"),
                "table_id": table_id,
                "source": "acs5",
                "acs_year": int(acs_year),
                "geo_level": geo_level_out,
            }
            w.writerow(r2)

    meta = {
        "dataset": "ACS 5-year marginals_long (derived)",
        "acs_year": int(acs_year),
        "statefp": statefp,
        "countyfp": countyfp,
        "geo_level_input": geo_level,
        "geo_level_output": geo_level_out,
        "group_col_output": group_col,
        "tables": tables,
        "created_utc": _utc_now_iso(),
        "out_path": str(out_path),
    }
    (out_path.with_suffix(out_path.suffix + ".metadata.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()
