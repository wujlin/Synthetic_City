#!/usr/bin/env python3
from __future__ import annotations

"""
Build external-condition v1 marginals for Michigan from ACS detailed tables.

Design goal:
- condition-led schema for a first realistic external-condition experiment
- keep everything explicit and reviewable
- default to PUMA-level output so it can connect to the current PUMA-level mainline

Schema v1 (all-population variables):
- SEX: categories {1, 2}
- AGEP_bin: 10 coarse bins from B01001
- SCHL_allpop: {not_25p, less_than_high_school, high_school_or_ged,
                some_college_or_assoc, bachelor_plus}
- ESR_allpop: {not_16p, employed, unemployed, armed_forces, not_in_labor_force}

Why this schema:
- B01001, B15003, and B23025 are already present in the repo
- education and employment universes differ from "all persons", so we convert them
  into explicit all-population variables by adding "not in universe" categories
- this avoids silently mixing universes inside the condition vector
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


def _ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _canon_puma5(value: Any) -> str:
    digits = "".join(ch for ch in str(value).strip() if ch.isdigit())
    if not digits:
        return ""
    return str(int(digits)).zfill(5)


def _load_csv_gz(path: pathlib.Path) -> Any:
    pd = _require("pandas")
    return pd.read_csv(path, compression="gzip", low_memory=False)


def _make_tract_geoid(df: Any) -> Any:
    cols = set(df.columns.astype(str).tolist())
    if "tract_geoid" in cols:
        return df["tract_geoid"].astype(str)
    if {"state", "county", "tract"} <= cols:
        state = df["state"].astype(str).str.zfill(2)
        county = df["county"].astype(str).str.zfill(3)
        tract = df["tract"].astype(str).str.zfill(6)
        return (state + county + tract).astype(str)
    for c in ("GEOID", "GEOID20", "geoid", "tract_geoid20"):
        if c in cols:
            s = df[c].astype(str).str.replace(r"[^0-9]", "", regex=True)
            return s.str[-11:].astype(str)
    if "GEO_ID" in cols:
        s = df["GEO_ID"].astype(str)
        s = s.str.replace("US", "", regex=False).str.replace(r"[^0-9]", "", regex=True)
        return s.str[-11:].astype(str)
    raise SystemExit(f"Cannot derive tract_geoid; columns={list(df.columns)[:40]}")


def _interval_labels(edges: list[float]) -> list[str]:
    pd = _require("pandas")
    labels = []
    for i in range(len(edges) - 1):
        labels.append(str(pd.Interval(float(edges[i]), float(edges[i + 1]), closed="left")))
    return labels


def _default_table_path(*, acs_dir: pathlib.Path, acs_year: int, table_id: str) -> pathlib.Path:
    cand = [
        acs_dir / f"acs5_{acs_year}_{table_id}_tract_state26_countyall.csv.gz",
        acs_dir / f"acs5_{acs_year}_{table_id}_tract_michigan.csv.gz",
    ]
    for p in cand:
        if p.exists():
            return p
    return cand[0]


def _read_tract_puma_csv(path: pathlib.Path) -> dict[str, str]:
    pd = _require("pandas")
    df = pd.read_csv(path, low_memory=False)
    needed = {"tract_geoid", "puma"}
    if not needed <= set(df.columns.astype(str).tolist()):
        raise SystemExit(f"tract_puma_csv must contain {needed}; got {list(df.columns)}")
    out = {}
    for r in df[["tract_geoid", "puma"]].itertuples(index=False):
        tract = str(r.tract_geoid)
        puma = str(r.puma)
        if tract and tract != "nan" and puma and puma != "nan":
            out[tract] = puma
    return out


def _build_tract_puma_map(*, tract_zip: pathlib.Path, puma_zip: pathlib.Path, statefp: str) -> dict[str, str]:
    gpd = _require("geopandas")
    pd = _require("pandas")

    tracts = gpd.read_file(f"zip://{tract_zip}")
    pumas = gpd.read_file(f"zip://{puma_zip}")

    tract_cols = list(tracts.columns)
    puma_cols = list(pumas.columns)

    def _find_col(columns: list[str], candidates: tuple[str, ...]) -> str | None:
        lookup = {c.upper(): c for c in columns}
        for cand in candidates:
            if cand.upper() in lookup:
                return lookup[cand.upper()]
        return None

    tract_geoid_col = _find_col(tract_cols, ("GEOID", "GEOID20"))
    statefp_col = _find_col(tract_cols, ("STATEFP", "STATEFP20"))
    countyfp_col = _find_col(tract_cols, ("COUNTYFP", "COUNTYFP20"))
    tractce_col = _find_col(tract_cols, ("TRACTCE", "TRACTCE20"))
    puma_col = _find_col(puma_cols, ("PUMACE20", "PUMACE", "GEOID20", "GEOID"))

    if puma_col is None:
        raise SystemExit(f"Cannot find PUMA code column in: {puma_cols}")

    if tract_geoid_col is None:
        if not (statefp_col and countyfp_col and tractce_col):
            raise SystemExit("Cannot build tract_geoid from tract shapefile columns.")
        tracts = tracts.copy()
        tracts["tract_geoid"] = (
            tracts[statefp_col].astype(str).str.zfill(2)
            + tracts[countyfp_col].astype(str).str.zfill(3)
            + tracts[tractce_col].astype(str).str.zfill(6)
        )
        tract_geoid_col = "tract_geoid"

    if statefp_col is not None:
        tracts = tracts[tracts[statefp_col].astype(str).str.zfill(2) == str(statefp).zfill(2)].copy()

    tracts_pts = tracts[[tract_geoid_col, "geometry"]].copy()
    tracts_pts["geometry"] = tracts_pts.geometry.representative_point()

    if tracts_pts.crs != pumas.crs:
        pumas = pumas.to_crs(tracts_pts.crs)

    joined = gpd.sjoin(
        tracts_pts,
        pumas[[puma_col, "geometry"]],
        how="left",
        predicate="within",
    )

    result = pd.DataFrame(
        {
            "tract_geoid": joined[tract_geoid_col].astype(str),
            "puma": joined[puma_col].astype(str),
        }
    )
    result = result.drop_duplicates(subset=["tract_geoid"], keep="first")
    result = result.dropna(subset=["puma"])
    result = result[~result["puma"].isin({"", "nan", "None"})].copy()
    result["puma"] = result["puma"].map(lambda v: str(int(float(v))) if str(v).replace(".", "", 1).isdigit() else str(v))
    return dict(zip(result["tract_geoid"], result["puma"]))


def _b01001_records(df: Any, *, group_col: str) -> list[dict[str, Any]]:
    pd = _require("pandas")

    def num(col: str) -> Any:
        return pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    age_edges = [0.0, 5.0, 18.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 1000.0]
    age_labels = _interval_labels(age_edges)

    sex_male = num("B01001_002E")
    sex_female = num("B01001_026E")

    def s(cols: list[str]) -> Any:
        out = None
        for c in cols:
            v = num(c)
            out = v if out is None else (out + v)
        return out if out is not None else 0.0

    age_bin_cols = {
        age_labels[0]: ["B01001_003E", "B01001_027E"],
        age_labels[1]: ["B01001_004E", "B01001_005E", "B01001_006E", "B01001_028E", "B01001_029E", "B01001_030E"],
        age_labels[2]: [
            "B01001_007E", "B01001_008E", "B01001_009E", "B01001_010E",
            "B01001_031E", "B01001_032E", "B01001_033E", "B01001_034E",
        ],
        age_labels[3]: ["B01001_011E", "B01001_012E", "B01001_035E", "B01001_036E"],
        age_labels[4]: ["B01001_013E", "B01001_014E", "B01001_037E", "B01001_038E"],
        age_labels[5]: ["B01001_015E", "B01001_016E", "B01001_039E", "B01001_040E"],
        age_labels[6]: ["B01001_017E", "B01001_018E", "B01001_019E", "B01001_041E", "B01001_042E", "B01001_043E"],
        age_labels[7]: ["B01001_020E", "B01001_021E", "B01001_022E", "B01001_044E", "B01001_045E", "B01001_046E"],
        age_labels[8]: ["B01001_023E", "B01001_024E", "B01001_047E", "B01001_048E"],
        age_labels[9]: ["B01001_025E", "B01001_049E"],
    }

    out: list[dict[str, Any]] = []
    for idx, g in df[group_col].astype(str).items():
        g = str(g)
        if not g or g == "nan":
            continue
        out.append({group_col: g, "variable": "SEX", "category": "1", "target": float(sex_male.loc[idx]), "table_id": "B01001", "universe": "all_persons"})
        out.append({group_col: g, "variable": "SEX", "category": "2", "target": float(sex_female.loc[idx]), "table_id": "B01001", "universe": "all_persons"})
        for label, cols in age_bin_cols.items():
            out.append({group_col: g, "variable": "AGEP_bin", "category": str(label), "target": float(s(cols).loc[idx]), "table_id": "B01001", "universe": "all_persons"})
    return out


def _b23025_records(df: Any, *, group_col: str, total_pop: Any) -> list[dict[str, Any]]:
    pd = _require("pandas")

    def num(col: str) -> Any:
        return pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    total_16p = num("B23025_001E")
    employed = num("B23025_004E")
    unemployed = num("B23025_005E")
    armed = num("B23025_006E")
    not_in_lf = num("B23025_007E")
    not_16p = (total_pop - total_16p).clip(lower=0.0)

    out: list[dict[str, Any]] = []
    for idx, g in df[group_col].astype(str).items():
        g = str(g)
        if not g or g == "nan":
            continue
        out.extend(
            [
                {group_col: g, "variable": "ESR_allpop", "category": "not_16p", "target": float(not_16p.loc[idx]), "table_id": "B23025", "universe": "all_persons"},
                {group_col: g, "variable": "ESR_allpop", "category": "employed", "target": float(employed.loc[idx]), "table_id": "B23025", "universe": "all_persons"},
                {group_col: g, "variable": "ESR_allpop", "category": "unemployed", "target": float(unemployed.loc[idx]), "table_id": "B23025", "universe": "all_persons"},
                {group_col: g, "variable": "ESR_allpop", "category": "armed_forces", "target": float(armed.loc[idx]), "table_id": "B23025", "universe": "all_persons"},
                {group_col: g, "variable": "ESR_allpop", "category": "not_in_labor_force", "target": float(not_in_lf.loc[idx]), "table_id": "B23025", "universe": "all_persons"},
            ]
        )
    return out


def _b15003_records(df: Any, *, group_col: str, total_pop: Any) -> list[dict[str, Any]]:
    pd = _require("pandas")

    def num(col: str) -> Any:
        return pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    def s(cols: list[str]) -> Any:
        out = None
        for c in cols:
            v = num(c)
            out = v if out is None else (out + v)
        return out if out is not None else 0.0

    total_25p = num("B15003_001E")
    not_25p = (total_pop - total_25p).clip(lower=0.0)
    lt_hs = s([f"B15003_{i:03d}E" for i in range(2, 17)])
    hs_ged = s([f"B15003_{i:03d}E" for i in range(17, 19)])
    # B15003_022E is bachelor's degree and should be grouped into bachelor_plus.
    some_assoc = s([f"B15003_{i:03d}E" for i in range(19, 22)])
    bach_plus = s([f"B15003_{i:03d}E" for i in range(22, 26)])

    out: list[dict[str, Any]] = []
    for idx, g in df[group_col].astype(str).items():
        g = str(g)
        if not g or g == "nan":
            continue
        out.extend(
            [
                {group_col: g, "variable": "SCHL_allpop", "category": "not_25p", "target": float(not_25p.loc[idx]), "table_id": "B15003", "universe": "all_persons"},
                {group_col: g, "variable": "SCHL_allpop", "category": "less_than_high_school", "target": float(lt_hs.loc[idx]), "table_id": "B15003", "universe": "all_persons"},
                {group_col: g, "variable": "SCHL_allpop", "category": "high_school_or_ged", "target": float(hs_ged.loc[idx]), "table_id": "B15003", "universe": "all_persons"},
                {group_col: g, "variable": "SCHL_allpop", "category": "some_college_or_assoc", "target": float(some_assoc.loc[idx]), "table_id": "B15003", "universe": "all_persons"},
                {group_col: g, "variable": "SCHL_allpop", "category": "bachelor_plus", "target": float(bach_plus.loc[idx]), "table_id": "B15003", "universe": "all_persons"},
            ]
        )
    return out


def _aggregate_to_puma(records: list[dict[str, Any]], tract_to_puma: dict[str, str], *, group_col: str) -> list[dict[str, Any]]:
    pd = _require("pandas")

    df = pd.DataFrame(records)
    df["puma"] = df[group_col].map(tract_to_puma)
    df = df[df["puma"].notna()].copy()
    grouped = (
        df.groupby(
            ["puma", "variable", "category", "table_id", "universe"],
            sort=False,
            as_index=False,
        )["target"]
        .sum()
    )
    grouped = grouped.rename(columns={"puma": "puma"})
    out = grouped.to_dict(orient="records")
    return out


def main() -> None:
    from src.synthpop.paths import data_root as default_data_root

    ap = argparse.ArgumentParser(prog="build_external_condition_v1_michigan")
    ap.add_argument("--data_root", default=str(default_data_root()))
    ap.add_argument("--acs_year", type=int, default=2022)
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--aggregate_to", choices=["tract", "puma"], default="puma")
    ap.add_argument("--tract_puma_csv", default="")
    ap.add_argument("--tract_zip", default="")
    ap.add_argument("--puma_zip", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    acs_year = int(args.acs_year)
    statefp = str(args.statefp).zfill(2)
    if statefp != "26":
        raise SystemExit("v1 script is Michigan-only (statefp=26).")

    acs_dir = data_root / "detroit" / "raw" / "census" / "acs" / f"acs5_{acs_year}"
    if not acs_dir.exists():
        raise SystemExit(f"ACS dir not found: {acs_dir}")

    table_paths = {
        "B01001": _default_table_path(acs_dir=acs_dir, acs_year=acs_year, table_id="B01001"),
        "B15003": _default_table_path(acs_dir=acs_dir, acs_year=acs_year, table_id="B15003"),
        "B23025": _default_table_path(acs_dir=acs_dir, acs_year=acs_year, table_id="B23025"),
    }
    for k, p in table_paths.items():
        if not p.exists():
            raise SystemExit(f"ACS table not found: {k} -> {p}")

    dfs = {k: _load_csv_gz(p) for k, p in table_paths.items()}
    for k, df in dfs.items():
        df["tract_geoid"] = _make_tract_geoid(df).astype(str)
        dfs[k] = df

    pd = _require("pandas")
    total_pop = pd.to_numeric(dfs["B01001"]["B01001_001E"], errors="coerce").fillna(0.0)

    records: list[dict[str, Any]] = []
    records.extend(_b01001_records(dfs["B01001"], group_col="tract_geoid"))
    records.extend(_b15003_records(dfs["B15003"], group_col="tract_geoid", total_pop=total_pop))
    records.extend(_b23025_records(dfs["B23025"], group_col="tract_geoid", total_pop=total_pop))

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
    default_name = f"extcond_v1_acs5_{acs_year}_{group_col}_state{statefp}_michigan.csv"
    out_path = pathlib.Path(args.out_path).expanduser().resolve() if args.out_path else (out_dir / default_name)
    if out_path.exists() and not args.overwrite:
        print(f"[skip] exists: {out_path}")
        return

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
                    "acs_year": acs_year,
                    "geo_level": group_col,
                    "schema": "external_condition_v1",
                }
            )

    meta = {
        "dataset": "Michigan ACS external condition v1",
        "schema": "external_condition_v1",
        "acs_year": acs_year,
        "statefp": statefp,
        "group_col": group_col,
        "tables": list(table_paths.keys()),
        "table_paths": {k: str(v) for k, v in table_paths.items()},
        "tract_puma_source": tract_puma_source,
        "variables": {
            "SEX": {"source_table": "B01001", "categories": ["1", "2"], "universe": "all_persons"},
            "AGEP_bin": {"source_table": "B01001", "categories": "10 coarse bins", "universe": "all_persons"},
            "SCHL_allpop": {"source_table": "B15003", "categories": ["not_25p", "less_than_high_school", "high_school_or_ged", "some_college_or_assoc", "bachelor_plus"], "universe": "all_persons"},
            "ESR_allpop": {"source_table": "B23025", "categories": ["not_16p", "employed", "unemployed", "armed_forces", "not_in_labor_force"], "universe": "all_persons"},
        },
        "created_utc": _utc_now_iso(),
        "out_path": str(out_path),
    }
    (out_path.with_suffix(out_path.suffix + ".metadata.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()
