#!/usr/bin/env python3
"""
Exp 1: Layer-1 base population reconstruction at Block Group (BG) using 2020 Decennial DHC.

Problem this answers:
- We need BG-level "base demographics" (age_group, sex, [race], [hispanic]) that match hard counts
  before Layer-2 attribute diffusion. This isolates "ecological inference on attributes" from basic
  population accounting.

Design (KISS):
- v0 implementation focuses on P12 (Sex by Age) and produces BG x (age_group, sex) counts exactly.
- Optional: download DHC tables via Census API (mode=fetch). Many environments will prefer manual
  download; mode=build works from local parquet/csv.
- Output defaults to a compact counts table; microdata expansion is optional (can be very large).

Planned extensions (explicitly not in v0):
- Integrate P12A-P12G (race-specific sex-by-age) + P5 (Hispanic by race) to build (age, sex, race, hisp).
- Integer optimization path (gurobi) for exact multi-table consistency.

Outputs:
  outputs/<run_id>/
    base_pop_bg_age_sex_counts.parquet   (ignored by git by default .gitignore; keep on workstation)
    internal_validation.json             (small, can be committed)
    run.metadata.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
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
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fetch_json(url: str, *, timeout_s: int = 60) -> Any:
    import urllib.parse
    import urllib.request

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to parse JSON from {url}") from e


def _dhc_api_base(year: int = 2020) -> str:
    if int(year) != 2020:
        raise ValueError("Only DHC 2020 is supported in v0.")
    return "https://api.census.gov/data/2020/dec/dhc"


def _fetch_counties(*, api_base: str, statefp: str, api_key: str | None) -> list[str]:
    statefp = str(statefp).zfill(2)
    key_q = f"&key={api_key}" if api_key else ""
    url = f"{api_base}?get=NAME&for=county:*&in=state:{statefp}{key_q}"
    data = _fetch_json(url)
    cols = data[0]
    rows = data[1:]
    if "county" not in cols:
        raise RuntimeError("Unexpected counties response: missing 'county' column.")
    idx = cols.index("county")
    return sorted({str(r[idx]).zfill(3) for r in rows})


def fetch_dhc_bg(
    *,
    out_path: pathlib.Path,
    table: str,
    statefp: str,
    year: int = 2020,
    api_key: str | None = None,
    counties: list[str] | None = None,
    timeout_s: int = 60,
) -> None:
    """
    Fetch a DHC group table at BG for a state (optionally subset by counties).

    We use get=group(P12) style, which returns all variables in the group in one call.
    This still needs county iteration for reliability/size.
    """
    pd = _require("pandas")

    api_base = _dhc_api_base(year)
    statefp = str(statefp).zfill(2)
    key_q = f"&key={api_key}" if api_key else ""
    if counties is None:
        counties = _fetch_counties(api_base=api_base, statefp=statefp, api_key=api_key)

    frames = []
    for cty in counties:
        cty = str(cty).zfill(3)
        url = (
            f"{api_base}?get=group({table})&for=block%20group:*&in=state:{statefp}%20county:{cty}{key_q}"
        )
        data = _fetch_json(url, timeout_s=timeout_s)
        cols = data[0]
        rows = data[1:]
        frames.append(pd.DataFrame(rows, columns=cols))

    df = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)


def _p12_age_bins() -> list[str]:
    # 23 bins (same binning used in B01001-style 23 age groups).
    return [
        "Under 5 years",
        "5 to 9 years",
        "10 to 14 years",
        "15 to 17 years",
        "18 and 19 years",
        "20 years",
        "21 years",
        "22 to 24 years",
        "25 to 29 years",
        "30 to 34 years",
        "35 to 39 years",
        "40 to 44 years",
        "45 to 49 years",
        "50 to 54 years",
        "55 to 59 years",
        "60 and 61 years",
        "62 to 64 years",
        "65 and 66 years",
        "67 to 69 years",
        "70 to 74 years",
        "75 to 79 years",
        "80 to 84 years",
        "85 years and over",
    ]


def _p12_var_map() -> dict[str, tuple[int, int]]:
    """
    Map P12 variables to (sex, age_idx) for the 23 age bins.

    P12 layout (49 vars, *_N suffix):
      P12_001N total
      P12_002N male total
      P12_003N..P12_025N male by age (23)
      P12_026N female total
      P12_027N..P12_049N female by age (23)
    """
    m: dict[str, tuple[int, int]] = {}
    # male: sex=1
    for k in range(23):
        var = f"P12_{(3 + k):03d}N"
        m[var] = (1, k)
    # female: sex=2
    for k in range(23):
        var = f"P12_{(27 + k):03d}N"
        m[var] = (2, k)
    return m


def build_bg_age_sex_counts(*, p12: Any) -> Any:
    """
    Convert a fetched/loaded P12 frame into a long counts table:
      (bg_geoid, sex, age_idx, count)
    """
    pd = _require("pandas")
    np = _require("numpy")

    if not isinstance(p12, pd.DataFrame):
        raise TypeError("p12 must be a pandas DataFrame")

    cols = set(p12.columns.astype(str).tolist())

    def _bg_geoid_from_frame(df: "Any") -> "Any":
        # Preferred: explicit bg_geoid.
        if "bg_geoid" in cols:
            return df["bg_geoid"].astype(str)
        # Common: GEOID / GEO_ID.
        for c in ("GEOID", "GEOID20", "geoid"):
            if c in cols:
                s = df[c].astype(str).str.replace(r"[^0-9]", "", regex=True)
                return s.str[-12:].astype(str)
        if "GEO_ID" in cols:
            s = df["GEO_ID"].astype(str)
            s = s.str.replace("US", "", regex=False).str.replace(r"[^0-9]", "", regex=True)
            return s.str[-12:].astype(str)

        # Census API schema.
        if {"state", "county", "tract", "block group"} <= cols:
            state = df["state"].astype(str).str.zfill(2)
            county = df["county"].astype(str).str.zfill(3)
            tract = df["tract"].astype(str).str.zfill(6)
            bg = df["block group"].astype(str).str.zfill(1)
            return (state + county + tract + bg).astype(str)

        # TIGER-like schema (rare in our pipeline but common in exports).
        if {"STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE"} <= cols:
            state = df["STATEFP"].astype(str).str.zfill(2)
            county = df["COUNTYFP"].astype(str).str.zfill(3)
            tract = df["TRACTCE"].astype(str).str.zfill(6)
            bg = df["BLKGRPCE"].astype(str).str.zfill(1)
            return (state + county + tract + bg).astype(str)

        raise ValueError(
            "Cannot derive bg_geoid. Provide columns (bg_geoid) or (GEOID/GEO_ID) or Census API geo columns."
        )

    var_map = _p12_var_map()
    vars_present = [v for v in var_map.keys() if v in p12.columns]
    if len(vars_present) != len(var_map):
        missing_vars = sorted(set(var_map.keys()) - set(vars_present))
        raise ValueError(f"P12 missing expected variables (first 10): {missing_vars[:10]}")

    df = p12.copy()
    for v in vars_present:
        df[v] = pd.to_numeric(df[v], errors="coerce").fillna(0.0).astype(int)

    # Build BG GEOID (12 digits): state(2)+county(3)+tract(6)+bg(1)
    df["bg_geoid"] = _bg_geoid_from_frame(df)

    rows = []
    for v in vars_present:
        sex, age_idx = var_map[v]
        sub = df[["bg_geoid", v]].copy()
        sub["sex"] = sex
        sub["age_idx"] = age_idx
        sub.rename(columns={v: "count"}, inplace=True)
        rows.append(sub)

    out = pd.concat(rows, ignore_index=True)
    out["count"] = pd.to_numeric(out["count"], errors="coerce").fillna(0.0).astype(int)
    out = out[out["count"] > 0].reset_index(drop=True)

    # Sanity: non-negative, finite.
    if (out["count"] < 0).any():
        raise RuntimeError("Negative counts found in P12-derived table (unexpected).")

    # Add human-readable age_bin label for convenience (small).
    bins = _p12_age_bins()
    out["age_bin"] = out["age_idx"].map(lambda i: bins[int(i)] if 0 <= int(i) < len(bins) else None)
    return out


def _internal_validate_p12(*, counts_long: Any, p12: Any) -> dict[str, Any]:
    pd = _require("pandas")
    np = _require("numpy")

    var_map = _p12_var_map()
    # reconstruct P12 vars from long counts
    wide = counts_long.pivot_table(
        index="bg_geoid", columns=["sex", "age_idx"], values="count", aggfunc="sum", fill_value=0
    )
    # compute max absolute deviation to original P12 for a quick exactness check
    # Note: since we kept integers directly from P12, this should be exact.
    # We validate totals for a small sample to avoid heavy wide joins.
    p12_df = p12.copy()
    p12_df["bg_geoid"] = (
        p12_df["state"].astype(str).str.zfill(2)
        + p12_df["county"].astype(str).str.zfill(3)
        + p12_df["tract"].astype(str).str.zfill(6)
        + p12_df["block group"].astype(str).str.zfill(1)
    )
    for v in var_map:
        p12_df[v] = pd.to_numeric(p12_df[v], errors="coerce").fillna(0.0).astype(int)

    sample_n = min(200, int(p12_df.shape[0]))
    sample = p12_df.sample(n=sample_n, random_state=0) if sample_n > 0 else p12_df

    max_abs = 0
    worst = None
    for _, r in sample.iterrows():
        bg = str(r["bg_geoid"])
        for v, (sex, age_idx) in var_map.items():
            ref = int(r[v])
            got = int(wide.loc[bg, (sex, age_idx)]) if bg in wide.index else 0
            d = abs(got - ref)
            if d > max_abs:
                max_abs = d
                worst = {"bg_geoid": bg, "var": v, "ref": ref, "got": got}

    return {
        "sample_n": int(sample_n),
        "max_abs_diff": int(max_abs),
        "worst": worst,
        "note": "Validation is sample-based; exactness is expected if input P12 is complete.",
    }


def main() -> None:
    pd = _require("pandas")

    from src.synthpop.pipeline.detroit_v0 import make_run_id
    from src.synthpop.paths import data_root as default_data_root

    p = argparse.ArgumentParser(prog="exp1_base_population")
    p.add_argument("--mode", choices=["fetch", "build"], default="build")
    p.add_argument("--data_root", default=str(default_data_root()))
    p.add_argument("--out_dir", default=None, help="Default: outputs/<run_id> under repo.")

    # fetch options
    p.add_argument("--dhc_year", type=int, default=2020)
    p.add_argument("--statefp", default="26")
    p.add_argument("--api_key", default=None, help="Optional Census API key.")
    p.add_argument("--tables", default="P12", help='Comma-separated, e.g. "P12,P5" (v0 uses P12 only).')
    p.add_argument("--timeout_s", type=int, default=60)
    p.add_argument("--out_raw_dir", default=None, help="Where to store fetched DHC files (default under data_root).")

    # build options
    p.add_argument("--p12_path", default=None, help="Path to DHC P12 file (parquet/csv) for build mode.")
    p.add_argument("--counts_only", action="store_true", help="Only write BG×age×sex counts (default).")
    p.add_argument("--expand_microdata", action="store_true", help="Expand to per-person rows (can be huge).")
    p.add_argument("--max_persons", type=int, default=None, help="Optional cap when expanding microdata (debug).")
    args = p.parse_args()

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    out_dir = (
        pathlib.Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (_REPO_ROOT / "outputs" / make_run_id(prefix="exp1_base_population"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        out_dir / "run.metadata.json",
        {
            "created_utc": _utc_now_iso(),
            "argv": sys.argv,
            "script": pathlib.Path(__file__).name,
            "mode": args.mode,
            "env": {"RAW_ROOT": os.environ.get("RAW_ROOT"), "SYNTHCITY_DATA_ROOT": os.environ.get("SYNTHCITY_DATA_ROOT")},
            "args": vars(args),
        },
    )

    if args.mode == "fetch":
        out_raw_dir = (
            pathlib.Path(args.out_raw_dir).expanduser().resolve()
            if args.out_raw_dir
            else (data_root / "detroit" / "raw" / "census" / "dhc_2020")
        )
        out_raw_dir.mkdir(parents=True, exist_ok=True)
        tables = [t.strip() for t in str(args.tables).split(",") if t.strip()]
        for t in tables:
            out_path = out_raw_dir / f"dhc_{int(args.dhc_year)}_{t}_bg_state{str(args.statefp).zfill(2)}.parquet"
            print(f"[info] fetching {t} -> {out_path}")
            fetch_dhc_bg(
                out_path=out_path,
                table=t,
                statefp=str(args.statefp),
                year=int(args.dhc_year),
                api_key=(str(args.api_key) if args.api_key else None),
                counties=None,
                timeout_s=int(args.timeout_s),
            )
        print(f"[ok] fetched tables into: {out_raw_dir}")
        return

    # --- build mode ---
    if not args.p12_path:
        # default location produced by mode=fetch
        default_p12 = data_root / "detroit" / "raw" / "census" / "dhc_2020" / f"dhc_2020_P12_bg_state{str(args.statefp).zfill(2)}.parquet"
        if default_p12.exists():
            p12_path = default_p12
        else:
            raise SystemExit("Missing --p12_path and default fetched P12 file not found.")
    else:
        p12_path = pathlib.Path(args.p12_path).expanduser().resolve()
        if not p12_path.exists():
            raise SystemExit(f"P12 file not found: {p12_path}")

    print(f"[info] loading P12: {p12_path}")
    if p12_path.suffix.lower() == ".parquet":
        p12 = pd.read_parquet(p12_path)
    else:
        p12 = pd.read_csv(p12_path, low_memory=False)

    counts_long = build_bg_age_sex_counts(p12=p12)

    # Write counts (parquet preferred; but parquet is ignored by git by default).
    out_counts = out_dir / "base_pop_bg_age_sex_counts.parquet"
    counts_long.to_parquet(out_counts, index=False)
    print(f"[ok] wrote: {out_counts}")

    internal_validation = {"p12_exactness": _internal_validate_p12(counts_long=counts_long, p12=p12)}
    _write_json(out_dir / "internal_validation.json", internal_validation)

    if args.expand_microdata:
        np = _require("numpy")
        # Expand counts into per-person rows. This can be very large for a full state.
        rows = []
        total = 0
        cap = int(args.max_persons) if args.max_persons is not None else None
        for r in counts_long.itertuples(index=False):
            n = int(getattr(r, "count"))
            if n <= 0:
                continue
            if cap is not None and total >= cap:
                break
            take = n if cap is None else min(n, cap - total)
            total += take
            rows.append(
                pd.DataFrame(
                    {
                        "bg_geoid": [getattr(r, "bg_geoid")] * take,
                        "sex": [int(getattr(r, "sex"))] * take,
                        "age_idx": [int(getattr(r, "age_idx"))] * take,
                    }
                )
            )
        micro = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["bg_geoid", "sex", "age_idx"])
        out_micro = out_dir / "base_pop_bg_age_sex_microdata.parquet"
        micro.to_parquet(out_micro, index=False)
        print(f"[ok] wrote: {out_micro} (n={int(micro.shape[0])})")


if __name__ == "__main__":
    main()
