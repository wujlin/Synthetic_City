#!/usr/bin/env python3
"""
Exp 1: Layer-1 base population reconstruction at Block Group (BG) using 2020 Decennial DHC.

Problem this answers:
- We need BG-level "base demographics" (age_group, sex, [race], [hispanic]) that match hard counts
  before Layer-2 attribute diffusion. This isolates "ecological inference on attributes" from basic
  population accounting.

Design (KISS):
- v0 implementation focuses on P12 (Sex by Age) and produces BG x (age_group, sex) counts exactly.
- v0.1 adds BG-level race totals using DHC P5 (Hispanic or Latino Origin by Race) and builds a
  BG-level (age, sex, race) count table that matches BOTH:
    1) P12 age×sex counts (exact), and
    2) P5 race totals (exact; Hispanic dimension optionally kept separate later).
  Since DHC does not provide age×sex×race cross-tabs in this dataset, we infer the joint structure
  via IPF initialized by a global seed from Michigan PUMS, then integerize with exact marginals.
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
import zipfile
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


def _write_table(df: Any, out_path: pathlib.Path) -> pathlib.Path:
    """
    Write a table with a parquet-first policy, but fall back to compressed CSV when parquet
    engines (pyarrow/fastparquet) are unavailable. This keeps the script runnable in minimal envs.
    """
    pd = _require("pandas")

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(out_path, index=False)
            return out_path
        except ImportError:
            # Fall back to CSV.GZ next to the parquet path.
            out_csv = out_path.with_suffix(".csv.gz")
            df.to_csv(out_csv, index=False, compression="gzip")
            return out_csv
    # For non-parquet paths, just write CSV (infer compression by suffix).
    compression = "gzip" if out_path.name.endswith(".gz") else None
    df.to_csv(out_path, index=False, compression=compression)
    return out_path


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


def _find_first_csv_in_zip(zip_path: pathlib.Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No .csv found inside: {zip_path}")
        return names[0]


def _resolve_pums_person_zip(*, data_root: pathlib.Path, pums_year: int, pums_period: str, statefp: str) -> pathlib.Path:
    statefp = str(statefp).zfill(2)
    state_postal_lower = "mi" if statefp == "26" else None
    raw_dir = data_root / "detroit" / "raw" / "pums" / f"pums_{pums_year}_{pums_period}"
    candidates: list[pathlib.Path] = [raw_dir / f"psam_p{statefp}.zip"]
    if state_postal_lower is not None:
        candidates.append(raw_dir / f"csv_p{state_postal_lower}i.zip")  # csv_pmi.zip
        candidates.append(raw_dir / f"csv_p{state_postal_lower}.zip")
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit(f"PUMS person zip not found. Tried: {candidates}")


def _age_to_p12_idx(age: int) -> int:
    # 23 bins (same semantics as DHC P12 / ACS B01001 23 age groups).
    if age < 0:
        age = 0
    if age <= 4:
        return 0
    if age <= 9:
        return 1
    if age <= 14:
        return 2
    if age <= 17:
        return 3
    if age <= 19:
        return 4
    if age == 20:
        return 5
    if age == 21:
        return 6
    if age <= 24:
        return 7
    if age <= 29:
        return 8
    if age <= 34:
        return 9
    if age <= 39:
        return 10
    if age <= 44:
        return 11
    if age <= 49:
        return 12
    if age <= 54:
        return 13
    if age <= 59:
        return 14
    if age <= 61:
        return 15
    if age <= 64:
        return 16
    if age <= 66:
        return 17
    if age <= 69:
        return 18
    if age <= 74:
        return 19
    if age <= 79:
        return 20
    if age <= 84:
        return 21
    return 22


def _rac1p_to_race7(code: int) -> int | None:
    """
    Map PUMS RAC1P (1..9) into DHC/P5 7-category race:
      0 white, 1 black, 2 aian, 3 asian, 4 nhpi, 5 other, 6 two_or_more
    """
    try:
        c = int(code)
    except Exception:
        return None
    if c == 1:
        return 0
    if c == 2:
        return 1
    if c in (3, 4, 5):
        return 2
    if c == 6:
        return 3
    if c == 7:
        return 4
    if c == 8:
        return 5
    if c == 9:
        return 6
    return None


def _p5_race7_totals(*, df: Any) -> Any:
    """
    Derive 7-category race totals from DHC P5 (Hispanic or Latino Origin by Race).
    We collapse Hispanic dimension:
      race_total = not_hisp_race + hisp_race
    """
    pd = _require("pandas")

    need = [
        "P5_003N",
        "P5_004N",
        "P5_005N",
        "P5_006N",
        "P5_007N",
        "P5_008N",
        "P5_009N",
        "P5_011N",
        "P5_012N",
        "P5_013N",
        "P5_014N",
        "P5_015N",
        "P5_016N",
        "P5_017N",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"P5 missing expected variables: {missing}")

    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(int)

    out = pd.DataFrame(
        {
            "race_white": df["P5_003N"] + df["P5_011N"],
            "race_black": df["P5_004N"] + df["P5_012N"],
            "race_aian": df["P5_005N"] + df["P5_013N"],
            "race_asian": df["P5_006N"] + df["P5_014N"],
            "race_nhpi": df["P5_007N"] + df["P5_015N"],
            "race_other": df["P5_008N"] + df["P5_016N"],
            "race_two_or_more": df["P5_009N"] + df["P5_017N"],
        }
    )
    return out


def _ipf_2d(*, seed_joint: Any, target_row: Any, target_col: Any, iters: int = 50, eps: float = 1e-12) -> Any:
    """
    2D IPF (raking) to match row/col marginals.
    """
    np = _require("numpy")

    x = np.asarray(seed_joint, dtype=float).copy()
    row = np.asarray(target_row, dtype=float)
    col = np.asarray(target_col, dtype=float)
    if row.size == 0 or col.size == 0:
        raise ValueError("target marginals must be non-empty")
    if x.shape != (row.size, col.size):
        raise ValueError(f"seed_joint shape {x.shape} != ({row.size},{col.size})")

    # Guard: avoid all-zeros rows/cols in seed.
    x = np.clip(x, 0.0, None)
    x = x + eps

    for _ in range(int(iters)):
        # row scaling
        rs = x.sum(axis=1)
        rf = row / np.maximum(rs, eps)
        x *= rf.reshape(-1, 1)
        # col scaling
        cs = x.sum(axis=0)
        cf = col / np.maximum(cs, eps)
        x *= cf.reshape(1, -1)
    return x


class _MaxFlow:
    def __init__(self, n: int) -> None:
        self.n = int(n)
        self.adj: list[list[int]] = [[] for _ in range(self.n)]
        self.to: list[int] = []
        self.cap: list[int] = []
        self.rev: list[int] = []

    def add_edge(self, u: int, v: int, c: int) -> None:
        if c <= 0:
            return
        u = int(u)
        v = int(v)
        c = int(c)
        fwd = len(self.to)
        bwd = fwd + 1
        self.to.append(v)
        self.cap.append(c)
        self.rev.append(bwd)
        self.to.append(u)
        self.cap.append(0)
        self.rev.append(fwd)
        self.adj[u].append(fwd)
        self.adj[v].append(bwd)

    def max_flow(self, s: int, t: int) -> int:
        from collections import deque

        flow = 0
        s = int(s)
        t = int(t)
        while True:
            parent = [-1] * self.n
            parent_edge = [-1] * self.n
            q: deque[int] = deque([s])
            parent[s] = s
            while q and parent[t] == -1:
                u = q.popleft()
                for ei in self.adj[u]:
                    if self.cap[ei] <= 0:
                        continue
                    v = self.to[ei]
                    if parent[v] != -1:
                        continue
                    parent[v] = u
                    parent_edge[v] = ei
                    q.append(v)
            if parent[t] == -1:
                break
            # augment 1 unit at a time (capacities are small)
            aug = 10**9
            v = t
            while v != s:
                ei = parent_edge[v]
                aug = min(aug, self.cap[ei])
                v = parent[v]
            v = t
            while v != s:
                ei = parent_edge[v]
                self.cap[ei] -= aug
                self.cap[self.rev[ei]] += aug
                v = parent[v]
            flow += aug
        return int(flow)


def _integerize_2d(*, x: Any, row_targets: Any, col_targets: Any) -> Any:
    """
    Integerize a non-negative matrix with fixed integer row/col sums:
    - Start with floor(x)
    - Add 1s according to residuals using a small max-flow on cells with positive fractional parts.
    """
    np = _require("numpy")

    x = np.asarray(x, dtype=float)
    row = np.asarray(row_targets, dtype=int)
    col = np.asarray(col_targets, dtype=int)
    if x.shape != (row.size, col.size):
        raise ValueError("shape mismatch in integerize")
    if row.sum() != col.sum():
        raise ValueError("row/col totals mismatch")

    base = np.floor(x).astype(int)
    # Fix any tiny negative due to numeric noise.
    base = np.clip(base, 0, None)

    row_res = row - base.sum(axis=1)
    col_res = col - base.sum(axis=0)
    if (row_res < 0).any() or (col_res < 0).any():
        # This can happen if x has large values and floor overshoots due to invalid targets; treat as error.
        raise RuntimeError("Negative residuals in integerize (check IPF output/targets).")
    need = int(row_res.sum())
    if need == 0:
        return base

    frac = x - np.floor(x)
    R, C = base.shape
    s = 0
    row0 = 1
    col0 = row0 + R
    t = col0 + C
    g = _MaxFlow(t + 1)
    for i in range(R):
        g.add_edge(s, row0 + i, int(row_res[i]))
    for j in range(C):
        g.add_edge(col0 + j, t, int(col_res[j]))

    # Prefer edges with positive fractional part; order columns by frac desc.
    for i in range(R):
        cols = list(range(C))
        cols.sort(key=lambda j: float(frac[i, j]), reverse=True)
        for j in cols:
            if frac[i, j] > 0.0:
                g.add_edge(row0 + i, col0 + j, 1)

    got = g.max_flow(s, t)
    if got != need:
        # Fallback: allow all edges.
        g = _MaxFlow(t + 1)
        for i in range(R):
            g.add_edge(s, row0 + i, int(row_res[i]))
        for j in range(C):
            g.add_edge(col0 + j, t, int(col_res[j]))
        for i in range(R):
            for j in range(C):
                g.add_edge(row0 + i, col0 + j, 1)
        got = g.max_flow(s, t)
        if got != need:
            raise RuntimeError(f"Integerize failed: need={need} got={got}")

    # Decode flow: look at reverse capacities from col->row edges.
    # In our edge structure, added edges are row->col with cap reduced; reverse edge has cap==1 for used.
    # We'll detect used by scanning adjacency from row nodes to col nodes.
    for i in range(R):
        u = row0 + i
        for ei in g.adj[u]:
            v = g.to[ei]
            if v < col0 or v >= col0 + C:
                continue
            # reverse edge capacity >0 means flow was sent on u->v
            rev_ei = g.rev[ei]
            if g.cap[rev_ei] > 0:
                j = v - col0
                base[i, j] += 1

    # Final asserts
    if not (base.sum(axis=1) == row).all():
        raise RuntimeError("Row sums mismatch after integerize.")
    if not (base.sum(axis=0) == col).all():
        raise RuntimeError("Col sums mismatch after integerize.")
    return base


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


def _compute_seed_age_sex_race(*, pums_person_zip: pathlib.Path, n_rows: int | None = None) -> tuple[Any, dict[str, Any]]:
    """
    Compute a global seed joint for (sex×age_idx) x race7 from Michigan PUMS.
    Returns:
      seed_prob: (46, 7) with sum=1
      meta: mapping info + totals
    """
    pd = _require("pandas")
    np = _require("numpy")

    member = _find_first_csv_in_zip(pums_person_zip)
    usecols = ["AGEP", "SEX", "RAC1P", "PWGTP"]
    with zipfile.ZipFile(pums_person_zip) as zf, zf.open(member) as f:
        df = pd.read_csv(f, nrows=n_rows, usecols=lambda c: c in set(usecols), low_memory=False)
    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise RuntimeError(f"PUMS missing required cols: {missing} (zip={pums_person_zip} member={member})")

    w = pd.to_numeric(df["PWGTP"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    age = pd.to_numeric(df["AGEP"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=99.0).to_numpy(dtype=int)
    sex = pd.to_numeric(df["SEX"], errors="coerce").fillna(1).astype(int).clip(lower=1, upper=2).to_numpy(dtype=int)
    rac = pd.to_numeric(df["RAC1P"], errors="coerce").fillna(-1).astype(int).to_numpy(dtype=int)

    age_idx = np.vectorize(_age_to_p12_idx, otypes=[int])(age)
    race7 = np.array([_rac1p_to_race7(int(c)) if int(c) >= 0 else None for c in rac], dtype=object)

    mask = (w > 0) & (race7 != np.array(None))
    # race7 is object array; build numeric
    race7_num = np.full(rac.shape[0], -1, dtype=int)
    for i, v in enumerate(race7.tolist()):
        if v is None:
            continue
        race7_num[i] = int(v)
    mask = (w > 0) & (race7_num >= 0)

    w = w[mask]
    sex = sex[mask]
    age_idx = age_idx[mask]
    race7_num = race7_num[mask]

    seed = np.zeros((46, 7), dtype=float)
    # rows: male age_idx (0..22) then female (23..45)
    row_idx = (sex - 1) * 23 + age_idx
    for r, c, ww in zip(row_idx.tolist(), race7_num.tolist(), w.tolist()):
        seed[int(r), int(c)] += float(ww)
    tot = float(seed.sum())
    if tot <= 0:
        raise RuntimeError("PUMS seed has zero total weight (unexpected).")
    seed_prob = seed / tot
    meta = {
        "pums_person_zip": str(pums_person_zip),
        "member": str(member),
        "n_rows_used": int(len(w)),
        "race7_labels": ["white", "black", "aian", "asian", "nhpi", "other", "two_or_more"],
        "total_weight": float(tot),
    }
    return seed_prob, meta


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
    p.add_argument(
        "--dhc_bg_path",
        default=None,
        help="Path to combined DHC BG file that includes P12 (+ optional P5). If set, overrides --p12_path.",
    )
    p.add_argument("--include_race", action="store_true", help="Also build BG×age×sex×race counts using P5 + IPF.")
    p.add_argument("--pums_person_zip", default=None, help="PUMS person zip for seed joint (recommended for --include_race).")
    p.add_argument("--pums_year", type=int, default=2022)
    p.add_argument("--pums_period", default="5-Year")
    p.add_argument("--ipf_iters", type=int, default=50)
    p.add_argument("--max_bgs", type=int, default=None, help="Optional cap of BG rows for a quick smoke run.")
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
    if args.dhc_bg_path:
        p12_path = pathlib.Path(args.dhc_bg_path).expanduser().resolve()
        if not p12_path.exists():
            raise SystemExit(f"DHC BG file not found: {p12_path}")
    elif not args.p12_path:
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

    if args.max_bgs is not None:
        p12 = p12.head(int(args.max_bgs)).copy()

    if not args.include_race:
        counts_long = build_bg_age_sex_counts(p12=p12)

        # Write counts (parquet preferred; may fall back to csv.gz if parquet engine missing).
        out_counts = _write_table(counts_long, out_dir / "base_pop_bg_age_sex_counts.parquet")
        print(f"[ok] wrote: {out_counts}")

        internal_validation = {"p12_exactness": _internal_validate_p12(counts_long=counts_long, p12=p12)}
        _write_json(out_dir / "internal_validation.json", internal_validation)
    else:
        np = _require("numpy")

        # Ensure required columns exist.
        var_map = _p12_var_map()
        for v in var_map:
            if v not in p12.columns:
                raise SystemExit(f"Missing P12 var {v} in DHC file (need full P12_003N..P12_049N).")
        race_totals_df = _p5_race7_totals(df=p12)

        # Build BG GEOID and numeric arrays.
        p12 = p12.copy()
        p12["bg_geoid"] = (
            p12["state"].astype(str).str.zfill(2)
            + p12["county"].astype(str).str.zfill(3)
            + p12["tract"].astype(str).str.zfill(6)
            + p12["block group"].astype(str).str.zfill(1)
        )
        # row targets: 46 = male(23) + female(23)
        male_vars = [f"P12_{(3 + k):03d}N" for k in range(23)]
        fem_vars = [f"P12_{(27 + k):03d}N" for k in range(23)]
        for c in male_vars + fem_vars:
            p12[c] = pd.to_numeric(p12[c], errors="coerce").fillna(0.0).astype(int)
        rows46 = np.concatenate([p12[male_vars].to_numpy(dtype=int), p12[fem_vars].to_numpy(dtype=int)], axis=1)
        race7 = race_totals_df.to_numpy(dtype=int)  # (N,7)

        # Seed joint from PUMS (recommended).
        seed_meta: dict[str, Any] = {"used": False}
        seed_prob = None
        try:
            if args.pums_person_zip:
                pums_zip = pathlib.Path(args.pums_person_zip).expanduser().resolve()
            else:
                pums_zip = _resolve_pums_person_zip(
                    data_root=data_root,
                    pums_year=int(args.pums_year),
                    pums_period=str(args.pums_period),
                    statefp=str(args.statefp),
                )
            seed_prob, seed_meta2 = _compute_seed_age_sex_race(pums_person_zip=pums_zip, n_rows=None)
            seed_meta = {"used": True, "meta": seed_meta2}
        except Exception as e:
            seed_meta = {"used": False, "error": str(e), "fallback": "independence_seed_from_DHC_marginals"}

        # Independence fallback seed if PUMS is unavailable.
        if seed_prob is None:
            row_global = rows46.sum(axis=0).astype(float)
            col_global = race7.sum(axis=0).astype(float)
            if row_global.sum() <= 0 or col_global.sum() <= 0:
                raise SystemExit("Cannot build fallback seed: DHC totals are zero.")
            seed_prob = (row_global / row_global.sum()).reshape(-1, 1) * (col_global / col_global.sum()).reshape(1, -1)
            seed_prob = seed_prob / float(seed_prob.sum())

        # Build per-BG tables.
        race_labels = ["white", "black", "aian", "asian", "nhpi", "other", "two_or_more"]
        bg_ids = p12["bg_geoid"].astype(str).to_numpy()

        bg_col: list[str] = []
        sex_col: list[int] = []
        age_col: list[int] = []
        race_col: list[str] = []
        cnt_col: list[int] = []

        max_p12_abs = 0
        max_race_abs = 0
        worst_p12 = None
        worst_race = None
        n_bg = int(bg_ids.shape[0])

        for i in range(n_bg):
            bg = str(bg_ids[i])
            row_t = rows46[i, :].astype(int)
            col_t = race7[i, :].astype(int)
            total = int(row_t.sum())
            if total <= 0:
                continue
            if int(col_t.sum()) != total:
                # DHC should be consistent; but guard and continue.
                continue

            seed_joint = seed_prob * float(total)
            x = _ipf_2d(seed_joint=seed_joint, target_row=row_t, target_col=col_t, iters=int(args.ipf_iters))
            x_int = _integerize_2d(x=x, row_targets=row_t, col_targets=col_t)

            # internal diffs (should be 0)
            d_p12 = int(np.abs(x_int.sum(axis=1) - row_t).max())
            d_r = int(np.abs(x_int.sum(axis=0) - col_t).max())
            if d_p12 > max_p12_abs:
                max_p12_abs = d_p12
                worst_p12 = {"bg_geoid": bg}
            if d_r > max_race_abs:
                max_race_abs = d_r
                worst_race = {"bg_geoid": bg}

            nz = np.nonzero(x_int)
            rr = nz[0].tolist()
            cc = nz[1].tolist()
            vv = x_int[nz].astype(int).tolist()
            for r, c, v in zip(rr, cc, vv):
                if v <= 0:
                    continue
                sex = 1 if int(r) < 23 else 2
                age_idx = int(r) % 23
                bg_col.append(bg)
                sex_col.append(int(sex))
                age_col.append(int(age_idx))
                race_col.append(str(race_labels[int(c)]))
                cnt_col.append(int(v))

        out = pd.DataFrame({"bg_geoid": bg_col, "sex": sex_col, "age_idx": age_col, "race": race_col, "count": cnt_col})
        out = out[out["count"] > 0].reset_index(drop=True)

        out_counts = _write_table(out, out_dir / "base_pop_bg_age_sex_race_counts.parquet")
        print(f"[ok] wrote: {out_counts}")

        _write_json(
            out_dir / "internal_validation.json",
            {
                "p12_exactness": {"max_abs_diff": int(max_p12_abs), "worst": worst_p12},
                "race_exactness": {"max_abs_diff": int(max_race_abs), "worst": worst_race},
                "seed": seed_meta,
                "race_labels": race_labels,
                "note": "Counts match DHC P12 (age×sex) and P5-derived race totals exactly; joint is inferred via IPF seed + integerization.",
            },
        )

    if args.expand_microdata:
        if args.include_race:
            raise SystemExit("--expand_microdata is not supported with --include_race in v0.1; use counts and sample later.")
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
        out_micro = _write_table(micro, out_dir / "base_pop_bg_age_sex_microdata.parquet")
        print(f"[ok] wrote: {out_micro} (n={int(micro.shape[0])})")


if __name__ == "__main__":
    main()
