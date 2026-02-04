#!/usr/bin/env python3
from __future__ import annotations

"""
PoC: ACS-supervised joint-to-joint diffusion on B19037 (Age of Householder × Household Income).

Motivation (PI-aligned):
- For weak-copula settings (e.g., age×sex), sample-level training with pseudo-individuals can be ill-posed.
- Here we directly learn a distribution-to-distribution map:
    cond = (p_age, p_income)  ->  p_joint(age, income)
  where each training sample is one tract-level joint distribution.

Model:
- x_model = joint_tabddpm_logp
  Gaussian DDPM on a 1D vector in R^K where K = n_age_bins * n_income_bins.
  We represent a distribution as log-probabilities:
    x0 = log(p_joint + eps)
  and decode via softmax.

Baselines:
- Independence: outer(p_age, p_income)
- IPF(train-seed): IPF using a seed joint built from TRAIN tracts only

External validation (optional):
- Aggregate tract predictions to PUMA and compare vs PUMS-derived joint distributions.
  Note: B19037 is household-based. We approximate with PUMS households:
    - householder age from PUMS person (RELP==0, AGEP)
    - household income from PUMS housing (HINCP)
    - household weight WGTP

Outputs (commit-friendly):
  outputs/<run_id>/
    run_summary.json
    metrics/internal_acs_holdout.json
    metrics/baselines_internal.json
    metrics/external_pums_by_puma.json (if --data_root provided)
    metrics/acs_pums_baseline_gap.json (if --data_root provided)
    metrics/ablation_summary.json
"""

import argparse
import json
import math
import pathlib
import re
import sys
from typing import Any


# Allow running as a plain script without installing the repo.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency: {pkg}. Install it in your conda env.\n"
            "Recommended: conda install -c conda-forge pandas numpy geopandas pyproj shapely\n"
            "and install torch (CUDA if available)."
        ) from e


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _parse_puma_blocks(spec: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    for part in str(spec).split(";"):
        part = part.strip()
        if not part:
            continue
        items = [s.strip() for s in part.split(",") if s.strip()]
        if len(items) < 2:
            raise ValueError(f"Each block must contain >=2 PUMAs; got: {part!r}")
        blocks.append([str(_normalize_puma(x) or x) for x in items])
    if not blocks:
        raise ValueError("Empty --puma_blocks spec.")
    return blocks


def _tvd(p: Any, q: Any) -> float:
    np = _require("numpy")
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    return 0.5 * float(np.abs(p - q).sum())


def _ipf_2d(*, seed_joint: Any, target_row: Any, target_col: Any, max_iter: int = 200, tol: float = 1e-10) -> Any:
    """
    IPF for a 2D table on probabilities (not counts).
    Returns a flattened joint distribution matching the target row/col marginals.
    """
    np = _require("numpy")
    seed = np.asarray(seed_joint, dtype=float).reshape(-1)
    r = np.asarray(target_row, dtype=float).reshape(-1)
    c = np.asarray(target_col, dtype=float).reshape(-1)
    if float(r.sum()) <= 0 or float(c.sum()) <= 0:
        raise ValueError("target marginals must be non-empty")
    r = np.clip(r, 0.0, None)
    c = np.clip(c, 0.0, None)
    r = r / float(r.sum())
    c = c / float(c.sum())

    n_row = int(r.size)
    n_col = int(c.size)
    if seed.size != n_row * n_col:
        raise ValueError(f"seed_joint size mismatch: seed={seed.size}, expected={n_row*n_col}")

    table = seed.reshape(n_row, n_col).astype(float)
    table = np.clip(table, 0.0, None)
    s = float(table.sum())
    if s <= 0:
        table[:] = 1.0 / float(n_row * n_col)
    else:
        table /= s

    for _ in range(int(max_iter)):
        # Row scaling.
        row_sum = table.sum(axis=1)
        row_factor = np.zeros_like(row_sum)
        m = row_sum > 0
        row_factor[m] = r[m] / row_sum[m]
        table = table * row_factor.reshape(-1, 1)
        if bool((r <= 0).any()):
            table[r <= 0, :] = 0.0

        # Column scaling.
        col_sum = table.sum(axis=0)
        col_factor = np.zeros_like(col_sum)
        m = col_sum > 0
        col_factor[m] = c[m] / col_sum[m]
        table = table * col_factor.reshape(1, -1)
        if bool((c <= 0).any()):
            table[:, c <= 0] = 0.0

        if float(np.max(np.abs(table.sum(axis=1) - r))) < float(tol) and float(np.max(np.abs(table.sum(axis=0) - c))) < float(tol):
            break

    out = table.reshape(-1)
    out = np.clip(out, 0.0, None)
    out = out / (float(out.sum()) if float(out.sum()) > 0 else 1.0)
    return out


def _marginals_from_joint(*, p_joint: Any, n_row: int, n_col: int) -> tuple[Any, Any]:
    np = _require("numpy")
    p = np.asarray(p_joint, dtype=float).reshape(-1)
    if p.size != int(n_row) * int(n_col):
        raise ValueError("p_joint size mismatch")
    tab = p.reshape(int(n_row), int(n_col))
    row = tab.sum(axis=1).astype(float)
    col = tab.sum(axis=0).astype(float)
    row = row / (float(row.sum()) if float(row.sum()) > 0 else 1.0)
    col = col / (float(col.sum()) if float(col.sum()) > 0 else 1.0)
    return row, col


def _outer_from_marginals(*, p_row: Any, p_col: Any) -> Any:
    np = _require("numpy")
    r = np.asarray(p_row, dtype=float).reshape(-1)
    c = np.asarray(p_col, dtype=float).reshape(-1)
    r = np.clip(r, 0.0, None)
    c = np.clip(c, 0.0, None)
    r = r / (float(r.sum()) if float(r.sum()) > 0 else 1.0)
    c = c / (float(c.sum()) if float(c.sum()) > 0 else 1.0)
    tab = r.reshape(-1, 1) * c.reshape(1, -1)
    out = tab.reshape(-1)
    out = out / (float(out.sum()) if float(out.sum()) > 0 else 1.0)
    return out


def _load_buildings_for_mapping(path: pathlib.Path) -> dict[str, str]:
    pd = _require("pandas")
    df = pd.read_csv(path, low_memory=False)
    needed = ["tract_geoid", "puma"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"buildings_csv missing columns: {missing}")
    df["tract_geoid"] = df["tract_geoid"].astype(str)
    df["puma"] = df["puma"].astype(str)
    tract_to_puma = df.groupby("tract_geoid", sort=False)["puma"].first().to_dict()
    if not tract_to_puma:
        raise SystemExit("Empty tract_to_puma mapping from buildings_csv.")
    return {str(k): str(v) for k, v in tract_to_puma.items()}


def _pick_zip_csv_member_with_cols(*, zip_path: pathlib.Path, required_cols: list[str]) -> tuple[str, list[str]]:
    """
    Pick a CSV member inside `zip_path` that contains all `required_cols`.
    This avoids brittle assumptions like `sorted(members)[0]`.
    """
    import zipfile

    def _norm(c: Any) -> str:
        return str(c).lstrip("\ufeff").strip().upper()

    required = {_norm(c) for c in required_cols}

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise SystemExit(f"No CSV members found inside: {zip_path}")

        members = sorted(members, key=lambda m: int(zf.getinfo(m).file_size), reverse=True)
        scanned: list[tuple[str, list[str]]] = []
        for m in members:
            cols: list[str] = []
            with zf.open(m) as f:
                try:
                    raw = f.readline(1024 * 1024)
                except Exception:
                    raw = b""

            if raw:
                try:
                    line = raw.decode("utf-8-sig", errors="replace").strip()
                except Exception:
                    line = ""

                if line:
                    # Heuristic delimiter detection for robust header parsing.
                    delims = [",", "\t", "|", ";"]
                    delim = max(delims, key=lambda d: line.count(d))
                    if line.count(delim) >= 1:
                        cols = [c.strip().strip('"') for c in line.split(delim)]

            if not cols:
                # Fallback to pandas, in case header parsing above fails.
                pd = _require("pandas")
                with zf.open(m) as f:
                    header = pd.read_csv(f, nrows=0, low_memory=False)
                cols = [str(c) for c in list(header.columns)]

            scanned.append((m, cols))
            cols_norm = {_norm(c) for c in cols}
            if required.issubset(cols_norm):
                return m, cols

    preview = "; ".join([f"{m}: {cols[:12]}" for (m, cols) in scanned[:5]])
    raise SystemExit(f"Cannot find a CSV member with columns={required_cols} inside: {zip_path}. Scanned: {preview}")


def _load_pums_households(*, data_root: pathlib.Path, pums_year: int, pums_period: str, statefp: str, pumas: set[str], n_rows: int) -> Any:
    """
    Load PUMS housing file for household-level validation.
    """
    import zipfile

    pd = _require("pandas")

    statefp = str(statefp).zfill(2)
    state_postal_lower = "mi" if statefp == "26" else None
    if state_postal_lower is None:
        raise SystemExit(f"Unsupported --statefp={statefp}. v0 only supports MI (26).")

    raw_dir = data_root / "detroit" / "raw" / "pums" / f"pums_{pums_year}_{pums_period}"
    candidates = [
        raw_dir / f"psam_h{statefp}.zip",
        raw_dir / f"csv_h{state_postal_lower}.zip",
    ]
    zip_path = next((p for p in candidates if p.exists()), candidates[0])
    if not zip_path.exists():
        raise SystemExit(f"PUMS housing zip not found. Tried: {candidates[0]} and {candidates[1]}")

    cols = ["SERIALNO", "PUMA", "HINCP", "WGTP"]
    member, _ = _pick_zip_csv_member_with_cols(zip_path=zip_path, required_cols=cols)
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as f:
        df = pd.read_csv(f, nrows=int(n_rows), low_memory=False)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"PUMS housing file missing columns: {missing}. zip={zip_path} member={member} cols={list(df.columns)[:30]}")

    df = df[cols].copy()
    df["SERIALNO"] = df["SERIALNO"].astype(str)
    df["PUMA"] = pd.to_numeric(df["PUMA"], errors="coerce")
    df["HINCP"] = pd.to_numeric(df["HINCP"], errors="coerce")
    df["WGTP"] = pd.to_numeric(df["WGTP"], errors="coerce")
    df = df.dropna().copy()
    df["PUMA_STR"] = df["PUMA"].astype(int).astype(str)
    df = df[df["PUMA_STR"].isin(set(map(str, pumas)))].copy()
    if df.empty:
        raise SystemExit("After filtering to study PUMAs, no PUMS housing rows remain.")
    return df.reset_index(drop=True)


def _load_pums_householder_age(*, data_root: pathlib.Path, pums_year: int, pums_period: str, statefp: str, n_rows: int) -> Any:
    """
    Load PUMS person file and extract (SERIALNO -> householder AGEP).
    """
    import zipfile

    pd = _require("pandas")

    statefp = str(statefp).zfill(2)
    state_postal_lower = "mi" if statefp == "26" else None
    if state_postal_lower is None:
        raise SystemExit(f"Unsupported --statefp={statefp}. v0 only supports MI (26).")

    raw_dir = data_root / "detroit" / "raw" / "pums" / f"pums_{pums_year}_{pums_period}"
    candidates = [
        raw_dir / f"psam_p{statefp}.zip",
        raw_dir / f"csv_p{state_postal_lower}.zip",
    ]
    def _norm(c: Any) -> str:
        return str(c).lstrip("\ufeff").strip().upper()

    def _try_load(zp: pathlib.Path, required_cols: list[str]) -> tuple[pathlib.Path, str, Any] | None:
        member, _ = _pick_zip_csv_member_with_cols(zip_path=zp, required_cols=required_cols)
        with zipfile.ZipFile(zp) as zf, zf.open(member) as f:
            df0 = pd.read_csv(f, nrows=int(n_rows), low_memory=False)
        df0 = df0.rename(columns={c: _norm(c) for c in df0.columns})
        missing = [c for c in required_cols if _norm(c) not in df0.columns]
        if missing:
            raise SystemExit(f"PUMS person file missing columns: {missing}. zip={zp} member={member} cols={list(df0.columns)[:30]}")
        return zp, member, df0

    errors: list[str] = []

    # Prefer RELP==0 for householder (best effort).
    cols_relp = ["SERIALNO", "RELP", "AGEP"]
    for zp in candidates:
        if not zp.exists():
            continue
        try:
            got = _try_load(zp, cols_relp)
        except SystemExit as e:
            errors.append(str(e))
            continue
        assert got is not None
        zip_path, member, df = got
        df = df[[_norm(c) for c in cols_relp]].copy()
        df["SERIALNO"] = df["SERIALNO"].astype(str)
        df["RELP"] = pd.to_numeric(df["RELP"], errors="coerce")
        df["AGEP"] = pd.to_numeric(df["AGEP"], errors="coerce")
        df = df.dropna().copy()
        df = df[df["RELP"].astype(int) == 0].copy()
        if df.empty:
            raise SystemExit(f"No RELP==0 (householder) rows found in PUMS person file. zip={zip_path} member={member}")
        out = df.groupby("SERIALNO", sort=False)["AGEP"].first().reset_index()
        out.attrs["householder_method"] = "RELP==0"
        out.attrs["source_zip"] = str(zip_path)
        out.attrs["source_member"] = str(member)
        return out

    # Fallback: SPORDER==1 (approximate householder) if RELP is unavailable.
    cols_sporder = ["SERIALNO", "SPORDER", "AGEP"]
    for zp in candidates:
        if not zp.exists():
            continue
        try:
            got = _try_load(zp, cols_sporder)
        except SystemExit as e:
            errors.append(str(e))
            continue
        assert got is not None
        zip_path, member, df = got
        print(
            f"[warn] RELP not available in PUMS person file; using SPORDER==1 as householder proxy (zip={zip_path} member={member})",
            file=sys.stderr,
        )
        df = df[[_norm(c) for c in cols_sporder]].copy()
        df["SERIALNO"] = df["SERIALNO"].astype(str)
        df["SPORDER"] = pd.to_numeric(df["SPORDER"], errors="coerce")
        df["AGEP"] = pd.to_numeric(df["AGEP"], errors="coerce")
        df = df.dropna().copy()
        df = df[df["SPORDER"].astype(int) == 1].copy()
        if df.empty:
            raise SystemExit(f"No SPORDER==1 rows found in PUMS person file. zip={zip_path} member={member}")
        out = df.groupby("SERIALNO", sort=False)["AGEP"].first().reset_index()
        out.attrs["householder_method"] = "SPORDER==1"
        out.attrs["source_zip"] = str(zip_path)
        out.attrs["source_member"] = str(member)
        return out

    tried = [str(z) for z in candidates]
    details = "\n  - " + "\n  - ".join(errors[:10]) if errors else ""
    raise SystemExit(f"PUMS person zip not found or missing required columns (tried {tried}). Details:{details}")


def _parse_age_bounds(age_bins: list[str]) -> list[tuple[float, float]]:
    """
    Convert age bin labels to numeric bounds [low, high).
    """
    out: list[tuple[float, float]] = []
    for b in age_bins:
        s = str(b).strip().lower()
        m = re.search(r"under\\s+(\\d+)", s)
        if m:
            hi = float(int(m.group(1)))
            out.append((-math.inf, hi))
            continue
        m = re.search(r"(\\d+)\\s+to\\s+(\\d+)", s)
        if m:
            lo = float(int(m.group(1)))
            hi = float(int(m.group(2)) + 1)
            out.append((lo, hi))
            continue
        m = re.search(r"(\\d+)\\s*(?:years?\\s*)?and\\s+over", s)
        if m:
            lo = float(int(m.group(1)))
            out.append((lo, math.inf))
            continue
        raise SystemExit(f"Cannot parse age bin label for B19037: {b!r}")
    return out


def _parse_income_edges(income_bins: list[str]) -> list[float]:
    """
    Build monotonically increasing edges for np.searchsorted, aligned to income_bins order.
    For N bins, returns N-1 edges (upper bounds of bins except the last open-ended bin).
    """
    uppers: list[float] = []
    for idx, b in enumerate(income_bins):
        s = str(b).strip().lower()
        nums = [int(x.replace(",", "")) for x in re.findall(r"(\\d[\\d,]*)", s)]
        if "less than" in s and nums:
            uppers.append(float(nums[0]))
            continue
        if " to " in s and len(nums) >= 2:
            # Use high+1 as exclusive upper bound.
            uppers.append(float(nums[1] + 1))
            continue
        if ("or more" in s or "and over" in s) and nums:
            # Last open-ended bin should not contribute an upper edge.
            if idx != len(income_bins) - 1:
                raise SystemExit(f"Open-ended income bin appears not-last: {b!r}")
            break
        raise SystemExit(f"Cannot parse income bin label for B19037: {b!r}")

    if len(uppers) != max(0, len(income_bins) - 1):
        raise SystemExit("Failed to parse B19037 income bin edges (unexpected bin count).")
    if any(uppers[i] >= uppers[i + 1] for i in range(len(uppers) - 1)):
        raise SystemExit("Parsed income bin edges are not strictly increasing.")
    return uppers


def _pums_b19037_by_puma(
    *,
    data_root: pathlib.Path,
    pums_year: int,
    pums_period: str,
    statefp: str,
    pumas: set[str],
    schema: dict[str, Any],
    n_rows: int,
) -> dict[str, dict[str, Any]]:
    """
    Approximate B19037 joint distributions from PUMS household microdata.
    """
    np = _require("numpy")
    pd = _require("pandas")

    hh = _load_pums_households(data_root=data_root, pums_year=pums_year, pums_period=pums_period, statefp=statefp, pumas=pumas, n_rows=n_rows)
    holder = _load_pums_householder_age(data_root=data_root, pums_year=pums_year, pums_period=pums_period, statefp=statefp, n_rows=n_rows)
    df = hh.merge(holder, on="SERIALNO", how="inner")
    if df.empty:
        raise SystemExit("After joining housing with householder ages, no PUMS rows remain.")

    age_bins = list(schema["age_bins"])
    income_bins = list(schema["income_bins"])
    n_row = int(len(age_bins))
    n_col = int(len(income_bins))

    age_bounds = _parse_age_bounds(age_bins)
    inc_edges = _parse_income_edges(income_bins)

    ages = pd.to_numeric(df["AGEP"], errors="coerce").to_numpy(dtype=float)
    inc = pd.to_numeric(df["HINCP"], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df["WGTP"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    puma = df["PUMA_STR"].astype(str).to_numpy(dtype=str)

    # Bin age.
    age_idx = np.full(ages.shape, -1, dtype=int)
    for i, (lo, hi) in enumerate(age_bounds):
        m = (ages >= float(lo)) & (ages < float(hi))
        age_idx[m] = int(i)

    # Bin income via edges (last bin is >= last edge).
    inc_idx = np.searchsorted(np.asarray(inc_edges, dtype=float), inc, side="left").astype(int)

    m = (age_idx >= 0) & (age_idx < n_row) & (inc_idx >= 0) & (inc_idx < n_col) & (w > 0) & ~np.isnan(inc)
    age_idx = age_idx[m]
    inc_idx = inc_idx[m]
    w = w[m]
    puma = puma[m]

    out: dict[str, dict[str, Any]] = {}
    for pu in sorted(set(puma.tolist())):
        pm = puma == pu
        if not bool(pm.any()):
            continue
        a = age_idx[pm]
        c = inc_idx[pm]
        ww = w[pm]
        flat = a.astype(int) * n_col + c.astype(int)
        counts = np.zeros((n_row * n_col,), dtype=float)
        np.add.at(counts, flat, ww)
        total = float(counts.sum())
        denom = total if total > 0 else 1.0
        p_joint = (counts / denom).astype(float)
        tab = counts.reshape(n_row, n_col)
        p_age = (tab.sum(axis=1) / denom).astype(float)
        p_inc = (tab.sum(axis=0) / denom).astype(float)
        out[str(pu)] = {"total_households": float(total), "p_joint": p_joint, "p_age": p_age, "p_income": p_inc}

    if not out:
        raise SystemExit("Failed to build PUMS household distributions (empty after binning).")
    return out


def _read_acs(path: pathlib.Path, *, table_id: str) -> Any:
    pd = _require("pandas")
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    needed = ["state", "county", "tract", f"{table_id}_001E"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"ACS {table_id} missing columns: {missing}. Columns: {list(df.columns)[:30]}")
    state = df["state"].astype(str).str.zfill(2)
    county = df["county"].astype(str).str.zfill(3)
    tract = df["tract"].astype(str).str.zfill(6)
    df["tract_geoid"] = (state + county + tract).astype(str)

    # Numericize all table estimate columns.
    pat = re.compile(rf"^{re.escape(table_id)}_\d{{3}}E$")
    for c in list(df.columns):
        if pat.match(c):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    return df


def _parse_b19037_schema(*, variables_csv: pathlib.Path) -> dict[str, Any]:
    """
    Parse B19037 estimate variables into a 2D schema:
      age_bins (rows), income_bins (cols), and var->(i,j) mapping.

    We rely on ACS variable label hierarchy like:
      Estimate!!Total:!!Householder under 25 years:!!Less than $10,000
    """
    import csv as _csv

    table_id = "B19037"
    rows = []
    with open(variables_csv, "r", encoding="utf-8", newline="") as f:
        r = _csv.DictReader(f)
        for row in r:
            name = (row.get("name") or "").strip()
            if not (name.startswith(table_id + "_") and name.endswith("E")):
                continue
            label = (row.get("label") or "").strip()
            rows.append((name, label))

    rows.sort(key=lambda x: x[0])
    if not rows:
        raise SystemExit(f"No {table_id} variables found in: {variables_csv}")

    age_bins: list[str] = []
    income_bins: list[str] = []
    var_to_ij: dict[str, tuple[int, int]] = {}

    def _canon(s: str) -> str:
        return s.strip().rstrip(":").strip()

    for name, label in rows:
        parts = [p.strip() for p in str(label).split("!!") if p.strip()]
        # Expect: Estimate!!Total:!!<age>:!!<income>
        if len(parts) < 4:
            continue
        if parts[0].lower() != "estimate":
            continue
        # parts[1] is often "Total:"; keep tolerant.
        age = _canon(parts[2])
        inc = _canon(parts[3])
        if not age or not inc:
            continue
        if inc.lower().startswith("total"):
            continue

        if age not in age_bins:
            age_bins.append(age)
        if inc not in income_bins:
            income_bins.append(inc)
        i = age_bins.index(age)
        j = income_bins.index(inc)
        var_to_ij[name] = (i, j)

    if not var_to_ij or not age_bins or not income_bins:
        raise SystemExit(f"Failed to parse {table_id} 2D schema from: {variables_csv}")

    return {"table_id": table_id, "age_bins": age_bins, "income_bins": income_bins, "var_to_ij": var_to_ij}


def _targets_by_tract(df: Any, *, tracts: set[str], schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    np = _require("numpy")
    table_id = str(schema["table_id"])
    age_bins = list(schema["age_bins"])
    income_bins = list(schema["income_bins"])
    var_to_ij: dict[str, tuple[int, int]] = dict(schema["var_to_ij"])
    n_row = int(len(age_bins))
    n_col = int(len(income_bins))

    out: dict[str, dict[str, Any]] = {}
    for r in df.itertuples(index=False):
        tg = str(getattr(r, "tract_geoid"))
        if tg not in tracts:
            continue
        counts = np.zeros((n_row, n_col), dtype=float)
        for var, (i, j) in var_to_ij.items():
            if not hasattr(r, var):
                continue
            counts[int(i), int(j)] = float(getattr(r, var))
        total = float(counts.sum())
        denom = total if total > 0 else 1.0
        p_joint = (counts.reshape(-1) / denom).astype(float)
        p_age = (counts.sum(axis=1) / denom).astype(float)
        p_income = (counts.sum(axis=0) / denom).astype(float)
        out[tg] = {
            "total_households": float(total),
            "p_joint": p_joint,
            "p_age": p_age,
            "p_income": p_income,
            "counts_joint": counts.reshape(-1).astype(float),
        }
    if not out:
        raise SystemExit("No matching tracts found in ACS B19037 for the given study area.")
    return out


def main() -> None:
    np = _require("numpy")
    pd = _require("pandas")
    torch = _require("torch")

    from src.synthpop.model.diffusion_tabular import DiffusionTabularModel, TabDDPMConfig
    from src.synthpop.pipeline.detroit_v0 import make_run_id

    p = argparse.ArgumentParser(prog="poc_tabddpm_acs_supervised_b19037")
    p.add_argument("--acs_b19037_csv_gz", required=True, help="ACS B19037 tract CSV.gz (downloaded by detroit_fetch_public_data.py).")
    p.add_argument("--acs_b19037_variables_csv", default=None, help="ACS B19037 variables.csv (generated by detroit_fetch_public_data.py). If not set, inferred from the CSV.gz directory.")
    p.add_argument("--buildings_csv", required=True, help="Buildings CSV with tract_geoid and puma (for tract->PUMA mapping).")
    p.add_argument("--data_root", default=None, help="Detroit data_root (only for external PUMS validation).")
    p.add_argument("--pums_year", type=int, default=2023)
    p.add_argument("--pums_period", default="5-Year")
    p.add_argument("--statefp", default="26")
    p.add_argument("--pums_n_rows", type=int, default=300_000)
    p.add_argument(
        "--exclude_pumas",
        default="",
        help='Optional comma-separated PUMA codes to exclude (e.g. "3202,3203").',
    )
    p.add_argument(
        "--puma_blocks",
        required=True,
        help='Explicit PUMA blocks. Example: "3208,3209;3210,3211;3212,3213".',
    )
    p.add_argument("--conditions", default="marginal", help='Comma-separated conditions: "marginal" or "none". Default: marginal.')
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--n_eval_joint_samples", type=int, default=64)
    p.add_argument("--out_dir", default=None, help="Output directory (default: outputs/<run_id>).")
    args = p.parse_args()

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))

    acs_path = pathlib.Path(args.acs_b19037_csv_gz).expanduser().resolve()
    buildings_csv = pathlib.Path(args.buildings_csv).expanduser().resolve()

    if args.acs_b19037_variables_csv:
        vars_path = pathlib.Path(args.acs_b19037_variables_csv).expanduser().resolve()
    else:
        # detroit_fetch_public_data.py writes: acs5_<year>_B19037_variables.csv next to the csv.gz
        candidates = sorted(acs_path.parent.glob("acs5_*_B19037_variables.csv"))
        if not candidates:
            raise SystemExit("Cannot infer --acs_b19037_variables_csv. Pass it explicitly (acs5_YYYY_B19037_variables.csv).")
        vars_path = candidates[-1]

    if args.out_dir:
        out_root = pathlib.Path(args.out_dir).expanduser().resolve()
    else:
        out_root = pathlib.Path("outputs") / make_run_id(prefix="poc_acs_supervised_b19037")
    out_root.mkdir(parents=True, exist_ok=True)

    blocks = _parse_puma_blocks(str(args.puma_blocks))

    tract_to_puma = _load_buildings_for_mapping(buildings_csv)
    exclude_pumas = {str(_normalize_puma(x) or x) for x in str(args.exclude_pumas).split(",") if str(x).strip()}
    exclude_pumas = {p for p in exclude_pumas if p and p.lower() not in {"nan", "none"}}

    study_tracts = {tg for tg, pu in tract_to_puma.items() if str(pu) not in exclude_pumas}
    study_pumas = sorted({str(pu) for tg, pu in tract_to_puma.items() if tg in study_tracts})
    if len(study_pumas) < 2:
        raise SystemExit(f"Too few study PUMAs inferred from buildings_csv: {study_pumas}")

    schema = _parse_b19037_schema(variables_csv=vars_path)
    age_bins = list(schema["age_bins"])
    income_bins = list(schema["income_bins"])
    n_row = int(len(age_bins))
    n_col = int(len(income_bins))
    K = int(n_row * n_col)

    df = _read_acs(acs_path, table_id="B19037")
    targets_by_tract = _targets_by_tract(df, tracts=study_tracts, schema=schema)

    cond_list = [c.strip() for c in str(args.conditions).split(",") if c.strip()]
    unknown = [c for c in cond_list if c not in {"none", "marginal"}]
    if unknown:
        raise SystemExit(f"Unknown condition(s): {unknown}")

    # Optional external validation (PUMS) setup.
    pums_puma = None
    if args.data_root:
        data_root = pathlib.Path(args.data_root).expanduser().resolve()
        pums_puma = _pums_b19037_by_puma(
            data_root=data_root,
            pums_year=int(args.pums_year),
            pums_period=str(args.pums_period),
            statefp=str(args.statefp),
            pumas=set(study_pumas),
            schema=schema,
            n_rows=int(args.pums_n_rows),
        )

    run_meta = {
        "out_root": str(out_root),
        "acs_b19037_csv_gz": str(acs_path),
        "acs_b19037_variables_csv": str(vars_path),
        "buildings_csv": str(buildings_csv),
        "study_pumas": study_pumas,
        "n_tracts": int(len(study_tracts)),
        "puma_blocks": blocks,
        "conditions": cond_list,
        "x_model": "joint_tabddpm_logp",
        "K": int(K),
        "age_bins": age_bins,
        "income_bins": income_bins,
        "timesteps": int(args.timesteps),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "n_eval_joint_samples": int(args.n_eval_joint_samples),
        "external_validation": {"enabled": bool(pums_puma is not None), "pums_year": int(args.pums_year), "pums_period": str(args.pums_period)},
    }
    _write_json(out_root / "run_summary.json", run_meta)

    folds = list(range(len(blocks)))
    internal_by_condition: dict[str, Any] = {}
    external_by_condition: dict[str, Any] = {}
    baselines_by_fold: dict[str, Any] = {"independence": {}, "ipf_train_seed": {}}
    baseline_gap: dict[str, Any] | None = None

    # Baseline gap (ACS->PUMA vs PUMS), method-independent.
    if pums_puma is not None:
        by_puma: dict[str, Any] = {}
        for pu in study_pumas:
            # Aggregate ACS counts across tracts in this PUMA.
            tracts = [tg for tg, p in tract_to_puma.items() if tg in study_tracts and str(p) == str(pu)]
            counts = np.zeros((K,), dtype=float)
            total = 0.0
            for tg in tracts:
                t = targets_by_tract.get(str(tg))
                if t is None:
                    continue
                c = np.asarray(t["counts_joint"], dtype=float).reshape(-1)
                counts += c
                total += float(t["total_households"])
            if float(counts.sum()) <= 0 or str(pu) not in pums_puma:
                continue
            p_acs = counts / float(counts.sum())
            p_ref = np.asarray(pums_puma[str(pu)]["p_joint"], dtype=float)
            p_age_acs, p_inc_acs = _marginals_from_joint(p_joint=p_acs, n_row=n_row, n_col=n_col)
            p_age_ref, p_inc_ref = _marginals_from_joint(p_joint=p_ref, n_row=n_row, n_col=n_col)
            by_puma[str(pu)] = {
                "tvd_joint": float(_tvd(p_acs, p_ref)),
                "tvd_age": float(_tvd(p_age_acs, p_age_ref)),
                "tvd_income": float(_tvd(p_inc_acs, p_inc_ref)),
                "n_tracts": int(len(tracts)),
                "acs_total_households": float(total),
                "pums_total_households": float(pums_puma[str(pu)]["total_households"]),
            }

        vals_joint = [float(v["tvd_joint"]) for v in by_puma.values()]
        vals_age = [float(v["tvd_age"]) for v in by_puma.values()]
        vals_inc = [float(v["tvd_income"]) for v in by_puma.values()]
        baseline_gap = {
            "by_puma": by_puma,
            "summary": {
                "tvd_joint": {"mean": float(np.mean(vals_joint)), "max": float(np.max(vals_joint))} if vals_joint else None,
                "tvd_age": {"mean": float(np.mean(vals_age)), "max": float(np.max(vals_age))} if vals_age else None,
                "tvd_income": {"mean": float(np.mean(vals_inc)), "max": float(np.max(vals_inc))} if vals_inc else None,
            },
        }
        _write_json(out_root / "metrics" / "acs_pums_baseline_gap.json", baseline_gap)

    for cond in cond_list:
        internal_by_condition[cond] = {"by_fold": {}}
        external_by_condition[cond] = {"by_fold": {}}

    for fold_idx in folds:
        test_pumas = set(blocks[int(fold_idx)])
        train_pumas = set(study_pumas) - set(test_pumas)
        train_tracts = {tg for tg, pu in tract_to_puma.items() if tg in study_tracts and str(pu) in train_pumas}
        test_tracts = {tg for tg, pu in tract_to_puma.items() if tg in study_tracts and str(pu) in test_pumas}
        if not train_tracts or not test_tracts:
            raise SystemExit(f"Empty train/test tracts in fold={fold_idx}. train={len(train_tracts)} test={len(test_tracts)}")

        # Seed joint from TRAIN tracts only.
        seed_counts = np.zeros((K,), dtype=float)
        for tg in sorted(train_tracts):
            t = targets_by_tract.get(str(tg))
            if t is None:
                continue
            seed_counts += np.asarray(t["counts_joint"], dtype=float).reshape(-1)
        seed_p = seed_counts / (float(seed_counts.sum()) if float(seed_counts.sum()) > 0 else 1.0)

        # Baselines on held-out tracts.
        ind_by_tract: dict[str, Any] = {}
        ipf_by_tract: dict[str, Any] = {}
        for tg in sorted(test_tracts):
            t = targets_by_tract.get(str(tg))
            if t is None:
                continue
            p_age = np.asarray(t["p_age"], dtype=float)
            p_inc = np.asarray(t["p_income"], dtype=float)
            p_true = np.asarray(t["p_joint"], dtype=float)

            p_ind = _outer_from_marginals(p_row=p_age, p_col=p_inc)
            p_ipf = _ipf_2d(seed_joint=seed_p, target_row=p_age, target_col=p_inc)
            ind_by_tract[str(tg)] = {"tvd_joint": float(_tvd(p_ind, p_true))}
            ipf_by_tract[str(tg)] = {"tvd_joint": float(_tvd(p_ipf, p_true))}

        def _summ(vals: list[float]) -> dict[str, float]:
            return {"mean": float(np.mean(vals)), "max": float(np.max(vals)), "p90": float(np.quantile(vals, 0.9))} if vals else {"mean": float("nan"), "max": float("nan"), "p90": float("nan")}

        baselines_by_fold["independence"][str(fold_idx)] = _summ([v["tvd_joint"] for v in ind_by_tract.values()])
        baselines_by_fold["ipf_train_seed"][str(fold_idx)] = _summ([v["tvd_joint"] for v in ipf_by_tract.values()])

        # Conditions: train one model per (fold, cond).
        for cond in cond_list:
            fold_dir = out_root / f"fold_{fold_idx}" / cond
            fold_dir.mkdir(parents=True, exist_ok=True)
            ckpt = fold_dir / "model.pt"

            # Build train set: one vector per tract.
            xs = []
            cs = []
            for tg in sorted(train_tracts):
                t = targets_by_tract.get(str(tg))
                if t is None:
                    continue
                p_joint = np.asarray(t["p_joint"], dtype=np.float32).reshape(-1)
                if p_joint.size != K:
                    continue
                xs.append(np.log(np.clip(p_joint, 0.0, None) + 1e-6).reshape(1, K))
                if cond == "marginal":
                    c = np.concatenate([np.asarray(t["p_age"], dtype=np.float32).reshape(-1), np.asarray(t["p_income"], dtype=np.float32).reshape(-1)], axis=0)
                    cs.append(c.reshape(1, -1))

            if not xs:
                raise SystemExit(f"No training tracts samples for fold={fold_idx}, cond={cond}")
            x_u_all = np.concatenate(xs, axis=0).astype(np.float32)
            x_mean = x_u_all.mean(axis=0).astype(np.float32)
            x_std = x_u_all.std(axis=0).astype(np.float32)
            x_std = np.where(x_std <= 1e-6, 1.0, x_std).astype(np.float32)
            x_z = ((x_u_all - x_mean) / x_std).astype(np.float32)

            x = torch.from_numpy(x_z)
            if cond == "marginal":
                cond_all = np.concatenate(cs, axis=0).astype(np.float32)
                cond_t = torch.from_numpy(cond_all)
            else:
                cond_t = None

            cfg = TabDDPMConfig(timesteps=int(args.timesteps))
            model = DiffusionTabularModel(input_dim=int(K), cond_dim=int(cond_t.shape[1]) if cond_t is not None else 0, seed=int(args.seed), config=cfg)
            train_metrics = model.fit(
                x=x,
                cond=cond_t,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                device=args.device,
                log_every=int(args.log_every),
            )
            model.save(ckpt)
            _write_json(
                fold_dir / "train_summary.json",
                {
                    "fold": int(fold_idx),
                    "condition": cond,
                    "x_model": "joint_tabddpm_logp",
                    "K": int(K),
                    "cond_dim": int(cond_t.shape[1]) if cond_t is not None else 0,
                    "x_mean": [float(v) for v in x_mean.tolist()],
                    "x_std": [float(v) for v in x_std.tolist()],
                    "train_metrics": train_metrics,
                },
            )

            # Internal evaluation on held-out tracts.
            internal_by_tract: dict[str, Any] = {}
            p_hat_by_tract: dict[str, Any] = {}
            for tg in sorted(test_tracts):
                t = targets_by_tract.get(str(tg))
                if t is None:
                    continue
                n_eval = int(args.n_eval_joint_samples)
                if cond == "marginal":
                    c = np.concatenate([np.asarray(t["p_age"], dtype=np.float32).reshape(-1), np.asarray(t["p_income"], dtype=np.float32).reshape(-1)], axis=0)
                    c_rep = np.repeat(c.reshape(1, -1), repeats=n_eval, axis=0)
                    c_t = torch.from_numpy(c_rep)
                else:
                    c_t = None

                z = model.sample(n=n_eval, cond=c_t, device=args.device).numpy().astype(np.float32)
                logp = (z * x_std.reshape(1, -1) + x_mean.reshape(1, -1)).astype(np.float32)
                logp = logp - logp.max(axis=1, keepdims=True)
                p = np.exp(logp)
                p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
                p_joint_raw = p.mean(axis=0).astype(float)
                p_joint = p_joint_raw
                if cond == "marginal":
                    p_joint = _ipf_2d(seed_joint=p_joint_raw, target_row=t["p_age"], target_col=t["p_income"])
                p_age_hat, p_inc_hat = _marginals_from_joint(p_joint=p_joint, n_row=n_row, n_col=n_col)
                p_hat_by_tract[str(tg)] = {"p_joint": p_joint, "p_age": p_age_hat, "p_income": p_inc_hat}
                internal_by_tract[str(tg)] = {
                    "tvd_joint": float(_tvd(p_joint, t["p_joint"])),
                    "tvd_age": float(_tvd(p_age_hat, t["p_age"])),
                    "tvd_income": float(_tvd(p_inc_hat, t["p_income"])),
                }

            vals_joint = [v["tvd_joint"] for v in internal_by_tract.values()]
            vals_age = [v["tvd_age"] for v in internal_by_tract.values()]
            vals_inc = [v["tvd_income"] for v in internal_by_tract.values()]
            internal_by_condition[cond]["by_fold"][str(fold_idx)] = {
                "tvd_joint": _summ(vals_joint),
                "tvd_age": _summ(vals_age),
                "tvd_income": _summ(vals_inc),
            }

            # External evaluation vs PUMS at PUMA level (held-out PUMAs only).
            if pums_puma is not None:
                by_puma: dict[str, Any] = {}
                for pu in sorted(test_pumas):
                    if str(pu) not in pums_puma:
                        continue
                    tracts = [tg for tg in test_tracts if str(tract_to_puma.get(str(tg))) == str(pu)]
                    counts_hat = np.zeros((K,), dtype=float)
                    for tg in tracts:
                        t = targets_by_tract.get(str(tg))
                        ph = p_hat_by_tract.get(str(tg))
                        if t is None or ph is None:
                            continue
                        w = float(t["total_households"])
                        counts_hat += w * np.asarray(ph["p_joint"], dtype=float)
                    if float(counts_hat.sum()) <= 0:
                        continue
                    p_hat = counts_hat / float(counts_hat.sum())
                    p_ref = np.asarray(pums_puma[str(pu)]["p_joint"], dtype=float)
                    p_age_hat, p_inc_hat = _marginals_from_joint(p_joint=p_hat, n_row=n_row, n_col=n_col)
                    p_age_ref, p_inc_ref = _marginals_from_joint(p_joint=p_ref, n_row=n_row, n_col=n_col)
                    by_puma[str(pu)] = {
                        "tvd_joint": float(_tvd(p_hat, p_ref)),
                        "tvd_age": float(_tvd(p_age_hat, p_age_ref)),
                        "tvd_income": float(_tvd(p_inc_hat, p_inc_ref)),
                        "n_tracts": int(len(tracts)),
                    }

                vals_joint = [float(v["tvd_joint"]) for v in by_puma.values()]
                vals_age = [float(v["tvd_age"]) for v in by_puma.values()]
                vals_inc = [float(v["tvd_income"]) for v in by_puma.values()]
                external_by_condition[cond]["by_fold"][str(fold_idx)] = {
                    "tvd_joint": _summ(vals_joint),
                    "tvd_age": _summ(vals_age),
                    "tvd_income": _summ(vals_inc),
                    "by_puma": by_puma,
                }
                _write_json(fold_dir / "metrics" / "external_pums_by_puma.json", external_by_condition[cond]["by_fold"][str(fold_idx)])

    # Aggregate baselines across folds.
    baselines_internal = {"by_baseline": {}}
    for bname in ["independence", "ipf_train_seed"]:
        per = baselines_by_fold[bname]
        # mean across folds (means of means).
        m = [float(per[str(i)]["mean"]) for i in folds]
        baselines_internal["by_baseline"][bname] = {"tvd_joint": {"mean": {"mean": float(np.mean(m)), "std": float(np.std(m, ddof=0))}}, "by_fold": per}
    _write_json(out_root / "metrics" / "baselines_internal.json", baselines_internal)

    # Aggregate internal across folds per condition.
    internal_acs = {"by_condition": {}, "by_fold": {}}
    for cond in cond_list:
        by_fold = internal_by_condition[cond]["by_fold"]
        vals_joint = [float(by_fold[str(i)]["tvd_joint"]["mean"]) for i in folds]
        vals_age = [float(by_fold[str(i)]["tvd_age"]["mean"]) for i in folds]
        vals_inc = [float(by_fold[str(i)]["tvd_income"]["mean"]) for i in folds]
        internal_acs["by_condition"][cond] = {
            "tvd_joint": {"mean": float(np.mean(vals_joint)), "max": float(np.max(vals_joint)), "p90": float(np.quantile(vals_joint, 0.9)), "n_folds": int(len(folds))},
            "tvd_age": {"mean": float(np.mean(vals_age)), "max": float(np.max(vals_age)), "p90": float(np.quantile(vals_age, 0.9)), "n_folds": int(len(folds))},
            "tvd_income": {"mean": float(np.mean(vals_inc)), "max": float(np.max(vals_inc)), "p90": float(np.quantile(vals_inc, 0.9)), "n_folds": int(len(folds))},
        }
        internal_acs["by_fold"][cond] = by_fold
    _write_json(out_root / "metrics" / "internal_acs_holdout.json", internal_acs)

    # Aggregate external across folds per condition.
    if pums_puma is not None:
        external = {"by_condition": {}, "by_fold": {}}
        for cond in cond_list:
            by_fold = external_by_condition[cond]["by_fold"]
            external["by_fold"][cond] = by_fold
            vals_joint = [float(by_fold[str(i)]["tvd_joint"]["mean"]) for i in folds if str(i) in by_fold]
            vals_age = [float(by_fold[str(i)]["tvd_age"]["mean"]) for i in folds if str(i) in by_fold]
            vals_inc = [float(by_fold[str(i)]["tvd_income"]["mean"]) for i in folds if str(i) in by_fold]
            external["by_condition"][cond] = {
                "tvd_joint": {"mean": float(np.mean(vals_joint)), "max": float(np.max(vals_joint)), "p90": float(np.quantile(vals_joint, 0.9)), "n_folds": int(len(vals_joint))} if vals_joint else None,
                "tvd_age": {"mean": float(np.mean(vals_age)), "max": float(np.max(vals_age)), "p90": float(np.quantile(vals_age, 0.9)), "n_folds": int(len(vals_age))} if vals_age else None,
                "tvd_income": {"mean": float(np.mean(vals_inc)), "max": float(np.max(vals_inc)), "p90": float(np.quantile(vals_inc, 0.9)), "n_folds": int(len(vals_inc))} if vals_inc else None,
            }
        _write_json(out_root / "metrics" / "external_pums_by_puma.json", external)
    else:
        _write_json(out_root / "metrics" / "external_pums_by_puma.json", {"by_condition": {}, "note": "external validation disabled (no --data_root)"})
        if baseline_gap is None:
            _write_json(out_root / "metrics" / "acs_pums_baseline_gap.json", {"note": "external validation disabled (no --data_root)"})

    # Final ablation summary.
    _write_json(
        out_root / "metrics" / "ablation_summary.json",
        {
            "conditions": cond_list,
            "folds": folds,
            "internal_acs": internal_by_condition,
            "baselines_internal": baselines_internal["by_baseline"],
            "external_pums": external_by_condition if pums_puma is not None else {"note": "external validation disabled (no --data_root)"},
            "baseline_gap": baseline_gap if baseline_gap is not None else {"note": "external validation disabled (no --data_root)"},
        },
    )

    print(f"[ok] wrote: {out_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
