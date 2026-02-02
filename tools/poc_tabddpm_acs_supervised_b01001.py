#!/usr/bin/env python3
from __future__ import annotations

"""
PoC (Scheme C idea): ACS-supervised diffusion on tract-level age×sex, with PUMS as external validation.

Core principle (PI-aligned):
- Training uses ONLY ACS tract-level distributions (B01001), plus tract_context (geo + built).
- PUMS microdata is used ONLY for external validation at the PUMA level (never used in training).

Why "pseudo-individuals":
- Diffusion models are trained on samples x0.
- ACS provides distribution-level supervision; we convert it into sample-level supervision by
  sampling pseudo-individuals from tract-level B01001 age×sex distributions.

This script implements:
1) Build tract_context (geo-only / built-only / geo+built) and a "none" ablation.
2) 4-fold CV by PUMA blocks (pairs of adjacent PUMAs; greedy pairing).
3) Internal evaluation: per-tract TVD vs ACS on held-out tracts.
4) External evaluation: aggregate tract predictions to PUMA and compare vs PUMS (TVD),
   plus a baseline gap (ACS->PUMA vs PUMS) as a method-independent lower bound.
"""

import argparse
import json
import math
import pathlib
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


def _pick_col(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _age23_bins() -> list[tuple[int, int, str]]:
    """
    ACS B01001 age groups (23 bins), shared by male/female.
    Uses left-closed, right-open integer intervals on AGEP.
    """
    return [
        (0, 5, "0-4"),
        (5, 10, "5-9"),
        (10, 15, "10-14"),
        (15, 18, "15-17"),
        (18, 20, "18-19"),
        (20, 21, "20"),
        (21, 22, "21"),
        (22, 25, "22-24"),
        (25, 30, "25-29"),
        (30, 35, "30-34"),
        (35, 40, "35-39"),
        (40, 45, "40-44"),
        (45, 50, "45-49"),
        (50, 55, "50-54"),
        (55, 60, "55-59"),
        (60, 62, "60-61"),
        (62, 65, "62-64"),
        (65, 67, "65-66"),
        (67, 70, "67-69"),
        (70, 75, "70-74"),
        (75, 80, "75-79"),
        (80, 85, "80-84"),
        (85, 200, "85+"),
    ]


def _age23_index(age: Any) -> int | None:
    try:
        a = int(float(age))
    except Exception:
        return None
    if a < 0:
        a = 0
    for i, (lo, hi, _lab) in enumerate(_age23_bins()):
        if lo <= a < hi:
            return i
    return len(_age23_bins()) - 1


def _b01001_columns() -> tuple[list[str], list[str]]:
    male = [f"B01001_{i:03d}E" for i in range(3, 26)]  # 003..025 (23 bins)
    female = [f"B01001_{i:03d}E" for i in range(27, 50)]  # 027..049 (23 bins)
    return male, female


def _read_acs_b01001(path: pathlib.Path) -> Any:
    pd = _require("pandas")
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    needed = ["state", "county", "tract", "B01001_001E"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"ACS B01001 missing columns: {missing}. Columns: {list(df.columns)[:30]}")

    state = df["state"].astype(str).str.zfill(2)
    county = df["county"].astype(str).str.zfill(3)
    tract = df["tract"].astype(str).str.zfill(6)
    df["tract_geoid"] = (state + county + tract).astype(str)

    # Numericize target columns.
    df["B01001_001E"] = pd.to_numeric(df["B01001_001E"], errors="coerce").fillna(0.0).clip(lower=0.0)
    male_cols, female_cols = _b01001_columns()
    for c in male_cols + female_cols + ["B01001_002E", "B01001_026E"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    return df


def _b01001_targets_by_tract(df_b01001: Any, *, tracts: set[str]) -> dict[str, dict[str, Any]]:
    """
    Build per-tract targets:
    - total_pop
    - p_joint (46)
    - p_age (23)
    - p_sex (2)
    - counts (for ACS->PUMA baseline)
    """
    np = _require("numpy")

    male_cols, female_cols = _b01001_columns()
    missing = [c for c in (male_cols + female_cols) if c not in df_b01001.columns]
    if missing:
        raise SystemExit(f"ACS B01001 missing detailed columns (need 003..025 and 027..049). Missing: {missing[:10]}")

    out: dict[str, dict[str, Any]] = {}
    for r in df_b01001.itertuples(index=False):
        tg = str(getattr(r, "tract_geoid"))
        if tg not in tracts:
            continue
        total = float(getattr(r, "B01001_001E"))
        male = np.array([float(getattr(r, c)) for c in male_cols], dtype=float)
        female = np.array([float(getattr(r, c)) for c in female_cols], dtype=float)
        joint = np.concatenate([male, female], axis=0)  # 46, male then female
        denom = total if total > 0 else float(joint.sum())
        denom = denom if denom > 0 else 1.0
        p_joint = (joint / denom).astype(float)
        p_age = ((male + female) / denom).astype(float)
        p_sex = (np.array([male.sum(), female.sum()], dtype=float) / denom).astype(float)
        out[tg] = {
            "total_pop": float(total),
            "p_joint": p_joint,
            "p_age": p_age,
            "p_sex": p_sex,
            "counts_joint": joint.astype(float),
            "counts_age": (male + female).astype(float),
            "counts_sex": np.array([male.sum(), female.sum()], dtype=float),
        }
    if not out:
        raise SystemExit("No matching tracts found in ACS B01001 for the given study area.")
    return out


def _load_buildings(buildings_csv: pathlib.Path, *, n_tiers: int) -> Any:
    pd = _require("pandas")
    b = pd.read_csv(buildings_csv, low_memory=False)
    needed = ["bldg_id", "puma", "tract_geoid", "footprint_area_m2", "height_m", "cap_proxy"]
    missing = [c for c in needed if c not in b.columns]
    if missing:
        raise SystemExit(f"buildings_csv missing columns: {missing}")
    b["tract_geoid"] = b["tract_geoid"].astype(str)
    b["puma"] = b["puma"].map(_normalize_puma)
    b["footprint_area_m2"] = pd.to_numeric(b["footprint_area_m2"], errors="coerce").fillna(0.0).clip(lower=0.0)
    b["height_m"] = pd.to_numeric(b["height_m"], errors="coerce").fillna(0.0).clip(lower=0.0)
    b["cap_proxy"] = pd.to_numeric(b["cap_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "price_tier" in b.columns:
        b["price_tier"] = pd.to_numeric(b["price_tier"], errors="coerce")
        b.loc[(b["price_tier"] < 1) | (b["price_tier"] > int(n_tiers)), "price_tier"] = float("nan")
    return b


def _tract_to_puma_from_buildings(buildings: Any) -> dict[str, str]:
    pd = _require("pandas")
    tract_to_puma: dict[str, str] = {}
    g = buildings.dropna(subset=["tract_geoid", "puma"]).copy()
    if g.empty:
        return tract_to_puma
    for tract, sub in g.groupby("tract_geoid", sort=False):
        mode = sub["puma"].astype(str).value_counts(dropna=True)
        if mode.empty:
            continue
        tract_to_puma[str(tract)] = str(mode.index[0])
    return tract_to_puma


def _build_built_context(buildings: Any, *, n_tiers: int) -> Any:
    pd = _require("pandas")
    import numpy as np  # type: ignore

    g = buildings.groupby("tract_geoid", sort=False)
    out = pd.DataFrame(
        {
            "tract_geoid": g.size().index.astype(str),
            "n_buildings": g.size().to_numpy(dtype=float),
            "cap_proxy_sum": g["cap_proxy"].sum().to_numpy(dtype=float),
            "height_mean": g["height_m"].mean().to_numpy(dtype=float),
            "footprint_mean": g["footprint_area_m2"].mean().to_numpy(dtype=float),
        }
    )
    out["n_buildings_log"] = np.log1p(out["n_buildings"].astype(float))
    out["cap_proxy_sum_log"] = np.log1p(out["cap_proxy_sum"].astype(float))
    out["footprint_mean_log"] = np.log1p(out["footprint_mean"].astype(float))

    # Price tier histogram (proportions) if available.
    for k in range(1, int(n_tiers) + 1):
        out[f"price_tier_p{k}"] = 0.0
    if "price_tier" in buildings.columns:
        b2 = buildings.dropna(subset=["price_tier"]).copy()
        if not b2.empty:
            b2["price_tier"] = pd.to_numeric(b2["price_tier"], errors="coerce")
            b2 = b2.dropna(subset=["price_tier"]).copy()
            b2["price_tier"] = b2["price_tier"].astype(int)
            b2 = b2[(b2["price_tier"] >= 1) & (b2["price_tier"] <= int(n_tiers))].copy()
        if not b2.empty:
            counts = (
                b2.groupby(["tract_geoid", "price_tier"], sort=False)["bldg_id"]
                .size()
                .unstack(fill_value=0)
                .reindex(columns=list(range(1, int(n_tiers) + 1)), fill_value=0)
            )
            denom = counts.sum(axis=1).replace(0, 1).astype(float)
            props = counts.div(denom, axis=0).astype(float)
            props.columns = [f"price_tier_p{k}" for k in range(1, int(n_tiers) + 1)]
            props = props.reset_index()
            props["tract_geoid"] = props["tract_geoid"].astype(str)
            out = out.merge(props, on="tract_geoid", how="left")
            for k in range(1, int(n_tiers) + 1):
                out[f"price_tier_p{k}"] = pd.to_numeric(out.get(f"price_tier_p{k}", 0.0), errors="coerce").fillna(0.0).astype(float)
    return out


def _build_geo_context(*, tiger_tract_zip: pathlib.Path, tracts: set[str], cbd_lon: float, cbd_lat: float) -> Any:
    gpd = _require("geopandas")
    pd = _require("pandas")
    pyproj = _require("pyproj")

    tract = gpd.read_file(f"zip://{tiger_tract_zip}")
    if tract.crs is None:
        tract = tract.set_crs(4269, allow_override=True)
    tract = tract.to_crs(3857)

    geoid_col = _pick_col(list(tract.columns), ("GEOID", "GEOID20", "GEOID10"))
    if geoid_col is None:
        raise SystemExit(f"Cannot find tract GEOID column in: {tiger_tract_zip}")
    tract = tract[[geoid_col, "geometry"]].rename(columns={geoid_col: "tract_geoid"})
    tract["tract_geoid"] = tract["tract_geoid"].astype(str)
    tract = tract[tract["tract_geoid"].isin(sorted(tracts))].copy()
    if tract.empty:
        raise SystemExit("No matching tracts found in TIGER tract zip for the given study area.")

    cent = tract.geometry.centroid
    cent_gdf = gpd.GeoDataFrame(tract[["tract_geoid"]].copy(), geometry=cent, crs=3857)
    cent_ll = cent_gdf.to_crs(4326)

    # Area (km^2)
    area_km2 = tract.geometry.area.astype(float) / 1e6

    # Dist to CBD (in km)
    tr = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
    cbd_x, cbd_y = tr.transform(float(cbd_lon), float(cbd_lat))
    dx = cent_gdf.geometry.x.to_numpy(dtype=float) - float(cbd_x)
    dy = cent_gdf.geometry.y.to_numpy(dtype=float) - float(cbd_y)
    dist_cbd_km = (dx * dx + dy * dy) ** 0.5 / 1000.0

    out = pd.DataFrame(
        {
            "tract_geoid": cent_ll["tract_geoid"].astype(str),
            "centroid_lon": cent_ll.geometry.x.astype(float),
            "centroid_lat": cent_ll.geometry.y.astype(float),
            "area_km2": area_km2.to_numpy(dtype=float),
            "dist_cbd_km": dist_cbd_km.astype(float),
        }
    )
    return out


def _standardize(df: Any, *, cols: list[str], mean: dict[str, float] | None = None, std: dict[str, float] | None = None) -> tuple[Any, dict[str, float], dict[str, float]]:
    pd = _require("pandas")
    out = df.copy()
    mean_out: dict[str, float] = {}
    std_out: dict[str, float] = {}
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
        mu = float(x.mean()) if mean is None else float(mean[c])
        sd = float(x.std()) if std is None else float(std[c])
        if not (sd > 1e-6):
            sd = 1.0
        out[c] = ((x - mu) / sd).astype(float)
        mean_out[c] = mu
        std_out[c] = sd
    return out, mean_out, std_out


def _tvd(p: Any, q: Any) -> float:
    np = _require("numpy")
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p - q).sum())


def _marginals_from_joint(p_joint: Any) -> tuple[Any, Any]:
    """
    Convert a 46-dim (sex x age23) joint distribution into:
      - p_age: (23,)
      - p_sex: (2,)
    """
    np = _require("numpy")
    p = np.asarray(p_joint, dtype=float).reshape(-1)
    if p.size != 46:
        raise ValueError(f"p_joint must have length 46, got {p.size}")
    male = p[:23]
    female = p[23:]
    p_age = (male + female).astype(float)
    p_sex = np.array([float(male.sum()), float(female.sum())], dtype=float)
    p_age = p_age / (float(p_age.sum()) if float(p_age.sum()) > 0 else 1.0)
    p_sex = p_sex / (float(p_sex.sum()) if float(p_sex.sum()) > 0 else 1.0)
    return p_age, p_sex


def _sample_pseudo(
    *,
    rng: Any,
    p_joint: Any,
    n: int,
) -> tuple[Any, Any]:
    """
    Sample pseudo individuals from a 46-dim joint distribution.
    Returns:
      age_idx: (n,) int in [0,22]
      sex01: (n,) int in {0,1} (0=male,1=female)
    """
    np = _require("numpy")
    p = np.asarray(p_joint, dtype=float)
    if p.size != 46:
        raise ValueError(f"p_joint must have length 46, got {p.size}")
    p = np.clip(p, 0.0, 1.0)
    s = float(p.sum())
    if s <= 0:
        p = np.full((46,), 1.0 / 46.0, dtype=float)
    else:
        p = p / s
    idx = rng.choice(46, size=int(n), replace=True, p=p)
    sex01 = (idx // 23).astype(int)
    age_idx = (idx % 23).astype(int)
    return age_idx, sex01


def _decode_samples(samples: Any) -> tuple[Any, Any]:
    """
    Decode sampled (age_u, sex_u) in [0,1]-like space into discrete bins.
    """
    np = _require("numpy")
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"samples must be (N,2), got {x.shape}")
    age_u = np.clip(x[:, 0], 0.0, 1.0)
    sex_u = np.clip(x[:, 1], 0.0, 1.0)
    age_idx = np.clip(np.rint(age_u * 22.0), 0, 22).astype(int)
    sex01 = np.clip(np.rint(sex_u), 0, 1).astype(int)
    return age_idx, sex01


def _p_from_samples(*, age_idx: Any, sex01: Any) -> dict[str, Any]:
    np = _require("numpy")
    n = int(len(age_idx))
    if n <= 0:
        return {"p_joint": np.full((46,), 1.0 / 46.0), "p_age": np.full((23,), 1.0 / 23.0), "p_sex": np.full((2,), 0.5)}

    joint_counts = np.zeros((46,), dtype=float)
    for a, s in zip(age_idx, sex01, strict=False):
        idx = int(s) * 23 + int(a)
        joint_counts[idx] += 1.0
    p_joint = joint_counts / float(n)

    age_counts = np.zeros((23,), dtype=float)
    for a in age_idx:
        age_counts[int(a)] += 1.0
    p_age = age_counts / float(n)

    sex_counts = np.zeros((2,), dtype=float)
    for s in sex01:
        sex_counts[int(s)] += 1.0
    p_sex = sex_counts / float(n)

    return {"p_joint": p_joint, "p_age": p_age, "p_sex": p_sex}


def _load_pums_persons(*, data_root: pathlib.Path, pums_year: int, pums_period: str, statefp: str, pumas: set[str], n_rows: int) -> Any:
    """
    Load PUMS person file (minimal columns) for external validation.
    Uses the same default path/search as tools/poc_tabddpm_pums_buildingcond.py.
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
    zip_path = next((p for p in candidates if p.exists()), candidates[0])
    if not zip_path.exists():
        raise SystemExit(f"PUMS zip not found. Tried: {candidates[0]} and {candidates[1]}")

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise SystemExit(f"No CSV members found inside: {zip_path}")
        member = sorted(members)[0]
        with zf.open(member) as f:
            df = pd.read_csv(f, nrows=int(n_rows), low_memory=False)

    cols = ["AGEP", "SEX", "PUMA"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"PUMS person file missing columns: {missing}")
    df = df[cols].copy()
    df["AGEP"] = pd.to_numeric(df["AGEP"], errors="coerce")
    df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
    df["PUMA"] = pd.to_numeric(df["PUMA"], errors="coerce")
    df = df.dropna().copy()
    df["PUMA_STR"] = df["PUMA"].astype(int).astype(str)
    df = df[df["PUMA_STR"].isin(set(map(str, pumas)))].copy()
    if df.empty:
        raise SystemExit("After filtering to study PUMAs, no PUMS rows remain.")
    return df.reset_index(drop=True)


def _pums_puma_distributions(df_pums: Any) -> dict[str, dict[str, Any]]:
    """
    Return per-PUMA distributions over:
      - p_joint (46)
      - p_age (23)
      - p_sex (2)
    """
    import numpy as np  # type: ignore
    pd = _require("pandas")

    out: dict[str, dict[str, Any]] = {}
    for puma, sub in df_pums.groupby("PUMA_STR", sort=False):
        age_idx = sub["AGEP"].apply(_age23_index)
        m = ~age_idx.isna()
        if not bool(m.any()):
            continue
        age_idx = age_idx[m].astype(int).to_numpy(dtype=int)
        sex = pd.to_numeric(sub.loc[m, "SEX"], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
        sex01 = np.where(sex == 2, 1, 0).astype(int)  # default male for unknown
        stats = _p_from_samples(age_idx=age_idx, sex01=sex01)
        out[str(puma)] = stats
    if not out:
        raise SystemExit("Failed to build PUMS per-PUMA distributions (empty after binning).")
    return out


def _build_puma_blocks(*, tiger_puma_zip: pathlib.Path, pumas: list[str]) -> list[list[str]]:
    """
    Create 4 blocks by pairing adjacent PUMAs (greedy).
    """
    gpd = _require("geopandas")

    puma_gdf = gpd.read_file(f"zip://{tiger_puma_zip}")
    if puma_gdf.crs is None:
        puma_gdf = puma_gdf.set_crs(4269, allow_override=True)
    puma_gdf = puma_gdf.to_crs(3857)

    puma_col = _pick_col(list(puma_gdf.columns), ("PUMACE20", "PUMA", "PUMACE10"))
    if puma_col is None:
        raise SystemExit(f"Cannot find PUMA code column in: {tiger_puma_zip}")

    puma_gdf[puma_col] = puma_gdf[puma_col].map(_normalize_puma)
    p = sorted(set(map(str, pumas)))
    puma_gdf = puma_gdf[puma_gdf[puma_col].astype(str).isin(p)].copy()
    if puma_gdf.empty:
        raise SystemExit("No study PUMAs found in TIGER puma zip.")

    # Build adjacency list.
    geoms = {str(r[puma_col]): r.geometry for _, r in puma_gdf.iterrows()}
    adj: dict[str, set[str]] = {k: set() for k in geoms}
    keys = list(geoms.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            ga, gb = geoms[a], geoms[b]
            try:
                touch = bool(ga.touches(gb))
            except Exception:
                touch = False
            if touch:
                adj[a].add(b)
                adj[b].add(a)

    # Greedy pairing.
    unpaired = set(keys)
    blocks: list[list[str]] = []
    while unpaired:
        a = min(unpaired, key=lambda k: (len(adj.get(k, set())), k))
        neigh = sorted([b for b in adj.get(a, set()) if b in unpaired and b != a], key=lambda k: (len(adj.get(k, set())), k))
        if neigh:
            b = neigh[0]
        else:
            # Fallback: pair with any remaining (still gives 4 folds, but not adjacency-guaranteed).
            b = sorted([k for k in unpaired if k != a])[0]
        blocks.append([a, b])
        unpaired.remove(a)
        unpaired.remove(b)

    return blocks


def main() -> None:
    np = _require("numpy")
    pd = _require("pandas")
    torch = _require("torch")

    from src.synthpop.model.diffusion_tabular import DiffusionTabularModel, TabDDPMConfig
    from src.synthpop.pipeline.detroit_v0 import make_run_id

    p = argparse.ArgumentParser(prog="poc_tabddpm_acs_supervised_b01001")
    p.add_argument("--acs_b01001_csv_gz", required=True, help="ACS B01001 tract CSV.gz (downloaded by detroit_fetch_public_data.py).")
    p.add_argument("--buildings_csv", required=True, help="Buildings CSV with tract_geoid and puma (optionally price_tier).")
    p.add_argument("--tiger_tract_zip", required=True, help="TIGER tract zip (tl_2023_26_tract.zip).")
    p.add_argument("--tiger_puma_zip", required=True, help="TIGER puma zip (tl_2023_26_puma20.zip).")
    p.add_argument("--data_root", default=None, help="Detroit data_root (only for external PUMS validation).")
    p.add_argument("--pums_year", type=int, default=2023)
    p.add_argument("--pums_period", default="5-Year")
    p.add_argument("--statefp", default="26")
    p.add_argument("--pums_n_rows", type=int, default=200_000)
    p.add_argument("--n_tiers", type=int, default=5)
    p.add_argument("--cbd_lon", type=float, default=-83.0458)
    p.add_argument("--cbd_lat", type=float, default=42.3314)

    p.add_argument("--n_pseudo_base", type=int, default=500, help="Base pseudo-individuals per tract (scaled by sqrt(pop)).")
    p.add_argument("--n_pseudo_min", type=int, default=100)
    p.add_argument("--n_pseudo_max", type=int, default=1500)
    p.add_argument("--n_eval_per_tract", type=int, default=2000)

    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=200)

    p.add_argument(
        "--conditions",
        default="none,geo-only,built-only,geo+built",
        help='Comma-separated conditions: "none", "geo-only", "built-only", "geo+built".',
    )
    p.add_argument("--fold", type=int, default=-1, help="Run a single fold index (0..3). -1 = run all folds.")
    p.add_argument("--out_dir", default=None, help="Output directory (default: outputs/<run_id>).")
    args = p.parse_args()

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))

    acs_path = pathlib.Path(args.acs_b01001_csv_gz).expanduser().resolve()
    buildings_csv = pathlib.Path(args.buildings_csv).expanduser().resolve()
    tiger_tract_zip = pathlib.Path(args.tiger_tract_zip).expanduser().resolve()
    tiger_puma_zip = pathlib.Path(args.tiger_puma_zip).expanduser().resolve()

    if args.out_dir:
        out_root = pathlib.Path(args.out_dir).expanduser().resolve()
    else:
        out_root = pathlib.Path("outputs") / make_run_id(prefix="poc_acs_supervised_b01001")
    out_root.mkdir(parents=True, exist_ok=True)

    buildings = _load_buildings(buildings_csv, n_tiers=int(args.n_tiers))
    tract_to_puma = _tract_to_puma_from_buildings(buildings)
    study_tracts = set(tract_to_puma.keys())
    study_pumas = sorted(set(tract_to_puma.values()))
    if len(study_pumas) < 2:
        raise SystemExit(f"Too few study PUMAs inferred from buildings_csv: {study_pumas}")

    # Targets from ACS.
    b01001 = _read_acs_b01001(acs_path)
    targets_by_tract = _b01001_targets_by_tract(b01001, tracts=study_tracts)

    # Context features.
    geo_ctx = _build_geo_context(tiger_tract_zip=tiger_tract_zip, tracts=study_tracts, cbd_lon=float(args.cbd_lon), cbd_lat=float(args.cbd_lat))
    built_ctx = _build_built_context(buildings, n_tiers=int(args.n_tiers))
    ctx = geo_ctx.merge(built_ctx, on="tract_geoid", how="left")
    for c in ctx.columns:
        if c == "tract_geoid":
            continue
        ctx[c] = pd.to_numeric(ctx[c], errors="coerce").fillna(0.0).astype(float)
    ctx["puma"] = ctx["tract_geoid"].map(lambda tg: tract_to_puma.get(str(tg)))
    ctx = ctx.dropna(subset=["puma"]).copy()

    # PUMA blocks for spatial holdout.
    blocks = _build_puma_blocks(tiger_puma_zip=tiger_puma_zip, pumas=study_pumas)
    blocks = [sorted(list(map(str, b))) for b in blocks]
    blocks = sorted(blocks, key=lambda b: ",".join(b))
    if len(blocks) < 2:
        raise SystemExit(f"Failed to build PUMA blocks. Blocks: {blocks}")

    # Load external PUMS (optional but recommended).
    pums_puma_dist = None
    if args.data_root:
        data_root = pathlib.Path(args.data_root).expanduser().resolve()
        df_pums = _load_pums_persons(
            data_root=data_root,
            pums_year=int(args.pums_year),
            pums_period=str(args.pums_period),
            statefp=str(args.statefp),
            pumas=set(study_pumas),
            n_rows=int(args.pums_n_rows),
        )
        pums_puma_dist = _pums_puma_distributions(df_pums)

    cond_list = [c.strip() for c in str(args.conditions).split(",") if c.strip()]
    valid_cond = {"none", "geo-only", "built-only", "geo+built"}
    for c in cond_list:
        if c not in valid_cond:
            raise SystemExit(f"Unknown condition: {c}. Valid: {sorted(valid_cond)}")

    # Baseline gap (ACS->PUMA vs PUMS), method-independent.
    baseline_gap = None
    if pums_puma_dist is not None:
        # Aggregate ACS counts to PUMA.
        acs_counts_by_puma: dict[str, Any] = {p: np.zeros((46,), dtype=float) for p in study_pumas}
        for tg, t in targets_by_tract.items():
            puma = tract_to_puma.get(str(tg))
            if not puma:
                continue
            acs_counts_by_puma[str(puma)] += np.asarray(t["counts_joint"], dtype=float)
        baseline_by_puma: dict[str, float] = {}
        for puma in study_pumas:
            ac = acs_counts_by_puma[str(puma)]
            ac_p = ac / (ac.sum() if ac.sum() > 0 else 1.0)
            pu_p = np.asarray(pums_puma_dist[str(puma)]["p_joint"], dtype=float)
            ac_age, ac_sex = _marginals_from_joint(ac_p)
            pu_age, pu_sex = _marginals_from_joint(pu_p)
            baseline_by_puma[str(puma)] = {
                "tvd_joint": float(_tvd(ac_p, pu_p)),
                "tvd_age": float(_tvd(ac_age, pu_age)),
                "tvd_sex": float(_tvd(ac_sex, pu_sex)),
            }
        vals_joint = [v["tvd_joint"] for v in baseline_by_puma.values()]
        vals_age = [v["tvd_age"] for v in baseline_by_puma.values()]
        vals_sex = [v["tvd_sex"] for v in baseline_by_puma.values()]
        baseline_gap = {
            "by_puma": baseline_by_puma,
            "summary": {
                "tvd_joint": {"mean": float(np.mean(vals_joint)), "max": float(np.max(vals_joint))} if vals_joint else None,
                "tvd_age": {"mean": float(np.mean(vals_age)), "max": float(np.max(vals_age))} if vals_age else None,
                "tvd_sex": {"mean": float(np.mean(vals_sex)), "max": float(np.max(vals_sex))} if vals_sex else None,
            },
        }
        _write_json(out_root / "metrics" / "acs_pums_baseline_gap.json", baseline_gap)

    # Run folds (or a single fold).
    fold_indices = list(range(len(blocks)))
    if int(args.fold) >= 0:
        if int(args.fold) >= len(blocks):
            raise SystemExit(f"--fold out of range: {args.fold} (n_folds={len(blocks)})")
        fold_indices = [int(args.fold)]

    run_meta = {
        "out_root": str(out_root),
        "acs_b01001_csv_gz": str(acs_path),
        "buildings_csv": str(buildings_csv),
        "tiger_tract_zip": str(tiger_tract_zip),
        "tiger_puma_zip": str(tiger_puma_zip),
        "study_pumas": study_pumas,
        "n_tracts": int(len(set(ctx["tract_geoid"].astype(str).tolist()))),
        "puma_blocks": blocks,
        "conditions": cond_list,
        "n_pseudo_base": int(args.n_pseudo_base),
        "n_eval_per_tract": int(args.n_eval_per_tract),
        "timesteps": int(args.timesteps),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "baseline_gap": baseline_gap,
        "external_validation": {"enabled": bool(pums_puma_dist is not None), "pums_year": int(args.pums_year), "pums_period": str(args.pums_period)},
    }
    _write_json(out_root / "run_summary.json", run_meta)

    geo_cols = ["centroid_lon", "centroid_lat", "area_km2", "dist_cbd_km"]
    built_cols = [
        "n_buildings_log",
        "cap_proxy_sum_log",
        "height_mean",
        "footprint_mean_log",
    ] + [f"price_tier_p{k}" for k in range(1, int(args.n_tiers) + 1)]

    # Helper: build a tract->cond vector dict for a given fold+condition.
    def _cond_for_fold(condition: str, train_tracts: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
        if condition == "none":
            return {}, {"cond_dim": 0, "cols": []}
        if condition == "geo-only":
            cols = geo_cols
        elif condition == "built-only":
            cols = built_cols
        else:
            cols = geo_cols + built_cols

        sub = ctx[ctx["tract_geoid"].astype(str).isin(sorted(train_tracts))].copy()
        sub_train, mu, sd = _standardize(sub, cols=cols)
        # Apply the same scaler to all tracts (train+test).
        full, _, _ = _standardize(ctx, cols=cols, mean=mu, std=sd)
        full = full.set_index("tract_geoid", drop=False)
        out_map = {str(tg): full.loc[str(tg), cols].to_numpy(dtype=float) for tg in full.index.astype(str).tolist()}
        return out_map, {"cond_dim": int(len(cols)), "cols": cols, "mean": mu, "std": sd}

    # Collect fold-level summaries for ablation report.
    ablation_internal: dict[str, dict[int, Any]] = {c: {} for c in cond_list}
    ablation_external: dict[str, dict[int, Any]] = {c: {} for c in cond_list}

    # Main loop.
    for fold_idx in fold_indices:
        test_pumas = set(blocks[int(fold_idx)])
        train_pumas = set(study_pumas) - set(test_pumas)
        train_tracts = set(ctx[ctx["puma"].astype(str).isin(sorted(train_pumas))]["tract_geoid"].astype(str).tolist())
        test_tracts = set(ctx[ctx["puma"].astype(str).isin(sorted(test_pumas))]["tract_geoid"].astype(str).tolist())
        if not train_tracts or not test_tracts:
            raise SystemExit(f"Empty train/test tracts in fold={fold_idx}. train={len(train_tracts)} test={len(test_tracts)}")

        for condition in cond_list:
            fold_dir = out_root / f"fold_{fold_idx}" / condition
            fold_dir.mkdir(parents=True, exist_ok=True)

            tract_cond, scaler = _cond_for_fold(condition, train_tracts=train_tracts)
            cond_dim = int(scaler["cond_dim"])

            # Build training dataset (pseudo individuals).
            xs = []
            cs = []
            pops = []
            weights = []
            for tg in sorted(train_tracts):
                t = targets_by_tract.get(str(tg))
                if t is None:
                    continue
                total = float(t["total_pop"])
                w = math.sqrt(max(1.0, total))
                pops.append(total)
                weights.append(w)
            w_mean = float(np.mean(weights)) if weights else 1.0

            for tg in sorted(train_tracts):
                t = targets_by_tract.get(str(tg))
                if t is None:
                    continue
                total = float(t["total_pop"])
                w = math.sqrt(max(1.0, total))
                n_i = int(round(float(args.n_pseudo_base) * (w / w_mean)))
                n_i = max(int(args.n_pseudo_min), min(int(args.n_pseudo_max), n_i))
                age_idx, sex01 = _sample_pseudo(rng=rng, p_joint=t["p_joint"], n=n_i)

                # Encode to [0,1] floats.
                age_u = age_idx.astype(float) / 22.0
                sex_u = sex01.astype(float)
                x_u = np.stack([age_u, sex_u], axis=1).astype(np.float32)

                if cond_dim > 0:
                    c = tract_cond.get(str(tg))
                    if c is None:
                        continue
                    c = np.asarray(c, dtype=np.float32)
                    c_rep = np.repeat(c.reshape(1, -1), repeats=int(n_i), axis=0)
                    cs.append(c_rep)
                xs.append(x_u)

            if not xs:
                raise SystemExit(f"No training samples constructed for fold={fold_idx}, condition={condition}.")
            x_u_all = np.concatenate(xs, axis=0).astype(np.float32)
            cond_all = np.concatenate(cs, axis=0).astype(np.float32) if cond_dim > 0 else None

            # Standardize x_u (train-only).
            x_mean = x_u_all.mean(axis=0).astype(np.float32)
            x_std = x_u_all.std(axis=0).astype(np.float32)
            x_std = np.where(x_std <= 1e-6, 1.0, x_std).astype(np.float32)
            x_z = ((x_u_all - x_mean) / x_std).astype(np.float32)

            x = torch.from_numpy(x_z)
            cond = torch.from_numpy(cond_all) if cond_all is not None else None

            cfg = TabDDPMConfig(timesteps=int(args.timesteps))
            model = DiffusionTabularModel(input_dim=int(x.shape[1]), cond_dim=int(cond.shape[1]) if cond is not None else 0, seed=int(args.seed), config=cfg)

            train_metrics = model.fit(
                x=x,
                cond=cond,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                device=args.device,
                log_every=int(args.log_every),
            )
            ckpt = fold_dir / "model.pt"
            model.save(ckpt)

            train_summary = {
                "fold": int(fold_idx),
                "condition": condition,
                "train_pumas": sorted(train_pumas),
                "test_pumas": sorted(test_pumas),
                "n_train_tracts": int(len(train_tracts)),
                "n_test_tracts": int(len(test_tracts)),
                "n_train_samples": int(x.shape[0]),
                "cond_dim": cond_dim,
                "cond_cols": scaler.get("cols", []),
                "x_mean": [float(v) for v in x_mean.tolist()],
                "x_std": [float(v) for v in x_std.tolist()],
                "train_metrics": train_metrics,
                "ckpt": str(ckpt),
            }
            _write_json(fold_dir / "train_summary.json", train_summary)

            # --- Internal evaluation vs ACS on held-out tracts ---
            internal_by_tract: dict[str, Any] = {}
            for tg in sorted(test_tracts):
                t = targets_by_tract.get(str(tg))
                if t is None:
                    continue
                n_eval = int(args.n_eval_per_tract)
                if cond_dim > 0:
                    c = tract_cond.get(str(tg))
                    if c is None:
                        continue
                    c = np.asarray(c, dtype=np.float32)
                    c_rep = np.repeat(c.reshape(1, -1), repeats=n_eval, axis=0)
                    c_t = torch.from_numpy(c_rep)
                else:
                    c_t = None
                z = model.sample(n=n_eval, cond=c_t, device=args.device).to_numpy(dtype=np.float32)
                x_u = (z * x_std.reshape(1, -1) + x_mean.reshape(1, -1)).astype(np.float32)
                age_idx, sex01 = _decode_samples(x_u)
                phat = _p_from_samples(age_idx=age_idx, sex01=sex01)
                tvd_joint = _tvd(phat["p_joint"], t["p_joint"])
                tvd_age = _tvd(phat["p_age"], t["p_age"])
                tvd_sex = _tvd(phat["p_sex"], t["p_sex"])
                internal_by_tract[str(tg)] = {"tvd_joint": float(tvd_joint), "tvd_age": float(tvd_age), "tvd_sex": float(tvd_sex)}

            vals_joint = [v["tvd_joint"] for v in internal_by_tract.values()]
            vals_age = [v["tvd_age"] for v in internal_by_tract.values()]
            vals_sex = [v["tvd_sex"] for v in internal_by_tract.values()]
            internal = {
                "by_tract": internal_by_tract,
                "summary": {
                    "tvd_joint": {"mean": float(np.mean(vals_joint)), "max": float(np.max(vals_joint)), "p90": float(np.quantile(vals_joint, 0.9))},
                    "tvd_age": {"mean": float(np.mean(vals_age)), "max": float(np.max(vals_age)), "p90": float(np.quantile(vals_age, 0.9))},
                    "tvd_sex": {"mean": float(np.mean(vals_sex)), "max": float(np.max(vals_sex)), "p90": float(np.quantile(vals_sex, 0.9))},
                },
            }
            _write_json(fold_dir / "metrics" / "internal_acs_holdout.json", internal)
            ablation_internal[condition][int(fold_idx)] = dict(internal.get("summary", {}))

            # Worst tracts diagnosis (top 10 by joint TVD).
            worst = sorted(internal_by_tract.items(), key=lambda kv: kv[1]["tvd_joint"], reverse=True)[:10]
            _write_json(
                fold_dir / "metrics" / "worst_tracts_diagnosis.json",
                {"worst_tracts": [{"tract_geoid": k, **v} for k, v in worst]},
            )

            # --- External evaluation vs PUMS at PUMA level ---
            if pums_puma_dist is not None:
                # First estimate p_hat for each tract (reuse internal estimates if tract in test; otherwise sample now).
                p_hat_by_tract: dict[str, Any] = {}
                for tg in sorted(set(ctx["tract_geoid"].astype(str).tolist())):
                    t = targets_by_tract.get(str(tg))
                    if t is None:
                        continue
                    n_eval = int(args.n_eval_per_tract)
                    if cond_dim > 0:
                        c = tract_cond.get(str(tg))
                        if c is None:
                            continue
                        c = np.asarray(c, dtype=np.float32)
                        c_rep = np.repeat(c.reshape(1, -1), repeats=n_eval, axis=0)
                        c_t = torch.from_numpy(c_rep)
                    else:
                        c_t = None
                    z = model.sample(n=n_eval, cond=c_t, device=args.device).to_numpy(dtype=np.float32)
                    x_u = (z * x_std.reshape(1, -1) + x_mean.reshape(1, -1)).astype(np.float32)
                    age_idx, sex01 = _decode_samples(x_u)
                    phat = _p_from_samples(age_idx=age_idx, sex01=sex01)
                    p_hat_by_tract[str(tg)] = phat

                # Aggregate to PUMA using ACS tract populations.
                counts_hat_by_puma: dict[str, Any] = {p: np.zeros((46,), dtype=float) for p in study_pumas}
                for tg, phat in p_hat_by_tract.items():
                    puma = tract_to_puma.get(str(tg))
                    t = targets_by_tract.get(str(tg))
                    if not puma or t is None:
                        continue
                    pop = float(t["total_pop"])
                    counts_hat_by_puma[str(puma)] += float(pop) * np.asarray(phat["p_joint"], dtype=float)

                external_by_puma: dict[str, Any] = {}
                for puma in study_pumas:
                    hat = counts_hat_by_puma[str(puma)]
                    hat_p = hat / (hat.sum() if hat.sum() > 0 else 1.0)
                    ref_p = np.asarray(pums_puma_dist[str(puma)]["p_joint"], dtype=float)
                    hat_age, hat_sex = _marginals_from_joint(hat_p)
                    ref_age, ref_sex = _marginals_from_joint(ref_p)
                    external_by_puma[str(puma)] = {
                        "tvd_joint": float(_tvd(hat_p, ref_p)),
                        "tvd_age": float(_tvd(hat_age, ref_age)),
                        "tvd_sex": float(_tvd(hat_sex, ref_sex)),
                    }

                vals_joint = [v["tvd_joint"] for v in external_by_puma.values()]
                vals_age = [v["tvd_age"] for v in external_by_puma.values()]
                vals_sex = [v["tvd_sex"] for v in external_by_puma.values()]
                external = {
                    "by_puma": external_by_puma,
                    "summary": {
                        "tvd_joint": {"mean": float(np.mean(vals_joint)), "max": float(np.max(vals_joint))} if vals_joint else None,
                        "tvd_age": {"mean": float(np.mean(vals_age)), "max": float(np.max(vals_age))} if vals_age else None,
                        "tvd_sex": {"mean": float(np.mean(vals_sex)), "max": float(np.max(vals_sex))} if vals_sex else None,
                    },
                }
                _write_json(fold_dir / "metrics" / "external_pums_by_puma.json", external)
                ablation_external[condition][int(fold_idx)] = dict(external.get("summary", {}))

    # --- Write ablation summary (mean±std across folds) ---
    def _mean_std(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    def _summarize_across_folds(per_fold: dict[int, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"by_fold": {str(k): v for k, v in sorted(per_fold.items())}}
        for metric in ["tvd_joint", "tvd_age", "tvd_sex"]:
            mean_vals = [float(v.get(metric, {}).get("mean")) for v in per_fold.values() if v.get(metric) and v[metric].get("mean") is not None]
            max_vals = [float(v.get(metric, {}).get("max")) for v in per_fold.values() if v.get(metric) and v[metric].get("max") is not None]
            p90_vals = [float(v.get(metric, {}).get("p90")) for v in per_fold.values() if v.get(metric) and v[metric].get("p90") is not None]
            out[metric] = {
                "mean": _mean_std(mean_vals),
                "max": _mean_std(max_vals),
            }
            if p90_vals:
                out[metric]["p90"] = _mean_std(p90_vals)
        return out

    ablation_summary: dict[str, Any] = {
        "folds": [int(i) for i in fold_indices],
        "conditions": cond_list,
        "internal_acs": {c: _summarize_across_folds(ablation_internal.get(c, {})) for c in cond_list},
        "external_pums": {c: _summarize_across_folds(ablation_external.get(c, {})) for c in cond_list},
        "baseline_gap": baseline_gap,
    }
    _write_json(out_root / "metrics" / "ablation_summary.json", ablation_summary)

    print(f"[ok] wrote: {out_root}")


if __name__ == "__main__":
    main()
