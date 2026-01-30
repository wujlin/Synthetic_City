from __future__ import annotations

"""
Veraset mobility data helpers (Scheme C-v2).

This module intentionally focuses on:
- robust IO (CSV/CSV.GZ/Parquet, file or directory)
- lightweight feature extraction that does not assume lat/lon availability

Source schema references:
- `docs/deway_data/home.md`
- `docs/deway_data/visit.md`
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("veraset.py requires pandas.") from e
    return pd


def _iter_data_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(str(path))
    files: list[Path] = []
    for ext in ("*.csv", "*.csv.gz", "*.parquet"):
        files.extend(sorted(path.glob(ext)))
    return files


def _read_any(path: Path, *, columns: list[str] | None = None) -> Any:
    pd = _require_pandas()
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix == ".gz" and path.name.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip", usecols=columns)
    if path.suffix == ".csv":
        return pd.read_csv(path, usecols=columns)
    raise ValueError(f"Unsupported file: {path}")


def _normalize_geohash_col(df: Any) -> Any:
    pd = _require_pandas()
    if "GEOHASH_5" in df.columns and "GEO_HASH5" in df.columns:
        # Prefer GEOHASH_5 if both exist; drop the other to avoid ambiguity.
        df = df.drop(columns=["GEO_HASH5"])
    if "GEO_HASH5" in df.columns and "GEOHASH_5" not in df.columns:
        df = df.rename(columns={"GEO_HASH5": "GEOHASH_5"})
    if "GEOHASH_5" in df.columns:
        df["GEOHASH_5"] = df["GEOHASH_5"].astype(str)
    if "CENSUS_BLOCK_GROUP" in df.columns:
        df["CENSUS_BLOCK_GROUP"] = df["CENSUS_BLOCK_GROUP"].astype(str)
    if "CAID" in df.columns:
        df["CAID"] = df["CAID"].astype(str)
    return df


def load_veraset_home(path: str | Path) -> Any:
    """
    Load device -> home location mapping.
    Required columns (minimum): CAID, CENSUS_BLOCK_GROUP, GEOHASH_5(or GEO_HASH5).
    """
    pd = _require_pandas()
    p = Path(path)
    files = _iter_data_files(p)
    if not files:
        raise FileNotFoundError(f"No data files found under: {p}")

    frames = []
    for f in files:
        df = _read_any(f, columns=None)
        df = _normalize_geohash_col(df)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


def load_veraset_visits(path: str | Path) -> Any:
    """
    Load POI visits data (event-level).
    Required columns (minimum): CAID, CENSUS_BLOCK_GROUP, GEOHASH_5(or GEO_HASH5), UTC_TIMESTAMP or LOCAL_TIMESTAMP.
    """
    pd = _require_pandas()
    p = Path(path)
    files = _iter_data_files(p)
    if not files:
        raise FileNotFoundError(f"No data files found under: {p}")

    frames = []
    for f in files:
        df = _read_any(f, columns=None)
        df = _normalize_geohash_col(df)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


@dataclass(frozen=True)
class DeviceFeatureSpec:
    category_col: str = "TOP_CATEGORY"
    ts_col_candidates: tuple[str, ...] = ("LOCAL_TIMESTAMP", "UTC_TIMESTAMP")
    geohash_col: str = "GEOHASH_5"
    cbg_col: str = "CENSUS_BLOCK_GROUP"
    device_id_col: str = "CAID"
    max_top_categories: int = 20
    category_prefix: str = "cat__"


def compute_device_features(visits: Any, *, spec: DeviceFeatureSpec | None = None) -> Any:
    """
    Compute simple per-device behavior features from visits.

    Output columns (v0, minimal & extensible):
    - n_visits
    - unique_cbg_count
    - unique_geohash5_count
    - unique_category_count
    - weekend_ratio (if timestamp parsable)
    - evening_ratio (if timestamp parsable)
    - optional dwell stats (if MINIMUM_DWELL exists)
    - top-category ratios (top-K over the input batch)
    """
    pd = _require_pandas()
    s = spec or DeviceFeatureSpec()

    if not isinstance(visits, pd.DataFrame):
        raise TypeError("visits must be a pandas DataFrame")
    if s.device_id_col not in visits.columns:
        raise ValueError(f"visits missing device id col: {s.device_id_col}")

    v = visits.copy()
    v = _normalize_geohash_col(v)

    for optional in [s.cbg_col, s.geohash_col, s.category_col]:
        if optional not in v.columns:
            v[optional] = None

    ts_col = None
    for cand in s.ts_col_candidates:
        if cand in v.columns:
            ts_col = cand
            break

    if ts_col is not None:
        # LOCAL_TIMESTAMP is often string; UTC_TIMESTAMP may be int seconds.
        if ts_col == "UTC_TIMESTAMP":
            v["_dt"] = pd.to_datetime(v[ts_col], unit="s", utc=True, errors="coerce")
        else:
            v["_dt"] = pd.to_datetime(v[ts_col], errors="coerce")
        v["_weekday"] = v["_dt"].dt.dayofweek
        v["_hour"] = v["_dt"].dt.hour
        v["_is_weekend"] = v["_weekday"].isin([5, 6])
        v["_is_evening"] = v["_hour"].isin(list(range(18, 24)))
    else:
        v["_is_weekend"] = False
        v["_is_evening"] = False

    g = v.groupby(s.device_id_col, sort=False)
    size = g.size()
    out = pd.DataFrame(
        {
            s.device_id_col: size.index.astype(str),
            "n_visits": size.to_numpy(),
            "unique_cbg_count": g[s.cbg_col].nunique(dropna=True).to_numpy(),
            "unique_geohash5_count": g[s.geohash_col].nunique(dropna=True).to_numpy(),
            "unique_category_count": g[s.category_col].nunique(dropna=True).to_numpy(),
            "weekend_ratio": g["_is_weekend"].mean().to_numpy(dtype=float),
            "evening_ratio": g["_is_evening"].mean().to_numpy(dtype=float),
        }
    )

    if "MINIMUM_DWELL" in v.columns:
        dwell = pd.to_numeric(v["MINIMUM_DWELL"], errors="coerce")
        v["_dwell"] = dwell
        gd = v.groupby(s.device_id_col, sort=False)["_dwell"]
        out["mean_dwell_min"] = gd.mean().to_numpy(dtype=float)
        out["median_dwell_min"] = gd.median().to_numpy(dtype=float)

    # Add per-device top-category ratios (top-K categories globally in this visits batch).
    max_k = int(s.max_top_categories)
    if max_k > 0 and s.category_col in v.columns:
        cats = v[[s.device_id_col, s.category_col]].dropna()
        if len(cats) > 0:
            top = cats[s.category_col].value_counts().head(max_k).index.tolist()
            cats["_cat"] = cats[s.category_col].where(cats[s.category_col].isin(top), other="__OTHER")
            counts = pd.crosstab(cats[s.device_id_col], cats["_cat"]).astype(float)
            probs = counts.div(counts.sum(axis=1).replace(0.0, 1.0), axis=0)

            def _safe_name(token: Any) -> str:
                t = str(token).strip()
                t = t.replace(" ", "_").replace("/", "_").replace("-", "_").replace("__", "_")
                return t[:64] if t else "EMPTY"

            probs = probs.rename(columns={c: f"{s.category_prefix}{_safe_name(c)}" for c in probs.columns})
            probs = probs.reset_index().rename(columns={s.device_id_col: s.device_id_col})
            out = out.merge(probs, on=s.device_id_col, how="left")
            for col in probs.columns:
                if col == s.device_id_col:
                    continue
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)
    return out


def compute_activity_center(visits: Any, *, spec: DeviceFeatureSpec | None = None, time_filter: str = "evening") -> Any:
    """
    Compute a per-device "activity center".

    v0 (no lat/lon): use the modal GEOHASH_5 within a time window as activity center.
    Returns a DataFrame: [CAID, GEOHASH_5] (and CENSUS_BLOCK_GROUP if available).
    """
    pd = _require_pandas()
    s = spec or DeviceFeatureSpec()
    if not isinstance(visits, pd.DataFrame):
        raise TypeError("visits must be a pandas DataFrame")

    v = visits.copy()
    v = _normalize_geohash_col(v)
    if s.device_id_col not in v.columns:
        raise ValueError(f"visits missing device id col: {s.device_id_col}")
    if s.geohash_col not in v.columns:
        raise ValueError(f"visits missing geohash col: {s.geohash_col}")

    if time_filter not in ("all", "evening"):
        raise ValueError("time_filter must be one of: all, evening")

    ts_col = None
    for cand in s.ts_col_candidates:
        if cand in v.columns:
            ts_col = cand
            break

    if ts_col is not None and time_filter == "evening":
        if ts_col == "UTC_TIMESTAMP":
            v["_dt"] = pd.to_datetime(v[ts_col], unit="s", utc=True, errors="coerce")
        else:
            v["_dt"] = pd.to_datetime(v[ts_col], errors="coerce")
        v["_hour"] = v["_dt"].dt.hour
        v = v[v["_hour"].isin(list(range(18, 24)))]

    def _mode(series: Any) -> Any:
        vc = series.value_counts(dropna=True)
        if len(vc) == 0:
            return None
        return vc.index[0]

    agg: dict[str, Any] = {s.geohash_col: _mode}
    if s.cbg_col in v.columns:
        agg[s.cbg_col] = _mode
    out = v.groupby(s.device_id_col, sort=False).agg(agg).reset_index()
    return out
