from __future__ import annotations

"""
Condition vectors for the diffusion-based hierarchical generation.

Design intent:
- Region conditions summarize macro constraints and built-environment context.
- Building conditions summarize local context (area/height/landuse/POI neighborhood).

v0 (Detroit) encoding policy (PI review aligned):
- Do NOT treat `building_id` as a categorical variable in the diffusion model (too many categories).
- Scheme B (current PI decision): separate attribute generation and spatial allocation.
  - The diffusion model learns attribute structure under macro geography:
    P(attrs | PUMA/tract_context) from PUMS-only (no synthetic person-building pairing in training).
  - Spatial anchoring is handled by an explicit, reviewable allocator (post-processing):
    f(attrs, group) -> building, supporting multiple strategies for ablation/sensitivity.

Scheme C-v2 (PI proposal):
- Move toward a *data-driven joint* approach by learning a shared latent space:
  z_person, z_device, z_building aligned via contrastive/distribution/spatial losses.
- Condition vectors expand to tract/device summaries, serving as conditioning inputs to latent diffusion.
"""

import pathlib
from typing import Any


def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("condition_vectors.py requires pandas.") from e
    return pd


def _read_df(path: pathlib.Path) -> Any:
    pd = _require_pandas()
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_df(df: Any, out_path: pathlib.Path) -> None:
    pd = _require_pandas()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
        return
    if out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
        return
    raise ValueError(f"Unsupported out_path suffix: {out_path}")


def build_tract_conditions(
    *,
    geo_units: Any,
    marginals: Any,
    poi_summary: Any | None = None,
    group_col: str = "tract_geoid",
    variable_col: str = "variable",
    category_col: str = "category",
    target_col: str = "target",
    prefix: str = "marg__",
    normalize: bool = True,
) -> Any:
    """
    Build tract-level condition vectors from long-format marginals + optional POI/built summaries.

    Expected marginals schema (aligned with tools/build_acs_marginals_long.py):
    - group_col (e.g., tract_geoid / puma)
    - variable (e.g., AGEP_bin, SEX)
    - category (string label)
    - target (count)
    """
    pd = _require_pandas()
    if not isinstance(geo_units, pd.DataFrame):
        raise TypeError("geo_units must be a pandas DataFrame")
    if not isinstance(marginals, pd.DataFrame):
        raise TypeError("marginals must be a pandas DataFrame")
    if group_col not in geo_units.columns:
        raise ValueError(f"geo_units missing group_col: {group_col}")
    for c in [group_col, variable_col, category_col, target_col]:
        if c not in marginals.columns:
            raise ValueError(f"marginals missing column: {c}")

    out = geo_units[[group_col]].drop_duplicates().copy()
    out[group_col] = out[group_col].astype(str)

    m = marginals[[group_col, variable_col, category_col, target_col]].copy()
    m[group_col] = m[group_col].astype(str)
    m[variable_col] = m[variable_col].astype(str)
    m[category_col] = m[category_col].astype(str)
    m[target_col] = pd.to_numeric(m[target_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    if normalize:
        denom = m.groupby([group_col, variable_col], sort=False)[target_col].transform("sum").replace(0.0, 1.0)
        m["_value"] = m[target_col] / denom
    else:
        m["_value"] = m[target_col]

    wide = m.pivot_table(index=group_col, columns=[variable_col, category_col], values="_value", fill_value=0.0)
    wide.columns = [f"{prefix}{v}__{cat}" for (v, cat) in wide.columns.to_list()]
    wide = wide.reset_index()

    out = out.merge(wide, on=group_col, how="left")
    for c in out.columns:
        if c == group_col:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)

    if poi_summary is not None:
        if not isinstance(poi_summary, pd.DataFrame):
            raise TypeError("poi_summary must be a pandas DataFrame when provided")
        if group_col not in poi_summary.columns:
            raise ValueError(f"poi_summary missing group_col: {group_col}")
        ps = poi_summary.copy()
        ps[group_col] = ps[group_col].astype(str)
        keep = [c for c in ps.columns if c == group_col or pd.api.types.is_numeric_dtype(ps[c])]
        ps = ps[keep]
        out = out.merge(ps, on=group_col, how="left")
        for c in out.columns:
            if c == group_col:
                continue
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)

    return out


def build_device_features(*, veraset_visits: Any, device_home: Any) -> Any:
    """
    Build device-level feature vectors from Veraset visits + home mapping.

    v0 behavior:
    - uses src.synthpop.data.veraset.compute_device_features / compute_activity_center
    - attaches device home CBG/geohash if present
    """
    pd = _require_pandas()
    if not isinstance(veraset_visits, pd.DataFrame) or not isinstance(device_home, pd.DataFrame):
        raise TypeError("veraset_visits/device_home must be pandas DataFrames")

    from ..data.veraset import compute_activity_center, compute_device_features

    feats = compute_device_features(veraset_visits)
    center = compute_activity_center(veraset_visits)

    home_cols = [c for c in ["CAID", "CENSUS_BLOCK_GROUP", "GEOHASH_5"] if c in device_home.columns]
    home = device_home[home_cols].drop_duplicates(subset=["CAID"]).copy() if "CAID" in home_cols else None

    out = feats.merge(center, on="CAID", how="left", suffixes=("", "_activity"))
    if home is not None:
        out = out.merge(home, on="CAID", how="left", suffixes=("", "_home"))
    return out


def build_region_conditions(*, geo_units_path: pathlib.Path, marginals_path: pathlib.Path, out_path: pathlib.Path) -> None:
    geo = _read_df(geo_units_path)
    m = _read_df(marginals_path)
    cond = build_tract_conditions(geo_units=geo, marginals=m)
    _write_df(cond, out_path)


def build_building_conditions(*, buildings_path: pathlib.Path, poi_path: pathlib.Path | None, out_path: pathlib.Path) -> None:
    """
    Minimal building condition builder.

    v0 behavior: keep numeric building attributes (and optionally merge POI summaries if a shared key exists).
    """
    pd = _require_pandas()
    b = _read_df(buildings_path)
    if not isinstance(b, pd.DataFrame):
        raise TypeError("buildings_path must resolve to a DataFrame")

    keep = [c for c in b.columns if pd.api.types.is_numeric_dtype(b[c]) or c in ("bldg_id", "bg_geoid", "tract_geoid", "puma")]
    out = b[keep].copy()

    if poi_path is not None:
        poi = _read_df(poi_path)
        if isinstance(poi, pd.DataFrame):
            # Merge on the first common geo key if exists.
            for key in ("bldg_id", "bg_geoid", "tract_geoid"):
                if key in out.columns and key in poi.columns:
                    poi_keep = [c for c in poi.columns if c == key or pd.api.types.is_numeric_dtype(poi[c])]
                    out = out.merge(poi[poi_keep], on=key, how="left")
                    break

    for c in out.columns:
        if c in ("bldg_id", "bg_geoid", "tract_geoid", "puma"):
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)

    _write_df(out, out_path)
