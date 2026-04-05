from __future__ import annotations

"""
SafeGraph POI processing.

SafeGraph is license-sensitive; we only register existing local shards and extract Detroit subsets
into processed outputs (which may still be restricted depending on agreement).
"""

import ast
import json
import pathlib
import re
from collections import defaultdict
from typing import Any


def extract_detroit_pois(*, safegraph_unzip_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    raise NotImplementedError("TODO(v0): filter POIs to Detroit study area and write processed POI table.")


def _require_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("poi_safegraph.py requires pandas.") from e
    return pd


def _sanitize_col_token(value: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    token = token.strip("_")
    return token or "unknown"


def _collapse_home_geoid(value: Any, *, group_level: str) -> str:
    digits = "".join(ch for ch in str(value).strip() if ch.isdigit())
    if group_level == "cbg":
        return digits[:12] if len(digits) >= 12 else ""
    if group_level == "tract":
        return digits[:11] if len(digits) >= 11 else ""
    if group_level == "county":
        return digits[:5] if len(digits) >= 5 else ""
    raise ValueError(f"Unsupported group_level: {group_level}")


def parse_safegraph_count_map(value: Any) -> dict[str, float]:
    """
    Parse SafeGraph-style JSON dict fields such as `visitor_home_cbgs`.
    """
    if value is None:
        return {}
    text = str(value).strip()
    if not text or text in {"nan", "None", "null", "{}", "[]"}:
        return {}

    obj: Any
    try:
        obj = json.loads(text)
    except Exception:
        try:
            obj = ast.literal_eval(text)
        except Exception:
            return {}

    if not isinstance(obj, dict):
        return {}

    out: dict[str, float] = {}
    for k, v in obj.items():
        key = "".join(ch for ch in str(k).strip() if ch.isdigit())
        if not key:
            continue
        try:
            val = float(v)
        except Exception:
            continue
        if val > 0.0:
            out[key] = out.get(key, 0.0) + val
    return out


def aggregate_home_origin_profiles(
    *,
    merged_poi: Any,
    group_level: str = "tract",
    region_filter: str | None = None,
    region_col: str = "region",
    visitor_home_col: str = "visitor_home_cbgs",
    category_col: str | None = "top_category",
    top_n_categories: int = 24,
    min_category_weight: float = 0.0,
    chunk_size: int = 50_000,
) -> Any:
    """
    Aggregate SafeGraph `visitor_home_cbgs` into home-origin profiles.

    Returns a DataFrame with:
    - `<group_level>_geoid`
    - `home_origin_count`
    - `home_origin_share`
    - optional category-share columns `cat__*`

    This is intentionally light-weight: it extracts residential-origin mass and
    a simple POI-category portrait for each home geography.
    """
    pd = _require_pandas()

    group_col = {
        "cbg": "cbg_geoid",
        "tract": "tract_geoid",
        "county": "county_geoid",
    }.get(str(group_level))
    if group_col is None:
        raise ValueError(f"group_level must be one of: cbg, tract, county. Got: {group_level}")

    use_category = category_col is not None and str(category_col).strip() != ""
    group_mass: dict[str, float] = defaultdict(float)
    category_mass: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def _consume_chunk(df: Any) -> None:
        if df.empty:
            return
        work = df
        if region_filter is not None:
            work = work[work[region_col].astype(str) == str(region_filter)].copy()
            if work.empty:
                return
        for row in work.itertuples(index=False):
            cbg_map = parse_safegraph_count_map(getattr(row, visitor_home_col))
            if not cbg_map:
                continue
            cat_val = None
            if use_category:
                raw_cat = getattr(row, str(category_col))
                cat_val = str(raw_cat).strip() if raw_cat is not None else ""
                if not cat_val or cat_val in {"nan", "None"}:
                    cat_val = "unknown"
            for raw_geoid, weight in cbg_map.items():
                gid = _collapse_home_geoid(raw_geoid, group_level=str(group_level))
                if not gid:
                    continue
                group_mass[gid] += float(weight)
                if cat_val is not None:
                    category_mass[gid][str(cat_val)] += float(weight)

    if isinstance(merged_poi, pd.DataFrame):
        need = [visitor_home_col]
        if region_filter is not None:
            need.append(region_col)
        if use_category:
            need.append(str(category_col))
        miss = [c for c in need if c not in merged_poi.columns]
        if miss:
            raise ValueError(f"merged_poi missing columns: {miss}")
        _consume_chunk(merged_poi[need].copy())
    else:
        path = pathlib.Path(merged_poi).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(str(path))
        usecols = [visitor_home_col]
        if region_filter is not None:
            usecols.append(region_col)
        if use_category:
            usecols.append(str(category_col))
        for chunk in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=int(chunk_size)):
            _consume_chunk(chunk)

    if not group_mass:
        cols = [group_col, "home_origin_count", "home_origin_share"]
        return pd.DataFrame(columns=cols)

    total_mass = float(sum(group_mass.values()))
    categories_ranked: list[str] = []
    if use_category:
        cat_totals: dict[str, float] = defaultdict(float)
        for gdict in category_mass.values():
            for cat, weight in gdict.items():
                cat_totals[str(cat)] += float(weight)
        ranked = sorted(cat_totals.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
        ranked = [kv for kv in ranked if float(kv[1]) >= float(min_category_weight)]
        categories_ranked = [str(cat) for cat, _ in ranked[: max(int(top_n_categories), 0)]]

    rows: list[dict[str, Any]] = []
    for gid in sorted(group_mass):
        mass = float(group_mass[gid])
        row: dict[str, Any] = {
            group_col: str(gid),
            "home_origin_count": mass,
            "home_origin_share": (mass / total_mass) if total_mass > 0 else 0.0,
        }
        if categories_ranked:
            denom = max(float(sum(category_mass[gid].values())), 1e-12)
            for cat in categories_ranked:
                col = f"cat__{_sanitize_col_token(cat)}"
                row[col] = float(category_mass[gid].get(cat, 0.0)) / denom
        rows.append(row)

    out = pd.DataFrame(rows)
    for col in out.columns:
        if col == group_col:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)
    return out
