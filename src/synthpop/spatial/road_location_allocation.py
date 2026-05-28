from __future__ import annotations

"""
Explicit small-area spatial allocation on road-constrained supports.

Design goal:
- accept tract/CBG-assigned synthetic population
- build home/work candidate points from road-network support sets
- assign home/work locations explicitly without mixing in extra behavioral priors
"""

import multiprocessing as mp
from typing import Any, Iterable


_ROAD_CANDIDATE_WORKER_STATE: dict[str, Any] = {}


def _require_geo_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import geopandas as gpd  # type: ignore
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        from shapely.geometry import LineString, LinearRing, MultiLineString, Point  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("road_location_allocation requires geopandas, pandas, numpy, shapely.") from e
    return gpd, np, pd, (LineString, LinearRing, MultiLineString, Point)


def _iter_line_parts(geom: Any, *, line_types: tuple[Any, Any, Any, Any]) -> Iterable[Any]:
    LineString, LinearRing, MultiLineString, _ = line_types
    if geom is None or getattr(geom, "is_empty", True):
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, LinearRing):
        return [LineString(geom)]
    if isinstance(geom, MultiLineString):
        out: list[Any] = []
        for part in geom.geoms:
            out.extend(list(_iter_line_parts(part, line_types=line_types)))
        return out
    if getattr(geom, "geom_type", "") == "GeometryCollection":
        out = []
        for part in geom.geoms:
            out.extend(list(_iter_line_parts(part, line_types=line_types)))
        return out
    return []


def _hash_line_coords(geom: Any, *, precision: int) -> tuple[tuple[float, float], ...]:
    parts: list[tuple[float, float]] = []
    for x, y in list(getattr(geom, "coords", [])):
        parts.append((round(float(x), int(precision)), round(float(y), int(precision))))
    return tuple(parts)


def _point_key(pt: Any, *, precision: int) -> tuple[float, float]:
    return (round(float(pt.x), int(precision)), round(float(pt.y), int(precision)))


def _nudge_point_toward(
    *,
    pt: Any,
    target: Any,
    fraction: float,
    point_type: Any,
) -> Any:
    if pt is None or target is None:
        return pt
    frac = float(fraction)
    if frac <= 0.0:
        return pt
    return point_type(
        float(pt.x) + (float(target.x) - float(pt.x)) * frac,
        float(pt.y) + (float(target.y) - float(pt.y)) * frac,
    )


def _legalize_points_to_area(
    *,
    points: list[dict[str, Any]],
    area_geom: Any,
    legalization_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    _, _, _, line_types = _require_geo_stack()
    _, _, _, Point = line_types
    if area_geom is None or getattr(area_geom, "is_empty", True) or not points:
        return points, {"input_points": int(len(points)), "kept_points": int(len(points)), "legalized_points": 0, "dropped_points": 0}

    rep = area_geom.representative_point()
    kept: list[dict[str, Any]] = []
    legalized = 0
    dropped = 0
    frac0 = float(legalization_fraction)
    frac_schedule = [frac0, frac0 * 10.0, frac0 * 100.0, frac0 * 1000.0]
    frac_schedule = [min(float(v), 1e-2) for v in frac_schedule if float(v) > 0.0]
    for row in points:
        pt = row.get("geometry")
        if pt is None or getattr(pt, "is_empty", True):
            dropped += 1
            continue
        if bool(area_geom.covers(pt)):
            kept.append(row)
            continue
        fixed_pt = None
        for frac in frac_schedule:
            nudged = _nudge_point_toward(pt=pt, target=rep, fraction=frac, point_type=Point)
            if nudged is not None and bool(area_geom.covers(nudged)):
                fixed_pt = nudged
                break
        if fixed_pt is not None:
            fixed = dict(row)
            fixed["geometry"] = fixed_pt
            fixed["geometry_legalized"] = True
            kept.append(fixed)
            legalized += 1
            continue
        dropped += 1
    return kept, {
        "input_points": int(len(points)),
        "kept_points": int(len(kept)),
        "legalized_points": int(legalized),
        "dropped_points": int(dropped),
    }


def _home_codes(home_mode: str) -> list[str]:
    mode = str(home_mode).strip().lower()
    if mode == "compatibility":
        return ["S1400", "S1740"]
    if mode == "conservative":
        return ["S1400"]
    raise ValueError(f"unknown home_mode: {home_mode}")


def _stage_uses_fallback(
    *,
    stage: str | None,
    allowed_non_primary_stages: set[str] | None = None,
) -> bool:
    st = str(stage or "").strip()
    if not st or st == "primary":
        return False
    if allowed_non_primary_stages and st in allowed_non_primary_stages:
        return False
    return True


def _query_road_subset(
    *,
    roads: Any,
    area_geom: Any,
    road_mtfcc_col: str,
    road_component_col: str | None,
    road_sindex: Any | None,
    mtfcc_values: list[str] | None,
) -> Any:
    _, _, pd, _ = _require_geo_stack()
    if road_sindex is not None:
        try:
            idx = list(road_sindex.query(area_geom, predicate="intersects"))
            subset = roads.iloc[idx].copy()
        except Exception:
            subset = roads[roads.geometry.intersects(area_geom)].copy()
    else:
        subset = roads[roads.geometry.intersects(area_geom)].copy()
    if mtfcc_values is not None:
        keep = {str(v) for v in mtfcc_values}
        subset = subset[subset[str(road_mtfcc_col)].astype(str).isin(keep)].copy()
    cols = [c for c in [str(road_mtfcc_col), ("geometry"), (str(road_component_col) if road_component_col else None)] if c]
    cols = [c for c in cols if c in subset.columns]
    if not cols:
        return pd.DataFrame(columns=["geometry", str(road_mtfcc_col)])
    return subset[cols].copy()


def _clip_road_lines(
    *,
    road_subset: Any,
    area_geom: Any,
    road_mtfcc_col: str,
    road_component_col: str | None,
    dedupe_precision: int,
) -> list[dict[str, Any]]:
    _, _, _, line_types = _require_geo_stack()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for rec in road_subset.to_dict(orient="records"):
        geom = rec.get("geometry")
        if geom is None or getattr(geom, "is_empty", True):
            continue
        clipped = geom.intersection(area_geom)
        for line in _iter_line_parts(clipped, line_types=line_types):
            if getattr(line, "is_empty", True):
                continue
            key = _hash_line_coords(line, precision=int(dedupe_precision))
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "geometry": line,
                    "source_mtfcc": str(rec.get(str(road_mtfcc_col), "")) or None,
                    "source_component": (rec.get(str(road_component_col)) if road_component_col else None),
                }
            )
    return rows


def _interpolate_points_from_lines(
    *,
    line_rows: list[dict[str, Any]],
    density: float,
    dedupe_precision: int,
) -> list[dict[str, Any]]:
    _, np, _, line_types = _require_geo_stack()
    _, _, _, Point = line_types
    out: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for row in line_rows:
        line = row["geometry"]
        length = float(getattr(line, "length", 0.0))
        if length <= 0.0:
            continue
        segs = np.arange(float(density), float(length), float(density))
        for seg in segs.tolist():
            pt = line.interpolate(float(seg))
            if not isinstance(pt, Point) or getattr(pt, "is_empty", True):
                continue
            key = _point_key(pt, precision=int(dedupe_precision))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "geometry": pt,
                    "source_mtfcc": row.get("source_mtfcc"),
                    "source_component": row.get("source_component"),
                }
            )
    return out


def _first_points_from_lines(
    *,
    line_rows: list[dict[str, Any]],
    dedupe_precision: int,
) -> list[dict[str, Any]]:
    _, _, _, line_types = _require_geo_stack()
    _, _, _, Point = line_types
    out: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for row in line_rows:
        line = row["geometry"]
        coords = list(getattr(line, "coords", []))
        if not coords:
            continue
        pt = Point(coords[0])
        key = _point_key(pt, precision=int(dedupe_precision))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "geometry": pt,
                "source_mtfcc": row.get("source_mtfcc"),
                "source_component": row.get("source_component"),
            }
        )
    return out


def _build_candidate_frame(
    *,
    groups: list[dict[str, Any]],
    group_col: str,
    role: str,
) -> Any:
    gpd, _, pd, _ = _require_geo_stack()
    if not groups:
        return gpd.GeoDataFrame(
            pd.DataFrame(
                columns=[
                    "candidate_id",
                    str(group_col),
                    "candidate_role",
                    "source_stage",
                    "source_mtfcc",
                    "source_component",
                ]
            ),
            geometry=[],
        )
    rows: list[dict[str, Any]] = []
    for item in groups:
        g = str(item[str(group_col)])
        stage = str(item["source_stage"])
        for i, cand in enumerate(item["points"]):
            rows.append(
                {
                    "candidate_id": f"{g}:{str(role)}:{i:06d}",
                    str(group_col): g,
                    "candidate_role": str(role),
                    "source_stage": stage,
                    "source_mtfcc": cand.get("source_mtfcc"),
                    "source_component": cand.get("source_component"),
                    "geometry": cand["geometry"],
                }
            )
    if not rows:
        return gpd.GeoDataFrame(
            pd.DataFrame(
                columns=[
                    "candidate_id",
                    str(group_col),
                    "candidate_role",
                    "source_stage",
                    "source_mtfcc",
                    "source_component",
                    "geometry_legalized",
                ]
            ),
            geometry=[],
            crs=None,
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=None)


def _road_candidate_state(
    *,
    road_g: Any,
    road_mtfcc_col: str,
    road_component_col: str | None,
    road_sindex: Any | None,
    home_primary: list[str],
    home_compat: list[str],
    work_mtfcc: list[str],
    work_gap_exception_mtfcc: list[str],
    allow_home_fallback: bool,
    allow_work_fallback: bool,
    legalization_fraction: float,
    home_interpolation_density: float,
    work_interpolation_density: float,
    dedupe_precision: int,
) -> dict[str, Any]:
    return {
        "road_g": road_g,
        "road_mtfcc_col": str(road_mtfcc_col),
        "road_component_col": (str(road_component_col) if road_component_col else None),
        "road_sindex": road_sindex,
        "home_primary": [str(v) for v in home_primary],
        "home_compat": [str(v) for v in home_compat],
        "work_mtfcc": [str(v) for v in work_mtfcc],
        "work_gap_exception_mtfcc": [str(v) for v in work_gap_exception_mtfcc],
        "allow_home_fallback": bool(allow_home_fallback),
        "allow_work_fallback": bool(allow_work_fallback),
        "legalization_fraction": float(legalization_fraction),
        "home_interpolation_density": float(home_interpolation_density),
        "work_interpolation_density": float(work_interpolation_density),
        "dedupe_precision": int(dedupe_precision),
    }


def _build_candidates_for_area(
    *,
    area: dict[str, Any],
    group_col: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    g = str(area[str(group_col)])
    geom = area["geometry"]
    if geom is None or getattr(geom, "is_empty", True):
        return {
            str(group_col): g,
            "home_group": {str(group_col): g, "source_stage": "no_candidates", "points": []},
            "work_group": {str(group_col): g, "source_stage": "no_candidates", "points": []},
            "home_geom_meta": {"input_points": 0, "kept_points": 0, "legalized_points": 0, "dropped_points": 0},
            "work_geom_meta": {"input_points": 0, "kept_points": 0, "legalized_points": 0, "dropped_points": 0},
        }

    road_g = state["road_g"]
    road_mtfcc_col = state["road_mtfcc_col"]
    road_component_col = state["road_component_col"]
    road_sindex = state["road_sindex"]
    dedupe_precision = int(state["dedupe_precision"])

    # Home candidates.
    home_stage = "primary"
    home_subset = _query_road_subset(
        roads=road_g,
        area_geom=geom,
        road_mtfcc_col=road_mtfcc_col,
        road_component_col=road_component_col,
        road_sindex=road_sindex,
        mtfcc_values=state["home_primary"],
    )
    home_lines = _clip_road_lines(
        road_subset=home_subset,
        area_geom=geom,
        road_mtfcc_col=road_mtfcc_col,
        road_component_col=road_component_col,
        dedupe_precision=dedupe_precision,
    )
    home_points = _interpolate_points_from_lines(
        line_rows=home_lines,
        density=float(state["home_interpolation_density"]),
        dedupe_precision=dedupe_precision,
    )
    if not home_points and bool(state["allow_home_fallback"]) and state["home_primary"] == ["S1400"]:
        home_stage = "compatibility_fallback"
        home_subset = _query_road_subset(
            roads=road_g,
            area_geom=geom,
            road_mtfcc_col=road_mtfcc_col,
            road_component_col=road_component_col,
            road_sindex=road_sindex,
            mtfcc_values=state["home_compat"],
        )
        home_lines = _clip_road_lines(
            road_subset=home_subset,
            area_geom=geom,
            road_mtfcc_col=road_mtfcc_col,
            road_component_col=road_component_col,
            dedupe_precision=dedupe_precision,
        )
        home_points = _interpolate_points_from_lines(
            line_rows=home_lines,
            density=float(state["home_interpolation_density"]),
            dedupe_precision=dedupe_precision,
        )
    if not home_points and bool(state["allow_home_fallback"]):
        home_stage = "all_roads_fallback"
        home_subset = _query_road_subset(
            roads=road_g,
            area_geom=geom,
            road_mtfcc_col=road_mtfcc_col,
            road_component_col=road_component_col,
            road_sindex=road_sindex,
            mtfcc_values=None,
        )
        home_lines = _clip_road_lines(
            road_subset=home_subset,
            area_geom=geom,
            road_mtfcc_col=road_mtfcc_col,
            road_component_col=road_component_col,
            dedupe_precision=dedupe_precision,
        )
        home_points = _interpolate_points_from_lines(
            line_rows=home_lines,
            density=float(state["home_interpolation_density"]),
            dedupe_precision=dedupe_precision,
        )
    if not home_points and bool(state["allow_home_fallback"]):
        home_stage = "representative_point"
        home_points = [{"geometry": geom.representative_point(), "source_mtfcc": None, "source_component": None}]
    if not home_points:
        home_stage = "no_candidates"
        home_points = []
    home_points, home_geom_meta = _legalize_points_to_area(
        points=home_points,
        area_geom=geom,
        legalization_fraction=float(state["legalization_fraction"]),
    )
    if not home_points:
        home_stage = "no_candidates"

    # Work candidates.
    work_stage = "primary"
    work_subset = _query_road_subset(
        roads=road_g,
        area_geom=geom,
        road_mtfcc_col=road_mtfcc_col,
        road_component_col=road_component_col,
        road_sindex=road_sindex,
        mtfcc_values=state["work_mtfcc"],
    )
    work_lines = _clip_road_lines(
        road_subset=work_subset,
        area_geom=geom,
        road_mtfcc_col=road_mtfcc_col,
        road_component_col=road_component_col,
        dedupe_precision=dedupe_precision,
    )
    work_points = _interpolate_points_from_lines(
        line_rows=work_lines,
        density=float(state["work_interpolation_density"]),
        dedupe_precision=dedupe_precision,
    )
    if not work_points and state["work_gap_exception_mtfcc"]:
        work_stage = "arterial_missing_exception"
        work_subset = _query_road_subset(
            roads=road_g,
            area_geom=geom,
            road_mtfcc_col=road_mtfcc_col,
            road_component_col=road_component_col,
            road_sindex=road_sindex,
            mtfcc_values=state["work_gap_exception_mtfcc"],
        )
        work_lines = _clip_road_lines(
            road_subset=work_subset,
            area_geom=geom,
            road_mtfcc_col=road_mtfcc_col,
            road_component_col=road_component_col,
            dedupe_precision=dedupe_precision,
        )
        work_points = _interpolate_points_from_lines(
            line_rows=work_lines,
            density=float(state["work_interpolation_density"]),
            dedupe_precision=dedupe_precision,
        )
    if not work_points and bool(state["allow_work_fallback"]):
        work_stage = "home_intersection_fallback"
        home_for_work_subset = _query_road_subset(
            roads=road_g,
            area_geom=geom,
            road_mtfcc_col=road_mtfcc_col,
            road_component_col=road_component_col,
            road_sindex=road_sindex,
            mtfcc_values=state["home_compat"],
        )
        home_for_work_lines = _clip_road_lines(
            road_subset=home_for_work_subset,
            area_geom=geom,
            road_mtfcc_col=road_mtfcc_col,
            road_component_col=road_component_col,
            dedupe_precision=dedupe_precision,
        )
        work_points = _first_points_from_lines(
            line_rows=home_for_work_lines,
            dedupe_precision=dedupe_precision,
        )
    if not work_points and bool(state["allow_work_fallback"]):
        work_stage = "all_roads_fallback"
        all_subset = _query_road_subset(
            roads=road_g,
            area_geom=geom,
            road_mtfcc_col=road_mtfcc_col,
            road_component_col=road_component_col,
            road_sindex=road_sindex,
            mtfcc_values=None,
        )
        all_lines = _clip_road_lines(
            road_subset=all_subset,
            area_geom=geom,
            road_mtfcc_col=road_mtfcc_col,
            road_component_col=road_component_col,
            dedupe_precision=dedupe_precision,
        )
        work_points = _interpolate_points_from_lines(
            line_rows=all_lines,
            density=float(state["work_interpolation_density"]),
            dedupe_precision=dedupe_precision,
        )
    if not work_points and bool(state["allow_work_fallback"]):
        work_stage = "representative_point"
        work_points = [{"geometry": geom.representative_point(), "source_mtfcc": None, "source_component": None}]
    if not work_points:
        work_stage = "no_candidates"
        work_points = []
    work_points, work_geom_meta = _legalize_points_to_area(
        points=work_points,
        area_geom=geom,
        legalization_fraction=float(state["legalization_fraction"]),
    )
    if not work_points:
        work_stage = "no_candidates"

    return {
        str(group_col): g,
        "home_group": {str(group_col): g, "source_stage": home_stage, "points": home_points},
        "work_group": {str(group_col): g, "source_stage": work_stage, "points": work_points},
        "home_geom_meta": home_geom_meta,
        "work_geom_meta": work_geom_meta,
    }


def _init_road_candidate_worker(state: dict[str, Any]) -> None:
    global _ROAD_CANDIDATE_WORKER_STATE
    worker_state = dict(state)
    road_sindex = None
    try:
        road_sindex = worker_state["road_g"].sindex
    except Exception:
        road_sindex = None
    worker_state["road_sindex"] = road_sindex
    _ROAD_CANDIDATE_WORKER_STATE = worker_state


def _build_candidates_for_area_worker(task: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    group_col, area = task
    return _build_candidates_for_area(
        area=area,
        group_col=str(group_col),
        state=_ROAD_CANDIDATE_WORKER_STATE,
    )


def build_road_location_candidates(
    *,
    areas: Any,
    roads: Any,
    group_col: str = "tract_geoid",
    road_mtfcc_col: str = "MTFCC",
    road_component_col: str | None = "component",
    home_mode: str = "compatibility",
    work_mtfcc_values: list[str] | None = None,
    work_gap_exception_mtfcc_values: list[str] | None = None,
    allow_home_fallback: bool = False,
    allow_work_fallback: bool = False,
    legalization_fraction: float = 1e-6,
    home_interpolation_density: float = 0.0005,
    work_interpolation_density: float = 0.0002,
    dedupe_precision: int = 6,
    n_jobs: int = 1,
    parallel_chunksize: int = 32,
) -> tuple[Any, Any, dict[str, Any]]:
    """
    Build explicit home/work candidate points on road-constrained support sets.

    Inputs are expected to be GeoDataFrames with polygon `areas` and line `roads`.
    """
    gpd, _, pd, line_types = _require_geo_stack()
    _, _, _, Point = line_types
    if not isinstance(areas, gpd.GeoDataFrame) or not isinstance(roads, gpd.GeoDataFrame):
        raise TypeError("areas and roads must be GeoDataFrame")
    if str(group_col) not in areas.columns:
        raise ValueError(f"areas missing column: {group_col}")
    if str(road_mtfcc_col) not in roads.columns:
        raise ValueError(f"roads missing column: {road_mtfcc_col}")
    if "geometry" not in areas.columns or "geometry" not in roads.columns:
        raise ValueError("areas and roads must contain geometry")

    area_g = areas[[str(group_col), "geometry"]].copy()
    area_g = area_g.dropna(subset=["geometry"]).copy()
    area_g[str(group_col)] = area_g[str(group_col)].astype(str)
    if areas.crs is not None and roads.crs is not None and areas.crs != roads.crs:
        road_g = roads.to_crs(areas.crs).copy()
    else:
        road_g = roads.copy()
    road_g[str(road_mtfcc_col)] = road_g[str(road_mtfcc_col)].astype(str)
    if road_component_col and str(road_component_col) in road_g.columns:
        road_g[str(road_component_col)] = road_g[str(road_component_col)]

    road_sindex = None
    try:
        road_sindex = road_g.sindex
    except Exception:
        road_sindex = None

    work_mtfcc = [str(v) for v in (work_mtfcc_values or ["S1200"])]
    work_gap_exception_mtfcc = [str(v) for v in (work_gap_exception_mtfcc_values or []) if str(v).strip()]
    home_primary = _home_codes(str(home_mode))
    home_compat = ["S1400", "S1740"]
    home_allowed_non_primary_stages: list[str] = []
    work_allowed_non_primary_stages: list[str] = []
    if work_gap_exception_mtfcc:
        work_allowed_non_primary_stages.append("arterial_missing_exception")

    area_records = area_g.to_dict(orient="records")
    state = _road_candidate_state(
        road_g=road_g,
        road_mtfcc_col=str(road_mtfcc_col),
        road_component_col=(str(road_component_col) if road_component_col else None),
        road_sindex=road_sindex,
        home_primary=home_primary,
        home_compat=home_compat,
        work_mtfcc=work_mtfcc,
        work_gap_exception_mtfcc=work_gap_exception_mtfcc,
        allow_home_fallback=bool(allow_home_fallback),
        allow_work_fallback=bool(allow_work_fallback),
        legalization_fraction=float(legalization_fraction),
        home_interpolation_density=float(home_interpolation_density),
        work_interpolation_density=float(work_interpolation_density),
        dedupe_precision=int(dedupe_precision),
    )

    use_parallel = int(n_jobs) > 1 and int(len(area_records)) > 1
    results: list[dict[str, Any]]
    if use_parallel:
        start_method = "fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method()
        ctx = mp.get_context(start_method)
        tasks = [(str(group_col), area) for area in area_records]
        with ctx.Pool(
            processes=int(n_jobs),
            initializer=_init_road_candidate_worker,
            initargs=(state,),
        ) as pool:
            results = list(pool.imap(_build_candidates_for_area_worker, tasks, chunksize=max(1, int(parallel_chunksize))))
    else:
        results = [
            _build_candidates_for_area(
                area=area,
                group_col=str(group_col),
                state=state,
            )
            for area in area_records
        ]

    home_groups: list[dict[str, Any]] = []
    work_groups: list[dict[str, Any]] = []
    home_stage_counts: dict[str, int] = {}
    work_stage_counts: dict[str, int] = {}
    home_geometry_meta = {"input_points": 0, "kept_points": 0, "legalized_points": 0, "dropped_points": 0}
    work_geometry_meta = {"input_points": 0, "kept_points": 0, "legalized_points": 0, "dropped_points": 0}
    for item in results:
        home_group = item["home_group"]
        work_group = item["work_group"]
        home_groups.append(home_group)
        work_groups.append(work_group)
        home_stage = str(home_group["source_stage"])
        work_stage = str(work_group["source_stage"])
        home_stage_counts[home_stage] = int(home_stage_counts.get(home_stage, 0) + 1)
        work_stage_counts[work_stage] = int(work_stage_counts.get(work_stage, 0) + 1)
        for k, v in item["home_geom_meta"].items():
            home_geometry_meta[k] = int(home_geometry_meta.get(k, 0) + int(v))
        for k, v in item["work_geom_meta"].items():
            work_geometry_meta[k] = int(work_geometry_meta.get(k, 0) + int(v))

    home_candidates = _build_candidate_frame(groups=home_groups, group_col=str(group_col), role="home")
    work_candidates = _build_candidate_frame(groups=work_groups, group_col=str(group_col), role="work")
    if areas.crs is not None:
        home_candidates.set_crs(areas.crs, allow_override=True, inplace=True)
        work_candidates.set_crs(areas.crs, allow_override=True, inplace=True)

    meta = {
        "group_col": str(group_col),
        "home_mode": str(home_mode),
        "work_mtfcc_values": work_mtfcc,
        "work_gap_exception_mtfcc_values": work_gap_exception_mtfcc,
        "allow_home_fallback": bool(allow_home_fallback),
        "allow_work_fallback": bool(allow_work_fallback),
        "home_allowed_non_primary_stages": home_allowed_non_primary_stages,
        "work_allowed_non_primary_stages": work_allowed_non_primary_stages,
        "legalization_fraction": float(legalization_fraction),
        "home_interpolation_density": float(home_interpolation_density),
        "work_interpolation_density": float(work_interpolation_density),
        "n_jobs": int(max(1, n_jobs)),
        "parallel_chunksize": int(max(1, parallel_chunksize)),
        "parallel_used": bool(use_parallel),
        "n_groups": int(area_g[str(group_col)].nunique()),
        "n_home_candidates": int(home_candidates.shape[0]),
        "n_work_candidates": int(work_candidates.shape[0]),
        "home_stage_counts": home_stage_counts,
        "work_stage_counts": work_stage_counts,
        "home_geometry_meta": home_geometry_meta,
        "work_geometry_meta": work_geometry_meta,
    }
    return home_candidates, work_candidates, meta


def _sample_candidate_indices(
    *,
    pool_size: int,
    n: int,
    rng: Any,
) -> list[int]:
    if int(pool_size) <= 0 or int(n) <= 0:
        return []
    return rng.integers(0, int(pool_size), size=int(n)).tolist()


def _assign_groupwise_candidates(
    *,
    groups: Any,
    pool_map: dict[str, Any],
    rng: Any,
    candidate_id_out: Any,
    geometry_out: Any,
    stage_out: Any,
    mode_out: Any,
    fallback_out: Any,
    assigned_mode: str,
    no_candidate_mode: str,
    allowed_non_primary_stages: set[str],
) -> None:
    _, np, _, _ = _require_geo_stack()
    if groups is None or int(getattr(groups, "shape", [0])[0]) == 0:
        return
    for g, idx in groups.groupby("_assign_group", sort=False).groups.items():
        take_idx = groups.loc[list(idx), "index"].to_numpy(dtype=int)
        pool = pool_map.get(str(g))
        if pool is None or pool.empty:
            stage_out[take_idx] = "no_candidates"
            mode_out[take_idx] = str(no_candidate_mode)
            continue
        chosen = rng.integers(0, int(pool.shape[0]), size=int(take_idx.shape[0]))
        picked = pool.iloc[chosen]
        stage_vals = picked["source_stage"].astype(str).to_numpy(dtype=object)
        candidate_id_out[take_idx] = picked["candidate_id"].astype(str).to_numpy(dtype=object)
        geometry_out[take_idx] = picked["geometry"].to_numpy(dtype=object)
        stage_out[take_idx] = stage_vals
        mode_out[take_idx] = str(assigned_mode)
        fallback_out[take_idx] = [
            _stage_uses_fallback(stage=str(v), allowed_non_primary_stages=allowed_non_primary_stages)
            for v in stage_vals.tolist()
        ]


def _resolve_work_eligible_mask(
    *,
    persons: Any,
    work_eligible_col: str | None,
    work_eligible_values: list[str] | None,
) -> Any:
    _, _, pd, _ = _require_geo_stack()
    if not work_eligible_col:
        return pd.Series([False] * int(persons.shape[0]), index=persons.index)
    col = str(work_eligible_col)
    if col not in persons.columns:
        raise ValueError(f"persons missing work_eligible_col: {col}")
    s = persons[col]
    if work_eligible_values:
        keep = {str(v).strip().lower() for v in work_eligible_values if str(v).strip()}
        return s.astype(str).str.strip().str.lower().isin(keep)
    if str(s.dtype) == "bool":
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0.0) > 0.0
    raise ValueError(
        "work_eligible_values must be provided when work_eligible_col is non-boolean and non-numeric"
    )


def assign_home_work_locations(
    *,
    persons: Any,
    home_candidates: Any,
    work_candidates: Any,
    group_col: str = "tract_geoid",
    work_group_col: str | None = None,
    person_id_col: str = "person_id",
    household_col: str | None = "household_id",
    work_eligible_col: str | None = None,
    work_eligible_values: list[str] | None = None,
    seed: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """
    Assign home/work locations from pre-built candidate points.
    """
    _, np, pd, _ = _require_geo_stack()
    if not isinstance(persons, pd.DataFrame):
        raise TypeError("persons must be a pandas DataFrame")
    need_person_cols = [str(person_id_col), str(group_col)]
    work_group = str(work_group_col).strip() if work_group_col else str(group_col)
    if work_group not in need_person_cols:
        need_person_cols.append(work_group)
    miss = [c for c in need_person_cols if c not in persons.columns]
    if miss:
        raise ValueError(f"persons missing columns: {miss}")

    rng = np.random.default_rng(int(seed))
    out = persons.copy().reset_index(drop=True)
    out[str(group_col)] = out[str(group_col)].astype(str)
    out[work_group] = out[work_group].astype(str)

    # Candidate pools.
    def _pool_map(cands: Any) -> dict[str, Any]:
        mp: dict[str, Any] = {}
        if cands is None or int(getattr(cands, "shape", [0])[0]) == 0:
            return mp
        tmp = cands.copy()
        tmp[str(group_col)] = tmp[str(group_col)].astype(str)
        for g, gg in tmp.groupby(str(group_col), sort=False):
            mp[str(g)] = gg.reset_index(drop=True)
        return mp

    home_pool = _pool_map(home_candidates)
    work_pool = _pool_map(work_candidates)

    # Home assignment.
    home_candidate_id = np.full((int(out.shape[0]),), None, dtype=object)
    home_geometry = np.full((int(out.shape[0]),), None, dtype=object)
    home_stage = np.full((int(out.shape[0]),), None, dtype=object)
    home_mode_out = np.full((int(out.shape[0]),), None, dtype=object)
    home_fallback = np.full((int(out.shape[0]),), False, dtype=bool)
    home_allowed_non_primary_stages: set[str] = set()

    use_household = bool(household_col) and str(household_col) in out.columns and out[str(household_col)].notna().any()
    if use_household:
        hh_mask = out[str(household_col)].notna()
        hh_groups = (
            out.loc[hh_mask, [str(group_col), str(household_col)]]
            .copy()
            .drop_duplicates([str(group_col), str(household_col)], keep="first")
            .reset_index(drop=False)
            .rename(columns={str(group_col): "_assign_group"})
        )
        hh_row_map = (
            out.loc[hh_mask, [str(group_col), str(household_col)]]
            .copy()
            .reset_index(drop=False)
            .rename(columns={str(group_col): "_assign_group"})
            .merge(
                hh_groups[["index", "_assign_group", str(household_col)]].rename(columns={"index": "_group_index"}),
                on=["_assign_group", str(household_col)],
                how="left",
            )
        )
        unique_home_groups = hh_groups.loc[:, ["index", "_assign_group"]].rename(columns={"index": "index"})
        _assign_groupwise_candidates(
            groups=unique_home_groups,
            pool_map=home_pool,
            rng=rng,
            candidate_id_out=home_candidate_id,
            geometry_out=home_geometry,
            stage_out=home_stage,
            mode_out=home_mode_out,
            fallback_out=home_fallback,
            assigned_mode="household",
            no_candidate_mode="unassigned_no_candidates",
            allowed_non_primary_stages=home_allowed_non_primary_stages,
        )
        group_to_choice = {
            int(idx): (
                home_candidate_id[int(idx)],
                home_geometry[int(idx)],
                home_stage[int(idx)],
                home_mode_out[int(idx)],
                bool(home_fallback[int(idx)]),
            )
            for idx in unique_home_groups["index"].tolist()
        }
        for row_idx, group_idx in zip(hh_row_map["index"].tolist(), hh_row_map["_group_index"].tolist()):
            picked = group_to_choice.get(int(group_idx))
            if picked is None:
                continue
            home_candidate_id[int(row_idx)] = picked[0]
            home_geometry[int(row_idx)] = picked[1]
            home_stage[int(row_idx)] = picked[2]
            home_mode_out[int(row_idx)] = picked[3]
            home_fallback[int(row_idx)] = picked[4]

        person_proxy_groups = (
            out.loc[~hh_mask, [str(group_col)]]
            .copy()
            .reset_index(drop=False)
            .rename(columns={str(group_col): "_assign_group", "index": "index"})
        )
        _assign_groupwise_candidates(
            groups=person_proxy_groups,
            pool_map=home_pool,
            rng=rng,
            candidate_id_out=home_candidate_id,
            geometry_out=home_geometry,
            stage_out=home_stage,
            mode_out=home_mode_out,
            fallback_out=home_fallback,
            assigned_mode="person_proxy",
            no_candidate_mode="unassigned_no_candidates",
            allowed_non_primary_stages=home_allowed_non_primary_stages,
        )
    else:
        person_proxy_groups = (
            out.loc[:, [str(group_col)]]
            .copy()
            .reset_index(drop=False)
            .rename(columns={str(group_col): "_assign_group", "index": "index"})
        )
        _assign_groupwise_candidates(
            groups=person_proxy_groups,
            pool_map=home_pool,
            rng=rng,
            candidate_id_out=home_candidate_id,
            geometry_out=home_geometry,
            stage_out=home_stage,
            mode_out=home_mode_out,
            fallback_out=home_fallback,
            assigned_mode="person_proxy",
            no_candidate_mode="unassigned_no_candidates",
            allowed_non_primary_stages=home_allowed_non_primary_stages,
        )

    # Work assignment.
    work_candidate_id = np.full((int(out.shape[0]),), None, dtype=object)
    work_geometry = np.full((int(out.shape[0]),), None, dtype=object)
    work_stage = np.full((int(out.shape[0]),), None, dtype=object)
    work_mode_out = np.full((int(out.shape[0]),), None, dtype=object)
    work_fallback = np.full((int(out.shape[0]),), False, dtype=bool)
    work_allowed_non_primary_stages: set[str] = set()
    if work_candidates is not None and "source_stage" in work_candidates.columns:
        present_work_stages = {str(v) for v in work_candidates["source_stage"].dropna().astype(str).unique().tolist()}
        if "arterial_missing_exception" in present_work_stages:
            work_allowed_non_primary_stages.add("arterial_missing_exception")

    eligible = _resolve_work_eligible_mask(
        persons=out,
        work_eligible_col=(str(work_eligible_col) if work_eligible_col else None),
        work_eligible_values=work_eligible_values,
    )
    eligible_groups = (
        out.loc[eligible, [work_group]]
        .copy()
        .reset_index(drop=False)
        .rename(columns={work_group: "_assign_group", "index": "index"})
    )
    _assign_groupwise_candidates(
        groups=eligible_groups,
        pool_map=work_pool,
        rng=rng,
        candidate_id_out=work_candidate_id,
        geometry_out=work_geometry,
        stage_out=work_stage,
        mode_out=work_mode_out,
        fallback_out=work_fallback,
        assigned_mode="worker",
        no_candidate_mode="unassigned_no_candidates",
        allowed_non_primary_stages=work_allowed_non_primary_stages,
    )
    work_mode_out[~eligible.to_numpy(dtype=bool)] = "ineligible"

    out["home_candidate_id"] = home_candidate_id.tolist()
    out["home_geometry"] = home_geometry.tolist()
    out["home_source_stage"] = home_stage.tolist()
    out["home_assignment_mode"] = home_mode_out.tolist()
    out["home_fallback_flag"] = home_fallback.tolist()

    out["work_candidate_id"] = work_candidate_id.tolist()
    out["work_geometry"] = work_geometry.tolist()
    out["work_source_stage"] = work_stage.tolist()
    out["work_assignment_mode"] = work_mode_out.tolist()
    out["work_fallback_flag"] = work_fallback.tolist()

    def _extract_xy(col: str, prefix: str) -> None:
        xs: list[float | None] = []
        ys: list[float | None] = []
        for geom in out[col].tolist():
            if geom is None or getattr(geom, "is_empty", True):
                xs.append(None)
                ys.append(None)
            else:
                xs.append(float(geom.x))
                ys.append(float(geom.y))
        out[f"{prefix}_x"] = xs
        out[f"{prefix}_y"] = ys

    _extract_xy("home_geometry", "home")
    _extract_xy("work_geometry", "work")

    meta = {
        "group_col": str(group_col),
        "work_group_col": work_group,
        "seed": int(seed),
        "home_assignment_mode": ("household" if use_household else "person_proxy"),
        "n_persons": int(out.shape[0]),
        "home_assigned": int(pd.notna(out["home_candidate_id"]).sum()),
        "home_unassigned": int(pd.isna(out["home_candidate_id"]).sum()),
        "home_fallback_assignments": int(pd.Series(home_fallback).sum()),
        "work_eligible": int(eligible.sum()),
        "work_assigned": int(pd.notna(out["work_candidate_id"]).sum()),
        "work_unassigned": int(eligible.sum() - pd.notna(out["work_candidate_id"]).sum()),
        "work_fallback_assignments": int(pd.Series(work_fallback).sum()),
    }
    return out, meta
