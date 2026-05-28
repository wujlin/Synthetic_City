from __future__ import annotations

import pathlib
import urllib.request
from typing import Any

import numpy as np
import pandas as pd


JOB_FAMILY_CNS_GROUPS: dict[str, tuple[str, ...]] = {
    "JF_SERVICE": ("CNS07", "CNS14", "CNS15", "CNS16", "CNS17", "CNS18"),
    "JF_INDUSTRIAL": ("CNS01", "CNS02", "CNS03", "CNS04", "CNS05", "CNS06", "CNS08"),
    "JF_PROFESSIONAL": ("CNS09", "CNS10", "CNS11", "CNS12", "CNS13", "CNS19", "CNS20"),
}


def _canon_geoid_series(s: pd.Series, *, width: int) -> pd.Series:
    out = s.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    missing = out.isna() | out.str.lower().isin({"", "nan", "none", "<na>"})
    numeric = out.str.fullmatch(r"\d+").fillna(False)
    out.loc[numeric] = out.loc[numeric].str.zfill(int(width))
    out.loc[missing] = pd.NA
    return out


def lodes_od_url(*, state_postal: str, year: int, part: str) -> str:
    st = str(state_postal).strip().lower()
    if part not in {"main", "aux"}:
        raise ValueError("part must be one of: main, aux")
    return f"https://lehd.ces.census.gov/data/lodes/LODES8/{st}/od/{st}_od_{part}_JT00_{int(year)}.csv.gz"


def ensure_lodes_od_file(
    *,
    state_postal: str,
    year: int,
    part: str,
    out_dir: str | pathlib.Path,
) -> pathlib.Path:
    out_dir = pathlib.Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{str(state_postal).strip().lower()}_od_{part}_JT00_{int(year)}.csv.gz"
    if dest.exists():
        return dest
    urllib.request.urlretrieve(lodes_od_url(state_postal=state_postal, year=year, part=part), dest)
    return dest


def load_lodes_od(
    *,
    main_path: str | pathlib.Path,
    aux_path: str | pathlib.Path,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    cols = usecols or ["w_geocode", "h_geocode", "S000", "SA01", "SA02", "SA03", "SE01", "SE02", "SE03"]

    def _read(path: pathlib.Path) -> pd.DataFrame:
        return pd.read_csv(
            path,
            usecols=cols,
            compression="gzip" if path.suffix == ".gz" else "infer",
            dtype={"w_geocode": str, "h_geocode": str},
            low_memory=False,
        )

    main_df = _read(pathlib.Path(main_path).expanduser().resolve())
    aux_df = _read(pathlib.Path(aux_path).expanduser().resolve())
    out = pd.concat([main_df, aux_df], ignore_index=True)
    out["w_geocode"] = _canon_geoid_series(out["w_geocode"], width=15)
    out["h_geocode"] = _canon_geoid_series(out["h_geocode"], width=15)
    for col in [c for c in out.columns if c not in {"w_geocode", "h_geocode"}]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def load_lodes_rac_or_wac(
    *,
    path: str | pathlib.Path,
    geocode_col: str,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    cols = usecols or [
        str(geocode_col),
        "C000",
        "CA01",
        "CA02",
        "CA03",
        "CE01",
        "CE02",
        "CE03",
        *[f"CNS{i:02d}" for i in range(1, 21)],
    ]
    path = pathlib.Path(path).expanduser().resolve()
    out = pd.read_csv(
        path,
        usecols=cols,
        compression="gzip" if path.suffix == ".gz" else "infer",
        dtype={str(geocode_col): str},
        low_memory=False,
    )
    out[str(geocode_col)] = _canon_geoid_series(out[str(geocode_col)], width=15)
    for col in [c for c in cols if c != str(geocode_col)]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def aggregate_lodes_to_tract_od(od: pd.DataFrame) -> pd.DataFrame:
    out = od.copy()
    out["home_tract_geoid"] = _canon_geoid_series(out["h_geocode"], width=15).str.slice(0, 11)
    out["work_tract_geoid"] = _canon_geoid_series(out["w_geocode"], width=15).str.slice(0, 11)
    value_cols = [c for c in out.columns if c not in {"w_geocode", "h_geocode", "home_tract_geoid", "work_tract_geoid"}]
    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out = (
        out.groupby(["home_tract_geoid", "work_tract_geoid"], as_index=False, sort=False)[value_cols]
        .sum()
        .sort_values(["home_tract_geoid", "work_tract_geoid"], kind="stable")
        .reset_index(drop=True)
    )
    return out


def aggregate_lodes_wac_to_tract(wac: pd.DataFrame) -> pd.DataFrame:
    need = ["w_geocode", "C000"]
    miss = [c for c in need if c not in wac.columns]
    if miss:
        raise ValueError(f"wac missing columns: {miss}")
    out = wac.copy()
    out["tract_geoid"] = _canon_geoid_series(out["w_geocode"], width=15).str.slice(0, 11)
    value_cols = [c for c in out.columns if c not in {"w_geocode", "tract_geoid"}]
    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out = (
        out.groupby("tract_geoid", as_index=False, sort=False)[value_cols]
        .sum()
        .sort_values("tract_geoid", kind="stable")
        .reset_index(drop=True)
    )
    out["tract_geoid"] = out["tract_geoid"].astype(str)
    return _add_lodes_wac_share_columns(out)


def _add_lodes_wac_share_columns(wac: pd.DataFrame) -> pd.DataFrame:
    out = wac.copy()
    total = out["C000"].replace(0.0, np.nan)
    for seg in ["CA01", "CA02", "CA03", "CE01", "CE02", "CE03"]:
        if seg in out.columns:
            out[f"share_{seg}"] = pd.to_numeric(out[seg], errors="coerce").fillna(0.0) / total
            out[f"share_{seg}"] = out[f"share_{seg}"].fillna(0.0)
    for family, cols in JOB_FAMILY_CNS_GROUPS.items():
        present = [c for c in cols if c in out.columns]
        if not present:
            continue
        out[family] = out[present].sum(axis=1)
        out[f"share_{family}"] = pd.to_numeric(out[family], errors="coerce").fillna(0.0) / total
        out[f"share_{family}"] = out[f"share_{family}"].fillna(0.0)
    return out


def _normalize_tract_geoid_crosswalk(
    crosswalk: pd.DataFrame,
    *,
    legacy_col: str = "legacy_tract_geoid",
    current_col: str = "tract_geoid",
    weight_col: str = "weight",
) -> pd.DataFrame:
    missing = [c for c in [legacy_col, current_col, weight_col] if c not in crosswalk.columns]
    if missing:
        raise ValueError(f"tract geoid crosswalk missing columns: {missing}")
    out = crosswalk[[legacy_col, current_col, weight_col]].copy()
    out[legacy_col] = _canon_geoid_series(out[legacy_col], width=11)
    out[current_col] = _canon_geoid_series(out[current_col], width=11)
    out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0)
    out = out[out[weight_col] > 0.0].copy()
    if out.empty:
        return pd.DataFrame(columns=["legacy_tract_geoid", "tract_geoid", "weight"])
    out = (
        out.groupby([legacy_col, current_col], as_index=False, sort=False)[weight_col]
        .sum()
        .rename(columns={legacy_col: "legacy_tract_geoid", current_col: "tract_geoid", weight_col: "weight"})
    )
    totals = out.groupby("legacy_tract_geoid", sort=False)["weight"].transform("sum").replace(0.0, np.nan)
    out["weight"] = (out["weight"] / totals).fillna(0.0)
    return out[out["weight"] > 0.0].reset_index(drop=True)


def remap_tract_od_geoids(
    tract_od: pd.DataFrame,
    tract_crosswalk: pd.DataFrame,
    *,
    legacy_col: str = "legacy_tract_geoid",
    current_col: str = "tract_geoid",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Remap OD tract GEOIDs through an area-weighted tract crosswalk."""
    cw = _normalize_tract_geoid_crosswalk(
        tract_crosswalk,
        legacy_col=legacy_col,
        current_col=current_col,
        weight_col=weight_col,
    )
    if cw.empty or tract_od.empty:
        return tract_od.copy()

    out = tract_od.copy()
    out["home_tract_geoid"] = _canon_geoid_series(out["home_tract_geoid"], width=11)
    out["work_tract_geoid"] = _canon_geoid_series(out["work_tract_geoid"], width=11)
    value_cols = [c for c in out.columns if c not in {"home_tract_geoid", "work_tract_geoid"}]

    def _remap_side(frame: pd.DataFrame, *, col: str, suffix: str) -> pd.DataFrame:
        side_cw = cw.rename(
            columns={
                "legacy_tract_geoid": f"__legacy_{suffix}",
                "tract_geoid": f"__current_{suffix}",
                "weight": f"__weight_{suffix}",
            }
        )
        merged = frame.merge(side_cw, left_on=col, right_on=f"__legacy_{suffix}", how="left")
        matched = merged[f"__current_{suffix}"].notna()
        merged[col] = merged[f"__current_{suffix}"].where(matched, merged[col])
        merged[f"__weight_{suffix}"] = merged[f"__weight_{suffix}"].where(matched, 1.0).astype(float)
        return merged.drop(columns=[f"__legacy_{suffix}", f"__current_{suffix}"])

    out = _remap_side(out, col="home_tract_geoid", suffix="home")
    out = _remap_side(out, col="work_tract_geoid", suffix="work")
    factor = out["__weight_home"].to_numpy(dtype=float) * out["__weight_work"].to_numpy(dtype=float)
    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=float) * factor
    out = out.drop(columns=["__weight_home", "__weight_work"])
    return (
        out.groupby(["home_tract_geoid", "work_tract_geoid"], as_index=False, sort=False)[value_cols]
        .sum()
        .sort_values(["home_tract_geoid", "work_tract_geoid"], kind="stable")
        .reset_index(drop=True)
    )


def remap_tract_wac_geoids(
    tract_wac: pd.DataFrame,
    tract_crosswalk: pd.DataFrame,
    *,
    legacy_col: str = "legacy_tract_geoid",
    current_col: str = "tract_geoid",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Remap tract-level WAC counts through a tract crosswalk and recompute shares."""
    cw = _normalize_tract_geoid_crosswalk(
        tract_crosswalk,
        legacy_col=legacy_col,
        current_col=current_col,
        weight_col=weight_col,
    )
    if cw.empty or tract_wac.empty:
        return tract_wac.copy()

    out = tract_wac.copy()
    out["tract_geoid"] = _canon_geoid_series(out["tract_geoid"], width=11)
    share_cols = [c for c in out.columns if str(c).startswith("share_")]
    if share_cols:
        out = out.drop(columns=share_cols)
    value_cols = [c for c in out.columns if c != "tract_geoid"]

    side_cw = cw.rename(
        columns={
            "legacy_tract_geoid": "__legacy",
            "tract_geoid": "__current",
            "weight": "__weight",
        }
    )
    out = out.merge(side_cw, left_on="tract_geoid", right_on="__legacy", how="left")
    matched = out["__current"].notna()
    out["tract_geoid"] = out["__current"].where(matched, out["tract_geoid"])
    weights = out["__weight"].where(matched, 1.0).astype(float).to_numpy(dtype=float)
    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=float) * weights
    out = out.drop(columns=["__legacy", "__current", "__weight"])
    out = (
        out.groupby("tract_geoid", as_index=False, sort=False)[value_cols]
        .sum()
        .sort_values("tract_geoid", kind="stable")
        .reset_index(drop=True)
    )
    return _add_lodes_wac_share_columns(out)


def build_tract_area_crosswalk(
    *,
    legacy_areas: Any,
    current_areas: Any,
    legacy_group_col: str = "GEOID",
    current_group_col: str = "GEOID",
    min_legacy_area_share: float = 1e-8,
    projected_crs: str = "EPSG:5070",
) -> pd.DataFrame:
    """Build an area-weighted legacy-to-current tract GEOID crosswalk."""
    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("build_tract_area_crosswalk requires geopandas.") from e

    if str(legacy_group_col) not in legacy_areas.columns:
        raise ValueError(f"legacy areas missing group column: {legacy_group_col}")
    if str(current_group_col) not in current_areas.columns:
        raise ValueError(f"current areas missing group column: {current_group_col}")

    legacy = legacy_areas[[str(legacy_group_col), "geometry"]].dropna(subset=["geometry"]).copy()
    current = current_areas[[str(current_group_col), "geometry"]].dropna(subset=["geometry"]).copy()
    legacy = legacy.rename(columns={str(legacy_group_col): "legacy_tract_geoid"})
    current = current.rename(columns={str(current_group_col): "tract_geoid"})
    legacy["legacy_tract_geoid"] = _canon_geoid_series(legacy["legacy_tract_geoid"], width=11)
    current["tract_geoid"] = _canon_geoid_series(current["tract_geoid"], width=11)
    legacy = legacy.drop_duplicates("legacy_tract_geoid", keep="first")
    current = current.drop_duplicates("tract_geoid", keep="first")
    if legacy.empty or current.empty:
        return pd.DataFrame(columns=["legacy_tract_geoid", "tract_geoid", "weight", "legacy_area_share", "overlap_area"])

    if legacy.crs is None:
        legacy = legacy.set_crs("EPSG:4269", allow_override=True)
    if current.crs is None:
        current = current.set_crs("EPSG:4269", allow_override=True)
    legacy = legacy.to_crs(projected_crs)
    current = current.to_crs(projected_crs)
    legacy["__legacy_area"] = legacy.geometry.area.astype(float)

    inter = gpd.overlay(
        legacy[["legacy_tract_geoid", "__legacy_area", "geometry"]],
        current[["tract_geoid", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if inter.empty:
        return pd.DataFrame(columns=["legacy_tract_geoid", "tract_geoid", "weight", "legacy_area_share", "overlap_area"])
    inter["overlap_area"] = inter.geometry.area.astype(float)
    inter = inter[inter["overlap_area"] > 0.0].copy()
    inter["legacy_area_share"] = inter["overlap_area"] / inter["__legacy_area"].replace(0.0, np.nan)
    inter = inter[inter["legacy_area_share"] >= float(min_legacy_area_share)].copy()
    if inter.empty:
        return pd.DataFrame(columns=["legacy_tract_geoid", "tract_geoid", "weight", "legacy_area_share", "overlap_area"])
    denom = inter.groupby("legacy_tract_geoid", sort=False)["overlap_area"].transform("sum").replace(0.0, np.nan)
    inter["weight"] = (inter["overlap_area"] / denom).fillna(0.0)
    out = inter.loc[:, ["legacy_tract_geoid", "tract_geoid", "weight", "legacy_area_share", "overlap_area"]].copy()
    return out.sort_values(["legacy_tract_geoid", "tract_geoid"], kind="stable").reset_index(drop=True)


def haversine_km(
    lon1: Any,
    lat1: Any,
    lon2: Any,
    lat2: Any,
) -> np.ndarray:
    lon1r = np.radians(np.asarray(lon1, dtype=float))
    lat1r = np.radians(np.asarray(lat1, dtype=float))
    lon2r = np.radians(np.asarray(lon2, dtype=float))
    lat2r = np.radians(np.asarray(lat2, dtype=float))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c


def compute_gravity_accessibility(
    *,
    tract_centroids: pd.DataFrame,
    tract_mass: pd.DataFrame,
    tract_col: str = "tract_geoid",
    mass_col: str = "C000",
    distance_beta: float = 0.1,
    out_col: str = "access_jobs_gravity",
) -> pd.DataFrame:
    if float(distance_beta) <= 0.0:
        raise ValueError("distance_beta must be > 0 for gravity accessibility")
    if str(tract_col) not in tract_centroids.columns:
        raise ValueError(f"tract_centroids missing tract column: {tract_col}")
    if str(tract_col) not in tract_mass.columns or str(mass_col) not in tract_mass.columns:
        raise ValueError(f"tract_mass missing columns: {[tract_col, mass_col]}")

    cent = tract_centroids[[str(tract_col), "centroid_x", "centroid_y"]].copy()
    cent[str(tract_col)] = cent[str(tract_col)].astype(str)
    mass = tract_mass[[str(tract_col), str(mass_col)]].copy()
    mass[str(tract_col)] = mass[str(tract_col)].astype(str)
    mass[str(mass_col)] = pd.to_numeric(mass[str(mass_col)], errors="coerce").fillna(0.0).clip(lower=0.0)
    merged = cent.merge(mass, on=str(tract_col), how="inner").drop_duplicates(str(tract_col), keep="first").reset_index(drop=True)
    if merged.empty:
        return pd.DataFrame({str(tract_col): [], str(out_col): []})

    x = merged["centroid_x"].to_numpy(dtype=float)
    y = merged["centroid_y"].to_numpy(dtype=float)
    jobs = merged[str(mass_col)].to_numpy(dtype=float)
    dist = haversine_km(x[:, None], y[:, None], x[None, :], y[None, :])
    gravity = np.exp(-float(distance_beta) * dist) @ jobs
    return pd.DataFrame(
        {
            str(tract_col): merged[str(tract_col)].astype(str).tolist(),
            str(out_col): gravity.astype(float).tolist(),
        }
    )


def compute_job_center_accessibility(
    *,
    tract_centroids: pd.DataFrame,
    tract_mass: pd.DataFrame,
    tract_col: str = "tract_geoid",
    mass_col: str = "C000",
    distance_beta: float = 0.1,
    top_quantile: float = 0.95,
    min_centers: int = 10,
    out_col: str = "access_job_centers_gravity",
) -> pd.DataFrame:
    if float(distance_beta) <= 0.0:
        raise ValueError("distance_beta must be > 0 for job-center accessibility")
    q = float(top_quantile)
    if not (0.0 < q < 1.0):
        raise ValueError("top_quantile must be in (0,1)")
    cent = tract_centroids[[str(tract_col), "centroid_x", "centroid_y"]].copy()
    cent[str(tract_col)] = cent[str(tract_col)].astype(str)
    mass = tract_mass[[str(tract_col), str(mass_col)]].copy()
    mass[str(tract_col)] = mass[str(tract_col)].astype(str)
    mass[str(mass_col)] = pd.to_numeric(mass[str(mass_col)], errors="coerce").fillna(0.0).clip(lower=0.0)
    merged = cent.merge(mass, on=str(tract_col), how="inner").drop_duplicates(str(tract_col), keep="first").reset_index(drop=True)
    if merged.empty:
        return pd.DataFrame({str(tract_col): [], str(out_col): []})
    cutoff = float(merged[str(mass_col)].quantile(q))
    centers = merged[merged[str(mass_col)] >= cutoff].copy()
    if centers.shape[0] < int(min_centers):
        centers = merged.sort_values(str(mass_col), ascending=False, kind="stable").head(int(min_centers)).copy()
    x = merged["centroid_x"].to_numpy(dtype=float)
    y = merged["centroid_y"].to_numpy(dtype=float)
    cx = centers["centroid_x"].to_numpy(dtype=float)
    cy = centers["centroid_y"].to_numpy(dtype=float)
    cjobs = centers[str(mass_col)].to_numpy(dtype=float)
    dist = haversine_km(x[:, None], y[:, None], cx[None, :], cy[None, :])
    gravity = np.exp(-float(distance_beta) * dist) @ cjobs
    return pd.DataFrame(
        {
            str(tract_col): merged[str(tract_col)].astype(str).tolist(),
            str(out_col): gravity.astype(float).tolist(),
        }
    )


def assign_job_center_membership(
    *,
    tract_centroids: pd.DataFrame,
    tract_mass: pd.DataFrame,
    tract_col: str = "tract_geoid",
    mass_col: str = "C000",
    county_col: str = "county_geoid",
    top_quantile: float = 0.95,
    min_centers_per_county: int = 3,
) -> pd.DataFrame:
    q = float(top_quantile)
    if not (0.0 < q < 1.0):
        raise ValueError("top_quantile must be in (0,1)")
    cent = tract_centroids[[str(tract_col), "centroid_x", "centroid_y"]].copy()
    cent[str(tract_col)] = cent[str(tract_col)].astype(str)
    if str(county_col) not in cent.columns:
        cent[str(county_col)] = cent[str(tract_col)].astype(str).str.slice(0, 5)
    mass = tract_mass[[str(tract_col), str(mass_col)]].copy()
    mass[str(tract_col)] = mass[str(tract_col)].astype(str)
    mass[str(mass_col)] = pd.to_numeric(mass[str(mass_col)], errors="coerce").fillna(0.0).clip(lower=0.0)
    merged = cent.merge(mass, on=str(tract_col), how="inner").drop_duplicates(str(tract_col), keep="first").reset_index(drop=True)
    if merged.empty:
        return pd.DataFrame(
            {
                str(tract_col): [],
                str(county_col): [],
                "center_geoid": [],
                "center_county_geoid": [],
                "center_distance_km": [],
                "center_mass": [],
            }
        )

    rows: list[pd.DataFrame] = []
    for county, grp in merged.groupby(str(county_col), sort=False):
        grp = grp.reset_index(drop=True)
        cutoff = float(grp[str(mass_col)].quantile(q))
        centers = grp[grp[str(mass_col)] >= cutoff].copy()
        if centers.shape[0] < int(min_centers_per_county):
            centers = grp.sort_values(str(mass_col), ascending=False, kind="stable").head(int(min_centers_per_county)).copy()
        gx = grp["centroid_x"].to_numpy(dtype=float)
        gy = grp["centroid_y"].to_numpy(dtype=float)
        cx = centers["centroid_x"].to_numpy(dtype=float)
        cy = centers["centroid_y"].to_numpy(dtype=float)
        dist = haversine_km(gx[:, None], gy[:, None], cx[None, :], cy[None, :])
        nearest = np.argmin(dist, axis=1)
        grp_out = pd.DataFrame(
            {
                str(tract_col): grp[str(tract_col)].astype(str).tolist(),
                str(county_col): grp[str(county_col)].astype(str).tolist(),
                "center_geoid": centers[str(tract_col)].astype(str).to_numpy()[nearest].tolist(),
                "center_county_geoid": [str(county)] * int(grp.shape[0]),
                "center_distance_km": dist[np.arange(dist.shape[0]), nearest].astype(float).tolist(),
                "center_mass": centers[str(mass_col)].to_numpy(dtype=float)[nearest].astype(float).tolist(),
            }
        )
        rows.append(grp_out)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out


def build_tract_centroid_table(
    *,
    areas: Any,
    group_col: str = "tract_geoid",
) -> pd.DataFrame:
    if str(group_col) not in areas.columns:
        raise ValueError(f"areas missing group column: {group_col}")
    geo = areas[[str(group_col), "geometry"]].copy()
    geo = geo.dropna(subset=["geometry"]).copy()
    geo[str(group_col)] = geo[str(group_col)].astype(str)
    reps = geo.geometry.representative_point()
    out = pd.DataFrame(
        {
            str(group_col): geo[str(group_col)].astype(str).tolist(),
            "centroid_x": reps.x.astype(float).tolist(),
            "centroid_y": reps.y.astype(float).tolist(),
        }
    )
    return out.drop_duplicates(str(group_col), keep="first").reset_index(drop=True)


def enrich_tract_od_with_geometry_and_wac(
    *,
    tract_od: pd.DataFrame,
    tract_centroids: pd.DataFrame,
    tract_wac: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = tract_od.copy()
    out["home_tract_geoid"] = out["home_tract_geoid"].astype(str)
    out["work_tract_geoid"] = out["work_tract_geoid"].astype(str)

    cent = tract_centroids.copy()
    cent["tract_geoid"] = cent["tract_geoid"].astype(str)
    home_cent = cent.rename(
        columns={
            "tract_geoid": "home_tract_geoid",
            "centroid_x": "home_centroid_x",
            "centroid_y": "home_centroid_y",
        }
    )
    work_cent = cent.rename(
        columns={
            "tract_geoid": "work_tract_geoid",
            "centroid_x": "work_centroid_x",
            "centroid_y": "work_centroid_y",
        }
    )
    out = out.merge(home_cent, on="home_tract_geoid", how="left")
    out = out.merge(work_cent, on="work_tract_geoid", how="left")
    out["distance_km"] = haversine_km(
        out["home_centroid_x"],
        out["home_centroid_y"],
        out["work_centroid_x"],
        out["work_centroid_y"],
    )

    if tract_wac is not None and not tract_wac.empty:
        wac = tract_wac.copy()
        wac["tract_geoid"] = wac["tract_geoid"].astype(str)
        work_rename = {
            "tract_geoid": "work_tract_geoid",
            "C000": "work_C000",
            "CA01": "work_CA01",
            "CA02": "work_CA02",
            "CA03": "work_CA03",
            "CE01": "work_CE01",
            "CE02": "work_CE02",
            "CE03": "work_CE03",
            "share_CA01": "work_share_CA01",
            "share_CA02": "work_share_CA02",
            "share_CA03": "work_share_CA03",
            "share_CE01": "work_share_CE01",
            "share_CE02": "work_share_CE02",
            "share_CE03": "work_share_CE03",
            "access_jobs_gravity": "work_access_jobs_gravity",
            "access_job_centers_gravity": "work_access_job_centers_gravity",
            "center_geoid": "work_center_geoid",
            "center_county_geoid": "work_center_county_geoid",
            "center_distance_km": "work_center_distance_km",
            "center_mass": "work_center_mass",
        }
        for family in JOB_FAMILY_CNS_GROUPS:
            if family in wac.columns:
                work_rename[family] = f"work_{family}"
            share_col = f"share_{family}"
            if share_col in wac.columns:
                work_rename[share_col] = f"work_share_{family}"
        work_keep = [c for c in work_rename if c in wac.columns]
        out = out.merge(wac[work_keep].rename(columns=work_rename), on="work_tract_geoid", how="left")
        home_rename = {
            "tract_geoid": "home_tract_geoid",
            "center_geoid": "home_center_geoid",
            "center_county_geoid": "home_center_county_geoid",
            "center_distance_km": "home_center_distance_km",
            "center_mass": "home_center_mass",
        }
        home_keep = [c for c in home_rename if c in wac.columns]
        out = out.merge(wac[home_keep].rename(columns=home_rename), on="home_tract_geoid", how="left")
        for col in [
            "work_C000",
            "work_CA01",
            "work_CA02",
            "work_CA03",
            "work_CE01",
            "work_CE02",
            "work_CE03",
            "work_share_CA01",
            "work_share_CA02",
            "work_share_CA03",
            "work_share_CE01",
            "work_share_CE02",
            "work_share_CE03",
            "work_access_jobs_gravity",
            "work_access_job_centers_gravity",
            "work_center_distance_km",
            "work_center_mass",
            "work_JF_SERVICE",
            "work_JF_INDUSTRIAL",
            "work_JF_PROFESSIONAL",
            "work_share_JF_SERVICE",
            "work_share_JF_INDUSTRIAL",
            "work_share_JF_PROFESSIONAL",
            "home_center_distance_km",
            "home_center_mass",
        ]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


def prepare_internal_study_tract_od(
    *,
    tract_od: pd.DataFrame,
    study_tracts: set[str] | list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    study = {str(x).replace(".0", "").strip().zfill(11) for x in study_tracts}
    od = tract_od.copy()
    od["home_tract_geoid"] = _canon_geoid_series(od["home_tract_geoid"], width=11)
    od["work_tract_geoid"] = _canon_geoid_series(od["work_tract_geoid"], width=11)
    od["S000"] = pd.to_numeric(od["S000"], errors="coerce").fillna(0.0)
    value_cols = [c for c in od.columns if c not in {"home_tract_geoid", "work_tract_geoid"}]
    for col in value_cols:
        od[col] = pd.to_numeric(od[col], errors="coerce").fillna(0.0)

    from_study = od[od["home_tract_geoid"].isin(sorted(study))].copy()
    internal = from_study[from_study["work_tract_geoid"].isin(sorted(study))].copy()
    internal = (
        internal.groupby(["home_tract_geoid", "work_tract_geoid"], as_index=False, sort=False)[value_cols]
        .sum()
        .sort_values(["home_tract_geoid", "work_tract_geoid"], kind="stable")
        .reset_index(drop=True)
    )

    origin_total = (
        from_study.groupby("home_tract_geoid", as_index=False, sort=False)["S000"]
        .sum()
        .rename(columns={"S000": "total_jobs_from_origin"})
    )
    origin_internal = (
        internal.groupby("home_tract_geoid", as_index=False, sort=False)["S000"]
        .sum()
        .rename(columns={"S000": "internal_jobs_from_origin"})
    )
    origin_stats = pd.DataFrame({"home_tract_geoid": sorted(study)}).merge(origin_total, on="home_tract_geoid", how="left")
    origin_stats = origin_stats.merge(origin_internal, on="home_tract_geoid", how="left")
    origin_stats["total_jobs_from_origin"] = origin_stats["total_jobs_from_origin"].fillna(0.0)
    origin_stats["internal_jobs_from_origin"] = origin_stats["internal_jobs_from_origin"].fillna(0.0)
    origin_stats["internal_share"] = origin_stats["internal_jobs_from_origin"] / origin_stats["total_jobs_from_origin"].replace(0.0, 1.0)
    origin_stats["has_internal_destination"] = origin_stats["internal_jobs_from_origin"] > 0.0

    summary = {
        "n_study_tracts": int(len(study)),
        "n_origin_tracts_with_any_jobs": int((origin_stats["total_jobs_from_origin"] > 0.0).sum()),
        "n_origin_tracts_with_internal_dest": int(origin_stats["has_internal_destination"].sum()),
        "share_origin_tracts_with_internal_dest": float(origin_stats["has_internal_destination"].mean()) if len(origin_stats) else float("nan"),
        "total_jobs_from_study_origins": float(origin_stats["total_jobs_from_origin"].sum()),
        "total_internal_jobs": float(origin_stats["internal_jobs_from_origin"].sum()),
        "overall_internal_share": float(origin_stats["internal_jobs_from_origin"].sum() / max(origin_stats["total_jobs_from_origin"].sum(), 1.0)),
    }
    return internal, origin_stats.sort_values("home_tract_geoid", kind="stable").reset_index(drop=True), summary
