from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _require_geopandas() -> Any:
    try:
        import geopandas as gpd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("mobility_anchor.py requires geopandas.") from e
    return gpd


@dataclass(frozen=True)
class AnchorSpec:
    min_home_secs: int = 6 * 3600
    min_work_secs: int = 3 * 3600
    night_start_hour: int = 20
    night_end_hour: int = 6
    work_start_hour: int = 9
    work_end_hour: int = 17
    min_home_work_distance_m: float = 500.0


def _extract_local_hour(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.slice(11, 13), errors="coerce")


def _night_mask(start_hour: pd.Series, end_hour: pd.Series, *, spec: AnchorSpec) -> pd.Series:
    return (start_hour >= spec.night_start_hour) | (end_hour < spec.night_end_hour) | (
        (start_hour < spec.night_end_hour) & (end_hour < 12)
    )


def _day_mask(start_hour: pd.Series, *, spec: AnchorSpec) -> pd.Series:
    return (start_hour >= spec.work_start_hour) & (start_hour < spec.work_end_hour)


def haversine_m(
    lon1: pd.Series | np.ndarray,
    lat1: pd.Series | np.ndarray,
    lon2: pd.Series | np.ndarray,
    lat2: pd.Series | np.ndarray,
) -> np.ndarray:
    lon1r = np.radians(np.asarray(lon1, dtype=float))
    lat1r = np.radians(np.asarray(lat1, dtype=float))
    lon2r = np.radians(np.asarray(lon2, dtype=float))
    lat2r = np.radians(np.asarray(lat2, dtype=float))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371000.0 * c


def load_events_in_bbox(
    *,
    path: str | Path,
    bbox: tuple[float, float, float, float],
    chunksize: int = 500_000,
) -> pd.DataFrame:
    usecols = ["ad_id", "latitude", "longitude", "time_spent", "start_time_local", "end_time_local"]
    path = Path(path).expanduser().resolve()
    frames: list[pd.DataFrame] = []
    minx, miny, maxx, maxy = bbox
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=int(chunksize)):
        det = chunk[
            chunk["latitude"].between(miny, maxy)
            & chunk["longitude"].between(minx, maxx)
        ].copy()
        if det.empty:
            continue
        frames.append(det)
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True)


def select_device_anchors(events: pd.DataFrame, *, spec: AnchorSpec | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    s = spec or AnchorSpec()
    if events.empty:
        empty = pd.DataFrame(columns=["ad_id", "latitude", "longitude", "time_spent"])
        return empty, empty, {
            "n_events": 0,
            "n_devices": 0,
            "n_home_anchor_devices": 0,
            "n_work_candidate_devices": 0,
            "n_work_anchor_devices": 0,
            "n_home_work_devices": 0,
        }

    ev = events.copy()
    ev["ad_id"] = ev["ad_id"].astype(str)
    ev["time_spent"] = pd.to_numeric(ev["time_spent"], errors="coerce").fillna(0).astype(float)
    ev["_start_hour"] = _extract_local_hour(ev["start_time_local"])
    ev["_end_hour"] = _extract_local_hour(ev["end_time_local"])
    ev["_is_home_candidate"] = _night_mask(ev["_start_hour"], ev["_end_hour"], spec=s) & (ev["time_spent"] >= float(s.min_home_secs))
    ev["_is_work_candidate"] = _day_mask(ev["_start_hour"], spec=s) & (ev["time_spent"] >= float(s.min_work_secs))

    home = (
        ev[ev["_is_home_candidate"]]
        .sort_values(["ad_id", "time_spent"], ascending=[True, False], kind="stable")
        .drop_duplicates("ad_id", keep="first")
        .loc[:, ["ad_id", "latitude", "longitude", "time_spent", "start_time_local", "end_time_local"]]
        .rename(
            columns={
                "latitude": "home_latitude",
                "longitude": "home_longitude",
                "time_spent": "home_time_spent",
                "start_time_local": "home_start_time_local",
                "end_time_local": "home_end_time_local",
            }
        )
        .reset_index(drop=True)
    )

    work = (
        ev[ev["_is_work_candidate"]]
        .sort_values(["ad_id", "time_spent"], ascending=[True, False], kind="stable")
        .drop_duplicates("ad_id", keep="first")
        .loc[:, ["ad_id", "latitude", "longitude", "time_spent", "start_time_local", "end_time_local"]]
        .rename(
            columns={
                "latitude": "work_latitude",
                "longitude": "work_longitude",
                "time_spent": "work_time_spent",
                "start_time_local": "work_start_time_local",
                "end_time_local": "work_end_time_local",
            }
        )
        .reset_index(drop=True)
    )

    work_merged = work.merge(home, on="ad_id", how="inner")
    if len(work_merged) > 0:
        work_merged["home_work_distance_m"] = haversine_m(
            work_merged["home_longitude"],
            work_merged["home_latitude"],
            work_merged["work_longitude"],
            work_merged["work_latitude"],
        )
        work_merged = work_merged[work_merged["home_work_distance_m"] >= float(s.min_home_work_distance_m)].copy()
    work = work_merged.reset_index(drop=True)

    summary = {
        "n_events": int(len(ev)),
        "n_devices": int(ev["ad_id"].nunique()),
        "n_home_anchor_devices": int(len(home)),
        "n_work_candidate_devices": int(ev.loc[ev["_is_work_candidate"], "ad_id"].nunique()),
        "n_work_anchor_devices": int(work["ad_id"].nunique()) if "ad_id" in work.columns else 0,
        "n_home_work_devices": int(work["ad_id"].nunique()) if "ad_id" in work.columns else 0,
    }
    return home, work, summary


def load_bg_units(*, tiger_bg_zip: str | Path, allowed_tracts: set[str] | None = None) -> Any:
    gpd = _require_geopandas()
    bg = gpd.read_file(f"zip://{Path(tiger_bg_zip).expanduser().resolve()}")
    if "GEOID" not in bg.columns:
        raise ValueError("BG file missing GEOID column.")
    out = bg.loc[:, ["GEOID", "geometry"]].copy()
    out["bg_geoid"] = out["GEOID"].astype(str)
    out["tract_geoid"] = out["bg_geoid"].str.slice(0, 11)
    if allowed_tracts:
        out = out[out["tract_geoid"].isin(sorted(set(str(x) for x in allowed_tracts)))].copy()
    return out[["bg_geoid", "tract_geoid", "geometry"]].reset_index(drop=True)


def spatial_join_points_to_bg(
    *,
    points: pd.DataFrame,
    x_col: str,
    y_col: str,
    bg_units: Any,
    keep_cols: list[str],
) -> pd.DataFrame:
    gpd = _require_geopandas()
    if points.empty:
        return pd.DataFrame(columns=keep_cols + ["bg_geoid", "tract_geoid"])
    selected_cols = list(dict.fromkeys(keep_cols + [x_col, y_col]))
    work = points.loc[:, selected_cols].copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=[x_col, y_col]).copy()
    if work.empty:
        return pd.DataFrame(columns=keep_cols + ["bg_geoid", "tract_geoid"])
    gdf = gpd.GeoDataFrame(
        work,
        geometry=gpd.points_from_xy(work[x_col], work[y_col], crs="EPSG:4326"),
    )
    target = bg_units
    if getattr(target, "crs", None) is None:
        target = target.set_crs("EPSG:4269")
    target = target.to_crs(gdf.crs)
    joined = gpd.sjoin(
        gdf,
        target[["bg_geoid", "tract_geoid", "geometry"]],
        how="left",
        predicate="intersects",
    )
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    dedupe_cols = [col for col in keep_cols if col in out.columns]
    if dedupe_cols:
        out = out.drop_duplicates(subset=dedupe_cols, keep="first")
    out = out.reset_index(drop=True)
    return out


def compare_share_frames(
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_cols: list[str],
    left_value_col: str,
    right_value_col: str,
    top_k: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = left.merge(right, on=key_cols, how="outer").fillna({left_value_col: 0.0, right_value_col: 0.0})
    merged[left_value_col] = pd.to_numeric(merged[left_value_col], errors="coerce").fillna(0.0).astype(float)
    merged[right_value_col] = pd.to_numeric(merged[right_value_col], errors="coerce").fillna(0.0).astype(float)

    left_total = float(merged[left_value_col].sum())
    right_total = float(merged[right_value_col].sum())
    merged["left_share"] = merged[left_value_col] / max(left_total, 1.0)
    merged["right_share"] = merged[right_value_col] / max(right_total, 1.0)

    valid = merged[["left_share", "right_share"]].copy()
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid) >= 2 and valid["left_share"].nunique() > 1 and valid["right_share"].nunique() > 1:
        spearman = float(valid["left_share"].corr(valid["right_share"], method="spearman"))
    else:
        spearman = float("nan")

    left_vec = merged["left_share"].to_numpy(dtype=float)
    right_vec = merged["right_share"].to_numpy(dtype=float)
    denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
    cosine = float(np.dot(left_vec, right_vec) / denom) if denom > 0.0 else float("nan")
    tvd = float(0.5 * np.abs(left_vec - right_vec).sum())

    left_top = set(
        merged.sort_values("left_share", ascending=False, kind="stable")
        .head(int(top_k))
        .loc[:, key_cols]
        .astype(str)
        .agg("|".join, axis=1)
        .tolist()
    )
    right_top = set(
        merged.sort_values("right_share", ascending=False, kind="stable")
        .head(int(top_k))
        .loc[:, key_cols]
        .astype(str)
        .agg("|".join, axis=1)
        .tolist()
    )

    summary = {
        "n_units": int(len(merged)),
        "left_total": left_total,
        "right_total": right_total,
        "spearman_share": spearman,
        "cosine_share": cosine,
        "tvd_share": tvd,
        "top_k": int(top_k),
        "top_k_overlap": int(len(left_top & right_top)),
    }
    return merged, summary


def within_tract_bg_spearman(
    *,
    synthetic_bg_counts: pd.DataFrame,
    mobility_bg_counts: pd.DataFrame,
    min_mobility_total: int = 20,
    min_bg_units: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    syn = synthetic_bg_counts.rename(columns={"count": "synthetic_count"}).copy()
    mob = mobility_bg_counts.rename(columns={"count": "mobility_count"}).copy()
    merged = syn.merge(mob, on=["tract_geoid", "bg_geoid"], how="outer").fillna(0.0)
    merged["synthetic_count"] = pd.to_numeric(merged["synthetic_count"], errors="coerce").fillna(0.0)
    merged["mobility_count"] = pd.to_numeric(merged["mobility_count"], errors="coerce").fillna(0.0)

    rows: list[dict[str, Any]] = []
    for tract, grp in merged.groupby("tract_geoid", sort=False):
        mob_total = float(grp["mobility_count"].sum())
        n_bg = int(len(grp))
        if mob_total < float(min_mobility_total) or n_bg < int(min_bg_units):
            rows.append(
                {
                    "tract_geoid": str(tract),
                    "n_bg_units": n_bg,
                    "mobility_total": mob_total,
                    "synthetic_total": float(grp["synthetic_count"].sum()),
                    "spearman_bg": np.nan,
                    "eligible": False,
                }
            )
            continue
        if grp["mobility_count"].nunique() <= 1 or grp["synthetic_count"].nunique() <= 1:
            sp = np.nan
        else:
            sp = float(grp["mobility_count"].corr(grp["synthetic_count"], method="spearman"))
        rows.append(
            {
                "tract_geoid": str(tract),
                "n_bg_units": n_bg,
                "mobility_total": mob_total,
                "synthetic_total": float(grp["synthetic_count"].sum()),
                "spearman_bg": sp,
                "eligible": True,
            }
        )

    out = pd.DataFrame(rows).sort_values("tract_geoid", kind="stable").reset_index(drop=True)
    valid = out[out["eligible"] & out["spearman_bg"].notna()].copy()
    summary = {
        "n_tracts_total": int(len(out)),
        "n_tracts_eligible": int(out["eligible"].sum()),
        "n_tracts_with_valid_spearman": int(len(valid)),
        "mean_spearman_bg": float(valid["spearman_bg"].mean()) if len(valid) else float("nan"),
        "median_spearman_bg": float(valid["spearman_bg"].median()) if len(valid) else float("nan"),
        "share_spearman_bg_ge_0_3": float((valid["spearman_bg"] >= 0.3).mean()) if len(valid) else float("nan"),
        "share_spearman_bg_ge_0_5": float((valid["spearman_bg"] >= 0.5).mean()) if len(valid) else float("nan"),
    }
    return out, summary


def summarize_distance_distribution(
    *,
    synthetic_distance_m: pd.Series | np.ndarray,
    mobility_distance_m: pd.Series | np.ndarray,
    bins_km: list[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bins = bins_km or [0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 40.0, 80.0, np.inf]
    syn = np.asarray(synthetic_distance_m, dtype=float)
    mob = np.asarray(mobility_distance_m, dtype=float)
    syn = syn[np.isfinite(syn)]
    mob = mob[np.isfinite(mob)]

    syn_km = syn / 1000.0
    mob_km = mob / 1000.0
    syn_hist, edges = np.histogram(syn_km, bins=bins)
    mob_hist, _ = np.histogram(mob_km, bins=bins)

    table = pd.DataFrame(
        {
            "bin_left_km": edges[:-1],
            "bin_right_km": edges[1:],
            "synthetic_count": syn_hist,
            "mobility_count": mob_hist,
        }
    )
    syn_share = syn_hist / max(int(syn_hist.sum()), 1)
    mob_share = mob_hist / max(int(mob_hist.sum()), 1)
    table["synthetic_share"] = syn_share
    table["mobility_share"] = mob_share

    denom = float(np.linalg.norm(syn_share) * np.linalg.norm(mob_share))
    cosine = float(np.dot(syn_share, mob_share) / denom) if denom > 0.0 else float("nan")
    tvd = float(0.5 * np.abs(syn_share - mob_share).sum())
    summary = {
        "synthetic_n": int(len(syn)),
        "mobility_n": int(len(mob)),
        "synthetic_median_km": float(np.median(syn_km)) if len(syn_km) else float("nan"),
        "mobility_median_km": float(np.median(mob_km)) if len(mob_km) else float("nan"),
        "synthetic_p90_km": float(np.quantile(syn_km, 0.9)) if len(syn_km) else float("nan"),
        "mobility_p90_km": float(np.quantile(mob_km, 0.9)) if len(mob_km) else float("nan"),
        "cosine_share": cosine,
        "tvd_share": tvd,
    }
    return table, summary
