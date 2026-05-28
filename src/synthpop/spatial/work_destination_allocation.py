from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


_INTERVAL_RE = re.compile(r"[\[\(]\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([^)\]]+)\s*[\)\]]")
_JOB_FAMILY_LABELS = ("JF_SERVICE", "JF_INDUSTRIAL", "JF_PROFESSIONAL")


def _canon_geoid_series(s: pd.Series, *, width: int) -> pd.Series:
    out = s.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    missing = out.isna() | out.str.lower().isin({"", "nan", "none", "<na>"})
    numeric = out.str.fullmatch(r"\d+").fillna(False)
    out.loc[numeric] = out.loc[numeric].str.zfill(int(width))
    out.loc[missing] = pd.NA
    return out


def _resolve_work_eligible_mask(
    *,
    persons: pd.DataFrame,
    work_eligible_col: str | None,
    work_eligible_values: list[str] | None,
) -> pd.Series:
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
    raise ValueError("work_eligible_values must be provided when work_eligible_col is non-boolean and non-numeric")


def _interval_midpoint(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    m = _INTERVAL_RE.search(s)
    if not m:
        return None
    lo = float(m.group(1))
    hi_s = m.group(2).strip().lower()
    if hi_s in {"inf", "+inf", "infinity"}:
        hi = lo + 50_000.0
    else:
        hi = float(hi_s)
    return 0.5 * (lo + hi)


def _map_earn_value_to_lodes_segment(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "not_in_earnings_universe"}:
        return None
    low = s.lower()
    if low in {"lt_25k", "lt25k"}:
        return "CE01"
    if low in {"25k_50k", "25k-50k"}:
        return "CE02"
    if low in {"50k_75k", "50k-75k", "75k_100k", "75k-100k", "ge_100k", "ge100k"}:
        return "CE03"
    mid = _interval_midpoint(s)
    if mid is None:
        try:
            mid = float(s)
        except Exception:
            return None
    if mid < 15_000.0:
        return "CE01"
    if mid < 40_000.0:
        return "CE02"
    return "CE03"


def _map_earn_series_to_lodes_segment(series: pd.Series) -> pd.Series:
    uniq = pd.Series(series.astype(object).drop_duplicates().tolist(), dtype=object)
    mapper = {value: _map_earn_value_to_lodes_segment(value) for value in uniq.tolist()}
    return series.map(mapper)


def _map_age_value_to_lodes_segment(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    mid = _interval_midpoint(s)
    if mid is None:
        try:
            mid = float(s)
        except Exception:
            return None
    if mid < 30.0:
        return "CA01"
    if mid < 55.0:
        return "CA02"
    return "CA03"


def _map_age_series_to_lodes_segment(series: pd.Series) -> pd.Series:
    uniq = pd.Series(series.astype(object).drop_duplicates().tolist(), dtype=object)
    mapper = {value: _map_age_value_to_lodes_segment(value) for value in uniq.tolist()}
    return series.map(mapper)


def _integerize_vector(weights: np.ndarray, total: int) -> np.ndarray:
    total = int(total)
    if total <= 0:
        return np.zeros_like(np.asarray(weights, dtype=float), dtype=int)
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0.0:
        w = np.ones_like(w, dtype=float)
        s = float(w.sum())
    expected = w / s * float(total)
    base = np.floor(expected).astype(int)
    rem = int(total - base.sum())
    if rem > 0:
        frac = expected - base
        order = np.argsort(-frac, kind="stable")
        base[order[:rem]] += 1
    return base


def _sinkhorn_plan(
    prior: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
    *,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> np.ndarray:
    K = np.clip(np.asarray(prior, dtype=float), 1e-12, None)
    r = np.asarray(row_target, dtype=float)
    c = np.asarray(col_target, dtype=float)
    if K.ndim != 2:
        raise ValueError("prior must be 2D")
    if K.shape != (int(r.shape[0]), int(c.shape[0])):
        raise ValueError("prior shape must match row_target x col_target")
    if float(r.sum()) <= 0.0 or float(c.sum()) <= 0.0:
        return np.zeros_like(K, dtype=float)
    u = np.ones((K.shape[0],), dtype=float)
    v = np.ones((K.shape[1],), dtype=float)
    for _ in range(int(max_iter)):
        Kv = K @ v
        Kv = np.clip(Kv, 1e-12, None)
        u = r / Kv
        KTu = K.T @ u
        KTu = np.clip(KTu, 1e-12, None)
        v = c / KTu
        plan = (u[:, None] * K) * v[None, :]
        row_err = float(np.max(np.abs(plan.sum(axis=1) - r))) if plan.size else 0.0
        col_err = float(np.max(np.abs(plan.sum(axis=0) - c))) if plan.size else 0.0
        if max(row_err, col_err) <= float(tol):
            break
    return (u[:, None] * K) * v[None, :]


def _integerize_plan_with_margins(plan: np.ndarray, row_target: np.ndarray, col_target: np.ndarray) -> np.ndarray:
    x = np.asarray(plan, dtype=float)
    r = np.asarray(row_target, dtype=int)
    c = np.asarray(col_target, dtype=int)
    base = np.floor(np.clip(x, 0.0, None)).astype(int)
    row_res = r - base.sum(axis=1)
    col_res = c - base.sum(axis=0)
    if np.any(row_res < 0) or np.any(col_res < 0):
        raise ValueError("floor(plan) exceeded row or column targets")
    frac = np.clip(x - base, 0.0, None)
    remain = int(row_res.sum())
    if remain != int(col_res.sum()):
        raise ValueError("row/column residual totals do not match")
    for _ in range(remain):
        mask = (row_res[:, None] > 0) & (col_res[None, :] > 0)
        if not bool(mask.any()):
            break
        score = np.where(mask, frac, -1.0)
        i, j = np.unravel_index(int(np.argmax(score)), score.shape)
        base[i, j] += 1
        row_res[i] -= 1
        col_res[j] -= 1
        frac[i, j] = 0.0
    if int(row_res.sum()) != 0 or int(col_res.sum()) != 0:
        raise ValueError("failed to integerize plan while preserving margins")
    return base


def _normalize_multiplier_map(spec: dict[str, Any] | None) -> dict[str, float]:
    if not spec:
        return {}
    out: dict[str, float] = {}
    for key, value in spec.items():
        k = str(key).strip()
        if not k:
            continue
        out[k] = float(value)
    return out


def _type_multiplier(
    *,
    earn_seg: Any,
    age_seg: Any,
    earn_map: dict[str, float],
    age_map: dict[str, float],
) -> float:
    mult = 1.0
    if earn_map and earn_seg is not None:
        mult *= float(earn_map.get(str(earn_seg), 1.0))
    if age_map and age_seg is not None:
        mult *= float(age_map.get(str(age_seg), 1.0))
    return float(mult)


def _infer_job_family_label(
    *,
    earn_seg: Any,
    age_seg: Any,
    schl_value: Any,
) -> str:
    scores = {
        "JF_SERVICE": 1.0,
        "JF_INDUSTRIAL": 1.0,
        "JF_PROFESSIONAL": 1.0,
    }
    earn = None if earn_seg is None else str(earn_seg)
    age = None if age_seg is None else str(age_seg)
    schl = "" if schl_value is None else str(schl_value).strip().lower()
    if earn == "CE01":
        scores["JF_SERVICE"] += 1.6
        scores["JF_INDUSTRIAL"] += 0.9
    elif earn == "CE02":
        scores["JF_INDUSTRIAL"] += 1.2
        scores["JF_SERVICE"] += 0.5
        scores["JF_PROFESSIONAL"] += 0.3
    elif earn == "CE03":
        scores["JF_PROFESSIONAL"] += 2.0
        scores["JF_INDUSTRIAL"] += 0.4
    if schl == "bachelor_plus":
        scores["JF_PROFESSIONAL"] += 2.4
    elif schl == "some_college_or_assoc":
        scores["JF_PROFESSIONAL"] += 0.8
        scores["JF_INDUSTRIAL"] += 0.8
    elif schl in {"high_school_or_ged", "less_than_high_school"}:
        scores["JF_INDUSTRIAL"] += 1.1
        scores["JF_SERVICE"] += 0.6
    elif schl == "not_25p":
        scores["JF_SERVICE"] += 0.4
    if age == "CA01":
        scores["JF_SERVICE"] += 0.3
    elif age == "CA03":
        scores["JF_PROFESSIONAL"] += 0.2
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _weighted_group_mean(
    values: np.ndarray,
    weights: np.ndarray,
    inverse: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    inv = np.asarray(inverse, dtype=int)
    num = np.bincount(inv, weights=vals * w, minlength=int(n_groups))
    den = np.bincount(inv, weights=w, minlength=int(n_groups))
    den = np.clip(den, 1e-12, None)
    return num / den


def assign_work_destination_tract(
    *,
    persons: pd.DataFrame,
    tract_od: pd.DataFrame,
    person_id_col: str = "person_id",
    home_group_col: str = "tract_geoid",
    out_col: str = "work_tract_geoid",
    work_eligible_col: str | None = None,
    work_eligible_values: list[str] | None = None,
    distance_col: str | None = "distance_km",
    distance_beta: float = 0.0,
    earn_col: str | None = None,
    age_col: str | None = None,
    schl_col: str | None = None,
    od_age_segment_weight: float = 0.0,
    od_earn_segment_weight: float = 0.0,
    destination_segment_weight: float = 0.0,
    destination_age_segment_weight: float = 0.0,
    destination_access_col: str | None = None,
    destination_access_weight: float = 0.0,
    od_pair_prior_col: str | None = None,
    od_pair_prior_weight: float = 0.0,
    destination_center_col: str | None = None,
    destination_center_weight: float = 0.0,
    same_tract_weight: float = 0.0,
    same_county_weight: float = 0.0,
    same_home_center_weight: float = 0.0,
    job_family_weight: float = 0.0,
    distance_earn_multiplier_map: dict[str, Any] | None = None,
    distance_age_multiplier_map: dict[str, Any] | None = None,
    destination_access_earn_multiplier_map: dict[str, Any] | None = None,
    destination_access_age_multiplier_map: dict[str, Any] | None = None,
    destination_center_earn_multiplier_map: dict[str, Any] | None = None,
    destination_center_age_multiplier_map: dict[str, Any] | None = None,
    same_tract_earn_multiplier_map: dict[str, Any] | None = None,
    same_tract_age_multiplier_map: dict[str, Any] | None = None,
    same_county_earn_multiplier_map: dict[str, Any] | None = None,
    same_county_age_multiplier_map: dict[str, Any] | None = None,
    same_home_center_earn_multiplier_map: dict[str, Any] | None = None,
    same_home_center_age_multiplier_map: dict[str, Any] | None = None,
    assignment_mode: str = "independent",
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(persons, pd.DataFrame):
        raise TypeError("persons must be a pandas DataFrame")
    if not isinstance(tract_od, pd.DataFrame):
        raise TypeError("tract_od must be a pandas DataFrame")
    need_person_cols = [str(person_id_col), str(home_group_col)]
    miss = [c for c in need_person_cols if c not in persons.columns]
    if miss:
        raise ValueError(f"persons missing columns: {miss}")
    need_od_cols = ["home_tract_geoid", "work_tract_geoid", "S000"]
    miss_od = [c for c in need_od_cols if c not in tract_od.columns]
    if miss_od:
        raise ValueError(f"tract_od missing columns: {miss_od}")

    out = persons.copy().reset_index(drop=True)
    out[str(home_group_col)] = _canon_geoid_series(out[str(home_group_col)], width=11)

    od = tract_od.copy()
    od["home_tract_geoid"] = _canon_geoid_series(od["home_tract_geoid"], width=11)
    od["work_tract_geoid"] = _canon_geoid_series(od["work_tract_geoid"], width=11)
    od["S000"] = pd.to_numeric(od["S000"], errors="coerce").fillna(0.0)
    od = od[od["S000"] > 0.0].copy()
    use_distance = bool(distance_col) and str(distance_col) in od.columns and float(distance_beta) > 0.0
    if use_distance:
        od[str(distance_col)] = pd.to_numeric(od[str(distance_col)], errors="coerce").fillna(0.0).clip(lower=0.0)
    age_od_cols = {seg: seg.replace("CA", "SA") for seg in ["CA01", "CA02", "CA03"]}
    earn_od_cols = {seg: seg.replace("CE", "SE") for seg in ["CE01", "CE02", "CE03"]}
    age_dest_cols = {seg: f"work_share_{seg}" for seg in ["CA01", "CA02", "CA03"]}
    earn_dest_cols = {seg: f"work_share_{seg}" for seg in ["CE01", "CE02", "CE03"]}
    use_od_age_prior = bool(age_col) and str(age_col) in out.columns and float(od_age_segment_weight) > 0.0 and all(
        col in od.columns for col in age_od_cols.values()
    )
    use_od_earn_prior = bool(earn_col) and str(earn_col) in out.columns and float(od_earn_segment_weight) > 0.0 and all(
        col in od.columns for col in earn_od_cols.values()
    )
    use_destination_earn_prior = (
        bool(earn_col)
        and str(earn_col) in out.columns
        and float(destination_segment_weight) > 0.0
        and all(col in od.columns for col in earn_dest_cols.values())
    )
    use_destination_age_prior = (
        bool(age_col)
        and str(age_col) in out.columns
        and float(destination_age_segment_weight) > 0.0
        and all(col in od.columns for col in age_dest_cols.values())
    )
    use_destination_access_prior = (
        bool(destination_access_col)
        and str(destination_access_col) in od.columns
        and float(destination_access_weight) > 0.0
    )
    use_od_pair_prior = (
        bool(od_pair_prior_col)
        and str(od_pair_prior_col) in od.columns
        and float(od_pair_prior_weight) > 0.0
    )
    use_destination_center_prior = (
        bool(destination_center_col)
        and str(destination_center_col) in od.columns
        and float(destination_center_weight) > 0.0
    )
    use_same_tract_prior = float(same_tract_weight) != 0.0
    use_same_home_center_prior = (
        float(same_home_center_weight) != 0.0
        and "work_center_geoid" in od.columns
        and "home_center_geoid" in od.columns
    )
    use_job_family_prior = (
        float(job_family_weight) > 0.0
        and any(bool(x) and str(x) in out.columns for x in [earn_col, age_col, schl_col])
        and all(f"work_share_{fam}" in od.columns for fam in _JOB_FAMILY_LABELS)
    )
    if use_destination_earn_prior:
        for col in earn_dest_cols.values():
            od[col] = pd.to_numeric(od[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    if use_destination_age_prior:
        for col in age_dest_cols.values():
            od[col] = pd.to_numeric(od[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    if use_destination_access_prior:
        od[str(destination_access_col)] = pd.to_numeric(od[str(destination_access_col)], errors="coerce").fillna(0.0).clip(lower=0.0)
    if use_od_pair_prior:
        od[str(od_pair_prior_col)] = pd.to_numeric(od[str(od_pair_prior_col)], errors="coerce").fillna(0.0).clip(lower=0.0)
    if use_destination_center_prior:
        od[str(destination_center_col)] = pd.to_numeric(od[str(destination_center_col)], errors="coerce").fillna(0.0).clip(lower=0.0)
    if use_od_earn_prior:
        for col in earn_od_cols.values():
            od[col] = pd.to_numeric(od[col], errors="coerce").fillna(0.0)
    if use_od_age_prior:
        for col in age_od_cols.values():
            od[col] = pd.to_numeric(od[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    distance_earn_map = _normalize_multiplier_map(distance_earn_multiplier_map)
    distance_age_map = _normalize_multiplier_map(distance_age_multiplier_map)
    access_earn_map = _normalize_multiplier_map(destination_access_earn_multiplier_map)
    access_age_map = _normalize_multiplier_map(destination_access_age_multiplier_map)
    center_earn_map = _normalize_multiplier_map(destination_center_earn_multiplier_map)
    center_age_map = _normalize_multiplier_map(destination_center_age_multiplier_map)
    same_tract_earn_map = _normalize_multiplier_map(same_tract_earn_multiplier_map)
    same_tract_age_map = _normalize_multiplier_map(same_tract_age_multiplier_map)
    county_earn_map = _normalize_multiplier_map(same_county_earn_multiplier_map)
    county_age_map = _normalize_multiplier_map(same_county_age_multiplier_map)
    home_center_earn_map = _normalize_multiplier_map(same_home_center_earn_multiplier_map)
    home_center_age_map = _normalize_multiplier_map(same_home_center_age_multiplier_map)
    use_earn_type_coefficients = bool(
        distance_earn_map or access_earn_map or center_earn_map or same_tract_earn_map or county_earn_map or home_center_earn_map
    )
    use_age_type_coefficients = bool(
        distance_age_map or access_age_map or center_age_map or same_tract_age_map or county_age_map or home_center_age_map
    )
    mode = str(assignment_mode).strip().lower()
    if mode not in {"independent", "balanced", "hierarchical_county", "hierarchical_county_center", "hierarchical_regime"}:
        raise ValueError(
            "assignment_mode must be one of: independent, balanced, hierarchical_county, hierarchical_county_center, hierarchical_regime"
        )
    use_center_hierarchy = mode == "hierarchical_county_center"
    if use_center_hierarchy and "work_center_geoid" not in od.columns:
        raise ValueError("tract_od missing work_center_geoid required by hierarchical_county_center mode")
    if use_center_hierarchy:
        od["work_center_geoid"] = od["work_center_geoid"].fillna(od["work_tract_geoid"]).astype(str)
    if use_same_home_center_prior:
        od["work_center_geoid"] = od["work_center_geoid"].fillna(od["work_tract_geoid"]).astype(str)
        od["home_center_geoid"] = od["home_center_geoid"].fillna(od["home_tract_geoid"]).astype(str)

    rng = np.random.default_rng(int(seed))
    pool: dict[str, dict[str, Any]] = {}
    for home, grp in od.groupby("home_tract_geoid", sort=False):
        base = grp["S000"].to_numpy(dtype=float)
        total = float(base.sum())
        if total <= 0.0:
            continue
        dests = grp["work_tract_geoid"].astype(str).to_numpy()
        item: dict[str, Any] = {
            "dests": dests,
            "base_probs": base / total,
        }
        if use_distance:
            item["distance_km"] = grp[str(distance_col)].to_numpy(dtype=float)
        if use_destination_earn_prior:
            for seg, col in earn_dest_cols.items():
                item[f"dest_share_{seg}"] = grp[col].to_numpy(dtype=float)
        if use_destination_age_prior:
            for seg, col in age_dest_cols.items():
                item[f"dest_share_{seg}"] = grp[col].to_numpy(dtype=float)
        if use_destination_access_prior:
            item["dest_access"] = grp[str(destination_access_col)].to_numpy(dtype=float)
        if use_od_pair_prior:
            item["od_pair_prior"] = grp[str(od_pair_prior_col)].to_numpy(dtype=float)
        if use_destination_center_prior:
            item["dest_center_access"] = grp[str(destination_center_col)].to_numpy(dtype=float)
        if use_job_family_prior:
            for fam in _JOB_FAMILY_LABELS:
                item[f"dest_share_{fam}"] = grp[f"work_share_{fam}"].to_numpy(dtype=float)
        if use_same_tract_prior or mode == "hierarchical_regime":
            item["same_tract"] = (grp["work_tract_geoid"].astype(str).to_numpy() == str(home)).astype(float)
        if float(same_county_weight) != 0.0:
            item["same_county"] = (
                grp["work_tract_geoid"].astype(str).str.slice(0, 5).to_numpy()
                == str(home)[:5]
            ).astype(float)
        if use_same_home_center_prior:
            item["same_home_center"] = (
                grp["work_center_geoid"].astype(str).to_numpy()
                == grp["home_center_geoid"].astype(str).to_numpy()
            ).astype(float)
        if use_od_earn_prior:
            denom = np.clip(grp["S000"].to_numpy(dtype=float), 1e-6, None)
            for seg, col in earn_od_cols.items():
                item[f"od_share_{seg}"] = grp[col].to_numpy(dtype=float) / denom
        if use_od_age_prior:
            denom = np.clip(grp["S000"].to_numpy(dtype=float), 1e-6, None)
            for seg, col in age_od_cols.items():
                item[f"od_share_{seg}"] = grp[col].to_numpy(dtype=float) / denom
        counties = pd.Series(dests, dtype=object).astype(str).str.slice(0, 5).to_numpy()
        uniq_counties, county_inv = np.unique(counties, return_inverse=True)
        county_base = np.bincount(county_inv, weights=item["base_probs"], minlength=int(uniq_counties.shape[0]))
        county_base = np.clip(county_base, 1e-12, None)
        county_item: dict[str, Any] = {
            "counties": uniq_counties.astype(object),
            "base_probs": county_base / float(county_base.sum()),
            "tract_indices": [np.flatnonzero(county_inv == i).astype(int) for i in range(int(uniq_counties.shape[0]))],
        }
        if use_distance and "distance_km" in item:
            county_item["distance_km"] = _weighted_group_mean(
                item["distance_km"],
                item["base_probs"],
                county_inv,
                int(uniq_counties.shape[0]),
            )
        if use_destination_earn_prior:
            for seg in ["CE01", "CE02", "CE03"]:
                seg_share = item.get(f"dest_share_{seg}")
                if seg_share is not None:
                    county_item[f"dest_share_{seg}"] = _weighted_group_mean(
                        seg_share,
                        item["base_probs"],
                        county_inv,
                        int(uniq_counties.shape[0]),
                    )
        if use_destination_age_prior:
            for seg in ["CA01", "CA02", "CA03"]:
                seg_share = item.get(f"dest_share_{seg}")
                if seg_share is not None:
                    county_item[f"dest_share_{seg}"] = _weighted_group_mean(
                        seg_share,
                        item["base_probs"],
                        county_inv,
                        int(uniq_counties.shape[0]),
                    )
        if use_destination_access_prior:
            county_item["dest_access"] = _weighted_group_mean(
                item["dest_access"],
                item["base_probs"],
                county_inv,
                int(uniq_counties.shape[0]),
            )
        if use_od_pair_prior:
            county_item["od_pair_prior"] = _weighted_group_mean(
                item["od_pair_prior"],
                item["base_probs"],
                county_inv,
                int(uniq_counties.shape[0]),
            )
        if use_destination_center_prior:
            county_item["dest_center_access"] = _weighted_group_mean(
                item["dest_center_access"],
                item["base_probs"],
                county_inv,
                int(uniq_counties.shape[0]),
            )
        if float(same_county_weight) != 0.0:
            county_item["same_county"] = (uniq_counties.astype(str) == str(home)[:5]).astype(float)
        if use_center_hierarchy:
            center_ids = grp["work_center_geoid"].astype(str).to_numpy()
            county_centers: list[dict[str, Any]] = []
            for tract_idx in county_item["tract_indices"]:
                tract_idx = np.asarray(tract_idx, dtype=int)
                local_centers = center_ids[tract_idx]
                uniq_centers, center_inv = np.unique(local_centers, return_inverse=True)
                local_weights = item["base_probs"][tract_idx]
                center_base = np.bincount(
                    center_inv,
                    weights=local_weights,
                    minlength=int(uniq_centers.shape[0]),
                )
                center_base = np.clip(center_base, 1e-12, None)
                center_item: dict[str, Any] = {
                    "centers": uniq_centers.astype(object),
                    "base_probs": center_base / float(center_base.sum()),
                    "tract_indices": [
                        tract_idx[np.flatnonzero(center_inv == i).astype(int)].astype(int)
                        for i in range(int(uniq_centers.shape[0]))
                    ],
                }
                if use_same_home_center_prior:
                    home_center_geoid = str(grp["home_center_geoid"].astype(str).iloc[0])
                    center_item["same_home_center"] = (uniq_centers.astype(str) == home_center_geoid).astype(float)
                if use_distance and "distance_km" in item:
                    center_item["distance_km"] = _weighted_group_mean(
                        item["distance_km"][tract_idx],
                        local_weights,
                        center_inv,
                        int(uniq_centers.shape[0]),
                    )
                if use_destination_earn_prior:
                    for seg in ["CE01", "CE02", "CE03"]:
                        seg_share = item.get(f"dest_share_{seg}")
                        if seg_share is not None:
                            center_item[f"dest_share_{seg}"] = _weighted_group_mean(
                                seg_share[tract_idx],
                                local_weights,
                                center_inv,
                                int(uniq_centers.shape[0]),
                            )
                if use_destination_age_prior:
                    for seg in ["CA01", "CA02", "CA03"]:
                        seg_share = item.get(f"dest_share_{seg}")
                        if seg_share is not None:
                            center_item[f"dest_share_{seg}"] = _weighted_group_mean(
                                seg_share[tract_idx],
                                local_weights,
                                center_inv,
                                int(uniq_centers.shape[0]),
                            )
                if use_destination_access_prior:
                    center_item["dest_access"] = _weighted_group_mean(
                        item["dest_access"][tract_idx],
                        local_weights,
                        center_inv,
                        int(uniq_centers.shape[0]),
                    )
                if use_od_pair_prior:
                    center_item["od_pair_prior"] = _weighted_group_mean(
                        item["od_pair_prior"][tract_idx],
                        local_weights,
                        center_inv,
                        int(uniq_centers.shape[0]),
                    )
                if use_destination_center_prior:
                    center_item["dest_center_access"] = _weighted_group_mean(
                        item["dest_center_access"][tract_idx],
                        local_weights,
                        center_inv,
                        int(uniq_centers.shape[0]),
                    )
                county_centers.append(center_item)
            county_item["centers_nested"] = county_centers
        if mode == "hierarchical_regime":
            same_county_arr = item.get("same_county")
            if same_county_arr is None:
                same_county_arr = (
                    grp["work_tract_geoid"].astype(str).str.slice(0, 5).to_numpy()
                    == str(home)[:5]
                ).astype(float)
            same_home_center_arr = None
            if "work_center_geoid" in grp.columns and "home_center_geoid" in grp.columns:
                same_home_center_arr = (
                    grp["work_center_geoid"].astype(str).to_numpy()
                    == grp["home_center_geoid"].astype(str).to_numpy()
                ).astype(float)
            same_tract_arr = item.get("same_tract")
            regime = np.full((int(grp.shape[0]),), "cross_county", dtype=object)
            if same_county_arr is not None:
                regime[np.asarray(same_county_arr, dtype=float) > 0.5] = "same_county"
            if same_home_center_arr is not None:
                regime[np.asarray(same_home_center_arr, dtype=float) > 0.5] = "same_center"
            if same_tract_arr is not None:
                regime[np.asarray(same_tract_arr, dtype=float) > 0.5] = "same_tract"
            regime_levels = np.asarray(["same_tract", "same_center", "same_county", "cross_county"], dtype=object)
            regime_item: dict[str, Any] = {
                "regimes": regime_levels,
                "tract_indices": [],
                "base_probs": np.zeros((int(regime_levels.shape[0]),), dtype=float),
                "same_tract": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
                "same_home_center": np.asarray([1.0, 1.0, 0.0, 0.0], dtype=float),
                "same_county": np.asarray([1.0, 1.0, 1.0, 0.0], dtype=float),
            }
            regime_to_idx = {str(r): i for i, r in enumerate(regime_levels.tolist())}
            regime_inv = np.asarray([regime_to_idx[str(r)] for r in regime], dtype=int)
            regime_base = np.bincount(regime_inv, weights=item["base_probs"], minlength=int(regime_levels.shape[0]))
            regime_item["base_probs"] = np.clip(regime_base, 1e-12, None)
            regime_item["base_probs"] = regime_item["base_probs"] / float(regime_item["base_probs"].sum())
            for i in range(int(regime_levels.shape[0])):
                regime_item["tract_indices"].append(np.flatnonzero(regime_inv == i).astype(int))
            if use_distance and "distance_km" in item:
                regime_item["distance_km"] = _weighted_group_mean(
                    item["distance_km"],
                    item["base_probs"],
                    regime_inv,
                    int(regime_levels.shape[0]),
                )
            if use_destination_earn_prior:
                for seg in ["CE01", "CE02", "CE03"]:
                    seg_share = item.get(f"dest_share_{seg}")
                    if seg_share is not None:
                        regime_item[f"dest_share_{seg}"] = _weighted_group_mean(
                            seg_share,
                            item["base_probs"],
                            regime_inv,
                            int(regime_levels.shape[0]),
                        )
            if use_destination_age_prior:
                for seg in ["CA01", "CA02", "CA03"]:
                    seg_share = item.get(f"dest_share_{seg}")
                    if seg_share is not None:
                        regime_item[f"dest_share_{seg}"] = _weighted_group_mean(
                            seg_share,
                            item["base_probs"],
                            regime_inv,
                            int(regime_levels.shape[0]),
                        )
            if use_job_family_prior:
                for fam in _JOB_FAMILY_LABELS:
                    seg_share = item.get(f"dest_share_{fam}")
                    if seg_share is not None:
                        regime_item[f"dest_share_{fam}"] = _weighted_group_mean(
                            seg_share,
                            item["base_probs"],
                            regime_inv,
                            int(regime_levels.shape[0]),
                        )
            if use_destination_access_prior:
                regime_item["dest_access"] = _weighted_group_mean(
                    item["dest_access"],
                    item["base_probs"],
                    regime_inv,
                    int(regime_levels.shape[0]),
                )
            if use_od_pair_prior:
                regime_item["od_pair_prior"] = _weighted_group_mean(
                    item["od_pair_prior"],
                    item["base_probs"],
                    regime_inv,
                    int(regime_levels.shape[0]),
                )
            if use_destination_center_prior:
                regime_item["dest_center_access"] = _weighted_group_mean(
                    item["dest_center_access"],
                    item["base_probs"],
                    regime_inv,
                    int(regime_levels.shape[0]),
                )
            item["regime"] = regime_item
        item["county"] = county_item
        pool[str(home)] = item

    eligible = _resolve_work_eligible_mask(
        persons=out,
        work_eligible_col=(str(work_eligible_col) if work_eligible_col else None),
        work_eligible_values=work_eligible_values,
    )

    work_dest = np.full((int(out.shape[0]),), None, dtype=object)
    work_mode = np.full((int(out.shape[0]),), "ineligible", dtype=object)
    work_earn_segment = np.full((int(out.shape[0]),), None, dtype=object)
    work_age_segment = np.full((int(out.shape[0]),), None, dtype=object)
    work_job_family = np.full((int(out.shape[0]),), None, dtype=object)

    eligible_idx = out.index[eligible]
    if len(eligible_idx):
        eligible_home = out.loc[eligible_idx, str(home_group_col)].astype(str)
        eligible_frame = pd.DataFrame({"index": eligible_idx.to_numpy(dtype=int), "_home": eligible_home.to_numpy(dtype=object)})
        group_cols = ["_home"]
        if use_od_earn_prior or use_destination_earn_prior or use_earn_type_coefficients:
            earn_segments = _map_earn_series_to_lodes_segment(out.loc[eligible_idx, str(earn_col)])
            eligible_frame["_earn_segment"] = earn_segments.to_numpy(dtype=object)
            work_earn_segment[eligible_idx.to_numpy(dtype=int)] = earn_segments.astype(object).to_numpy()
            group_cols.append("_earn_segment")
        if use_od_age_prior or use_destination_age_prior or use_age_type_coefficients:
            age_segments = _map_age_series_to_lodes_segment(out.loc[eligible_idx, str(age_col)])
            eligible_frame["_age_segment"] = age_segments.to_numpy(dtype=object)
            work_age_segment[eligible_idx.to_numpy(dtype=int)] = age_segments.astype(object).to_numpy()
            group_cols.append("_age_segment")
        if use_job_family_prior:
            if schl_col and str(schl_col) in out.columns:
                schl_series = out.loc[eligible_idx, str(schl_col)].astype(object)
            else:
                schl_series = pd.Series([None] * int(len(eligible_idx)), index=eligible_idx, dtype=object)
            earn_series = (
                pd.Series(work_earn_segment[eligible_idx.to_numpy(dtype=int)], index=eligible_idx, dtype=object)
                if (use_od_earn_prior or use_destination_earn_prior or use_earn_type_coefficients or earn_col)
                else pd.Series([None] * int(len(eligible_idx)), index=eligible_idx, dtype=object)
            )
            age_series = (
                pd.Series(work_age_segment[eligible_idx.to_numpy(dtype=int)], index=eligible_idx, dtype=object)
                if (use_od_age_prior or use_destination_age_prior or use_age_type_coefficients or age_col)
                else pd.Series([None] * int(len(eligible_idx)), index=eligible_idx, dtype=object)
            )
            family_series = pd.Series(
                [
                    _infer_job_family_label(
                        earn_seg=earn_series.iloc[i],
                        age_seg=age_series.iloc[i],
                        schl_value=schl_series.iloc[i],
                    )
                    for i in range(int(len(eligible_idx)))
                ],
                index=eligible_idx,
                dtype=object,
            )
            eligible_frame["_job_family"] = family_series.to_numpy(dtype=object)
            work_job_family[eligible_idx.to_numpy(dtype=int)] = family_series.to_numpy(dtype=object)
            group_cols.append("_job_family")
        for home, grp_home in eligible_frame.groupby("_home", sort=False, dropna=False):
            home = str(home)
            item = pool.get(home)
            take_idx_all = grp_home["index"].to_numpy(dtype=int)
            if item is None:
                work_mode[take_idx_all] = "unassigned_no_destination"
                continue

            grouped_types = []
            type_group_cols = group_cols[1:]
            if not type_group_cols:
                grouped_types.append({"earn_seg": None, "age_seg": None, "indices": take_idx_all})
            else:
                for key, grp in grp_home.groupby(type_group_cols, sort=False, dropna=False):
                    if not isinstance(key, tuple):
                        key = (key,)
                    rem = list(key)
                    earn_seg = None
                    age_seg = None
                    job_family = None
                    if "_earn_segment" in group_cols:
                        earn_seg = rem.pop(0) if rem else None
                    if "_age_segment" in group_cols:
                        age_seg = rem.pop(0) if rem else None
                    if "_job_family" in group_cols:
                        job_family = rem.pop(0) if rem else None
                    grouped_types.append(
                        {
                            "earn_seg": earn_seg,
                            "age_seg": age_seg,
                            "job_family": job_family,
                            "indices": grp["index"].to_numpy(dtype=int),
                        }
                    )

            def _weights_for_type(earn_seg: Any, age_seg: Any, job_family: Any) -> np.ndarray:
                weights = item["base_probs"].astype(float).copy()
                if use_distance and "distance_km" in item:
                    dist_mult = _type_multiplier(
                        earn_seg=earn_seg,
                        age_seg=age_seg,
                        earn_map=distance_earn_map,
                        age_map=distance_age_map,
                    )
                    weights = weights * np.exp(-(float(distance_beta) * dist_mult) * item["distance_km"].astype(float))
                if use_od_earn_prior and earn_seg in {"CE01", "CE02", "CE03"}:
                    od_share = item.get(f"od_share_{str(earn_seg)}")
                    if od_share is not None:
                        weights = weights * np.power(np.clip(od_share.astype(float), 1e-6, None), float(od_earn_segment_weight))
                if use_od_age_prior and age_seg in {"CA01", "CA02", "CA03"}:
                    od_share = item.get(f"od_share_{str(age_seg)}")
                    if od_share is not None:
                        weights = weights * np.power(np.clip(od_share.astype(float), 1e-6, None), float(od_age_segment_weight))
                if use_destination_earn_prior and earn_seg in {"CE01", "CE02", "CE03"}:
                    seg_share = item.get(f"dest_share_{str(earn_seg)}")
                    if seg_share is not None:
                        weights = weights * np.power(np.clip(seg_share.astype(float), 1e-6, None), float(destination_segment_weight))
                if use_destination_age_prior and age_seg in {"CA01", "CA02", "CA03"}:
                    seg_share = item.get(f"dest_share_{str(age_seg)}")
                    if seg_share is not None:
                        weights = weights * np.power(np.clip(seg_share.astype(float), 1e-6, None), float(destination_age_segment_weight))
                if use_job_family_prior and job_family in _JOB_FAMILY_LABELS:
                    fam_share = item.get(f"dest_share_{str(job_family)}")
                    if fam_share is not None:
                        weights = weights * np.power(np.clip(fam_share.astype(float), 1e-6, None), float(job_family_weight))
                if use_destination_access_prior:
                    access = item.get("dest_access")
                    if access is not None:
                        access_rel = np.clip(access.astype(float) / max(float(np.mean(access.astype(float))), 1e-6), 1e-6, None)
                        access_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=access_earn_map,
                            age_map=access_age_map,
                        )
                        weights = weights * np.power(access_rel, float(destination_access_weight) * access_mult)
                if use_od_pair_prior:
                    pair_prior = item.get("od_pair_prior")
                    if pair_prior is not None:
                        weights = weights * np.power(np.clip(pair_prior.astype(float), 1e-6, None), float(od_pair_prior_weight))
                if use_destination_center_prior:
                    center_access = item.get("dest_center_access")
                    if center_access is not None:
                        center_rel = np.clip(center_access.astype(float) / max(float(np.mean(center_access.astype(float))), 1e-6), 1e-6, None)
                        center_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=center_earn_map,
                            age_map=center_age_map,
                        )
                        weights = weights * np.power(center_rel, float(destination_center_weight) * center_mult)
                if use_same_tract_prior:
                    same_tract = item.get("same_tract")
                    if same_tract is not None:
                        same_tract_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=same_tract_earn_map,
                            age_map=same_tract_age_map,
                        )
                        weights = weights * np.exp((float(same_tract_weight) * same_tract_mult) * same_tract.astype(float))
                if float(same_county_weight) != 0.0:
                    same_county = item.get("same_county")
                    if same_county is not None:
                        county_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=county_earn_map,
                            age_map=county_age_map,
                        )
                        weights = weights * np.exp((float(same_county_weight) * county_mult) * same_county.astype(float))
                if use_same_home_center_prior:
                    same_home_center = item.get("same_home_center")
                    if same_home_center is not None:
                        home_center_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=home_center_earn_map,
                            age_map=home_center_age_map,
                        )
                        weights = weights * np.exp(
                            (float(same_home_center_weight) * home_center_mult) * same_home_center.astype(float)
                        )
                return np.clip(weights.astype(float), 1e-12, None)

            def _county_weights_for_type(county_item_local: dict[str, Any], earn_seg: Any, age_seg: Any, job_family: Any) -> np.ndarray:
                weights = county_item_local["base_probs"].astype(float).copy()
                if use_distance and "distance_km" in county_item_local:
                    dist_mult = _type_multiplier(
                        earn_seg=earn_seg,
                        age_seg=age_seg,
                        earn_map=distance_earn_map,
                        age_map=distance_age_map,
                    )
                    weights = weights * np.exp(
                        -(float(distance_beta) * dist_mult) * county_item_local["distance_km"].astype(float)
                    )
                if use_destination_earn_prior and earn_seg in {"CE01", "CE02", "CE03"}:
                    seg_share = county_item_local.get(f"dest_share_{str(earn_seg)}")
                    if seg_share is not None:
                        weights = weights * np.power(
                            np.clip(seg_share.astype(float), 1e-6, None),
                            float(destination_segment_weight),
                        )
                if use_destination_age_prior and age_seg in {"CA01", "CA02", "CA03"}:
                    seg_share = county_item_local.get(f"dest_share_{str(age_seg)}")
                    if seg_share is not None:
                        weights = weights * np.power(
                            np.clip(seg_share.astype(float), 1e-6, None),
                            float(destination_age_segment_weight),
                        )
                if use_job_family_prior and job_family in _JOB_FAMILY_LABELS:
                    fam_share = county_item_local.get(f"dest_share_{str(job_family)}")
                    if fam_share is not None:
                        weights = weights * np.power(
                            np.clip(fam_share.astype(float), 1e-6, None),
                            float(job_family_weight),
                        )
                if use_destination_access_prior:
                    access = county_item_local.get("dest_access")
                    if access is not None:
                        access_rel = np.clip(access.astype(float) / max(float(np.mean(access.astype(float))), 1e-6), 1e-6, None)
                        access_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=access_earn_map,
                            age_map=access_age_map,
                        )
                        weights = weights * np.power(access_rel, float(destination_access_weight) * access_mult)
                if use_od_pair_prior:
                    pair_prior = county_item_local.get("od_pair_prior")
                    if pair_prior is not None:
                        weights = weights * np.power(np.clip(pair_prior.astype(float), 1e-6, None), float(od_pair_prior_weight))
                if use_destination_center_prior:
                    center_access = county_item_local.get("dest_center_access")
                    if center_access is not None:
                        center_rel = np.clip(
                            center_access.astype(float) / max(float(np.mean(center_access.astype(float))), 1e-6),
                            1e-6,
                            None,
                        )
                        center_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=center_earn_map,
                            age_map=center_age_map,
                        )
                        weights = weights * np.power(center_rel, float(destination_center_weight) * center_mult)
                if float(same_county_weight) != 0.0:
                    same_county = county_item_local.get("same_county")
                    if same_county is not None:
                        county_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=county_earn_map,
                            age_map=county_age_map,
                        )
                        weights = weights * np.exp((float(same_county_weight) * county_mult) * same_county.astype(float))
                return np.clip(weights.astype(float), 1e-12, None)

            def _center_weights_for_type(center_item_local: dict[str, Any], earn_seg: Any, age_seg: Any, job_family: Any) -> np.ndarray:
                weights = center_item_local["base_probs"].astype(float).copy()
                if use_distance and "distance_km" in center_item_local:
                    dist_mult = _type_multiplier(
                        earn_seg=earn_seg,
                        age_seg=age_seg,
                        earn_map=distance_earn_map,
                        age_map=distance_age_map,
                    )
                    weights = weights * np.exp(
                        -(float(distance_beta) * dist_mult) * center_item_local["distance_km"].astype(float)
                    )
                if use_destination_earn_prior and earn_seg in {"CE01", "CE02", "CE03"}:
                    seg_share = center_item_local.get(f"dest_share_{str(earn_seg)}")
                    if seg_share is not None:
                        weights = weights * np.power(
                            np.clip(seg_share.astype(float), 1e-6, None),
                            float(destination_segment_weight),
                        )
                if use_destination_age_prior and age_seg in {"CA01", "CA02", "CA03"}:
                    seg_share = center_item_local.get(f"dest_share_{str(age_seg)}")
                    if seg_share is not None:
                        weights = weights * np.power(
                            np.clip(seg_share.astype(float), 1e-6, None),
                            float(destination_age_segment_weight),
                        )
                if use_job_family_prior and job_family in _JOB_FAMILY_LABELS:
                    fam_share = center_item_local.get(f"dest_share_{str(job_family)}")
                    if fam_share is not None:
                        weights = weights * np.power(
                            np.clip(fam_share.astype(float), 1e-6, None),
                            float(job_family_weight),
                        )
                if use_destination_access_prior:
                    access = center_item_local.get("dest_access")
                    if access is not None:
                        access_rel = np.clip(access.astype(float) / max(float(np.mean(access.astype(float))), 1e-6), 1e-6, None)
                        access_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=access_earn_map,
                            age_map=access_age_map,
                        )
                        weights = weights * np.power(access_rel, float(destination_access_weight) * access_mult)
                if use_od_pair_prior:
                    pair_prior = center_item_local.get("od_pair_prior")
                    if pair_prior is not None:
                        weights = weights * np.power(np.clip(pair_prior.astype(float), 1e-6, None), float(od_pair_prior_weight))
                if use_destination_center_prior:
                    center_access = center_item_local.get("dest_center_access")
                    if center_access is not None:
                        center_rel = np.clip(
                            center_access.astype(float) / max(float(np.mean(center_access.astype(float))), 1e-6),
                            1e-6,
                            None,
                        )
                        center_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=center_earn_map,
                            age_map=center_age_map,
                        )
                        weights = weights * np.power(center_rel, float(destination_center_weight) * center_mult)
                if use_same_home_center_prior:
                    same_home_center = center_item_local.get("same_home_center")
                    if same_home_center is not None:
                        home_center_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=home_center_earn_map,
                            age_map=home_center_age_map,
                        )
                        weights = weights * np.exp(
                            (float(same_home_center_weight) * home_center_mult) * same_home_center.astype(float)
                        )
                return np.clip(weights.astype(float), 1e-12, None)

            def _regime_weights_for_type(regime_item_local: dict[str, Any], earn_seg: Any, age_seg: Any, job_family: Any) -> np.ndarray:
                weights = regime_item_local["base_probs"].astype(float).copy()
                if use_distance and "distance_km" in regime_item_local:
                    dist_mult = _type_multiplier(
                        earn_seg=earn_seg,
                        age_seg=age_seg,
                        earn_map=distance_earn_map,
                        age_map=distance_age_map,
                    )
                    weights = weights * np.exp(
                        -(float(distance_beta) * dist_mult) * regime_item_local["distance_km"].astype(float)
                    )
                if use_destination_earn_prior and earn_seg in {"CE01", "CE02", "CE03"}:
                    seg_share = regime_item_local.get(f"dest_share_{str(earn_seg)}")
                    if seg_share is not None:
                        weights = weights * np.power(np.clip(seg_share.astype(float), 1e-6, None), float(destination_segment_weight))
                if use_destination_age_prior and age_seg in {"CA01", "CA02", "CA03"}:
                    seg_share = regime_item_local.get(f"dest_share_{str(age_seg)}")
                    if seg_share is not None:
                        weights = weights * np.power(np.clip(seg_share.astype(float), 1e-6, None), float(destination_age_segment_weight))
                if use_job_family_prior and job_family in _JOB_FAMILY_LABELS:
                    fam_share = regime_item_local.get(f"dest_share_{str(job_family)}")
                    if fam_share is not None:
                        weights = weights * np.power(np.clip(fam_share.astype(float), 1e-6, None), float(job_family_weight))
                if use_destination_access_prior:
                    access = regime_item_local.get("dest_access")
                    if access is not None:
                        access_rel = np.clip(access.astype(float) / max(float(np.mean(access.astype(float))), 1e-6), 1e-6, None)
                        access_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=access_earn_map,
                            age_map=access_age_map,
                        )
                        weights = weights * np.power(access_rel, float(destination_access_weight) * access_mult)
                if use_od_pair_prior:
                    pair_prior = regime_item_local.get("od_pair_prior")
                    if pair_prior is not None:
                        weights = weights * np.power(np.clip(pair_prior.astype(float), 1e-6, None), float(od_pair_prior_weight))
                if use_destination_center_prior:
                    center_access = regime_item_local.get("dest_center_access")
                    if center_access is not None:
                        center_rel = np.clip(center_access.astype(float) / max(float(np.mean(center_access.astype(float))), 1e-6), 1e-6, None)
                        center_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=center_earn_map,
                            age_map=center_age_map,
                        )
                        weights = weights * np.power(center_rel, float(destination_center_weight) * center_mult)
                if use_same_tract_prior:
                    same_tract = regime_item_local.get("same_tract")
                    if same_tract is not None:
                        same_tract_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=same_tract_earn_map,
                            age_map=same_tract_age_map,
                        )
                        weights = weights * np.exp((float(same_tract_weight) * same_tract_mult) * same_tract.astype(float))
                if float(same_county_weight) != 0.0:
                    same_county = regime_item_local.get("same_county")
                    if same_county is not None:
                        county_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=county_earn_map,
                            age_map=county_age_map,
                        )
                        weights = weights * np.exp((float(same_county_weight) * county_mult) * same_county.astype(float))
                if use_same_home_center_prior:
                    same_home_center = regime_item_local.get("same_home_center")
                    if same_home_center is not None:
                        home_center_mult = _type_multiplier(
                            earn_seg=earn_seg,
                            age_seg=age_seg,
                            earn_map=home_center_earn_map,
                            age_map=home_center_age_map,
                        )
                        weights = weights * np.exp((float(same_home_center_weight) * home_center_mult) * same_home_center.astype(float))
                return np.clip(weights.astype(float), 1e-12, None)

            if mode == "independent":
                for info in grouped_types:
                    take_idx = info["indices"]
                    weights = _weights_for_type(info["earn_seg"], info["age_seg"], info.get("job_family"))
                    total_w = float(weights.sum())
                    probs = (weights / total_w) if total_w > 0.0 else item["base_probs"]
                    sampled = rng.choice(item["dests"], size=int(take_idx.shape[0]), p=probs)
                    work_dest[take_idx] = sampled.astype(object)
                    work_mode[take_idx] = "sampled_from_od"
            elif mode in {"hierarchical_county", "hierarchical_county_center"}:
                county_item = item["county"]
                for info in grouped_types:
                    take_idx = info["indices"]
                    county_weights = _county_weights_for_type(county_item, info["earn_seg"], info["age_seg"], info.get("job_family"))
                    county_probs = county_weights / float(county_weights.sum())
                    chosen_county_idx = rng.choice(
                        np.arange(county_item["counties"].shape[0], dtype=int),
                        size=int(take_idx.shape[0]),
                        p=county_probs,
                    )
                    for county_idx in np.unique(chosen_county_idx):
                        mask = chosen_county_idx == int(county_idx)
                        idxs = take_idx[mask]
                        if mode == "hierarchical_county_center":
                            center_item = county_item["centers_nested"][int(county_idx)]
                            center_weights = _center_weights_for_type(center_item, info["earn_seg"], info["age_seg"], info.get("job_family"))
                            center_probs = center_weights / float(center_weights.sum())
                            chosen_center_idx = rng.choice(
                                np.arange(center_item["centers"].shape[0], dtype=int),
                                size=int(idxs.shape[0]),
                                p=center_probs,
                            )
                            for center_idx in np.unique(chosen_center_idx):
                                center_mask = chosen_center_idx == int(center_idx)
                                center_take = idxs[center_mask]
                                tract_idx = center_item["tract_indices"][int(center_idx)]
                                tract_weights = _weights_for_type(info["earn_seg"], info["age_seg"], info.get("job_family"))[tract_idx]
                                tract_probs = tract_weights / float(tract_weights.sum())
                                sampled = rng.choice(item["dests"][tract_idx], size=int(center_take.shape[0]), p=tract_probs)
                                work_dest[center_take] = sampled.astype(object)
                                work_mode[center_take] = "sampled_from_od"
                        else:
                            tract_idx = county_item["tract_indices"][int(county_idx)]
                            tract_weights = _weights_for_type(info["earn_seg"], info["age_seg"], info.get("job_family"))[tract_idx]
                            tract_probs = tract_weights / float(tract_weights.sum())
                            sampled = rng.choice(item["dests"][tract_idx], size=int(idxs.shape[0]), p=tract_probs)
                            work_dest[idxs] = sampled.astype(object)
                            work_mode[idxs] = "sampled_from_od"
            elif mode == "hierarchical_regime":
                regime_item = item["regime"]
                for info in grouped_types:
                    take_idx = info["indices"]
                    regime_weights = _regime_weights_for_type(regime_item, info["earn_seg"], info["age_seg"], info.get("job_family"))
                    regime_probs = regime_weights / float(regime_weights.sum())
                    chosen_regime_idx = rng.choice(
                        np.arange(regime_item["regimes"].shape[0], dtype=int),
                        size=int(take_idx.shape[0]),
                        p=regime_probs,
                    )
                    for regime_idx in np.unique(chosen_regime_idx):
                        mask = chosen_regime_idx == int(regime_idx)
                        idxs = take_idx[mask]
                        tract_idx = regime_item["tract_indices"][int(regime_idx)]
                        tract_weights = _weights_for_type(info["earn_seg"], info["age_seg"], info.get("job_family"))[tract_idx]
                        tract_probs = tract_weights / float(tract_weights.sum())
                        sampled = rng.choice(item["dests"][tract_idx], size=int(idxs.shape[0]), p=tract_probs)
                        work_dest[idxs] = sampled.astype(object)
                        work_mode[idxs] = "sampled_from_od"
            else:
                row_target = np.asarray([int(info["indices"].shape[0]) for info in grouped_types], dtype=int)
                col_target = _integerize_vector(item["base_probs"], int(take_idx_all.shape[0]))
                prior = np.vstack([_weights_for_type(info["earn_seg"], info["age_seg"], info.get("job_family")) for info in grouped_types])
                plan = _sinkhorn_plan(prior, row_target.astype(float), col_target.astype(float))
                plan_i = _integerize_plan_with_margins(plan, row_target=row_target, col_target=col_target)
                for ridx, info in enumerate(grouped_types):
                    idxs = info["indices"].copy()
                    rng.shuffle(idxs)
                    alloc = plan_i[ridx]
                    dests = np.repeat(item["dests"], alloc.astype(int))
                    rng.shuffle(dests)
                    if int(dests.shape[0]) != int(idxs.shape[0]):
                        raise ValueError("balanced assignment produced mismatched row total")
                    work_dest[idxs] = dests.astype(object)
                    work_mode[idxs] = "sampled_from_od"

    out[str(out_col)] = work_dest.tolist()
    out["work_destination_mode"] = work_mode.tolist()
    out["work_destination_unassigned_flag"] = pd.isna(out[str(out_col)])
    out["work_destination_earn_segment"] = work_earn_segment.tolist()
    out["work_destination_age_segment"] = work_age_segment.tolist()
    out["work_destination_job_family"] = work_job_family.tolist()

    sampled = out[out["work_destination_mode"] == "sampled_from_od"].copy()
    same_tract_share = (
        float((sampled[str(home_group_col)].astype(str) == sampled[str(out_col)].astype(str)).mean())
        if len(sampled)
        else float("nan")
    )
    meta = {
        "person_id_col": str(person_id_col),
        "home_group_col": str(home_group_col),
        "out_col": str(out_col),
        "distance_col": (str(distance_col) if distance_col else None),
        "distance_beta": float(distance_beta),
        "earn_col": (str(earn_col) if earn_col else None),
        "age_col": (str(age_col) if age_col else None),
        "schl_col": (str(schl_col) if schl_col else None),
        "od_age_segment_weight": float(od_age_segment_weight),
        "od_earn_segment_weight": float(od_earn_segment_weight),
        "destination_segment_weight": float(destination_segment_weight),
        "destination_age_segment_weight": float(destination_age_segment_weight),
        "destination_access_col": (str(destination_access_col) if destination_access_col else None),
        "destination_access_weight": float(destination_access_weight),
        "od_pair_prior_col": (str(od_pair_prior_col) if od_pair_prior_col else None),
        "od_pair_prior_weight": float(od_pair_prior_weight),
        "destination_center_col": (str(destination_center_col) if destination_center_col else None),
        "destination_center_weight": float(destination_center_weight),
        "same_tract_weight": float(same_tract_weight),
        "same_county_weight": float(same_county_weight),
        "same_home_center_weight": float(same_home_center_weight),
        "job_family_weight": float(job_family_weight),
        "distance_earn_multiplier_map": distance_earn_map,
        "distance_age_multiplier_map": distance_age_map,
        "destination_access_earn_multiplier_map": access_earn_map,
        "destination_access_age_multiplier_map": access_age_map,
        "destination_center_earn_multiplier_map": center_earn_map,
        "destination_center_age_multiplier_map": center_age_map,
        "same_tract_earn_multiplier_map": same_tract_earn_map,
        "same_tract_age_multiplier_map": same_tract_age_map,
        "same_county_earn_multiplier_map": county_earn_map,
        "same_county_age_multiplier_map": county_age_map,
        "same_home_center_earn_multiplier_map": home_center_earn_map,
        "same_home_center_age_multiplier_map": home_center_age_map,
        "assignment_mode": mode,
        "use_distance": bool(use_distance),
        "use_od_age_prior": bool(use_od_age_prior),
        "use_od_earn_prior": bool(use_od_earn_prior),
        "use_destination_earn_prior": bool(use_destination_earn_prior),
        "use_destination_age_prior": bool(use_destination_age_prior),
        "use_destination_access_prior": bool(use_destination_access_prior),
        "use_od_pair_prior": bool(use_od_pair_prior),
        "use_destination_center_prior": bool(use_destination_center_prior),
        "use_same_tract_prior": bool(use_same_tract_prior),
        "use_same_home_center_prior": bool(use_same_home_center_prior),
        "use_job_family_prior": bool(use_job_family_prior),
        "use_earn_type_coefficients": bool(use_earn_type_coefficients),
        "use_age_type_coefficients": bool(use_age_type_coefficients),
        "seed": int(seed),
        "n_persons": int(out.shape[0]),
        "work_eligible": int(eligible.sum()),
        "work_destination_assigned": int((out["work_destination_mode"] == "sampled_from_od").sum()),
        "work_destination_unassigned": int((out["work_destination_mode"] == "unassigned_no_destination").sum()),
        "same_tract_share_among_assigned": same_tract_share,
        "destination_mode_counts": {
            str(k): int(v)
            for k, v in out["work_destination_mode"].value_counts(dropna=False).to_dict().items()
        },
        "segment_counts_among_eligible": (
            {
                str(k): int(v)
                for k, v in pd.Series(work_earn_segment[eligible_idx.to_numpy(dtype=int)]).fillna("missing").value_counts(dropna=False).to_dict().items()
            }
            if len(eligible_idx)
            else {}
        ),
        "age_segment_counts_among_eligible": (
            {
                str(k): int(v)
                for k, v in pd.Series(work_age_segment[eligible_idx.to_numpy(dtype=int)]).fillna("missing").value_counts(dropna=False).to_dict().items()
            }
            if len(eligible_idx)
            else {}
        ),
        "job_family_counts_among_eligible": (
            {
                str(k): int(v)
                for k, v in pd.Series(work_job_family[eligible_idx.to_numpy(dtype=int)]).fillna("missing").value_counts(dropna=False).to_dict().items()
            }
            if len(eligible_idx)
            else {}
        ),
    }
    return out, meta
