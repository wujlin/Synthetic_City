#!/usr/bin/env python3
"""
Exp5: Tract-level posterior alignment (Layer-3) on top of Exp4 synthetic sample.

Core question:
- After enforcing tract-level ACS marginals, does a diffusion-derived seed still
  preserve better joint structure than a global seed?

Comparison:
1) IPF(diffusion_seed)
2) IPF(global_seed)

Inputs:
- synthetic sample from Exp4 (csv/csv.gz/parquet), with at least:
    tract_geoid, puma (or mappable), age/sex/race info, PINCP, SCHL, ESR
- ACS targets_long at tract level (from build_acs_targets_long_michigan.py)
  with variables:
    PINCP_16p_bin, ESR_16p, SCHL_25p
- PUMS microdata (for external PUMA-level validation)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import pathlib
import random
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


def _age_idx_to_midpoint(age_idx_series: Any) -> Any:
    pd = _require("pandas")
    mids = [2.0, 7.0, 12.0, 16.0, 19.0, 20.0, 21.0, 23.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 61.0, 64.0, 66.0, 69.0, 74.0, 79.0, 84.0, 90.0]
    age_idx = pd.to_numeric(age_idx_series, errors="coerce").fillna(0).astype(int).clip(lower=0, upper=22)
    return age_idx.map(lambda x: float(mids[int(x)]))


def _tvd(p: Any, q: Any) -> float:
    np = _require("numpy")
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p - q).sum())


def _weighted_rank(u: Any, w: Any) -> Any:
    np = _require("numpy")
    u = np.asarray(u, dtype=float)
    w = np.asarray(w, dtype=float)
    if u.shape != w.shape:
        raise ValueError(f"u and w must have same shape, got {u.shape} vs {w.shape}")
    mask = np.isfinite(u) & np.isfinite(w) & (w > 0)
    out = np.full(u.shape, np.nan, dtype=float)
    if not bool(mask.any()):
        return out
    u_m = u[mask]
    w_m = w[mask]
    order = np.argsort(u_m, kind="mergesort")
    w_sorted = w_m[order]
    cw = np.cumsum(w_sorted)
    tot = float(cw[-1])
    r_sorted = (cw - 0.5 * w_sorted) / max(tot, 1e-12)
    r = np.empty_like(r_sorted)
    r[order] = r_sorted
    out[mask] = np.clip(r, 0.0, 1.0)
    return out


def _copula_hist2d(*, u: Any, v: Any, w: Any, bins: int = 10) -> Any:
    np = _require("numpy")
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(u) & np.isfinite(v) & np.isfinite(w) & (w > 0)
    u = np.clip(u[mask], 0.0, 1.0)
    v = np.clip(v[mask], 0.0, 1.0)
    w = w[mask]
    if u.size == 0:
        return np.full((bins, bins), 1.0 / float(bins * bins), dtype=float)
    h, _, _ = np.histogram2d(u, v, bins=int(bins), range=[[0.0, 1.0], [0.0, 1.0]], weights=w)
    s = float(h.sum())
    if s <= 0 or not math.isfinite(s):
        return np.full((bins, bins), 1.0 / float(bins * bins), dtype=float)
    return (h / s).astype(float)


def _weighted_cat_dist(df: Any, col: str, wcol: str) -> dict[str, float]:
    pd = _require("pandas")
    s = df[[col, wcol]].copy()
    s[wcol] = pd.to_numeric(s[wcol], errors="coerce").fillna(0.0).clip(lower=0.0)
    s[col] = s[col].astype(str)
    g = s.groupby(col, dropna=False)[wcol].sum()
    tot = float(g.sum())
    if tot <= 0:
        return {}
    return {str(k): float(v / tot) for k, v in g.to_dict().items()}


def _weighted_joint_dist(df: Any, cols: list[str], wcol: str) -> dict[str, float]:
    pd = _require("pandas")
    s = df[cols + [wcol]].copy()
    s[wcol] = pd.to_numeric(s[wcol], errors="coerce").fillna(0.0).clip(lower=0.0)
    for c in cols:
        s[c] = s[c].astype(str)
    g = s.groupby(cols, dropna=False)[wcol].sum()
    tot = float(g.sum())
    if tot <= 0:
        return {}
    out: dict[str, float] = {}
    for k, v in g.to_dict().items():
        out["|".join(str(x) for x in k)] = float(v / tot)
    return out


def _tvd_from_dists(p: dict[str, float], q: dict[str, float]) -> float | None:
    if not p or not q:
        return None
    keys = sorted(set(p.keys()) | set(q.keys()))
    pv = [float(p.get(k, 0.0)) for k in keys]
    qv = [float(q.get(k, 0.0)) for k in keys]
    return _tvd(pv, qv)


def _harmonize_synthetic_columns(df: Any, *, tract_col: str, puma_col: str) -> Any:
    pd = _require("pandas")

    out = df.copy()
    lower = {str(c).lower(): str(c) for c in out.columns}

    def _pick(*cands: str) -> str | None:
        for c in cands:
            if c in lower:
                return lower[c]
        return None

    if "AGEP" not in out.columns:
        age_idx_col = _pick("age_idx", "agebin", "age_bin", "age_group_idx")
        if age_idx_col is not None:
            out["AGEP"] = _age_idx_to_midpoint(out[age_idx_col])
    if "SEX" not in out.columns:
        sex_col = _pick("sex")
        if sex_col is not None:
            out["SEX"] = pd.to_numeric(out[sex_col], errors="coerce").fillna(1).astype(int).clip(lower=1, upper=2)
    if "PINCP" not in out.columns:
        inc_col = _pick("income", "pincp")
        if inc_col is not None:
            out["PINCP"] = pd.to_numeric(out[inc_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "SCHL" not in out.columns:
        c = _pick("schl")
        if c is not None:
            out["SCHL"] = out[c]
    if "ESR" not in out.columns:
        c = _pick("esr")
        if c is not None:
            out["ESR"] = out[c]

    if tract_col not in out.columns:
        c = _pick("tract_geoid", "tract", "tractid")
        if c is not None:
            out[tract_col] = out[c].astype(str)
    if puma_col not in out.columns:
        c = _pick("puma", "puma20")
        if c is not None:
            out[puma_col] = pd.to_numeric(out[c], errors="coerce").fillna(-9).astype(int).astype(str)

    req = [tract_col, puma_col, "AGEP", "SEX", "PINCP", "SCHL", "ESR"]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise SystemExit(f"synthetic missing required columns after harmonization: {missing}")

    out[tract_col] = out[tract_col].astype(str)
    out[puma_col] = out[puma_col].astype(str)
    out["AGEP"] = pd.to_numeric(out["AGEP"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=99.0)
    out["SEX"] = pd.to_numeric(out["SEX"], errors="coerce").fillna(1).astype(int).clip(lower=1, upper=2)
    out["PINCP"] = pd.to_numeric(out["PINCP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["SCHL"] = pd.to_numeric(out["SCHL"], errors="coerce").fillna(0).astype(int).astype(str)
    out["ESR"] = pd.to_numeric(out["ESR"], errors="coerce").fillna(0).astype(int).astype(str)
    if "W" not in out.columns:
        out["W"] = 1.0
    out["W"] = pd.to_numeric(out["W"], errors="coerce").fillna(1.0).clip(lower=0.0)
    return out


def _derive_scope_columns(df: Any) -> Any:
    pd = _require("pandas")
    out = df.copy()

    age = pd.to_numeric(out["AGEP"], errors="coerce").fillna(0.0)
    inc = pd.to_numeric(out["PINCP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    esr = out["ESR"].astype(str)
    schl = pd.to_numeric(out["SCHL"], errors="coerce").fillna(0).astype(int)

    out["is_16p"] = age >= 16
    out["is_25p"] = age >= 25
    out["has_earnings_16p"] = (age >= 16) & (inc > 0)

    # ESR_16p
    esr16 = pd.Series([None] * int(out.shape[0]), index=out.index, dtype=object)
    esr16.loc[out["is_16p"]] = "not_in_labor_force"
    esr16.loc[out["is_16p"] & esr.isin({"1", "2"})] = "employed"
    esr16.loc[out["is_16p"] & esr.isin({"3"})] = "unemployed"
    esr16.loc[out["is_16p"] & esr.isin({"4", "5"})] = "armed_forces"
    out["ESR_16p"] = esr16.astype(object)

    # SCHL_25p
    schl25 = pd.Series([None] * int(out.shape[0]), index=out.index, dtype=object)
    schl25.loc[out["is_25p"]] = "less_than_high_school"
    schl25.loc[out["is_25p"] & schl.isin([19, 20])] = "high_school_or_ged"
    schl25.loc[out["is_25p"] & schl.isin([21, 22, 23])] = "some_college_or_assoc"
    schl25.loc[out["is_25p"] & (schl >= 24)] = "bachelor_plus"
    out["SCHL_25p"] = schl25.astype(object)

    # PINCP_16p_bin
    inc16 = pd.Series([None] * int(out.shape[0]), index=out.index, dtype=object)
    inc16.loc[out["has_earnings_16p"] & (inc < 25_000.0)] = "lt_25k"
    inc16.loc[out["has_earnings_16p"] & (inc >= 25_000.0) & (inc < 50_000.0)] = "25k_50k"
    inc16.loc[out["has_earnings_16p"] & (inc >= 50_000.0) & (inc < 75_000.0)] = "50k_75k"
    inc16.loc[out["has_earnings_16p"] & (inc >= 75_000.0) & (inc < 100_000.0)] = "75k_100k"
    inc16.loc[out["has_earnings_16p"] & (inc >= 100_000.0)] = "ge_100k"
    out["PINCP_16p_bin"] = inc16.astype(object)
    return out


def _build_global_seed_like(df: Any, *, seed: int) -> Any:
    """
    Build a global seed by re-drawing attributes from the global donor pool
    within the same demographic key (age_idx/sex/race if available).
    This removes local spatial copula while keeping global demographic consistency.
    """
    np = _require("numpy")
    pd = _require("pandas")

    src = df.copy().reset_index(drop=True)
    out = src.copy()

    key_cols = [c for c in ["age_idx", "sex", "race"] if c in src.columns]
    if not key_cols:
        key_cols = [c for c in ["SEX"] if c in src.columns]

    rng = np.random.default_rng(int(seed))
    all_idx = np.arange(int(src.shape[0]), dtype=int)

    if key_cols:
        key = src[key_cols].astype(str).agg("|".join, axis=1)
        pools: dict[str, Any] = {}
        for k, gidx in pd.Series(np.arange(len(key))).groupby(key.values):
            pools[str(k)] = gidx.to_numpy(dtype=int)
        row_key = key.to_numpy(dtype=object)
        donor_idx = np.empty(len(row_key), dtype=int)
        for i, k in enumerate(row_key.tolist()):
            pool = pools.get(str(k))
            if pool is None or pool.size == 0:
                donor_idx[i] = int(rng.choice(all_idx))
            else:
                donor_idx[i] = int(rng.choice(pool))
    else:
        donor_idx = rng.choice(all_idx, size=int(src.shape[0]), replace=True).astype(int)

    for c in ["PINCP", "SCHL", "ESR"]:
        out[c] = src[c].iloc[donor_idx].to_numpy()
    return out


def _load_targets_long(path: pathlib.Path, *, tract_col: str) -> dict[str, dict[str, dict[str, float]]]:
    pd = _require("pandas")

    if path.suffix.lower() == ".parquet":
        tgt = pd.read_parquet(path)
    else:
        tgt = pd.read_csv(path, low_memory=False)

    req = [tract_col, "variable", "category", "target"]
    missing = [c for c in req if c not in tgt.columns]
    if missing:
        raise SystemExit(f"targets_long missing columns: {missing}")

    vars_keep = {"PINCP_16p_bin", "ESR_16p", "SCHL_25p"}
    t = tgt[tgt["variable"].astype(str).isin(vars_keep)].copy()
    t[tract_col] = t[tract_col].astype(str)
    t["variable"] = t["variable"].astype(str)
    t["category"] = t["category"].astype(str)
    t["target"] = pd.to_numeric(t["target"], errors="coerce").fillna(0.0).clip(lower=0.0)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for (g, var), sub in t.groupby([tract_col, "variable"], sort=False):
        cat_sum = sub.groupby("category", sort=False)["target"].sum()
        total = float(cat_sum.sum())
        if total <= 0:
            continue
        probs = {str(k): float(v / total) for k, v in cat_sum.to_dict().items()}
        out.setdefault(str(g), {})[str(var)] = probs
    return out


def _rake_weights_by_tract(
    *,
    df: Any,
    tract_col: str,
    targets: dict[str, dict[str, dict[str, float]]],
    base_weight_col: str,
    out_weight_col: str,
    iters: int,
    clip_factor: float,
) -> tuple[Any, dict[str, Any]]:
    """
    Multi-marginal raking on overlapping universes:
    - PINCP_16p_bin over AGEP>=16 & PINCP>0
    - ESR_16p over AGEP>=16
    - SCHL_25p over AGEP>=25
    """
    np = _require("numpy")
    pd = _require("pandas")

    d = df.copy()
    d[out_weight_col] = pd.to_numeric(d[base_weight_col], errors="coerce").fillna(1.0).clip(lower=0.0).to_numpy(dtype=float)

    def _eligible_mask(x: Any, var: str) -> Any:
        if var == "PINCP_16p_bin":
            return x["has_earnings_16p"].to_numpy(dtype=bool)
        if var == "ESR_16p":
            return x["is_16p"].to_numpy(dtype=bool)
        if var == "SCHL_25p":
            return x["is_25p"].to_numpy(dtype=bool)
        return np.zeros(int(x.shape[0]), dtype=bool)

    diagnostics = {"iters": int(iters), "tracts_seen": 0, "tracts_used": 0, "infeasible_cells": 0}
    tracts = sorted(set(d[tract_col].astype(str).unique().tolist()) & set(targets.keys()))
    diagnostics["tracts_seen"] = int(len(tracts))

    for _ in range(int(iters)):
        for tg in tracts:
            idx = d.index[d[tract_col].astype(str) == str(tg)].to_numpy(dtype=int)
            if idx.size == 0:
                continue
            sub = d.loc[idx]
            w = d.loc[idx, out_weight_col].to_numpy(dtype=float)
            if not np.isfinite(w).any() or float(np.nansum(w)) <= 0:
                continue
            tract_targets = targets.get(str(tg), {})
            if not tract_targets:
                continue

            for var in ["PINCP_16p_bin", "ESR_16p", "SCHL_25p"]:
                tprob = tract_targets.get(var)
                if not tprob:
                    continue
                mask = _eligible_mask(sub, var)
                if not bool(mask.any()):
                    continue
                sub_var = sub.loc[mask, var].astype(str).to_numpy(dtype=object)
                sub_w = w[mask]
                tot = float(sub_w.sum())
                if tot <= 0:
                    continue

                cur_counts: dict[str, float] = {}
                for c in set(sub_var.tolist()):
                    cur_counts[str(c)] = float(sub_w[sub_var == c].sum())

                factors = np.ones_like(sub_w, dtype=float)
                for cat, p in tprob.items():
                    target_count = float(p) * tot
                    cur = float(cur_counts.get(str(cat), 0.0))
                    if cur <= 0:
                        if target_count > 0:
                            diagnostics["infeasible_cells"] += 1
                        continue
                    ratio = target_count / max(cur, 1e-12)
                    if clip_factor > 1.0:
                        ratio = float(np.clip(ratio, 1.0 / clip_factor, clip_factor))
                    factors[sub_var == str(cat)] = factors[sub_var == str(cat)] * ratio

                w_new = w.copy()
                w_new[mask] = sub_w * factors
                # keep tract total weight stable
                s0 = float(w.sum())
                s1 = float(w_new.sum())
                if s1 > 0 and s0 > 0:
                    w_new = w_new * (s0 / s1)
                d.loc[idx, out_weight_col] = w_new

            diagnostics["tracts_used"] += 1

    return d, diagnostics


def _weighted_tvd_to_targets(
    *,
    df: Any,
    tract_col: str,
    wcol: str,
    targets: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    np = _require("numpy")
    pd = _require("pandas")

    def _eligible_mask(x: Any, var: str) -> Any:
        if var == "PINCP_16p_bin":
            return x["has_earnings_16p"].to_numpy(dtype=bool)
        if var == "ESR_16p":
            return x["is_16p"].to_numpy(dtype=bool)
        if var == "SCHL_25p":
            return x["is_25p"].to_numpy(dtype=bool)
        return np.zeros(int(x.shape[0]), dtype=bool)

    out: dict[str, Any] = {}
    for var in ["PINCP_16p_bin", "ESR_16p", "SCHL_25p"]:
        by_group: dict[str, float] = {}
        for tg, tvars in targets.items():
            tprob = tvars.get(var)
            if not tprob:
                continue
            sub = df[df[tract_col].astype(str) == str(tg)]
            if sub.empty:
                continue
            mask = _eligible_mask(sub, var)
            if not bool(mask.any()):
                continue
            s = sub.loc[mask, [var, wcol]].copy()
            s[wcol] = pd.to_numeric(s[wcol], errors="coerce").fillna(0.0).clip(lower=0.0)
            tot = float(s[wcol].sum())
            if tot <= 0:
                continue
            scount = s.groupby(var, dropna=False)[wcol].sum()
            cats = sorted(set([str(c) for c in scount.index.tolist()]) | set(tprob.keys()))
            p = np.array([float(scount.get(c, 0.0) / tot) for c in cats], dtype=float)
            q = np.array([float(tprob.get(c, 0.0)) for c in cats], dtype=float)
            by_group[str(tg)] = _tvd(p, q)
        vals = list(by_group.values())
        out[var] = {
            "mean": (None if not vals else float(np.mean(vals))),
            "max": (None if not vals else float(np.max(vals))),
            "n_tracts": int(len(vals)),
            "by_tract": by_group,
        }
    return out


def _puma_metrics_vs_pums(
    *,
    syn: Any,
    puma_col: str,
    wcol: str,
    ref: Any,
    ref_wcol: str,
) -> dict[str, Any]:
    pd = _require("pandas")
    np = _require("numpy")

    s = syn.copy()
    r = ref.copy()

    income_edges = [0.0, 10_000.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0, 150_000.0, 250_000.0, 10_000_000.0]
    s["PINCP_bin"] = pd.cut(s["PINCP"], bins=income_edges, include_lowest=True, right=False).astype(str)
    r["PINCP_bin"] = pd.cut(r["PINCP"], bins=income_edges, include_lowest=True, right=False).astype(str)

    per_puma: dict[str, Any] = {}
    for p in sorted(set(s[puma_col].astype(str).unique().tolist())):
        s_p = s[s[puma_col].astype(str) == str(p)]
        r_p = r[r[puma_col].astype(str) == str(p)]
        if s_p.empty or r_p.empty:
            continue
        if float(pd.to_numeric(s_p[wcol], errors="coerce").fillna(0.0).sum()) <= 0:
            continue
        if float(pd.to_numeric(r_p[ref_wcol], errors="coerce").fillna(0.0).sum()) <= 0:
            continue

        m = {
            "tvd_income_bin": _tvd_from_dists(_weighted_cat_dist(s_p, "PINCP_bin", wcol), _weighted_cat_dist(r_p, "PINCP_bin", ref_wcol)),
            "tvd_schl": _tvd_from_dists(_weighted_cat_dist(s_p, "SCHL", wcol), _weighted_cat_dist(r_p, "SCHL", ref_wcol)),
            "tvd_esr": _tvd_from_dists(_weighted_cat_dist(s_p, "ESR", wcol), _weighted_cat_dist(r_p, "ESR", ref_wcol)),
        }

        u_s = _weighted_rank(s_p["AGEP"].to_numpy(dtype=float), pd.to_numeric(s_p[wcol], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        v_s = _weighted_rank(s_p["PINCP"].to_numpy(dtype=float), pd.to_numeric(s_p[wcol], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        u_r = _weighted_rank(r_p["AGEP"].to_numpy(dtype=float), pd.to_numeric(r_p[ref_wcol], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        v_r = _weighted_rank(r_p["PINCP"].to_numpy(dtype=float), pd.to_numeric(r_p[ref_wcol], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        cop_s = _copula_hist2d(u=u_s, v=v_s, w=pd.to_numeric(s_p[wcol], errors="coerce").fillna(0.0).to_numpy(dtype=float), bins=10)
        cop_r = _copula_hist2d(u=u_r, v=v_r, w=pd.to_numeric(r_p[ref_wcol], errors="coerce").fillna(0.0).to_numpy(dtype=float), bins=10)
        m["copula_tvd_age_income"] = _tvd(cop_s, cop_r)

        s_joint = s_p.assign(age_idx=s_p["AGEP"].astype(int).map(_age_to_p12_idx).astype(str))
        r_joint = r_p.assign(age_idx=r_p["AGEP"].astype(int).map(_age_to_p12_idx).astype(str))
        m["joint_tvd_age_income_bin"] = _tvd_from_dists(
            _weighted_joint_dist(s_joint, ["age_idx", "PINCP_bin"], wcol),
            _weighted_joint_dist(r_joint, ["age_idx", "PINCP_bin"], ref_wcol),
        )
        per_puma[str(p)] = m

    def _agg(metric: str) -> dict[str, float] | None:
        vals = [per_puma[p][metric] for p in per_puma if per_puma[p].get(metric) is not None]
        if not vals:
            return None
        arr = np.asarray(vals, dtype=float)
        return {"mean": float(arr.mean()), "max": float(arr.max()), "n_pumas": int(arr.size)}

    return {
        "by_puma": per_puma,
        "summary": {
            "tvd_income_bin": _agg("tvd_income_bin"),
            "tvd_schl": _agg("tvd_schl"),
            "tvd_esr": _agg("tvd_esr"),
            "copula_tvd_age_income": _agg("copula_tvd_age_income"),
            "joint_tvd_age_income_bin": _agg("joint_tvd_age_income_bin"),
        },
    }


def _load_pums_reference(*, data_root: pathlib.Path, pums_year: int, pums_period: str, statefp: str, pums_person_zip: str | None, puma_col_out: str) -> Any:
    pd = _require("pandas")

    if pums_person_zip:
        pzip = pathlib.Path(pums_person_zip).expanduser().resolve()
    else:
        pzip = _resolve_pums_person_zip(data_root=data_root, pums_year=pums_year, pums_period=pums_period, statefp=statefp)
    if not pzip.exists():
        raise SystemExit(f"pums_person_zip not found: {pzip}")

    member = _find_first_csv_in_zip(pzip)
    usecols = ["PUMA", "PUMA20", "PWGTP", "AGEP", "PINCP", "SCHL", "ESR"]
    with zipfile.ZipFile(pzip) as zf, zf.open(member) as f:
        ref = pd.read_csv(f, usecols=lambda c: c in set(usecols), low_memory=False)

    if "PUMA20" in ref.columns:
        ref["PUMA"] = ref["PUMA20"]
    if "PUMA" not in ref.columns:
        raise SystemExit("PUMS reference missing PUMA/PUMA20")
    puma_num = pd.to_numeric(ref["PUMA"], errors="coerce")
    ref = ref[puma_num.notna() & (puma_num != -9)].copy()
    ref[puma_col_out] = ref["PUMA"].astype(str)
    ref["PWGTP"] = pd.to_numeric(ref["PWGTP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    ref["AGEP"] = pd.to_numeric(ref["AGEP"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=99.0)
    ref["PINCP"] = pd.to_numeric(ref["PINCP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    ref["SCHL"] = pd.to_numeric(ref["SCHL"], errors="coerce").fillna(0).astype(int).astype(str)
    ref["ESR"] = pd.to_numeric(ref["ESR"], errors="coerce").fillna(0).astype(int).astype(str)
    ref = ref[ref["PWGTP"] > 0].copy()
    return ref


def _summary_delta(*, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in ["tvd_income_bin", "tvd_schl", "tvd_esr", "copula_tvd_age_income", "joint_tvd_age_income_bin"]:
        b = (before.get("summary", {}).get(metric) or {}).get("mean")
        a = (after.get("summary", {}).get(metric) or {}).get("mean")
        if b is None or a is None:
            out[metric] = {"before_mean": b, "after_mean": a, "delta": None, "delta_pct": None}
            continue
        delta = float(a - b)
        pct = (None if abs(float(b)) < 1e-12 else float(delta / float(b)))
        out[metric] = {"before_mean": float(b), "after_mean": float(a), "delta": delta, "delta_pct": pct}
    return out


def main() -> None:
    pd = _require("pandas")
    np = _require("numpy")

    from src.synthpop.pipeline.detroit_v0 import make_run_id
    from src.synthpop.paths import data_root as default_data_root

    ap = argparse.ArgumentParser(prog="exp5_tract_postalign")
    ap.add_argument("--synthetic_path", required=True, help="Exp4 synthetic sample file (csv/csv.gz/parquet).")
    ap.add_argument("--acs_targets_long", required=True, help="Tract-level ACS targets_long (csv/parquet).")
    ap.add_argument("--data_root", default=str(default_data_root()))
    ap.add_argument("--pums_year", type=int, default=2023)
    ap.add_argument("--pums_period", default="5-Year")
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--pums_person_zip", default=None, help="Optional override for PUMS person zip.")
    ap.add_argument("--tract_col", default="tract_geoid")
    ap.add_argument("--puma_col", default="puma")
    ap.add_argument("--ipf_iters", type=int, default=25)
    ap.add_argument("--clip_factor", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_post_samples", action="store_true", help="Write post-aligned microdata csv.gz for both seeds.")
    ap.add_argument("--out_dir", default=None, help="Default: outputs/<run_id> under repo.")
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    out_dir = (
        pathlib.Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (_REPO_ROOT / "outputs" / make_run_id(prefix="exp5_tract_postalign"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        out_dir / "run.metadata.json",
        {
            "created_utc": _utc_now_iso(),
            "argv": sys.argv,
            "script": pathlib.Path(__file__).name,
            "env": {"RAW_ROOT": os.environ.get("RAW_ROOT"), "SYNTHCITY_DATA_ROOT": os.environ.get("SYNTHCITY_DATA_ROOT")},
            "args": vars(args),
        },
    )

    syn_path = pathlib.Path(args.synthetic_path).expanduser().resolve()
    if not syn_path.exists():
        raise SystemExit(f"synthetic_path not found: {syn_path}")
    if syn_path.suffix.lower() == ".parquet":
        syn_raw = pd.read_parquet(syn_path)
    else:
        syn_raw = pd.read_csv(syn_path, low_memory=False)
    syn = _harmonize_synthetic_columns(syn_raw, tract_col=str(args.tract_col), puma_col=str(args.puma_col))
    syn = _derive_scope_columns(syn)

    # Keep only rows with non-empty tract/puma.
    syn = syn[(syn[str(args.tract_col)].astype(str) != "") & (syn[str(args.puma_col)].astype(str) != "")].copy()
    if syn.empty:
        raise SystemExit("synthetic becomes empty after tract/puma filtering.")

    targets = _load_targets_long(pathlib.Path(args.acs_targets_long).expanduser().resolve(), tract_col=str(args.tract_col))
    if not targets:
        raise SystemExit("No valid targets loaded from acs_targets_long.")

    # Restrict to tract intersection for fair comparison.
    valid_tracts = set(targets.keys())
    syn = syn[syn[str(args.tract_col)].astype(str).isin(valid_tracts)].copy()
    if syn.empty:
        raise SystemExit("No overlapping tracts between synthetic and targets_long.")

    # Build seeds.
    diffusion_seed = syn.copy().reset_index(drop=True)
    global_seed = _build_global_seed_like(syn, seed=int(args.seed)).reset_index(drop=True)
    global_seed = _derive_scope_columns(global_seed)

    for d in [diffusion_seed, global_seed]:
        d["W_pre"] = pd.to_numeric(d["W"], errors="coerce").fillna(1.0).clip(lower=0.0)

    # Pre-alignment tract fit.
    pre_tract_diff = _weighted_tvd_to_targets(
        df=diffusion_seed, tract_col=str(args.tract_col), wcol="W_pre", targets=targets
    )
    pre_tract_glob = _weighted_tvd_to_targets(
        df=global_seed, tract_col=str(args.tract_col), wcol="W_pre", targets=targets
    )

    # Post-alignment (raking/IPF-style).
    diffusion_post, diag_diff = _rake_weights_by_tract(
        df=diffusion_seed,
        tract_col=str(args.tract_col),
        targets=targets,
        base_weight_col="W_pre",
        out_weight_col="W_post",
        iters=int(args.ipf_iters),
        clip_factor=float(args.clip_factor),
    )
    global_post, diag_glob = _rake_weights_by_tract(
        df=global_seed,
        tract_col=str(args.tract_col),
        targets=targets,
        base_weight_col="W_pre",
        out_weight_col="W_post",
        iters=int(args.ipf_iters),
        clip_factor=float(args.clip_factor),
    )

    post_tract_diff = _weighted_tvd_to_targets(
        df=diffusion_post, tract_col=str(args.tract_col), wcol="W_post", targets=targets
    )
    post_tract_glob = _weighted_tvd_to_targets(
        df=global_post, tract_col=str(args.tract_col), wcol="W_post", targets=targets
    )

    # External PUMA validation (vs PUMS).
    ref = _load_pums_reference(
        data_root=pathlib.Path(args.data_root).expanduser().resolve(),
        pums_year=int(args.pums_year),
        pums_period=str(args.pums_period),
        statefp=str(args.statefp),
        pums_person_zip=args.pums_person_zip,
        puma_col_out=str(args.puma_col),
    )

    metrics = {
        "diffusion_seed": {
            "pre": _puma_metrics_vs_pums(syn=diffusion_seed, puma_col=str(args.puma_col), wcol="W_pre", ref=ref, ref_wcol="PWGTP"),
            "post": _puma_metrics_vs_pums(syn=diffusion_post, puma_col=str(args.puma_col), wcol="W_post", ref=ref, ref_wcol="PWGTP"),
        },
        "global_seed": {
            "pre": _puma_metrics_vs_pums(syn=global_seed, puma_col=str(args.puma_col), wcol="W_pre", ref=ref, ref_wcol="PWGTP"),
            "post": _puma_metrics_vs_pums(syn=global_post, puma_col=str(args.puma_col), wcol="W_post", ref=ref, ref_wcol="PWGTP"),
        },
    }

    summary = {
        "created_utc": _utc_now_iso(),
        "n_rows": int(syn.shape[0]),
        "n_tracts_overlap": int(syn[str(args.tract_col)].nunique()),
        "n_pumas_overlap": int(syn[str(args.puma_col)].nunique()),
        "targets_variables": ["PINCP_16p_bin", "ESR_16p", "SCHL_25p"],
        "ipf": {
            "iters": int(args.ipf_iters),
            "clip_factor": float(args.clip_factor),
            "diagnostics": {"diffusion_seed": diag_diff, "global_seed": diag_glob},
        },
        "tract_fit_tvd": {
            "diffusion_seed": {"pre": pre_tract_diff, "post": post_tract_diff},
            "global_seed": {"pre": pre_tract_glob, "post": post_tract_glob},
        },
        "puma_eval_delta": {
            "diffusion_seed_pre_to_post": _summary_delta(before=metrics["diffusion_seed"]["pre"], after=metrics["diffusion_seed"]["post"]),
            "global_seed_pre_to_post": _summary_delta(before=metrics["global_seed"]["pre"], after=metrics["global_seed"]["post"]),
            "post_diffusion_minus_global": _summary_delta(before=metrics["global_seed"]["post"], after=metrics["diffusion_seed"]["post"]),
        },
    }

    _write_json(out_dir / "exp5_metrics_by_seed.json", metrics)
    _write_json(out_dir / "exp5_summary.json", summary)
    _write_json(
        out_dir / "tract_fit_by_seed.json",
        {
            "diffusion_seed": {"pre": pre_tract_diff, "post": post_tract_diff},
            "global_seed": {"pre": pre_tract_glob, "post": post_tract_glob},
        },
    )

    if bool(args.save_post_samples):
        diffusion_post.to_csv(out_dir / "diffusion_seed_post.csv.gz", index=False, compression="gzip")
        global_post.to_csv(out_dir / "global_seed_post.csv.gz", index=False, compression="gzip")

    print(f"[ok] wrote: {out_dir}")


if __name__ == "__main__":
    main()

