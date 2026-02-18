#!/usr/bin/env python3
"""
Exp 2: Layer-2 attribute extension via conditional tabular diffusion (TabDDPM v0).

Problem this answers:
- Given base demographics (age_group, sex, race) and optional area features, generate additional
  attributes (income, education, employment) while preserving joint structure beyond marginals.

Training data (independent from validation targets):
- Michigan PUMS person microdata (AGEP, SEX, RAC1P, PINCP, SCHL, ESR, PUMA, PWGTP).

Validation (does not participate in training):
- PUMS holdout by PUMA folds (joint/marginal/correlation + copula TVD).
- (Optional, future) ACS tract-level summaries for education/employment/earnings.

Design choices (KISS, aligned with repo constraints):
- Use the existing Gaussian DDPM (DiffusionTabularModel) on a continuous vector:
    x = [income_z] + onehot(SCHL) + onehot(ESR)
  Conditioning is concatenated (demo-only) to keep implementation minimal.
- Race conditioning is supported if columns exist, but is optional.
- Optional PUMA-level context features (computed from PUMS itself) can be concatenated to condition.
- Checkpoints are written but ignored by git via .gitignore.

Outputs:
  outputs/<run_id>/
    <condition_id>/fold_<k>/{model.pt,encoder.json,train_summary.json}
    metrics/pums_holdout_marginal_tvd.json
    metrics/pums_holdout_joint_tvd.json
    metrics/pums_holdout_copula_tvd.json
    ablation_summary.json
    run.metadata.json
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
from dataclasses import asdict, dataclass
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
    # 23 bins (same semantics as DHC P12 / ACS B01001 23 age groups).
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


def _one_hot_codes(codes: "Any", depth: int) -> "Any":
    np = _require("numpy")
    codes = np.asarray(codes, dtype=int)
    if codes.ndim != 1:
        raise ValueError("codes must be 1D")
    if (codes < 0).any() or (codes >= depth).any():
        raise ValueError("codes out of range for one-hot")
    return np.eye(depth, dtype=np.float32)[codes]


def _softmax(x: "Any", axis: int = 1) -> "Any":
    np = _require("numpy")
    x = np.asarray(x, dtype=np.float32)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def _tvd(p: "Any", q: "Any") -> float:
    np = _require("numpy")
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p - q).sum())


def _weighted_rank(u: "Any", w: "Any") -> "Any":
    np = _require("numpy")
    u = np.asarray(u, dtype=float)
    w = np.asarray(w, dtype=float)
    if u.shape != w.shape:
        raise ValueError(f"u and w must have the same shape, got {u.shape} vs {w.shape}")
    mask = np.isfinite(u) & np.isfinite(w) & (w > 0)
    # Keep output aligned to input length; invalid rows are NaN and will be filtered later
    # when we build copulas/joints.
    out = np.full(u.shape, np.nan, dtype=float)
    u_m = u[mask]
    w_m = w[mask]
    if u_m.size == 0:
        return out
    order = np.argsort(u_m, kind="mergesort")
    w_sorted = w_m[order]
    cw = np.cumsum(w_sorted)
    tot = float(cw[-1])
    r_sorted = (cw - 0.5 * w_sorted) / max(tot, 1e-12)
    r = np.empty_like(r_sorted)
    r[order] = r_sorted
    out[mask] = np.clip(r, 0.0, 1.0)
    return out


def _copula_hist2d(*, u: "Any", v: "Any", w: "Any", bins: int = 10) -> "Any":
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


def _stable_hash_fold(values: list[str], *, n_folds: int, seed: int) -> dict[str, int]:
    """
    Deterministic fold assignment without external deps.
    Not geographic, but prevents row-level leakage across folds.
    """
    import hashlib

    out: dict[str, int] = {}
    for v in values:
        h = hashlib.sha1((str(seed) + "::" + str(v)).encode("utf-8")).hexdigest()
        out[str(v)] = int(h[:8], 16) % int(n_folds)
    return out


@dataclass(frozen=True)
class _Encoder:
    # condition
    cond_cols: list[str]
    age_depth: int
    sex_depth: int
    race_cats: list[str] | None
    # target
    income_mean: float
    income_std: float
    schl_cats: list[str]
    esr_cats: list[str]


def _build_puma_stats(*, df: Any, puma_col: str, wcol: str) -> tuple[dict[str, "Any"], list[str]]:
    """
    Build lightweight PUMA context features from PUMS itself.
    These are intended to be available at inference time via external aggregates (ACS etc.).
    """
    pd = _require("pandas")
    np = _require("numpy")

    d = df[[puma_col, wcol, "AGEP", "PINCP"]].copy()
    d[puma_col] = d[puma_col].astype(str)
    d[wcol] = pd.to_numeric(d[wcol], errors="coerce").fillna(0.0).clip(lower=0.0)
    d["AGEP"] = pd.to_numeric(d["AGEP"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=99.0)
    d["PINCP"] = pd.to_numeric(d["PINCP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    d["PINCP_log"] = np.log1p(d["PINCP"].to_numpy(dtype=np.float32))

    def _wmean(x: "Any", w: "Any") -> float:
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        s = float(w.sum())
        if s <= 0:
            return float("nan")
        return float((x * w).sum() / s)

    rows = []
    for puma, g in d.groupby(puma_col, sort=False):
        w = g[wcol].to_numpy(dtype=float)
        age = g["AGEP"].to_numpy(dtype=float)
        inc = g["PINCP_log"].to_numpy(dtype=float)
        pop = float(w.sum())
        rows.append(
            {
                "puma": str(puma),
                "pop_log": float(np.log(max(pop, 1.0))),
                "mean_age": _wmean(age, w),
                "mean_income_log": _wmean(inc, w),
                "pct_child": float(w[age < 18].sum() / max(pop, 1e-9)),
                "pct_elderly": float(w[age >= 65].sum() / max(pop, 1e-9)),
            }
        )
    feat = pd.DataFrame(rows)
    cols = ["pop_log", "mean_age", "mean_income_log", "pct_child", "pct_elderly"]
    # z-score across PUMAs
    mu = feat[cols].mean(axis=0, numeric_only=True)
    sd = feat[cols].std(axis=0, ddof=0, numeric_only=True).replace(0.0, 1.0)
    feat_z = (feat[cols] - mu) / sd
    out: dict[str, Any] = {}
    for i, r in feat.iterrows():
        out[str(r["puma"])] = feat_z.iloc[i].to_numpy(dtype=np.float32)
    return out, [f"puma_{c}_z" for c in cols]


def _resolve_buildings_csv(*, data_root: pathlib.Path, buildings_csv: str | None) -> pathlib.Path | None:
    if buildings_csv:
        p = pathlib.Path(buildings_csv).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"buildings_csv not found: {p}")
        return p
    cand = data_root / "detroit" / "processed" / "buildings" / "buildings_detroit_features_price.csv"
    if cand.exists():
        return cand
    return None


def _build_puma_built_stats(*, buildings_df: Any, puma_col: str = "puma") -> tuple[dict[str, "Any"], list[str]]:
    """
    Build PUMA-level built-environment context from building table.

    Required: a PUMA column and at least one numeric feature among known candidates.
    """
    pd = _require("pandas")
    np = _require("numpy")

    d = buildings_df.copy()
    if puma_col not in d.columns:
        # Common fallback names.
        for c in ["PUMA", "puma20", "PUMA20"]:
            if c in d.columns:
                d[puma_col] = d[c]
                break
    if puma_col not in d.columns:
        raise SystemExit("buildings_csv missing puma/PUMA/PUMA20 column for built context.")

    d[puma_col] = pd.to_numeric(d[puma_col], errors="coerce")
    d = d[d[puma_col].notna()].copy()
    d[puma_col] = d[puma_col].astype(int).astype(str)
    if d.empty:
        raise SystemExit("buildings_csv has no valid puma rows.")

    # Candidate feature columns (take those that exist).
    num_cols = []
    for c in ["cap_proxy", "height_m", "height", "footprint_area_m2", "footprint_area", "dist_cbd_km"]:
        if c in d.columns:
            num_cols.append(c)
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0).astype(float)

    # Price tier entropy/proportions if available.
    has_price_tier = "price_tier" in d.columns
    if has_price_tier:
        d["price_tier"] = pd.to_numeric(d["price_tier"], errors="coerce").fillna(0).astype(int)

    rows = []
    for puma, g in d.groupby(puma_col, sort=False):
        row: dict[str, Any] = {"puma": str(puma)}
        row["n_buildings_log"] = float(np.log(max(int(g.shape[0]), 1)))
        row["cap_proxy_sum_log"] = float(np.log1p(float(g["cap_proxy"].sum()))) if "cap_proxy" in g.columns else 0.0
        if "height_m" in g.columns:
            row["height_mean"] = float(g["height_m"].mean())
        elif "height" in g.columns:
            row["height_mean"] = float(g["height"].mean())
        else:
            row["height_mean"] = 0.0
        if "footprint_area_m2" in g.columns:
            row["footprint_mean_log"] = float(np.log1p(float(g["footprint_area_m2"].mean())))
        elif "footprint_area" in g.columns:
            row["footprint_mean_log"] = float(np.log1p(float(g["footprint_area"].mean())))
        else:
            row["footprint_mean_log"] = 0.0
        row["dist_cbd_mean"] = float(g["dist_cbd_km"].mean()) if "dist_cbd_km" in g.columns else 0.0
        if has_price_tier:
            valid = g["price_tier"][g["price_tier"] > 0]
            if valid.empty:
                probs = [0.0] * 5
            else:
                counts = np.bincount(valid.to_numpy(dtype=int), minlength=6)[1:6].astype(float)
                probs = (counts / max(float(counts.sum()), 1e-12)).tolist()
            for i, p in enumerate(probs, start=1):
                row[f"price_tier_p{i}"] = float(p)
        rows.append(row)

    feat = pd.DataFrame(rows)
    cols = [c for c in feat.columns if c != "puma"]
    if not cols:
        raise SystemExit("No usable built features found in buildings_csv.")

    mu = feat[cols].mean(axis=0, numeric_only=True)
    sd = feat[cols].std(axis=0, ddof=0, numeric_only=True).replace(0.0, 1.0)
    feat_z = (feat[cols] - mu) / sd

    out: dict[str, Any] = {}
    for i, r in feat.iterrows():
        out[str(r["puma"])] = feat_z.iloc[i].to_numpy(dtype=np.float32)
    return out, [f"built_{c}_z" for c in cols]


def _encode_condition(
    *,
    df: Any,
    puma_col: str,
    use_race: bool,
    use_area_stats: bool,
    area_stats: dict[str, "Any"] | None,
    area_stat_cols: list[str] | None,
    race_cats_global: list[int] | None,
) -> tuple[Any, Any, _Encoder]:
    pd = _require("pandas")
    np = _require("numpy")

    age_idx = df["AGEP"].astype(int).map(_age_to_p12_idx).to_numpy(dtype=int)
    age_oh = _one_hot_codes(age_idx, 23)

    sex = pd.to_numeric(df["SEX"], errors="coerce").fillna(1).astype(int).clip(lower=1, upper=2).to_numpy(dtype=int)
    sex_idx = sex - 1
    sex_oh = _one_hot_codes(sex_idx, 2)

    cond_parts = [age_oh, sex_oh]
    cond_cols = [f"age_{i}" for i in range(23)] + ["sex_m", "sex_f"]

    race_cats: list[str] | None = None
    if use_race:
        race = pd.to_numeric(df.get("RAC1P"), errors="coerce").fillna(-1).astype(int)
        if not race_cats_global:
            raise RuntimeError("use_race requested but race_cats_global is None/empty")
        race_cat = pd.Categorical(race, categories=race_cats_global, ordered=False)
        # Keep only known races (codes >=0)
        mask = (race_cat.codes >= 0)
        mask_np = np.asarray(mask, dtype=bool)
        if not bool(mask_np.all()):
            df = df.loc[mask_np].copy()
            age_oh = age_oh[mask_np]
            sex_oh = sex_oh[mask_np]
            cond_parts = [age_oh, sex_oh]
        race_cats = [str(x) for x in race_cat.categories.tolist()]
        race_oh = np.eye(len(race_cats), dtype=np.float32)[race_cat.codes[mask_np]]
        cond_parts.append(race_oh)
        cond_cols += [f"race_{c}" for c in race_cats]

    if use_area_stats:
        if area_stats is None or area_stat_cols is None:
            raise RuntimeError("use_area_stats requested but area_stats/area_stat_cols is None")
        p = df[puma_col].astype(str).to_numpy()
        dim = int(len(area_stat_cols))
        zero = np.zeros(dim, dtype=np.float32)
        stats = np.stack([np.asarray(area_stats.get(str(x), zero), dtype=np.float32) for x in p], axis=0)
        if stats.ndim != 2:
            raise RuntimeError("Invalid area stats shape")
        cond_parts.append(stats.astype(np.float32))
        cond_cols += list(area_stat_cols)

    cond = np.concatenate(cond_parts, axis=1).astype(np.float32)
    enc = _Encoder(
        cond_cols=cond_cols,
        age_depth=23,
        sex_depth=2,
        race_cats=race_cats,
        income_mean=0.0,
        income_std=1.0,
        schl_cats=[],
        esr_cats=[],
    )
    return df, cond, enc


def _encode_targets(*, df: Any) -> tuple[Any, _Encoder, Any]:
    """
    Encode targets into a continuous vector:
      x = [income_z] + onehot(SCHL) + onehot(ESR)
    Returns:
      x_np, encoder, decode_meta
    """
    pd = _require("pandas")
    np = _require("numpy")

    if getattr(df, "empty", False):
        raise RuntimeError("encode_targets received an empty DataFrame (check fold split / filtering).")

    income = pd.to_numeric(df["PINCP"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32)
    income_log = np.log1p(income)
    mu = float(income_log.mean())
    sd = float(income_log.std(ddof=0))
    if not math.isfinite(sd) or sd <= 1e-6:
        sd = 1.0
    income_z = ((income_log - mu) / sd).reshape(-1, 1).astype(np.float32)

    schl = pd.to_numeric(df.get("SCHL"), errors="coerce").fillna(0).astype(int)
    schl_cat = pd.Categorical(schl)
    schl_cats = [str(x) for x in schl_cat.categories.tolist()]
    schl_oh = np.eye(len(schl_cats), dtype=np.float32)[schl_cat.codes]

    esr = pd.to_numeric(df.get("ESR"), errors="coerce").fillna(0).astype(int)
    esr_cat = pd.Categorical(esr)
    esr_cats = [str(x) for x in esr_cat.categories.tolist()]
    esr_oh = np.eye(len(esr_cats), dtype=np.float32)[esr_cat.codes]

    x = np.concatenate([income_z, schl_oh, esr_oh], axis=1).astype(np.float32)
    enc = _Encoder(
        cond_cols=[],
        age_depth=23,
        sex_depth=2,
        race_cats=None,
        income_mean=mu,
        income_std=sd,
        schl_cats=schl_cats,
        esr_cats=esr_cats,
    )
    decode_meta = {"income_z_col": 0, "schl_slice": [1, 1 + len(schl_cats)], "esr_slice": [1 + len(schl_cats), x.shape[1]]}
    return x, enc, decode_meta


def _decode_samples(*, x_hat: Any, enc_target: _Encoder, decode_meta: dict[str, Any]) -> Any:
    pd = _require("pandas")
    np = _require("numpy")

    x_hat = np.asarray(x_hat, dtype=np.float32)
    inc_z = x_hat[:, 0]
    inc_log = inc_z * float(enc_target.income_std) + float(enc_target.income_mean)
    # Prevent overflow during expm1 for pathological samples.
    inc_log = np.clip(inc_log, -10.0, 20.0)
    income = np.expm1(inc_log).clip(min=0.0)

    s0, s1 = decode_meta["schl_slice"]
    e0, e1 = decode_meta["esr_slice"]
    schl_logits = x_hat[:, s0:s1]
    esr_logits = x_hat[:, e0:e1]
    schl_idx = schl_logits.argmax(axis=1) if schl_logits.shape[1] > 0 else np.zeros(x_hat.shape[0], dtype=int)
    esr_idx = esr_logits.argmax(axis=1) if esr_logits.shape[1] > 0 else np.zeros(x_hat.shape[0], dtype=int)

    # Be robust to pathological folds where a category list is empty.
    if enc_target.schl_cats:
        schl = [enc_target.schl_cats[int(i)] for i in schl_idx]
    else:
        schl = ["0"] * int(x_hat.shape[0])
    if enc_target.esr_cats:
        esr = [enc_target.esr_cats[int(i)] for i in esr_idx]
    else:
        esr = ["0"] * int(x_hat.shape[0])

    out = pd.DataFrame({"PINCP": income.astype(float), "SCHL": schl, "ESR": esr})
    # Confidence diagnostics (optional)
    if schl_logits.shape[1] > 0:
        out["SCHL_conf"] = _softmax(schl_logits).max(axis=1).astype(float)
    if esr_logits.shape[1] > 0:
        out["ESR_conf"] = _softmax(esr_logits).max(axis=1).astype(float)
    return out


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
        # k is a tuple
        out["|".join(str(x) for x in k)] = float(v / tot)
    return out


def _tvd_from_dists(p: dict[str, float], q: dict[str, float]) -> float | None:
    if not p or not q:
        return None
    keys = sorted(set(p.keys()) | set(q.keys()))
    pv = [float(p.get(k, 0.0)) for k in keys]
    qv = [float(q.get(k, 0.0)) for k in keys]
    return _tvd(pv, qv)


def _cosine_from_dists(p: dict[str, float], q: dict[str, float]) -> float | None:
    np = _require("numpy")
    if not p or not q:
        return None
    keys = sorted(set(p.keys()) | set(q.keys()))
    pv = np.asarray([float(p.get(k, 0.0)) for k in keys], dtype=float)
    qv = np.asarray([float(q.get(k, 0.0)) for k in keys], dtype=float)
    pn = float(np.linalg.norm(pv))
    qn = float(np.linalg.norm(qv))
    if pn <= 0 or qn <= 0:
        return None
    c = float(np.dot(pv, qv) / (pn * qn))
    return c if np.isfinite(c) else None


def main() -> None:
    pd = _require("pandas")
    np = _require("numpy")
    torch = _require("torch")

    from src.synthpop.model.diffusion_tabular import DiffusionTabularModel, TabDDPMConfig
    from src.synthpop.pipeline.detroit_v0 import make_run_id
    from src.synthpop.paths import data_root as default_data_root

    ap = argparse.ArgumentParser(prog="exp2_attribute_extension")
    ap.add_argument("--data_root", default=str(default_data_root()))
    ap.add_argument("--pums_year", type=int, default=2022)
    ap.add_argument("--pums_period", default="5-Year")
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--pums_person_zip", default=None, help="Optional override for PUMS person zip (recommended for strict reproducibility).")
    ap.add_argument("--buildings_csv", default=None, help="Optional building features CSV for built-context conditions.")
    ap.add_argument("--mode", choices=["train_eval", "eval_only"], default="train_eval")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If set, reuse existing fold checkpoints under out_dir and only run missing folds.",
    )
    ap.add_argument("--n_rows", type=int, default=None, help="Optional cap for faster iteration.")
    ap.add_argument(
        "--conditions",
        default="demo_only",
        help=(
            "Comma-separated condition sets: "
            "demo_only, demo_race, demo_puma, demo_race_puma, demo_race_puma_built"
        ),
    )
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--fold_split", choices=["hash"], default="hash")
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--hidden_dims", default="512,512")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--out_dir", default=None, help="Default: outputs/<run_id> under repo.")
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    out_dir = (
        pathlib.Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (_REPO_ROOT / "outputs" / make_run_id(prefix="exp2_attribute_extension"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Avoid overwriting the original training metadata when doing eval-only recovery.
    meta_path = out_dir / "run.metadata.json"
    if args.mode == "eval_only" and meta_path.exists():
        meta_path = out_dir / "eval.metadata.json"

    _write_json(
        meta_path,
        {
            "created_utc": _utc_now_iso(),
            "argv": sys.argv,
            "script": pathlib.Path(__file__).name,
            "env": {"RAW_ROOT": os.environ.get("RAW_ROOT"), "SYNTHCITY_DATA_ROOT": os.environ.get("SYNTHCITY_DATA_ROOT")},
            "args": vars(args),
        },
    )

    person_zip = (
        pathlib.Path(str(args.pums_person_zip)).expanduser().resolve()
        if args.pums_person_zip
        else _resolve_pums_person_zip(
            data_root=data_root, pums_year=int(args.pums_year), pums_period=str(args.pums_period), statefp=str(args.statefp)
        )
    )
    if not person_zip.exists():
        raise SystemExit(f"pums_person_zip not found: {person_zip}")
    member = _find_first_csv_in_zip(person_zip)

    usecols = ["PUMA", "PUMA20", "PWGTP", "AGEP", "SEX", "PINCP", "SCHL", "ESR", "RAC1P"]
    with zipfile.ZipFile(person_zip) as zf, zf.open(member) as f:
        # Read the full file first (no head bias), then sample later if --n_rows is set.
        df = pd.read_csv(f, usecols=lambda c: c in set(usecols), low_memory=False)

    # Keep geography boundary-consistent with the selected PUMS release.
    if int(args.pums_year) >= 2022:
        if "PUMA20" not in df.columns:
            raise SystemExit(
                f"PUMS {int(args.pums_year)} requires PUMA20, but column is missing "
                f"(zip={person_zip} member={member})."
            )
        puma_col = "PUMA20"
        df["PUMA"] = df["PUMA20"]
    elif "PUMA" in df.columns:
        puma_col = "PUMA"
    else:
        raise SystemExit(f"PUMS missing legacy PUMA column (zip={person_zip} member={member})")

    missing = [c for c in ["PWGTP", "AGEP", "SEX", "PINCP", "SCHL", "ESR"] if c not in df.columns]
    if missing:
        raise SystemExit(f"PUMS missing required cols: {missing} (zip={person_zip} member={member})")

    # Filter invalid PUMA codes before sampling (PUMS uses -9 for NIU/invalid in some extracts).
    puma_num = pd.to_numeric(df["PUMA"], errors="coerce")
    df = df[puma_num.notna() & (puma_num != -9)].copy()

    # Clean.
    df["PUMA"] = df["PUMA"].astype(str)
    df["PWGTP"] = pd.to_numeric(df["PWGTP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["AGEP"] = pd.to_numeric(df["AGEP"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=99.0)
    df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce").fillna(1).astype(int).clip(lower=1, upper=2)
    df["PINCP"] = pd.to_numeric(df["PINCP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["SCHL"] = pd.to_numeric(df["SCHL"], errors="coerce").fillna(0).astype(int)
    df["ESR"] = pd.to_numeric(df["ESR"], errors="coerce").fillna(0).astype(int)
    if "RAC1P" in df.columns:
        df["RAC1P"] = pd.to_numeric(df["RAC1P"], errors="coerce").fillna(-1).astype(int)

    df = df[df["PWGTP"] > 0].copy()
    if df.empty:
        raise SystemExit("No valid PUMS rows after cleaning.")

    # Randomly subsample after cleaning + invalid-PUMA filtering.
    if args.n_rows is not None and int(args.n_rows) > 0 and int(df.shape[0]) > int(args.n_rows):
        df = df.sample(n=int(args.n_rows), random_state=int(args.seed)).reset_index(drop=True)

    pumas = sorted(df["PUMA"].unique().tolist())
    fold_of = _stable_hash_fold(pumas, n_folds=int(args.n_folds), seed=int(args.seed))

    conditions = [c.strip() for c in str(args.conditions).split(",") if c.strip()]
    hidden_dims = tuple(int(x) for x in str(args.hidden_dims).split(",") if x.strip())

    # Global race categories for stable one-hot across folds.
    race_cats_global: list[int] | None = None
    if "RAC1P" in df.columns:
        rc = sorted({int(x) for x in df["RAC1P"].to_numpy(dtype=int).tolist() if int(x) >= 0})
        race_cats_global = rc if rc else None

    # PUMA stats for "puma" conditions.
    puma_stats, puma_stat_cols = _build_puma_stats(df=df, puma_col="PUMA", wcol="PWGTP")
    built_stats = None
    built_stat_cols = None
    if any("built" in c for c in conditions):
        bcsv = _resolve_buildings_csv(data_root=data_root, buildings_csv=args.buildings_csv)
        if bcsv is None:
            raise SystemExit(
                "built condition requested but buildings CSV not found. "
                "Pass --buildings_csv or place file at detroit/processed/buildings/buildings_detroit_features_price.csv"
            )
        bdf = pd.read_csv(bcsv, low_memory=False)
        built_stats, built_stat_cols = _build_puma_built_stats(buildings_df=bdf, puma_col="puma")

    metrics_by_condition: dict[str, Any] = {}
    for cond_id in conditions:
        use_race = "race" in cond_id
        use_puma_stats = "puma" in cond_id
        use_built_stats = "built" in cond_id
        use_area_stats = use_puma_stats or use_built_stats
        area_stats = None
        area_stat_cols = None
        if use_area_stats:
            base_map = puma_stats if use_puma_stats else {}
            base_cols = puma_stat_cols if use_puma_stats else []
            extra_map = built_stats if use_built_stats and built_stats is not None else {}
            extra_cols = built_stat_cols if use_built_stats and built_stat_cols is not None else []
            # Union by PUMA; missing pieces are zero-filled.
            keys = sorted(set(base_map.keys()) | set(extra_map.keys()))
            area_stats = {}
            for k in keys:
                a = np.asarray(base_map.get(k, np.zeros(len(base_cols), dtype=np.float32)), dtype=np.float32)
                b = np.asarray(extra_map.get(k, np.zeros(len(extra_cols), dtype=np.float32)), dtype=np.float32)
                area_stats[str(k)] = np.concatenate([a, b], axis=0).astype(np.float32)
            area_stat_cols = list(base_cols) + list(extra_cols)
        cond_root = out_dir / cond_id
        cond_root.mkdir(parents=True, exist_ok=True)

        by_fold_metrics = {}
        for fold in range(int(args.n_folds)):
            fold_dir = cond_root / f"fold_{fold}"
            test_pumas = {p for p, f in fold_of.items() if f == fold}
            train_df = df[~df["PUMA"].isin(test_pumas)].copy().reset_index(drop=True)
            test_df = df[df["PUMA"].isin(test_pumas)].copy().reset_index(drop=True)

            if train_df.empty:
                by_fold_metrics[str(fold)] = {
                    "note": "empty train fold (check n_folds / unique PUMA count)",
                    "n_train_rows": 0,
                    "n_test_rows": int(test_df.shape[0]),
                    "n_test_pumas": int(len(test_pumas)),
                }
                continue

            model = None
            enc = None
            decode_meta = None
            if args.mode == "eval_only":
                model_path = fold_dir / "model.pt"
                enc_path = fold_dir / "encoder.json"
                if not model_path.exists() or not enc_path.exists():
                    by_fold_metrics[str(fold)] = {
                        "note": "missing_checkpoint_or_encoder (eval_only)",
                        "model_path": str(model_path),
                        "encoder_path": str(enc_path),
                        "n_test_rows": int(test_df.shape[0]),
                        "n_test_pumas": int(len(test_pumas)),
                    }
                    continue

                payload = json.loads(enc_path.read_text(encoding="utf-8"))
                enc = _Encoder(**payload["condition"])
                decode_meta = payload["decode_meta"]
                model = DiffusionTabularModel(input_dim=1, cond_dim=0, seed=int(args.seed))
                model.load(model_path)
            else:
                # Resume a partially completed run (common when evaluation crashes after long training).
                if args.resume:
                    model_path = fold_dir / "model.pt"
                    enc_path = fold_dir / "encoder.json"
                    if model_path.exists() and enc_path.exists():
                        payload = json.loads(enc_path.read_text(encoding="utf-8"))
                        enc = _Encoder(**payload["condition"])
                        decode_meta = payload["decode_meta"]
                        model = DiffusionTabularModel(input_dim=1, cond_dim=0, seed=int(args.seed))
                        model.load(model_path)
                        print(f"[info] resume: using existing checkpoint for cond={cond_id} fold={fold} -> {model_path}")
                    else:
                        model = None

                if model is not None and enc is not None and decode_meta is not None:
                    # Already loaded via resume path; skip training.
                    pass
                else:
                    # Encode condition & targets on train.
                    train_df2, cond_train, enc_cond = _encode_condition(
                        df=train_df,
                        puma_col="PUMA",
                        use_race=use_race,
                        use_area_stats=use_area_stats,
                        area_stats=area_stats,
                        area_stat_cols=area_stat_cols,
                        race_cats_global=race_cats_global,
                    )
                    if train_df2.empty:
                        by_fold_metrics[str(fold)] = {
                            "note": "empty train fold after condition filtering (likely missing/invalid RAC1P categories)",
                            "n_train_rows": 0,
                            "n_test_rows": int(test_df.shape[0]),
                            "n_test_pumas": int(len(test_pumas)),
                        }
                        continue
                    x_train, enc_target, decode_meta = _encode_targets(df=train_df2)
                    # Patch encoder with condition info.
                    enc = _Encoder(
                        cond_cols=enc_cond.cond_cols,
                        age_depth=enc_cond.age_depth,
                        sex_depth=enc_cond.sex_depth,
                        race_cats=enc_cond.race_cats,
                        income_mean=enc_target.income_mean,
                        income_std=enc_target.income_std,
                        schl_cats=enc_target.schl_cats,
                        esr_cats=enc_target.esr_cats,
                    )

                    # Torch tensors.
                    x_t = torch.as_tensor(x_train, dtype=torch.float32)
                    c_t = torch.as_tensor(cond_train, dtype=torch.float32)

                    cfg = TabDDPMConfig(
                        timesteps=int(args.timesteps),
                        hidden_dims=hidden_dims,
                        lr=float(args.lr),
                    )
                    model = DiffusionTabularModel(
                        input_dim=int(x_train.shape[1]),
                        cond_dim=int(cond_train.shape[1]),
                        seed=int(args.seed),
                        config=cfg,
                    )
                    train_summary = model.fit(
                        x=x_t,
                        cond=c_t,
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        device=args.device,
                        log_every=int(args.log_every),
                    )

                    fold_dir.mkdir(parents=True, exist_ok=True)
                    model.save(fold_dir / "model.pt")
                    _write_json(fold_dir / "encoder.json", {"condition": asdict(enc), "decode_meta": decode_meta})
                    _write_json(fold_dir / "train_summary.json", train_summary)

            # --- Evaluate on test rows (paired generation: keep conditions from test_df) ---
            assert model is not None
            assert enc is not None
            assert decode_meta is not None
            race_eval = race_cats_global
            if use_race and enc.race_cats:
                try:
                    race_eval = [int(x) for x in enc.race_cats]
                except Exception:
                    race_eval = race_cats_global
            test_df2, cond_test, _ = _encode_condition(
                df=test_df,
                puma_col="PUMA",
                use_race=use_race,
                use_area_stats=use_area_stats,
                area_stats=area_stats,
                area_stat_cols=area_stat_cols,
                race_cats_global=race_eval,
            )
            n_test = int(test_df2.shape[0])
            if n_test == 0:
                by_fold_metrics[str(fold)] = {"note": "empty test fold", "n_test": 0}
                continue

            c_test_t = torch.as_tensor(cond_test, dtype=torch.float32)
            x_hat_t = model.sample(n=n_test, cond=c_test_t, device=args.device)
            x_hat = x_hat_t.detach().cpu().numpy()
            gen = _decode_samples(x_hat=x_hat, enc_target=enc, decode_meta=decode_meta)
            syn = test_df2.reset_index(drop=True).copy()
            syn["PINCP"] = gen["PINCP"].to_numpy(dtype=float)
            syn["SCHL"] = gen["SCHL"].astype(str)
            syn["ESR"] = gen["ESR"].astype(str)

            ref = test_df2.reset_index(drop=True).copy()
            ref["SCHL"] = ref["SCHL"].astype(str)
            ref["ESR"] = ref["ESR"].astype(str)

            # Marginals @ PUMA (PINCP binned, SCHL, ESR)
            # Income bins: reuse scheme-B bins.
            income_edges = [0.0, 10_000.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0, 150_000.0, 250_000.0, 10_000_000.0]
            syn["PINCP_bin"] = pd.cut(syn["PINCP"], bins=income_edges, include_lowest=True, right=False).astype(str)
            ref["PINCP_bin"] = pd.cut(ref["PINCP"], bins=income_edges, include_lowest=True, right=False).astype(str)

            per_puma = {}
            for puma in sorted(test_pumas):
                s_p = syn[syn["PUMA"] == str(puma)]
                r_p = ref[ref["PUMA"] == str(puma)]
                if s_p.empty or r_p.empty:
                    continue
                wcol = "PWGTP"
                m = {
                    "tvd_income_bin": _tvd_from_dists(_weighted_cat_dist(s_p, "PINCP_bin", wcol), _weighted_cat_dist(r_p, "PINCP_bin", wcol)),
                    "tvd_schl": _tvd_from_dists(_weighted_cat_dist(s_p, "SCHL", wcol), _weighted_cat_dist(r_p, "SCHL", wcol)),
                    "tvd_esr": _tvd_from_dists(_weighted_cat_dist(s_p, "ESR", wcol), _weighted_cat_dist(r_p, "ESR", wcol)),
                }

                # Copula TVD (age-income) within PUMA
                u_s = _weighted_rank(s_p["AGEP"].to_numpy(dtype=float), s_p[wcol].to_numpy(dtype=float))
                v_s = _weighted_rank(s_p["PINCP"].to_numpy(dtype=float), s_p[wcol].to_numpy(dtype=float))
                u_r = _weighted_rank(r_p["AGEP"].to_numpy(dtype=float), r_p[wcol].to_numpy(dtype=float))
                v_r = _weighted_rank(r_p["PINCP"].to_numpy(dtype=float), r_p[wcol].to_numpy(dtype=float))
                cop_s = _copula_hist2d(u=u_s, v=v_s, w=s_p[wcol].to_numpy(dtype=float), bins=10)
                cop_r = _copula_hist2d(u=u_r, v=v_r, w=r_p[wcol].to_numpy(dtype=float), bins=10)
                m["copula_tvd_age_income"] = _tvd(cop_s, cop_r)

                # Joint TVD on a coarse joint: (age_idx, income_bin)
                age_idx_s = s_p["AGEP"].astype(int).map(_age_to_p12_idx).astype(int).astype(str)
                age_idx_r = r_p["AGEP"].astype(int).map(_age_to_p12_idx).astype(int).astype(str)
                s_joint = s_p.assign(age_idx=age_idx_s)
                r_joint = r_p.assign(age_idx=age_idx_r)
                m["joint_tvd_age_income_bin"] = _tvd_from_dists(
                    _weighted_joint_dist(s_joint, ["age_idx", "PINCP_bin"], wcol),
                    _weighted_joint_dist(r_joint, ["age_idx", "PINCP_bin"], wcol),
                )
                m["puma_cosine_age_income_bin_joint"] = _cosine_from_dists(
                    _weighted_joint_dist(s_joint, ["age_idx", "PINCP_bin"], wcol),
                    _weighted_joint_dist(r_joint, ["age_idx", "PINCP_bin"], wcol),
                )

                per_puma[str(puma)] = m

            # Summaries over heldout PUMAs
            def _agg(metric: str) -> dict[str, float] | None:
                vals = [per_puma[p][metric] for p in per_puma if per_puma[p].get(metric) is not None]
                if not vals:
                    return None
                arr = np.asarray(vals, dtype=float)
                return {"mean": float(arr.mean()), "max": float(arr.max())}

            def _agg_cos(metric: str) -> dict[str, float] | None:
                vals = [per_puma[p][metric] for p in per_puma if per_puma[p].get(metric) is not None]
                if not vals:
                    return None
                arr = np.asarray(vals, dtype=float)
                return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}

            by_fold_metrics[str(fold)] = {
                "n_test_rows": int(n_test),
                "n_test_pumas": int(len(test_pumas)),
                "by_puma": per_puma,
                "summary": {
                    "tvd_income_bin": _agg("tvd_income_bin"),
                    "tvd_schl": _agg("tvd_schl"),
                    "tvd_esr": _agg("tvd_esr"),
                    "copula_tvd_age_income": _agg("copula_tvd_age_income"),
                    "joint_tvd_age_income_bin": _agg("joint_tvd_age_income_bin"),
                    "puma_cosine_age_income_bin_joint": _agg_cos("puma_cosine_age_income_bin_joint"),
                },
            }

        # condition-level aggregate
        metrics_by_condition[cond_id] = {"by_fold": by_fold_metrics}
        _write_json(cond_root / "metrics_pums_holdout.json", metrics_by_condition[cond_id])

    _write_json(out_dir / "ablation_summary.json", {"by_condition": metrics_by_condition})
    print(f"[ok] wrote: {out_dir}")


if __name__ == "__main__":
    main()
