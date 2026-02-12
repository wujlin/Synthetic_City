#!/usr/bin/env python3
"""
Exp4: End-to-end (Layer 1 -> Layer 2) smoke experiment.

Goal:
- Use Exp1 BG-level base population (age_idx, sex, race, count) as *inputs*.
- Map BG -> PUMA20 (via TIGER) to enable PUMA-context conditioning.
- Apply Exp2 diffusion models (per-fold checkpoints) to generate attributes (PINCP,SCHL,ESR).
- Validate at PUMA level against PUMS microdata (same metric family as Exp2).

Notes:
- This script is designed for workstation execution (CUDA + torch).
- It samples a manageable number of individuals from the Exp1 counts table to avoid expanding 10M+ rows.
- Models (.pt) are usually gitignored; they must exist locally on the workstation under the Exp2 run dir.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import pathlib
import random
import re
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


def _pick_latest_tiger_zip(cands: list[pathlib.Path]) -> pathlib.Path | None:
    """
    Pick the latest-year TIGER zip among candidates like tl_2023_26_bg.zip.
    Falls back to lexicographic order if year cannot be parsed.
    """
    if not cands:
        return None

    def _key(p: pathlib.Path) -> tuple[int, str]:
        m = re.match(r"tl_(\d{4})_", p.name)
        year = int(m.group(1)) if m else -1
        return (year, p.name)

    return sorted(cands, key=_key, reverse=True)[0]


def _auto_find_tiger_zips(
    *, data_root: pathlib.Path, statefp: str
) -> tuple[pathlib.Path | None, pathlib.Path | None, dict[str, list[str]]]:
    """
    Try to locate TIGER BG + PUMA20 zip files under common data_root layouts.
    Returns: (bg_zip, puma_zip, debug_candidates)
    """
    statefp2 = str(statefp).zfill(2)
    search_roots = [
        data_root / "detroit" / "raw" / "census" / "tiger",
        data_root / "detroit" / "raw" / "census",
        data_root / "detroit" / "raw",
        data_root / "detroit",
    ]
    bg_cands: list[pathlib.Path] = []
    puma_cands: list[pathlib.Path] = []
    for root in search_roots:
        root = pathlib.Path(root)
        if not root.exists():
            continue
        bg_cands.extend(list(root.rglob(f"tl_*_{statefp2}_bg.zip")))
        puma_cands.extend(list(root.rglob(f"tl_*_{statefp2}_puma20.zip")))

    bg_cands_u = sorted({p.resolve() for p in bg_cands})
    puma_cands_u = sorted({p.resolve() for p in puma_cands})
    bg_zip = _pick_latest_tiger_zip(bg_cands_u)
    puma_zip = _pick_latest_tiger_zip(puma_cands_u)
    debug = {
        "bg": [str(p) for p in bg_cands_u[:20]],
        "puma20": [str(p) for p in puma_cands_u[:20]],
    }
    return bg_zip, puma_zip, debug


def _stable_hash_fold(values: list[str], *, n_folds: int, seed: int) -> dict[str, int]:
    import hashlib

    out: dict[str, int] = {}
    for v in values:
        h = hashlib.sha1((str(seed) + "::" + str(v)).encode("utf-8")).hexdigest()
        out[str(v)] = int(h[:8], 16) % int(n_folds)
    return out


def _one_hot_codes(codes: Any, depth: int) -> Any:
    np = _require("numpy")
    codes = np.asarray(codes, dtype=int)
    if (codes < 0).any() or (codes >= depth).any():
        raise ValueError("codes out of range for one-hot")
    return np.eye(depth, dtype=np.float32)[codes]


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


def _tvd(p: Any, q: Any) -> float:
    np = _require("numpy")
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p - q).sum())


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


def _load_base_counts(path: pathlib.Path) -> Any:
    pd = _require("pandas")
    p = pathlib.Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"base_counts_path not found: {p}")
    if p.suffix.lower() == ".parquet":
        # Requires pyarrow/fastparquet; we keep the error message explicit.
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv", ".gz"} or p.name.lower().endswith(".csv.gz"):
        return pd.read_csv(p, low_memory=False)
    raise SystemExit(f"Unsupported base_counts format: {p}")


def _sample_from_counts(counts_df: Any, *, n_samples: int, seed: int) -> Any:
    np = _require("numpy")
    pd = _require("pandas")

    req = {"bg_geoid", "sex", "age_idx", "race", "count"}
    missing = [c for c in sorted(req) if c not in counts_df.columns]
    if missing:
        raise SystemExit(f"base_counts missing columns: {missing}")

    d = counts_df.copy()
    d["count"] = pd.to_numeric(d["count"], errors="coerce").fillna(0).astype(int)
    d = d[d["count"] > 0].reset_index(drop=True)
    if d.empty:
        raise SystemExit("base_counts is empty after filtering count>0")

    w = d["count"].to_numpy(dtype=float)
    tot = float(w.sum())
    if tot <= 0:
        raise SystemExit("base_counts total count is 0")
    p = (w / tot).astype(float)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(int(d.shape[0]), size=int(n_samples), replace=True, p=p)
    s = d.iloc[idx].reset_index(drop=True)
    s["bg_geoid"] = s["bg_geoid"].astype(str)
    s["tract_geoid"] = s["bg_geoid"].str.slice(0, 11)
    s["sex"] = pd.to_numeric(s["sex"], errors="coerce").fillna(1).astype(int).clip(lower=1, upper=2)
    s["age_idx"] = pd.to_numeric(s["age_idx"], errors="coerce").fillna(0).astype(int).clip(lower=0, upper=22)
    s["race"] = s["race"].astype(str)
    # weight column for distribution calculations (constant is fine for sampled individuals)
    s["W"] = 1.0
    return s[["bg_geoid", "tract_geoid", "sex", "age_idx", "race", "W"]]


def _bg_to_puma_map(*, tiger_bg_zip: pathlib.Path, tiger_puma_zip: pathlib.Path) -> dict[str, str]:
    gpd = _require("geopandas")

    bg = gpd.read_file(f"zip://{pathlib.Path(tiger_bg_zip).expanduser().resolve()}")
    puma = gpd.read_file(f"zip://{pathlib.Path(tiger_puma_zip).expanduser().resolve()}")

    # BG GEOID
    if "GEOID" in bg.columns:
        bg["bg_geoid"] = bg["GEOID"].astype(str)
    else:
        # Fallback: build from components.
        for c in ["STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE"]:
            if c not in bg.columns:
                raise SystemExit(f"TIGER BG missing {c}; cannot build bg_geoid")
        bg["bg_geoid"] = (
            bg["STATEFP"].astype(str).str.zfill(2)
            + bg["COUNTYFP"].astype(str).str.zfill(3)
            + bg["TRACTCE"].astype(str).str.zfill(6)
            + bg["BLKGRPCE"].astype(str).str.zfill(1)
        )

    # PUMA code (match Exp2's str(int(code)) behavior)
    if "GEOID20" in puma.columns:
        puma_geoid = puma["GEOID20"].astype(str)
        puma_code = puma_geoid.str.slice(-5)
    elif "PUMACE20" in puma.columns:
        puma_code = puma["PUMACE20"].astype(str).str.zfill(5)
    else:
        raise SystemExit("TIGER PUMA missing GEOID20/PUMACE20")
    puma["puma"] = puma_code.astype(int).astype(str)

    bg_cent = bg[["bg_geoid", "geometry"]].copy()
    # Use representative points to avoid centroid-in-geographic-CRS pitfalls and keep points inside polygons.
    bg_cent["geometry"] = bg_cent.geometry.representative_point()
    if bg_cent.crs != puma.crs:
        bg_cent = bg_cent.to_crs(puma.crs)

    joined = gpd.sjoin(bg_cent, puma[["puma", "geometry"]], how="left", predicate="within")
    if "puma" not in joined.columns:
        raise SystemExit("spatial join failed to produce puma column")
    out = {str(r["bg_geoid"]): (None if r["puma"] is None else str(r["puma"])) for _, r in joined.iterrows()}
    # drop missing
    out2 = {k: v for k, v in out.items() if v is not None}
    return out2


def _build_puma_stats(*, pums_df: Any, puma_col: str, wcol: str) -> tuple[dict[str, Any], list[str]]:
    pd = _require("pandas")
    np = _require("numpy")

    d = pums_df[[puma_col, wcol, "AGEP", "PINCP"]].copy()
    d[puma_col] = d[puma_col].astype(str)
    d[wcol] = pd.to_numeric(d[wcol], errors="coerce").fillna(0.0).clip(lower=0.0)
    d["AGEP"] = pd.to_numeric(d["AGEP"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=99.0)
    d["PINCP"] = pd.to_numeric(d["PINCP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    d["PINCP_log"] = np.log1p(d["PINCP"].to_numpy(dtype=np.float32))

    def _wmean(x: Any, w: Any) -> float:
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
    mu = feat[cols].mean(axis=0, numeric_only=True)
    sd = feat[cols].std(axis=0, ddof=0, numeric_only=True).replace(0.0, 1.0)
    feat_z = (feat[cols] - mu) / sd
    out: dict[str, Any] = {}
    for i, r in feat.iterrows():
        out[str(r["puma"])] = feat_z.iloc[i].to_numpy(dtype=np.float32)
    return out, [f"puma_{c}_z" for c in cols]


def _decode_samples(*, x_hat: Any, enc: dict[str, Any], decode_meta: dict[str, Any]) -> Any:
    pd = _require("pandas")
    np = _require("numpy")

    x_hat = np.asarray(x_hat, dtype=np.float32)
    inc_z = x_hat[:, 0]
    inc_log = inc_z * float(enc["income_std"]) + float(enc["income_mean"])
    inc_log = np.clip(inc_log, -10.0, 20.0)
    income = np.expm1(inc_log).clip(min=0.0)

    s0, s1 = decode_meta["schl_slice"]
    e0, e1 = decode_meta["esr_slice"]
    schl_logits = x_hat[:, s0:s1]
    esr_logits = x_hat[:, e0:e1]
    schl_idx = schl_logits.argmax(axis=1) if schl_logits.shape[1] > 0 else np.zeros(x_hat.shape[0], dtype=int)
    esr_idx = esr_logits.argmax(axis=1) if esr_logits.shape[1] > 0 else np.zeros(x_hat.shape[0], dtype=int)

    schl_cats = list(enc.get("schl_cats") or [])
    esr_cats = list(enc.get("esr_cats") or [])
    schl = [schl_cats[int(i)] if schl_cats else "0" for i in schl_idx]
    esr = [esr_cats[int(i)] if esr_cats else "0" for i in esr_idx]

    return pd.DataFrame({"PINCP": income.astype(float), "SCHL": [str(x) for x in schl], "ESR": [str(x) for x in esr]})


def main() -> None:
    pd = _require("pandas")
    np = _require("numpy")
    torch = _require("torch")

    from src.synthpop.model.diffusion_tabular import DiffusionTabularModel
    from src.synthpop.pipeline.detroit_v0 import make_run_id
    from src.synthpop.paths import data_root as default_data_root

    ap = argparse.ArgumentParser(prog="exp4_end_to_end_l1_l2")
    ap.add_argument("--data_root", default=str(default_data_root()))
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--pums_year", type=int, default=2023)
    ap.add_argument("--pums_period", default="5-Year")
    ap.add_argument("--exp1_counts_path", required=True, help="Exp1 counts table (parquet/csv)")
    ap.add_argument("--exp2_run_dir", required=True, help="Exp2 run dir (must contain model.pt/encoder.json)")
    ap.add_argument("--condition", default="demo_race_puma", help="Which Exp2 condition to use for sampling.")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=200000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiger_bg_zip", default=None, help="Optional TIGER BG zip (tl_2023_26_bg.zip)")
    ap.add_argument("--tiger_puma_zip", default=None, help="Optional TIGER PUMA zip (tl_2023_26_puma20.zip)")
    ap.add_argument("--cache_bg_to_puma", action="store_true", help="Cache BG->PUMA map to out_dir for reuse.")
    ap.add_argument("--save_samples_csv_gz", action="store_true", help="Write sampled synthetic microdata (csv.gz).")
    ap.add_argument("--out_dir", default=None, help="Default: outputs/<run_id> under repo.")
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    out_dir = (
        pathlib.Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (_REPO_ROOT / "outputs" / make_run_id(prefix="exp4_end_to_end_l1_l2"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        out_dir / "run.metadata.json",
        {
            "created_utc": _utc_now_iso(),
            "argv": sys.argv,
            "env": {"RAW_ROOT": os.environ.get("RAW_ROOT"), "SYNTHCITY_DATA_ROOT": os.environ.get("SYNTHCITY_DATA_ROOT")},
            "args": vars(args),
        },
    )

    # --- Load base population counts and sample individuals ---
    counts_df = _load_base_counts(pathlib.Path(args.exp1_counts_path))
    base_sample = _sample_from_counts(counts_df, n_samples=int(args.n_samples), seed=int(args.seed))

    # --- Map BG -> PUMA20 via TIGER ---
    cache_path = out_dir / "bg_to_puma.json"
    bg_to_puma = None
    if cache_path.exists():
        bg_to_puma = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        tiger_bg_zip = pathlib.Path(args.tiger_bg_zip).expanduser().resolve() if args.tiger_bg_zip else None
        tiger_puma_zip = pathlib.Path(args.tiger_puma_zip).expanduser().resolve() if args.tiger_puma_zip else None
        debug_cands = None
        if tiger_bg_zip is None or tiger_puma_zip is None:
            bg2, puma2, debug_cands = _auto_find_tiger_zips(data_root=data_root, statefp=str(args.statefp))
            if tiger_bg_zip is None:
                tiger_bg_zip = bg2
            if tiger_puma_zip is None:
                tiger_puma_zip = puma2
        if tiger_bg_zip is None or tiger_puma_zip is None:
            msg = "Missing TIGER zips. Provide --tiger_bg_zip and --tiger_puma_zip.\n"
            if debug_cands is not None:
                msg += f"Auto-search candidates (first 20 each): {json.dumps(debug_cands, ensure_ascii=False)}\n"
            msg += "Expected filenames like tl_2023_<STATEFP>_bg.zip and tl_2023_<STATEFP>_puma20.zip under $DATA_ROOT/detroit/raw/census/.\n"
            raise SystemExit(msg)
        bg_to_puma = _bg_to_puma_map(tiger_bg_zip=tiger_bg_zip, tiger_puma_zip=tiger_puma_zip)
        if bool(args.cache_bg_to_puma):
            _write_json(cache_path, bg_to_puma)

    base_sample["puma"] = base_sample["bg_geoid"].map(lambda x: bg_to_puma.get(str(x)))
    base_sample = base_sample[base_sample["puma"].notna()].reset_index(drop=True)
    base_sample["puma"] = base_sample["puma"].astype(str)
    if base_sample.empty:
        raise SystemExit("No sampled rows mapped to PUMA (BG->PUMA mapping failed?)")

    # --- Load PUMS reference (for evaluation + PUMA stats) ---
    pums_zip = _resolve_pums_person_zip(
        data_root=data_root, pums_year=int(args.pums_year), pums_period=str(args.pums_period), statefp=str(args.statefp)
    )
    member = _find_first_csv_in_zip(pums_zip)
    usecols = ["PUMA", "PUMA20", "PWGTP", "AGEP", "SEX", "PINCP", "SCHL", "ESR", "RAC1P"]
    with zipfile.ZipFile(pums_zip) as zf, zf.open(member) as f:
        ref = pd.read_csv(f, usecols=lambda c: c in set(usecols), low_memory=False)
    if "PUMA20" in ref.columns:
        ref["PUMA"] = ref["PUMA20"]
    if "PUMA" not in ref.columns:
        raise SystemExit("PUMS reference missing PUMA/PUMA20")
    puma_num = pd.to_numeric(ref["PUMA"], errors="coerce")
    ref = ref[puma_num.notna() & (puma_num != -9)].copy()
    ref["puma"] = ref["PUMA"].astype(str)
    ref["PWGTP"] = pd.to_numeric(ref["PWGTP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    ref["AGEP"] = pd.to_numeric(ref["AGEP"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=99.0)
    ref["PINCP"] = pd.to_numeric(ref["PINCP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    ref["SCHL"] = pd.to_numeric(ref["SCHL"], errors="coerce").fillna(0).astype(int).astype(str)
    ref["ESR"] = pd.to_numeric(ref["ESR"], errors="coerce").fillna(0).astype(int).astype(str)
    ref = ref[ref["PWGTP"] > 0].copy()

    # ref already includes both numeric-ish 'PUMA' and string 'puma'.
    # Avoid duplicate column labels by using 'puma' for grouping.
    puma_stats, puma_stat_cols = _build_puma_stats(pums_df=ref, puma_col="puma", wcol="PWGTP")

    # --- Fold assignment (same as Exp2) ---
    pumas_all = sorted(set(ref["puma"].unique().tolist()) | set(base_sample["puma"].unique().tolist()))
    fold_of = _stable_hash_fold(pumas_all, n_folds=int(args.n_folds), seed=int(args.seed))
    base_sample["fold"] = base_sample["puma"].map(lambda p: fold_of.get(str(p), 0)).astype(int)

    # --- Load models & encoders ---
    exp2_dir = pathlib.Path(args.exp2_run_dir).expanduser().resolve()
    cond_dir = exp2_dir / str(args.condition)
    if not cond_dir.exists():
        raise SystemExit(f"Condition dir not found: {cond_dir}")

    models: dict[int, Any] = {}
    encoders: dict[int, dict[str, Any]] = {}
    decode_metas: dict[int, dict[str, Any]] = {}
    for fold in range(int(args.n_folds)):
        fold_dir = cond_dir / f"fold_{fold}"
        model_path = fold_dir / "model.pt"
        enc_path = fold_dir / "encoder.json"
        if not model_path.exists() or not enc_path.exists():
            raise SystemExit(f"Missing model/encoder for fold={fold}: {model_path} {enc_path}")
        payload = json.loads(enc_path.read_text(encoding="utf-8"))
        enc = dict(payload["condition"])
        decode_meta = dict(payload["decode_meta"])
        model = DiffusionTabularModel(input_dim=1, cond_dim=0, seed=int(args.seed))
        model.load(model_path)
        models[int(fold)] = model
        encoders[int(fold)] = enc
        decode_metas[int(fold)] = decode_meta

    # --- Prepare conditioning vectors (per fold) and sample attributes ---
    # Exp1 uses DHC/P5 7-race labels; Exp2 conditions on PUMS RAC1P (1..9).
    # We map each 7-race label back to a representative RAC1P code:
    # - AIAN corresponds to RAC1P in (3,4,5); we pick 3 for determinism.
    # - Asian=6, NHPI=7, Other=8, Two+ = 9 in ACS PUMS.
    race_label_to_code = {
        "white": 1,
        "black": 2,
        "aian": 3,
        "asian": 6,
        "nhpi": 7,
        "other": 8,
        "two_or_more": 9,
    }

    out_rows = []
    for fold in range(int(args.n_folds)):
        part = base_sample[base_sample["fold"] == int(fold)].reset_index(drop=True)
        if part.empty:
            continue
        enc = encoders[int(fold)]
        race_cats = [int(x) for x in (enc.get("race_cats") or [])] if enc.get("race_cats") else []
        # Condition parts in the same order as Exp2 _encode_condition.
        age_oh = _one_hot_codes(part["age_idx"].to_numpy(dtype=int), 23)
        sex_idx = part["sex"].to_numpy(dtype=int) - 1
        sex_oh = _one_hot_codes(sex_idx, 2)
        cond_parts = [age_oh, sex_oh]
        if race_cats:
            codes = part["race"].map(lambda s: race_label_to_code.get(str(s), -1)).to_numpy(dtype=int)
            if (codes < 0).any():
                raise SystemExit("Unknown race label in base population sample; cannot encode race.")
            # Map to category index
            cat_to_idx = {int(c): i for i, c in enumerate(race_cats)}
            r_idx = np.asarray([cat_to_idx.get(int(c), -1) for c in codes], dtype=int)
            if (r_idx < 0).any():
                raise SystemExit("Race codes not covered by encoder categories; check Exp2 training race_cats.")
            race_oh = _one_hot_codes(r_idx, len(race_cats))
            cond_parts.append(race_oh)
        # PUMA stats (if present in encoder cond_cols)
        if any(str(c).startswith("puma_") for c in (enc.get("cond_cols") or [])):
            p = part["puma"].astype(str).to_numpy()
            stats = np.stack([np.asarray(puma_stats.get(str(x), np.zeros(5, dtype=np.float32)), dtype=np.float32) for x in p], axis=0)
            cond_parts.append(stats.astype(np.float32))

        cond_np = np.concatenate(cond_parts, axis=1).astype(np.float32)
        cond_dim_expected = int(models[int(fold)].cond_dim)
        if int(cond_np.shape[1]) != int(cond_dim_expected):
            raise SystemExit(f"cond_dim mismatch for fold={fold}: built {cond_np.shape[1]} vs model {cond_dim_expected}")

        cond_t = torch.as_tensor(cond_np, dtype=torch.float32)
        x_hat_t = models[int(fold)].sample(n=int(cond_np.shape[0]), cond=cond_t, device=str(args.device))
        x_hat = x_hat_t.detach().cpu().numpy()
        gen = _decode_samples(x_hat=x_hat, enc=enc, decode_meta=decode_metas[int(fold)])
        part_out = part.copy()
        part_out["PINCP"] = gen["PINCP"].to_numpy(dtype=float)
        part_out["SCHL"] = gen["SCHL"].astype(str)
        part_out["ESR"] = gen["ESR"].astype(str)
        out_rows.append(part_out)

    syn = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    if syn.empty:
        raise SystemExit("No synthetic rows produced.")

    # Optional: persist sampled microdata for debugging (not meant for git sync).
    if bool(args.save_samples_csv_gz):
        out_csv = out_dir / "synthetic_sample.csv.gz"
        syn.to_csv(out_csv, index=False, compression="gzip")

    # --- Evaluate vs PUMS reference at PUMA level (same metric family as Exp2) ---
    income_edges = [0.0, 10_000.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0, 150_000.0, 250_000.0, 10_000_000.0]
    syn["PINCP_bin"] = pd.cut(syn["PINCP"], bins=income_edges, include_lowest=True, right=False).astype(str)
    ref["PINCP_bin"] = pd.cut(ref["PINCP"], bins=income_edges, include_lowest=True, right=False).astype(str)

    per_puma = {}
    for puma in sorted(set(syn["puma"].unique().tolist())):
        s_p = syn[syn["puma"] == str(puma)]
        r_p = ref[ref["puma"] == str(puma)]
        if s_p.empty or r_p.empty:
            continue
        if int(s_p.shape[0]) < 200:
            # Too noisy for diagnostics.
            continue
        m = {
            "tvd_income_bin": _tvd_from_dists(_weighted_cat_dist(s_p, "PINCP_bin", "W"), _weighted_cat_dist(r_p, "PINCP_bin", "PWGTP")),
            "tvd_schl": _tvd_from_dists(_weighted_cat_dist(s_p, "SCHL", "W"), _weighted_cat_dist(r_p, "SCHL", "PWGTP")),
            "tvd_esr": _tvd_from_dists(_weighted_cat_dist(s_p, "ESR", "W"), _weighted_cat_dist(r_p, "ESR", "PWGTP")),
        }
        # Copula TVD (AGEP,PINCP) on ranks; approximate AGEP from age_idx midpoints.
        age_mid = np.array([2.0, 7.0, 12.0, 16.0, 19.0, 20.0, 21.0, 23.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0, 57.0, 61.0, 64.0, 66.0, 69.0, 74.0, 79.0, 84.0, 90.0])
        s_age = age_mid[s_p["age_idx"].to_numpy(dtype=int)]
        r_age = r_p["AGEP"].to_numpy(dtype=float)
        u_s = _weighted_rank(s_age, s_p["W"].to_numpy(dtype=float))
        v_s = _weighted_rank(s_p["PINCP"].to_numpy(dtype=float), s_p["W"].to_numpy(dtype=float))
        u_r = _weighted_rank(r_age, r_p["PWGTP"].to_numpy(dtype=float))
        v_r = _weighted_rank(r_p["PINCP"].to_numpy(dtype=float), r_p["PWGTP"].to_numpy(dtype=float))
        cop_s = _copula_hist2d(u=u_s, v=v_s, w=s_p["W"].to_numpy(dtype=float), bins=10)
        cop_r = _copula_hist2d(u=u_r, v=v_r, w=r_p["PWGTP"].to_numpy(dtype=float), bins=10)
        m["copula_tvd_age_income"] = _tvd(cop_s, cop_r)

        # Joint TVD (age_idx, income_bin)
        s_joint = s_p.assign(age_idx=s_p["age_idx"].astype(int).astype(str))
        r_joint = r_p.assign(age_idx=r_p["AGEP"].astype(int).map(lambda a: int(a) if a >= 0 else 0).map(lambda a: a))  # placeholder
        # Reuse age_idx bins from midpoints: map ref AGEP to same 23 bins
        r_joint["age_idx"] = r_p["AGEP"].astype(int).map(lambda a: int(a)).map(lambda a: a)
        # Map to bins with the same cutpoints by reusing the edges in Exp2 attribute script.
        # Simpler: approximate by mapping to closest midpoint.
        r_age = r_p["AGEP"].to_numpy(dtype=int)
        # vectorized mapping
        def _age_to_idx_vec(a: Any) -> Any:
            a = np.asarray(a, dtype=int)
            out = np.zeros_like(a, dtype=int)
            for i, v in enumerate(a.tolist()):
                # reuse the 23-bin logic via midpoint thresholds
                # (fast enough for eval sizes; correctness is not critical for this end-to-end smoke test)
                if v <= 4:
                    out[i] = 0
                elif v <= 9:
                    out[i] = 1
                elif v <= 14:
                    out[i] = 2
                elif v <= 17:
                    out[i] = 3
                elif v <= 19:
                    out[i] = 4
                elif v == 20:
                    out[i] = 5
                elif v == 21:
                    out[i] = 6
                elif v <= 24:
                    out[i] = 7
                elif v <= 29:
                    out[i] = 8
                elif v <= 34:
                    out[i] = 9
                elif v <= 39:
                    out[i] = 10
                elif v <= 44:
                    out[i] = 11
                elif v <= 49:
                    out[i] = 12
                elif v <= 54:
                    out[i] = 13
                elif v <= 59:
                    out[i] = 14
                elif v <= 61:
                    out[i] = 15
                elif v <= 64:
                    out[i] = 16
                elif v <= 66:
                    out[i] = 17
                elif v <= 69:
                    out[i] = 18
                elif v <= 74:
                    out[i] = 19
                elif v <= 79:
                    out[i] = 20
                elif v <= 84:
                    out[i] = 21
                else:
                    out[i] = 22
            return out

        r_joint["age_idx"] = _age_to_idx_vec(r_age).astype(int).astype(str)
        m["joint_tvd_age_income_bin"] = _tvd_from_dists(
            _weighted_joint_dist(s_joint, ["age_idx", "PINCP_bin"], "W"),
            _weighted_joint_dist(r_p.assign(age_idx=r_joint["age_idx"]), ["age_idx", "PINCP_bin"], "PWGTP"),
        )
        m["puma_cosine_age_income_bin_joint"] = _cosine_from_dists(
            _weighted_joint_dist(s_joint, ["age_idx", "PINCP_bin"], "W"),
            _weighted_joint_dist(r_p.assign(age_idx=r_joint["age_idx"]), ["age_idx", "PINCP_bin"], "PWGTP"),
        )

        per_puma[str(puma)] = m

    # Summary across PUMAs
    def _agg(metric: str) -> dict[str, float] | None:
        vals = [per_puma[p][metric] for p in per_puma if per_puma[p].get(metric) is not None]
        if not vals:
            return None
        arr = np.asarray(vals, dtype=float)
        return {"mean": float(arr.mean()), "max": float(arr.max()), "n_pumas": int(arr.size)}

    def _agg_cos(metric: str) -> dict[str, float] | None:
        vals = [per_puma[p][metric] for p in per_puma if per_puma[p].get(metric) is not None]
        if not vals:
            return None
        arr = np.asarray(vals, dtype=float)
        return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max()), "n_pumas": int(arr.size)}

    out_metrics = {
        "by_puma": per_puma,
        "summary": {
            **{k: _agg(k) for k in ["tvd_income_bin", "tvd_schl", "tvd_esr", "copula_tvd_age_income", "joint_tvd_age_income_bin"]},
            "puma_cosine_age_income_bin_joint": _agg_cos("puma_cosine_age_income_bin_joint"),
        },
    }
    _write_json(out_dir / "end_to_end_puma_metrics.json", out_metrics)

    print(f"[ok] wrote: {out_dir}")


if __name__ == "__main__":
    main()
