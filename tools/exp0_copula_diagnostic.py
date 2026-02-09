#!/usr/bin/env python3
"""
Exp 0: Copula spatial heterogeneity diagnostic (Michigan PUMS).

Goal (problem-driven):
- Measure whether a target dependence structure ("copula") varies across areas (PUMA).
- If copulas are near-constant, IPF with a global seed joint is likely sufficient and diffusion
  offers limited incremental value for copula learning.

Method (KISS, plan v2):
- Use rank transform (weighted, if weights available) to map each variable to ~Uniform(0,1).
- Estimate copula per PUMA as a 2D histogram on (u, v) in [0,1]^2.
- Quantify heterogeneity via TVD to a global copula and sampled pairwise TVD.
- (Optional) basic predictability check via simple feature-based binning (no sklearn dependency).

Outputs (small, git-friendly):
  outputs/<run_id>/
    copula_by_puma.json
    tvd_to_global.json
    pairwise_tvd_summary.json
    cluster_analysis.json
    diagnostic_summary.json
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
from typing import Any

# Allow running as a plain script without installing the repo.
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


def _resolve_pums_person_zip(
    *,
    data_root: pathlib.Path,
    pums_year: int,
    pums_period: str,
    statefp: str,
) -> pathlib.Path:
    """
    Resolve a PUMS person zip under the project layout.
    We accept either naming convention:
      - psam_p{statefp}.zip
      - csv_p{state_postal_lower}i.zip  (MI example: csv_pmi.zip)
    """
    statefp = str(statefp).zfill(2)
    # For v0 we only special-case MI for the "csv_*" naming.
    state_postal_lower = "mi" if statefp == "26" else None

    raw_dir = data_root / "detroit" / "raw" / "pums" / f"pums_{pums_year}_{pums_period}"
    candidates: list[pathlib.Path] = [raw_dir / f"psam_p{statefp}.zip"]
    if state_postal_lower is not None:
        candidates.append(raw_dir / f"csv_p{state_postal_lower}i.zip")  # csv_pmi.zip
        candidates.append(raw_dir / f"csv_p{state_postal_lower}.zip")  # fallback

    for p in candidates:
        if p.exists():
            return p

    # Fallback: scan detroit/raw/pums.
    search_root = data_root / "detroit" / "raw" / "pums"
    patterns = [f"psam_p{statefp}.zip"]
    if state_postal_lower is not None:
        patterns.extend([f"csv_p{state_postal_lower}i.zip", f"csv_p{state_postal_lower}.zip"])
    found: list[pathlib.Path] = []
    for pat in patterns:
        found.extend(sorted(search_root.glob(f"**/{pat}")))
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        msg = "\n".join([str(p) for p in found[:10]])
        raise SystemExit(
            "Multiple PUMS person zips found; please keep only one or pass --pums_person_zip.\n"
            f"Candidates (first 10):\n{msg}"
        )
    raise SystemExit(
        "PUMS person zip not found.\n"
        f"Tried:\n  - " + "\n  - ".join(str(p) for p in candidates)
    )


def _weighted_rank(u: "Any", w: "Any") -> "Any":
    """
    Weighted rank transform to (0,1), using mid-point CDF:
      r_i = (cum_w_i - 0.5*w_i) / total_w
    """
    np = _require("numpy")
    u = np.asarray(u)
    w = np.asarray(w, dtype=float)
    if u.ndim != 1 or w.ndim != 1 or u.shape[0] != w.shape[0]:
        raise ValueError("u and w must be 1D arrays with same length")
    if u.shape[0] == 0:
        return np.asarray([], dtype=float)
    w = np.clip(w, 0.0, None)
    tot = float(w.sum())
    if not math.isfinite(tot) or tot <= 0.0:
        # fallback: unweighted ranks
        order = np.argsort(u, kind="mergesort")
        r = np.empty_like(order, dtype=float)
        r[order] = (np.arange(u.shape[0], dtype=float) + 0.5) / float(u.shape[0])
        return r

    order = np.argsort(u, kind="mergesort")
    w_sorted = w[order]
    cw = np.cumsum(w_sorted, dtype=float)
    r_sorted = (cw - 0.5 * w_sorted) / tot
    r = np.empty_like(r_sorted)
    r[order] = r_sorted
    # Guard against numeric drift.
    return np.clip(r, 0.0, 1.0)


def _weighted_median(x: "Any", w: "Any") -> float | None:
    np = _require("numpy")
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return None
    x = x[mask]
    w = w[mask]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    w = w[order]
    cw = np.cumsum(w)
    cutoff = 0.5 * float(w.sum())
    idx = int(np.searchsorted(cw, cutoff, side="left"))
    idx = max(0, min(idx, x.shape[0] - 1))
    return float(x[idx])


def _tvd(p: "Any", q: "Any") -> float:
    np = _require("numpy")
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p - q).sum())


def _hist2d_copula(*, u: "Any", v: "Any", w: "Any", bins: int) -> "Any":
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


def main() -> None:
    pd = _require("pandas")
    np = _require("numpy")

    from src.synthpop.pipeline.detroit_v0 import make_run_id
    from src.synthpop.paths import data_root as default_data_root

    p = argparse.ArgumentParser(prog="exp0_copula_diagnostic")
    p.add_argument("--data_root", default=str(default_data_root()), help="Project data root (RAW_ROOT layout).")
    p.add_argument("--pums_year", type=int, default=2023)
    p.add_argument("--pums_period", default="5-Year")
    p.add_argument("--statefp", default="26")
    p.add_argument("--pums_person_zip", default=None, help="Override PUMS person zip path (psam_pXX.zip).")
    p.add_argument("--n_rows", type=int, default=None, help="Optional row cap for quick smoke runs.")
    p.add_argument("--group_col", default="PUMA")
    p.add_argument("--x_col", default="AGEP")
    p.add_argument("--y_col", default="PINCP")
    p.add_argument("--weight_col", default="PWGTP")
    p.add_argument("--use_weights", action="store_true", help="Use weights if available (recommended).")
    p.add_argument("--bins", type=int, default=10, help="Copula histogram bins (e.g., 10).")
    p.add_argument("--pairwise_pairs", type=int, default=20000, help="Sampled pairwise TVD pairs (0 disables).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=None, help="Default: outputs/<run_id> under repo.")
    args = p.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    if args.pums_person_zip:
        person_zip = pathlib.Path(args.pums_person_zip).expanduser().resolve()
    else:
        person_zip = _resolve_pums_person_zip(
            data_root=data_root,
            pums_year=int(args.pums_year),
            pums_period=str(args.pums_period),
            statefp=str(args.statefp),
        )

    out_dir = (
        pathlib.Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (_REPO_ROOT / "outputs" / make_run_id(prefix="exp0_copula_diagnostic"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load PUMS person file ---
    member = _find_first_csv_in_zip(person_zip)
    usecols = [str(args.group_col), str(args.x_col), str(args.y_col)]
    if args.use_weights:
        usecols.append(str(args.weight_col))

    with zipfile.ZipFile(person_zip) as zf, zf.open(member) as f:
        df = pd.read_csv(f, nrows=args.n_rows, usecols=lambda c: c in set(usecols), low_memory=False)

    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise SystemExit(f"PUMS person file missing required columns: {missing} (zip={person_zip} member={member})")

    gcol = str(args.group_col)
    xcol = str(args.x_col)
    ycol = str(args.y_col)
    wcol = str(args.weight_col)

    df[gcol] = df[gcol].astype(str)
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce").fillna(0.0).clip(lower=0.0)
    if args.use_weights:
        df[wcol] = pd.to_numeric(df[wcol], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df[wcol] = 1.0

    df = df[np.isfinite(df[xcol]) & np.isfinite(df[ycol]) & np.isfinite(df[wcol]) & (df[wcol] > 0)].copy()
    if df.empty:
        raise SystemExit("No valid rows after cleaning; check input columns and filters.")

    # --- compute copula per group ---
    bins = int(args.bins)
    copula_by_group: dict[str, list[list[float]]] = {}
    group_pop: dict[str, float] = {}
    group_n: dict[str, int] = {}

    for g, gdf in df.groupby(gcol, sort=False):
        w = gdf[wcol].to_numpy(dtype=float)
        u = _weighted_rank(gdf[xcol].to_numpy(dtype=float), w)
        v = _weighted_rank(gdf[ycol].to_numpy(dtype=float), w)
        cop = _hist2d_copula(u=u, v=v, w=w, bins=bins)
        copula_by_group[str(g)] = cop.tolist()
        group_pop[str(g)] = float(w.sum())
        group_n[str(g)] = int(gdf.shape[0])

    groups = sorted(copula_by_group.keys())
    pops = np.array([group_pop[g] for g in groups], dtype=float)
    cops = np.stack([np.asarray(copula_by_group[g], dtype=float) for g in groups], axis=0)  # (G,B,B)
    pop_tot = float(pops.sum())
    if pop_tot <= 0:
        pop_tot = float(len(groups))
        pops = np.ones_like(pops)
    global_copula = (cops * pops.reshape(-1, 1, 1)).sum(axis=0) / pop_tot

    tvd_to_global: dict[str, float] = {}
    for i, g in enumerate(groups):
        tvd_to_global[g] = _tvd(cops[i], global_copula)

    tvd_vals = np.array(list(tvd_to_global.values()), dtype=float)
    tvd_summary = {
        "mean": float(tvd_vals.mean()),
        "p50": float(np.quantile(tvd_vals, 0.50)),
        "p90": float(np.quantile(tvd_vals, 0.90)),
        "max": float(tvd_vals.max()),
    }

    # --- sampled pairwise tvd ---
    pairwise_summary: dict[str, Any] = {"n_pairs": 0, "mean": None, "p50": None, "p90": None, "max": None}
    n_pairs = int(args.pairwise_pairs)
    if n_pairs > 0 and len(groups) >= 2:
        rng = random.Random(int(args.seed))
        tvds = []
        for _ in range(n_pairs):
            i = rng.randrange(len(groups))
            j = rng.randrange(len(groups) - 1)
            if j >= i:
                j += 1
            tvds.append(_tvd(cops[i], cops[j]))
        arr = np.asarray(tvds, dtype=float)
        pairwise_summary = {
            "n_pairs": int(arr.size),
            "mean": float(arr.mean()),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "max": float(arr.max()),
        }

    # --- lightweight "predictability" check (feature binning) ---
    # Use two simple PUMA features from PUMS: weighted median age & income.
    feat_rows = []
    for g, gdf in df.groupby(gcol, sort=False):
        w = gdf[wcol].to_numpy(dtype=float)
        feat_rows.append(
            {
                "puma": str(g),
                "pop_w": float(w.sum()),
                "median_age": _weighted_median(gdf[xcol].to_numpy(dtype=float), w),
                "median_income": _weighted_median(gdf[ycol].to_numpy(dtype=float), w),
                "pct_elderly": float((w[gdf[xcol].to_numpy(dtype=float) >= 65].sum()) / max(float(w.sum()), 1e-9)),
                "pct_child": float((w[gdf[xcol].to_numpy(dtype=float) < 18].sum()) / max(float(w.sum()), 1e-9)),
            }
        )
    feat_df = pd.DataFrame(feat_rows)
    feat_df["median_age"] = pd.to_numeric(feat_df["median_age"], errors="coerce")
    feat_df["median_income"] = pd.to_numeric(feat_df["median_income"], errors="coerce")

    # Bin by income quintiles (fallback if too few groups).
    q = 5 if feat_df.shape[0] >= 25 else max(2, min(5, feat_df.shape[0] // 5 + 1))
    try:
        feat_df["cluster"] = pd.qcut(feat_df["median_income"].rank(method="first"), q=q, labels=False, duplicates="drop")
    except Exception:
        feat_df["cluster"] = 0

    cluster_map = {str(r["puma"]): int(r["cluster"]) for _, r in feat_df.iterrows()}
    clusters = sorted(set(cluster_map.values()))

    intra = []
    inter = []
    rng = random.Random(int(args.seed) + 1)
    # sample up to 20k pairs for stability
    pair_budget = min(20000, max(0, len(groups) * 50))
    for _ in range(pair_budget):
        i = rng.randrange(len(groups))
        j = rng.randrange(len(groups) - 1)
        if j >= i:
            j += 1
        gi = groups[i]
        gj = groups[j]
        d = _tvd(cops[i], cops[j])
        if cluster_map.get(gi) == cluster_map.get(gj):
            intra.append(d)
        else:
            inter.append(d)
    intra_arr = np.asarray(intra, dtype=float) if intra else None
    inter_arr = np.asarray(inter, dtype=float) if inter else None
    cluster_analysis = {
        "n_clusters": int(len(clusters)),
        "cluster_sizes": {str(c): int(sum(1 for g in groups if cluster_map.get(g) == c)) for c in clusters},
        "intra_tvd": None
        if intra_arr is None
        else {
            "n": int(intra_arr.size),
            "mean": float(intra_arr.mean()),
            "p90": float(np.quantile(intra_arr, 0.90)),
        },
        "inter_tvd": None
        if inter_arr is None
        else {
            "n": int(inter_arr.size),
            "mean": float(inter_arr.mean()),
            "p90": float(np.quantile(inter_arr, 0.90)),
        },
        "note": "Clusters are income-quantile bins (no sklearn). If inter>>intra, features may predict copula variation.",
    }

    # --- decision heuristic (plan v2 thresholds) ---
    mean_tvd = float(tvd_summary["mean"])
    if mean_tvd < 0.03:
        decision = "copula_global_constant"
        recommendation = "Diffusion is unlikely to add value for copula; consider IPF with global seed joint."
    elif mean_tvd < 0.08:
        decision = "copula_weakly_heterogeneous"
        recommendation = "Diffusion may add limited value; proceed but prioritize strong signals and regularization."
    else:
        decision = "copula_heterogeneous"
        recommendation = "Copula varies across areas; diffusion has a plausible target to learn beyond IPF."

    diagnostic_summary = {
        "created_utc": _utc_now_iso(),
        "inputs": {
            "pums_person_zip": str(person_zip),
            "member": member,
            "n_rows": int(df.shape[0]),
            "group_col": gcol,
            "x_col": xcol,
            "y_col": ycol,
            "weight_col": wcol,
            "use_weights": bool(args.use_weights),
            "bins": bins,
        },
        "n_groups": int(len(groups)),
        "tvd_to_global": tvd_summary,
        "pairwise_tvd": pairwise_summary,
        "cluster_analysis": cluster_analysis,
        "decision": {"label": decision, "recommendation": recommendation},
    }

    # --- write outputs ---
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
    _write_json(out_dir / "copula_by_puma.json", {"bins": bins, "groups": groups, "copula": copula_by_group, "pop_w": group_pop, "n_rows": group_n})
    _write_json(out_dir / "tvd_to_global.json", {"tvd_to_global": tvd_to_global, "global_copula": global_copula.tolist()})
    _write_json(out_dir / "pairwise_tvd_summary.json", pairwise_summary)
    _write_json(out_dir / "cluster_analysis.json", cluster_analysis)
    _write_json(out_dir / "diagnostic_summary.json", diagnostic_summary)

    print(f"[ok] wrote: {out_dir}")


if __name__ == "__main__":
    main()

