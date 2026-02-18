#!/usr/bin/env python3
"""
Exp C: Copula baseline comparison on Exp2 holdout folds.

Goal:
- Compare diffusion predictions against a simple train-average baseline:
  for each fold, use the *training* global copula / joint distribution as predictor
  for all heldout PUMAs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
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
        candidates.append(raw_dir / f"csv_p{state_postal_lower}i.zip")
        candidates.append(raw_dir / f"csv_p{state_postal_lower}.zip")
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit(f"PUMS person zip not found. Tried: {candidates}")


def _tvd(p: Any, q: Any) -> float:
    np = _require("numpy")
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p - q).sum())


def _weighted_rank(u: Any, w: Any) -> Any:
    np = _require("numpy")
    u = np.asarray(u, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(u) & np.isfinite(w) & (w > 0)
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
    if s <= 0:
        return np.full((bins, bins), 1.0 / float(bins * bins), dtype=float)
    return (h / s).astype(float)


def _age_to_p12_idx(age: int) -> int:
    age = max(int(age), 0)
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
    return {"|".join(str(x) for x in k): float(v / tot) for k, v in g.to_dict().items()}


def _tvd_from_dists(p: dict[str, float], q: dict[str, float]) -> float | None:
    if not p or not q:
        return None
    keys = sorted(set(p.keys()) | set(q.keys()))
    pv = [float(p.get(k, 0.0)) for k in keys]
    qv = [float(q.get(k, 0.0)) for k in keys]
    return _tvd(pv, qv)


def _stable_hash_fold(values: list[str], *, n_folds: int, seed: int) -> dict[str, int]:
    import hashlib
    out: dict[str, int] = {}
    for v in values:
        h = hashlib.sha1((str(seed) + "::" + str(v)).encode("utf-8")).hexdigest()
        out[str(v)] = int(h[:8], 16) % int(n_folds)
    return out


def _agg(vals: list[float]) -> dict[str, float] | None:
    np = _require("numpy")
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    return {"mean": float(arr.mean()), "max": float(arr.max()), "n": int(arr.size)}


def main() -> None:
    pd = _require("pandas")
    np = _require("numpy")
    from src.synthpop.pipeline.detroit_v0 import make_run_id
    from src.synthpop.paths import data_root as default_data_root

    ap = argparse.ArgumentParser(prog="expc_copula_baseline")
    ap.add_argument("--exp2_run_dir", required=True)
    ap.add_argument("--condition", default="demo_race_puma")
    ap.add_argument("--data_root", default=str(default_data_root()))
    ap.add_argument("--pums_year", type=int, default=2022)
    ap.add_argument("--pums_period", default="5-Year")
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--pums_person_zip", default=None)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.exp2_run_dir).expanduser().resolve()
    metrics_path = run_dir / str(args.condition) / "metrics_pums_holdout.json"
    if not metrics_path.exists():
        raise SystemExit(f"metrics file not found: {metrics_path}")
    exp2_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    pums_zip = (
        pathlib.Path(str(args.pums_person_zip)).expanduser().resolve()
        if args.pums_person_zip
        else _resolve_pums_person_zip(
            data_root=data_root, pums_year=int(args.pums_year), pums_period=str(args.pums_period), statefp=str(args.statefp)
        )
    )
    if not pums_zip.exists():
        raise SystemExit(f"pums_person_zip not found: {pums_zip}")
    member = _find_first_csv_in_zip(pums_zip)

    usecols = ["PUMA", "PUMA20", "PWGTP", "AGEP", "PINCP"]
    with zipfile.ZipFile(pums_zip) as zf, zf.open(member) as f:
        df = pd.read_csv(f, usecols=lambda c: c in set(usecols), low_memory=False)

    if int(args.pums_year) >= 2022:
        if "PUMA20" not in df.columns:
            raise SystemExit(f"PUMS {int(args.pums_year)} requires PUMA20, but it is missing.")
        df["PUMA"] = df["PUMA20"]
    elif "PUMA" not in df.columns:
        raise SystemExit("Legacy PUMS requires PUMA, but it is missing.")

    puma_num = pd.to_numeric(df["PUMA"], errors="coerce")
    df = df[puma_num.notna() & (puma_num != -9)].copy()
    df["PUMA"] = df["PUMA"].astype(str)
    df["PWGTP"] = pd.to_numeric(df["PWGTP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["AGEP"] = pd.to_numeric(df["AGEP"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=99.0)
    df["PINCP"] = pd.to_numeric(df["PINCP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df[df["PWGTP"] > 0].copy()
    if df.empty:
        raise SystemExit("No valid PUMS rows after cleaning.")

    pumas = sorted(df["PUMA"].unique().tolist())
    fold_of = _stable_hash_fold(pumas, n_folds=int(args.n_folds), seed=int(args.seed))

    income_edges = [0.0, 10_000.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0, 150_000.0, 250_000.0, 10_000_000.0]
    by_fold: dict[str, Any] = {}

    for fold in range(int(args.n_folds)):
        test_pumas = {p for p, f in fold_of.items() if f == fold}
        train_df = df[~df["PUMA"].isin(test_pumas)].copy()
        test_df = df[df["PUMA"].isin(test_pumas)].copy()
        if train_df.empty or test_df.empty:
            continue

        # Baseline predictor: one global train copula/joint for all heldout PUMAs.
        u_tr = _weighted_rank(train_df["AGEP"].to_numpy(dtype=float), train_df["PWGTP"].to_numpy(dtype=float))
        v_tr = _weighted_rank(train_df["PINCP"].to_numpy(dtype=float), train_df["PWGTP"].to_numpy(dtype=float))
        cop_train = _copula_hist2d(u=u_tr, v=v_tr, w=train_df["PWGTP"].to_numpy(dtype=float), bins=int(args.bins))

        train_joint_df = train_df.copy()
        train_joint_df["age_idx"] = train_joint_df["AGEP"].astype(int).map(_age_to_p12_idx).astype(str)
        train_joint_df["PINCP_bin"] = pd.cut(train_joint_df["PINCP"], bins=income_edges, include_lowest=True, right=False).astype(str)
        joint_train = _weighted_joint_dist(train_joint_df, ["age_idx", "PINCP_bin"], "PWGTP")

        fold_metrics: dict[str, Any] = {}
        exp2_fold = exp2_metrics.get("by_fold", {}).get(str(fold), {})
        exp2_by_puma = exp2_fold.get("by_puma", {})

        for p in sorted(test_pumas):
            r_p = test_df[test_df["PUMA"] == str(p)].copy()
            if r_p.empty:
                continue
            u_r = _weighted_rank(r_p["AGEP"].to_numpy(dtype=float), r_p["PWGTP"].to_numpy(dtype=float))
            v_r = _weighted_rank(r_p["PINCP"].to_numpy(dtype=float), r_p["PWGTP"].to_numpy(dtype=float))
            cop_ref = _copula_hist2d(u=u_r, v=v_r, w=r_p["PWGTP"].to_numpy(dtype=float), bins=int(args.bins))
            tvd_cop_base = _tvd(cop_train, cop_ref)

            r_p["age_idx"] = r_p["AGEP"].astype(int).map(_age_to_p12_idx).astype(str)
            r_p["PINCP_bin"] = pd.cut(r_p["PINCP"], bins=income_edges, include_lowest=True, right=False).astype(str)
            joint_ref = _weighted_joint_dist(r_p, ["age_idx", "PINCP_bin"], "PWGTP")
            tvd_joint_base = _tvd_from_dists(joint_train, joint_ref)

            d = exp2_by_puma.get(str(p), {})
            fold_metrics[str(p)] = {
                "baseline_copula_tvd_age_income": tvd_cop_base,
                "baseline_joint_tvd_age_income_bin": tvd_joint_base,
                "diffusion_copula_tvd_age_income": d.get("copula_tvd_age_income"),
                "diffusion_joint_tvd_age_income_bin": d.get("joint_tvd_age_income_bin"),
            }

        # Fold summary
        base_cop = [v["baseline_copula_tvd_age_income"] for v in fold_metrics.values() if v["baseline_copula_tvd_age_income"] is not None]
        base_jnt = [v["baseline_joint_tvd_age_income_bin"] for v in fold_metrics.values() if v["baseline_joint_tvd_age_income_bin"] is not None]
        diff_cop = [v["diffusion_copula_tvd_age_income"] for v in fold_metrics.values() if v["diffusion_copula_tvd_age_income"] is not None]
        diff_jnt = [v["diffusion_joint_tvd_age_income_bin"] for v in fold_metrics.values() if v["diffusion_joint_tvd_age_income_bin"] is not None]

        by_fold[str(fold)] = {
            "n_test_pumas": int(len(fold_metrics)),
            "by_puma": fold_metrics,
            "summary": {
                "baseline_copula_tvd_age_income": _agg(base_cop),
                "baseline_joint_tvd_age_income_bin": _agg(base_jnt),
                "diffusion_copula_tvd_age_income": _agg(diff_cop),
                "diffusion_joint_tvd_age_income_bin": _agg(diff_jnt),
            },
        }

    # Overall summary across folds (mean of fold means)
    def _overall(metric_key: str) -> dict[str, float] | None:
        vals = []
        for f in by_fold.values():
            s = f.get("summary", {}).get(metric_key)
            if isinstance(s, dict) and s.get("mean") is not None:
                vals.append(float(s["mean"]))
        if not vals:
            return None
        arr = np.asarray(vals, dtype=float)
        return {"mean": float(arr.mean()), "max": float(arr.max()), "n_folds": int(arr.size)}

    overall = {
        "baseline_copula_tvd_age_income": _overall("baseline_copula_tvd_age_income"),
        "baseline_joint_tvd_age_income_bin": _overall("baseline_joint_tvd_age_income_bin"),
        "diffusion_copula_tvd_age_income": _overall("diffusion_copula_tvd_age_income"),
        "diffusion_joint_tvd_age_income_bin": _overall("diffusion_joint_tvd_age_income_bin"),
    }
    # Positive means diffusion is better (lower TVD).
    if overall["baseline_copula_tvd_age_income"] and overall["diffusion_copula_tvd_age_income"]:
        overall["copula_tvd_improvement"] = float(
            overall["baseline_copula_tvd_age_income"]["mean"] - overall["diffusion_copula_tvd_age_income"]["mean"]
        )
    if overall["baseline_joint_tvd_age_income_bin"] and overall["diffusion_joint_tvd_age_income_bin"]:
        overall["joint_tvd_improvement"] = float(
            overall["baseline_joint_tvd_age_income_bin"]["mean"] - overall["diffusion_joint_tvd_age_income_bin"]["mean"]
        )

    payload = {
        "created_utc": _utc_now_iso(),
        "inputs": {
            "exp2_run_dir": str(run_dir),
            "condition": str(args.condition),
            "pums_zip": str(pums_zip),
            "member": str(member),
            "pums_year": int(args.pums_year),
            "n_folds": int(args.n_folds),
            "seed": int(args.seed),
            "bins": int(args.bins),
        },
        "by_fold": by_fold,
        "overall": overall,
        "note": "If copula_tvd_improvement > 0, diffusion captures age-income dependence better than train-average baseline.",
    }

    out_path = (
        pathlib.Path(args.out_path).expanduser().resolve()
        if args.out_path
        else (run_dir / f"copula_baseline_{str(args.condition)}.json")
    )
    _write_json(out_path, payload)
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()

