#!/usr/bin/env python3
"""
Exp2 Diagnosis: explain worst-case PUMAs from Exp2 holdout metrics.

This script reads an Exp2 run directory (ablation_summary.json) and produces a compact
diagnostic JSON for PI review:
- which PUMAs are worst for each metric (per condition)
- per-PUMA metric table across conditions
- optional: PUMS-derived profile stats for these PUMAs (sample size, weighted means)

Goal: separate "model failure" from "data/structure outliers".
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
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
        candidates.append(raw_dir / f"csv_p{state_postal_lower}i.zip")  # csv_pmi.zip
        candidates.append(raw_dir / f"csv_p{state_postal_lower}.zip")
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit(f"PUMS person zip not found. Tried: {candidates}")


def _weighted_mean(x: Any, w: Any) -> float | None:
    np = _require("numpy")
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not bool(mask.any()):
        return None
    s = float(w[mask].sum())
    if s <= 0:
        return None
    return float((x[mask] * w[mask]).sum() / s)


def main() -> None:
    from src.synthpop.paths import data_root as default_data_root

    ap = argparse.ArgumentParser(prog="exp2_worst_puma_diagnosis")
    ap.add_argument("--exp2_run_dir", required=True, help="Path to Exp2 run dir containing ablation_summary.json")
    ap.add_argument("--conditions", default=None, help="Comma-separated conditions; default: from ablation_summary.json")
    ap.add_argument(
        "--metrics",
        default="tvd_income_bin,tvd_schl,tvd_esr,copula_tvd_age_income,joint_tvd_age_income_bin",
        help="Comma-separated metrics to rank PUMAs by.",
    )
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--pumas", default=None, help="Optional comma-separated PUMAs to diagnose (overrides worst-k).")
    ap.add_argument("--with_pums_profile", action="store_true", help="If set, load PUMS and add per-PUMA stats.")
    ap.add_argument("--data_root", default=str(default_data_root()))
    ap.add_argument("--pums_year", type=int, default=2023)
    ap.add_argument("--pums_period", default="5-Year")
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--n_rows", type=int, default=None, help="Optional cap when reading PUMS for profiling.")
    ap.add_argument(
        "--income_edges",
        default="0,10000,25000,50000,75000,100000,150000,250000,10000000",
        help="Income bin edges (for optional profiling).",
    )
    ap.add_argument("--out_path", default=None, help="Default: <run_dir>/worst_pumas_diagnosis.json")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.exp2_run_dir).expanduser().resolve()
    ablation_path = run_dir / "ablation_summary.json"
    if not ablation_path.exists():
        raise SystemExit(f"ablation_summary.json not found: {ablation_path}")
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    by_condition: dict[str, Any] = ablation.get("by_condition", {})
    if not by_condition:
        raise SystemExit("ablation_summary.json has no by_condition")

    conditions = (
        [c.strip() for c in str(args.conditions).split(",") if c.strip()]
        if args.conditions
        else sorted(by_condition.keys())
    )
    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]

    # Build per-condition per-PUMA metric dict.
    cond_puma: dict[str, dict[str, dict[str, float]]] = {}
    cond_puma_fold: dict[str, dict[str, int]] = {}
    for cond in conditions:
        d = by_condition.get(cond, {})
        bf = d.get("by_fold", {})
        out: dict[str, dict[str, float]] = {}
        out_fold: dict[str, int] = {}
        for fk, f in bf.items():
            by_puma = f.get("by_puma", {}) or {}
            for puma, mm in by_puma.items():
                if not isinstance(mm, dict):
                    continue
                out[str(puma)] = {k: float(mm.get(k)) for k in metrics if mm.get(k) is not None}
                out_fold[str(puma)] = int(fk)
        cond_puma[cond] = out
        cond_puma_fold[cond] = out_fold

    # Select PUMAs to report.
    if args.pumas:
        selected = {p.strip() for p in str(args.pumas).split(",") if p.strip()}
    else:
        selected = set()
        top_k = int(args.top_k)
        for cond in conditions:
            for metric in metrics:
                vals = [(p, mm.get(metric)) for p, mm in cond_puma.get(cond, {}).items() if metric in mm]
                vals = [(p, float(v)) for p, v in vals if v is not None and math.isfinite(float(v))]
                vals.sort(key=lambda t: t[1], reverse=True)
                for p, _ in vals[:top_k]:
                    selected.add(str(p))
    selected = {str(p) for p in selected if str(p)}

    # Worst tables.
    worst_by_condition: dict[str, Any] = {}
    for cond in conditions:
        worst_by_metric: dict[str, list[dict[str, Any]]] = {}
        for metric in metrics:
            vals = [(p, mm.get(metric)) for p, mm in cond_puma.get(cond, {}).items() if metric in mm]
            vals = [(p, float(v)) for p, v in vals if v is not None and math.isfinite(float(v))]
            vals.sort(key=lambda t: t[1], reverse=True)
            worst_by_metric[metric] = [
                {"puma": str(p), "value": float(v), "fold": int(cond_puma_fold.get(cond, {}).get(str(p), -1))}
                for p, v in vals[: int(args.top_k)]
            ]
        worst_by_condition[cond] = worst_by_metric

    # Per-PUMA table.
    puma_table: dict[str, Any] = {}
    for puma in sorted(selected):
        entry = {"metrics_by_condition": {}, "fold_by_condition": {}}
        for cond in conditions:
            mm = cond_puma.get(cond, {}).get(puma, {})
            entry["metrics_by_condition"][cond] = mm
            entry["fold_by_condition"][cond] = int(cond_puma_fold.get(cond, {}).get(puma, -1))
        puma_table[puma] = entry

    # Optional PUMS profiling for selected PUMAs.
    pums_profiles = None
    if bool(args.with_pums_profile) and selected:
        pd = _require("pandas")
        data_root = pathlib.Path(args.data_root).expanduser().resolve()
        pums_zip = _resolve_pums_person_zip(
            data_root=data_root, pums_year=int(args.pums_year), pums_period=str(args.pums_period), statefp=str(args.statefp)
        )
        member = _find_first_csv_in_zip(pums_zip)
        usecols = ["PUMA", "PUMA20", "PWGTP", "AGEP", "PINCP", "SCHL", "ESR", "SEX", "RAC1P"]
        with zipfile.ZipFile(pums_zip) as zf, zf.open(member) as f:
            df = pd.read_csv(f, usecols=lambda c: c in set(usecols), low_memory=False)

        if "PUMA20" in df.columns:
            df["PUMA"] = df["PUMA20"]
        if "PUMA" not in df.columns:
            raise SystemExit("PUMS missing PUMA/PUMA20 for profiling.")

        puma_num = pd.to_numeric(df["PUMA"], errors="coerce")
        df = df[puma_num.notna() & (puma_num != -9)].copy()
        df["PUMA"] = df["PUMA"].astype(str)
        df["PWGTP"] = pd.to_numeric(df["PWGTP"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df["AGEP"] = pd.to_numeric(df["AGEP"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=99.0)
        df["PINCP"] = pd.to_numeric(df["PINCP"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df = df[df["PWGTP"] > 0].copy()
        if args.n_rows is not None and int(args.n_rows) > 0 and int(df.shape[0]) > int(args.n_rows):
            df = df.sample(n=int(args.n_rows), random_state=0).reset_index(drop=True)

        income_edges = [float(x) for x in str(args.income_edges).split(",") if str(x).strip()]
        pums_profiles = {
            "pums_zip": str(pums_zip),
            "member": str(member),
            "income_edges": income_edges,
            "by_puma": {},
        }
        for puma in sorted(selected):
            d = df[df["PUMA"] == str(puma)]
            if d.empty:
                continue
            w = d["PWGTP"].to_numpy(dtype=float)
            age = d["AGEP"].to_numpy(dtype=float)
            inc = d["PINCP"].to_numpy(dtype=float)
            pop_w = float(w.sum())
            pums_profiles["by_puma"][str(puma)] = {
                "n_rows": int(d.shape[0]),
                "total_weight": pop_w,
                "mean_age_w": _weighted_mean(age, w),
                "mean_income_w": _weighted_mean(inc, w),
                "pct_child_w": (None if pop_w <= 0 else float(w[age < 18].sum() / pop_w)),
                "pct_elderly_w": (None if pop_w <= 0 else float(w[age >= 65].sum() / pop_w)),
            }

    out = {
        "created_utc": _utc_now_iso(),
        "argv": sys.argv,
        "run_dir": str(run_dir),
        "inputs": {"ablation_summary": str(ablation_path)},
        "conditions": conditions,
        "metrics": metrics,
        "top_k": int(args.top_k),
        "selected_pumas": sorted(selected),
        "worst_by_condition": worst_by_condition,
        "puma_table": puma_table,
        "pums_profiles": pums_profiles,
    }

    out_path = pathlib.Path(args.out_path).expanduser().resolve() if args.out_path else (run_dir / "worst_pumas_diagnosis.json")
    _write_json(out_path, out)
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()

