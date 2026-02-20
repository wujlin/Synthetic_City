#!/usr/bin/env python3
from __future__ import annotations

"""
Summarize US-PUMA diffusion ablations across multiple run directories.

Outputs:
- ablation_long.csv/json
- zscore_effect_summary.csv/json
- training_curve_summary.csv/json
- condition_dim_summary.csv/json
"""

import argparse
import datetime as _dt
import json
import math
import pathlib
from typing import Any

import numpy as np
import pandas as pd


def _utc_now_tag() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _metric_mean(node: Any) -> float | None:
    if isinstance(node, dict):
        v = node.get("mean")
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None
    try:
        return float(node)
    except Exception:
        return None


def _infer_condition_dim(cond: str, shape_dims: list[int]) -> int | None:
    if not shape_dims:
        return None
    s = [int(x) for x in shape_dims]
    marginal = int(sum(s))
    pairwise = int(sum(s[i] * s[j] for i in range(len(s)) for j in range(i + 1, len(s))))
    c = str(cond).strip().lower()
    if c == "none":
        return 0
    if c == "marginal":
        return marginal
    if c == "pairwise":
        return pairwise
    if c == "marginal_pairwise":
        return marginal + pairwise
    return None


def _shape_dims_from_summary(rs: dict[str, Any] | None) -> list[int]:
    if not isinstance(rs, dict):
        return []
    shape = rs.get("shape")
    if isinstance(shape, dict):
        out: list[int] = []
        for _, v in shape.items():
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    return []


def _is_zscore(rs: dict[str, Any] | None) -> bool | None:
    if not isinstance(rs, dict):
        return None
    xr = rs.get("x_representation")
    if xr is None:
        return None
    s = str(xr).lower()
    return ("z-score" in s) or ("zscore" in s)


def _run_rows(run_dir: pathlib.Path) -> list[dict[str, Any]]:
    rs = _read_json(run_dir / "run_summary.json")
    if rs is None:
        rs = _read_json(run_dir / "run.metadata.json")
    ab = _read_json(run_dir / "metrics" / "ablation_summary.json")
    bi = _read_json(run_dir / "metrics" / "baselines_internal.json")
    if not isinstance(ab, dict):
        return []

    baseline_ind = None
    baseline_ipf = None
    if isinstance(ab.get("baselines"), dict):
        baseline_ind = _metric_mean(ab["baselines"].get("independence", {}).get("tvd_joint"))
        baseline_ipf = _metric_mean(ab["baselines"].get("ipf_train_seed", {}).get("tvd_joint"))
    if (baseline_ind is None or baseline_ipf is None) and isinstance(bi, dict):
        bb = bi.get("by_baseline", {})
        if baseline_ind is None:
            baseline_ind = _metric_mean(bb.get("independence", {}).get("tvd_joint"))
        if baseline_ipf is None:
            baseline_ipf = _metric_mean(bb.get("ipf_train_seed", {}).get("tvd_joint"))

    cond_dims_from_run = rs.get("condition_dims", {}) if isinstance(rs, dict) else {}
    shape_dims = _shape_dims_from_summary(rs)
    zscore_flag = _is_zscore(rs)
    epochs = int(rs.get("epochs")) if isinstance(rs, dict) and rs.get("epochs") is not None else None
    batch_size = int(rs.get("batch_size")) if isinstance(rs, dict) and rs.get("batch_size") is not None else None
    n_train = int(rs.get("n_non_mi_rows")) if isinstance(rs, dict) and rs.get("n_non_mi_rows") is not None else None
    steps_per_epoch = None
    total_steps = None
    if epochs is not None and batch_size is not None and n_train is not None and batch_size > 0:
        steps_per_epoch = int(math.ceil(float(n_train) / float(batch_size)))
        total_steps = int(epochs * steps_per_epoch)

    k_dim = int(rs.get("K_joint_dim")) if isinstance(rs, dict) and rs.get("K_joint_dim") is not None else None
    hidden_dims = rs.get("hidden_dims") if isinstance(rs, dict) else None
    cond_inj = rs.get("condition_injection") if isinstance(rs, dict) else None
    eval_mode = rs.get("eval_mode") if isinstance(rs, dict) else None
    seed = rs.get("seed") if isinstance(rs, dict) else None

    out: list[dict[str, Any]] = []
    conds = ab.get("conditions", {})
    if not isinstance(conds, dict):
        return out
    for cond_name, obj in conds.items():
        tvd_joint = _metric_mean(obj.get("tvd_joint")) if isinstance(obj, dict) else None
        tvd_joint_raw = _metric_mean(obj.get("tvd_joint_raw")) if isinstance(obj, dict) else None
        cos_joint = _metric_mean(obj.get("cosine_joint")) if isinstance(obj, dict) else None
        cos_joint_raw = _metric_mean(obj.get("cosine_joint_raw")) if isinstance(obj, dict) else None

        cond_dim = None
        if isinstance(cond_dims_from_run, dict) and cond_name in cond_dims_from_run:
            try:
                cond_dim = int(cond_dims_from_run[cond_name])
            except Exception:
                cond_dim = None
        if cond_dim is None:
            cond_dim = _infer_condition_dim(cond_name, shape_dims)

        row = {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "condition": str(cond_name),
            "cond_dim": cond_dim,
            "k_joint_dim": k_dim,
            "shape_dims": ",".join(str(x) for x in shape_dims) if shape_dims else None,
            "tvd_joint": tvd_joint,
            "tvd_joint_raw": tvd_joint_raw,
            "cosine_joint": cos_joint,
            "cosine_joint_raw": cos_joint_raw,
            "baseline_independence_tvd_joint": baseline_ind,
            "baseline_ipf_seed_tvd_joint": baseline_ipf,
            "delta_vs_ipf": (tvd_joint - baseline_ipf) if (tvd_joint is not None and baseline_ipf is not None) else None,
            "delta_vs_independence": (tvd_joint - baseline_ind) if (tvd_joint is not None and baseline_ind is not None) else None,
            "zscore_enabled": zscore_flag,
            "epochs": epochs,
            "batch_size": batch_size,
            "n_train": n_train,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "hidden_dims": ",".join(str(int(x)) for x in hidden_dims) if isinstance(hidden_dims, list) else hidden_dims,
            "condition_injection": cond_inj,
            "eval_mode": eval_mode,
            "seed": seed,
        }
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="summarize_us_puma_ablation")
    ap.add_argument(
        "--run_dirs",
        required=True,
        help="Comma-separated run dirs. Example: outputs/runA,outputs/runB",
    )
    ap.add_argument("--out_dir", default=None, help="Default: outputs/_us_puma_ablation_summary_<utc>")
    args = ap.parse_args()

    run_dirs = []
    for x in str(args.run_dirs).split(","):
        s = x.strip()
        if s:
            run_dirs.append(pathlib.Path(s).expanduser().resolve())
    if not run_dirs:
        raise SystemExit("No run dirs provided.")

    ts = _utc_now_tag()
    out_dir = (
        pathlib.Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else pathlib.Path("outputs").resolve() / f"_us_puma_ablation_summary_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for rd in run_dirs:
        if not rd.exists():
            print(f"[warn] missing run dir: {rd}")
            continue
        rows.extend(_run_rows(rd))
    if not rows:
        raise SystemExit("No valid rows parsed from given run dirs.")

    long_df = pd.DataFrame(rows)
    long_df = long_df.sort_values(["k_joint_dim", "condition", "total_steps", "run_name"], na_position="last").reset_index(drop=True)

    z_df = (
        long_df.groupby(["k_joint_dim", "condition", "zscore_enabled"], dropna=False, as_index=False)
        .agg(
            n_runs=("run_name", "count"),
            tvd_joint_mean=("tvd_joint", "mean"),
            tvd_joint_std=("tvd_joint", "std"),
            delta_vs_ipf_mean=("delta_vs_ipf", "mean"),
            delta_vs_ipf_std=("delta_vs_ipf", "std"),
        )
        .sort_values(["k_joint_dim", "condition", "zscore_enabled"])
        .reset_index(drop=True)
    )

    curve_df = (
        long_df[long_df["total_steps"].notna()]
        .sort_values(["k_joint_dim", "condition", "total_steps"])
        .reset_index(drop=True)
    )

    cond_df = (
        long_df[long_df["cond_dim"].notna()]
        .sort_values(["k_joint_dim", "condition", "cond_dim", "tvd_joint"])
        .reset_index(drop=True)
    )

    long_csv = out_dir / "ablation_long.csv"
    z_csv = out_dir / "zscore_effect_summary.csv"
    curve_csv = out_dir / "training_curve_summary.csv"
    cond_csv = out_dir / "condition_dim_summary.csv"
    long_json = out_dir / "ablation_long.json"
    z_json = out_dir / "zscore_effect_summary.json"
    curve_json = out_dir / "training_curve_summary.json"
    cond_json = out_dir / "condition_dim_summary.json"
    meta_json = out_dir / "run.metadata.json"

    long_df.to_csv(long_csv, index=False)
    z_df.to_csv(z_csv, index=False)
    curve_df.to_csv(curve_csv, index=False)
    cond_df.to_csv(cond_csv, index=False)

    long_json.write_text(long_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    z_json.write_text(z_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    curve_json.write_text(curve_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    cond_json.write_text(cond_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "created_utc": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "n_runs_input": int(len(run_dirs)),
        "n_rows_parsed": int(long_df.shape[0]),
        "run_dirs": [str(x) for x in run_dirs],
        "outputs": {
            "ablation_long_csv": str(long_csv),
            "zscore_effect_csv": str(z_csv),
            "training_curve_csv": str(curve_csv),
            "condition_dim_csv": str(cond_csv),
        },
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] wrote: {out_dir}")


if __name__ == "__main__":
    main()
