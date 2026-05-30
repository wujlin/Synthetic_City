#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import pathlib
import statistics
from typing import Any


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summ(xs: list[float]) -> dict[str, float | int]:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    std = 0.0 if len(vals) == 1 else float(statistics.pstdev(vals))
    return {
        "mean": float(statistics.fmean(vals)),
        "std": float(std),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n": int(len(vals)),
    }


def _collect_run(path: pathlib.Path) -> dict[str, Any]:
    run_summary = _read_json(path / "run_summary.json")
    metrics = _read_json(path / "metrics" / "hierarchical_summary.json")
    model = metrics["hierarchical_joint"]
    earn = metrics["earn_aux"]
    baselines = metrics["baselines"]
    return {
        "run_dir": str(path),
        "seed": int(run_summary["seed"]),
        "fine_input_mode": str(run_summary.get("fine_input_mode", "z_only")),
        "cond_dim": int(run_summary["cond_dim"]),
        "tvd_joint": float(model["tvd_joint"]["mean"]),
        "tvd_joint_raw": float(model["tvd_joint_raw"]["mean"]) if "tvd_joint_raw" in model else float("nan"),
        "tvd_coarse_head": float(model["tvd_coarse_head"]["mean"]),
        "tvd_coarse_from_fine": float(model["tvd_coarse_from_fine"]["mean"]),
        "tvd_earn": float(earn["tvd_earn"]["mean"]),
        "cosine_earn": float(earn["cosine_earn"]["mean"]),
        "mae_earn": float(earn["mae_earn"]["mean"]),
        "ipf_tvd_joint": float(baselines["ipf_train_seed_external"]["tvd_joint"]["mean"]),
        "baseline_tvd_earn": float(baselines["train_mean_earn"]["tvd_earn"]["mean"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="summarize_external_joint_hier_full_earn_aux_runs")
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--label", default="external_joint_hier_full_earn_aux")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    run_dirs = [pathlib.Path(p).expanduser().resolve() for p in args.run_dirs]
    missing = [str(p) for p in run_dirs if not (p / "run_summary.json").exists()]
    if missing:
        raise SystemExit(f"missing run_summary.json under: {missing}")

    rows = [_collect_run(p) for p in run_dirs]
    payload = {
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z"),
        "label": str(args.label),
        "n_runs": int(len(rows)),
        "runs": rows,
        "summary": {
            "cond_dim": _summ([r["cond_dim"] for r in rows]),
            "tvd_joint": _summ([r["tvd_joint"] for r in rows]),
            "tvd_joint_raw": _summ([r["tvd_joint_raw"] for r in rows]),
            "tvd_coarse_head": _summ([r["tvd_coarse_head"] for r in rows]),
            "tvd_coarse_from_fine": _summ([r["tvd_coarse_from_fine"] for r in rows]),
            "tvd_earn": _summ([r["tvd_earn"] for r in rows]),
            "cosine_earn": _summ([r["cosine_earn"] for r in rows]),
            "mae_earn": _summ([r["mae_earn"] for r in rows]),
            "ipf_tvd_joint": _summ([r["ipf_tvd_joint"] for r in rows]),
            "baseline_tvd_earn": _summ([r["baseline_tvd_earn"] for r in rows]),
            "relative_gain_vs_ipf_pct": _summ(
                [
                    100.0 * (float(r["ipf_tvd_joint"]) - float(r["tvd_joint"]))
                    / max(float(r["ipf_tvd_joint"]), 1e-12)
                    for r in rows
                ]
            ),
            "relative_gain_vs_earn_baseline_pct": _summ(
                [
                    100.0 * (float(r["baseline_tvd_earn"]) - float(r["tvd_earn"]))
                    / max(float(r["baseline_tvd_earn"]), 1e-12)
                    for r in rows
                ]
            ),
            "seeds": [int(r["seed"]) for r in rows],
            "run_dirs": [str(r["run_dir"]) for r in rows],
        },
    }

    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    else:
        stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = pathlib.Path("outputs") / f"_{args.label}_summary_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()
