#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import statistics
from typing import Any


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summ(xs: list[float]) -> dict[str, float | int]:
    vals = [float(x) for x in xs]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    std = 0.0 if len(vals) == 1 else float(statistics.pstdev(vals))
    return {"mean": float(statistics.fmean(vals)), "std": std, "n": int(len(vals))}


def _collect_run(path: pathlib.Path) -> dict[str, Any]:
    run_summary = _read_json(path / "run_summary.json")
    results = run_summary["results"]
    one_shot = results.get("references", {})
    one_shot_metrics = one_shot.get("metrics", {})
    one_shot_joint = float(one_shot_metrics.get("tvd_joint", {}).get("mean", float("nan")))
    one_shot_ipf = float(one_shot_metrics.get("ipf", {}).get("mean", float("nan")))
    return {
        "run_dir": str(path),
        "stage1_checkpoint": str(run_summary["stage1_checkpoint"]),
        "stage2_checkpoint": str(run_summary["stage2_checkpoint"]),
        "stage1_coarse_tvd_raw": float(results["stage1_coarse"]["tvd_raw"]["mean"]),
        "stage1_coarse_tvd_ipf": float(results["stage1_coarse"]["tvd_ipf"]["mean"]),
        "pipeline_stage1_raw_tvd_joint_raw": float(results["pipeline_stage1_raw"]["tvd_joint_raw"]["mean"]),
        "pipeline_stage1_raw_tvd_joint": float(results["pipeline_stage1_raw"]["tvd_joint"]["mean"]),
        "pipeline_stage1_coarse_ipf_tvd_joint_raw": float(results["pipeline_stage1_coarse_ipf"]["tvd_joint_raw"]["mean"]),
        "pipeline_stage1_coarse_ipf_tvd_joint": float(results["pipeline_stage1_coarse_ipf"]["tvd_joint"]["mean"]),
        "oracle_stage2_tvd_joint": float(results["oracle_stage2_true_coarse"]["tvd_joint"]["mean"]),
        "uniform_refine_tvd_joint": float(results["uniform_refine_with_stage1_coarse_ipf"]["tvd_joint"]["mean"]),
        "one_shot_tvd_joint": one_shot_joint,
        "ipf_tvd_joint": one_shot_ipf,
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="summarize_external_c2f_full_earn_eval_runs")
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--label", default="external_c2f_full_earn_eval")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    run_dirs = [pathlib.Path(p).expanduser().resolve() for p in args.run_dirs]
    missing = [str(p) for p in run_dirs if not (p / "run_summary.json").exists()]
    if missing:
        raise SystemExit(f"missing run_summary.json under: {missing}")

    rows = [_collect_run(p) for p in run_dirs]
    payload = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "label": str(args.label),
        "n_runs": int(len(rows)),
        "runs": rows,
        "metrics": {
            "stage1_coarse_tvd_raw": _summ([r["stage1_coarse_tvd_raw"] for r in rows]),
            "stage1_coarse_tvd_ipf": _summ([r["stage1_coarse_tvd_ipf"] for r in rows]),
            "pipeline_stage1_raw_tvd_joint_raw": _summ([r["pipeline_stage1_raw_tvd_joint_raw"] for r in rows]),
            "pipeline_stage1_raw_tvd_joint": _summ([r["pipeline_stage1_raw_tvd_joint"] for r in rows]),
            "pipeline_stage1_coarse_ipf_tvd_joint_raw": _summ([r["pipeline_stage1_coarse_ipf_tvd_joint_raw"] for r in rows]),
            "pipeline_stage1_coarse_ipf_tvd_joint": _summ([r["pipeline_stage1_coarse_ipf_tvd_joint"] for r in rows]),
            "oracle_stage2_tvd_joint": _summ([r["oracle_stage2_tvd_joint"] for r in rows]),
            "uniform_refine_tvd_joint": _summ([r["uniform_refine_tvd_joint"] for r in rows]),
            "one_shot_tvd_joint": _summ([r["one_shot_tvd_joint"] for r in rows]),
            "ipf_tvd_joint": _summ([r["ipf_tvd_joint"] for r in rows]),
        },
    }
    one_shot_mean = float(payload["metrics"]["one_shot_tvd_joint"]["mean"])
    c2f_mean = float(payload["metrics"]["pipeline_stage1_coarse_ipf_tvd_joint"]["mean"])
    ipf_mean = float(payload["metrics"]["ipf_tvd_joint"]["mean"])
    payload["relative_gain_vs_one_shot_pct"] = 100.0 * (one_shot_mean - c2f_mean) / max(one_shot_mean, 1e-12)
    payload["relative_gap_vs_ipf_pct"] = 100.0 * (c2f_mean - ipf_mean) / max(ipf_mean, 1e-12)

    out_json = pathlib.Path(args.out_json).expanduser().resolve() if args.out_json else pathlib.Path("outputs") / f"_{args.label}_summary_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()
