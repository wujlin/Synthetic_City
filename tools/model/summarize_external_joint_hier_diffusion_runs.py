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
    metrics = _read_json(path / "metrics" / "hier_diffusion_summary.json")
    joint = metrics["hier_diffusion_joint"]
    ipf = metrics["baselines"]["ipf_train_seed_external"]["tvd_joint"]["mean"]
    return {
        "run_dir": str(path),
        "condition_injection": str(run_summary["condition_injection"]),
        "timesteps": int(run_summary["timesteps"]),
        "epochs": int(run_summary["epochs"]),
        "tvd_joint_raw": float(joint["tvd_joint_raw"]["mean"]),
        "tvd_joint": float(joint["tvd_joint"]["mean"]),
        "tvd_coarse_head": float(joint["tvd_coarse_head"]["mean"]),
        "tvd_coarse_from_fine": float(joint["tvd_coarse_from_fine"]["mean"]),
        "ipf_tvd_joint": float(ipf),
        "selection_by_fold": run_summary.get("selection_by_fold", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="summarize_external_joint_hier_diffusion_runs")
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--label", default="external_joint_hier_diffusion")
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
            "tvd_joint": _summ([r["tvd_joint"] for r in rows]),
            "tvd_joint_raw": _summ([r["tvd_joint_raw"] for r in rows]),
            "tvd_coarse_head": _summ([r["tvd_coarse_head"] for r in rows]),
            "tvd_coarse_from_fine": _summ([r["tvd_coarse_from_fine"] for r in rows]),
            "ipf": _summ([r["ipf_tvd_joint"] for r in rows]),
        },
    }
    ipf_mean = float(payload["metrics"]["ipf"]["mean"])
    joint_mean = float(payload["metrics"]["tvd_joint"]["mean"])
    payload["relative_gain_vs_ipf_pct"] = 100.0 * (ipf_mean - joint_mean) / max(ipf_mean, 1e-12)

    out_json = pathlib.Path(args.out_json).expanduser().resolve() if args.out_json else pathlib.Path("outputs") / f"_{args.label}_summary_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()
