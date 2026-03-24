#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import statistics
from collections import defaultdict
from typing import Any


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summ(xs: list[float]) -> dict[str, float | int]:
    vals = [float(x) for x in xs]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    if len(vals) == 1:
        std = 0.0
    else:
        std = statistics.pstdev(vals)
    return {
        "mean": float(statistics.fmean(vals)),
        "std": float(std),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n": int(len(vals)),
    }


def _collect_run(path: pathlib.Path) -> dict[str, Any]:
    run_summary = _read_json(path / "run_summary.json")
    hier = _read_json(path / "metrics" / "hierarchical_summary.json")
    results = hier["hierarchical_joint"]
    baselines = hier["baselines"]
    return {
        "run_dir": str(path),
        "seed": int(run_summary["seed"]),
        "fine_input_mode": str(run_summary.get("fine_input_mode", "z_coarse_prob")),
        "tvd_joint_raw": float(results["tvd_joint_raw"]["mean"]),
        "tvd_joint": float(results["tvd_joint"]["mean"]),
        "tvd_coarse_head": float(results["tvd_coarse_head"]["mean"]),
        "tvd_coarse_from_fine": float(results["tvd_coarse_from_fine"]["mean"]),
        "ipf_tvd_joint": float(baselines["ipf_train_seed_external"]["tvd_joint"]["mean"]),
        "ind_tvd_joint": float(baselines["independence_external"]["tvd_joint"]["mean"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="summarize_external_joint_hier_runs")
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--label", default="external_joint_hier_age_schl")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    run_dirs = [pathlib.Path(p).expanduser().resolve() for p in args.run_dirs]
    missing = [str(p) for p in run_dirs if not (p / "run_summary.json").exists()]
    if missing:
        raise SystemExit(f"missing run_summary.json under: {missing}")

    rows = [_collect_run(p) for p in run_dirs]
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["fine_input_mode"]].append(row)

    grouped: dict[str, Any] = {}
    for mode, items in sorted(by_mode.items()):
        grouped[mode] = {
            "tvd_joint_raw": _summ([x["tvd_joint_raw"] for x in items]),
            "tvd_joint": _summ([x["tvd_joint"] for x in items]),
            "tvd_coarse_head": _summ([x["tvd_coarse_head"] for x in items]),
            "tvd_coarse_from_fine": _summ([x["tvd_coarse_from_fine"] for x in items]),
            "ipf_tvd_joint": _summ([x["ipf_tvd_joint"] for x in items]),
            "relative_gain_vs_ipf_pct": _summ(
                [
                    100.0 * (float(x["ipf_tvd_joint"]) - float(x["tvd_joint"])) / max(float(x["ipf_tvd_joint"]), 1e-12)
                    for x in items
                ]
            ),
            "seeds": [int(x["seed"]) for x in items],
            "run_dirs": [str(x["run_dir"]) for x in items],
        }

    payload = {
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z"),
        "label": str(args.label),
        "n_runs": int(len(rows)),
        "runs": rows,
        "grouped_by_fine_input_mode": grouped,
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
