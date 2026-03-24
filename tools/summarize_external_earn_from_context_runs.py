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
    metrics = _read_json(path / "metrics" / "earn_from_context_summary.json")
    model = metrics["earn_from_context"]
    baseline = metrics["baselines"]["train_mean_earn"]
    return {
        "run_dir": str(path),
        "seed": int(run_summary["seed"]),
        "tvd_earn": float(model["tvd_earn"]["mean"]),
        "cosine_earn": float(model["cosine_earn"]["mean"]),
        "mae_earn": float(model["mae_earn"]["mean"]),
        "latent_norm": float(model["latent_norm"]["mean"]),
        "baseline_tvd_earn": float(baseline["tvd_earn"]["mean"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="summarize_external_earn_from_context_runs")
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--label", default="external_earn_from_context")
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
            "tvd_earn": _summ([r["tvd_earn"] for r in rows]),
            "cosine_earn": _summ([r["cosine_earn"] for r in rows]),
            "mae_earn": _summ([r["mae_earn"] for r in rows]),
            "latent_norm": _summ([r["latent_norm"] for r in rows]),
            "baseline_tvd_earn": _summ([r["baseline_tvd_earn"] for r in rows]),
            "relative_gain_vs_baseline_pct": _summ(
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
