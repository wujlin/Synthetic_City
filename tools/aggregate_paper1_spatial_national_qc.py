#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from typing import Any

import pandas as pd


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _weighted_rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _build_sample(run_dir: pathlib.Path, *, sample_n: int, seed: int) -> str | None:
    sample_paths = sorted((run_dir / "samples").glob("state*_sample_*.parquet"))
    sample_paths = [p for p in sample_paths if p.name != "national_sample_100k.parquet"]
    if not sample_paths or sample_n <= 0:
        return None
    parts = [pd.read_parquet(p) for p in sample_paths]
    sample = pd.concat(parts, ignore_index=True)
    if sample.shape[0] > sample_n:
        sample = sample.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    out = run_dir / "samples" / "national_sample_100k.parquet"
    sample.to_parquet(out, index=False)
    return str(out)


def main() -> int:
    ap = argparse.ArgumentParser(prog="aggregate_paper1_spatial_national_qc")
    ap.add_argument("--run_dir", required=True, type=pathlib.Path)
    ap.add_argument("--sample_n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    metrics_dir = run_dir / "metrics"
    state_qc_path = metrics_dir / "state_qc_summary.csv"
    puma_qc_path = metrics_dir / "puma_qc_summary.csv"
    inventory_candidates = [
        metrics_dir / "state_input_inventory.csv",
        metrics_dir / "asset_inventory_ready_joint_intersection.csv",
    ]
    inventory_path = next((p for p in inventory_candidates if p.exists()), inventory_candidates[0])
    failure_path = metrics_dir / "assignment_failure_summary.csv"

    if not state_qc_path.exists():
        raise SystemExit(f"state QC file not found: {state_qc_path}")

    state_qc = pd.read_csv(state_qc_path, dtype={"statefp": str})
    for col in [
        "n_persons",
        "missing_home_coordinate_count",
        "work_eligible_persons",
        "missing_work_coordinate_count_among_workers",
        "output_file_size_bytes",
        "runtime_seconds",
    ]:
        if col in state_qc.columns:
            state_qc[col] = pd.to_numeric(state_qc[col], errors="coerce").fillna(0.0)

    n_persons = float(state_qc.get("n_persons", pd.Series(dtype=float)).sum())
    missing_home = float(state_qc.get("missing_home_coordinate_count", pd.Series(dtype=float)).sum())
    work_eligible = float(state_qc.get("work_eligible_persons", pd.Series(dtype=float)).sum())
    missing_work = float(state_qc.get("missing_work_coordinate_count_among_workers", pd.Series(dtype=float)).sum())

    puma_summary: dict[str, Any] = {}
    if puma_qc_path.exists():
        puma_qc = pd.read_csv(puma_qc_path, dtype={"statefp": str, "puma_uid": str})
        puma_qc["population_abs_error"] = pd.to_numeric(puma_qc.get("population_abs_error"), errors="coerce").fillna(0.0)
        puma_qc["population_error"] = pd.to_numeric(puma_qc.get("population_error"), errors="coerce").fillna(0.0)
        puma_summary = {
            "puma_qc_rows": int(puma_qc.shape[0]),
            "puma_population_abs_error_max": float(puma_qc["population_abs_error"].max()) if not puma_qc.empty else 0.0,
            "puma_population_abs_error_mean": float(puma_qc["population_abs_error"].mean()) if not puma_qc.empty else 0.0,
            "puma_population_error_sum": float(puma_qc["population_error"].sum()) if not puma_qc.empty else 0.0,
        }

    input_inventory: dict[str, Any] = {}
    expected_states: set[str] | None = None
    if inventory_path.exists():
        inv = pd.read_csv(inventory_path, dtype={"statefp": str})
        inv["statefp"] = inv["statefp"].astype(str).str.zfill(2)
        if "ready_for_state_smoke_or_batch" in inv.columns:
            ready = inv["ready_for_state_smoke_or_batch"].astype(str).str.lower().eq("true")
        elif "status" in inv.columns:
            ready = inv["status"].astype(str).eq("ready")
        else:
            ready = pd.Series([True] * int(inv.shape[0]), index=inv.index)
        expected_states = set(inv.loc[ready, "statefp"].tolist())
        input_inventory = {
            "inventory_states": int(inv.shape[0]),
            "inventory_ready_states": int(ready.sum()),
            "inventory_blocked_states": int((~ready).sum()),
            "inventory_path": str(inventory_path),
        }

    failure_summary: dict[str, Any] = {}
    if failure_path.exists():
        fail = pd.read_csv(failure_path, dtype={"statefp": str})
        failure_summary = {
            "failure_summary_rows": int(fail.shape[0]),
            "failure_states": int(fail["statefp"].astype(str).str.zfill(2).nunique()) if "statefp" in fail.columns else None,
        }

    national_sample = _build_sample(run_dir, sample_n=int(args.sample_n), seed=int(args.seed))

    completed_states = set(state_qc["statefp"].astype(str).str.zfill(2).unique())
    expected_state_count = len(expected_states) if expected_states is not None else 50
    missing_expected_states = sorted((expected_states or set()) - completed_states)
    is_complete = int(len(completed_states)) >= int(expected_state_count) and not missing_expected_states

    summary = {
        "created_utc": _utc_now(),
        "run_dir": str(run_dir),
        "status": "state_outputs_complete_pending_manual_review" if is_complete else "partial",
        "expected_state_count": int(expected_state_count),
        "missing_expected_states": missing_expected_states,
        "number_of_states_completed": int(state_qc["statefp"].nunique()),
        "number_of_pumas_completed": int(pd.to_numeric(state_qc.get("n_pumas", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "national_total_synthetic_persons": int(n_persons),
        "home_assignment_rate": _weighted_rate(n_persons - missing_home, n_persons),
        "work_eligible_persons": int(work_eligible),
        "work_coordinate_assignment_rate_among_workers": _weighted_rate(work_eligible - missing_work, work_eligible),
        "missing_home_coordinate_count": int(missing_home),
        "missing_work_coordinate_count_among_workers": int(missing_work),
        "output_file_size_bytes": int(pd.to_numeric(state_qc.get("output_file_size_bytes", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "runtime_seconds_sum_over_states": float(pd.to_numeric(state_qc.get("runtime_seconds", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "national_sample_100k_parquet": national_sample,
        **puma_summary,
        **input_inventory,
        **failure_summary,
    }
    _write_json(metrics_dir / "national_qc_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
