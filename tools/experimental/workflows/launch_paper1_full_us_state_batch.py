#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import json
import pathlib
import subprocess
import sys
import time
from typing import Any

import pandas as pd


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_state(row: dict[str, Any], *, args: argparse.Namespace, batch_log: pathlib.Path) -> dict[str, Any]:
    statefp = str(row["statefp"]).zfill(2)
    out_parquet = pathlib.Path(args.run_dir) / "synthetic" / f"state={statefp}" / "persons.parquet"
    status: dict[str, Any] = {
        "statefp": statefp,
        "state_postal": str(row.get("state_postal", "")),
        "started_utc": _utc_now(),
        "status": "started",
        "returncode": "",
        "seconds": "",
        "output_parquet": str(out_parquet),
        "error": "",
    }
    if out_parquet.exists() and out_parquet.stat().st_size > 0 and not bool(args.overwrite):
        status.update({"status": "skipped_existing", "finished_utc": _utc_now(), "returncode": 0, "seconds": 0.0})
        return status

    required_cols = [
        "targets_long_csv",
        "tract_puma_csv",
        "tract_zip",
        "roads_path",
        "lodes_main_path",
        "lodes_aux_path",
        "wac_path",
    ]
    missing = [c for c in required_cols if not str(row.get(c, "")).strip() or not pathlib.Path(str(row.get(c, ""))).exists()]
    if missing:
        status.update({"status": "blocked_missing_input", "error": ",".join(missing), "finished_utc": _utc_now()})
        return status

    cmd = [
        str(args.python_bin),
        "tools/experimental/workflows/run_paper1_spatial_state_pipeline.py",
        "--repo_root",
        str(args.repo_root),
        "--run_dir",
        str(args.run_dir),
        "--statefp",
        statefp,
        "--state_postal",
        str(row["state_postal"]),
        "--joint_wide_csv",
        str(args.joint_wide_csv),
        "--schema_json",
        str(args.schema_json),
        "--targets_long_csv",
        str(row["targets_long_csv"]),
        "--tract_puma_csv",
        str(row["tract_puma_csv"]),
        "--areas_path",
        str(row["tract_zip"]),
        "--roads_path",
        str(row["roads_path"]),
        "--lodes_main_path",
        str(row["lodes_main_path"]),
        "--lodes_aux_path",
        str(row["lodes_aux_path"]),
        "--wac_path",
        str(row["wac_path"]),
        "--n_jobs",
        str(int(args.n_jobs_per_state)),
        "--seed",
        str(int(args.seed)),
        "--sample_n",
        str(int(args.sample_n_per_state)),
        "--work_destination_profile",
        str(args.work_destination_profile),
    ]
    if bool(args.allow_cross_state_work):
        cmd.extend(
            [
                "--allow_cross_state_work",
                "--cross_state_asset_inventory_csv",
                str(args.asset_inventory_csv),
            ]
        )
        if str(args.cross_state_home_outbound_cache_dir).strip():
            cmd.extend(
                [
                    "--cross_state_home_outbound_cache_dir",
                    str(args.cross_state_home_outbound_cache_dir),
                ]
            )
    t0 = time.perf_counter()
    with batch_log.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] STATE {statefp} COMMAND {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(args.repo_root), stdout=log, stderr=subprocess.STDOUT)
    seconds = float(time.perf_counter() - t0)
    status.update(
        {
            "status": "completed" if int(proc.returncode) == 0 else "failed",
            "returncode": int(proc.returncode),
            "seconds": seconds,
            "finished_utc": _utc_now(),
        }
    )
    if int(proc.returncode) != 0:
        status["error"] = f"nonzero_returncode_{proc.returncode}"
    return status


def _append_status(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(prog="launch_paper1_full_us_state_batch")
    ap.add_argument("--repo_root", type=pathlib.Path, default=pathlib.Path.cwd())
    ap.add_argument("--run_dir", type=pathlib.Path, required=True)
    ap.add_argument("--asset_inventory_csv", type=pathlib.Path, required=True)
    ap.add_argument("--joint_wide_csv", type=pathlib.Path, required=True)
    ap.add_argument("--schema_json", type=pathlib.Path, required=True)
    ap.add_argument("--states", default="ready")
    ap.add_argument("--python_bin", default="/home/jinlin/miniconda3/envs/dpl/bin/python")
    ap.add_argument("--state_workers", type=int, default=1)
    ap.add_argument("--n_jobs_per_state", type=int, default=4)
    ap.add_argument("--sample_n_per_state", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow_cross_state_work", action="store_true")
    ap.add_argument("--cross_state_home_outbound_cache_dir", default="", type=pathlib.Path)
    ap.add_argument(
        "--work_destination_profile",
        default="od_preserving",
        choices=["od_preserving", "detroit_weighted"],
        help=(
            "Paper1 product work-destination profile. Default od_preserving keeps LODES OD as "
            "the binding work-destination constraint. detroit_weighted is the older Detroit/Paper2 utility profile."
        ),
    )
    args = ap.parse_args()

    args.repo_root = args.repo_root.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.asset_inventory_csv = args.asset_inventory_csv.expanduser().resolve()
    args.joint_wide_csv = args.joint_wide_csv.expanduser().resolve()
    args.schema_json = args.schema_json.expanduser().resolve()
    if str(args.cross_state_home_outbound_cache_dir).strip():
        args.cross_state_home_outbound_cache_dir = args.cross_state_home_outbound_cache_dir.expanduser().resolve()
    metrics_dir = _ensure_dir(args.run_dir / "metrics")
    batch_log = args.run_dir / "state_batch.log"
    status_csv = metrics_dir / "state_batch_status.csv"

    inv = pd.read_csv(args.asset_inventory_csv, dtype={"statefp": str})
    inv["statefp"] = inv["statefp"].astype(str).str.zfill(2)
    inv = inv[inv["status"].astype(str) == "ready"].copy()
    if str(args.states).strip().lower() != "ready":
        keep = {s.strip().zfill(2) for s in str(args.states).split(",") if s.strip()}
        inv = inv[inv["statefp"].isin(keep)].copy()
    rows = inv.sort_values("statefp").to_dict("records")
    _write_json(
        args.run_dir / "state_batch_summary.json",
        {
            "created_utc": _utc_now(),
            "status": "running",
            "asset_inventory_csv": str(args.asset_inventory_csv),
            "n_ready_states_selected": int(len(rows)),
            "state_workers": int(args.state_workers),
            "n_jobs_per_state": int(args.n_jobs_per_state),
            "allow_cross_state_work": bool(args.allow_cross_state_work),
            "cross_state_home_outbound_cache_dir": (
                str(args.cross_state_home_outbound_cache_dir)
                if str(args.cross_state_home_outbound_cache_dir).strip()
                else None
            ),
            "work_destination_profile": str(args.work_destination_profile),
        },
    )

    statuses: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, int(args.state_workers))) as ex:
        futs = {ex.submit(_run_state, row, args=args, batch_log=batch_log): row["statefp"] for row in rows}
        for fut in cf.as_completed(futs):
            status = fut.result()
            statuses.append(status)
            _append_status(status_csv, statuses)
            print(json.dumps(status, ensure_ascii=False), flush=True)

    completed = [s for s in statuses if s.get("status") in {"completed", "skipped_existing"}]
    failed = [s for s in statuses if s.get("status") not in {"completed", "skipped_existing"}]
    payload = {
        "created_utc": _utc_now(),
        "status": "completed" if not failed else "completed_with_failures",
        "state_batch_status_csv": str(status_csv),
        "states_selected": int(len(rows)),
        "states_completed_or_existing": int(len(completed)),
        "states_failed_or_blocked": int(len(failed)),
        "failed_statefps": [str(s.get("statefp")) for s in failed],
        "allow_cross_state_work": bool(args.allow_cross_state_work),
        "cross_state_home_outbound_cache_dir": (
            str(args.cross_state_home_outbound_cache_dir)
            if str(args.cross_state_home_outbound_cache_dir).strip()
            else None
        ),
        "work_destination_profile": str(args.work_destination_profile),
    }
    _write_json(args.run_dir / "state_batch_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
