#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_step(
    *,
    name: str,
    cmd: list[str],
    log_path: pathlib.Path,
    cwd: pathlib.Path,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] STEP {name}\n")
        log.write("COMMAND " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
    seconds = float(time.perf_counter() - t0)
    return {
        "name": name,
        "command": cmd,
        "returncode": int(proc.returncode),
        "seconds": seconds,
        "finished_utc": _utc_now(),
    }


def _append_failure(path: pathlib.Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fields = [
        "statefp",
        "stage",
        "reason",
        "missing_or_failed_path",
        "created_utc",
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def _stop_after_failed_step(
    *,
    step: dict[str, Any],
    statefp: str,
    stage: str,
    failure_csv: pathlib.Path,
) -> None:
    if int(step.get("returncode", 0)) == 0:
        return
    _append_failure(
        failure_csv,
        {
            "statefp": statefp,
            "stage": stage,
            "reason": f"nonzero_returncode_{step.get('returncode')}",
            "missing_or_failed_path": " ".join(str(x) for x in step.get("command", [])),
            "created_utc": _utc_now(),
        },
    )
    raise SystemExit(f"{stage} failed for state {statefp}; see {failure_csv}")


def _check_inputs(paths: dict[str, pathlib.Path], *, statefp: str, failure_csv: pathlib.Path) -> None:
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        _append_failure(
            failure_csv,
            {
                "statefp": statefp,
                "stage": "input_check",
                "reason": "missing_required_input",
                "missing_or_failed_path": ";".join(missing),
                "created_utc": _utc_now(),
            },
        )
        raise SystemExit(f"missing required inputs for state {statefp}: {missing}")


def _work_destination_profile_args(profile: str) -> list[str]:
    """Return work-destination arguments for a named paper/product profile."""
    profile = str(profile).strip().lower()
    if profile == "od_preserving":
        return [
            "--distance_beta",
            "0.0",
            "--destination_segment_weight",
            "0.0",
            "--destination_access_weight",
            "0.0",
            "--destination_center_weight",
            "0.0",
            "--same_county_weight",
            "0.0",
            "--same_home_center_weight",
            "0.0",
            "--assignment_mode",
            "balanced",
        ]
    if profile == "detroit_weighted":
        return [
            "--distance_beta",
            "0.08",
            "--destination_segment_weight",
            "1.0",
            "--destination_access_col",
            "work_access_jobs_gravity",
            "--destination_access_weight",
            "0.5",
            "--destination_center_col",
            "work_access_job_centers_gravity",
            "--destination_center_weight",
            "0.5",
            "--same_county_weight",
            "0.15",
            "--same_home_center_weight",
            "0.0",
            "--assignment_mode",
            "hierarchical_county",
            "--distance_earn_multiplier_map",
            "CE01:1.15,CE02:1.0,CE03:0.9",
            "--destination_access_earn_multiplier_map",
            "CE01:0.95,CE02:1.0,CE03:1.05",
            "--destination_center_earn_multiplier_map",
            "CE01:0.85,CE02:1.0,CE03:1.15",
            "--same_county_earn_multiplier_map",
            "CE01:1.1,CE02:1.0,CE03:0.9",
            "--same_home_center_earn_multiplier_map",
            "",
        ]
    raise ValueError(f"unknown work_destination_profile: {profile}")


def _finalize_person_locations(
    *,
    input_csv: pathlib.Path,
    output_parquet: pathlib.Path,
    sample_parquet: pathlib.Path,
    statefp: str,
    coordinate_crs: str,
    chunksize: int,
    sample_n: int,
    seed: int,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    sample_parquet.parent.mkdir(parents=True, exist_ok=True)

    keep_cols = [
        "person_id",
        "puma_uid",
        "tract_geoid",
        "type_idx",
        "AGEP_bin",
        "SEX",
        "SCHL_allpop",
        "ESR_allpop",
        "EARN_16p_bin",
        "is_worker",
        "work_tract_geoid",
        "work_destination_mode",
        "work_destination_unassigned_flag",
        "home_x",
        "home_y",
        "work_x",
        "work_y",
        "home_source_stage",
        "work_source_stage",
        "home_assignment_mode",
        "work_assignment_mode",
        "home_fallback_flag",
        "work_fallback_flag",
    ]
    rename = {
        "home_x": "home_lon",
        "home_y": "home_lat",
        "work_x": "work_lon",
        "work_y": "work_lat",
    }

    rng = np.random.default_rng(int(seed))
    writer: pq.ParquetWriter | None = None
    sample_parts: list[pd.DataFrame] = []
    n_rows = 0
    n_home_missing = 0
    n_work_eligible = 0
    n_work_tract_assigned = 0
    n_work_coord_missing = 0
    puma_counts: dict[str, int] = {}

    for chunk in pd.read_csv(input_csv, chunksize=int(chunksize), low_memory=False):
        present = [c for c in keep_cols if c in chunk.columns]
        out = chunk[present].copy()
        out = out.rename(columns=rename)
        out["statefp"] = str(statefp).zfill(2)
        for c in ["puma_uid", "tract_geoid", "work_tract_geoid"]:
            if c in out.columns:
                width = 7 if c == "puma_uid" else 11
                out[c] = (
                    out[c]
                    .astype("string")
                    .str.replace(r"\.0$", "", regex=True)
                    .str.strip()
                    .str.zfill(width)
                )
        if "is_worker" in out.columns:
            out["is_worker"] = out["is_worker"].fillna(False).astype(bool)

        n = int(out.shape[0])
        n_rows += n
        if "home_lon" in out.columns and "home_lat" in out.columns:
            n_home_missing += int((out["home_lon"].isna() | out["home_lat"].isna()).sum())
        if "is_worker" in out.columns:
            worker = out["is_worker"].astype(bool)
            n_work_eligible += int(worker.sum())
            if "work_tract_geoid" in out.columns:
                n_work_tract_assigned += int(out.loc[worker, "work_tract_geoid"].notna().sum())
            if "work_lon" in out.columns and "work_lat" in out.columns:
                n_work_coord_missing += int((out.loc[worker, "work_lon"].isna() | out.loc[worker, "work_lat"].isna()).sum())
        if "puma_uid" in out.columns:
            vc = out["puma_uid"].astype(str).value_counts()
            for k, v in vc.items():
                puma_counts[str(k)] = int(puma_counts.get(str(k), 0) + int(v))

        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_parquet, table.schema, compression="zstd")
        writer.write_table(table)

        if sample_n > 0:
            take = min(int(sample_n), n)
            if take > 0:
                sample_parts.append(out.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))

    if writer is not None:
        writer.close()

    if sample_parts and sample_n > 0:
        sample = pd.concat(sample_parts, ignore_index=True)
        if sample.shape[0] > int(sample_n):
            sample = sample.sample(n=int(sample_n), random_state=int(seed)).reset_index(drop=True)
        sample.to_parquet(sample_parquet, index=False)

    output_bytes = output_parquet.stat().st_size if output_parquet.exists() else 0
    return {
        "statefp": str(statefp).zfill(2),
        "n_persons": int(n_rows),
        "n_pumas": int(len(puma_counts)),
        "home_assignment_rate": float(1.0 - n_home_missing / max(n_rows, 1)),
        "missing_home_coordinate_count": int(n_home_missing),
        "work_eligible_persons": int(n_work_eligible),
        "work_tract_assignment_rate_among_workers": float(n_work_tract_assigned / max(n_work_eligible, 1)),
        "missing_work_coordinate_count_among_workers": int(n_work_coord_missing),
        "output_parquet": str(output_parquet),
        "sample_parquet": str(sample_parquet) if sample_parquet.exists() else None,
        "output_file_size_bytes": int(output_bytes),
        "coordinate_crs": str(coordinate_crs),
    }


def _normalize_id_series(s: pd.Series, width: int) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(width)
    )


def _write_puma_qc(
    *,
    joint_wide_csv: pathlib.Path,
    statefp: str,
    puma_counts: dict[str, int],
    output_csv: pathlib.Path,
) -> dict[str, Any]:
    joint = pd.read_csv(joint_wide_csv, usecols=lambda c: c in {"statefp", "puma_uid", "puma5", "total_person_weight"}, low_memory=False)
    if "statefp" not in joint.columns:
        joint["statefp"] = _normalize_id_series(joint["puma_uid"], 7).str.slice(0, 2)
    joint["statefp"] = _normalize_id_series(joint["statefp"], 2)
    if "puma_uid" not in joint.columns:
        joint["puma_uid"] = joint["statefp"] + _normalize_id_series(joint["puma5"], 5)
    joint["puma_uid"] = _normalize_id_series(joint["puma_uid"], 7)
    target = joint.loc[joint["statefp"] == statefp, ["statefp", "puma_uid", "total_person_weight"]].copy()
    target["target_persons"] = pd.to_numeric(target["total_person_weight"], errors="coerce").fillna(0.0)
    target = target.drop(columns=["total_person_weight"])
    observed = pd.DataFrame(
        {
            "puma_uid": list(puma_counts.keys()),
            "synthetic_persons": list(puma_counts.values()),
        }
    )
    if observed.empty:
        observed = pd.DataFrame(columns=["puma_uid", "synthetic_persons"])
    observed["puma_uid"] = _normalize_id_series(observed["puma_uid"], 7)
    observed["synthetic_persons"] = pd.to_numeric(observed["synthetic_persons"], errors="coerce").fillna(0).astype(int)
    out = target.merge(observed, on="puma_uid", how="left")
    out["synthetic_persons"] = out["synthetic_persons"].fillna(0).astype(int)
    out["population_error"] = out["synthetic_persons"] - out["target_persons"]
    out["population_abs_error"] = out["population_error"].abs()
    out["population_relative_error"] = out["population_error"] / out["target_persons"].replace(0, np.nan)

    if output_csv.exists():
        old = pd.read_csv(output_csv, dtype={"statefp": str, "puma_uid": str})
        old = old[old["statefp"].astype(str).str.zfill(2) != statefp].copy()
        pd.concat([old, out], ignore_index=True).sort_values(["statefp", "puma_uid"]).to_csv(output_csv, index=False)
    else:
        out.sort_values(["statefp", "puma_uid"]).to_csv(output_csv, index=False)

    return {
        "puma_target_count": int(out.shape[0]),
        "puma_population_abs_error_max": float(out["population_abs_error"].max()) if not out.empty else 0.0,
        "puma_population_abs_error_mean": float(out["population_abs_error"].mean()) if not out.empty else 0.0,
        "puma_population_error_sum": float(out["population_error"].sum()) if not out.empty else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="run_paper1_spatial_state_pipeline")
    ap.add_argument("--repo_root", type=pathlib.Path, default=pathlib.Path.cwd())
    ap.add_argument("--run_dir", required=True, type=pathlib.Path)
    ap.add_argument("--statefp", required=True)
    ap.add_argument("--state_postal", required=True)
    ap.add_argument("--joint_wide_csv", required=True, type=pathlib.Path)
    ap.add_argument("--schema_json", required=True, type=pathlib.Path)
    ap.add_argument("--targets_long_csv", required=True, type=pathlib.Path)
    ap.add_argument("--tract_puma_csv", required=True, type=pathlib.Path)
    ap.add_argument("--areas_path", required=True, type=pathlib.Path)
    ap.add_argument("--roads_path", required=True, type=pathlib.Path)
    ap.add_argument("--lodes_main_path", required=True, type=pathlib.Path)
    ap.add_argument("--lodes_aux_path", required=True, type=pathlib.Path)
    ap.add_argument("--wac_path", default="", type=pathlib.Path)
    ap.add_argument("--cross_state_asset_inventory_csv", default="", type=pathlib.Path)
    ap.add_argument("--cross_state_home_outbound_cache_dir", default="", type=pathlib.Path)
    ap.add_argument("--allow_cross_state_work", action="store_true")
    ap.add_argument(
        "--work_destination_profile",
        default="od_preserving",
        choices=["od_preserving", "detroit_weighted"],
        help=(
            "Paper1 product profile for assigning work destination tracts. "
            "od_preserving keeps LODES OD as the binding destination constraint; "
            "detroit_weighted preserves the older Detroit/Paper2 utility profile."
        ),
    )
    ap.add_argument("--n_jobs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coordinate_crs", default="EPSG:4269")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--sample_n", type=int, default=100_000)
    args = ap.parse_args()

    repo = args.repo_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    statefp = str(args.statefp).zfill(2)
    state_dir = _ensure_dir(run_dir / "stages" / f"state={statefp}")
    metrics_dir = _ensure_dir(run_dir / "metrics")
    final_dir = _ensure_dir(run_dir / "synthetic" / f"state={statefp}")
    sample_dir = _ensure_dir(run_dir / "samples")
    log_path = run_dir / "run.log"
    failure_csv = metrics_dir / "assignment_failure_summary.csv"

    required = {
        "joint_wide_csv": args.joint_wide_csv.expanduser().resolve(),
        "schema_json": args.schema_json.expanduser().resolve(),
        "targets_long_csv": args.targets_long_csv.expanduser().resolve(),
        "tract_puma_csv": args.tract_puma_csv.expanduser().resolve(),
        "areas_path": args.areas_path.expanduser().resolve(),
        "roads_path": args.roads_path.expanduser().resolve(),
        "lodes_main_path": args.lodes_main_path.expanduser().resolve(),
        "lodes_aux_path": args.lodes_aux_path.expanduser().resolve(),
    }
    if str(args.wac_path):
        required["wac_path"] = args.wac_path.expanduser().resolve()
    if bool(args.allow_cross_state_work):
        required["cross_state_asset_inventory_csv"] = args.cross_state_asset_inventory_csv.expanduser().resolve()
        if str(args.cross_state_home_outbound_cache_dir):
            required["cross_state_home_outbound_cache_dir"] = args.cross_state_home_outbound_cache_dir.expanduser().resolve()
    _check_inputs(required, statefp=statefp, failure_csv=failure_csv)

    steps: list[dict[str, Any]] = []
    py = "/home/jinlin/miniconda3/envs/dpl/bin/python"

    phase2_dir = state_dir / "phase2_alloc"
    steps.append(
        _run_step(
            name=f"state{statefp}_phase2_alloc",
            cwd=repo,
            log_path=log_path,
            cmd=[
                py,
                "tools/spatial/exp_phase2_puma_to_small_area.py",
                "--joint_wide_csv",
                str(required["joint_wide_csv"]),
                "--schema_json",
                str(required["schema_json"]),
                "--targets_long_csv",
                str(required["targets_long_csv"]),
                "--tract_puma_csv",
                str(required["tract_puma_csv"]),
                "--statefp",
                statefp,
                "--run_dir",
                str(phase2_dir),
                "--label",
                f"paper1_fullus_state{statefp}_phase2",
                "--hard_variables",
                "AGEP_SEX_cross",
                "--prior_variables",
                "SCHL_25p,ESR_16p,PINCP_16p_bin",
                "--strict_prior_variables",
                "--max_iters",
                "200",
                "--tol",
                "1e-6",
            ],
        )
    )
    _stop_after_failed_step(step=steps[-1], statefp=statefp, stage="phase2_alloc", failure_csv=failure_csv)

    expand_dir = state_dir / "phase2_expand"
    steps.append(
        _run_step(
            name=f"state{statefp}_phase2_expand",
            cwd=repo,
            log_path=log_path,
            cmd=[
                py,
                "tools/spatial/exp_phase2_expand_to_persons.py",
                "--allocation_long_csv",
                str(phase2_dir / "synthetic" / "type_assignment_long.csv"),
                "--run_dir",
                str(expand_dir),
                "--person_id_prefix",
                f"synp{statefp}",
                "--skip_persons_csv",
            ],
        )
    )
    _stop_after_failed_step(step=steps[-1], statefp=statefp, stage="phase2_expand", failure_csv=failure_csv)

    lodes_dir = state_dir / "lodes_tract_od"
    lodes_cmd = [
        py,
        "tools/spatial/prepare_detroit_lodes_tract_od.py",
        "--areas_path",
        str(required["areas_path"]),
        "--areas_group_col",
        "GEOID",
        "--study_persons_path",
        str(expand_dir / "synthetic" / "persons.parquet"),
        "--study_persons_group_col",
        "tract_geoid",
        "--state_postal",
        str(args.state_postal).lower(),
        "--year",
        "2020",
        "--main_path",
        str(required["lodes_main_path"]),
        "--aux_path",
        str(required["lodes_aux_path"]),
        "--run_dir",
        str(lodes_dir),
        "--label",
        f"paper1_fullus_state{statefp}_lodes",
    ]
    if "wac_path" in required:
        lodes_cmd.extend(["--wac_path", str(required["wac_path"])])
    if bool(args.allow_cross_state_work):
        lodes_cmd.extend(
            [
                "--allow_cross_state_work",
                "--cross_state_asset_inventory_csv",
                str(required["cross_state_asset_inventory_csv"]),
            ]
        )
        if "cross_state_home_outbound_cache_dir" in required:
            lodes_cmd.extend(["--cross_state_home_outbound_cache_dir", str(required["cross_state_home_outbound_cache_dir"])])
    steps.append(_run_step(name=f"state{statefp}_lodes_tract_od", cwd=repo, log_path=log_path, cmd=lodes_cmd))
    _stop_after_failed_step(step=steps[-1], statefp=statefp, stage="lodes_tract_od", failure_csv=failure_csv)

    work_dir = state_dir / "work_dest"
    work_dest_cmd = [
        py,
        "tools/spatial/exp_phase3b_assign_work_destinations.py",
        "--persons_path",
        str(expand_dir / "synthetic" / "persons.parquet"),
        "--tract_od_path",
        str(lodes_dir / "tract_od.csv"),
        "--run_dir",
        str(work_dir),
        "--label",
        f"paper1_fullus_state{statefp}_workdest_{args.work_destination_profile}",
        *_work_destination_profile_args(str(args.work_destination_profile)),
    ]
    steps.append(
        _run_step(
            name=f"state{statefp}_work_dest",
            cwd=repo,
            log_path=log_path,
            cmd=work_dest_cmd,
        )
    )
    _stop_after_failed_step(step=steps[-1], statefp=statefp, stage="work_dest", failure_csv=failure_csv)

    road_dir = state_dir / "road_locations"
    steps.append(
        _run_step(
            name=f"state{statefp}_road_locations",
            cwd=repo,
            log_path=log_path,
            cmd=[
                py,
                "tools/spatial/exp_phase3_road_locations.py",
                "--persons_path",
                str(work_dir / "synthetic" / "persons_with_worktract.parquet"),
                "--areas_path",
                str(required["areas_path"]),
                "--roads_path",
                str(required["roads_path"]),
                *(
                    [
                        "--cross_state_asset_inventory_csv",
                        str(required["cross_state_asset_inventory_csv"]),
                    ]
                    if bool(args.allow_cross_state_work)
                    else []
                ),
                "--areas_group_col",
                "GEOID",
                "--work_group_col",
                "work_tract_geoid",
                "--work_eligible_col",
                "is_worker",
                "--work_mtfcc_values",
                "S1100,S1200",
                "--work_gap_exception_mtfcc_values",
                "S1400",
                "--home_mode",
                "conservative",
                "--allow_home_fallback",
                "--allow_work_fallback",
                "--legalization_fraction",
                "1e-6",
                "--home_interpolation_density",
                "0.0005",
                "--work_interpolation_density",
                "0.0002",
                "--n_jobs",
                str(int(args.n_jobs)),
                "--parallel_chunksize",
                "32",
                "--low_memory",
                "--run_dir",
                str(road_dir),
                "--label",
                f"paper1_fullus_state{statefp}_roadloc",
            ],
        )
    )
    _stop_after_failed_step(step=steps[-1], statefp=statefp, stage="road_locations", failure_csv=failure_csv)

    t0 = time.perf_counter()
    state_qc = _finalize_person_locations(
        input_csv=road_dir / "synthetic" / "person_locations.csv",
        output_parquet=final_dir / "persons.parquet",
        sample_parquet=sample_dir / f"state{statefp}_sample_100k.parquet",
        statefp=statefp,
        coordinate_crs=str(args.coordinate_crs),
        chunksize=int(args.chunksize),
        sample_n=int(args.sample_n),
        seed=int(args.seed),
    )
    finalize_step = {
        "name": f"state{statefp}_finalize_parquet",
        "returncode": 0,
        "seconds": float(time.perf_counter() - t0),
        "finished_utc": _utc_now(),
    }
    steps.append(finalize_step)

    puma_qc = _write_puma_qc(
        joint_wide_csv=required["joint_wide_csv"],
        statefp=statefp,
        puma_counts={str(k): int(v) for k, v in pd.read_parquet(final_dir / "persons.parquet", columns=["puma_uid"])["puma_uid"].astype(str).value_counts().items()},
        output_csv=metrics_dir / "puma_qc_summary.csv",
    )

    state_qc["runtime_seconds"] = float(sum(float(s.get("seconds", 0.0)) for s in steps))
    state_qc["work_destination_profile"] = str(args.work_destination_profile)
    state_qc["allow_cross_state_work"] = bool(args.allow_cross_state_work)
    state_qc.update(puma_qc)
    state_qc["steps"] = steps
    _write_json(metrics_dir / f"state{statefp}_qc_summary.json", state_qc)

    state_qc_csv = metrics_dir / "state_qc_summary.csv"
    row = {k: v for k, v in state_qc.items() if k != "steps"}
    if state_qc_csv.exists():
        old = pd.read_csv(state_qc_csv, dtype={"statefp": str})
        old = old[old["statefp"].astype(str).str.zfill(2) != statefp].copy()
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).sort_values("statefp").to_csv(state_qc_csv, index=False)
    else:
        pd.DataFrame([row]).to_csv(state_qc_csv, index=False)

    print(json.dumps(state_qc, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
