#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.paths import ensure_dir, project_root
from src.synthpop.spatial.work_destination_allocation import assign_work_destination_tract


def _utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_frame(path: pathlib.Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _parse_multiplier_map_arg(spec: str) -> dict[str, float]:
    s = str(spec or "").strip()
    if not s:
        return {}
    out: dict[str, float] = {}
    for token in s.split(","):
        token = str(token).strip()
        if not token:
            continue
        if ":" not in token:
            raise SystemExit(f"invalid multiplier map token: {token}")
        key, value = token.split(":", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"invalid multiplier map token: {token}")
        out[key] = float(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp_phase3b_assign_work_destinations")
    ap.add_argument("--persons_path", required=True)
    ap.add_argument("--tract_od_path", required=True)
    ap.add_argument("--person_id_col", default="person_id")
    ap.add_argument("--home_group_col", default="tract_geoid")
    ap.add_argument("--out_col", default="work_tract_geoid")
    ap.add_argument("--work_eligible_col", default="is_worker")
    ap.add_argument("--work_eligible_values", default="")
    ap.add_argument("--distance_col", default="distance_km")
    ap.add_argument("--distance_beta", type=float, default=0.0)
    ap.add_argument("--earn_col", default="EARN_16p_bin")
    ap.add_argument("--age_col", default="AGEP_bin")
    ap.add_argument("--schl_col", default="SCHL_allpop")
    ap.add_argument("--od_age_segment_weight", type=float, default=0.0)
    ap.add_argument("--od_earn_segment_weight", type=float, default=0.0)
    ap.add_argument("--destination_segment_weight", type=float, default=0.0)
    ap.add_argument("--destination_age_segment_weight", type=float, default=0.0)
    ap.add_argument("--destination_access_col", default="")
    ap.add_argument("--destination_access_weight", type=float, default=0.0)
    ap.add_argument("--od_pair_prior_col", default="")
    ap.add_argument("--od_pair_prior_weight", type=float, default=0.0)
    ap.add_argument("--destination_center_col", default="")
    ap.add_argument("--destination_center_weight", type=float, default=0.0)
    ap.add_argument("--same_tract_weight", type=float, default=0.0)
    ap.add_argument("--same_county_weight", type=float, default=0.0)
    ap.add_argument("--same_home_center_weight", type=float, default=0.0)
    ap.add_argument("--job_family_weight", type=float, default=0.0)
    ap.add_argument("--distance_earn_multiplier_map", default="")
    ap.add_argument("--distance_age_multiplier_map", default="")
    ap.add_argument("--destination_access_earn_multiplier_map", default="")
    ap.add_argument("--destination_access_age_multiplier_map", default="")
    ap.add_argument("--destination_center_earn_multiplier_map", default="")
    ap.add_argument("--destination_center_age_multiplier_map", default="")
    ap.add_argument("--same_tract_earn_multiplier_map", default="")
    ap.add_argument("--same_tract_age_multiplier_map", default="")
    ap.add_argument("--same_county_earn_multiplier_map", default="")
    ap.add_argument("--same_county_age_multiplier_map", default="")
    ap.add_argument("--same_home_center_earn_multiplier_map", default="")
    ap.add_argument("--same_home_center_age_multiplier_map", default="")
    ap.add_argument("--assignment_mode", default="independent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--label", default="phase3b_assign_work_destinations")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    synthetic_dir = ensure_dir(run_dir / "synthetic")
    metrics_dir = ensure_dir(run_dir / "metrics")

    persons_path = pathlib.Path(args.persons_path).expanduser().resolve()
    tract_od_path = pathlib.Path(args.tract_od_path).expanduser().resolve()
    for p in [persons_path, tract_od_path]:
        if not p.exists():
            raise SystemExit(f"input not found: {p}")

    persons = _read_frame(persons_path)
    tract_od = _read_frame(tract_od_path)
    work_values = [x.strip() for x in str(args.work_eligible_values).split(",") if x.strip()]
    distance_earn_multiplier_map = _parse_multiplier_map_arg(args.distance_earn_multiplier_map)
    distance_age_multiplier_map = _parse_multiplier_map_arg(args.distance_age_multiplier_map)
    destination_access_earn_multiplier_map = _parse_multiplier_map_arg(args.destination_access_earn_multiplier_map)
    destination_access_age_multiplier_map = _parse_multiplier_map_arg(args.destination_access_age_multiplier_map)
    destination_center_earn_multiplier_map = _parse_multiplier_map_arg(args.destination_center_earn_multiplier_map)
    destination_center_age_multiplier_map = _parse_multiplier_map_arg(args.destination_center_age_multiplier_map)
    same_tract_earn_multiplier_map = _parse_multiplier_map_arg(args.same_tract_earn_multiplier_map)
    same_tract_age_multiplier_map = _parse_multiplier_map_arg(args.same_tract_age_multiplier_map)
    same_county_earn_multiplier_map = _parse_multiplier_map_arg(args.same_county_earn_multiplier_map)
    same_county_age_multiplier_map = _parse_multiplier_map_arg(args.same_county_age_multiplier_map)
    same_home_center_earn_multiplier_map = _parse_multiplier_map_arg(args.same_home_center_earn_multiplier_map)
    same_home_center_age_multiplier_map = _parse_multiplier_map_arg(args.same_home_center_age_multiplier_map)
    assigned, meta = assign_work_destination_tract(
        persons=persons,
        tract_od=tract_od,
        person_id_col=str(args.person_id_col),
        home_group_col=str(args.home_group_col),
        out_col=str(args.out_col),
        work_eligible_col=(str(args.work_eligible_col) if args.work_eligible_col else None),
        work_eligible_values=work_values,
        distance_col=(str(args.distance_col) if args.distance_col else None),
        distance_beta=float(args.distance_beta),
        earn_col=(str(args.earn_col) if args.earn_col else None),
        age_col=(str(args.age_col) if args.age_col else None),
        schl_col=(str(args.schl_col) if args.schl_col else None),
        od_age_segment_weight=float(args.od_age_segment_weight),
        od_earn_segment_weight=float(args.od_earn_segment_weight),
        destination_segment_weight=float(args.destination_segment_weight),
        destination_age_segment_weight=float(args.destination_age_segment_weight),
        destination_access_col=(str(args.destination_access_col) if args.destination_access_col else None),
        destination_access_weight=float(args.destination_access_weight),
        od_pair_prior_col=(str(args.od_pair_prior_col) if args.od_pair_prior_col else None),
        od_pair_prior_weight=float(args.od_pair_prior_weight),
        destination_center_col=(str(args.destination_center_col) if args.destination_center_col else None),
        destination_center_weight=float(args.destination_center_weight),
        same_tract_weight=float(args.same_tract_weight),
        same_county_weight=float(args.same_county_weight),
        same_home_center_weight=float(args.same_home_center_weight),
        job_family_weight=float(args.job_family_weight),
        distance_earn_multiplier_map=distance_earn_multiplier_map,
        distance_age_multiplier_map=distance_age_multiplier_map,
        destination_access_earn_multiplier_map=destination_access_earn_multiplier_map,
        destination_access_age_multiplier_map=destination_access_age_multiplier_map,
        destination_center_earn_multiplier_map=destination_center_earn_multiplier_map,
        destination_center_age_multiplier_map=destination_center_age_multiplier_map,
        same_tract_earn_multiplier_map=same_tract_earn_multiplier_map,
        same_tract_age_multiplier_map=same_tract_age_multiplier_map,
        same_county_earn_multiplier_map=same_county_earn_multiplier_map,
        same_county_age_multiplier_map=same_county_age_multiplier_map,
        same_home_center_earn_multiplier_map=same_home_center_earn_multiplier_map,
        same_home_center_age_multiplier_map=same_home_center_age_multiplier_map,
        assignment_mode=str(args.assignment_mode),
        seed=int(args.seed),
    )

    out_path = synthetic_dir / "persons_with_worktract.parquet"
    assigned.to_parquet(out_path, index=False)
    payload = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "timestamp_utc": _utc_now_compact(),
        "persons_path": str(persons_path),
        "tract_od_path": str(tract_od_path),
        "output_path": str(out_path),
        "distance_col": (str(args.distance_col) if args.distance_col else None),
        "distance_beta": float(args.distance_beta),
        "earn_col": (str(args.earn_col) if args.earn_col else None),
        "age_col": (str(args.age_col) if args.age_col else None),
        "schl_col": (str(args.schl_col) if args.schl_col else None),
        "od_age_segment_weight": float(args.od_age_segment_weight),
        "od_earn_segment_weight": float(args.od_earn_segment_weight),
        "destination_segment_weight": float(args.destination_segment_weight),
        "destination_age_segment_weight": float(args.destination_age_segment_weight),
        "destination_access_col": (str(args.destination_access_col) if args.destination_access_col else None),
        "destination_access_weight": float(args.destination_access_weight),
        "od_pair_prior_col": (str(args.od_pair_prior_col) if args.od_pair_prior_col else None),
        "od_pair_prior_weight": float(args.od_pair_prior_weight),
        "destination_center_col": (str(args.destination_center_col) if args.destination_center_col else None),
        "destination_center_weight": float(args.destination_center_weight),
        "same_tract_weight": float(args.same_tract_weight),
        "same_county_weight": float(args.same_county_weight),
        "same_home_center_weight": float(args.same_home_center_weight),
        "job_family_weight": float(args.job_family_weight),
        "distance_earn_multiplier_map": distance_earn_multiplier_map,
        "distance_age_multiplier_map": distance_age_multiplier_map,
        "destination_access_earn_multiplier_map": destination_access_earn_multiplier_map,
        "destination_access_age_multiplier_map": destination_access_age_multiplier_map,
        "destination_center_earn_multiplier_map": destination_center_earn_multiplier_map,
        "destination_center_age_multiplier_map": destination_center_age_multiplier_map,
        "same_tract_earn_multiplier_map": same_tract_earn_multiplier_map,
        "same_tract_age_multiplier_map": same_tract_age_multiplier_map,
        "same_county_earn_multiplier_map": same_county_earn_multiplier_map,
        "same_county_age_multiplier_map": same_county_age_multiplier_map,
        "same_home_center_earn_multiplier_map": same_home_center_earn_multiplier_map,
        "same_home_center_age_multiplier_map": same_home_center_age_multiplier_map,
        "assignment_mode": str(args.assignment_mode),
        "meta": meta,
    }
    _write_json(run_dir / "run_summary.json", payload)
    _write_json(metrics_dir / "summary.json", payload)


if __name__ == "__main__":
    main()
