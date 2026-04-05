#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Any

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_geodata(path: pathlib.Path) -> Any:
    try:
        import geopandas as gpd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("exp_phase3_roadloc_validate requires geopandas.") from e

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return gpd.read_parquet(path)
    if suffix == ".zip":
        return gpd.read_file(f"zip://{path}")
    return gpd.read_file(path)


def _candidate_geometry_check(
    *,
    candidates: pd.DataFrame,
    areas_path: pathlib.Path,
    group_col: str,
    areas_group_col: str,
    sample_n: int,
    seed: int,
) -> dict[str, Any]:
    if candidates.empty:
        return {"n_checked": 0, "within_group_polygon_rate": None, "bad_examples": []}

    try:
        import geopandas as gpd
    except Exception:
        return {"n_checked": 0, "within_group_polygon_rate": None, "bad_examples": [], "skipped": "missing_geopandas"}

    areas = _read_geodata(areas_path)
    if str(areas_group_col) not in areas.columns:
        raise SystemExit(f"areas missing column: {areas_group_col}")
    areas = areas[[str(areas_group_col), "geometry"]].copy()
    areas = areas.rename(columns={str(areas_group_col): str(group_col)})
    areas[str(group_col)] = areas[str(group_col)].astype(str)

    sample = candidates.copy()
    if int(sample_n) > 0 and int(sample.shape[0]) > int(sample_n):
        sample = sample.sample(n=int(sample_n), random_state=int(seed)).copy()
    sample[str(group_col)] = sample[str(group_col)].astype(str)
    sample = sample.merge(areas, on=str(group_col), how="left", suffixes=("", "_group"))
    sample["point_geom"] = gpd.points_from_xy(sample["x"], sample["y"], crs=areas.crs)

    ok: list[bool] = []
    bad_examples: list[dict[str, Any]] = []
    for row in sample[["candidate_id", str(group_col), "source_stage", "point_geom", "geometry"]].itertuples(index=False):
        area_geom = getattr(row, "geometry")
        point_geom = getattr(row, "point_geom")
        good = bool(area_geom is not None and not getattr(area_geom, "is_empty", True) and area_geom.covers(point_geom))
        ok.append(good)
        if not good and len(bad_examples) < 5:
            bad_examples.append(
                {
                    "candidate_id": str(getattr(row, "candidate_id")),
                    str(group_col): str(getattr(row, str(group_col))),
                    "source_stage": str(getattr(row, "source_stage")),
                    "x": float(point_geom.x),
                    "y": float(point_geom.y),
                }
            )

    return {
        "n_checked": int(len(ok)),
        "within_group_polygon_rate": float(sum(ok) / len(ok)) if ok else None,
        "bad_examples": bad_examples,
    }


def _integrity_check(
    *,
    persons: pd.DataFrame,
    candidates: pd.DataFrame,
    person_candidate_col: str,
    person_stage_col: str,
    person_fallback_col: str,
    person_group_col: str,
    candidate_group_col: str,
) -> dict[str, Any]:
    assigned = persons[persons[str(person_candidate_col)].notna()].copy()
    if assigned.empty:
        return {
            "n_assigned": 0,
            "candidate_ref_hit_rate": None,
            "group_match_rate": None,
            "stage_match_rate": None,
            "fallback_share_among_assigned": None,
        }

    merged = assigned.merge(
        candidates[["candidate_id", str(candidate_group_col), "source_stage"]].rename(
            columns={str(candidate_group_col): "_cand_group", "source_stage": "_cand_stage"}
        ),
        left_on=str(person_candidate_col),
        right_on="candidate_id",
        how="left",
    )
    hit = merged["candidate_id"].notna()
    group_match = merged[str(person_group_col)].astype(str) == merged["_cand_group"].astype(str)
    stage_match = merged[str(person_stage_col)].astype(str) == merged["_cand_stage"].astype(str)
    return {
        "n_assigned": int(assigned.shape[0]),
        "candidate_ref_hit_rate": float(hit.mean()),
        "group_match_rate": float(group_match[hit].mean()) if bool(hit.any()) else None,
        "stage_match_rate": float(stage_match[hit].mean()) if bool(hit.any()) else None,
        "fallback_share_among_assigned": float(assigned[str(person_fallback_col)].astype(bool).mean()),
    }


def _non_primary_stage_count(stage_counts: dict[str, Any]) -> int:
    total = 0
    for stage, count in dict(stage_counts).items():
        if str(stage) != "primary":
            total += int(count)
    return int(total)


def _failure_exposure_stage_count(
    stage_counts: dict[str, Any],
    *,
    allowed_non_primary_stages: list[str] | None = None,
) -> int:
    allowed = {str(v) for v in (allowed_non_primary_stages or []) if str(v).strip()}
    total = 0
    for stage, count in dict(stage_counts).items():
        st = str(stage)
        if st == "primary":
            continue
        if st in allowed:
            continue
        total += int(count)
    return int(total)


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp_phase3_roadloc_validate")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--areas_path", default="")
    ap.add_argument("--areas_group_col", default="")
    ap.add_argument("--group_col", default="")
    ap.add_argument("--candidate_sample_n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")
    run_summary_path = run_dir / "run_summary.json"
    summary_path = run_dir / "metrics" / "summary.json"
    if not run_summary_path.exists() or not summary_path.exists():
        raise SystemExit(f"missing run summaries under: {run_dir}")

    run_summary = _read_json(run_summary_path)
    summary = _read_json(summary_path)
    group_col = str(args.group_col).strip() or str(summary.get("group_col") or run_summary.get("candidate_meta", {}).get("group_col") or "tract_geoid")
    work_group_col = str(summary.get("work_group_col") or run_summary.get("assignment_meta", {}).get("work_group_col") or group_col)
    areas_group_col = str(args.areas_group_col).strip() or str(summary.get("areas_group_col") or group_col)
    areas_path = pathlib.Path(args.areas_path).expanduser().resolve() if args.areas_path else pathlib.Path(summary["input_paths"]["areas_path"])
    out_json = pathlib.Path(args.out_json).expanduser().resolve() if args.out_json else (run_dir / "metrics" / "roadloc_validation.json")

    artifacts = summary.get("artifacts", {})
    home_csv = pathlib.Path(artifacts["home_candidates_csv"]).expanduser().resolve()
    work_csv = pathlib.Path(artifacts["work_candidates_csv"]).expanduser().resolve()
    persons_csv = pathlib.Path(artifacts["person_locations_csv"]).expanduser().resolve()
    for p in [home_csv, work_csv, persons_csv, areas_path]:
        if not p.exists():
            raise SystemExit(f"missing artifact: {p}")

    home_candidates = pd.read_csv(
        home_csv,
        usecols=["candidate_id", group_col, "source_stage", "x", "y"],
        dtype={"candidate_id": str, group_col: str, "source_stage": str},
        low_memory=False,
    )
    work_candidates = pd.read_csv(
        work_csv,
        usecols=["candidate_id", group_col, "source_stage", "x", "y"],
        dtype={"candidate_id": str, group_col: str, "source_stage": str},
        low_memory=False,
    )
    person_usecols = list(
        dict.fromkeys(
            [
                "person_id",
                group_col,
                work_group_col,
                "home_candidate_id",
                "home_source_stage",
                "home_fallback_flag",
                "work_candidate_id",
                "work_source_stage",
                "work_fallback_flag",
                "work_assignment_mode",
            ]
        )
    )
    person_dtype = {
        "person_id": str,
        group_col: str,
        work_group_col: str,
        "home_candidate_id": str,
        "home_source_stage": str,
        "work_candidate_id": str,
        "work_source_stage": str,
        "work_assignment_mode": str,
    }
    persons = pd.read_csv(persons_csv, usecols=person_usecols, dtype=person_dtype, low_memory=False)
    for df in [home_candidates, work_candidates, persons]:
        df[group_col] = df[group_col].astype(str)
    if work_group_col in persons.columns:
        persons[work_group_col] = persons[work_group_col].astype(str)

    home_integrity = _integrity_check(
        persons=persons,
        candidates=home_candidates,
        person_candidate_col="home_candidate_id",
        person_stage_col="home_source_stage",
        person_fallback_col="home_fallback_flag",
        person_group_col=group_col,
        candidate_group_col=group_col,
    )
    work_integrity = _integrity_check(
        persons=persons[persons["work_assignment_mode"].astype(str) != "ineligible"].copy().rename(columns={work_group_col: "_work_group"}),
        candidates=work_candidates,
        person_candidate_col="work_candidate_id",
        person_stage_col="work_source_stage",
        person_fallback_col="work_fallback_flag",
        person_group_col="_work_group",
        candidate_group_col=group_col,
    )
    home_geom = _candidate_geometry_check(
        candidates=home_candidates,
        areas_path=areas_path,
        group_col=group_col,
        areas_group_col=areas_group_col,
        sample_n=int(args.candidate_sample_n),
        seed=int(args.seed),
    )
    work_geom = _candidate_geometry_check(
        candidates=work_candidates,
        areas_path=areas_path,
        group_col=group_col,
        areas_group_col=areas_group_col,
        sample_n=int(args.candidate_sample_n),
        seed=int(args.seed),
    )

    assignment_meta = run_summary.get("assignment_meta", {})
    candidate_meta = run_summary.get("candidate_meta", {})
    home_stage_counts = dict(candidate_meta.get("home_stage_counts", {}))
    work_stage_counts = dict(candidate_meta.get("work_stage_counts", {}))
    home_allowed_non_primary_stages = [str(v) for v in candidate_meta.get("home_allowed_non_primary_stages", [])]
    work_allowed_non_primary_stages = [str(v) for v in candidate_meta.get("work_allowed_non_primary_stages", [])]
    home_unassigned = int(assignment_meta.get("home_unassigned", 0))
    work_unassigned = int(assignment_meta.get("work_unassigned", 0))
    home_fallback_assignments = int(assignment_meta.get("home_fallback_assignments", 0))
    work_fallback_assignments = int(assignment_meta.get("work_fallback_assignments", 0))
    home_non_primary_groups = _non_primary_stage_count(home_stage_counts)
    work_non_primary_groups = _non_primary_stage_count(work_stage_counts)
    home_failure_exposure_groups = _failure_exposure_stage_count(
        home_stage_counts,
        allowed_non_primary_stages=home_allowed_non_primary_stages,
    )
    work_failure_exposure_groups = _failure_exposure_stage_count(
        work_stage_counts,
        allowed_non_primary_stages=work_allowed_non_primary_stages,
    )
    result = {
        "created_utc": _utc_now_iso(),
        "run_dir": str(run_dir),
        "group_col": group_col,
        "work_group_col": work_group_col,
        "areas_group_col": areas_group_col,
        "coverage_failure_definition": {
            "hard_failure": "存在 unassigned persons，或 assigned person 的 candidate id 无法在候选表中解析",
            "strict_failure": "任何 fallback assignment、no-candidate，或未声明的 non-primary candidate stage 都视为 failure exposure；显式声明的 exception stage 不算黑箱 fallback。",
        },
        "coverage": {
            "home_unassigned": home_unassigned,
            "work_unassigned": work_unassigned,
            "home_fallback_assignments": home_fallback_assignments,
            "work_fallback_assignments": work_fallback_assignments,
            "home_stage_counts": home_stage_counts,
            "work_stage_counts": work_stage_counts,
            "home_allowed_non_primary_stages": home_allowed_non_primary_stages,
            "work_allowed_non_primary_stages": work_allowed_non_primary_stages,
            "home_non_primary_groups": home_non_primary_groups,
            "work_non_primary_groups": work_non_primary_groups,
            "home_failure_exposure_groups": home_failure_exposure_groups,
            "work_failure_exposure_groups": work_failure_exposure_groups,
            "home_zero_fallback_success": bool(home_unassigned == 0 and home_fallback_assignments == 0 and home_failure_exposure_groups == 0),
            "work_zero_fallback_success": bool(work_unassigned == 0 and work_fallback_assignments == 0 and work_failure_exposure_groups == 0),
        },
        "integrity": {
            "home": home_integrity,
            "work": work_integrity,
        },
        "geometry": {
            "home_candidates": home_geom,
            "work_candidates": work_geom,
        },
        "limits": {
            "work_validation_scope": (
                "当前 work 点验证 tract 内 support consistency。"
                if work_group_col != group_col
                else "当前 work 点只验证 tract 内 support consistency，不验证 commute realism，因为输入 persons 还没有独立 work tract。"
            ),
        },
    }
    result["coverage"]["overall_zero_fallback_success"] = bool(
        result["coverage"]["home_zero_fallback_success"] and result["coverage"]["work_zero_fallback_success"]
    )
    _write_json(out_json, result)
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()
