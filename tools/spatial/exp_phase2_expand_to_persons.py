#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Any

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.paths import ensure_dir, project_root
from src.synthpop.spatial.allocation_expansion import (
    expand_integer_allocation_to_persons,
    integerize_type_allocation_long,
)


def _utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _canon_id_series(s: pd.Series, *, col: str) -> pd.Series:
    out = s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    if str(col) == "puma_uid":
        out = out.str.zfill(7)
    elif str(col) in {"tract_geoid", "work_tract_geoid"}:
        out = out.str.zfill(11)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp_phase2_expand_to_persons")
    ap.add_argument("--allocation_long_csv", required=True)
    ap.add_argument("--region_col", default="puma_uid")
    ap.add_argument("--group_col", default="tract_geoid")
    ap.add_argument("--type_idx_col", default="type_idx")
    ap.add_argument("--count_col", default="count")
    ap.add_argument("--out_count_col", default="count_int")
    ap.add_argument("--person_id_col", default="person_id")
    ap.add_argument("--person_id_prefix", default="synp")
    ap.add_argument("--esr_col", default="ESR_allpop")
    ap.add_argument("--employed_values", default="employed")
    ap.add_argument("--skip_persons_csv", action="store_true")
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--label", default="phase2_expand_to_persons")
    args = ap.parse_args()

    allocation_path = pathlib.Path(args.allocation_long_csv).expanduser().resolve()
    if not allocation_path.exists():
        raise SystemExit(f"allocation_long_csv not found: {allocation_path}")

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    synthetic_dir = ensure_dir(run_dir / "synthetic")
    metrics_dir = ensure_dir(run_dir / "metrics")

    alloc = pd.read_csv(allocation_path, low_memory=False)
    for col in [str(args.region_col), str(args.group_col)]:
        if col in alloc.columns:
            alloc[col] = _canon_id_series(alloc[col], col=col)
    alloc_int, int_meta = integerize_type_allocation_long(
        allocation_long=alloc,
        region_col=str(args.region_col),
        group_col=str(args.group_col),
        type_idx_col=str(args.type_idx_col),
        count_col=str(args.count_col),
        out_count_col=str(args.out_count_col),
    )
    persons, person_meta = expand_integer_allocation_to_persons(
        integer_allocation_long=alloc_int,
        count_col=str(args.out_count_col),
        person_id_col=str(args.person_id_col),
        person_id_prefix=str(args.person_id_prefix),
        esr_col=str(args.esr_col),
        employed_values=tuple(x.strip() for x in str(args.employed_values).split(",") if x.strip()),
    )

    alloc_int_csv = synthetic_dir / "type_assignment_long_integer.csv"
    persons_parquet = synthetic_dir / "persons.parquet"
    persons_csv = synthetic_dir / "persons.csv"
    alloc_int.to_csv(alloc_int_csv, index=False)
    persons.to_parquet(persons_parquet, index=False)
    if not bool(args.skip_persons_csv):
        persons.to_csv(persons_csv, index=False)

    summary = {
        "allocation_long_csv": str(allocation_path),
        "region_col": str(args.region_col),
        "group_col": str(args.group_col),
        "type_idx_col": str(args.type_idx_col),
        "count_col": str(args.count_col),
        "out_count_col": str(args.out_count_col),
        "person_id_col": str(args.person_id_col),
        "person_id_prefix": str(args.person_id_prefix),
        "esr_col": str(args.esr_col),
        "employed_values": [x.strip() for x in str(args.employed_values).split(",") if x.strip()],
        "integerize_meta": int_meta,
        "person_meta": person_meta,
        "artifacts": {
            "integer_allocation_csv": str(alloc_int_csv),
            "persons_parquet": str(persons_parquet),
            "persons_csv": (None if bool(args.skip_persons_csv) else str(persons_csv)),
        },
    }
    run_summary = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "created_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "summary_json": str(metrics_dir / "summary.json"),
        "artifacts": summary["artifacts"],
        "person_meta": person_meta,
    }
    _write_json(metrics_dir / "summary.json", summary)
    _write_json(run_dir / "run_summary.json", run_summary)
    print(f"[ok] wrote run summary: {run_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
