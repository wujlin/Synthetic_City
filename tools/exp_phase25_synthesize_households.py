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

from src.synthpop.paths import ensure_dir, project_root
from src.synthpop.spatial.tract_householding import (
    b11016_to_household_shell_targets,
    b19001_to_income_targets,
    load_or_fetch_acs_table,
    synthesize_households_from_persons,
)


def _utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_table(path: pathlib.Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".gz":
        return pd.read_csv(path, compression="gzip", low_memory=False)
    raise SystemExit(f"unsupported input format: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="exp_phase25_synthesize_households")
    ap.add_argument("--persons_path", required=True)
    ap.add_argument("--group_col", default="tract_geoid")
    ap.add_argument("--person_id_col", default="person_id")
    ap.add_argument("--age_col", default="AGEP_bin")
    ap.add_argument("--earn_col", default="EARN_16p_bin")
    ap.add_argument("--b11016_csv", default=str(project_root() / "dataset" / "census" / "acs5_2022_B11016_tract_michigan.csv.gz"))
    ap.add_argument("--b19001_csv", default=str(project_root() / "dataset" / "census" / "acs5_2022_B19001_tract_michigan.csv.gz"))
    ap.add_argument("--acs_year", default="2022")
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--household_id_prefix", default="hh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_dir", default="")
    ap.add_argument("--label", default="phase25_tract_households")
    args = ap.parse_args()

    persons_path = pathlib.Path(args.persons_path).expanduser().resolve()
    if not persons_path.exists():
        raise SystemExit(f"persons_path not found: {persons_path}")

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else (
        project_root() / "outputs" / f"_{args.label}_{_utc_now_compact()}"
    )
    synthetic_dir = ensure_dir(run_dir / "synthetic")
    metrics_dir = ensure_dir(run_dir / "metrics")

    persons = _read_table(persons_path)

    b11016_path = pathlib.Path(args.b11016_csv).expanduser().resolve()
    b19001_path = pathlib.Path(args.b19001_csv).expanduser().resolve()
    b11016 = load_or_fetch_acs_table(path=b11016_path, table_id="B11016", year=str(args.acs_year), statefp=str(args.statefp))
    b19001 = load_or_fetch_acs_table(path=b19001_path, table_id="B19001", year=str(args.acs_year), statefp=str(args.statefp))

    shell_targets = b11016_to_household_shell_targets(b11016_df=b11016, group_col=str(args.group_col))
    income_targets = b19001_to_income_targets(b19001_df=b19001, group_col=str(args.group_col))

    persons_hh, households, hh_meta = synthesize_households_from_persons(
        persons=persons,
        shell_targets=shell_targets,
        income_targets=income_targets,
        group_col=str(args.group_col),
        person_id_col=str(args.person_id_col),
        age_col=str(args.age_col),
        earn_col=str(args.earn_col),
        household_id_prefix=str(args.household_id_prefix),
        seed=int(args.seed),
    )

    group_diag = (
        persons_hh.groupby(str(args.group_col), as_index=False)
        .agg(
            n_persons=(str(args.person_id_col), "size"),
            n_households=("household_id", "nunique"),
            n_family_households=("household_type", lambda s: int((s == "family").sum())),
            n_nonfamily_households=("household_type", lambda s: int((s == "nonfamily").sum())),
        )
        .sort_values(str(args.group_col))
        .reset_index(drop=True)
    )
    group_diag["persons_per_household"] = (
        pd.to_numeric(group_diag["n_persons"], errors="coerce").fillna(0.0)
        / pd.to_numeric(group_diag["n_households"], errors="coerce").replace(0, pd.NA).astype(float)
    )

    households_parquet = synthetic_dir / "households.parquet"
    households_csv = synthetic_dir / "households.csv"
    persons_hh_parquet = synthetic_dir / "persons_with_households.parquet"
    persons_hh_csv = synthetic_dir / "persons_with_households.csv"
    shell_targets_csv = synthetic_dir / "household_shell_targets.csv"
    income_targets_csv = synthetic_dir / "household_income_targets.csv"
    group_diag_csv = metrics_dir / "group_diagnostics.csv"

    households.to_parquet(households_parquet, index=False)
    households.to_csv(households_csv, index=False)
    persons_hh.to_parquet(persons_hh_parquet, index=False)
    persons_hh.to_csv(persons_hh_csv, index=False)
    shell_targets.to_csv(shell_targets_csv, index=False)
    income_targets.to_csv(income_targets_csv, index=False)
    group_diag.to_csv(group_diag_csv, index=False)

    summary = {
        "label": str(args.label),
        "run_dir": str(run_dir),
        "created_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "persons_path": str(persons_path),
        "group_col": str(args.group_col),
        "person_id_col": str(args.person_id_col),
        "age_col": str(args.age_col),
        "earn_col": str(args.earn_col),
        "seed": int(args.seed),
        "acs_year": str(args.acs_year),
        "statefp": str(args.statefp),
        "b11016_csv": str(b11016_path),
        "b19001_csv": str(b19001_path),
        "householding_meta": hh_meta,
        "artifacts": {
            "households_parquet": str(households_parquet),
            "households_csv": str(households_csv),
            "persons_with_households_parquet": str(persons_hh_parquet),
            "persons_with_households_csv": str(persons_hh_csv),
            "household_shell_targets_csv": str(shell_targets_csv),
            "household_income_targets_csv": str(income_targets_csv),
            "group_diagnostics_csv": str(group_diag_csv),
        },
    }
    _write_json(metrics_dir / "summary.json", summary)
    _write_json(run_dir / "run_summary.json", summary)

    print(f"[ok] wrote: {households_parquet}")
    print(f"[ok] wrote: {persons_hh_parquet}")
    print(f"[ok] wrote: {metrics_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
