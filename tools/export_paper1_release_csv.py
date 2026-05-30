#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
from typing import Any

import pandas as pd


RELEASE_COLUMNS = [
    "person_id",
    "age",
    "gender",
    "education",
    "employment",
    "income",
    "home_lon",
    "home_lat",
    "work_lon",
    "work_lat",
]

SOURCE_COLUMNS = {
    "person_id": "person_id",
    "age": "AGEP_bin",
    "gender": "SEX",
    "education": "SCHL_allpop",
    "employment": "ESR_allpop",
    "income": "EARN_16p_bin",
    "home_lon": "home_lon",
    "home_lat": "home_lat",
    "work_lon": "work_lon",
    "work_lat": "work_lat",
}


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _statefp_from_parquet_path(path: pathlib.Path) -> str:
    parent = path.parent.name
    if parent.startswith("state="):
        return parent.split("=", 1)[1].zfill(2)
    return path.stem.replace("state", "").zfill(2)


def _selected_statefps(value: str) -> set[str] | None:
    text = str(value).strip().lower()
    if text in {"", "ready", "all"}:
        return None
    return {part.strip().zfill(2) for part in str(value).split(",") if part.strip()}


def _is_missing_pair(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    return df[x_col].isna() | df[y_col].isna()


def _transform_batch(batch_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = pd.DataFrame()
    for target, source in SOURCE_COLUMNS.items():
        out[target] = batch_df[source]

    out["person_id"] = (
        out["person_id"]
        .astype("string")
        .str.replace(r"^synp\d+_", "", regex=True)
        .str.strip()
    )
    out["income"] = out["income"].astype("string").replace(
        {"not_in_earnings_universe": "not_in_earnings"}
    )
    out["gender"] = pd.to_numeric(out["gender"], errors="coerce").astype("Int64")
    for col in ["home_lon", "home_lat", "work_lon", "work_lat"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    home_missing = _is_missing_pair(out, "home_lon", "home_lat")
    work_missing = _is_missing_pair(out, "work_lon", "work_lat")
    fill_work = work_missing & (~home_missing)
    out.loc[fill_work, ["work_lon", "work_lat"]] = out.loc[fill_work, ["home_lon", "home_lat"]].to_numpy()

    home_missing_after = _is_missing_pair(out, "home_lon", "home_lat")
    work_missing_after = _is_missing_pair(out, "work_lon", "work_lat")
    stats = {
        "rows": int(out.shape[0]),
        "missing_home_before": int(home_missing.sum()),
        "missing_work_before": int(work_missing.sum()),
        "missing_both_before": int((home_missing & work_missing).sum()),
        "work_filled_from_home": int(fill_work.sum()),
        "missing_home_after": int(home_missing_after.sum()),
        "missing_work_after": int(work_missing_after.sum()),
        "missing_both_after": int((home_missing_after & work_missing_after).sum()),
    }
    return out[RELEASE_COLUMNS], stats


def _merge_stats(total: dict[str, int], part: dict[str, int]) -> None:
    for k, v in part.items():
        total[k] = int(total.get(k, 0) + int(v))


def _export_state(
    *,
    parquet_path: pathlib.Path,
    out_csv: pathlib.Path,
    chunksize: int,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    required = list(SOURCE_COLUMNS.values())
    missing = [c for c in required if c not in pf.schema.names]
    if missing:
        raise SystemExit(f"{parquet_path} missing release source columns: {missing}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()

    totals: dict[str, int] = {}
    wrote_header = False
    for batch in pf.iter_batches(batch_size=int(chunksize), columns=required):
        release, stats = _transform_batch(batch.to_pandas())
        _merge_stats(totals, stats)
        release.to_csv(
            out_csv,
            index=False,
            mode="a",
            header=not wrote_header,
            quoting=csv.QUOTE_MINIMAL,
        )
        wrote_header = True

    return {
        "statefp": _statefp_from_parquet_path(parquet_path),
        **totals,
        "input_parquet": str(parquet_path),
        "output_csv": str(out_csv),
        "output_file_size_bytes": int(out_csv.stat().st_size if out_csv.exists() else 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="export_paper1_release_csv")
    ap.add_argument("--run_dir", required=True, type=pathlib.Path)
    ap.add_argument("--states", default="ready")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--fail_on_missing_home", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    out_dir_arg = str(args.out_dir).strip()
    out_dir = pathlib.Path(out_dir_arg).expanduser().resolve() if out_dir_arg else run_dir / "release_csv"
    selected = _selected_statefps(str(args.states))

    parquet_paths = sorted((run_dir / "synthetic").glob("state=*/persons.parquet"))
    if selected is not None:
        parquet_paths = [p for p in parquet_paths if _statefp_from_parquet_path(p) in selected]
    if not parquet_paths:
        raise SystemExit(f"no state parquet files found under {run_dir / 'synthetic'}")

    rows = []
    for path in parquet_paths:
        statefp = _statefp_from_parquet_path(path)
        state_out = out_dir / f"state={statefp}" / f"synthetic_individuals_state{statefp}.csv"
        row = _export_state(parquet_path=path, out_csv=state_out, chunksize=int(args.chunksize))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    metrics_dir = run_dir / "metrics"
    summary_csv = metrics_dir / "release_csv_summary.csv"
    summary_json = metrics_dir / "release_csv_summary.json"
    summary_df = pd.DataFrame(rows).sort_values("statefp")
    summary_df.to_csv(summary_csv, index=False)
    total_rows = int(summary_df["rows"].sum())
    payload = {
        "created_utc": _utc_now(),
        "run_dir": str(run_dir),
        "release_csv_dir": str(out_dir),
        "release_columns": RELEASE_COLUMNS,
        "states_exported": int(summary_df.shape[0]),
        "rows": total_rows,
        "missing_home_after": int(summary_df["missing_home_after"].sum()),
        "missing_work_after": int(summary_df["missing_work_after"].sum()),
        "missing_both_after": int(summary_df["missing_both_after"].sum()),
        "work_filled_from_home": int(summary_df["work_filled_from_home"].sum()),
        "summary_csv": str(summary_csv),
    }
    _write_json(summary_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if bool(args.fail_on_missing_home) and int(payload["missing_home_after"]) > 0:
        raise SystemExit(f"release CSV still has missing home coordinates: {payload['missing_home_after']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
