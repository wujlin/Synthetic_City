#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import pathlib
import subprocess
import sys
from typing import Any

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DC_STATEFP = "11"


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_tag() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _patch_dc_state_mapping() -> None:
    from tools import detroit_fetch_public_data as detroit

    detroit._STATEFP_TO_POSTAL_50[DC_STATEFP] = "dc"


def _run_module_main(module_name: str, argv: list[str]) -> None:
    _patch_dc_state_mapping()
    module = importlib.import_module(module_name)
    old_argv = sys.argv[:]
    try:
        sys.argv = [module_name.rsplit(".", 1)[-1], *argv]
        module.main()
    finally:
        sys.argv = old_argv


def _build_dc_condition_and_target(
    *,
    run_dir: pathlib.Path,
    acs_year: int,
    pums_year: int,
    pums_dir: pathlib.Path,
    api_key: str | None,
    overwrite: bool,
) -> dict[str, str]:
    data_dir = run_dir / "data"
    condition_dir = data_dir / "external_conditions"
    target_dir = data_dir / "external_targets"
    condition_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    base_condition = condition_dir / f"extcond_v1_acs5_{acs_year}_puma_state11.csv"
    earn_condition = condition_dir / f"extcond_earn_v1_acs5_{acs_year}_puma_state11.csv"
    merged_condition = condition_dir / f"extcond_v1_earn_v1_acs5_{acs_year}_puma_state11.csv"

    common = ["--statefp", DC_STATEFP]
    if overwrite:
        common.append("--overwrite")
    if api_key:
        common.extend(["--api_key", api_key])

    _run_module_main(
        "tools.data.build_external_condition_v1_acs_puma",
        [
            "--acs_year",
            str(acs_year),
            *common,
            "--out_path",
            str(base_condition),
        ],
    )
    _run_module_main(
        "tools.data.build_external_condition_earn_v1_acs_puma",
        [
            "--acs_year",
            str(acs_year),
            *common,
            "--out_path",
            str(earn_condition),
        ],
    )
    _run_module_main(
        "tools.data.merge_external_condition_v1_with_earn",
        [
            "--base_condition_csv",
            str(base_condition),
            "--earn_condition_csv",
            str(earn_condition),
            "--out_path",
            str(merged_condition),
            *(["--overwrite"] if overwrite else []),
        ],
    )

    _run_module_main(
        "tools.data.build_external_target_earn_conditional_v1_us",
        [
            "--statefp",
            DC_STATEFP,
            "--pums_year",
            str(pums_year),
            "--pums_dir",
            str(pums_dir),
            "--out_dir",
            str(target_dir),
        ],
    )
    cond_target = target_dir / f"exttarget_earn_cond_v1_pums_{pums_year}_puma_state11.csv"
    _run_module_main(
        "tools.data.build_external_target_v1_full_earn",
        [
            "--conditional_target_csv",
            str(cond_target),
            "--condition_csv",
            str(merged_condition),
            "--pums_year",
            str(pums_year),
            "--scope_tag",
            "state11",
            "--out_dir",
            str(target_dir),
            *(["--overwrite"] if overwrite else []),
        ],
    )
    wide_target = target_dir / f"exttarget_v1_full_earn_pums_{pums_year}_puma_state11_joint_wide.csv"
    schema_json = target_dir / f"exttarget_v1_full_earn_pums_{pums_year}_puma_state11.schema.json"

    return {
        "base_condition_csv": str(base_condition),
        "earn_condition_csv": str(earn_condition),
        "merged_condition_csv": str(merged_condition),
        "condition_schema_json": str(merged_condition.with_suffix(merged_condition.suffix + ".schema.json")),
        "conditional_target_csv": str(cond_target),
        "target_joint_wide_csv": str(wide_target),
        "target_schema_json": str(schema_json),
    }


def _export_dc_hierarchical_prediction(
    *,
    run_dir: pathlib.Path,
    seed: int,
    target_joint_wide_csv: pathlib.Path,
    target_schema_json: pathlib.Path,
    condition_csv: pathlib.Path,
    device: str,
    n_eval_joint_samples: int | None,
    ipf_iters: int | None,
) -> dict[str, str | int | float]:
    import dataclasses

    from tools.model.eval_external_c2f_full_earn_pipeline import _load_full_joint_wide
    from tools.experimental.workflows.export_michigan_pairwise_tvd_long import (
        _build_hierarchical_vectors,
        _load_ext_inputs,
        _select_device,
    )
    from tools.figures.make_fig4_michigan_regional_validation import PIPELINE_CONFIGS

    cfg = PIPELINE_CONFIGS[int(seed)]
    if n_eval_joint_samples is not None:
        cfg = dataclasses.replace(cfg, stage2_n_eval_joint_samples=int(n_eval_joint_samples))
    if ipf_iters is not None:
        cfg = dataclasses.replace(cfg, ipf_iters=int(ipf_iters))

    df, p_true, ids = _load_full_joint_wide(joint_wide_csv=target_joint_wide_csv, schema_json=target_schema_json)
    cond_raw, ext_marg = _load_ext_inputs(
        condition_csv=condition_csv,
        condition_schema_json=target_schema_json,
        ids=ids,
        stage1_schema_json=target_schema_json,
    )
    idx = np.arange(p_true.shape[0], dtype=np.int64)
    selected_device = _select_device(device)
    p_hier = _build_hierarchical_vectors(
        cfg=cfg,
        p_true_all=p_true,
        cond_raw=cond_raw,
        ext_marg=ext_marg,
        heldout_idx=idx,
        heldout_statefp=DC_STATEFP,
        device=selected_device,
    )
    p_hier = p_hier / np.maximum(p_hier.sum(axis=1, keepdims=True), 1e-12)

    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    key = f"hierarchical_seed{int(seed)}"
    npz_path = model_dir / "hierarchical_full_earn_dc_vectors.npz"
    np.savez_compressed(
        npz_path,
        p_true=p_true.astype(np.float32),
        **{key: p_hier.astype(np.float32)},
        puma_uid=df["puma_uid"].astype(str).to_numpy(),
        statefp=df["statefp"].astype(str).to_numpy(),
        puma5=df["puma5"].astype(str).to_numpy(),
    )

    pred_csv = model_dir / f"predicted_joint_wide_dc_seed{int(seed)}_hierarchical.csv"
    pred_summary = model_dir / f"predicted_joint_wide_dc_seed{int(seed)}_hierarchical.summary.json"
    cmd = [
        sys.executable,
        "tools/model/export_predicted_joint_wide_from_npz.py",
        "--npz",
        str(npz_path),
        "--key",
        key,
        "--reference_joint_wide_csv",
        str(target_joint_wide_csv),
        "--schema_json",
        str(target_schema_json),
        "--out_csv",
        str(pred_csv),
        "--out_summary_json",
        str(pred_summary),
        "--normalize",
    ]
    subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)

    tvd = 0.5 * np.abs(p_hier - p_true).sum(axis=1)
    payload: dict[str, str | int | float] = {
        "npz": str(npz_path),
        "prediction_key": key,
        "predicted_joint_wide_csv": str(pred_csv),
        "prediction_summary_json": str(pred_summary),
        "seed": int(seed),
        "device": selected_device,
        "n_dc_pumas": int(p_hier.shape[0]),
        "target_total_person_weight": float(df["total_person_weight"].astype(float).sum()),
        "diagnostic_tvd_vs_dc_pums_mean": float(np.mean(tvd)),
        "diagnostic_tvd_vs_dc_pums_median": float(np.median(tvd)),
        "stage1_checkpoint": str(cfg.stage1_checkpoint),
        "stage2_checkpoint": str(cfg.stage2_checkpoint),
        "stage2_n_eval_joint_samples": int(cfg.stage2_n_eval_joint_samples),
        "ipf_iters": int(cfg.ipf_iters),
    }
    _write_json(model_dir / "dc_prediction_summary.json", payload)
    return payload


def _run_dc_spatial(
    *,
    run_dir: pathlib.Path,
    asset_inventory_csv: pathlib.Path,
    joint_wide_csv: pathlib.Path,
    schema_json: pathlib.Path,
    seed: int,
    n_jobs: int,
    sample_n: int,
) -> dict[str, Any]:
    import pandas as pd

    inv = pd.read_csv(asset_inventory_csv, dtype={"statefp": str})
    inv["statefp"] = inv["statefp"].astype(str).str.zfill(2)
    rows = inv[inv["statefp"].eq(DC_STATEFP)].to_dict("records")
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one DC row in asset inventory, got {len(rows)}")
    row = rows[0]
    cmd = [
        sys.executable,
        "tools/experimental/workflows/run_paper1_spatial_state_pipeline.py",
        "--repo_root",
        str(_REPO_ROOT),
        "--run_dir",
        str(run_dir),
        "--statefp",
        DC_STATEFP,
        "--state_postal",
        str(row["state_postal"]),
        "--joint_wide_csv",
        str(joint_wide_csv),
        "--schema_json",
        str(schema_json),
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
        str(int(n_jobs)),
        "--seed",
        str(int(seed)),
        "--sample_n",
        str(int(sample_n)),
    ]
    log_path = run_dir / "run.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_utc_now()}] DC spatial command {' '.join(cmd)}\n")
        log.flush()
        subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True, stdout=log, stderr=subprocess.STDOUT)
    qc_path = run_dir / "metrics" / "state11_qc_summary.json"
    return json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {}


def main() -> int:
    ap = argparse.ArgumentParser(prog="run_paper1_dc_patch_pipeline")
    ap.add_argument("--run_dir", type=pathlib.Path, default=None)
    ap.add_argument("--acs_year", type=int, default=2022)
    ap.add_argument("--pums_year", type=int, default=2023)
    ap.add_argument(
        "--pums_dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/raw/pums/pums_2023_5-Year"),
    )
    ap.add_argument("--api_key_file", type=pathlib.Path, default=pathlib.Path("/home/jinlin/.config/synthetic_city/census_api_key"))
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n_eval_joint_samples", type=int, default=None)
    ap.add_argument("--ipf_iters", type=int, default=None)
    ap.add_argument(
        "--asset_inventory_csv",
        type=pathlib.Path,
        default=pathlib.Path("/home/jinlin/projects/Synthetic_City/outputs/_paper1_full_us_spatial_assets_2023_20260515T080630Z/metrics/state_asset_inventory.csv"),
    )
    ap.add_argument("--run_spatial", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=32)
    ap.add_argument("--sample_n", type=int, default=100000)
    ap.add_argument("--overwrite_intermediates", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir or pathlib.Path(f"outputs/_paper1_dc_spatial_population_2023_{_utc_tag()}")
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    api_key = None
    if args.api_key_file and args.api_key_file.exists():
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()

    pums_zip = args.pums_dir.expanduser().resolve() / "csv_pdc.zip"
    if not pums_zip.exists():
        raise SystemExit(f"DC PUMS zip not found: {pums_zip}")

    artifacts = _build_dc_condition_and_target(
        run_dir=run_dir,
        acs_year=int(args.acs_year),
        pums_year=int(args.pums_year),
        pums_dir=args.pums_dir.expanduser().resolve(),
        api_key=api_key,
        overwrite=bool(args.overwrite_intermediates),
    )
    prediction = _export_dc_hierarchical_prediction(
        run_dir=run_dir,
        seed=int(args.seed),
        target_joint_wide_csv=pathlib.Path(artifacts["target_joint_wide_csv"]),
        target_schema_json=pathlib.Path(artifacts["target_schema_json"]),
        condition_csv=pathlib.Path(artifacts["merged_condition_csv"]),
        device=str(args.device),
        n_eval_joint_samples=args.n_eval_joint_samples,
        ipf_iters=args.ipf_iters,
    )

    spatial_qc: dict[str, Any] | None = None
    if bool(args.run_spatial):
        spatial_qc = _run_dc_spatial(
            run_dir=run_dir,
            asset_inventory_csv=args.asset_inventory_csv.expanduser().resolve(),
            joint_wide_csv=pathlib.Path(str(prediction["predicted_joint_wide_csv"])),
            schema_json=pathlib.Path(artifacts["target_schema_json"]),
            seed=int(args.seed),
            n_jobs=int(args.n_jobs),
            sample_n=int(args.sample_n),
        )

    summary = {
        "created_utc": _utc_now(),
        "run_dir": str(run_dir),
        "status": "completed",
        "scope": "District of Columbia patch for Paper 1 full-US spatial product",
        "statefp": DC_STATEFP,
        "acs_year": int(args.acs_year),
        "pums_year": int(args.pums_year),
        "pums_zip": str(pums_zip),
        "artifacts": artifacts,
        "prediction": prediction,
        "spatial_qc": spatial_qc,
    }
    _write_json(run_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
