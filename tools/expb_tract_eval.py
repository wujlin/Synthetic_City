#!/usr/bin/env python3
"""
Exp B: tract-level value test (diffusion seed vs global seed baseline).

Question this script answers:
- At sub-PUMA scale (tract), does diffusion seed retain measurable value over
  a non-spatial global seed baseline?

Design:
- Reuse Exp5 helper functions for full metric consistency.
- Compare *pre-alignment* results only (no post IPF), to isolate model value.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import sys
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Missing dependency: {pkg}. Install it in your conda env.") from e


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_delta(before: float | None, after: float | None) -> dict[str, float | None]:
    if before is None or after is None:
        return {"before": before, "after": after, "delta": None}
    return {"before": float(before), "after": float(after), "delta": float(after - before)}


def main() -> None:
    pd = _require("pandas")

    from src.synthpop.pipeline.detroit_v0 import make_run_id
    from src.synthpop.paths import data_root as default_data_root
    from tools.exp5_tract_postalign import (
        _build_global_seed_like,
        _derive_scope_columns,
        _harmonize_synthetic_columns,
        _load_pums_reference,
        _load_targets_long,
        _puma_metrics_vs_pums,
        _weighted_tvd_to_targets,
    )

    ap = argparse.ArgumentParser(prog="expb_tract_eval")
    ap.add_argument("--synthetic_path", required=True, help="Exp4 synthetic sample file (csv/csv.gz/parquet).")
    ap.add_argument("--acs_targets_long", required=True, help="Tract-level ACS targets_long (csv/parquet).")
    ap.add_argument("--data_root", default=str(default_data_root()))
    ap.add_argument("--pums_year", type=int, default=2022)
    ap.add_argument("--pums_period", default="5-Year")
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--pums_person_zip", default=None)
    ap.add_argument("--tract_col", default="tract_geoid")
    ap.add_argument("--puma_col", default="puma")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = (
        pathlib.Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (_REPO_ROOT / "outputs" / make_run_id(prefix="expb_tract_eval"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        out_dir / "run.metadata.json",
        {
            "created_utc": _utc_now_iso(),
            "argv": sys.argv,
            "script": pathlib.Path(__file__).name,
            "env": {"RAW_ROOT": os.environ.get("RAW_ROOT"), "SYNTHCITY_DATA_ROOT": os.environ.get("SYNTHCITY_DATA_ROOT")},
            "args": vars(args),
        },
    )

    syn_path = pathlib.Path(args.synthetic_path).expanduser().resolve()
    if not syn_path.exists():
        raise SystemExit(f"synthetic_path not found: {syn_path}")
    if syn_path.suffix.lower() == ".parquet":
        syn_raw = pd.read_parquet(syn_path)
    else:
        syn_raw = pd.read_csv(syn_path, low_memory=False)

    syn = _harmonize_synthetic_columns(syn_raw, tract_col=str(args.tract_col), puma_col=str(args.puma_col))
    syn = _derive_scope_columns(syn)
    syn = syn[(syn[str(args.tract_col)].astype(str) != "") & (syn[str(args.puma_col)].astype(str) != "")].copy()
    if syn.empty:
        raise SystemExit("synthetic becomes empty after tract/puma filtering.")

    targets = _load_targets_long(pathlib.Path(args.acs_targets_long).expanduser().resolve(), tract_col=str(args.tract_col))
    if not targets:
        raise SystemExit("No valid targets loaded from acs_targets_long.")
    valid_tracts = set(targets.keys())
    syn = syn[syn[str(args.tract_col)].astype(str).isin(valid_tracts)].copy()
    if syn.empty:
        raise SystemExit("No overlapping tracts between synthetic and targets_long.")

    diffusion_seed = syn.copy().reset_index(drop=True)
    global_seed = _build_global_seed_like(syn, seed=int(args.seed)).reset_index(drop=True)
    global_seed = _derive_scope_columns(global_seed)

    for d in [diffusion_seed, global_seed]:
        if "W_eval" not in d.columns:
            d["W_eval"] = 1.0
        d["W_eval"] = pd.to_numeric(d["W_eval"], errors="coerce").fillna(1.0).clip(lower=0.0)

    tract_diff = _weighted_tvd_to_targets(
        df=diffusion_seed,
        tract_col=str(args.tract_col),
        wcol="W_eval",
        targets=targets,
    )
    tract_glob = _weighted_tvd_to_targets(
        df=global_seed,
        tract_col=str(args.tract_col),
        wcol="W_eval",
        targets=targets,
    )

    ref = _load_pums_reference(
        data_root=pathlib.Path(args.data_root).expanduser().resolve(),
        pums_year=int(args.pums_year),
        pums_period=str(args.pums_period),
        statefp=str(args.statefp),
        pums_person_zip=args.pums_person_zip,
        puma_col_out=str(args.puma_col),
    )
    puma_diff = _puma_metrics_vs_pums(
        syn=diffusion_seed,
        puma_col=str(args.puma_col),
        wcol="W_eval",
        ref=ref,
        ref_wcol="PWGTP",
    )
    puma_glob = _puma_metrics_vs_pums(
        syn=global_seed,
        puma_col=str(args.puma_col),
        wcol="W_eval",
        ref=ref,
        ref_wcol="PWGTP",
    )

    tract_cmp: dict[str, Any] = {}
    for var in ["PINCP_16p_bin", "ESR_16p", "SCHL_25p"]:
        b = tract_glob.get(var, {}).get("mean")
        a = tract_diff.get(var, {}).get("mean")
        tract_cmp[var] = _safe_delta(b, a)

    puma_cmp: dict[str, Any] = {}
    for metric in [
        "tvd_income_bin",
        "tvd_schl",
        "tvd_esr",
        "copula_tvd_age_income",
        "joint_tvd_age_income_bin",
        "puma_cosine_age_income_bin_joint",
    ]:
        b = puma_glob.get("summary", {}).get(metric, {}).get("mean")
        a = puma_diff.get("summary", {}).get(metric, {}).get("mean")
        puma_cmp[metric] = _safe_delta(b, a)

    summary = {
        "created_utc": _utc_now_iso(),
        "n_rows": int(syn.shape[0]),
        "n_tracts_overlap": int(syn[str(args.tract_col)].nunique()),
        "n_pumas_overlap": int(syn[str(args.puma_col)].nunique()),
        "tract_mean_tvd_global_to_diffusion": tract_cmp,
        "puma_mean_metric_global_to_diffusion": puma_cmp,
        "note": "delta = diffusion - global. For TVD lower is better, for cosine higher is better.",
    }

    _write_json(out_dir / "tract_metrics_diffusion_seed.json", tract_diff)
    _write_json(out_dir / "tract_metrics_global_seed.json", tract_glob)
    _write_json(out_dir / "puma_metrics_diffusion_seed.json", puma_diff)
    _write_json(out_dir / "puma_metrics_global_seed.json", puma_glob)
    _write_json(out_dir / "expb_summary.json", summary)
    print(f"[ok] wrote: {out_dir}")


if __name__ == "__main__":
    main()

