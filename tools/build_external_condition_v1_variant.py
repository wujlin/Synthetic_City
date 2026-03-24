#!/usr/bin/env python3
from __future__ import annotations

"""
Project external-condition v1 marginals into a named refinement-ablation variant schema.
"""

import argparse
import json
import pathlib
import sys
from typing import Any

import pandas as pd


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.build_external_condition_v1_michigan import _utc_now_iso
from tools.external_v1_variant_presets import get_variant_spec
from tools.train_us_puma_5var_diffusion import _canon_puma5, _canon_statefp, _canon_uid_loose


def _normalize_geo_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "statefp" in out.columns:
        out["statefp"] = out["statefp"].map(_canon_statefp)
    if "puma" in out.columns:
        out["puma"] = out["puma"].map(_canon_puma5)
    if "puma_uid" in out.columns:
        out["puma_uid"] = out["puma_uid"].map(_canon_uid_loose)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_external_condition_v1_variant")
    ap.add_argument("--condition_csv", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out_path", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    spec = get_variant_spec(str(args.variant))
    in_path = pathlib.Path(args.condition_csv).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"condition_csv not found: {in_path}")

    default_name = in_path.name.replace("extcond_v1_", f"extcond_v1_{spec.name}_")
    out_path = pathlib.Path(args.out_path).expanduser().resolve() if args.out_path else in_path.with_name(default_name)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"out_path exists: {out_path} (use --overwrite)")

    header_cols = pd.read_csv(in_path, nrows=0).columns.astype(str).tolist()
    dtype_map = {k: str for k in ["statefp", "puma", "puma_uid", "variable", "category"] if k in header_cols}
    df = pd.read_csv(in_path, dtype=dtype_map, low_memory=False)
    df = _normalize_geo_cols(df)

    required = {"variable", "category", "target"}
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise SystemExit(f"condition_csv missing columns: {miss}")

    unsupported = sorted(set(df["variable"].astype(str)) - set(spec.variable_order))
    if unsupported:
        raise SystemExit(f"condition_csv has unsupported variables for variant projection: {unsupported}")

    def _map_category(var: str, cat: str) -> str:
        try:
            return spec.mappings[var][cat]
        except KeyError as e:
            raise SystemExit(f"Unmapped category for variant={spec.name}: var={var}, cat={cat}") from e

    geo_cols = [c for c in ["statefp", "puma", "puma_uid"] if c in df.columns]
    keep_cols = [c for c in ["table_id", "universe", "source", "acs_year", "geo_level"] if c in df.columns]

    df["variable"] = df["variable"].astype(str)
    df["category"] = df.apply(lambda r: _map_category(str(r["variable"]), str(r["category"])), axis=1)
    group_cols = geo_cols + ["variable", "category"]
    agg = df.groupby(group_cols, as_index=False, sort=False)["target"].sum()
    for c in keep_cols:
        agg[c] = df.groupby(group_cols, as_index=False, sort=False)[c].first()[c]
    agg["schema"] = f"external_condition_v1_{spec.name}"

    ordered_cols = geo_cols + ["variable", "category", "target"] + keep_cols + ["schema"]
    agg = agg[ordered_cols]
    agg.to_csv(out_path, index=False)

    meta = {
        "dataset": f"External condition v1 {spec.name}",
        "schema": f"external_condition_v1_{spec.name}",
        "variant": spec.name,
        "source_condition_csv": str(in_path),
        "out_path": str(out_path),
        "variable_order": spec.variable_order,
        "shape": spec.shape,
        "K": spec.K,
        "variables": spec.categories,
        "created_utc": _utc_now_iso(),
    }
    out_path.with_suffix(out_path.suffix + ".metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()

