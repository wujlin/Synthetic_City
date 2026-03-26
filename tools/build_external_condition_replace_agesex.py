#!/usr/bin/env python3
from __future__ import annotations

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


def _load_schema(path: pathlib.Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"invalid schema json: {path}")
    return obj


def _filter_schema(
    schema: dict[str, Any],
    *,
    keep_variables: list[str],
    schema_name: str,
    reference_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_categories = dict(schema.get("categories", {}))
    ref_categories = dict(reference_schema.get("categories", {})) if isinstance(reference_schema, dict) else {}
    categories: dict[str, list[str]] = {}
    for var in keep_variables:
        if str(var) in ref_categories:
            categories[str(var)] = [str(x) for x in list(ref_categories[str(var)])]
            continue
        if str(var) not in raw_categories:
            raise SystemExit(f"schema missing categories for variable={var}")
        categories[str(var)] = [str(x) for x in list(raw_categories[str(var)])]
    return {
        "schema": schema_name,
        "variable_order": [str(v) for v in keep_variables],
        "categories": categories,
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_external_condition_replace_agesex")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--input_schema_json", required=True)
    ap.add_argument(
        "--reference_schema_json",
        default=None,
        help="Optional schema JSON that defines canonical category order for overlapping variables.",
    )
    ap.add_argument("--out_csv", required=True)
    ap.add_argument(
        "--keep_variables",
        default="AGEP_SEX_cross,SCHL_allpop,ESR_allpop,EARN_16p_bin",
        help="Comma-separated variable order for the output condition schema.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    input_csv = pathlib.Path(args.input_csv).expanduser().resolve()
    input_schema_json = pathlib.Path(args.input_schema_json).expanduser().resolve()
    reference_schema_json = pathlib.Path(args.reference_schema_json).expanduser().resolve() if args.reference_schema_json else None
    out_csv = pathlib.Path(args.out_csv).expanduser().resolve()
    keep_variables = [str(x).strip() for x in str(args.keep_variables).split(",") if str(x).strip()]

    if not input_csv.exists():
        raise SystemExit(f"input_csv not found: {input_csv}")
    if not input_schema_json.exists():
        raise SystemExit(f"input_schema_json not found: {input_schema_json}")
    if reference_schema_json is not None and not reference_schema_json.exists():
        raise SystemExit(f"reference_schema_json not found: {reference_schema_json}")
    if len(keep_variables) == 0:
        raise SystemExit("keep_variables is empty")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not args.overwrite:
        raise SystemExit(f"out_csv exists: {out_csv} (use --overwrite)")

    df = pd.read_csv(input_csv, low_memory=False)
    need = {"statefp", "puma", "puma_uid", "variable", "category", "target"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"input_csv missing columns: {miss}")

    input_schema = _load_schema(input_schema_json)
    reference_schema = _load_schema(reference_schema_json) if reference_schema_json is not None else None
    keep_set = {str(v) for v in keep_variables}
    filtered = df.loc[df["variable"].astype(str).isin(keep_set)].copy()
    filtered["variable"] = filtered["variable"].astype(str)
    filtered["category"] = filtered["category"].astype(str)

    have_variables = filtered["variable"].drop_duplicates().tolist()
    missing_vars = [v for v in keep_variables if v not in have_variables]
    if missing_vars:
        raise SystemExit(f"input_csv missing requested variables: {missing_vars}")

    order_map = {str(v): i for i, v in enumerate(keep_variables)}
    category_order_map: dict[str, dict[str, int]] = {}
    raw_categories = dict(input_schema.get("categories", {}))
    for var in keep_variables:
        cats = [str(x) for x in list(raw_categories[str(var)])]
        category_order_map[str(var)] = {cat: i for i, cat in enumerate(cats)}

    filtered["_var_order"] = filtered["variable"].map(order_map)
    filtered["_cat_order"] = filtered.apply(
        lambda r: category_order_map[str(r["variable"])].get(str(r["category"]), 10**9), axis=1
    )
    filtered = filtered.sort_values(
        ["statefp", "puma", "_var_order", "_cat_order"],
        kind="stable",
    ).drop(columns=["_var_order", "_cat_order"]).reset_index(drop=True)

    output_schema = _filter_schema(
        input_schema,
        keep_variables=keep_variables,
        schema_name="external_condition_v1_agesex_replace_earn_v1",
        reference_schema=reference_schema,
    )

    filtered.to_csv(out_csv, index=False)
    schema_path = out_csv.with_suffix(out_csv.suffix + ".schema.json")
    schema_path.write_text(json.dumps(output_schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta = {
        "schema": output_schema["schema"],
        "created_utc": _utc_now_iso(),
        "input_csv": str(input_csv),
        "input_schema_json": str(input_schema_json),
        "reference_schema_json": str(reference_schema_json) if reference_schema_json is not None else None,
        "out_csv": str(out_csv),
        "keep_variables": keep_variables,
        "n_rows": int(filtered.shape[0]),
        "n_pumas": int(filtered["puma_uid"].astype(str).nunique()),
        "variables": keep_variables,
    }
    meta_path = out_csv.with_suffix(out_csv.suffix + ".metadata.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_csv}")


if __name__ == "__main__":
    main()
