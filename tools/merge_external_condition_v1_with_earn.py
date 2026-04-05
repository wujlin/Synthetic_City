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


def _infer_schema_from_df(df: pd.DataFrame, *, schema_name: str) -> dict[str, Any]:
    variable_order = df["variable"].astype(str).drop_duplicates().tolist()
    categories: dict[str, list[str]] = {}
    for var in variable_order:
        cats = df.loc[df["variable"].astype(str) == str(var), "category"].astype(str).drop_duplicates().tolist()
        categories[str(var)] = cats
    return {
        "schema": schema_name,
        "variable_order": variable_order,
        "categories": categories,
    }


def _load_schema_sidecar(csv_path: pathlib.Path) -> dict[str, Any] | None:
    schema_path = csv_path.with_suffix(csv_path.suffix + ".schema.json")
    if not schema_path.exists():
        return None
    try:
        return json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_schema_json(path: pathlib.Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"invalid schema json: {path}")
    return obj


def _merge_condition_schemas(
    *,
    base_schema: dict[str, Any] | None,
    earn_schema: dict[str, Any] | None,
    reference_schema: dict[str, Any] | None,
    merged_df: pd.DataFrame,
) -> dict[str, Any]:
    if base_schema is None and earn_schema is None:
        return _infer_schema_from_df(merged_df, schema_name="external_condition_v1_plus_earn_v1")

    variable_order: list[str] = []
    categories: dict[str, list[str]] = {}
    ref_categories = dict(reference_schema.get("categories", {})) if isinstance(reference_schema, dict) else {}
    for obj in [base_schema, earn_schema]:
        if not isinstance(obj, dict):
            continue
        for var in [str(x) for x in obj.get("variable_order", [])]:
            if var not in variable_order:
                variable_order.append(var)
        for var, cats in dict(obj.get("categories", {})).items():
            var = str(var)
            if var in ref_categories:
                categories[var] = [str(x) for x in list(ref_categories[var])]
            else:
                categories[var] = [str(x) for x in list(cats)]
    inferred = _infer_schema_from_df(merged_df, schema_name="external_condition_v1_plus_earn_v1")
    for var in inferred["variable_order"]:
        if var not in variable_order:
            variable_order.append(var)
    for var, cats in inferred["categories"].items():
        var = str(var)
        if var in ref_categories:
            categories.setdefault(var, [str(x) for x in list(ref_categories[var])])
        else:
            categories.setdefault(var, [str(x) for x in list(cats)])
    return {
        "schema": "external_condition_v1_plus_earn_v1",
        "variable_order": variable_order,
        "categories": categories,
    }


def _usable_geo_cols(df: pd.DataFrame) -> list[str]:
    candidates = ["puma_uid", "tract_geoid", "cbg_geoid", "county_geoid", "puma", "tract"]
    out: list[str] = []
    for col in candidates:
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.strip()
        s = s.where(~s.isin({"nan", "None", "null"}), "")
        if bool((s != "").any()):
            out.append(str(col))
    return out


def _select_merge_geo_cols(base: pd.DataFrame, earn: pd.DataFrame) -> list[str]:
    base_geo = _usable_geo_cols(base)
    earn_geo = _usable_geo_cols(earn)
    shared = [c for c in base_geo if c in set(earn_geo)]
    if not shared:
        raise SystemExit("cannot infer shared non-empty geography columns for merge")
    return shared


def main() -> None:
    ap = argparse.ArgumentParser(prog="merge_external_condition_v1_with_earn")
    ap.add_argument("--base_condition_csv", required=True)
    ap.add_argument("--earn_condition_csv", required=True)
    ap.add_argument(
        "--reference_schema_json",
        default=None,
        help="Optional schema JSON that defines canonical category order for overlapping variables.",
    )
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base_path = pathlib.Path(args.base_condition_csv).expanduser().resolve()
    earn_path = pathlib.Path(args.earn_condition_csv).expanduser().resolve()
    reference_schema_path = pathlib.Path(args.reference_schema_json).expanduser().resolve() if args.reference_schema_json else None
    out_path = pathlib.Path(args.out_path).expanduser().resolve()
    if not base_path.exists():
        raise SystemExit(f"base_condition_csv not found: {base_path}")
    if not earn_path.exists():
        raise SystemExit(f"earn_condition_csv not found: {earn_path}")
    if reference_schema_path is not None and not reference_schema_path.exists():
        raise SystemExit(f"reference_schema_json not found: {reference_schema_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"out_path exists: {out_path} (use --overwrite)")

    base = pd.read_csv(base_path, low_memory=False)
    earn = pd.read_csv(earn_path, low_memory=False)
    need = {"variable", "category", "target"}
    for name, df in [("base", base), ("earn", earn)]:
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise SystemExit(f"{name} condition missing columns: {miss}")

    geo_cols = _select_merge_geo_cols(base, earn)
    key_cols = geo_cols + ["variable", "category"]
    dup_base = base.duplicated(subset=key_cols).sum()
    dup_earn = earn.duplicated(subset=key_cols).sum()
    if int(dup_base) > 0 or int(dup_earn) > 0:
        raise SystemExit(f"duplicate condition rows detected: base={int(dup_base)} earn={int(dup_earn)}")

    overlap = set(map(tuple, base[key_cols].astype(str).to_numpy().tolist())) & set(
        map(tuple, earn[key_cols].astype(str).to_numpy().tolist())
    )
    if overlap:
        raise SystemExit(f"base/earn condition overlap detected, example={list(overlap)[:3]}")

    merged = pd.concat([base, earn], axis=0, ignore_index=True)
    sort_cols = [c for c in ["statefp", "puma", "puma_uid", "tract_geoid", "cbg_geoid", "county_geoid", "variable", "category"] if c in merged.columns]
    merged = merged.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    merged.to_csv(out_path, index=False)

    merged_schema = _merge_condition_schemas(
        base_schema=_load_schema_sidecar(base_path),
        earn_schema=_load_schema_sidecar(earn_path),
        reference_schema=_load_schema_json(reference_schema_path),
        merged_df=merged,
    )

    meta = {
        "schema": "external_condition_v1_plus_earn_v1",
        "created_utc": _utc_now_iso(),
        "base_condition_csv": str(base_path),
        "earn_condition_csv": str(earn_path),
        "reference_schema_json": str(reference_schema_path) if reference_schema_path is not None else None,
        "out_path": str(out_path),
        "n_rows": int(merged.shape[0]),
        "merge_geo_cols": geo_cols,
        "n_geo_units": int(merged[geo_cols].astype(str).drop_duplicates().shape[0]),
        "variables": sorted(merged["variable"].astype(str).unique().tolist()),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".metadata.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    schema_path = out_path.with_suffix(out_path.suffix + ".schema.json")
    schema_path.write_text(json.dumps(merged_schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()
