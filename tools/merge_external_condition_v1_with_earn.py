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


def main() -> None:
    ap = argparse.ArgumentParser(prog="merge_external_condition_v1_with_earn")
    ap.add_argument("--base_condition_csv", required=True)
    ap.add_argument("--earn_condition_csv", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base_path = pathlib.Path(args.base_condition_csv).expanduser().resolve()
    earn_path = pathlib.Path(args.earn_condition_csv).expanduser().resolve()
    out_path = pathlib.Path(args.out_path).expanduser().resolve()
    if not base_path.exists():
        raise SystemExit(f"base_condition_csv not found: {base_path}")
    if not earn_path.exists():
        raise SystemExit(f"earn_condition_csv not found: {earn_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"out_path exists: {out_path} (use --overwrite)")

    base = pd.read_csv(base_path, low_memory=False)
    earn = pd.read_csv(earn_path, low_memory=False)
    need = {"statefp", "puma", "puma_uid", "variable", "category", "target"}
    for name, df in [("base", base), ("earn", earn)]:
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise SystemExit(f"{name} condition missing columns: {miss}")

    key_cols = ["puma_uid", "variable", "category"]
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
    merged = merged.sort_values(["statefp", "puma", "variable", "category"], kind="stable").reset_index(drop=True)
    merged.to_csv(out_path, index=False)

    meta = {
        "schema": "external_condition_v1_plus_earn_v1",
        "created_utc": _utc_now_iso(),
        "base_condition_csv": str(base_path),
        "earn_condition_csv": str(earn_path),
        "out_path": str(out_path),
        "n_rows": int(merged.shape[0]),
        "n_pumas": int(merged["puma_uid"].astype(str).nunique()),
        "variables": sorted(merged["variable"].astype(str).unique().tolist()),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".metadata.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()
