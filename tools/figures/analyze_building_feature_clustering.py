#!/usr/bin/env python3
from __future__ import annotations

"""
Analyze whether building features are "clustered" by tract (or other group).

Motivation (KISS, Scheme B diagnostics):
- If buildings within the same tract have similar features (high between-group ratio),
  then tract-level structure exists in the building feature space.
- This is a necessary (but not sufficient) condition for any "conditioning on building features"
  approach to potentially help tract-level validation.

Output:
- A small JSON with per-feature total variance, within-group variance, and between_ratio.
  between_ratio is analogous to a 1-way ANOVA / eta^2:
    between_ratio = 1 - (within_var / total_var)
  where within_var is the pooled within-group variance (ddof=0).
"""

import argparse
import json
import pathlib
from typing import Any


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as e:
        raise RuntimeError(f"Missing dependency: {pkg}. Install it in your conda env.") from e


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _analyze_one_feature(df, *, group_col: str, feat_col: str) -> dict[str, Any]:
    pd = _require("pandas")
    np = _require("numpy")

    g = df[group_col].astype(str)
    x = pd.to_numeric(df[feat_col], errors="coerce")
    m = (~x.isna()) & (~g.isna()) & (g != "") & (g != "nan")
    g = g[m]
    x = x[m].astype(float)

    n = int(x.shape[0])
    if n < 2:
        return {"n": n, "n_groups": int(g.nunique(dropna=False)), "total_var": None, "within_var": None, "between_ratio": None}

    # Total variance (population, ddof=0).
    mu = float(x.mean())
    sst = float(((x - mu) ** 2).sum())
    total_var = float(sst / n) if sst >= 0 else None

    # Pooled within-group variance (population, ddof=0).
    mean_g = x.groupby(g, sort=False).transform("mean")
    ssw = float(((x - mean_g) ** 2).sum())
    within_var = float(ssw / n) if ssw >= 0 else None

    between_ratio = None
    if total_var is not None and total_var > 0 and within_var is not None:
        between_ratio = float(max(0.0, min(1.0, 1.0 - within_var / total_var)))

    # Helpful group-size stats for interpretation.
    sizes = g.value_counts(dropna=False)
    sizes_v = sizes.to_numpy(dtype=float)
    size_stats = {
        "n_groups": int(sizes.shape[0]),
        "min": int(sizes.min()) if not sizes.empty else None,
        "p10": int(np.quantile(sizes_v, 0.10)) if sizes_v.size else None,
        "p50": int(np.quantile(sizes_v, 0.50)) if sizes_v.size else None,
        "p90": int(np.quantile(sizes_v, 0.90)) if sizes_v.size else None,
        "max": int(sizes.max()) if not sizes.empty else None,
    }

    return {
        "n": n,
        "total_var": total_var,
        "within_var": within_var,
        "between_ratio": between_ratio,
        "group_size": size_stats,
    }


def main() -> None:
    pd = _require("pandas")

    p = argparse.ArgumentParser(prog="analyze_building_feature_clustering")
    p.add_argument("--buildings_csv", required=True, help="Buildings CSV (must include group_col).")
    p.add_argument("--group_col", default="tract_geoid", help='Grouping column (default: "tract_geoid").')
    p.add_argument(
        "--features",
        default="price_tier,dist_cbd_km,footprint_area_m2,height_m,cap_proxy",
        help="Comma-separated numeric features to analyze.",
    )
    p.add_argument(
        "--log1p_features",
        default="footprint_area_m2,cap_proxy",
        help="Comma-separated features to also analyze under log1p(x) (empty to disable).",
    )
    p.add_argument("--out_json", default=None, help="Output JSON path (default: <buildings_csv>.tract_clustering.json).")
    args = p.parse_args()

    buildings_csv = pathlib.Path(args.buildings_csv).expanduser().resolve()
    if not buildings_csv.exists():
        raise SystemExit(f"buildings_csv not found: {buildings_csv}")

    df = pd.read_csv(buildings_csv, low_memory=False)
    group_col = str(args.group_col)
    if group_col not in df.columns:
        raise SystemExit(f"buildings_csv missing group_col={group_col}. Columns: {list(df.columns)[:30]}")

    feats = [f.strip() for f in str(args.features).split(",") if f.strip()]
    if not feats:
        raise SystemExit("--features cannot be empty")

    log1p_feats = [f.strip() for f in str(args.log1p_features).split(",") if f.strip()]

    out_json = pathlib.Path(args.out_json).expanduser().resolve() if args.out_json else buildings_csv.with_suffix(".tract_clustering.json")

    results: dict[str, Any] = {}
    for feat in feats:
        if feat not in df.columns:
            results[feat] = {"error": f"missing column: {feat}"}
            continue
        results[feat] = _analyze_one_feature(df, group_col=group_col, feat_col=feat)

    for feat in log1p_feats:
        if feat not in df.columns:
            continue
        col = f"{feat}__log1p"
        df[col] = pd.to_numeric(df[feat], errors="coerce")
        df[col] = df[col].clip(lower=0.0)
        df[col] = df[col].apply(lambda v: None if pd.isna(v) else float(__import__("math").log1p(float(v))))
        results[col] = _analyze_one_feature(df, group_col=group_col, feat_col=col)

    meta = {
        "buildings_csv": str(buildings_csv),
        "group_col": group_col,
        "n_buildings": int(df.shape[0]),
        "features": feats,
        "log1p_features": log1p_feats,
    }

    _write_json(out_json, {"meta": meta, "results": results})
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()

