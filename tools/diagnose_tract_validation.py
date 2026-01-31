#!/usr/bin/env python3
from __future__ import annotations

"""
Diagnose tract-level validation failures for Scheme B runs.

This script is intentionally KISS:
- It consumes existing run artifacts (stats_metrics_acs_tract.json) and raw inputs (samples_building.csv, buildings_csv).
- It outputs a small JSON that can be committed under outputs/<run_id>/metrics/.

Typical usage (workstation):
  python tools/diagnose_tract_validation.py \
    --run_dir "$OUT_DIR" \
    --buildings_csv "$BLDG_CSV" \
    --acs_targets_long_tract "$ACS_LONG_TRACT"
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


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_value_counts(series, *, max_items: int = 20) -> dict[str, int]:
    vc = series.value_counts(dropna=False)
    out: dict[str, int] = {}
    for k, v in vc.iloc[: max(1, int(max_items))].items():
        out[str(k)] = int(v)
    return out


def _age_bin_labels(*, edges: list[float]) -> list[str]:
    pd = _require("pandas")
    labels = []
    for i in range(len(edges) - 1):
        labels.append(str(pd.Interval(float(edges[i]), float(edges[i + 1]), closed="left")))
    return labels


def main() -> None:
    pd = _require("pandas")
    np = _require("numpy")

    p = argparse.ArgumentParser(prog="diagnose_tract_validation")
    p.add_argument("--run_dir", required=True, help="Workstation run dir that contains samples_building.csv and metrics/*.json.")
    p.add_argument("--buildings_csv", required=True, help="Buildings CSV used for the run (must include tract_geoid and ideally price_tier).")
    p.add_argument("--acs_targets_long_tract", required=True, help="Tract-level ACS targets_long CSV (group_col=tract_geoid).")
    p.add_argument(
        "--stats_metrics_acs_tract",
        default=None,
        help="Path to metrics/stats_metrics_acs_tract.json (default: <run_dir>/metrics/stats_metrics_acs_tract.json).",
    )
    p.add_argument(
        "--samples_csv",
        default=None,
        help="Path to samples_building.csv (default: <run_dir>/samples_building.csv).",
    )
    p.add_argument(
        "--out_json",
        default=None,
        help="Output JSON path (default: <run_dir>/metrics/tract_diagnosis.json).",
    )
    p.add_argument("--top_k", type=int, default=5, help="How many worst tracts to diagnose (default: 5).")
    p.add_argument("--min_samples_warn", type=int, default=50, help="Warn threshold for small synthetic sample size (default: 50).")
    args = p.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve()
    buildings_csv = pathlib.Path(args.buildings_csv).expanduser().resolve()
    acs_targets = pathlib.Path(args.acs_targets_long_tract).expanduser().resolve()

    stats_path = (
        pathlib.Path(args.stats_metrics_acs_tract).expanduser().resolve()
        if args.stats_metrics_acs_tract
        else (run_dir / "metrics" / "stats_metrics_acs_tract.json")
    )
    samples_csv = pathlib.Path(args.samples_csv).expanduser().resolve() if args.samples_csv else (run_dir / "samples_building.csv")
    out_json = pathlib.Path(args.out_json).expanduser().resolve() if args.out_json else (run_dir / "metrics" / "tract_diagnosis.json")

    if not stats_path.exists():
        raise SystemExit(f"stats_metrics_acs_tract not found: {stats_path}")
    if not samples_csv.exists():
        raise SystemExit(f"samples_building.csv not found: {samples_csv}")
    if not buildings_csv.exists():
        raise SystemExit(f"buildings_csv not found: {buildings_csv}")
    if not acs_targets.exists():
        raise SystemExit(f"acs_targets_long_tract not found: {acs_targets}")

    stats = _read_json(stats_path)
    tvd_age_by_tract = dict(stats.get("marginal_tvd", {}).get("AGEP_bin", {}).get("by_group", {}))
    tvd_sex_by_tract = dict(stats.get("marginal_tvd", {}).get("SEX", {}).get("by_group", {}))
    meta = dict(stats.get("meta", {}))
    age_edges = [float(x) for x in meta.get("bin_edges", {}).get("AGEP", [0, 5, 18, 25, 35, 45, 55, 65, 75, 85, 1000])]
    age_labels = _age_bin_labels(edges=age_edges)

    # Worst tracts by AGEP_bin TVD.
    items = [(str(k), float(v)) for k, v in tvd_age_by_tract.items()]
    items.sort(key=lambda kv: kv[1], reverse=True)
    worst_tracts = [k for k, _v in items[: max(1, int(args.top_k))]]

    # Load samples (only needed cols).
    usecols = ["tract_geoid", "AGEP", "bldg_id", "puma"]
    df_samples = pd.read_csv(samples_csv, usecols=lambda c: c in set(usecols), low_memory=False)
    if "tract_geoid" not in df_samples.columns:
        raise SystemExit("samples_csv missing tract_geoid (re-run PoC with buildings_csv including tract_geoid).")
    df_samples["tract_geoid"] = df_samples["tract_geoid"].astype(str)
    df_samples["AGEP"] = pd.to_numeric(df_samples.get("AGEP"), errors="coerce")
    df_samples = df_samples.dropna(subset=["tract_geoid", "AGEP"]).copy()

    # Load buildings.
    df_b = pd.read_csv(buildings_csv, low_memory=False)
    if "tract_geoid" not in df_b.columns:
        raise SystemExit("buildings_csv missing tract_geoid (build it with prepare_detroit_buildings_gba.py --tiger_tract_zip).")
    if "puma" not in df_b.columns:
        raise SystemExit("buildings_csv missing puma.")
    df_b["tract_geoid"] = df_b["tract_geoid"].astype(str)
    df_b["puma"] = df_b["puma"].astype(str)

    # Load ACS targets_long (tract-level).
    df_acs = pd.read_csv(acs_targets, low_memory=False)
    required_cols = {"tract_geoid", "variable", "category", "target"}
    if not required_cols.issubset(set(df_acs.columns)):
        raise SystemExit(f"acs_targets_long_tract missing columns: {sorted(required_cols - set(df_acs.columns))}")
    df_acs["tract_geoid"] = df_acs["tract_geoid"].astype(str)
    df_acs["variable"] = df_acs["variable"].astype(str)
    df_acs["category"] = df_acs["category"].astype(str)
    df_acs["target"] = pd.to_numeric(df_acs["target"], errors="coerce").fillna(0.0).clip(lower=0.0)

    def _syn_age_dist_for_tract(tract: str) -> dict[str, float]:
        sub = df_samples[df_samples["tract_geoid"] == tract]
        if sub.empty:
            return {}
        bins = pd.cut(sub["AGEP"], bins=age_edges, include_lowest=True, right=False).astype(str)
        p = bins.value_counts(normalize=True, dropna=False)
        return {k: float(p.get(k, 0.0)) for k in age_labels}

    def _acs_age_dist_for_tract(tract: str) -> dict[str, float]:
        t = df_acs[(df_acs["tract_geoid"] == tract) & (df_acs["variable"] == "AGEP_bin")].copy()
        if t.empty:
            return {}
        total = float(t["target"].sum())
        if total <= 0:
            return {}
        p = (t.groupby("category", sort=False)["target"].sum() / total).astype(float)
        return {k: float(p.get(k, 0.0)) for k in age_labels}

    def _syn_sex_dist_for_tract(tract: str) -> dict[str, float]:
        if "SEX" not in df_samples.columns:
            return {}
        sub = df_samples[df_samples["tract_geoid"] == tract]
        if sub.empty:
            return {}
        p = sub["SEX"].astype(str).value_counts(normalize=True, dropna=False)
        cats = sorted(p.index.astype(str).tolist())
        return {k: float(p.get(k, 0.0)) for k in cats}

    def _acs_sex_dist_for_tract(tract: str) -> dict[str, float]:
        t = df_acs[(df_acs["tract_geoid"] == tract) & (df_acs["variable"] == "SEX")].copy()
        if t.empty:
            return {}
        total = float(t["target"].sum())
        if total <= 0:
            return {}
        p = (t.groupby("category", sort=False)["target"].sum() / total).astype(float)
        cats = sorted(p.index.astype(str).tolist())
        return {k: float(p.get(k, 0.0)) for k in cats}

    def _prob_gap(p_syn: dict[str, float], p_tgt: dict[str, float]) -> dict[str, float]:
        keys = sorted(set(p_syn.keys()) | set(p_tgt.keys()))
        return {k: float(p_syn.get(k, 0.0) - p_tgt.get(k, 0.0)) for k in keys}

    worst_diag: dict[str, Any] = {}
    for tract in worst_tracts:
        sub_syn = df_samples[df_samples["tract_geoid"] == tract]
        n_syn = int(sub_syn.shape[0])
        sub_b = df_b[df_b["tract_geoid"] == tract]
        n_bldg = int(sub_b.shape[0])

        p_syn_age = _syn_age_dist_for_tract(tract)
        p_tgt_age = _acs_age_dist_for_tract(tract)
        age_gap = _prob_gap(p_syn_age, p_tgt_age) if (p_syn_age and p_tgt_age) else {}

        p_syn_sex = _syn_sex_dist_for_tract(tract)
        p_tgt_sex = _acs_sex_dist_for_tract(tract)
        sex_gap = _prob_gap(p_syn_sex, p_tgt_sex) if (p_syn_sex and p_tgt_sex) else {}

        likely: list[str] = []
        if n_syn < int(args.min_samples_warn):
            likely.append("样本量过小（高方差）")
        if n_bldg <= 0:
            likely.append("该tract无建筑或未被建筑数据覆盖")
        if "price_tier" in sub_b.columns and n_bldg > 0:
            tiers = pd.to_numeric(sub_b["price_tier"], errors="coerce").dropna().astype(int)
            if not tiers.empty:
                top_frac = float(tiers.value_counts(normalize=True).iloc[0])
                if top_frac >= 0.9:
                    likely.append("建筑price_tier极端集中（空间经济结构单一）")

        worst_diag[tract] = {
            "tvd_agep_bin": float(tvd_age_by_tract.get(tract, float("nan"))),
            "tvd_sex": float(tvd_sex_by_tract.get(tract, float("nan"))) if tract in tvd_sex_by_tract else None,
            "n_synthetic": n_syn,
            "n_buildings": n_bldg,
            "price_tier_dist": (_safe_value_counts(sub_b["price_tier"].astype(str)) if "price_tier" in sub_b.columns else {}),
            "agep_bin_gap": age_gap,
            "sex_gap": sex_gap,
            "likely_cause": " + ".join(likely) if likely else None,
        }

    # TVD by PUMA (using tract->puma mapping from buildings).
    tract_to_puma = df_b.groupby("tract_geoid", sort=False)["puma"].first().to_dict()
    tvd_by_puma: dict[str, Any] = {}
    for tract, tvd in tvd_age_by_tract.items():
        puma = tract_to_puma.get(str(tract))
        if puma is None:
            continue
        tvd_by_puma.setdefault(str(puma), {})[str(tract)] = float(tvd)

    puma_summary: dict[str, Any] = {}
    for puma, tract_map in tvd_by_puma.items():
        tvds = np.array(list(tract_map.values()), dtype=float)
        if tvds.size == 0:
            continue
        worst_t = max(tract_map, key=lambda k: tract_map[k])
        puma_summary[str(puma)] = {
            "n_tracts": int(tvds.size),
            "tvd_mean": float(tvds.mean()),
            "tvd_median": float(np.median(tvds)),
            "tvd_p90": float(np.quantile(tvds, 0.90)),
            "worst_tract": str(worst_t),
            "worst_tvd": float(tract_map[worst_t]),
        }

    # Global diagnostics: how much is driven by small-n tracts?
    syn_counts = df_samples["tract_geoid"].value_counts()
    tvd_pairs = [(t, float(v)) for t, v in tvd_age_by_tract.items() if t in syn_counts.index.astype(str)]
    xs = np.array([float(syn_counts.get(t, 0)) for t, _v in tvd_pairs], dtype=float)
    ys = np.array([v for _t, v in tvd_pairs], dtype=float)
    corr = None
    if xs.size >= 3 and float(xs.std()) > 0 and float(ys.std()) > 0:
        corr = float(np.corrcoef(xs, ys)[0, 1])

    out = {
        "meta": {
            "run_dir": str(run_dir),
            "stats_metrics_acs_tract": str(stats_path),
            "samples_csv": str(samples_csv),
            "buildings_csv": str(buildings_csv),
            "acs_targets_long_tract": str(acs_targets),
            "top_k": int(args.top_k),
            "min_samples_warn": int(args.min_samples_warn),
            "n_tracts_eval": int(len(tvd_age_by_tract)),
            "n_tracts_samples": int(df_samples["tract_geoid"].nunique(dropna=False)),
            "n_tracts_buildings": int(df_b["tract_geoid"].nunique(dropna=False)),
            "corr_n_synthetic_vs_tvd": corr,
        },
        "worst_tracts": worst_tracts,
        "worst_tracts_diagnosis": worst_diag,
        "tvd_by_puma": puma_summary,
    }

    _write_json(out_json, out)
    print(f"[ok] wrote: {out_json}")


if __name__ == "__main__":
    main()

