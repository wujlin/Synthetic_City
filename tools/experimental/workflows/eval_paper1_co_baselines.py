#!/usr/bin/env python3
from __future__ import annotations

"""
Evaluate candidate-pool combinatorial-optimization baselines for Paper 1.

The manuscript compares regional joint distributions over a fixed cell schema.
For that target, individual-level CO can be evaluated after aggregating PUMS
records into the same cells: all candidates in the same cell are equivalent for
TVD. We therefore report the entropy-balanced continuous relaxation, plus an
integerized version at the target PUMA population size.

Variants:
- CO-local: candidate pool is the target PUMA PUMS distribution. This is an
  oracle information setting because target microdata are available.
- CO-national: candidate pool is the non-heldout national PUMS distribution,
  aggregated with PUMS person weights. This is the fair held-out comparison.
"""

import argparse
import datetime as _dt
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.model.train_external_joint_hier_diffusion_full as stage1_base
from tools.model.train_us_puma_5var_diffusion import (
    _canon_puma5,
    _canon_statefp,
    _canon_uid,
    _cosine,
    _ipf_nd,
    _nd_independence,
    _summ,
    _tvd,
    _utc_now_iso,
    _write_json,
)
from tools.model.train_us_puma_external_v1_diffusion import (
    _load_condition_specs_from_schema,
    _load_external_condition_matrix,
    _load_var_specs_from_schema,
)


def _load_schema(schema_json: pathlib.Path) -> tuple[list[str], tuple[int, ...], int]:
    schema = json.loads(schema_json.read_text(encoding="utf-8"))
    variable_order = [str(x) for x in schema.get("variable_order", [])]
    shape = tuple(int(x) for x in schema.get("shape", []))
    if not variable_order or not shape:
        raise SystemExit(f"Invalid schema_json: {schema_json}")
    if len(variable_order) != len(shape):
        raise SystemExit(f"schema variable_order/shape mismatch: {schema_json}")
    k = int(np.prod(shape))
    schema_k = schema.get("K", schema.get("target_dim", k))
    if schema_k is not None and int(schema_k) != k:
        raise SystemExit(f"schema K mismatch: schema={schema_k}, shape_product={k}")
    return variable_order, shape, k


def _load_joint_wide(*, joint_wide_csv: pathlib.Path, schema_json: pathlib.Path) -> tuple[pd.DataFrame, np.ndarray, list[str], int]:
    _, _, k = _load_schema(schema_json)
    df = pd.read_csv(joint_wide_csv, low_memory=False)
    req = {"statefp", "puma", "puma_uid", "total_person_weight"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"joint_wide_csv missing columns: {miss}")
    p_cols = [f"p_joint_{i:03d}" for i in range(k)]
    miss_p = [c for c in p_cols if c not in df.columns]
    if miss_p:
        raise SystemExit(f"joint_wide_csv missing joint columns: {miss_p[:5]}")

    df["statefp"] = df["statefp"].map(_canon_statefp)
    df["puma5"] = df["puma"].map(_canon_puma5)
    df["puma_uid"] = df.apply(lambda r: _canon_uid(r["statefp"], r["puma5"]), axis=1)
    df["total_person_weight"] = pd.to_numeric(df["total_person_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)

    p = df[p_cols].to_numpy(dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return df, p, df["puma_uid"].astype(str).tolist(), k


def _run_ipf(*, seed: np.ndarray, marginals: list[np.ndarray], shape: tuple[int, ...], ipf_iters: int) -> np.ndarray:
    out = _ipf_nd(
        seed_joint=np.asarray(seed, dtype=np.float64).reshape(shape),
        target_marginals=[np.asarray(x, dtype=np.float64) for x in marginals],
        shape=shape,
        max_iter=int(ipf_iters),
    ).reshape(-1)
    out = np.clip(out, 0.0, None)
    out = out / max(float(out.sum()), 1e-12)
    return out


def _integerize_prob(p: np.ndarray, n: int) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.clip(p, 0.0, None)
    p = p / max(float(p.sum()), 1e-12)
    n = max(int(n), 1)
    raw = p * float(n)
    counts = np.floor(raw).astype(np.int64)
    rem = int(n - int(counts.sum()))
    if rem > 0:
        frac = raw - counts
        add_idx = np.argpartition(-frac, kth=min(rem, frac.size - 1))[:rem]
        counts[add_idx] += 1
    out = counts.astype(np.float64) / float(n)
    out = out / max(float(out.sum()), 1e-12)
    return out


def _marginal_gap(p: np.ndarray, marginals: list[np.ndarray], shape: tuple[int, ...]) -> float:
    tab = np.asarray(p, dtype=np.float64).reshape(shape)
    gaps: list[float] = []
    axes = tuple(range(len(shape)))
    for axis, target in enumerate(marginals):
        cur = tab.sum(axis=tuple(a for a in axes if a != axis))
        cur = cur / max(float(cur.sum()), 1e-12)
        tgt = np.asarray(target, dtype=np.float64).reshape(-1)
        tgt = np.clip(tgt, 0.0, None)
        tgt = tgt / max(float(tgt.sum()), 1e-12)
        gaps.append(_tvd(cur, tgt))
    return float(np.mean(gaps)) if gaps else 0.0


def _summary_from_rows(rows: list[dict[str, Any]], col: str) -> dict[str, float] | None:
    return _summ([float(r[col]) for r in rows if pd.notna(r.get(col))])


def main() -> None:
    ap = argparse.ArgumentParser(prog="eval_paper1_co_baselines")
    ap.add_argument("--joint_wide_csv", required=True)
    ap.add_argument("--schema_json", required=True)
    ap.add_argument("--condition_csv", required=True)
    ap.add_argument("--condition_schema_json", default=None)
    ap.add_argument("--heldout_statefp", default="26")
    ap.add_argument("--ipf_iters", type=int, default=500)
    ap.add_argument("--reference_by_puma_csv", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    joint_wide_csv = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    schema_json = pathlib.Path(args.schema_json).expanduser().resolve()
    condition_csv = pathlib.Path(args.condition_csv).expanduser().resolve()
    condition_schema_json = pathlib.Path(args.condition_schema_json).expanduser().resolve() if args.condition_schema_json else None
    reference_by_puma_csv = pathlib.Path(args.reference_by_puma_csv).expanduser().resolve() if args.reference_by_puma_csv else None
    for p in [joint_wide_csv, schema_json, condition_csv]:
        if not p.exists():
            raise SystemExit(f"Required path not found: {p}")
    if condition_schema_json is not None and not condition_schema_json.exists():
        raise SystemExit(f"condition_schema_json not found: {condition_schema_json}")
    if reference_by_puma_csv is not None and not reference_by_puma_csv.exists():
        raise SystemExit(f"reference_by_puma_csv not found: {reference_by_puma_csv}")

    variable_order, shape, k = _load_schema(schema_json)
    df, p_true_all, ids, _ = _load_joint_wide(joint_wide_csv=joint_wide_csv, schema_json=schema_json)
    heldout_statefp = _canon_statefp(args.heldout_statefp)
    is_heldout = (df["statefp"] == heldout_statefp).to_numpy(dtype=bool)
    heldout_idx = np.where(is_heldout)[0]
    train_idx = np.where(~is_heldout)[0]
    if heldout_idx.size == 0:
        raise SystemExit(f"No held-out PUMAs found for statefp={heldout_statefp}")
    if train_idx.size == 0:
        raise SystemExit(f"No training PUMAs after holding out statefp={heldout_statefp}")

    var_specs = _load_var_specs_from_schema(schema_json=schema_json)
    cond_specs = _load_condition_specs_from_schema(
        condition_schema_json=condition_schema_json,
        fallback_var_specs=var_specs,
    )
    cond_raw, block_slices, cond_meta = _load_external_condition_matrix(
        condition_csv=condition_csv,
        ids=ids,
        var_specs=cond_specs,
    )
    ext_marg = {var: cond_raw[:, sl].copy() for var, sl in block_slices.items() if var in variable_order}
    ext_marg = stage1_base._augment_ext_marginals_from_cross(
        cond_raw=cond_raw,
        block_slices=block_slices,
        ext_marg=ext_marg,
    )
    missing_marg = [v for v in variable_order if v not in ext_marg]
    if missing_marg:
        raise SystemExit(f"Missing condition marginals for variables: {missing_marg}")

    weights = df["total_person_weight"].to_numpy(dtype=np.float64)
    train_weights = np.clip(weights[train_idx], 0.0, None)
    if float(train_weights.sum()) <= 0:
        train_weights = np.ones_like(train_weights, dtype=np.float64)
    seed_national_weighted = (p_true_all[train_idx] * train_weights.reshape(-1, 1)).sum(axis=0)
    seed_national_weighted = seed_national_weighted / max(float(seed_national_weighted.sum()), 1e-12)

    seed_puma_mean = p_true_all[train_idx].mean(axis=0)
    seed_puma_mean = seed_puma_mean / max(float(seed_puma_mean.sum()), 1e-12)

    rows: list[dict[str, Any]] = []
    for idx in heldout_idx.tolist():
        p_true = np.asarray(p_true_all[idx], dtype=np.float64)
        target_marginals = [np.asarray(ext_marg[var][idx], dtype=np.float64) for var in variable_order]
        n_pop = int(round(float(weights[idx])))

        p_local = _run_ipf(seed=p_true, marginals=target_marginals, shape=shape, ipf_iters=int(args.ipf_iters))
        p_local_int = _integerize_prob(p_local, n=n_pop)
        p_nat = _run_ipf(seed=seed_national_weighted, marginals=target_marginals, shape=shape, ipf_iters=int(args.ipf_iters))
        p_nat_int = _integerize_prob(p_nat, n=n_pop)
        p_ipf_mean = _run_ipf(seed=seed_puma_mean, marginals=target_marginals, shape=shape, ipf_iters=int(args.ipf_iters))
        p_ind = _nd_independence(target_marginals)

        rows.append(
            {
                "puma_uid": str(df.iloc[idx]["puma_uid"]),
                "statefp": str(df.iloc[idx]["statefp"]),
                "puma5": str(df.iloc[idx]["puma5"]),
                "total_person_weight": float(weights[idx]),
                "tvd_co_local": float(_tvd(p_local, p_true)),
                "tvd_co_local_integerized": float(_tvd(p_local_int, p_true)),
                "tvd_co_national": float(_tvd(p_nat, p_true)),
                "tvd_co_national_integerized": float(_tvd(p_nat_int, p_true)),
                "tvd_ipf_puma_mean_seed": float(_tvd(p_ipf_mean, p_true)),
                "tvd_independence": float(_tvd(p_ind, p_true)),
                "cosine_co_local": float(_cosine(p_local, p_true)),
                "cosine_co_national": float(_cosine(p_nat, p_true)),
                "mean_marginal_gap_co_local": _marginal_gap(p_local, target_marginals, shape),
                "mean_marginal_gap_co_national": _marginal_gap(p_nat, target_marginals, shape),
                "mean_marginal_gap_ipf_puma_mean_seed": _marginal_gap(p_ipf_mean, target_marginals, shape),
            }
        )

    out_dir = (
        pathlib.Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else _REPO_ROOT / "outputs" / f"_paper1_CO_baselines_state{heldout_statefp}_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    by_puma = pd.DataFrame(rows)
    if reference_by_puma_csv is not None:
        ref = pd.read_csv(reference_by_puma_csv)
        if "puma_uid" in ref.columns:
            ref["puma_uid"] = ref["puma_uid"].map(lambda x: str(x).zfill(7))
            by_puma = by_puma.merge(ref, on="puma_uid", how="left", suffixes=("", "_reference"))
    by_puma.to_csv(metrics_dir / "co_baselines_by_puma.csv", index=False)

    summary = {
        "created_utc": _utc_now_iso(),
        "experiment": "paper1_candidate_pool_co_baselines",
        "interpretation": {
            "co_local": "Oracle candidate-pool baseline: target PUMA PUMS joint is used as the seed/support before fitting ACS marginals.",
            "co_national": "Fair held-out candidate-pool baseline: non-heldout national PUMS joint, weighted by PUMS person weights, is used as the seed/support.",
            "integerized": "Largest-remainder integerization at target PUMA total_person_weight; included to approximate finite synthetic population counts.",
        },
        "inputs": {
            "joint_wide_csv": str(joint_wide_csv),
            "schema_json": str(schema_json),
            "condition_csv": str(condition_csv),
            "condition_schema_json": str(condition_schema_json) if condition_schema_json else None,
            "reference_by_puma_csv": str(reference_by_puma_csv) if reference_by_puma_csv else None,
        },
        "heldout_statefp": str(heldout_statefp),
        "n_heldout_pumas": int(heldout_idx.size),
        "n_train_pumas": int(train_idx.size),
        "schema": {
            "variable_order": variable_order,
            "shape": list(shape),
            "K": int(k),
        },
        "condition_meta": cond_meta,
        "ipf_iters": int(args.ipf_iters),
        "summaries": {
            "co_local": {
                "tvd_joint": _summary_from_rows(rows, "tvd_co_local"),
                "cosine_joint": _summary_from_rows(rows, "cosine_co_local"),
                "mean_marginal_gap": _summary_from_rows(rows, "mean_marginal_gap_co_local"),
            },
            "co_local_integerized": {
                "tvd_joint": _summary_from_rows(rows, "tvd_co_local_integerized"),
            },
            "co_national": {
                "tvd_joint": _summary_from_rows(rows, "tvd_co_national"),
                "cosine_joint": _summary_from_rows(rows, "cosine_co_national"),
                "mean_marginal_gap": _summary_from_rows(rows, "mean_marginal_gap_co_national"),
            },
            "co_national_integerized": {
                "tvd_joint": _summary_from_rows(rows, "tvd_co_national_integerized"),
            },
            "ipf_puma_mean_seed": {
                "tvd_joint": _summary_from_rows(rows, "tvd_ipf_puma_mean_seed"),
                "mean_marginal_gap": _summary_from_rows(rows, "mean_marginal_gap_ipf_puma_mean_seed"),
            },
            "independence": {
                "tvd_joint": _summary_from_rows(rows, "tvd_independence"),
            },
        },
    }

    if reference_by_puma_csv is not None:
        ref_cols = [
            "pipeline_tvd_mean",
            "ipf_tvd",
            "one_shot_tvd_mean",
            "delta_tvd_ipf_minus_pipeline",
        ]
        ref_summary = {}
        for c in ref_cols:
            if c in by_puma.columns:
                vals = pd.to_numeric(by_puma[c], errors="coerce").dropna().astype(float).tolist()
                ref_summary[c] = _summ(vals)
        summary["reference_summaries"] = ref_summary

        paired_specs = {
            "co_national_minus_hierarchical": ("tvd_co_national", "pipeline_tvd_mean"),
            "co_national_minus_ipf": ("tvd_co_national", "ipf_tvd"),
            "co_local_minus_hierarchical": ("tvd_co_local", "pipeline_tvd_mean"),
            "co_local_minus_ipf": ("tvd_co_local", "ipf_tvd"),
        }
        paired = {}
        for name, (a_col, b_col) in paired_specs.items():
            if a_col not in by_puma.columns or b_col not in by_puma.columns:
                continue
            a = pd.to_numeric(by_puma[a_col], errors="coerce")
            b = pd.to_numeric(by_puma[b_col], errors="coerce")
            ok = a.notna() & b.notna()
            vals = (a[ok].astype(float) - b[ok].astype(float)).tolist()
            paired[name] = _summ(vals)
        summary["paired_deltas"] = paired

    _write_json(metrics_dir / "co_baselines_summary.json", summary)
    _write_json(out_dir / "run_summary.json", summary)
    print(f"[ok] wrote: {out_dir}")
    print(json.dumps(summary["summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
