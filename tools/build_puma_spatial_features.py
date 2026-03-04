#!/usr/bin/env python3
from __future__ import annotations

"""
Build PUMA-level spatial feature table for diffusion conditions.

Design choices:
- Unified neighborhood definition: KNN by centroid distance in EPSG:3857.
- No fallback neighborhood mechanism is used.
- Neighbor summaries are computed from the marginal vectors in joint_wide_csv.

Example:
    python tools/build_puma_spatial_features.py \
      --puma_zip /path/to/tl_2023_us_puma10.zip \
      --joint_wide_csv /path/to/puma_b19037_joint_wide.csv \
      --out_csv /path/to/puma_spatial_features_b19037.csv \
      --knn_k 6 \
      --pe_levels 8
"""

import argparse
import datetime as _dt
import json
import pathlib
import sys
from typing import Sequence

import numpy as np
import pandas as pd


def _find_col(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    mp = {c.upper(): c for c in columns}
    for cand in candidates:
        c = mp.get(str(cand).upper())
        if c is not None:
            return c
    return None


def _sorted_bin_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    if not cols:
        return []

    def _key(c: str) -> tuple[int, str]:
        try:
            return (int(str(c).split("_")[-1]), str(c))
        except Exception:
            return (10**9, str(c))

    return sorted(cols, key=_key)


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_puma(v: str) -> str:
    s = str(v).strip()
    if s.isdigit():
        return str(int(s))
    return s


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_puma_spatial_features")
    ap.add_argument("--puma_zip", required=True, help="TIGER PUMA20 zip (US-wide recommended).")
    ap.add_argument("--joint_wide_csv", required=True, help="joint_wide CSV used by training.")
    ap.add_argument("--out_csv", required=True, help="Output CSV path.")
    ap.add_argument("--knn_k", type=int, default=6, help="K for centroid KNN (default: 6).")
    ap.add_argument("--pe_levels", type=int, default=8, help="Sinusoidal PE levels L (default: 8).")
    ap.add_argument("--epsg_metric", type=int, default=3857, help="Projected CRS EPSG for metric distance.")
    args = ap.parse_args()

    puma_zip = pathlib.Path(args.puma_zip).expanduser().resolve()
    joint_csv = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    out_csv = pathlib.Path(args.out_csv).expanduser().resolve()
    out_meta = out_csv.with_suffix(out_csv.suffix + ".metadata.json")

    if str(args.puma_zip).strip() == "":
        raise SystemExit("--puma_zip is empty. Please set a valid TIGER PUMA zip path.")
    if not puma_zip.exists():
        raise SystemExit(f"puma_zip not found: {puma_zip}")
    if not puma_zip.is_file():
        raise SystemExit(f"puma_zip must be a .zip file, got: {puma_zip}")
    if puma_zip.suffix.lower() != ".zip":
        raise SystemExit(f"puma_zip must end with .zip, got: {puma_zip.name}")
    if not joint_csv.exists():
        raise SystemExit(f"joint_wide_csv not found: {joint_csv}")
    if int(args.knn_k) <= 0:
        raise SystemExit("--knn_k must be positive.")
    if int(args.pe_levels) <= 0:
        raise SystemExit("--pe_levels must be positive.")

    import geopandas as gpd

    print(f"[info] loading puma shapefile: {puma_zip}", file=sys.stderr)
    g = gpd.read_file(f"zip://{puma_zip}")

    state_col = _find_col(g.columns, ["STATEFP20", "STATEFP"])
    puma_col = _find_col(g.columns, ["PUMACE20", "PUMACE"])
    if state_col is None or puma_col is None:
        raise SystemExit(
            "Cannot locate STATEFP/PUMA columns in puma shapefile. "
            f"columns={list(g.columns)}"
        )

    g = g[[state_col, puma_col, "geometry"]].copy()
    g["statefp"] = g[state_col].astype(str).str.zfill(2)
    g["puma"] = g[puma_col].astype(str).str.zfill(5).map(_normalize_puma)
    g["puma_uid"] = g["statefp"] + g[puma_col].astype(str).str.zfill(5)

    # Metric projection for centroid distance and geometry shape metrics.
    g = g.to_crs(epsg=int(args.epsg_metric))
    cent = g.geometry.centroid
    g["centroid_x"] = cent.x.astype(float)
    g["centroid_y"] = cent.y.astype(float)
    area_m2 = g.geometry.area.astype(float)
    peri_m = g.geometry.length.astype(float)
    g["area_km2"] = area_m2 / 1_000_000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        compact = (4.0 * np.pi * area_m2) / np.maximum(peri_m * peri_m, 1e-12)
    g["compactness"] = np.where(np.isfinite(compact), compact, 0.0).astype(float)

    # Keep one row per puma_uid.
    g = g.drop_duplicates(subset=["puma_uid"], keep="first").reset_index(drop=True)

    print(f"[info] loading joint_wide csv: {joint_csv}", file=sys.stderr)
    j = pd.read_csv(joint_csv)
    required = {"puma_uid", "statefp", "puma"}
    missing_req = [c for c in required if c not in j.columns]
    if missing_req:
        raise SystemExit(f"joint_wide csv missing columns: {missing_req}")

    # Marginal columns in fixed order.
    marg_cols: list[str] = []
    for pref in ["p_age_", "p_sex_", "p_income_", "p_schl_", "p_esr_"]:
        marg_cols.extend(_sorted_bin_cols(j, pref))
    if not marg_cols:
        raise SystemExit("No marginal columns found in joint_wide csv (expected p_* prefixes).")

    j = j[["puma_uid", "statefp", "puma"] + marg_cols].copy()
    j["puma_uid"] = j["puma_uid"].astype(str)
    j = j.drop_duplicates(subset=["puma_uid"], keep="first").reset_index(drop=True)

    d = j.merge(
        g[["puma_uid", "centroid_x", "centroid_y", "area_km2", "compactness"]],
        on="puma_uid",
        how="left",
    )
    if d[["centroid_x", "centroid_y"]].isna().any().any():
        bad = d.loc[d["centroid_x"].isna() | d["centroid_y"].isna(), "puma_uid"].tolist()
        raise SystemExit(f"Missing geometry for {len(bad)} pumas in joint_wide csv. Example={bad[:5]}")

    n = int(d.shape[0])
    k = int(args.knn_k)
    if n <= k:
        raise SystemExit(f"Not enough rows for knn_k={k}: n={n}")

    xy = d[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float64)
    marg = d[marg_cols].to_numpy(dtype=np.float64)

    # Full pairwise distances (n=2456 => manageable), then take top-k nearest.
    dx = xy[:, None, 0] - xy[None, :, 0]
    dy = xy[:, None, 1] - xy[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy, dtype=np.float64)
    np.fill_diagonal(dist, np.inf)
    nn_idx = np.argsort(dist, axis=1)[:, :k]  # (n, k)
    nn_dist = np.take_along_axis(dist, nn_idx, axis=1)  # (n, k)

    inv = 1.0 / np.maximum(nn_dist, 1e-6)
    w = inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-12)

    neigh1_avg = np.zeros((n, marg.shape[1]), dtype=np.float64)
    neigh_std_mean = np.zeros((n,), dtype=np.float64)
    neigh_std_max = np.zeros((n,), dtype=np.float64)
    neigh2_avg = np.zeros((n, marg.shape[1]), dtype=np.float64)

    for i in range(n):
        n1 = nn_idx[i]
        neigh1_avg[i] = np.sum(w[i][:, None] * marg[n1], axis=0)
        std_vec = np.std(marg[n1], axis=0, ddof=0)
        neigh_std_mean[i] = float(np.mean(std_vec))
        neigh_std_max[i] = float(np.max(std_vec))

        # 2-hop from KNN graph: union(neighbors of 1-hop) \ {self, 1-hop}
        n2_set: set[int] = set()
        for j_idx in n1:
            n2_set.update(nn_idx[int(j_idx)].tolist())
        n2_set.discard(i)
        for x in n1.tolist():
            n2_set.discard(int(x))
        if not n2_set:
            raise SystemExit(f"Empty 2-hop neighborhood for row={i}, puma_uid={d.iloc[i]['puma_uid']}")
        n2 = np.array(sorted(n2_set), dtype=int)
        d2 = dist[i, n2]
        inv2 = 1.0 / np.maximum(d2, 1e-6)
        w2 = inv2 / np.maximum(inv2.sum(), 1e-12)
        neigh2_avg[i] = np.sum(w2[:, None] * marg[n2], axis=0)

    # Global z-score for centroid raw and PE base.
    cx = d["centroid_x"].to_numpy(dtype=np.float64)
    cy = d["centroid_y"].to_numpy(dtype=np.float64)
    cx_mu, cx_sd = float(np.mean(cx)), float(np.std(cx, ddof=0))
    cy_mu, cy_sd = float(np.mean(cy)), float(np.std(cy, ddof=0))
    cx_sd = cx_sd if cx_sd > 1e-12 else 1.0
    cy_sd = cy_sd if cy_sd > 1e-12 else 1.0
    cx_z = (cx - cx_mu) / cx_sd
    cy_z = (cy - cy_mu) / cy_sd

    out = pd.DataFrame(
        {
            "puma_uid": d["puma_uid"].astype(str),
            "statefp": d["statefp"].astype(str).str.zfill(2),
            "puma": d["puma"].astype(str).map(_normalize_puma),
            "centroid_x": cx,
            "centroid_y": cy,
            "centroid_x_z": cx_z.astype(np.float32),
            "centroid_y_z": cy_z.astype(np.float32),
            "area_km2": d["area_km2"].to_numpy(dtype=np.float64),
            "compactness": d["compactness"].to_numpy(dtype=np.float64),
            "n_neighbors": np.full((n,), k, dtype=int),
            "neigh_marg_std_mean": neigh_std_mean.astype(np.float32),
            "neigh_marg_std_max": neigh_std_max.astype(np.float32),
        }
    )

    # Positional encoding on z-scored centroid.
    L = int(args.pe_levels)
    for l in range(L):
        f = (2**l) * np.pi
        out[f"pe_x_sin_{l}"] = np.sin(f * cx_z).astype(np.float32)
        out[f"pe_x_cos_{l}"] = np.cos(f * cx_z).astype(np.float32)
        out[f"pe_y_sin_{l}"] = np.sin(f * cy_z).astype(np.float32)
        out[f"pe_y_cos_{l}"] = np.cos(f * cy_z).astype(np.float32)

    # Neighbor summaries of marginals.
    for i_col, c in enumerate(marg_cols):
        out[f"neigh1_marg_{i_col:03d}"] = neigh1_avg[:, i_col].astype(np.float32)
        out[f"neigh2_marg_{i_col:03d}"] = neigh2_avg[:, i_col].astype(np.float32)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    _write_json(
        out_meta,
        {
            "created_utc": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "inputs": {
                "puma_zip": str(puma_zip),
                "joint_wide_csv": str(joint_csv),
            },
            "params": {
                "knn_k": k,
                "pe_levels": L,
                "epsg_metric": int(args.epsg_metric),
            },
            "n_rows": int(out.shape[0]),
            "n_marg_dims": int(len(marg_cols)),
            "marginal_columns_used": marg_cols,
            "feature_groups": {
                "centroid_raw": ["centroid_x_z", "centroid_y_z"],
                "centroid_pe": [f"pe_* (L={L})"],
                "geo_shape": ["area_km2", "compactness"],
                "neigh_1hop": [f"neigh1_marg_{i:03d}" for i in range(len(marg_cols))],
                "neigh_2hop": [f"neigh2_marg_{i:03d}" for i in range(len(marg_cols))],
                "neigh_stats": ["n_neighbors", "neigh_marg_std_mean", "neigh_marg_std_max"],
            },
            "neighborhood_definition": "KNN only, inverse-distance weighted, no fallback.",
        },
    )
    print(f"[ok] wrote: {out_csv}", file=sys.stderr)
    print(f"[ok] wrote: {out_meta}", file=sys.stderr)


if __name__ == "__main__":
    main()
