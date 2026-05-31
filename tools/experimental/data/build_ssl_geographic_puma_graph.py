#!/usr/bin/env python3
from __future__ import annotations

"""Build PUMA-level geographic-neighborhood graph for SSL contrastive probes."""

import argparse
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _digits(v: object) -> str:
    return "".join(ch for ch in str(v).strip() if ch.isdigit())


def _canon_statefp(v: object) -> str:
    d = _digits(v)
    return str(int(d)).zfill(2) if d else ""


def _canon_puma5(v: object) -> str:
    d = _digits(v)
    return str(int(d)).zfill(5) if d else ""


def _add_uid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"statefp", "puma"}.issubset(out.columns):
        out["statefp"] = out["statefp"].map(_canon_statefp)
        out["puma5"] = out["puma"].map(_canon_puma5)
        out["puma_uid_key"] = out["statefp"] + out["puma5"]
    elif "puma_uid" in out.columns:
        raw = out["puma_uid"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)
        out["puma_uid_key"] = raw
        out["statefp"] = raw.str[:2]
        out["puma5"] = raw.str[2:]
    else:
        raise SystemExit("input must contain either statefp+puma or puma_uid")
    out = out[out["puma_uid_key"].str.len() == 7].copy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_ssl_geographic_puma_graph")
    ap.add_argument("--spatial_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--distance_col_x", default="centroid_x")
    ap.add_argument("--distance_col_y", default="centroid_y")
    ap.add_argument("--tau", type=float, default=0.0, help="Distance decay scale. Default: median top-k distance.")
    args = ap.parse_args()

    spatial_csv = pathlib.Path(args.spatial_csv).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not spatial_csv.exists():
        raise SystemExit(f"spatial_csv not found: {spatial_csv}")

    d = _add_uid(pd.read_csv(spatial_csv, low_memory=False))
    for c in [args.distance_col_x, args.distance_col_y]:
        if c not in d.columns:
            raise SystemExit(f"missing coordinate column {c}: {spatial_csv}")
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[args.distance_col_x, args.distance_col_y]).drop_duplicates("puma_uid_key").reset_index(drop=True)
    if d.shape[0] <= int(args.top_k):
        raise SystemExit(f"not enough rows for top_k={args.top_k}: n={d.shape[0]}")

    keys = d["puma_uid_key"].astype(str).tolist()
    xy = d[[args.distance_col_x, args.distance_col_y]].to_numpy(dtype=np.float64)
    dx = xy[:, None, 0] - xy[None, :, 0]
    dy = xy[:, None, 1] - xy[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, : int(args.top_k)]
    nn_dist = np.take_along_axis(dist, nn, axis=1)
    tau = float(args.tau)
    if tau <= 0:
        tau = float(np.median(nn_dist[np.isfinite(nn_dist)]))
    tau = max(tau, 1e-9)

    rows = []
    for i, nbrs in enumerate(nn):
        raw_w = np.exp(-nn_dist[i] / tau)
        raw_w = raw_w / max(float(raw_w.sum()), 1e-12)
        for rank, (j, dij, wij) in enumerate(zip(nbrs.tolist(), nn_dist[i].tolist(), raw_w.tolist()), start=1):
            rows.append(
                {
                    "home_puma_uid": keys[i],
                    "work_puma_uid": keys[j],
                    "neighbor_rank": rank,
                    "geo_distance_m": float(dij),
                    "geo_weight": float(wij),
                    # Reuse generic graph-probe naming where only home/work columns are required.
                    "sym_share": float(wij),
                    "sym_count": float(1.0 / rank),
                }
            )

    edges = pd.DataFrame(rows)
    out_path = out_dir / "puma_geographic_top_neighbors.csv"
    edges.to_csv(out_path, index=False)
    summary = {
        "created_utc": _utc_now(),
        "input": str(spatial_csv),
        "n_pumas": int(d.shape[0]),
        "top_k": int(args.top_k),
        "tau": tau,
        "n_edges": int(edges.shape[0]),
        "outputs": {"top_geographic_neighbors": str(out_path)},
        "interpretation": "Geographic KNN graph for SSL contrastive positive pairs; not a functional OD graph.",
    }
    (out_dir / "geographic_puma_graph_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
