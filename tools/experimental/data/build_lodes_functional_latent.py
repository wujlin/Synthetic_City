#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.experimental.representation.ssl_copula_residual_probe import _load_target, _write_json  # noqa: E402


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    h = -np.sum(np.where(p > 0, p * np.log(np.clip(p, 1e-12, None)), 0.0), axis=1)
    return h / max(np.log(max(p.shape[1], 2)), 1e-12)


def _top_share(p: np.ndarray, k: int) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    kk = max(1, min(int(k), p.shape[1]))
    return np.sort(p, axis=1)[:, -kk:].sum(axis=1)


def _svd_features(x: np.ndarray, n_components: int, prefix: str, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_components = max(1, min(int(n_components), min(x.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=int(seed))
    z = svd.fit_transform(x)
    cols = [f"{prefix}_{i:02d}" for i in range(n_components)]
    meta = {
        f"{prefix}_n_components": int(n_components),
        f"{prefix}_explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
    }
    return pd.DataFrame(z, columns=cols), meta


def _within_group_rank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        vals = values[idx]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        if len(idx) == 1:
            ranks[order] = 0.5
        else:
            ranks[order] = np.arange(len(idx), dtype=np.float64) / float(len(idx) - 1)
        out[idx] = ranks
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build PUMA-level functional latent features from LODES PUMA OD edges.")
    ap.add_argument("--target_wide_csv", type=pathlib.Path, required=True)
    ap.add_argument("--directed_edges_csv", type=pathlib.Path, required=True)
    ap.add_argument(
        "--n_components",
        type=int,
        default=16,
        help="SVD components per OD view. Use 0 to keep only transferable scalar role summaries.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", type=pathlib.Path, required=True)
    args = ap.parse_args()

    keys, _, _, _ = _load_target(args.target_wide_csv)
    puma_uids = keys["puma_uid_key"].astype(str).str.zfill(7).tolist()
    index = {uid: i for i, uid in enumerate(puma_uids)}
    n = len(puma_uids)

    edges = pd.read_csv(args.directed_edges_csv, low_memory=False)
    required = {"home_puma_uid", "work_puma_uid", "od_count"}
    missing = sorted(required - set(edges.columns))
    if missing:
        raise SystemExit(f"directed_edges_csv missing columns: {missing}")
    edges["home_puma_uid"] = edges["home_puma_uid"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)
    edges["work_puma_uid"] = edges["work_puma_uid"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)
    edges["od_count"] = pd.to_numeric(edges["od_count"], errors="coerce").fillna(0.0).clip(lower=0.0)

    mat = np.zeros((n, n), dtype=np.float64)
    used_edges = 0
    used_count = 0.0
    for row in edges.itertuples(index=False):
        i = index.get(row.home_puma_uid)
        j = index.get(row.work_puma_uid)
        if i is None or j is None:
            continue
        c = float(row.od_count)
        if c <= 0:
            continue
        mat[i, j] += c
        used_edges += 1
        used_count += c

    out_total = mat.sum(axis=1)
    in_total = mat.sum(axis=0)
    out_share = mat / np.clip(out_total[:, None], 1e-12, None)
    in_origin_share = mat.T / np.clip(in_total[:, None], 1e-12, None)
    sym = mat + mat.T
    sym_share = sym / np.clip(sym.sum(axis=1, keepdims=True), 1e-12, None)
    self_flow = np.diag(mat)
    self_share = self_flow / np.clip(out_total, 1e-12, None)

    statefp = keys["statefp"].astype(str).str.zfill(2).to_numpy()
    base = pd.DataFrame(
        {
            "statefp": statefp,
            "puma5": keys["puma5"].astype(str).str.zfill(5).to_numpy(),
            "puma_uid": puma_uids,
            "func__log1p_out_total": np.log1p(out_total),
            "func__log1p_in_total": np.log1p(in_total),
            "func__out_in_log_ratio": np.log1p(out_total) - np.log1p(in_total),
            "func__self_share": self_share,
            "func__out_entropy": _entropy(out_share),
            "func__in_entropy": _entropy(in_origin_share),
            "func__sym_entropy": _entropy(sym_share),
            "func__out_top1_share": _top_share(out_share, 1),
            "func__out_top3_share": _top_share(out_share, 3),
            "func__in_top1_share": _top_share(in_origin_share, 1),
            "func__in_top3_share": _top_share(in_origin_share, 3),
            "func__has_lodes": ((out_total > 0) | (in_total > 0)).astype(float),
            "func__out_total_state_rank": _within_group_rank(np.log1p(out_total), statefp),
            "func__in_total_state_rank": _within_group_rank(np.log1p(in_total), statefp),
            "func__self_share_state_rank": _within_group_rank(self_share, statefp),
            "func__out_entropy_state_rank": _within_group_rank(_entropy(out_share), statefp),
            "func__in_entropy_state_rank": _within_group_rank(_entropy(in_origin_share), statefp),
        }
    )

    frames = [base]
    meta: dict[str, Any] = {}
    if int(args.n_components) > 0:
        for arr, prefix in [
            (out_share, "func__out_svd"),
            (in_origin_share, "func__in_svd"),
            (sym_share, "func__sym_svd"),
        ]:
            feats, m = _svd_features(arr, int(args.n_components), prefix, int(args.seed))
            frames.append(feats)
            meta.update(m)
    out = pd.concat(frames, axis=1)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    _write_json(
        args.out_csv.with_suffix(args.out_csv.suffix + ".metadata.json"),
        {
            "created_utc": _utc_ts(),
            "target_wide_csv": str(args.target_wide_csv),
            "directed_edges_csv": str(args.directed_edges_csv),
            "out_csv": str(args.out_csv),
            "n_pumas": int(n),
            "input_edge_rows": int(len(edges)),
            "used_edge_rows": int(used_edges),
            "used_od_count": float(used_count),
            "pumas_with_lodes": int(np.sum((out_total > 0) | (in_total > 0))),
            "n_numeric_features": int(len([c for c in out.columns if c.startswith("func__")])),
            **meta,
        },
    )
    print(f"[done] wrote {args.out_csv} rows={len(out)} cols={len(out.columns)} used_edges={used_edges}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
