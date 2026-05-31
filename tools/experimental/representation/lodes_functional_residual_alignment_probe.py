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
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.experimental.representation.ssl_copula_residual_probe import (  # noqa: E402
    FULL_VARIABLE_ORDER,
    _aggregate_full_to_coarse,
    _fit_pc_scores,
    _load_acs_conditions,
    _load_target,
    _outer_joint,
    _parse_csv_ints,
    _residual_arrays,
    _write_json,
)


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _canon_uid(v: object) -> str:
    raw = "".join(ch for ch in str(v).replace(".0", "").strip() if ch.isdigit())
    return raw.zfill(7) if raw else ""


def _weighted_pc_mse(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.clip(w.sum(), 1e-12, None)
    return np.sum((np.asarray(a) - np.asarray(b)) ** 2 * w.reshape(1, -1), axis=1)


def _load_symmetric_top_edges(path: pathlib.Path, top_k: int, min_weight: float) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"home_puma_uid": str, "work_puma_uid": str}, low_memory=False)
    required = {"home_puma_uid", "work_puma_uid", "od_count"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"directed_edges_csv missing columns: {missing}")
    df["home_puma_uid"] = df["home_puma_uid"].map(_canon_uid)
    df["work_puma_uid"] = df["work_puma_uid"].map(_canon_uid)
    df["od_count"] = pd.to_numeric(df["od_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "origin_share" not in df.columns:
        totals = df.groupby("home_puma_uid")["od_count"].transform("sum")
        df["origin_share"] = df["od_count"] / np.clip(totals, 1e-12, None)
    df["origin_share"] = pd.to_numeric(df["origin_share"], errors="coerce").fillna(0.0).clip(lower=0.0)

    a = df.loc[:, ["home_puma_uid", "work_puma_uid", "od_count", "origin_share"]].copy()
    b = a.rename(
        columns={
            "home_puma_uid": "work_puma_uid",
            "work_puma_uid": "home_puma_uid",
            "od_count": "reverse_od_count",
            "origin_share": "reverse_origin_share",
        }
    )
    sym = a.merge(b, on=["home_puma_uid", "work_puma_uid"], how="outer").fillna(0.0)
    sym = sym[sym["home_puma_uid"] != sym["work_puma_uid"]].copy()
    sym["sym_count"] = sym["od_count"] + sym["reverse_od_count"]
    sym["sym_share"] = 0.5 * (sym["origin_share"] + sym["reverse_origin_share"])
    sym = sym[sym["sym_share"] >= float(min_weight)].copy()
    sym = sym.sort_values(["home_puma_uid", "sym_share"], ascending=[True, False])
    return sym.groupby("home_puma_uid", sort=False).head(int(top_k)).reset_index(drop=True)


def _sample_random_pairs(
    *,
    src_indices: np.ndarray,
    train_indices: np.ndarray,
    statefp: np.ndarray,
    rng: np.random.Generator,
    same_state: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dst = np.empty_like(src_indices)
    for pos, src in enumerate(src_indices):
        candidates = train_indices
        if same_state:
            mask = statefp[train_indices] == statefp[src]
            candidates = train_indices[mask]
        if candidates.size <= 1:
            candidates = train_indices
        if candidates.size <= 1:
            dst[pos] = src
            continue
        pick = int(rng.choice(candidates))
        if pick == src:
            alt = candidates[candidates != src]
            pick = int(rng.choice(alt)) if alt.size else pick
        dst[pos] = pick
    return src_indices, dst


def _make_eval_splits(
    statefp: np.ndarray,
    heldouts: list[str],
    *,
    split_mode: str,
    test_fraction: float,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    splits: list[tuple[str, np.ndarray]] = []
    rng = np.random.default_rng(int(seed))
    for heldout in heldouts:
        state_mask = statefp == heldout
        if not np.any(state_mask):
            continue
        if split_mode == "state_holdout":
            splits.append((heldout, ~state_mask))
            continue

        idx = np.where(state_mask)[0]
        if idx.size < 2:
            continue
        n_test = int(np.ceil(float(test_fraction) * idx.size))
        n_test = min(max(n_test, 1), idx.size - 1)
        test_idx = rng.choice(idx, size=n_test, replace=False)
        train_mask = np.ones(statefp.shape[0], dtype=bool)
        train_mask[test_idx] = False
        splits.append((heldout, train_mask))
    return splits


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Test whether LODES functional-neighbor pairs align with demographic copula-residual PC similarity. "
            "Lower weighted PC MSE means better alignment."
        )
    )
    data_root = pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data")
    ap.add_argument("--target_wide_csv", type=pathlib.Path, default=data_root / "us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv")
    ap.add_argument("--condition_csv", type=pathlib.Path, default=data_root / "us/processed/external_conditions/extcond_v1_agesex_earn_v1_acs5_2022_puma_us.csv")
    ap.add_argument("--directed_edges_csv", type=pathlib.Path, required=True)
    ap.add_argument("--heldout_statefps", default="26,12,48,55")
    ap.add_argument("--split_mode", choices=["state_holdout", "within_state_random"], default="state_holdout")
    ap.add_argument("--test_fraction", type=float, default=0.25)
    ap.add_argument("--reference_mode", choices=["target_marginals", "acs_marginals"], default="acs_marginals")
    ap.add_argument("--joint_space", choices=["full", "coarse"], default="full")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--min_weight", type=float, default=0.0)
    ap.add_argument("--max_pcs", type=int, default=40)
    ap.add_argument("--random_repeats", type=int, default=25)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    out_dir = args.output_dir or pathlib.Path(f"outputs/_lodes_functional_residual_alignment_{_utc_ts()}")
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    target_keys, p_true, p_eq_target, _ = _load_target(args.target_wide_csv)
    _, acs_marginals, *_ = _load_acs_conditions(args.condition_csv, target_keys)
    p_eq_acs = _outer_joint([acs_marginals[v] for v in FULL_VARIABLE_ORDER])
    if args.joint_space == "coarse":
        p_true = _aggregate_full_to_coarse(p_true)
        p_eq_target = _aggregate_full_to_coarse(p_eq_target)
        p_eq_acs = _aggregate_full_to_coarse(p_eq_acs)
    reference = p_eq_target if args.reference_mode == "target_marginals" else p_eq_acs
    _, _, log_ratio = _residual_arrays(p_true, reference, eps=float(args.eps))

    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()
    puma_uid = target_keys["puma_uid_key"].astype(str).str.zfill(7).to_numpy()
    uid_to_idx = {uid: i for i, uid in enumerate(puma_uid)}
    heldouts = [str(x).zfill(2) for x in _parse_csv_ints(args.heldout_statefps)]
    top_edges = _load_symmetric_top_edges(args.directed_edges_csv, top_k=int(args.top_k), min_weight=float(args.min_weight))
    top_edges = top_edges[
        top_edges["home_puma_uid"].isin(uid_to_idx) & top_edges["work_puma_uid"].isin(uid_to_idx)
    ].copy()
    top_edges["src_idx"] = top_edges["home_puma_uid"].map(uid_to_idx).astype(int)
    top_edges["dst_idx"] = top_edges["work_puma_uid"].map(uid_to_idx).astype(int)

    rng = np.random.default_rng(int(args.seed))
    pair_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []

    splits = _make_eval_splits(
        statefp,
        heldouts,
        split_mode=str(args.split_mode),
        test_fraction=float(args.test_fraction),
        seed=int(args.seed),
    )
    for heldout, train_mask in splits:
        test_mask = ~train_mask
        if int(test_mask.sum()) == 0:
            continue
        train_indices = np.where(train_mask)[0]
        scores, explained_ratio, cumulative = _fit_pc_scores(
            log_ratio,
            train_mask=train_mask,
            n_components=int(args.max_pcs),
            seed=int(args.seed),
        )
        scaler = StandardScaler()
        z = np.empty_like(scores, dtype=np.float64)
        z[train_mask] = scaler.fit_transform(scores[train_mask])
        z[test_mask] = scaler.transform(scores[test_mask])
        weights = explained_ratio[: z.shape[1]]

        train_pairs = top_edges[top_edges["src_idx"].isin(train_indices) & top_edges["dst_idx"].isin(train_indices)].copy()
        func_dist = _weighted_pc_mse(z[train_pairs["src_idx"].to_numpy()], z[train_pairs["dst_idx"].to_numpy()], weights) if not train_pairs.empty else np.array([])
        for label, same_state in [("random_global", False), ("random_same_state", True)]:
            vals = []
            src = train_pairs["src_idx"].to_numpy(dtype=int)
            for _ in range(max(1, int(args.random_repeats))):
                src_r, dst_r = _sample_random_pairs(
                    src_indices=src,
                    train_indices=train_indices,
                    statefp=statefp,
                    rng=rng,
                    same_state=same_state,
                )
                vals.append(_weighted_pc_mse(z[src_r], z[dst_r], weights))
            rand_dist = np.concatenate(vals) if vals else np.array([])
            pair_rows.append(
                {
                    "heldout_statefp": heldout,
                    "split_mode": str(args.split_mode),
                    "joint_space": args.joint_space,
                    "reference_mode": args.reference_mode,
                    "comparison": f"functional_vs_{label}",
                    "n_functional_pairs": int(func_dist.size),
                    "functional_mse_mean": float(np.mean(func_dist)) if func_dist.size else float("nan"),
                    "random_mse_mean": float(np.mean(rand_dist)) if rand_dist.size else float("nan"),
                    "ratio_functional_to_random": float(np.mean(func_dist) / np.mean(rand_dist)) if func_dist.size and rand_dist.size else float("nan"),
                    "pc_cumulative_explained": float(cumulative),
                }
            )

        # OOD retrieval gate: can held-out PUMAs find training-state functional analogs?
        test_indices = np.where(test_mask)[0]
        pred_rows = []
        for src in test_indices:
            neigh = top_edges[(top_edges["src_idx"] == int(src)) & (top_edges["dst_idx"].isin(train_indices))]
            if neigh.empty:
                continue
            neigh = neigh.sort_values("sym_share", ascending=False).head(int(args.top_k))
            dst = neigh["dst_idx"].to_numpy(dtype=int)
            w = neigh["sym_share"].to_numpy(dtype=np.float64)
            w = w / np.clip(w.sum(), 1e-12, None)
            pred = np.sum(z[dst] * w.reshape(-1, 1), axis=0, keepdims=True)
            pred_rows.append(float(_weighted_pc_mse(z[[src]], pred, weights)[0]))

        random_global_rows = []
        random_same_state_rows = []
        for _ in range(max(1, int(args.random_repeats))):
            for src in test_indices:
                dst = rng.choice(train_indices, size=min(int(args.top_k), train_indices.size), replace=False)
                pred = z[dst].mean(axis=0, keepdims=True)
                random_global_rows.append(float(_weighted_pc_mse(z[[src]], pred, weights)[0]))

                same_state_candidates = train_indices[statefp[train_indices] == statefp[src]]
                if same_state_candidates.size == 0:
                    same_state_candidates = train_indices
                dst_same = rng.choice(
                    same_state_candidates,
                    size=min(int(args.top_k), same_state_candidates.size),
                    replace=False,
                )
                pred_same = z[dst_same].mean(axis=0, keepdims=True)
                random_same_state_rows.append(float(_weighted_pc_mse(z[[src]], pred_same, weights)[0]))

        retrieval_rows.append(
            {
                "heldout_statefp": heldout,
                "split_mode": str(args.split_mode),
                "joint_space": args.joint_space,
                "reference_mode": args.reference_mode,
                "n_test_pumas": int(test_indices.size),
                "n_test_with_train_functional_neighbors": int(len(pred_rows)),
                "functional_neighbor_coverage": float(len(pred_rows) / max(test_indices.size, 1)),
                "functional_retrieval_mse_mean": float(np.mean(pred_rows)) if pred_rows else float("nan"),
                "random_train_retrieval_mse_mean": float(np.mean(random_global_rows)) if random_global_rows else float("nan"),
                "random_same_state_retrieval_mse_mean": float(np.mean(random_same_state_rows)) if random_same_state_rows else float("nan"),
                "ratio_functional_to_random": float(np.mean(pred_rows) / np.mean(random_global_rows)) if pred_rows and random_global_rows else float("nan"),
                "ratio_functional_to_random_same_state": float(np.mean(pred_rows) / np.mean(random_same_state_rows)) if pred_rows and random_same_state_rows else float("nan"),
                "pc_cumulative_explained": float(cumulative),
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    retrieval_df = pd.DataFrame(retrieval_rows)
    pair_df.to_csv(metrics_dir / "functional_pair_alignment_summary.csv", index=False)
    retrieval_df.to_csv(metrics_dir / "functional_ood_retrieval_summary.csv", index=False)
    _write_json(
        out_dir / "run_summary.json",
        {
            "target_wide_csv": str(args.target_wide_csv),
            "condition_csv": str(args.condition_csv),
            "directed_edges_csv": str(args.directed_edges_csv),
            "heldout_statefps": heldouts,
            "split_mode": str(args.split_mode),
            "test_fraction": float(args.test_fraction),
            "reference_mode": args.reference_mode,
            "joint_space": args.joint_space,
            "top_k": int(args.top_k),
            "min_weight": float(args.min_weight),
            "max_pcs": int(args.max_pcs),
            "random_repeats": int(args.random_repeats),
            "n_top_edges": int(len(top_edges)),
            "metrics": {
                "pair_alignment": str(metrics_dir / "functional_pair_alignment_summary.csv"),
                "ood_retrieval": str(metrics_dir / "functional_ood_retrieval_summary.csv"),
            },
        },
    )
    print("[pair alignment]")
    print(pair_df.to_string(index=False))
    print("[ood retrieval]")
    print(retrieval_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
