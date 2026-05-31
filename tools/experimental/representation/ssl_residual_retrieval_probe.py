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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.experimental.representation.ssl_copula_residual_probe import (  # noqa: E402
    FULL_VARIABLE_ORDER,
    _aggregate_full_to_coarse,
    _fit_pc_scores,
    _load_acs_conditions,
    _load_spatial,
    _load_target,
    _outer_joint,
    _parse_csv_ints,
    _residual_arrays,
    _write_json,
)


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / np.clip(b, 1e-12, None)


def _retrieval_metrics(
    *,
    x: np.ndarray,
    scores: np.ndarray,
    train_mask: np.ndarray,
    explained_ratio: np.ndarray,
    k: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_mask])
    x_test = scaler.transform(x[~train_mask])
    y_train = scores[train_mask]
    y_test = scores[~train_mask]

    k_eff = max(1, min(int(k), y_train.shape[0]))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(x_train)
    distances, indices = nn.kneighbors(x_test)

    pred = np.empty_like(y_test)
    for row_idx in range(y_test.shape[0]):
        pred[row_idx] = np.mean(y_train[indices[row_idx]], axis=0)

    train_var = np.var(y_train, axis=0)
    sq_norm = _safe_div((pred - y_test) ** 2, train_var)
    weights = explained_ratio[: scores.shape[1]].astype(np.float64)
    weights = weights / np.clip(weights.sum(), 1e-12, None)
    weighted_norm_mse_by_row = sq_norm @ weights
    top5 = min(5, scores.shape[1])
    top5_weights = weights[:top5] / np.clip(weights[:top5].sum(), 1e-12, None)
    top5_weighted_norm_mse_by_row = sq_norm[:, :top5] @ top5_weights

    per_row = pd.DataFrame(
        {
            "weighted_norm_mse": weighted_norm_mse_by_row,
            "top5_weighted_norm_mse": top5_weighted_norm_mse_by_row,
            "mean_neighbor_distance": distances.mean(axis=1),
            "min_neighbor_distance": distances[:, 0],
        }
    )
    summary = {
        "k": int(k_eff),
        "n_train": int(train_mask.sum()),
        "n_test": int((~train_mask).sum()),
        "weighted_norm_mse": float(np.mean(weighted_norm_mse_by_row)),
        "weighted_norm_mse_median": float(np.median(weighted_norm_mse_by_row)),
        "top5_weighted_norm_mse": float(np.mean(top5_weighted_norm_mse_by_row)),
        "mean_neighbor_distance": float(np.mean(distances)),
        "mean_min_neighbor_distance": float(np.mean(distances[:, 0])),
    }
    return summary, per_row


def _load_optional_view(path: pathlib.Path | None, target_keys: pd.DataFrame) -> np.ndarray | None:
    if path is None:
        return None
    return _load_spatial(path, target_keys)[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Nearest-neighbor analog retrieval probe for copula-residual PC scores. "
            "Lower retrieval error means the feature space retrieves PUMAs with more similar residual structure."
        )
    )
    data_root = pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data")
    ap.add_argument(
        "--target_wide_csv",
        type=pathlib.Path,
        default=data_root / "us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv",
    )
    ap.add_argument(
        "--condition_csv",
        type=pathlib.Path,
        default=data_root / "us/processed/external_conditions/extcond_v1_agesex_earn_v1_acs5_2022_puma_us.csv",
    )
    ap.add_argument(
        "--spatial_csv",
        type=pathlib.Path,
        default=data_root / "us/processed/features/puma_spatial_features_5var_knn6.csv",
    )
    ap.add_argument("--morphology_csv", type=pathlib.Path, default=None)
    ap.add_argument("--external_csv", type=pathlib.Path, default=None)
    ap.add_argument("--external_label", default="external")
    ap.add_argument("--uncertainty_csv", type=pathlib.Path, default=None)
    ap.add_argument("--heldout_statefps", default="26,12,48,55")
    ap.add_argument("--reference_mode", choices=["target_marginals", "acs_marginals"], default="acs_marginals")
    ap.add_argument("--joint_space", choices=["full", "coarse"], default="full")
    ap.add_argument("--max_pcs", type=int, default=40)
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    out_dir = args.output_dir or pathlib.Path(f"outputs/_ssl_residual_retrieval_probe_{_utc_ts()}")
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    target_keys, p_true, p_eq_target, target_marginals = _load_target(args.target_wide_csv)
    _, acs_marginals, x_acs_1d, x_acs_all, x_scale, _, _, _ = _load_acs_conditions(args.condition_csv, target_keys)
    x_spatial = _load_spatial(args.spatial_csv, target_keys)[0]
    x_morphology = _load_optional_view(args.morphology_csv, target_keys)
    x_external = _load_optional_view(args.external_csv, target_keys)
    x_uncertainty = _load_optional_view(args.uncertainty_csv, target_keys)

    p_eq_acs = _outer_joint([acs_marginals[v] for v in FULL_VARIABLE_ORDER])
    if args.joint_space == "coarse":
        p_true = _aggregate_full_to_coarse(p_true)
        p_eq_target = _aggregate_full_to_coarse(p_eq_target)
        p_eq_acs = _aggregate_full_to_coarse(p_eq_acs)
    reference = p_eq_target if args.reference_mode == "target_marginals" else p_eq_acs
    _, _, log_ratio = _residual_arrays(p_true, reference, eps=float(args.eps))

    feature_sets: dict[str, np.ndarray] = {
        "acs_1d_scale": np.concatenate([x_acs_1d, x_scale], axis=1),
        "acs_all_scale": np.concatenate([x_acs_all, x_scale], axis=1),
        "acs_all_spatial_scale": np.concatenate([x_acs_all, x_spatial, x_scale], axis=1),
    }
    if x_uncertainty is not None:
        feature_sets["acs_all_scale_uncertainty"] = np.concatenate([x_acs_all, x_scale, x_uncertainty], axis=1)
    if x_morphology is not None:
        feature_sets["acs_all_scale_morphology"] = np.concatenate([x_acs_all, x_scale, x_morphology], axis=1)
    if x_morphology is not None and x_uncertainty is not None:
        feature_sets["acs_all_scale_morphology_uncertainty"] = np.concatenate(
            [x_acs_all, x_scale, x_morphology, x_uncertainty],
            axis=1,
        )
    if x_external is not None:
        label = str(args.external_label).strip().replace(" ", "_") or "external"
        feature_sets[label] = x_external
        feature_sets[f"acs_all_scale_{label}"] = np.concatenate([x_acs_all, x_scale, x_external], axis=1)

    heldout_statefps = [str(x).zfill(2) for x in _parse_csv_ints(args.heldout_statefps)]
    ks = [int(x) for x in _parse_csv_ints(args.ks)]
    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()
    puma_uid = target_keys["puma_uid_key"].astype(str).to_numpy()

    summary_rows: list[dict[str, Any]] = []
    long_frames: list[pd.DataFrame] = []
    for heldout in heldout_statefps:
        train_mask = statefp != heldout
        if int((~train_mask).sum()) == 0:
            print(f"[warn] heldout state {heldout} has no rows; skipped", file=sys.stderr)
            continue
        scores, explained_ratio, cumulative = _fit_pc_scores(
            log_ratio,
            train_mask=train_mask,
            n_components=int(args.max_pcs),
            seed=int(args.seed),
        )
        for feature_name, x in feature_sets.items():
            for k in ks:
                row_summary, per_row = _retrieval_metrics(
                    x=x,
                    scores=scores,
                    train_mask=train_mask,
                    explained_ratio=explained_ratio,
                    k=k,
                )
                row_summary.update(
                    {
                        "reference_mode": args.reference_mode,
                        "joint_space": args.joint_space,
                        "heldout_statefp": heldout,
                        "feature_set": feature_name,
                        "n_features": int(x.shape[1]),
                        "n_pcs": int(scores.shape[1]),
                        "pc_cumulative_explained": float(cumulative),
                    }
                )
                summary_rows.append(row_summary)
                test_keys = target_keys.loc[~train_mask, ["statefp", "puma5", "puma_uid_key"]].reset_index(drop=True)
                per_row = pd.concat([test_keys, per_row], axis=1)
                per_row.insert(0, "feature_set", feature_name)
                per_row.insert(0, "heldout_statefp", heldout)
                per_row.insert(0, "k", int(k))
                long_frames.append(per_row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(metrics_dir / "residual_retrieval_summary.csv", index=False)
    if long_frames:
        pd.concat(long_frames, ignore_index=True).to_csv(metrics_dir / "residual_retrieval_by_puma.csv", index=False)

    _write_json(
        out_dir / "run_summary.json",
        {
            "target_wide_csv": str(args.target_wide_csv),
            "condition_csv": str(args.condition_csv),
            "spatial_csv": str(args.spatial_csv),
            "morphology_csv": str(args.morphology_csv) if args.morphology_csv is not None else "",
            "external_csv": str(args.external_csv) if args.external_csv is not None else "",
            "external_label": str(args.external_label),
            "uncertainty_csv": str(args.uncertainty_csv) if args.uncertainty_csv is not None else "",
            "heldout_statefps": heldout_statefps,
            "reference_mode": args.reference_mode,
            "joint_space": args.joint_space,
            "max_pcs": int(args.max_pcs),
            "ks": ks,
            "metrics": {
                "summary": str(metrics_dir / "residual_retrieval_summary.csv"),
                "by_puma": str(metrics_dir / "residual_retrieval_by_puma.csv"),
            },
        },
    )

    if not summary.empty:
        best = (
            summary.sort_values(["heldout_statefp", "k", "weighted_norm_mse"])
            .groupby(["heldout_statefp", "k"], as_index=False)
            .head(5)
            .loc[:, ["heldout_statefp", "k", "feature_set", "weighted_norm_mse", "top5_weighted_norm_mse"]]
        )
        print(best.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
