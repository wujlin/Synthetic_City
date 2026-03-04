#!/usr/bin/env python3
from __future__ import annotations

"""
Train distribution-level diffusion on US PUMA B19037-style joints and evaluate on Michigan.

Input:
- puma_b19037_joint_wide.csv from tools/build_us_puma_b19037_joint.py

Core setting:
- Distribution-level target: p_joint(age_householder, hh_income)
- Condition ablation: none vs marginal (p_age + p_income)
- Default split: leave_mi_out (train on non-MI, test on MI)

Outputs:
- run_summary.json
- metrics/internal_acs_holdout.json
- metrics/baselines_internal.json
- metrics/ablation_summary.json
"""

import argparse
import datetime as _dt
import json
import math
import pathlib
import random
import sys
from typing import Any

import numpy as np
import pandas as pd


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.model.diffusion_tabular import DiffusionTabularModel, TabDDPMConfig


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: torch. Please run in your conda env with PyTorch installed.") from e
    return torch


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _tvd(p: Any, q: Any) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    return 0.5 * float(np.abs(p - q).sum())


def _cosine(p: Any, q: Any) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    dp = float(np.dot(p, q))
    np_p = float(np.linalg.norm(p))
    np_q = float(np.linalg.norm(q))
    if np_p <= 0 or np_q <= 0:
        return 0.0
    return dp / (np_p * np_q)


def _summ(vals: list[float]) -> dict[str, float] | None:
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "max": float(np.max(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "n": int(arr.size),
    }


def _parse_hidden_dims(spec: str) -> tuple[int, ...]:
    xs = [int(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not xs:
        raise ValueError("--hidden_dims cannot be empty")
    if any(x <= 0 for x in xs):
        raise ValueError("--hidden_dims must be positive")
    return tuple(xs)


def _split_tokens(spec: str) -> list[str]:
    return [x.strip() for x in str(spec).split(",") if x.strip()]


def _sorted_suffix_cols(cols: list[str]) -> list[str]:
    def _key(c: str) -> tuple[int, str]:
        try:
            return (int(str(c).split("_")[-1]), str(c))
        except Exception:
            return (10**9, str(c))

    return sorted(cols, key=_key)


def _select_spatial_feature_cols(*, df: pd.DataFrame, sets: list[str], explicit_cols: list[str]) -> list[str]:
    if explicit_cols:
        miss = [c for c in explicit_cols if c not in df.columns]
        if miss:
            raise SystemExit(f"spatial_features_csv missing explicit columns: {miss}")
        return explicit_cols

    group_cols: dict[str, list[str]] = {
        "centroid_raw": [c for c in ["centroid_x_z", "centroid_y_z"] if c in df.columns],
        "centroid_pe": _sorted_suffix_cols([c for c in df.columns if c.startswith("pe_")]),
        "geo_shape": [c for c in ["area_km2", "compactness"] if c in df.columns],
        "neigh_1hop": _sorted_suffix_cols([c for c in df.columns if c.startswith("neigh1_marg_")]),
        "neigh_2hop": _sorted_suffix_cols([c for c in df.columns if c.startswith("neigh2_marg_")]),
        "neigh_stats": [c for c in ["n_neighbors", "neigh_marg_std_mean", "neigh_marg_std_max"] if c in df.columns],
    }
    bad_sets = [s for s in sets if s not in group_cols]
    if bad_sets:
        raise SystemExit(f"Unsupported --spatial_feature_sets: {bad_sets}")
    out: list[str] = []
    for s in sets:
        out.extend(group_cols[s])
    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for c in out:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    if not uniq:
        raise SystemExit("No spatial feature columns selected. Provide --spatial_feature_sets or --spatial_feature_cols.")
    return uniq


def _load_spatial_features(*, path: pathlib.Path, ids: list[str], sets: list[str], explicit_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    sdf = pd.read_csv(path)
    if "puma_uid" not in sdf.columns:
        raise SystemExit(f"spatial_features_csv missing required column: puma_uid ({path})")
    sdf["puma_uid"] = sdf["puma_uid"].astype(str)
    sdf = sdf.drop_duplicates(subset=["puma_uid"], keep="first").set_index("puma_uid")
    miss_ids = [pid for pid in ids if pid not in sdf.index]
    if miss_ids:
        raise SystemExit(f"spatial_features_csv missing {len(miss_ids)} puma_uid rows. Example={miss_ids[:5]}")
    cols = _select_spatial_feature_cols(df=sdf.reset_index(), sets=sets, explicit_cols=explicit_cols)
    arr = sdf.loc[ids, cols].to_numpy(dtype=np.float32)
    return arr, cols


def _zscore_selected(train: np.ndarray, test: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train.ndim != 2 or test.ndim != 2 or mask.ndim != 1:
        raise ValueError("invalid shape for zscore_selected")
    if train.shape[1] != test.shape[1] or train.shape[1] != mask.size:
        raise ValueError("dimension mismatch for zscore_selected")
    if not bool(mask.any()):
        return train, test
    tr = train.copy()
    te = test.copy()
    mu = tr[:, mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = tr[:, mask].std(axis=0, dtype=np.float64).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    tr[:, mask] = (tr[:, mask] - mu.reshape(1, -1)) / sd.reshape(1, -1)
    te[:, mask] = (te[:, mask] - mu.reshape(1, -1)) / sd.reshape(1, -1)
    return tr, te


def _ipf_2d(*, seed_joint: Any, target_row: Any, target_col: Any, max_iter: int = 200, tol: float = 1e-10) -> Any:
    seed = np.asarray(seed_joint, dtype=float).reshape(-1)
    r = np.asarray(target_row, dtype=float).reshape(-1)
    c = np.asarray(target_col, dtype=float).reshape(-1)
    if float(r.sum()) <= 0 or float(c.sum()) <= 0:
        raise ValueError("target marginals must be non-empty")
    r = np.clip(r, 0.0, None)
    c = np.clip(c, 0.0, None)
    r = r / float(r.sum())
    c = c / float(c.sum())
    n_row = int(r.size)
    n_col = int(c.size)
    if seed.size != n_row * n_col:
        raise ValueError(f"seed_joint size mismatch: seed={seed.size}, expected={n_row*n_col}")

    table = seed.reshape(n_row, n_col).astype(float)
    table = np.clip(table, 0.0, None)
    s = float(table.sum())
    if s <= 0:
        table[:] = 1.0 / float(n_row * n_col)
    else:
        table /= s

    for _ in range(int(max_iter)):
        row_sum = table.sum(axis=1)
        row_factor = np.zeros_like(row_sum)
        m = row_sum > 0
        row_factor[m] = r[m] / row_sum[m]
        table = table * row_factor.reshape(-1, 1)
        if bool((r <= 0).any()):
            table[r <= 0, :] = 0.0

        col_sum = table.sum(axis=0)
        col_factor = np.zeros_like(col_sum)
        m = col_sum > 0
        col_factor[m] = c[m] / col_sum[m]
        table = table * col_factor.reshape(1, -1)
        if bool((c <= 0).any()):
            table[:, c <= 0] = 0.0

        if float(np.max(np.abs(table.sum(axis=1) - r))) < float(tol) and float(np.max(np.abs(table.sum(axis=0) - c))) < float(tol):
            break

    out = table.reshape(-1)
    out = np.clip(out, 0.0, None)
    out = out / (float(out.sum()) if float(out.sum()) > 0 else 1.0)
    return out


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(z)
    denom = np.sum(e, axis=1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return e / denom


def _stable_hash_fold(values: list[str], *, n_folds: int, seed: int) -> dict[str, int]:
    import hashlib

    out: dict[str, int] = {}
    for v in values:
        h = hashlib.sha1((str(seed) + "::" + str(v)).encode("utf-8")).hexdigest()
        out[str(v)] = int(h[:8], 16) % int(n_folds)
    return out


def _apply_posthoc_ipf(*, policy: str, cond_name: str) -> bool:
    p = str(policy).strip().lower()
    if p == "none":
        return False
    if p == "all":
        return True
    if p == "marginal":
        return str(cond_name).strip().lower() in {"marginal", "marginal_spatial"}
    raise ValueError(f"Unsupported posthoc_ipf_policy: {policy}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="train_us_puma_b19037_diffusion")
    ap.add_argument("--joint_wide_csv", required=True, help="Path to puma_b19037_joint_wide.csv")
    ap.add_argument(
        "--conditions",
        default="none,marginal",
        help='Comma-separated: "none,marginal,spatial,marginal_spatial"',
    )
    ap.add_argument("--eval_mode", choices=["leave_mi_out", "mi_kfold"], default="leave_mi_out")
    ap.add_argument("--n_folds", type=int, default=5, help="Used when eval_mode=mi_kfold")
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--hidden_dims", default="512,512")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--condition_injection", choices=["concat", "film"], default="concat")
    ap.add_argument("--film_hidden_dim", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--n_eval_joint_samples", type=int, default=256)
    ap.add_argument("--spatial_features_csv", default=None, help="Optional PUMA-level spatial feature CSV.")
    ap.add_argument(
        "--spatial_feature_sets",
        default="",
        help='Comma-separated groups from {centroid_raw,centroid_pe,geo_shape,neigh_1hop,neigh_2hop,neigh_stats}.',
    )
    ap.add_argument("--spatial_feature_cols", default="", help="Optional explicit spatial feature columns (comma-separated).")
    ap.add_argument(
        "--posthoc_ipf_policy",
        choices=["none", "marginal", "all"],
        default="marginal",
        help="Apply post-hoc IPF on diffusion output during evaluation: none | marginal | all.",
    )
    ap.add_argument("--out_dir", default=None, help="Default: outputs/<run_id>")
    args = ap.parse_args()
    torch = _require_torch()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    in_path = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"joint_wide_csv not found: {in_path}")

    run_id = f"_us_puma_b19037_diffusion_{_dt.datetime.now(_dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else (_REPO_ROOT / "outputs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    required = {"statefp", "puma", "puma_uid", "total_households"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"joint_wide_csv missing columns: {missing}")

    p_joint_cols = sorted([c for c in df.columns if c.startswith("p_joint_")], key=lambda x: int(x.split("_")[-1]))
    p_age_cols = sorted([c for c in df.columns if c.startswith("p_age_")], key=lambda x: int(x.split("_")[-1]))
    p_inc_cols = sorted([c for c in df.columns if c.startswith("p_income_")], key=lambda x: int(x.split("_")[-1]))
    if not p_joint_cols or not p_age_cols or not p_inc_cols:
        raise SystemExit("joint_wide_csv missing p_joint_/p_age_/p_income_ columns.")

    n_row = len(p_age_cols)
    n_col = len(p_inc_cols)
    K = len(p_joint_cols)
    if K != n_row * n_col:
        raise SystemExit(f"shape mismatch: K={K}, n_row={n_row}, n_col={n_col}")

    df["statefp"] = df["statefp"].astype(str).str.zfill(2)
    is_mi = df["statefp"] == "26"
    if int(is_mi.sum()) == 0:
        raise SystemExit("No Michigan rows found (statefp==26).")

    p_joint = df[p_joint_cols].to_numpy(dtype=np.float32)
    p_age = df[p_age_cols].to_numpy(dtype=np.float32)
    p_inc = df[p_inc_cols].to_numpy(dtype=np.float32)
    totals = pd.to_numeric(df["total_households"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ids = df["puma_uid"].astype(str).tolist()

    # Normalize probabilities defensively.
    p_joint = np.clip(p_joint, 0.0, None)
    p_age = np.clip(p_age, 0.0, None)
    p_inc = np.clip(p_inc, 0.0, None)
    p_joint = p_joint / np.maximum(p_joint.sum(axis=1, keepdims=True), 1e-12)
    p_age = p_age / np.maximum(p_age.sum(axis=1, keepdims=True), 1e-12)
    p_inc = p_inc / np.maximum(p_inc.sum(axis=1, keepdims=True), 1e-12)

    # Work in log-prob space, then z-score per training fold to match DDPM's N(0,1) operating regime.
    x_log_all = np.log(np.clip(p_joint, 0.0, None) + 1e-6).astype(np.float32)
    cond_marg = np.concatenate([p_age, p_inc], axis=1).astype(np.float32)

    # Build folds.
    folds: list[tuple[str, np.ndarray, np.ndarray]] = []  # (fold_name, train_idx, test_idx)
    if str(args.eval_mode) == "leave_mi_out":
        tr = np.where(~is_mi.to_numpy(dtype=bool))[0]
        te = np.where(is_mi.to_numpy(dtype=bool))[0]
        if tr.size == 0 or te.size == 0:
            raise SystemExit("Invalid split in leave_mi_out mode.")
        folds.append(("leave_mi_out", tr, te))
    else:
        mi_ids = [ids[i] for i in np.where(is_mi.to_numpy(dtype=bool))[0]]
        fold_map = _stable_hash_fold(sorted(set(mi_ids)), n_folds=int(args.n_folds), seed=int(args.seed))
        for f in range(int(args.n_folds)):
            te_mask = np.array([(is_mi.iloc[i] and fold_map.get(ids[i], -1) == f) for i in range(len(ids))], dtype=bool)
            tr_mask = ~te_mask
            tr = np.where(tr_mask)[0]
            te = np.where(te_mask)[0]
            if tr.size == 0 or te.size == 0:
                continue
            folds.append((f"mi_fold_{f}", tr, te))
        if not folds:
            raise SystemExit("No valid folds built in mi_kfold mode.")

    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    conditions = [c.strip() for c in str(args.conditions).split(",") if c.strip()]
    allowed = {"none", "marginal", "spatial", "marginal_spatial"}
    bad = [c for c in conditions if c not in allowed]
    if bad:
        raise SystemExit(f"Unsupported conditions: {bad}. allowed={sorted(allowed)}")
    needs_spatial = any("spatial" in c for c in conditions)
    spatial_arr: np.ndarray | None = None
    spatial_cols: list[str] = []
    spatial_sets = _split_tokens(args.spatial_feature_sets)
    spatial_explicit = _split_tokens(args.spatial_feature_cols)
    if needs_spatial:
        if not args.spatial_features_csv:
            raise SystemExit("Spatial condition requested but --spatial_features_csv is not provided.")
        spatial_path = pathlib.Path(args.spatial_features_csv).expanduser().resolve()
        if not spatial_path.exists():
            raise SystemExit(f"spatial_features_csv not found: {spatial_path}")
        spatial_arr, spatial_cols = _load_spatial_features(
            path=spatial_path,
            ids=ids,
            sets=spatial_sets,
            explicit_cols=spatial_explicit,
        )
        if spatial_arr.shape[0] != len(ids):
            raise SystemExit("spatial feature rows mismatch with joint rows.")

    cond_map: dict[str, np.ndarray | None] = {
        "none": None,
        "marginal": cond_marg,
        "spatial": spatial_arr,
        "marginal_spatial": (np.concatenate([cond_marg, spatial_arr], axis=1).astype(np.float32) if spatial_arr is not None else None),
    }
    spatial_mask_map: dict[str, np.ndarray | None] = {
        "none": None,
        "marginal": np.zeros((cond_marg.shape[1],), dtype=bool),
        "spatial": (np.ones((spatial_arr.shape[1],), dtype=bool) if spatial_arr is not None else None),
        "marginal_spatial": (
            np.concatenate(
                [
                    np.zeros((cond_marg.shape[1],), dtype=bool),
                    np.ones((spatial_arr.shape[1],), dtype=bool),
                ]
            )
            if spatial_arr is not None
            else None
        ),
    }

    internal_by_condition: dict[str, Any] = {}
    baselines_by_fold: dict[str, dict[str, Any]] = {"independence": {}, "ipf_train_seed": {}}

    for cond_name in conditions:
        cond_arr = cond_map[cond_name]
        cond_dim = 0 if cond_arr is None else int(cond_arr.shape[1])
        cond_fold_metrics: dict[str, Any] = {}

        for fold_name, train_idx, test_idx in folds:
            x_train_log = x_log_all[train_idx]
            x_mean = x_train_log.mean(axis=0, dtype=np.float64).astype(np.float32)
            x_std = x_train_log.std(axis=0, dtype=np.float64).astype(np.float32)
            x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
            x_train = ((x_train_log - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)).astype(np.float32)
            x_test_true = p_joint[test_idx]
            tot_train = totals[train_idx]
            tot_test = totals[test_idx]
            cond_train = cond_arr[train_idx] if cond_arr is not None else None
            cond_test = cond_arr[test_idx] if cond_arr is not None else None
            m = spatial_mask_map[cond_name]
            if cond_train is not None and cond_test is not None and m is not None and bool(m.any()):
                cond_train, cond_test = _zscore_selected(cond_train, cond_test, m)

            model = DiffusionTabularModel(
                input_dim=K,
                cond_dim=cond_dim,
                seed=int(args.seed),
                config=TabDDPMConfig(
                    timesteps=int(args.timesteps),
                    hidden_dims=hidden_dims,
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    condition_injection=str(args.condition_injection),
                    film_hidden_dim=int(args.film_hidden_dim),
                ),
            )
            fit_kwargs = {
                "x": torch.from_numpy(x_train),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "device": args.device,
                "log_every": int(args.log_every),
            }
            if cond_train is not None:
                fit_kwargs["cond"] = torch.from_numpy(cond_train)
            model.fit(**fit_kwargs)

            tvd_joint_vals: list[float] = []
            tvd_age_vals: list[float] = []
            tvd_inc_vals: list[float] = []
            cos_vals: list[float] = []
            # Keep raw (pre-IPF) metrics for fairness diagnostics.
            raw_tvd_joint_vals: list[float] = []
            raw_tvd_age_vals: list[float] = []
            raw_tvd_inc_vals: list[float] = []
            raw_cos_vals: list[float] = []

            # Baselines for this fold (computed once; same for all conditions).
            seed_counts = np.zeros((K,), dtype=float)
            for i, idx in enumerate(train_idx):
                w = float(max(tot_train[i], 0.0))
                seed_counts += w * p_joint[idx]
            if float(seed_counts.sum()) <= 0:
                seed_joint = np.mean(p_joint[train_idx], axis=0)
                seed_joint = seed_joint / max(float(seed_joint.sum()), 1e-12)
            else:
                seed_joint = seed_counts / float(seed_counts.sum())
            seed_joint_2d = seed_joint.reshape(n_row, n_col)

            ind_tvd_vals: list[float] = []
            ipf_tvd_vals: list[float] = []

            for j, idx in enumerate(test_idx):
                p_true = p_joint[idx]
                p_age_t = p_age[idx]
                p_inc_t = p_inc[idx]

                # Baselines.
                p_ind = (p_age_t.reshape(-1, 1) * p_inc_t.reshape(1, -1)).reshape(-1)
                p_ipf = _ipf_2d(seed_joint=seed_joint_2d, target_row=p_age_t, target_col=p_inc_t)
                ind_tvd_vals.append(_tvd(p_ind, p_true))
                ipf_tvd_vals.append(_tvd(p_ipf, p_true))

                n_eval = int(args.n_eval_joint_samples)
                if cond_test is None:
                    z = model.sample(n=n_eval, cond=None, device=args.device).numpy()
                else:
                    c = np.repeat(cond_test[j : j + 1], repeats=n_eval, axis=0).astype(np.float32)
                    z = model.sample(n=n_eval, cond=torch.from_numpy(c), device=args.device).numpy()
                # Inverse z-score back to log-prob space before simplex projection.
                logp = z.astype(np.float64) * x_std.reshape(1, -1).astype(np.float64) + x_mean.reshape(1, -1).astype(np.float64)
                p_draws = _softmax_rows(logp)
                p_hat_raw = np.mean(p_draws, axis=0)
                p_hat_raw = p_hat_raw / max(float(p_hat_raw.sum()), 1e-12)

                p_hat_eval = p_hat_raw
                if _apply_posthoc_ipf(policy=str(args.posthoc_ipf_policy), cond_name=cond_name):
                    p_hat_eval = _ipf_2d(
                        seed_joint=p_hat_raw.reshape(n_row, n_col),
                        target_row=p_age_t,
                        target_col=p_inc_t,
                    )

                # Raw metrics (before optional post-hoc IPF).
                raw_tvd_joint_vals.append(_tvd(p_hat_raw, p_true))
                p_hat_raw_2d = p_hat_raw.reshape(n_row, n_col)
                p_true_2d = p_true.reshape(n_row, n_col)
                raw_tvd_age_vals.append(_tvd(p_hat_raw_2d.sum(axis=1), p_true_2d.sum(axis=1)))
                raw_tvd_inc_vals.append(_tvd(p_hat_raw_2d.sum(axis=0), p_true_2d.sum(axis=0)))
                raw_cos_vals.append(_cosine(p_hat_raw, p_true))

                # Eval metrics (after policy-selected post-hoc treatment).
                tvd_joint_vals.append(_tvd(p_hat_eval, p_true))
                p_hat_2d = p_hat_eval.reshape(n_row, n_col)
                p_true_2d = p_true.reshape(n_row, n_col)
                tvd_age_vals.append(_tvd(p_hat_2d.sum(axis=1), p_true_2d.sum(axis=1)))
                tvd_inc_vals.append(_tvd(p_hat_2d.sum(axis=0), p_true_2d.sum(axis=0)))
                cos_vals.append(_cosine(p_hat_eval, p_true))

            if fold_name not in baselines_by_fold["independence"]:
                baselines_by_fold["independence"][fold_name] = {"tvd_joint": _summ(ind_tvd_vals)}
                baselines_by_fold["ipf_train_seed"][fold_name] = {"tvd_joint": _summ(ipf_tvd_vals)}

            cond_fold_metrics[fold_name] = {
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "posthoc_ipf_applied": bool(_apply_posthoc_ipf(policy=str(args.posthoc_ipf_policy), cond_name=cond_name)),
                "tvd_joint": _summ(tvd_joint_vals),
                "tvd_age": _summ(tvd_age_vals),
                "tvd_income": _summ(tvd_inc_vals),
                "cosine_joint": _summ(cos_vals),
                "tvd_joint_raw": _summ(raw_tvd_joint_vals),
                "tvd_age_raw": _summ(raw_tvd_age_vals),
                "tvd_income_raw": _summ(raw_tvd_inc_vals),
                "cosine_joint_raw": _summ(raw_cos_vals),
            }

        # Aggregate across folds on fold mean.
        fold_names = sorted(cond_fold_metrics.keys())
        vals_joint = [float(cond_fold_metrics[f]["tvd_joint"]["mean"]) for f in fold_names if cond_fold_metrics[f].get("tvd_joint")]
        vals_age = [float(cond_fold_metrics[f]["tvd_age"]["mean"]) for f in fold_names if cond_fold_metrics[f].get("tvd_age")]
        vals_inc = [float(cond_fold_metrics[f]["tvd_income"]["mean"]) for f in fold_names if cond_fold_metrics[f].get("tvd_income")]
        vals_cos = [float(cond_fold_metrics[f]["cosine_joint"]["mean"]) for f in fold_names if cond_fold_metrics[f].get("cosine_joint")]
        vals_joint_raw = [float(cond_fold_metrics[f]["tvd_joint_raw"]["mean"]) for f in fold_names if cond_fold_metrics[f].get("tvd_joint_raw")]
        vals_age_raw = [float(cond_fold_metrics[f]["tvd_age_raw"]["mean"]) for f in fold_names if cond_fold_metrics[f].get("tvd_age_raw")]
        vals_inc_raw = [float(cond_fold_metrics[f]["tvd_income_raw"]["mean"]) for f in fold_names if cond_fold_metrics[f].get("tvd_income_raw")]
        vals_cos_raw = [float(cond_fold_metrics[f]["cosine_joint_raw"]["mean"]) for f in fold_names if cond_fold_metrics[f].get("cosine_joint_raw")]
        internal_by_condition[cond_name] = {
            "overall": {
                "tvd_joint": _summ(vals_joint),
                "tvd_age": _summ(vals_age),
                "tvd_income": _summ(vals_inc),
                "cosine_joint": _summ(vals_cos),
                "tvd_joint_raw": _summ(vals_joint_raw),
                "tvd_age_raw": _summ(vals_age_raw),
                "tvd_income_raw": _summ(vals_inc_raw),
                "cosine_joint_raw": _summ(vals_cos_raw),
            },
            "by_fold": cond_fold_metrics,
        }

    baselines_internal = {"by_baseline": {}}
    for bname, bfold in baselines_by_fold.items():
        fn = sorted(bfold.keys())
        vals = [float(bfold[f]["tvd_joint"]["mean"]) for f in fn if bfold[f].get("tvd_joint")]
        baselines_internal["by_baseline"][bname] = {"tvd_joint": _summ(vals), "by_fold": bfold}

    ablation_summary: dict[str, Any] = {"conditions": {}, "baselines": {}}
    for cond_name, obj in internal_by_condition.items():
        ablation_summary["conditions"][cond_name] = obj["overall"]
    for bname, obj in baselines_internal["by_baseline"].items():
        ablation_summary["baselines"][bname] = {"tvd_joint": obj["tvd_joint"]}

    run_summary = {
        "created_utc": _utc_now_iso(),
        "input_csv": str(in_path),
        "n_rows_total": int(df.shape[0]),
        "n_mi_rows": int(is_mi.sum()),
        "n_non_mi_rows": int((~is_mi).sum()),
        "n_age_bins": int(n_row),
        "n_income_bins": int(n_col),
        "K_joint_dim": int(K),
        "eval_mode": str(args.eval_mode),
        "n_folds": int(len(folds)),
        "conditions": conditions,
        "condition_dims": {c: 0 if cond_map[c] is None else int(cond_map[c].shape[1]) for c in conditions},
        "timesteps": int(args.timesteps),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_dims": list(hidden_dims),
        "condition_injection": str(args.condition_injection),
        "film_hidden_dim": int(args.film_hidden_dim),
        "n_eval_joint_samples": int(args.n_eval_joint_samples),
        "posthoc_ipf_policy": str(args.posthoc_ipf_policy),
        "x_representation": "logp + per-fold z-score",
        "seed": int(args.seed),
        "device": args.device,
        "spatial_features_csv": str(pathlib.Path(args.spatial_features_csv).expanduser().resolve()) if args.spatial_features_csv else None,
        "spatial_feature_sets": spatial_sets,
        "spatial_feature_cols": spatial_cols if spatial_cols else spatial_explicit,
    }

    _write_json(out_dir / "run_summary.json", run_summary)
    _write_json(out_dir / "metrics" / "internal_acs_holdout.json", {"by_condition": internal_by_condition})
    _write_json(out_dir / "metrics" / "baselines_internal.json", baselines_internal)
    _write_json(out_dir / "metrics" / "ablation_summary.json", ablation_summary)
    print(f"[ok] wrote: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
