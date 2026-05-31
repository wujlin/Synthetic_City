#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.model.external_c2f_full_earn_schema import COARSE_SHAPE, FULL_SHAPE, FULL_VARIABLE_ORDER
from tools.experimental.representation.ssl_copula_residual_probe import (  # noqa: E402
    _aggregate_full_to_coarse,
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


def _canon_statefps(raw: str) -> list[str]:
    return [str(x).zfill(2) for x in _parse_csv_ints(raw)]


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    sums = arr.sum(axis=1, keepdims=True)
    fallback = np.full_like(arr, 1.0 / max(arr.shape[1], 1))
    return np.where(sums > eps, arr / np.clip(sums, eps, None), fallback)


def _row_tvd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).sum(axis=1)


def _marginals_from_rows(p_joint: np.ndarray, *, shape: tuple[int, ...]) -> list[np.ndarray]:
    n = int(p_joint.shape[0])
    tab = np.asarray(p_joint, dtype=np.float64).reshape((n,) + tuple(shape))
    out: list[np.ndarray] = []
    for axis in range(len(shape)):
        sum_axes = tuple(j + 1 for j in range(len(shape)) if j != axis)
        out.append(_normalize_rows(tab.sum(axis=sum_axes)))
    return out


def _ipf_project_rows_np(
    seed_joint: np.ndarray,
    target_marginals: list[np.ndarray],
    *,
    shape: tuple[int, ...],
    max_iter: int,
    eps: float,
) -> np.ndarray:
    n = int(seed_joint.shape[0])
    x = _normalize_rows(seed_joint, eps=eps).reshape((n,) + tuple(shape))
    targets = [_normalize_rows(t, eps=eps) for t in target_marginals]
    d = len(shape)
    for _ in range(int(max_iter)):
        for axis, target in enumerate(targets):
            sum_axes = tuple(j + 1 for j in range(d) if j != axis)
            current = x.sum(axis=sum_axes)
            factor = target / np.clip(current, eps, None)
            reshape = [n] + [1] * d
            reshape[axis + 1] = int(shape[axis])
            x *= factor.reshape(reshape)
        x = _normalize_rows(x.reshape(n, -1), eps=eps).reshape((n,) + tuple(shape))
    return x.reshape(n, -1)


def _standardize_train_test(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    out = np.empty_like(np.asarray(x, dtype=np.float64), dtype=np.float64)
    out[train_mask] = scaler.fit_transform(x[train_mask])
    out[~train_mask] = scaler.transform(x[~train_mask])
    return out, scaler


def _fit_log_ratio_basis(
    log_ratio: np.ndarray,
    train_mask: np.ndarray,
    *,
    basis_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    z_train = np.asarray(log_ratio[train_mask], dtype=np.float64)
    mean = z_train.mean(axis=0)
    centered = z_train - mean.reshape(1, -1)
    n_components = max(1, min(int(basis_dim), centered.shape[0] - 1, centered.shape[1]))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(seed))
    pca.fit(centered)
    basis = np.asarray(pca.components_, dtype=np.float32)
    return mean.astype(np.float32), basis, float(np.sum(pca.explained_variance_ratio_))


def _safe_torch_import():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - remote environment dependent
        raise SystemExit(f"PyTorch is required for projection-aware training: {exc}") from exc
    return torch, nn


@dataclass
class TorchBatchData:
    x: Any
    p_base: Any
    p_true: Any
    targets: list[Any]


def _torch_ipf_project(seed_joint: Any, targets: list[Any], *, shape: tuple[int, ...], n_iter: int, eps: float, torch: Any) -> Any:
    n = int(seed_joint.shape[0])
    x = torch.clamp(seed_joint, min=0.0)
    x = x / torch.clamp(x.sum(dim=1, keepdim=True), min=eps)
    x = x.reshape((n,) + tuple(shape))
    d = len(shape)
    for _ in range(int(n_iter)):
        for axis, target in enumerate(targets):
            sum_dims = tuple(j + 1 for j in range(d) if j != axis)
            current = x.sum(dim=sum_dims)
            factor = target / torch.clamp(current, min=eps)
            reshape = [n] + [1] * d
            reshape[axis + 1] = int(shape[axis])
            x = x * factor.reshape(reshape)
        flat = x.reshape(n, -1)
        flat = torch.clamp(flat, min=0.0)
        flat = flat / torch.clamp(flat.sum(dim=1, keepdim=True), min=eps)
        x = flat.reshape((n,) + tuple(shape))
    return x.reshape(n, -1)


def _torch_distribution_from_model(
    *,
    model: Any,
    x: Any,
    p_base: Any,
    mean_log_ratio: Any,
    basis: Any,
    clip_log_ratio: float,
    eps: float,
    torch: Any,
) -> tuple[Any, Any]:
    coeff = model(x)
    u = mean_log_ratio.reshape(1, -1) + coeff @ basis
    u = torch.clamp(u, min=-float(clip_log_ratio), max=float(clip_log_ratio))
    q = p_base * torch.exp(u)
    q = q / torch.clamp(q.sum(dim=1, keepdim=True), min=eps)
    return q, coeff


def _make_model(input_dim: int, basis_dim: int, hidden_dim: int, dropout: float, nn: Any) -> Any:
    if int(hidden_dim) <= 0:
        return nn.Linear(int(input_dim), int(basis_dim))
    return nn.Sequential(
        nn.Linear(int(input_dim), int(hidden_dim)),
        nn.SiLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(int(hidden_dim), int(basis_dim)),
    )


def _train_projection_aware_model(
    *,
    x: np.ndarray,
    p_base: np.ndarray,
    p_base_eval: np.ndarray | None,
    p_true: np.ndarray,
    target_marginals: list[np.ndarray],
    mean_log_ratio: np.ndarray,
    basis: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    shape: tuple[int, ...],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    coeff_l2: float,
    hidden_dim: int,
    dropout: float,
    ipf_iters_train: int,
    ipf_iters_eval: int,
    clip_log_ratio: float,
    eps: float,
    seed: int,
    device_name: str,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    torch, nn = _safe_torch_import()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available() and device_name != "cpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    p_base_t = torch.as_tensor(p_base, dtype=torch.float32, device=device)
    p_base_eval_t = torch.as_tensor(p_base if p_base_eval is None else p_base_eval, dtype=torch.float32, device=device)
    p_true_t = torch.as_tensor(p_true, dtype=torch.float32, device=device)
    targets_t = [torch.as_tensor(m, dtype=torch.float32, device=device) for m in target_marginals]
    mean_t = torch.as_tensor(mean_log_ratio, dtype=torch.float32, device=device)
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=device)

    train_idx = np.where(train_mask)[0].astype(np.int64)
    test_idx = np.where(test_mask)[0].astype(np.int64)
    model = _make_model(x.shape[1], basis.shape[0], hidden_dim, dropout, nn).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = np.random.default_rng(int(seed))
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        rng.shuffle(train_idx)
        loss_vals: list[float] = []
        tvd_vals: list[float] = []
        coeff_vals: list[float] = []
        for start in range(0, train_idx.size, int(batch_size)):
            idx_np = train_idx[start : start + int(batch_size)]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
            q_pre, coeff = _torch_distribution_from_model(
                model=model,
                x=x_t[idx],
                p_base=p_base_t[idx],
                mean_log_ratio=mean_t,
                basis=basis_t,
                clip_log_ratio=clip_log_ratio,
                eps=eps,
                torch=torch,
            )
            q_post = _torch_ipf_project(
                q_pre,
                [m[idx] for m in targets_t],
                shape=shape,
                n_iter=int(ipf_iters_train),
                eps=eps,
                torch=torch,
            )
            tvd = 0.5 * torch.abs(q_post - p_true_t[idx]).sum(dim=1).mean()
            coeff_penalty = torch.mean(coeff**2)
            loss = tvd + float(coeff_l2) * coeff_penalty
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            loss_vals.append(float(loss.detach().cpu()))
            tvd_vals.append(float(tvd.detach().cpu()))
            coeff_vals.append(float(coeff_penalty.detach().cpu()))
        if epoch == 1 or epoch % max(1, int(epochs) // 10) == 0 or epoch == int(epochs):
            history_rows.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(np.mean(loss_vals)),
                    "train_tvd": float(np.mean(tvd_vals)),
                    "train_coeff_l2": float(np.mean(coeff_vals)),
                }
            )

    def _eval(mask: np.ndarray) -> tuple[np.ndarray, float]:
        model.eval()
        idx_all = np.where(mask)[0].astype(np.int64)
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, idx_all.size, int(batch_size)):
                idx_np = idx_all[start : start + int(batch_size)]
                idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
                q_pre, _ = _torch_distribution_from_model(
                    model=model,
                    x=x_t[idx],
                    p_base=p_base_eval_t[idx],
                    mean_log_ratio=mean_t,
                    basis=basis_t,
                    clip_log_ratio=clip_log_ratio,
                    eps=eps,
                    torch=torch,
                )
                q_post = _torch_ipf_project(
                    q_pre,
                    [m[idx] for m in targets_t],
                    shape=shape,
                    n_iter=int(ipf_iters_eval),
                    eps=eps,
                    torch=torch,
                )
                preds.append(q_post.detach().cpu().numpy())
        pred = np.vstack(preds)
        tvd = float(np.mean(_row_tvd(pred, p_true[idx_all])))
        return pred, tvd

    pred_test, test_tvd = _eval(test_mask)
    _, train_tvd = _eval(train_mask)
    metrics = {
        "train_tvd_eval": float(train_tvd),
        "test_tvd_eval": float(test_tvd),
        "device_cuda": float(device.type == "cuda"),
    }
    return metrics, pd.DataFrame(history_rows), pred_test


def _build_feature_sets(
    *,
    x_context: np.ndarray,
    x_external: np.ndarray,
    train_mask: np.ndarray,
    requested: list[str],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    context_z, _ = _standardize_train_test(x_context, train_mask)
    external_z, _ = _standardize_train_test(x_external, train_mask)
    if "acs" in requested:
        out["acs"] = context_z
    raw_names = [name for name in ("acs_external_raw", "acs_poi_raw") if name in requested]
    for name in raw_names:
        out[name] = np.concatenate([context_z, external_z], axis=1)
    resid_names = [name for name in ("acs_external_resid", "acs_poi_resid") if name in requested]
    if resid_names:
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=10.0)
        model.fit(context_z[train_mask], external_z[train_mask])
        residual = external_z - model.predict(context_z)
        for name in resid_names:
            out[name] = np.concatenate([context_z, residual], axis=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Train projection-aware external-view correction. The loss is post-IPF TVD, "
            "not residual-PC prediction error."
        )
    )
    data_root = pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data")
    ap.add_argument("--target_wide_csv", type=pathlib.Path, default=data_root / "us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv")
    ap.add_argument("--condition_csv", type=pathlib.Path, default=data_root / "us/processed/external_conditions/extcond_v1_agesex_earn_v1_acs5_2022_puma_us.csv")
    ap.add_argument("--external_csv", type=pathlib.Path, required=True)
    ap.add_argument(
        "--eligible_statefps",
        default="",
        help="Optional comma-separated state FIPS subset with real external-view coverage.",
    )
    ap.add_argument("--heldout_statefps", default="26")
    ap.add_argument("--joint_space", choices=["full", "coarse"], default="full")
    ap.add_argument("--models", default="acs,acs_external_raw,acs_external_resid")
    ap.add_argument("--external_label", default="external", help="Human-readable label for the external feature table.")
    ap.add_argument("--base_npz_all", type=pathlib.Path, default=None, help="Optional all-PUMA base joint vectors aligned with target_wide_csv.")
    ap.add_argument("--base_key_all", default="", help="NPZ key for --base_npz_all.")
    ap.add_argument("--base_label", default="acs_independence", help="Label for the active training/evaluation base.")
    ap.add_argument("--test_base_npz", type=pathlib.Path, default=None, help="Optional held-out test-base vectors, e.g., hierarchical outputs.")
    ap.add_argument("--test_base_keys", default="", help="Comma-separated npz keys to use as held-out test bases. Requires one held-out state.")
    ap.add_argument("--basis_dim", type=int, default=20)
    ap.add_argument("--mean_mode", choices=["fixed", "none"], default="fixed")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--coeff_l2", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--ipf_iters_train", type=int, default=20)
    ap.add_argument("--ipf_iters_eval", type=int, default=80)
    ap.add_argument("--clip_log_ratio", type=float, default=8.0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    out_dir = args.output_dir or pathlib.Path(f"outputs/_projection_aware_external_correction_{args.joint_space}_{_utc_ts()}")
    metrics_dir = out_dir / "metrics"
    history_dir = metrics_dir / "history"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    target_keys_all, p_true_full_all, _, _ = _load_target(args.target_wide_csv)
    eligible_statefps = _canon_statefps(args.eligible_statefps) if str(args.eligible_statefps).strip() else []
    eligible_mask_all = np.ones(int(target_keys_all.shape[0]), dtype=bool)
    if eligible_statefps:
        state_all = target_keys_all["statefp"].astype(str).str.zfill(2).to_numpy()
        eligible_mask_all = np.isin(state_all, np.asarray(eligible_statefps, dtype=object))
        if not np.any(eligible_mask_all):
            raise SystemExit(f"--eligible_statefps matched no target rows: {eligible_statefps}")
    target_keys = target_keys_all.loc[eligible_mask_all].reset_index(drop=True)
    p_true_full = p_true_full_all[eligible_mask_all]
    _, acs_marginals_full, _, x_acs_all, x_scale, *_ = _load_acs_conditions(args.condition_csv, target_keys)
    x_context = np.concatenate([x_acs_all, x_scale], axis=1)
    x_external, external_cols = _load_spatial(args.external_csv, target_keys)

    p_eq_full = _outer_joint([acs_marginals_full[v] for v in FULL_VARIABLE_ORDER])
    if args.joint_space == "full":
        shape = tuple(FULL_SHAPE)
        p_true = p_true_full
        p_base = p_eq_full
        target_marginals = [acs_marginals_full[v] for v in FULL_VARIABLE_ORDER]
    else:
        shape = tuple(COARSE_SHAPE)
        p_true = _aggregate_full_to_coarse(p_true_full)
        p_base = _aggregate_full_to_coarse(p_eq_full)
        target_marginals = _marginals_from_rows(p_base, shape=shape)

    base_label = str(args.base_label)
    if args.base_npz_all is not None:
        payload = np.load(args.base_npz_all, allow_pickle=False)
        key = str(args.base_key_all).strip()
        if not key:
            candidates = [k for k in payload.files if k != "p_true" and not k.endswith("_uid")]
            if len(candidates) != 1:
                raise SystemExit(f"--base_key_all required; candidates={candidates}")
            key = candidates[0]
        if key not in payload.files:
            raise SystemExit(f"--base_key_all not found in {args.base_npz_all}: {key}")
        loaded_base = _normalize_rows(np.asarray(payload[key], dtype=np.float64))
        if loaded_base.shape[0] == eligible_mask_all.shape[0] and loaded_base.shape[0] != p_true.shape[0]:
            loaded_base = loaded_base[eligible_mask_all]
        if args.joint_space == "coarse" and loaded_base.shape[1] == int(np.prod(FULL_SHAPE)):
            loaded_base = _aggregate_full_to_coarse(loaded_base)
        if loaded_base.shape != p_true.shape:
            raise SystemExit(f"base_npz_all shape mismatch: {loaded_base.shape} vs {p_true.shape}")
        if "p_true" in payload.files:
            loaded_true = np.asarray(payload["p_true"], dtype=np.float64)
            if loaded_true.shape[0] == eligible_mask_all.shape[0] and loaded_true.shape[0] != p_true.shape[0]:
                loaded_true = loaded_true[eligible_mask_all]
            if args.joint_space == "coarse" and loaded_true.shape[1] == int(np.prod(FULL_SHAPE)):
                loaded_true = _aggregate_full_to_coarse(loaded_true)
            if loaded_true.shape == p_true.shape:
                mismatch = float(np.mean(_row_tvd(_normalize_rows(loaded_true), p_true)))
                if mismatch > 1e-5:
                    raise SystemExit(f"base_npz_all p_true does not align with target; mean TVD mismatch={mismatch}")
        p_base = loaded_base
        base_label = str(args.base_label) if str(args.base_label) != "acs_independence" else key

    _, _, log_ratio = _residual_arrays(p_true, p_base, eps=float(args.eps))
    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()
    heldouts = _canon_statefps(args.heldout_statefps)
    requested_models = [x.strip() for x in str(args.models).split(",") if x.strip()]
    test_base_payload = None
    test_base_keys = [x.strip() for x in str(args.test_base_keys).split(",") if x.strip()]
    if args.test_base_npz is not None:
        test_base_payload = np.load(args.test_base_npz, allow_pickle=False)
        if not test_base_keys:
            test_base_keys = [k for k in test_base_payload.files if k != "p_true"]
        if len(heldouts) != 1:
            raise SystemExit("--test_base_npz currently supports exactly one held-out state.")

    eval_rows: list[dict[str, Any]] = []
    by_puma_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []

    for heldout in heldouts:
        test_mask = statefp == heldout
        if not np.any(test_mask):
            continue
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        mean_log_ratio, basis, basis_evr = _fit_log_ratio_basis(
            log_ratio,
            train_mask,
            basis_dim=int(args.basis_dim),
            seed=int(args.seed),
        )
        if args.mean_mode == "none":
            mean_log_ratio = np.zeros_like(mean_log_ratio)
        basis_rows.append(
            {
                "heldout_statefp": heldout,
                "basis_dim": int(basis.shape[0]),
                "basis_explained_variance": float(basis_evr),
                "mean_mode": args.mean_mode,
            }
        )
        feature_sets = _build_feature_sets(
            x_context=x_context,
            x_external=x_external,
            train_mask=train_mask,
            requested=requested_models,
        )

        p_base_test = p_base[test_idx]
        p_true_test = p_true[test_idx]
        test_bases: list[tuple[str, np.ndarray]] = [(base_label, p_base.copy())]
        if test_base_payload is not None:
            if "p_true" in test_base_payload.files:
                loaded_true = np.asarray(test_base_payload["p_true"], dtype=np.float64)
                if loaded_true.shape == p_true_test.shape:
                    mismatch = float(np.mean(_row_tvd(loaded_true, p_true_test)))
                    if mismatch > 1e-5:
                        raise SystemExit(f"test_base_npz p_true does not align with heldout target; mean TVD mismatch={mismatch}")
            for key in test_base_keys:
                if key not in test_base_payload.files:
                    raise SystemExit(f"missing --test_base_keys entry in npz: {key}")
                arr = _normalize_rows(np.asarray(test_base_payload[key], dtype=np.float64))
                if arr.shape != p_true_test.shape:
                    raise SystemExit(f"test base {key} shape mismatch: {arr.shape} vs {p_true_test.shape}")
                p_eval = p_base.copy()
                p_eval[test_idx] = arr
                test_bases.append((key, p_eval))
        mean_pre = _normalize_rows(p_base_test * np.exp(np.clip(mean_log_ratio.reshape(1, -1), -float(args.clip_log_ratio), float(args.clip_log_ratio))))
        mean_post = _ipf_project_rows_np(
            mean_pre,
            [m[test_idx] for m in target_marginals],
            shape=shape,
            max_iter=int(args.ipf_iters_eval),
            eps=float(args.eps),
        )
        mean_tvd = _row_tvd(mean_post, p_true_test)

        for base_name, p_base_eval in test_bases:
            p_eval_test = p_base_eval[test_idx]
            base_tvd = _row_tvd(p_eval_test, p_true_test)
            eval_rows.append(
                {
                    "heldout_statefp": heldout,
                    "test_base": base_name,
                    "model": "uncorrected_base",
                    "n_test": int(test_idx.size),
                    "test_tvd_mean": float(np.mean(base_tvd)),
                    "test_tvd_median": float(np.median(base_tvd)),
                    "delta_vs_base_mean": 0.0,
                }
            )
            if base_name == base_label and args.base_npz_all is None:
                eval_rows.append(
                    {
                        "heldout_statefp": heldout,
                        "test_base": base_name,
                        "model": "global_mean_logratio",
                        "n_test": int(test_idx.size),
                        "test_tvd_mean": float(np.mean(mean_tvd)),
                        "test_tvd_median": float(np.median(mean_tvd)),
                        "delta_vs_base_mean": float(np.mean(base_tvd - mean_tvd)),
                    }
                )

            for model_name, x in feature_sets.items():
                metrics, history, pred_test = _train_projection_aware_model(
                    x=x,
                    p_base=p_base,
                    p_base_eval=p_base_eval,
                    p_true=p_true,
                    target_marginals=target_marginals,
                    mean_log_ratio=mean_log_ratio,
                    basis=basis,
                    train_mask=train_mask,
                    test_mask=test_mask,
                    shape=shape,
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    coeff_l2=float(args.coeff_l2),
                    hidden_dim=int(args.hidden_dim),
                    dropout=float(args.dropout),
                    ipf_iters_train=int(args.ipf_iters_train),
                    ipf_iters_eval=int(args.ipf_iters_eval),
                    clip_log_ratio=float(args.clip_log_ratio),
                    eps=float(args.eps),
                    seed=int(args.seed),
                    device_name=str(args.device),
                )
                history["heldout_statefp"] = heldout
                history["test_base"] = base_name
                history["model"] = model_name
                history.to_csv(history_dir / f"history_state{heldout}_{base_name}_{model_name}.csv", index=False)
                pred_tvd = _row_tvd(pred_test, p_true_test)
                eval_rows.append(
                    {
                        "heldout_statefp": heldout,
                        "test_base": base_name,
                        "model": model_name,
                        "n_test": int(test_idx.size),
                        "test_tvd_mean": float(np.mean(pred_tvd)),
                        "test_tvd_median": float(np.median(pred_tvd)),
                        "delta_vs_base_mean": float(np.mean(base_tvd - pred_tvd)),
                        "delta_vs_global_mean_mean": float(np.mean(mean_tvd - pred_tvd)) if base_name == "acs_independence" else float("nan"),
                        **metrics,
                    }
                )
                for pos, idx in enumerate(test_idx):
                    by_puma_rows.append(
                        {
                            "heldout_statefp": heldout,
                            "test_base": base_name,
                            "statefp": str(target_keys.iloc[idx]["statefp"]).zfill(2),
                            "puma5": str(target_keys.iloc[idx]["puma5"]).zfill(5),
                            "puma_uid_key": str(target_keys.iloc[idx]["puma_uid_key"]),
                            "model": model_name,
                            "base_tvd": float(base_tvd[pos]),
                            "global_mean_tvd": float(mean_tvd[pos]),
                            "test_tvd": float(pred_tvd[pos]),
                            "delta_vs_base": float(base_tvd[pos] - pred_tvd[pos]),
                        }
                    )

    eval_df = pd.DataFrame(eval_rows)
    by_puma_df = pd.DataFrame(by_puma_rows)
    basis_df = pd.DataFrame(basis_rows)
    eval_df.to_csv(metrics_dir / "eval_by_state_model.csv", index=False)
    by_puma_df.to_csv(metrics_dir / "eval_by_puma.csv", index=False)
    basis_df.to_csv(metrics_dir / "basis_diagnostics.csv", index=False)

    run_summary = {
        "output_dir": str(out_dir),
        "joint_space": args.joint_space,
        "eligible_statefps": eligible_statefps,
        "heldout_statefps": heldouts,
        "models": requested_models,
        "base_npz_all": str(args.base_npz_all) if args.base_npz_all is not None else None,
        "base_key_all": str(args.base_key_all),
        "base_label": base_label,
        "test_base_npz": str(args.test_base_npz) if args.test_base_npz is not None else None,
        "test_base_keys": test_base_keys,
        "basis_dim": int(args.basis_dim),
        "mean_mode": args.mean_mode,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "coeff_l2": float(args.coeff_l2),
        "hidden_dim": int(args.hidden_dim),
        "ipf_iters_train": int(args.ipf_iters_train),
        "ipf_iters_eval": int(args.ipf_iters_eval),
        "external_csv": str(args.external_csv),
        "external_label": str(args.external_label),
        "external_feature_count": int(len(external_cols)),
        "eval_csv": str(metrics_dir / "eval_by_state_model.csv"),
        "by_puma_csv": str(metrics_dir / "eval_by_puma.csv"),
    }
    _write_json(out_dir / "run_summary.json", run_summary)
    if not eval_df.empty:
        print(eval_df.sort_values(["heldout_statefp", "test_tvd_mean"]).to_string(index=False))
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
