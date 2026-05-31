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
from sklearn.linear_model import Ridge

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.model.external_c2f_full_earn_schema import FULL_SHAPE, FULL_VARIABLE_ORDER
from tools.experimental.representation.projection_aware_external_correction import (
    _fit_log_ratio_basis,
    _ipf_project_rows_np,
    _marginals_from_rows,
    _normalize_rows,
    _row_tvd,
    _standardize_train_test,
    _train_projection_aware_model,
)
from tools.experimental.representation.ssl_copula_residual_probe import (
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


@dataclass(frozen=True)
class Candidate:
    name: str
    view: str
    mode: str


def _parse_candidates(raw: str) -> list[Candidate]:
    out: list[Candidate] = []
    for item in [x.strip() for x in str(raw).split(",") if x.strip()]:
        parts = item.split(":")
        if len(parts) == 1:
            name = parts[0]
            if name == "base":
                out.append(Candidate(name=name, view="base", mode="base"))
            elif name == "acs":
                out.append(Candidate(name=name, view="acs", mode="acs"))
            else:
                raise SystemExit(f"candidate must be name:view:mode for external views: {item}")
        elif len(parts) == 3:
            name, view, mode = parts
            if mode not in {"raw", "resid"}:
                raise SystemExit(f"external candidate mode must be raw or resid: {item}")
            out.append(Candidate(name=name, view=view, mode=mode))
        else:
            raise SystemExit(f"invalid candidate spec: {item}")
    names = [c.name for c in out]
    if len(names) != len(set(names)):
        raise SystemExit(f"duplicate candidate names: {names}")
    return out


def _make_features(
    *,
    candidate: Candidate,
    x_context: np.ndarray,
    external_by_view: dict[str, np.ndarray],
    train_mask: np.ndarray,
) -> np.ndarray | None:
    if candidate.mode == "base":
        return None
    context_z, _ = _standardize_train_test(x_context, train_mask)
    if candidate.mode == "acs":
        return context_z
    if candidate.view not in external_by_view:
        raise SystemExit(f"candidate view not found: {candidate.view}")
    external_z, _ = _standardize_train_test(external_by_view[candidate.view], train_mask)
    if candidate.mode == "raw":
        return np.concatenate([context_z, external_z], axis=1)
    model = Ridge(alpha=10.0)
    model.fit(context_z[train_mask], external_z[train_mask])
    residual = external_z - model.predict(context_z)
    return np.concatenate([context_z, residual], axis=1)


def _load_base_vectors(path: pathlib.Path, key: str, expected_shape: tuple[int, int]) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if key not in payload.files:
        raise SystemExit(f"base key not found in {path}: {key}; candidates={payload.files}")
    arr = _normalize_rows(np.asarray(payload[key], dtype=np.float64))
    if arr.shape != expected_shape:
        raise SystemExit(f"base vector shape mismatch: {arr.shape} vs {expected_shape}")
    return arr


def _fit_predict_candidate(
    *,
    candidate: Candidate,
    x_context: np.ndarray,
    external_by_view: dict[str, np.ndarray],
    p_base: np.ndarray,
    p_true: np.ndarray,
    target_marginals: list[np.ndarray],
    log_ratio: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    shape: tuple[int, ...],
    basis_dim: int,
    mean_mode: str,
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
    device: str,
) -> tuple[np.ndarray, dict[str, float]]:
    eval_idx = np.where(eval_mask)[0]
    if candidate.mode == "base":
        pred = p_base[eval_idx]
        return pred, {
            "train_tvd_eval": float("nan"),
            "eval_tvd": float(np.mean(_row_tvd(pred, p_true[eval_idx]))),
            "basis_evr": float("nan"),
        }
    mean_log_ratio, basis, basis_evr = _fit_log_ratio_basis(
        log_ratio,
        train_mask,
        basis_dim=int(basis_dim),
        seed=int(seed),
    )
    if mean_mode == "none":
        mean_log_ratio = np.zeros_like(mean_log_ratio)
    x = _make_features(
        candidate=candidate,
        x_context=x_context,
        external_by_view=external_by_view,
        train_mask=train_mask,
    )
    if x is None:
        raise RuntimeError("unreachable")
    metrics, _, pred = _train_projection_aware_model(
        x=x,
        p_base=p_base,
        p_base_eval=p_base,
        p_true=p_true,
        target_marginals=target_marginals,
        mean_log_ratio=mean_log_ratio,
        basis=basis,
        train_mask=train_mask,
        test_mask=eval_mask,
        shape=shape,
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        coeff_l2=float(coeff_l2),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        ipf_iters_train=int(ipf_iters_train),
        ipf_iters_eval=int(ipf_iters_eval),
        clip_log_ratio=float(clip_log_ratio),
        eps=float(eps),
        seed=int(seed),
        device_name=device,
    )
    metrics = dict(metrics)
    metrics["eval_tvd"] = float(np.mean(_row_tvd(pred, p_true[eval_idx])))
    metrics["basis_evr"] = float(basis_evr)
    return pred, metrics


def _fit_simplex_stack_weights(
    preds: np.ndarray,
    p_true: np.ndarray,
    *,
    maxiter: int = 300,
) -> tuple[np.ndarray, float, str]:
    n_candidates = int(preds.shape[0])
    x0 = np.zeros(n_candidates, dtype=np.float64)
    x0[0] = 1.0

    def objective(w: np.ndarray) -> float:
        w = np.asarray(w, dtype=np.float64)
        pred = np.tensordot(w, preds, axes=(0, 0))
        return float(np.mean(_row_tvd(pred, p_true)))

    try:
        from scipy.optimize import minimize

        best = None
        starts = [x0, np.full(n_candidates, 1.0 / n_candidates)]
        for i in range(1, n_candidates):
            s = np.zeros(n_candidates, dtype=np.float64)
            s[i] = 1.0
            starts.append(s)
        for start in starts:
            res = minimize(
                objective,
                start,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n_candidates,
                constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
                options={"maxiter": int(maxiter), "ftol": 1e-10, "disp": False},
            )
            if res.success:
                val = objective(res.x)
                if best is None or val < best[0]:
                    best = (val, np.asarray(res.x, dtype=np.float64), "slsqp")
        if best is not None:
            w = np.clip(best[1], 0.0, 1.0)
            w = w / np.clip(w.sum(), 1e-12, None)
            return w, objective(w), best[2]
    except Exception:
        pass

    # Coarse fallback. This is intentionally conservative and keeps the base
    # candidate in the simplex, so stacking cannot be forced to use external views.
    step = 0.1
    units = int(round(1.0 / step))
    best_w = x0.copy()
    best_val = objective(best_w)

    def rec(prefix: list[int], remaining: int, k_left: int) -> None:
        nonlocal best_w, best_val
        if k_left == 1:
            counts = prefix + [remaining]
            w = np.asarray(counts, dtype=np.float64) / units
            val = objective(w)
            if val < best_val:
                best_val = val
                best_w = w
            return
        for c in range(remaining + 1):
            rec(prefix + [c], remaining - c, k_left - 1)

    rec([], units, n_candidates)
    return best_w, best_val, "grid_0.1"


def main() -> int:
    data_root = pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data")
    ap = argparse.ArgumentParser(description="Projection-aware view gate and stacking over single-view residual adapters.")
    ap.add_argument("--target_wide_csv", type=pathlib.Path, default=data_root / "us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv")
    ap.add_argument("--condition_csv", type=pathlib.Path, default=data_root / "us/processed/external_conditions/extcond_v1_agesex_earn_v1_acs5_2022_puma_us.csv")
    ap.add_argument("--external_view", action="append", default=[], help="view_name=csv_path. Repeat for each external view.")
    ap.add_argument("--eligible_statefps", default="12,26,48,55")
    ap.add_argument("--heldout_statefp", default="26")
    ap.add_argument("--validation_statefps", default="12,48,55")
    ap.add_argument("--base_npz_all", type=pathlib.Path, required=True)
    ap.add_argument("--base_key_all", required=True)
    ap.add_argument(
        "--candidates",
        default="base,acs,poi_raw:poi:raw,lodes_resid:lodes:resid,viirs_resid:viirs:resid",
    )
    ap.add_argument("--basis_dim", type=int, default=40)
    ap.add_argument("--mean_mode", choices=["none", "fixed"], default="none")
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

    out_dir = args.output_dir or pathlib.Path(f"outputs/_projection_aware_view_gate_{_utc_ts()}")
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    candidates = _parse_candidates(args.candidates)
    external_paths: dict[str, pathlib.Path] = {}
    for item in args.external_view:
        if "=" not in item:
            raise SystemExit(f"--external_view must be view_name=csv_path: {item}")
        name, path = item.split("=", 1)
        external_paths[name.strip()] = pathlib.Path(path)

    needed_views = {c.view for c in candidates if c.mode not in {"base", "acs"}}
    missing_views = sorted(v for v in needed_views if v not in external_paths)
    if missing_views:
        raise SystemExit(f"missing --external_view entries for: {missing_views}")

    target_keys_all, p_true_full_all, _, _ = _load_target(args.target_wide_csv)
    eligible_statefps = _canon_statefps(args.eligible_statefps)
    state_all = target_keys_all["statefp"].astype(str).str.zfill(2).to_numpy()
    eligible_mask_all = np.isin(state_all, np.asarray(eligible_statefps, dtype=object))
    target_keys = target_keys_all.loc[eligible_mask_all].reset_index(drop=True)
    p_true = p_true_full_all[eligible_mask_all]
    _, acs_marginals_full, _, x_acs_all, x_scale, *_ = _load_acs_conditions(args.condition_csv, target_keys)
    x_context = np.concatenate([x_acs_all, x_scale], axis=1)
    target_marginals = [acs_marginals_full[v] for v in FULL_VARIABLE_ORDER]
    p_eq_full = _outer_joint(target_marginals)
    p_base_all = _load_base_vectors(args.base_npz_all, args.base_key_all, p_true_full_all.shape)
    p_base = p_base_all[eligible_mask_all]
    if p_base.shape != p_true.shape:
        raise SystemExit(f"filtered base shape mismatch: {p_base.shape} vs {p_true.shape}")
    _, _, log_ratio = _residual_arrays(p_true, p_base, eps=float(args.eps))
    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()
    heldout = str(args.heldout_statefp).zfill(2)
    validation_states = _canon_statefps(args.validation_statefps)
    if heldout in validation_states:
        raise SystemExit("--validation_statefps must not include held-out state")

    external_by_view = {
        name: _load_spatial(path, target_keys)[0]
        for name, path in external_paths.items()
    }
    shape = tuple(FULL_SHAPE)

    validation_rows: list[dict[str, Any]] = []
    val_pred_by_candidate: dict[str, list[np.ndarray]] = {c.name: [] for c in candidates}
    val_true_parts: list[np.ndarray] = []

    for val_state in validation_states:
        val_mask = statefp == val_state
        train_mask = (statefp != heldout) & (statefp != val_state)
        if not np.any(val_mask):
            raise SystemExit(f"validation state has no rows after filtering: {val_state}")
        if not np.any(train_mask):
            raise SystemExit(f"empty training mask for validation state: {val_state}")
        val_idx = np.where(val_mask)[0]
        val_true_parts.append(p_true[val_idx])
        for candidate in candidates:
            pred, metrics = _fit_predict_candidate(
                candidate=candidate,
                x_context=x_context,
                external_by_view=external_by_view,
                p_base=p_base,
                p_true=p_true,
                target_marginals=target_marginals,
                log_ratio=log_ratio,
                train_mask=train_mask,
                eval_mask=val_mask,
                shape=shape,
                basis_dim=int(args.basis_dim),
                mean_mode=args.mean_mode,
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
                device=args.device,
            )
            val_pred_by_candidate[candidate.name].append(pred)
            tvd = _row_tvd(pred, p_true[val_idx])
            validation_rows.append(
                {
                    "validation_statefp": val_state,
                    "candidate": candidate.name,
                    "n_val": int(val_idx.size),
                    "val_tvd_mean": float(np.mean(tvd)),
                    "val_tvd_median": float(np.median(tvd)),
                    **metrics,
                }
            )

    val_true = np.vstack(val_true_parts)
    candidate_names = [c.name for c in candidates]
    val_preds = np.stack([np.vstack(val_pred_by_candidate[name]) for name in candidate_names], axis=0)
    val_summary = []
    for i, name in enumerate(candidate_names):
        tvd = _row_tvd(val_preds[i], val_true)
        val_summary.append({"candidate": name, "validation_tvd_mean": float(np.mean(tvd)), "validation_tvd_median": float(np.median(tvd))})
    val_summary_df = pd.DataFrame(val_summary).sort_values("validation_tvd_mean")
    selected_candidate = str(val_summary_df.iloc[0]["candidate"])
    weights, stack_val_tvd, stack_method = _fit_simplex_stack_weights(val_preds, val_true)

    final_train_mask = statefp != heldout
    test_mask = statefp == heldout
    test_idx = np.where(test_mask)[0]
    test_true = p_true[test_idx]
    test_pred_by_candidate: dict[str, np.ndarray] = {}
    test_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        pred, metrics = _fit_predict_candidate(
            candidate=candidate,
            x_context=x_context,
            external_by_view=external_by_view,
            p_base=p_base,
            p_true=p_true,
            target_marginals=target_marginals,
            log_ratio=log_ratio,
            train_mask=final_train_mask,
            eval_mask=test_mask,
            shape=shape,
            basis_dim=int(args.basis_dim),
            mean_mode=args.mean_mode,
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
            device=args.device,
        )
        test_pred_by_candidate[candidate.name] = pred
        tvd = _row_tvd(pred, test_true)
        test_rows.append(
            {
                "method": f"candidate::{candidate.name}",
                "heldout_statefp": heldout,
                "n_test": int(test_idx.size),
                "test_tvd_mean": float(np.mean(tvd)),
                "test_tvd_median": float(np.median(tvd)),
                **metrics,
            }
        )

    selected_pred = test_pred_by_candidate[selected_candidate]
    selected_tvd = _row_tvd(selected_pred, test_true)
    test_rows.append(
        {
            "method": "validation_selected",
            "selected_candidate": selected_candidate,
            "heldout_statefp": heldout,
            "n_test": int(test_idx.size),
            "test_tvd_mean": float(np.mean(selected_tvd)),
            "test_tvd_median": float(np.median(selected_tvd)),
        }
    )
    test_preds = np.stack([test_pred_by_candidate[name] for name in candidate_names], axis=0)
    stack_pred = np.tensordot(weights, test_preds, axes=(0, 0))
    # The convex combination of feasible projected distributions remains feasible
    # for each held-out PUMA, but one final IPF pass removes numerical drift.
    stack_pred = _ipf_project_rows_np(
        stack_pred,
        [m[test_idx] for m in target_marginals],
        shape=shape,
        max_iter=int(args.ipf_iters_eval),
        eps=float(args.eps),
    )
    stack_tvd = _row_tvd(stack_pred, test_true)
    test_rows.append(
        {
            "method": "validation_stack",
            "selected_candidate": "",
            "heldout_statefp": heldout,
            "n_test": int(test_idx.size),
            "test_tvd_mean": float(np.mean(stack_tvd)),
            "test_tvd_median": float(np.median(stack_tvd)),
            "stack_val_tvd": float(stack_val_tvd),
            "stack_method": stack_method,
        }
    )

    pd.DataFrame(validation_rows).to_csv(metrics_dir / "validation_by_fold_candidate.csv", index=False)
    val_summary_df.to_csv(metrics_dir / "validation_summary_by_candidate.csv", index=False)
    pd.DataFrame(test_rows).sort_values("test_tvd_mean").to_csv(metrics_dir / "test_summary.csv", index=False)
    pd.DataFrame({"candidate": candidate_names, "weight": weights}).to_csv(metrics_dir / "stack_weights.csv", index=False)
    np.savez_compressed(
        metrics_dir / "test_predictions.npz",
        p_true=test_true,
        stack_pred=stack_pred,
        **{f"pred_{name}": arr for name, arr in test_pred_by_candidate.items()},
    )

    run_summary = {
        "output_dir": str(out_dir),
        "eligible_statefps": eligible_statefps,
        "heldout_statefp": heldout,
        "validation_statefps": validation_states,
        "base_npz_all": str(args.base_npz_all),
        "base_key_all": str(args.base_key_all),
        "external_views": {k: str(v) for k, v in external_paths.items()},
        "candidates": [c.__dict__ for c in candidates],
        "selected_candidate": selected_candidate,
        "stack_method": stack_method,
        "stack_validation_tvd": float(stack_val_tvd),
        "basis_dim": int(args.basis_dim),
        "mean_mode": args.mean_mode,
        "epochs": int(args.epochs),
        "coeff_l2": float(args.coeff_l2),
        "metrics_dir": str(metrics_dir),
    }
    _write_json(out_dir / "run_summary.json", run_summary)
    print(pd.read_csv(metrics_dir / "validation_summary_by_candidate.csv").to_string(index=False))
    print(pd.read_csv(metrics_dir / "stack_weights.csv").to_string(index=False))
    print(pd.read_csv(metrics_dir / "test_summary.csv").to_string(index=False))
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
