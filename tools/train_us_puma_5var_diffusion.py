#!/usr/bin/env python3
from __future__ import annotations

"""
Train 5-variable distribution-level diffusion on US PUMA joints and evaluate on Michigan.

Input:
- puma_5var_joint_wide.csv from tools/build_us_puma_5var_joint.py

Core setup:
- Train on non-MI PUMAs, test on MI PUMAs (leave_mi_out by default)
- Condition ablation: none vs marginal (all 5 marginals concatenated)
- Report raw and post-hoc-IPF metrics for fair comparison with IPF baseline
"""

import argparse
import datetime as _dt
import json
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


def _stable_hash_fold(values: list[str], *, n_folds: int, seed: int) -> dict[str, int]:
    import hashlib

    out: dict[str, int] = {}
    for v in values:
        h = hashlib.sha1((str(seed) + "::" + str(v)).encode("utf-8")).hexdigest()
        out[str(v)] = int(h[:8], 16) % int(n_folds)
    return out


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(z)
    d = np.sum(e, axis=1, keepdims=True)
    d = np.where(d <= 0, 1.0, d)
    return e / d


def _nd_independence(marginals: list[np.ndarray]) -> np.ndarray:
    out = np.asarray(marginals[0], dtype=float)
    for m in marginals[1:]:
        out = np.multiply.outer(out, np.asarray(m, dtype=float))
    out = np.asarray(out, dtype=float)
    out = out / max(float(out.sum()), 1e-12)
    return out.reshape(-1)


def _ipf_nd(
    *,
    seed_joint: np.ndarray,
    target_marginals: list[np.ndarray],
    shape: tuple[int, ...],
    max_iter: int = 200,
    tol: float = 1e-10,
    eps: float = 1e-12,
) -> np.ndarray:
    x = np.asarray(seed_joint, dtype=float).reshape(shape).copy()
    x = np.clip(x, 0.0, None)
    x = x / max(float(x.sum()), eps)

    targets = []
    for i, t in enumerate(target_marginals):
        tt = np.asarray(t, dtype=float).reshape(-1)
        if tt.size != shape[i]:
            raise ValueError(f"target marginal size mismatch at axis={i}: {tt.size} vs {shape[i]}")
        tt = np.clip(tt, 0.0, None)
        tt = tt / max(float(tt.sum()), eps)
        targets.append(tt)

    axes = list(range(len(shape)))
    for _ in range(int(max_iter)):
        max_err = 0.0
        for axis, tgt in enumerate(targets):
            sum_axes = tuple(a for a in axes if a != axis)
            cur = x.sum(axis=sum_axes)
            err = float(np.max(np.abs(cur - tgt)))
            max_err = max(max_err, err)
            factor = np.zeros_like(cur)
            m = cur > 0
            factor[m] = tgt[m] / (cur[m] + eps)
            reshape = [1] * len(shape)
            reshape[axis] = shape[axis]
            x = x * factor.reshape(reshape)
            if bool((tgt <= 0).any()):
                slicer = [slice(None)] * len(shape)
                for k, tv in enumerate(tgt):
                    if tv <= 0:
                        slicer[axis] = k
                        x[tuple(slicer)] = 0.0
        x = np.clip(x, 0.0, None)
        x = x / max(float(x.sum()), eps)
        if max_err < float(tol):
            break
    return x.reshape(-1)


def _marginal_from_joint(p: np.ndarray, *, shape: tuple[int, ...], axis: int) -> np.ndarray:
    tab = np.asarray(p, dtype=float).reshape(shape)
    sum_axes = tuple(i for i in range(len(shape)) if i != int(axis))
    out = tab.sum(axis=sum_axes)
    out = out / max(float(out.sum()), 1e-12)
    return out


def _apply_posthoc_ipf(*, policy: str, cond_name: str) -> bool:
    p = str(policy).strip().lower()
    if p == "none":
        return False
    if p == "all":
        return True
    if p == "marginal":
        return str(cond_name).strip().lower() == "marginal"
    raise ValueError(f"Unsupported posthoc_ipf_policy: {policy}")


def main() -> None:
    torch = _require_torch()

    ap = argparse.ArgumentParser(prog="train_us_puma_5var_diffusion")
    ap.add_argument("--joint_wide_csv", required=True, help="Path to puma_5var_joint_wide.csv")
    ap.add_argument("--conditions", default="none,marginal", help='Comma-separated: "none,marginal"')
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
    ap.add_argument("--n_eval_joint_samples", type=int, default=128)
    ap.add_argument("--ipf_iters", type=int, default=200)
    ap.add_argument(
        "--posthoc_ipf_policy",
        choices=["none", "marginal", "all"],
        default="marginal",
        help="Apply post-hoc IPF on diffusion output during evaluation: none | marginal | all.",
    )
    ap.add_argument("--out_dir", default=None, help="Default: outputs/<run_id>")
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    in_path = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"joint_wide_csv not found: {in_path}")

    run_id = f"_us_puma_5var_diffusion_{_dt.datetime.now(_dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else (_REPO_ROOT / "outputs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    required = {"statefp", "puma", "puma_uid", "total_person_weight"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"joint_wide_csv missing columns: {missing}")

    p_joint_cols = sorted([c for c in df.columns if c.startswith("p_joint_")], key=lambda x: int(x.split("_")[-1]))
    var_specs = [
        ("age", "p_age_"),
        ("sex", "p_sex_"),
        ("income", "p_income_"),
        ("schl", "p_schl_"),
        ("esr", "p_esr_"),
    ]
    marg_cols_by_var: dict[str, list[str]] = {}
    for vn, pref in var_specs:
        cols = sorted([c for c in df.columns if c.startswith(pref)], key=lambda x: int(x.split("_")[-1]))
        if not cols:
            raise SystemExit(f"Missing marginal columns for {vn} ({pref}*)")
        marg_cols_by_var[vn] = cols

    shape = tuple(len(marg_cols_by_var[vn]) for vn, _ in var_specs)
    K = int(np.prod(shape))
    if len(p_joint_cols) != K:
        raise SystemExit(f"Joint dim mismatch: got {len(p_joint_cols)} joint cols, expected {K} from shape={shape}")

    df["statefp"] = df["statefp"].astype(str).str.zfill(2)
    is_mi = df["statefp"] == "26"
    if int(is_mi.sum()) == 0:
        raise SystemExit("No Michigan rows found (statefp==26).")

    p_joint = df[p_joint_cols].to_numpy(dtype=np.float32)
    p_joint = np.clip(p_joint, 0.0, None)
    p_joint = p_joint / np.maximum(p_joint.sum(axis=1, keepdims=True), 1e-12)
    totals = pd.to_numeric(df["total_person_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ids = df["puma_uid"].astype(str).tolist()

    marg_by_var: dict[str, np.ndarray] = {}
    for vn, _ in var_specs:
        arr = df[marg_cols_by_var[vn]].to_numpy(dtype=np.float32)
        arr = np.clip(arr, 0.0, None)
        arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-12)
        marg_by_var[vn] = arr

    x_all = np.log(np.clip(p_joint, 0.0, None) + 1e-6).astype(np.float32)
    cond_marg = np.concatenate([marg_by_var[vn] for vn, _ in var_specs], axis=1).astype(np.float32)

    folds: list[tuple[str, np.ndarray, np.ndarray]] = []
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
    allowed = {"none", "marginal"}
    bad = [c for c in conditions if c not in allowed]
    if bad:
        raise SystemExit(f"Unsupported conditions: {bad}, allowed={sorted(allowed)}")

    internal_by_condition: dict[str, Any] = {}
    baselines_by_fold: dict[str, dict[str, Any]] = {"independence": {}, "ipf_train_seed": {}}

    for cond_name in conditions:
        cond_dim = 0 if cond_name == "none" else int(cond_marg.shape[1])
        cond_fold_metrics: dict[str, Any] = {}

        for fold_name, train_idx, test_idx in folds:
            x_train = x_all[train_idx]
            cond_train = cond_marg[train_idx] if cond_name == "marginal" else None
            cond_test = cond_marg[test_idx] if cond_name == "marginal" else None

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

            # Build train seed for IPF baseline (weighted mean joint).
            seed_counts = np.zeros((K,), dtype=float)
            for i, idx in enumerate(train_idx):
                w = float(max(totals[train_idx][i], 0.0))
                seed_counts += w * p_joint[idx]
            if float(seed_counts.sum()) <= 0:
                seed_joint = np.mean(p_joint[train_idx], axis=0)
                seed_joint = seed_joint / max(float(seed_joint.sum()), 1e-12)
            else:
                seed_joint = seed_counts / float(seed_counts.sum())

            ind_tvd_vals: list[float] = []
            ipf_tvd_vals: list[float] = []

            tvd_joint_vals: list[float] = []
            cos_vals: list[float] = []
            raw_tvd_joint_vals: list[float] = []
            raw_cos_vals: list[float] = []
            var_eval: dict[str, list[float]] = {vn: [] for vn, _ in var_specs}
            var_raw: dict[str, list[float]] = {vn: [] for vn, _ in var_specs}

            for j, idx in enumerate(test_idx):
                p_true = p_joint[idx]
                t_margs = [marg_by_var[vn][idx] for vn, _ in var_specs]

                p_ind = _nd_independence(t_margs)
                p_ipf = _ipf_nd(
                    seed_joint=seed_joint,
                    target_marginals=t_margs,
                    shape=shape,
                    max_iter=int(args.ipf_iters),
                )
                ind_tvd_vals.append(_tvd(p_ind, p_true))
                ipf_tvd_vals.append(_tvd(p_ipf, p_true))

                n_eval = int(args.n_eval_joint_samples)
                if cond_test is None:
                    z = model.sample(n=n_eval, cond=None, device=args.device).numpy()
                else:
                    c = np.repeat(cond_test[j : j + 1], repeats=n_eval, axis=0).astype(np.float32)
                    z = model.sample(n=n_eval, cond=torch.from_numpy(c), device=args.device).numpy()
                p_draws = _softmax_rows(z.astype(np.float64))
                p_hat_raw = np.mean(p_draws, axis=0)
                p_hat_raw = p_hat_raw / max(float(p_hat_raw.sum()), 1e-12)

                p_hat_eval = p_hat_raw
                if _apply_posthoc_ipf(policy=str(args.posthoc_ipf_policy), cond_name=cond_name):
                    p_hat_eval = _ipf_nd(
                        seed_joint=p_hat_raw,
                        target_marginals=t_margs,
                        shape=shape,
                        max_iter=int(args.ipf_iters),
                    )

                raw_tvd_joint_vals.append(_tvd(p_hat_raw, p_true))
                tvd_joint_vals.append(_tvd(p_hat_eval, p_true))
                raw_cos_vals.append(_cosine(p_hat_raw, p_true))
                cos_vals.append(_cosine(p_hat_eval, p_true))

                for axis, (vn, _) in enumerate(var_specs):
                    mr = _marginal_from_joint(p_hat_raw, shape=shape, axis=axis)
                    me = _marginal_from_joint(p_hat_eval, shape=shape, axis=axis)
                    mt = np.asarray(t_margs[axis], dtype=float)
                    var_raw[vn].append(_tvd(mr, mt))
                    var_eval[vn].append(_tvd(me, mt))

            if fold_name not in baselines_by_fold["independence"]:
                baselines_by_fold["independence"][fold_name] = {"tvd_joint": _summ(ind_tvd_vals)}
                baselines_by_fold["ipf_train_seed"][fold_name] = {"tvd_joint": _summ(ipf_tvd_vals)}

            fold_obj: dict[str, Any] = {
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "posthoc_ipf_applied": bool(_apply_posthoc_ipf(policy=str(args.posthoc_ipf_policy), cond_name=cond_name)),
                "tvd_joint": _summ(tvd_joint_vals),
                "cosine_joint": _summ(cos_vals),
                "tvd_joint_raw": _summ(raw_tvd_joint_vals),
                "cosine_joint_raw": _summ(raw_cos_vals),
            }
            for vn, _ in var_specs:
                fold_obj[f"tvd_{vn}"] = _summ(var_eval[vn])
                fold_obj[f"tvd_{vn}_raw"] = _summ(var_raw[vn])
            cond_fold_metrics[fold_name] = fold_obj

        fold_names = sorted(cond_fold_metrics.keys())
        out_overall: dict[str, Any] = {
            "tvd_joint": _summ([float(cond_fold_metrics[f]["tvd_joint"]["mean"]) for f in fold_names]),
            "cosine_joint": _summ([float(cond_fold_metrics[f]["cosine_joint"]["mean"]) for f in fold_names]),
            "tvd_joint_raw": _summ([float(cond_fold_metrics[f]["tvd_joint_raw"]["mean"]) for f in fold_names]),
            "cosine_joint_raw": _summ([float(cond_fold_metrics[f]["cosine_joint_raw"]["mean"]) for f in fold_names]),
        }
        for vn, _ in var_specs:
            out_overall[f"tvd_{vn}"] = _summ([float(cond_fold_metrics[f][f"tvd_{vn}"]["mean"]) for f in fold_names])
            out_overall[f"tvd_{vn}_raw"] = _summ([float(cond_fold_metrics[f][f"tvd_{vn}_raw"]["mean"]) for f in fold_names])
        internal_by_condition[cond_name] = {"overall": out_overall, "by_fold": cond_fold_metrics}

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
        "shape": {"age": shape[0], "sex": shape[1], "income": shape[2], "schl": shape[3], "esr": shape[4]},
        "K_joint_dim": int(K),
        "eval_mode": str(args.eval_mode),
        "n_folds": int(len(folds)),
        "conditions": conditions,
        "timesteps": int(args.timesteps),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_dims": list(hidden_dims),
        "condition_injection": str(args.condition_injection),
        "film_hidden_dim": int(args.film_hidden_dim),
        "n_eval_joint_samples": int(args.n_eval_joint_samples),
        "ipf_iters": int(args.ipf_iters),
        "posthoc_ipf_policy": str(args.posthoc_ipf_policy),
        "seed": int(args.seed),
        "device": args.device,
    }

    _write_json(out_dir / "run_summary.json", run_summary)
    _write_json(out_dir / "metrics" / "internal_acs_holdout.json", {"by_condition": internal_by_condition})
    _write_json(out_dir / "metrics" / "baselines_internal.json", baselines_internal)
    _write_json(out_dir / "metrics" / "ablation_summary.json", ablation_summary)
    print(f"[ok] wrote: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()

