#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _digits_only(v: object) -> str:
    return "".join(ch for ch in str(v).strip() if ch.isdigit())


def _canon_statefp(v: object) -> str:
    d = _digits_only(v)
    if not d:
        return ""
    return str(int(d)).zfill(2)


def _canon_puma5(v: object) -> str:
    d = _digits_only(v)
    if not d:
        return ""
    return str(int(d)).zfill(5)


def _canon_uid(statefp: object, puma: object) -> str:
    s = _canon_statefp(statefp)
    p = _canon_puma5(puma)
    if not s or not p:
        return ""
    return s + p


def _stable_hash_fold(values: list[str], *, n_folds: int, seed: int) -> dict[str, int]:
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


def _tvd(p: Any, q: Any) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    return 0.5 * float(np.abs(p - q).sum())


def _summ(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
        "n": int(arr.size),
    }


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


def _pairwise_from_joint(
    *,
    p_joint: np.ndarray,
    shape: tuple[int, ...],
    var_names: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    n = int(p_joint.shape[0])
    d = len(shape)
    tab = np.asarray(p_joint, dtype=np.float32).reshape((n,) + tuple(shape))
    blocks: list[np.ndarray] = []
    pair_dims: dict[str, int] = {}
    for i in range(d):
        for j in range(i + 1, d):
            keep = {i, j}
            reduce_axes = tuple(ax + 1 for ax in range(d) if ax not in keep)
            m = tab.sum(axis=reduce_axes)
            m = m.reshape(n, -1)
            m = m / np.maximum(m.sum(axis=1, keepdims=True), 1e-12)
            m = m.astype(np.float32)
            blocks.append(m)
            pair_dims[f"{var_names[i]}__{var_names[j]}"] = int(m.shape[1])
    if not blocks:
        return np.zeros((n, 0), dtype=np.float32), {}
    out = np.concatenate(blocks, axis=1).astype(np.float32)
    return out, pair_dims


@dataclass
class EvalData5Var:
    df: pd.DataFrame
    ids: list[str]
    p_joint: np.ndarray
    x_log_all: np.ndarray
    totals: np.ndarray
    shape: tuple[int, ...]
    var_names: list[str]
    marg_by_var: dict[str, np.ndarray]
    cond_map: dict[str, np.ndarray | None]
    test_idx: np.ndarray
    train_idx: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray


def load_eval_data(
    *,
    joint_wide_csv: pathlib.Path,
    condition_names: list[str],
    eval_mode: str = "leave_mi_out",
    n_folds: int = 5,
    fold_index: int = 0,
    seed: int = 0,
) -> EvalData5Var:
    df = pd.read_csv(joint_wide_csv)
    required = {"statefp", "puma", "puma_uid", "total_person_weight"}
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise SystemExit(f"joint_wide csv missing columns: {miss}")

    df["statefp"] = df["statefp"].map(_canon_statefp)
    df["puma5"] = df["puma"].map(_canon_puma5)
    df["puma"] = df["puma5"].map(lambda x: str(int(x)) if x else "")
    df["puma_uid"] = df.apply(lambda r: _canon_uid(r["statefp"], r["puma5"]), axis=1)
    bad_uid = int((df["puma_uid"] == "").sum())
    if bad_uid > 0:
        raise SystemExit(f"Invalid puma_uid rows after canonicalization: {bad_uid}")

    p_joint_cols = sorted([c for c in df.columns if c.startswith("p_joint_")], key=lambda x: int(x.split("_")[-1]))
    var_specs = [("age", "p_age_"), ("sex", "p_sex_"), ("income", "p_income_"), ("schl", "p_schl_"), ("esr", "p_esr_")]
    marg_cols_by_var: dict[str, list[str]] = {}
    for vn, pref in var_specs:
        cols = sorted([c for c in df.columns if c.startswith(pref)], key=lambda x: int(x.split("_")[-1]))
        if not cols:
            raise SystemExit(f"Missing marginal columns for {vn} ({pref}*)")
        marg_cols_by_var[vn] = cols

    shape = tuple(len(marg_cols_by_var[vn]) for vn, _ in var_specs)
    k = int(np.prod(shape))
    if len(p_joint_cols) != k:
        raise SystemExit(f"Joint dim mismatch: {len(p_joint_cols)} vs expected {k} from shape={shape}")

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

    cond_marg = np.concatenate([marg_by_var[vn] for vn, _ in var_specs], axis=1).astype(np.float32)
    cond_pairwise, _ = _pairwise_from_joint(
        p_joint=p_joint,
        shape=shape,
        var_names=[vn for vn, _ in var_specs],
    )
    cond_map: dict[str, np.ndarray | None] = {
        "none": None,
        "marginal": cond_marg,
        "pairwise": cond_pairwise,
        "marginal_pairwise": np.concatenate([cond_marg, cond_pairwise], axis=1).astype(np.float32),
    }
    for c in condition_names:
        if c not in cond_map:
            raise SystemExit(f"Unsupported condition for eval helper: {c}")

    is_mi = df["statefp"] == "26"
    if int(is_mi.sum()) == 0:
        raise SystemExit("No Michigan rows found (statefp==26).")

    if str(eval_mode) == "leave_mi_out":
        train_idx = np.where(~is_mi.to_numpy(dtype=bool))[0]
        test_idx = np.where(is_mi.to_numpy(dtype=bool))[0]
    else:
        mi_ids = [ids[i] for i in np.where(is_mi.to_numpy(dtype=bool))[0]]
        fold_map = _stable_hash_fold(sorted(set(mi_ids)), n_folds=int(n_folds), seed=int(seed))
        te_mask = np.array([(is_mi.iloc[i] and fold_map.get(ids[i], -1) == int(fold_index)) for i in range(len(ids))], dtype=bool)
        tr_mask = ~te_mask
        train_idx = np.where(tr_mask)[0]
        test_idx = np.where(te_mask)[0]
    if train_idx.size == 0 or test_idx.size == 0:
        raise SystemExit("Invalid train/test split in eval helper.")

    x_log_all = np.log(np.clip(p_joint, 0.0, None) + 1e-6).astype(np.float32)
    x_train_log = x_log_all[train_idx]
    x_mean = x_train_log.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = x_train_log.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)

    return EvalData5Var(
        df=df,
        ids=ids,
        p_joint=p_joint,
        x_log_all=x_log_all,
        totals=totals,
        shape=shape,
        var_names=[vn for vn, _ in var_specs],
        marg_by_var=marg_by_var,
        cond_map=cond_map,
        test_idx=test_idx,
        train_idx=train_idx,
        x_mean=x_mean,
        x_std=x_std,
    )


def infer_one_region(
    *,
    model: Any,
    data: EvalData5Var,
    row_idx: int,
    condition: str,
    n_eval_joint_samples: int,
    device: str | None,
    posthoc_ipf: bool,
    ipf_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    cond_arr = data.cond_map[condition]
    if cond_arr is None:
        z = model.sample(n=int(n_eval_joint_samples), cond=None, device=device).numpy()
    else:
        c = np.repeat(cond_arr[row_idx : row_idx + 1], repeats=int(n_eval_joint_samples), axis=0).astype(np.float32)
        import torch  # type: ignore

        z = model.sample(n=int(n_eval_joint_samples), cond=torch.from_numpy(c), device=device).numpy()

    logp = z.astype(np.float64) * data.x_std.reshape(1, -1).astype(np.float64) + data.x_mean.reshape(1, -1).astype(np.float64)
    p_draws = _softmax_rows(logp)
    p_hat_raw = np.mean(p_draws, axis=0)
    p_hat_raw = p_hat_raw / max(float(p_hat_raw.sum()), 1e-12)

    p_hat_eval = p_hat_raw
    if bool(posthoc_ipf):
        t_margs = [data.marg_by_var[vn][row_idx] for vn in data.var_names]
        p_hat_eval = _ipf_nd(
            seed_joint=p_hat_raw,
            target_marginals=t_margs,
            shape=data.shape,
            max_iter=int(ipf_iters),
        )
    return p_hat_raw.astype(np.float64), p_hat_eval.astype(np.float64)


def write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

