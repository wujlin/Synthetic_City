#!/usr/bin/env python3
from __future__ import annotations

"""
Train 5-variable distribution-level diffusion on US PUMA joints and evaluate on Michigan.

Input:
- puma_5var_joint_wide.csv from tools/build_us_puma_5var_joint.py

Core setup:
- Train on non-MI PUMAs, test on MI PUMAs (leave_mi_out by default)
- Condition ablation:
  - none
  - marginal (all 5 marginals concatenated)
  - pairwise (all pairwise 2D marginals derived from joint)
  - marginal_pairwise (marginal + pairwise)
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


def _split_tokens(spec: str) -> list[str]:
    return [x.strip() for x in str(spec).split(",") if x.strip()]


def _parse_int_tokens(spec: str) -> list[int]:
    out: list[int] = []
    for tok in _split_tokens(spec):
        try:
            v = int(tok)
        except Exception as e:
            raise ValueError(f"Invalid integer token in list: {tok}") from e
        out.append(v)
    return out


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


def _canon_uid_loose(v: object) -> str:
    d = _digits_only(v)
    if not d:
        return ""
    if len(d) > 7:
        d = d[-7:]
    return d.zfill(7)


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
    if "statefp" in sdf.columns and "puma" in sdf.columns:
        sdf["puma_uid"] = sdf.apply(lambda r: _canon_uid(r["statefp"], r["puma"]), axis=1)
    else:
        sdf["puma_uid"] = sdf["puma_uid"].map(_canon_uid_loose)
    sdf = sdf[sdf["puma_uid"] != ""].copy()
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
            m = tab.sum(axis=reduce_axes)  # (n, shape[i], shape[j])
            m = m.reshape(n, -1)
            m = m / np.maximum(m.sum(axis=1, keepdims=True), 1e-12)
            m = m.astype(np.float32)
            blocks.append(m)
            pair_dims[f"{var_names[i]}__{var_names[j]}"] = int(m.shape[1])
    if not blocks:
        return np.zeros((n, 0), dtype=np.float32), {}
    out = np.concatenate(blocks, axis=1).astype(np.float32)
    return out, pair_dims


def _apply_posthoc_ipf(*, policy: str, cond_name: str) -> bool:
    p = str(policy).strip().lower()
    if p == "none":
        return False
    if p == "all":
        return True
    if p == "marginal":
        return str(cond_name).strip().lower() in {
            "marginal",
            "marginal_pairwise",
            "marginal_spatial",
            "marginal_pairwise_spatial",
        }
    raise ValueError(f"Unsupported posthoc_ipf_policy: {policy}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="train_us_puma_5var_diffusion")
    ap.add_argument("--joint_wide_csv", required=True, help="Path to puma_5var_joint_wide.csv")
    ap.add_argument(
        "--conditions",
        default="none,marginal",
        help=(
            'Comma-separated: '
            '"none,marginal,pairwise,marginal_pairwise,spatial,marginal_spatial,'
            'pairwise_spatial,marginal_pairwise_spatial"'
        ),
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
    ap.add_argument("--n_eval_joint_samples", type=int, default=128)
    ap.add_argument("--ipf_iters", type=int, default=200)
    ap.add_argument("--spatial_features_csv", default=None, help="Optional PUMA-level spatial feature CSV.")
    ap.add_argument(
        "--spatial_feature_sets",
        default="",
        help='Comma-separated groups from {centroid_raw,centroid_pe,geo_shape,neigh_1hop,neigh_2hop,neigh_stats}.',
    )
    ap.add_argument("--spatial_feature_cols", default="", help="Optional explicit spatial feature columns (comma-separated).")
    ap.add_argument(
        "--save_epochs",
        default="",
        help="Optional comma-separated epoch list for checkpoint saving (e.g., 100,200,500,1000).",
    )
    ap.add_argument(
        "--save_final_model",
        action="store_true",
        help="If set, save final checkpoint per (condition, fold) under out_dir/checkpoints.",
    )
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

    df["statefp"] = df["statefp"].map(_canon_statefp)
    df["puma5"] = df["puma"].map(_canon_puma5)
    df["puma"] = df["puma5"].map(lambda x: str(int(x)) if x else "")
    df["puma_uid"] = df.apply(lambda r: _canon_uid(r["statefp"], r["puma5"]), axis=1)
    bad_uid = int((df["puma_uid"] == "").sum())
    if bad_uid > 0:
        raise SystemExit(f"Invalid puma_uid rows after canonicalization: {bad_uid}")
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

    # Work in log-prob space, then z-score per training fold to match DDPM's N(0,1) operating regime.
    x_log_all = np.log(np.clip(p_joint, 0.0, None) + 1e-6).astype(np.float32)
    cond_marg = np.concatenate([marg_by_var[vn] for vn, _ in var_specs], axis=1).astype(np.float32)
    cond_pairwise, pair_dims = _pairwise_from_joint(
        p_joint=p_joint,
        shape=shape,
        var_names=[vn for vn, _ in var_specs],
    )
    spatial_sets = _split_tokens(args.spatial_feature_sets)
    spatial_explicit = _split_tokens(args.spatial_feature_cols)
    conditions = [c.strip() for c in str(args.conditions).split(",") if c.strip()]
    allowed = {
        "none",
        "marginal",
        "pairwise",
        "marginal_pairwise",
        "spatial",
        "marginal_spatial",
        "pairwise_spatial",
        "marginal_pairwise_spatial",
    }
    bad = [c for c in conditions if c not in allowed]
    if bad:
        raise SystemExit(f"Unsupported conditions: {bad}, allowed={sorted(allowed)}")
    needs_spatial = any("spatial" in c for c in conditions)
    spatial_arr: np.ndarray | None = None
    spatial_cols: list[str] = []
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
        "pairwise": cond_pairwise,
        "marginal_pairwise": np.concatenate([cond_marg, cond_pairwise], axis=1).astype(np.float32),
        "spatial": spatial_arr,
        "marginal_spatial": (np.concatenate([cond_marg, spatial_arr], axis=1).astype(np.float32) if spatial_arr is not None else None),
        "pairwise_spatial": (np.concatenate([cond_pairwise, spatial_arr], axis=1).astype(np.float32) if spatial_arr is not None else None),
        "marginal_pairwise_spatial": (
            np.concatenate([cond_marg, cond_pairwise, spatial_arr], axis=1).astype(np.float32) if spatial_arr is not None else None
        ),
    }
    spatial_mask_map: dict[str, np.ndarray | None] = {
        "none": None,
        "marginal": np.zeros((cond_marg.shape[1],), dtype=bool),
        "pairwise": np.zeros((cond_pairwise.shape[1],), dtype=bool),
        "marginal_pairwise": np.zeros((cond_marg.shape[1] + cond_pairwise.shape[1],), dtype=bool),
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
        "pairwise_spatial": (
            np.concatenate(
                [
                    np.zeros((cond_pairwise.shape[1],), dtype=bool),
                    np.ones((spatial_arr.shape[1],), dtype=bool),
                ]
            )
            if spatial_arr is not None
            else None
        ),
        "marginal_pairwise_spatial": (
            np.concatenate(
                [
                    np.zeros((cond_marg.shape[1] + cond_pairwise.shape[1],), dtype=bool),
                    np.ones((spatial_arr.shape[1],), dtype=bool),
                ]
            )
            if spatial_arr is not None
            else None
        ),
    }

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
    internal_by_condition: dict[str, Any] = {}
    baselines_by_fold: dict[str, dict[str, Any]] = {"independence": {}, "ipf_train_seed": {}}
    save_epochs = sorted(set([e for e in _parse_int_tokens(args.save_epochs) if e > 0]))
    save_epoch_set = set(save_epochs)
    saved_checkpoints: dict[str, dict[str, list[str]]] = {}

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
            ckpt_dir = out_dir / "checkpoints" / cond_name / fold_name
            saved_this_fold: list[str] = []

            def _on_epoch(epoch: int, info: dict[str, Any]) -> None:
                if epoch in save_epoch_set:
                    ckpt_path = ckpt_dir / f"epoch_{epoch:05d}.pt"
                    model.save(ckpt_path)
                    saved_this_fold.append(str(ckpt_path))
                    print(
                        f"[ckpt] condition={cond_name} fold={fold_name} epoch={epoch} "
                        f"loss={float(info.get('loss', float('nan'))):.6f} path={ckpt_path}",
                        file=sys.stderr,
                    )

            fit_kwargs["epoch_callback"] = _on_epoch
            model.fit(**fit_kwargs)

            if bool(args.save_final_model):
                ckpt_path = ckpt_dir / "final.pt"
                model.save(ckpt_path)
                saved_this_fold.append(str(ckpt_path))
                print(f"[ckpt] condition={cond_name} fold={fold_name} final path={ckpt_path}", file=sys.stderr)

            saved_checkpoints.setdefault(cond_name, {})[fold_name] = saved_this_fold

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
                # Inverse z-score back to log-prob space before simplex projection.
                logp = z.astype(np.float64) * x_std.reshape(1, -1).astype(np.float64) + x_mean.reshape(1, -1).astype(np.float64)
                p_draws = _softmax_rows(logp)
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
        "condition_dims": {c: 0 if cond_map[c] is None else int(cond_map[c].shape[1]) for c in conditions},
        "pairwise_dims": pair_dims,
        "timesteps": int(args.timesteps),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_dims": list(hidden_dims),
        "condition_injection": str(args.condition_injection),
        "film_hidden_dim": int(args.film_hidden_dim),
        "n_eval_joint_samples": int(args.n_eval_joint_samples),
        "ipf_iters": int(args.ipf_iters),
        "posthoc_ipf_policy": str(args.posthoc_ipf_policy),
        "x_representation": "logp + per-fold z-score",
        "seed": int(args.seed),
        "device": args.device,
        "spatial_features_csv": str(pathlib.Path(args.spatial_features_csv).expanduser().resolve()) if args.spatial_features_csv else None,
        "spatial_feature_sets": spatial_sets,
        "spatial_feature_cols": spatial_cols if spatial_cols else spatial_explicit,
        "save_epochs": save_epochs,
        "save_final_model": bool(args.save_final_model),
        "saved_checkpoints": saved_checkpoints,
    }

    _write_json(out_dir / "run_summary.json", run_summary)
    _write_json(out_dir / "metrics" / "internal_acs_holdout.json", {"by_condition": internal_by_condition})
    _write_json(out_dir / "metrics" / "baselines_internal.json", baselines_internal)
    _write_json(out_dir / "metrics" / "ablation_summary.json", ablation_summary)
    print(f"[ok] wrote: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
