#!/usr/bin/env python3
from __future__ import annotations

"""
Train distribution-level diffusion on US PUMA external-target v1 and evaluate on Michigan.

Input:
- exttarget_v1_pums_2023_puma_us_joint_wide.csv
- extcond_v1_acs5_2022_puma_us.csv

Core setup:
- Train on non-MI PUMAs, test on MI PUMAs (leave_mi_out by default)
- Conditions:
  - none
  - external
- Report raw and external-marginal-calibrated metrics
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
from tools.build_external_target_v1_michigan import AGE_LABELS, ESR_LABELS, SCHL_LABELS, SEX_LABELS, SHAPE
from tools.train_us_puma_5var_diffusion import (
    _canon_puma5,
    _canon_statefp,
    _canon_uid,
    _canon_uid_loose,
    _cosine,
    _ipf_nd,
    _marginal_from_joint,
    _nd_independence,
    _parse_hidden_dims,
    _require_torch,
    _softmax_rows,
    _stable_hash_fold,
    _summ,
    _tvd,
    _utc_now_iso,
    _write_json,
)


PREFIX_BY_VAR: dict[str, str] = {
    "AGEP_bin": "p_age_",
    "SEX": "p_sex_",
    "SCHL_allpop": "p_schl_",
    "ESR_allpop": "p_esr_",
}


def _default_var_specs() -> list[tuple[str, str, list[str]]]:
    return [
        ("AGEP_bin", "p_age_", AGE_LABELS),
        ("SEX", "p_sex_", SEX_LABELS),
        ("SCHL_allpop", "p_schl_", SCHL_LABELS),
        ("ESR_allpop", "p_esr_", ESR_LABELS),
    ]


def _load_var_specs_from_schema(*, schema_json: pathlib.Path | None) -> list[tuple[str, str, list[str]]]:
    if schema_json is None:
        return _default_var_specs()
    obj = json.loads(schema_json.read_text(encoding="utf-8"))
    order = [str(x) for x in obj.get("variable_order", [])]
    cats_map = obj.get("categories", {})
    if not order or not isinstance(cats_map, dict):
        raise SystemExit(f"Invalid schema_json: {schema_json}")
    specs: list[tuple[str, str, list[str]]] = []
    for var in order:
        pref = PREFIX_BY_VAR.get(var)
        if pref is None:
            raise SystemExit(f"Unsupported variable in schema_json: {var}")
        cats = cats_map.get(var)
        if not isinstance(cats, list) or not cats:
            raise SystemExit(f"schema_json missing categories for variable={var}")
        specs.append((var, pref, [str(x) for x in cats]))
    return specs


def _normalize_geo_from_condition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = set(out.columns.astype(str).tolist())
    if "puma_uid" in cols:
        out["puma_uid"] = out["puma_uid"].map(_canon_uid_loose)
        return out
    if "statefp" in cols and "puma" in cols:
        out["statefp"] = out["statefp"].map(_canon_statefp)
        out["puma"] = out["puma"].map(_canon_puma5)
        out["puma_uid"] = out.apply(lambda r: _canon_uid(r["statefp"], r["puma"]), axis=1)
        return out
    raise SystemExit("condition_csv missing geography columns: expected puma_uid or (statefp, puma)")


def _load_external_condition_matrix(
    *,
    condition_csv: pathlib.Path,
    ids: list[str],
    var_specs: list[tuple[str, str, list[str]]],
) -> tuple[np.ndarray, dict[str, slice], dict[str, Any]]:
    dtype_map = {"puma_uid": str, "statefp": str, "puma": str, "variable": str, "category": str}
    cond = pd.read_csv(condition_csv, dtype={k: v for k, v in dtype_map.items() if k in pd.read_csv(condition_csv, nrows=0).columns})
    cond = _normalize_geo_from_condition(cond)

    need = {"variable", "category", "target", "puma_uid"}
    miss = [c for c in need if c not in cond.columns]
    if miss:
        raise SystemExit(f"condition_csv missing columns: {miss}")

    cond["target"] = pd.to_numeric(cond["target"], errors="coerce").fillna(0.0).astype(float)
    cond = cond[cond["puma_uid"] != ""].copy()

    cond = cond.groupby(["puma_uid", "variable", "category"], as_index=False, sort=False)["target"].sum()

    block_slices: dict[str, slice] = {}
    start = 0
    for var, _, cats in var_specs:
        block_slices[var] = slice(start, start + len(cats))
        start += len(cats)
    cond_dim = start

    lookup: dict[tuple[str, str, str], float] = {}
    for r in cond.itertuples(index=False):
        lookup[(str(r.puma_uid), str(r.variable), str(r.category))] = float(r.target)

    out = np.zeros((len(ids), cond_dim), dtype=np.float32)
    missing_uids: list[str] = []
    for row_idx, uid in enumerate(ids):
        uid_seen = False
        for var, _, cats in var_specs:
            vals = np.asarray([lookup.get((uid, var, cat), 0.0) for cat in cats], dtype=np.float64)
            if float(vals.sum()) > 0:
                uid_seen = True
                vals = vals / float(vals.sum())
            out[row_idx, block_slices[var]] = vals.astype(np.float32)
        if not uid_seen:
            missing_uids.append(uid)
    if missing_uids:
        raise SystemExit(f"condition_csv missing {len(missing_uids)} puma_uid rows. Example={missing_uids[:5]}")

    meta = {
        "condition_csv": str(condition_csv),
        "cond_dim": int(cond_dim),
        "block_dims": {var: int(len(cats)) for var, _, cats in var_specs},
        "variable_order": [var for var, _, _ in var_specs],
    }
    return out, block_slices, meta


def _apply_posthoc_ipf(*, policy: str, cond_name: str) -> bool:
    p = str(policy).strip().lower()
    if p == "none":
        return False
    if p == "all":
        return True
    if p == "external":
        return str(cond_name).strip().lower() == "external"
    raise ValueError(f"Unsupported posthoc_ipf_policy: {policy}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="train_us_puma_external_v1_diffusion")
    ap.add_argument("--joint_wide_csv", required=True, help="Path to exttarget_v1_pums_*_joint_wide.csv")
    ap.add_argument("--condition_csv", required=True, help="Path to extcond_v1_acs5_*_puma_*.csv")
    ap.add_argument("--schema_json", default=None, help="Optional schema JSON for external target/condition categories.")
    ap.add_argument("--conditions", default="none,external", help='Comma-separated: "none,external"')
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
        choices=["none", "external", "all"],
        default="external",
        help="Apply post-hoc IPF using external marginals during evaluation.",
    )
    ap.add_argument(
        "--save_final_model",
        action="store_true",
        help="If set, save final checkpoint per (condition, fold) under out_dir/checkpoints.",
    )
    ap.add_argument("--out_dir", default=None, help="Default: outputs/<run_id>")
    args = ap.parse_args()

    torch = _require_torch()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    in_path = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    cond_path = pathlib.Path(args.condition_csv).expanduser().resolve()
    schema_path = pathlib.Path(args.schema_json).expanduser().resolve() if args.schema_json else None
    if not in_path.exists():
        raise SystemExit(f"joint_wide_csv not found: {in_path}")
    if not cond_path.exists():
        raise SystemExit(f"condition_csv not found: {cond_path}")
    if schema_path is not None and not schema_path.exists():
        raise SystemExit(f"schema_json not found: {schema_path}")

    var_specs = _load_var_specs_from_schema(schema_json=schema_path)

    run_id = f"_us_puma_external_v1_diffusion_{_dt.datetime.now(_dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else (_REPO_ROOT / "outputs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    required = {"statefp", "puma", "puma_uid", "total_person_weight"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"joint_wide_csv missing columns: {missing}")

    p_joint_cols = sorted([c for c in df.columns if c.startswith("p_joint_")], key=lambda x: int(x.split("_")[-1]))
    marg_cols_by_var: dict[str, list[str]] = {}
    for _, pref, cats in var_specs:
        cols = sorted([c for c in df.columns if c.startswith(pref)], key=lambda x: int(x.split("_")[-1]))
        if len(cols) != len(cats):
            raise SystemExit(f"Marginal column mismatch for prefix={pref}: got {len(cols)}, expected {len(cats)}")
        marg_cols_by_var[pref] = cols

    shape = tuple(len(cats) for _, _, cats in var_specs)
    K = int(np.prod(shape))
    if len(p_joint_cols) != K:
        raise SystemExit(f"Joint dim mismatch: got {len(p_joint_cols)} joint cols, expected {K}")

    df["statefp"] = df["statefp"].map(_canon_statefp)
    df["puma5"] = df["puma"].map(_canon_puma5)
    df["puma"] = df["puma5"].map(lambda x: str(int(x)) if x else "")
    df["puma_uid"] = df.apply(lambda r: _canon_uid(r["statefp"], r["puma5"]), axis=1)
    if int((df["puma_uid"] == "").sum()) > 0:
        raise SystemExit("Invalid puma_uid rows after canonicalization.")

    is_mi = df["statefp"] == "26"
    if int(is_mi.sum()) == 0:
        raise SystemExit("No Michigan rows found (statefp==26).")

    p_joint = df[p_joint_cols].to_numpy(dtype=np.float32)
    p_joint = np.clip(p_joint, 0.0, None)
    p_joint = p_joint / np.maximum(p_joint.sum(axis=1, keepdims=True), 1e-12)
    totals = pd.to_numeric(df["total_person_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    ids = df["puma_uid"].astype(str).tolist()

    target_marg_by_var: dict[str, np.ndarray] = {}
    for var, pref, _ in var_specs:
        arr = df[marg_cols_by_var[pref]].to_numpy(dtype=np.float32)
        arr = np.clip(arr, 0.0, None)
        arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-12)
        target_marg_by_var[var] = arr

    cond_ext, cond_block_slices, cond_meta = _load_external_condition_matrix(condition_csv=cond_path, ids=ids, var_specs=var_specs)
    ext_marg_by_var: dict[str, np.ndarray] = {var: cond_ext[:, s].copy() for var, s in cond_block_slices.items()}

    x_log_all = np.log(np.clip(p_joint, 0.0, None) + 1e-6).astype(np.float32)

    conditions = [c.strip() for c in str(args.conditions).split(",") if c.strip()]
    allowed = {"none", "external"}
    bad = [c for c in conditions if c not in allowed]
    if bad:
        raise SystemExit(f"Unsupported conditions: {bad}, allowed={sorted(allowed)}")

    cond_map: dict[str, np.ndarray | None] = {
        "none": None,
        "external": cond_ext,
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
    baselines_by_fold: dict[str, dict[str, Any]] = {"independence_external": {}, "ipf_train_seed_external": {}}
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
            fit_kwargs: dict[str, Any] = {
                "x": torch.from_numpy(x_train),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "device": args.device,
                "log_every": int(args.log_every),
            }
            if cond_train is not None:
                fit_kwargs["cond"] = torch.from_numpy(cond_train)
            model.fit(**fit_kwargs)

            saved_this_fold: list[str] = []
            if bool(args.save_final_model):
                ckpt_path = out_dir / "checkpoints" / cond_name / fold_name / "final.pt"
                model.save(ckpt_path)
                saved_this_fold.append(str(ckpt_path))
            saved_checkpoints.setdefault(cond_name, {})[fold_name] = saved_this_fold

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
            var_eval: dict[str, list[float]] = {var: [] for var, _, _ in var_specs}
            var_raw: dict[str, list[float]] = {var: [] for var, _, _ in var_specs}

            for j, idx in enumerate(test_idx):
                p_true = p_joint[idx]
                ext_margs = [ext_marg_by_var[var][idx] for var, _, _ in var_specs]
                tgt_margs = [target_marg_by_var[var][idx] for var, _, _ in var_specs]

                p_ind = _nd_independence(ext_margs)
                p_ipf = _ipf_nd(
                    seed_joint=seed_joint,
                    target_marginals=ext_margs,
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

                logp = z.astype(np.float64) * x_std.reshape(1, -1).astype(np.float64) + x_mean.reshape(1, -1).astype(np.float64)
                p_draws = _softmax_rows(logp)
                p_hat_raw = np.mean(p_draws, axis=0)
                p_hat_raw = p_hat_raw / max(float(p_hat_raw.sum()), 1e-12)

                p_hat_eval = p_hat_raw
                if _apply_posthoc_ipf(policy=str(args.posthoc_ipf_policy), cond_name=cond_name):
                    p_hat_eval = _ipf_nd(
                        seed_joint=p_hat_raw,
                        target_marginals=ext_margs,
                        shape=shape,
                        max_iter=int(args.ipf_iters),
                    )

                raw_tvd_joint_vals.append(_tvd(p_hat_raw, p_true))
                tvd_joint_vals.append(_tvd(p_hat_eval, p_true))
                raw_cos_vals.append(_cosine(p_hat_raw, p_true))
                cos_vals.append(_cosine(p_hat_eval, p_true))

                for axis, (var, _, _) in enumerate(var_specs):
                    mr = _marginal_from_joint(p_hat_raw, shape=shape, axis=axis)
                    me = _marginal_from_joint(p_hat_eval, shape=shape, axis=axis)
                    mt = np.asarray(tgt_margs[axis], dtype=float)
                    var_raw[var].append(_tvd(mr, mt))
                    var_eval[var].append(_tvd(me, mt))

            if fold_name not in baselines_by_fold["independence_external"]:
                baselines_by_fold["independence_external"][fold_name] = {"tvd_joint": _summ(ind_tvd_vals)}
                baselines_by_fold["ipf_train_seed_external"][fold_name] = {"tvd_joint": _summ(ipf_tvd_vals)}

            fold_obj: dict[str, Any] = {
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "posthoc_ipf_applied": bool(_apply_posthoc_ipf(policy=str(args.posthoc_ipf_policy), cond_name=cond_name)),
                "tvd_joint": _summ(tvd_joint_vals),
                "cosine_joint": _summ(cos_vals),
                "tvd_joint_raw": _summ(raw_tvd_joint_vals),
                "cosine_joint_raw": _summ(raw_cos_vals),
            }
            for var, _, _ in var_specs:
                fold_obj[f"tvd_{var}"] = _summ(var_eval[var])
                fold_obj[f"tvd_{var}_raw"] = _summ(var_raw[var])
            cond_fold_metrics[fold_name] = fold_obj

        fold_names = sorted(cond_fold_metrics.keys())
        out_overall: dict[str, Any] = {
            "tvd_joint": _summ([float(cond_fold_metrics[f]["tvd_joint"]["mean"]) for f in fold_names]),
            "cosine_joint": _summ([float(cond_fold_metrics[f]["cosine_joint"]["mean"]) for f in fold_names]),
            "tvd_joint_raw": _summ([float(cond_fold_metrics[f]["tvd_joint_raw"]["mean"]) for f in fold_names]),
            "cosine_joint_raw": _summ([float(cond_fold_metrics[f]["cosine_joint_raw"]["mean"]) for f in fold_names]),
        }
        for var, _, _ in var_specs:
            out_overall[f"tvd_{var}"] = _summ([float(cond_fold_metrics[f][f"tvd_{var}"]["mean"]) for f in fold_names])
            out_overall[f"tvd_{var}_raw"] = _summ([float(cond_fold_metrics[f][f"tvd_{var}_raw"]["mean"]) for f in fold_names])
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
        "condition_csv": str(cond_path),
        "schema_json": str(schema_path) if schema_path is not None else None,
        "n_rows_total": int(df.shape[0]),
        "n_mi_rows": int(is_mi.sum()),
        "n_non_mi_rows": int((~is_mi).sum()),
        "shape": {"age": shape[0], "sex": shape[1], "schl": shape[2], "esr": shape[3]},
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
        "ipf_iters": int(args.ipf_iters),
        "posthoc_ipf_policy": str(args.posthoc_ipf_policy),
        "x_representation": "logp + per-fold z-score",
        "seed": int(args.seed),
        "device": args.device,
        "save_final_model": bool(args.save_final_model),
        "saved_checkpoints": saved_checkpoints,
        "external_condition": cond_meta,
        "target_variable_order": [var for var, _, _ in var_specs],
    }

    _write_json(out_dir / "run_summary.json", run_summary)
    _write_json(out_dir / "metrics" / "internal_external_holdout.json", {"by_condition": internal_by_condition})
    _write_json(out_dir / "metrics" / "baselines_internal.json", baselines_internal)
    _write_json(out_dir / "metrics" / "ablation_summary.json", ablation_summary)
    print(f"[ok] wrote: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
