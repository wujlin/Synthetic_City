#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import random
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.model.train_external_joint_hier_diffusion_full as one_shot_base
import tools.model.train_external_joint_hier_diffusion_full as stage1_base
import tools.model.train_external_joint_hier_diffusion_full_earn as _full_earn  # noqa: F401
from tools.model.eval_external_c2f_full_earn_pipeline import (
    _coarse_marginals_from_full_ext,
    _combine_from_coarse,
    _compute_stage2_scaler,
    _load_full_joint_wide,
    _load_stage1_model,
    _run_full_ipf,
)
from tools.model.external_c2f_full_earn_schema import COARSE_SHAPE, FULL_SHAPE, FULL_VARIABLE_ORDER
from tools.figures.make_fig4_michigan_regional_validation import ONE_SHOT_CONFIGS, PIPELINE_CONFIGS
from tools.model.train_us_puma_5var_diffusion import _canon_statefp, _ipf_nd, _require_torch, _softmax_rows, _tvd
from tools.model.train_us_puma_external_v1_diffusion import _load_condition_specs_from_schema, _load_external_condition_matrix


PAIR_NAMES = [
    "AGEP_bin__SEX",
    "AGEP_bin__SCHL_allpop",
    "AGEP_bin__ESR_allpop",
    "AGEP_bin__EARN_16p_bin",
    "SEX__SCHL_allpop",
    "SEX__ESR_allpop",
    "SEX__EARN_16p_bin",
    "SCHL_allpop__ESR_allpop",
    "SCHL_allpop__EARN_16p_bin",
    "ESR_allpop__EARN_16p_bin",
]


def _utc_now_compact() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch = _require_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _select_device(requested: str | None) -> str:
    torch = _require_torch()
    if requested is None or str(requested).lower().strip() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(requested)


def _load_ext_inputs(
    *,
    condition_csv: pathlib.Path,
    condition_schema_json: pathlib.Path,
    ids: list[str],
    stage1_schema_json: pathlib.Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    stage1_var_specs = stage1_base._load_var_specs_from_schema(schema_json=stage1_schema_json)
    cond_specs = _load_condition_specs_from_schema(
        condition_schema_json=condition_schema_json,
        fallback_var_specs=stage1_var_specs,
    )
    cond_raw, block_slices, _ = _load_external_condition_matrix(
        condition_csv=condition_csv,
        ids=ids,
        var_specs=cond_specs,
    )
    ext_marg = {var: cond_raw[:, sl].copy() for var, sl in block_slices.items()}
    ext_marg = stage1_base._augment_ext_marginals_from_cross(
        cond_raw=cond_raw,
        block_slices=block_slices,
        ext_marg=ext_marg,
    )
    missing = [v for v in FULL_VARIABLE_ORDER if v not in ext_marg]
    if missing:
        raise SystemExit(f"Missing external marginals for variables: {missing}")
    return cond_raw, ext_marg


def _pair_axes(pair: str) -> tuple[int, int]:
    left, right = pair.split("__", 1)
    return FULL_VARIABLE_ORDER.index(left), FULL_VARIABLE_ORDER.index(right)


def _pair_marginal(p: np.ndarray, pair: str) -> np.ndarray:
    axes_keep = _pair_axes(pair)
    tab = np.asarray(p, dtype=np.float64).reshape(FULL_SHAPE)
    axes_sum = tuple(i for i in range(len(FULL_SHAPE)) if i not in axes_keep)
    out = tab.sum(axis=axes_sum).reshape(-1)
    return out / max(float(out.sum()), 1e-12)


def _pairwise_rows(
    *,
    df: pd.DataFrame,
    heldout_idx: np.ndarray,
    p_true: np.ndarray,
    p_pred: np.ndarray,
    model: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_pos, idx in enumerate(heldout_idx.tolist()):
        true_vec = np.asarray(p_true[local_pos], dtype=np.float64)
        pred_vec = np.asarray(p_pred[local_pos], dtype=np.float64)
        for pair in PAIR_NAMES:
            rows.append(
                {
                    "puma_uid": str(df.iloc[idx]["puma_uid"]),
                    "statefp": str(df.iloc[idx]["statefp"]),
                    "puma5": str(df.iloc[idx]["puma5"]),
                    "model": str(model),
                    "seed": int(seed),
                    "pair": str(pair),
                    "tvd_pair": float(_tvd(_pair_marginal(pred_vec, pair), _pair_marginal(true_vec, pair))),
                }
            )
    return rows


def _build_ipf_vectors(
    *,
    p_true_all: np.ndarray,
    ext_marg: dict[str, np.ndarray],
    heldout_idx: np.ndarray,
    train_idx: np.ndarray,
    ipf_iters: int,
) -> np.ndarray:
    train_seed = np.asarray(p_true_all[train_idx], dtype=np.float64).mean(axis=0)
    train_seed = train_seed / max(float(train_seed.sum()), 1e-12)
    out = np.zeros((heldout_idx.size, p_true_all.shape[1]), dtype=np.float64)
    for local_pos, idx in enumerate(heldout_idx.tolist()):
        ext_row = {var: np.asarray(ext_marg[var][idx], dtype=np.float64) for var in FULL_VARIABLE_ORDER}
        out[local_pos] = _run_full_ipf(seed_joint=train_seed, ext_row=ext_row, ipf_iters=int(ipf_iters))
    return out


def _build_co_national_vectors(
    *,
    df: pd.DataFrame,
    p_true_all: np.ndarray,
    ext_marg: dict[str, np.ndarray],
    heldout_idx: np.ndarray,
    train_idx: np.ndarray,
    ipf_iters: int,
) -> np.ndarray:
    if "total_person_weight" in df.columns:
        weights = pd.to_numeric(df["total_person_weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
        train_weights = weights[train_idx]
        if float(train_weights.sum()) <= 0.0:
            train_weights = np.ones(train_idx.size, dtype=np.float64)
    else:
        train_weights = np.ones(train_idx.size, dtype=np.float64)
    seed = (np.asarray(p_true_all[train_idx], dtype=np.float64) * train_weights.reshape(-1, 1)).sum(axis=0)
    seed = seed / max(float(seed.sum()), 1e-12)

    out = np.zeros((heldout_idx.size, p_true_all.shape[1]), dtype=np.float64)
    for local_pos, idx in enumerate(heldout_idx.tolist()):
        ext_row = {var: np.asarray(ext_marg[var][idx], dtype=np.float64) for var in FULL_VARIABLE_ORDER}
        out[local_pos] = _run_full_ipf(seed_joint=seed, ext_row=ext_row, ipf_iters=int(ipf_iters))
    return out


def _build_one_shot_vectors(
    *,
    cfg: Any,
    df: pd.DataFrame,
    p_true_all: np.ndarray,
    cond_raw: np.ndarray,
    ext_marg: dict[str, np.ndarray],
    heldout_idx: np.ndarray,
    train_idx: np.ndarray,
    device: str,
) -> np.ndarray:
    _set_all_seeds(int(cfg.seed))
    checkpoint = pathlib.Path(cfg.checkpoint).expanduser().resolve()

    if str(cfg.support_mask_mode).lower().strip() == "dataset_nonzero":
        active_cols = np.where((p_true_all > float(cfg.support_mask_eps)).any(axis=0))[0].astype(np.int64)
    else:
        active_cols = np.arange(p_true_all.shape[1], dtype=np.int64)

    p_fine = p_true_all[:, active_cols].astype(np.float32)
    p_fine = p_fine / np.maximum(p_fine.sum(axis=1, keepdims=True), 1e-12)
    x_log_all = np.log(np.clip(p_fine, 0.0, None) + 1e-6).astype(np.float32)
    x_train_log = x_log_all[train_idx]
    x_mean = x_train_log.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = x_train_log.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)

    if 0.0 <= float(cfg.logp_clip_quantile_low) < float(cfg.logp_clip_quantile_high) <= 1.0:
        logp_clip_lo = np.quantile(x_train_log, float(cfg.logp_clip_quantile_low), axis=0).astype(np.float32)
        logp_clip_hi = np.quantile(x_train_log, float(cfg.logp_clip_quantile_high), axis=0).astype(np.float32)
    else:
        logp_clip_lo = None
        logp_clip_hi = None

    torch = _require_torch()
    model, _ = _load_stage1_model(checkpoint_path=checkpoint, timesteps=int(cfg.timesteps), seed=int(cfg.seed))
    model.to(device)
    cond_eval_t = torch.from_numpy(cond_raw[heldout_idx]).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        z_eval = model.encoder(cond_eval_t)
    x_samples = model.sample_latent_conditioned(
        z_cond=z_eval,
        n_draws=int(cfg.n_eval_joint_samples),
        device=device,
    ).numpy()

    logp = x_samples.astype(np.float64) * x_std.reshape(1, 1, -1).astype(np.float64) + x_mean.reshape(1, 1, -1).astype(np.float64)
    if logp_clip_lo is not None and logp_clip_hi is not None:
        logp = np.clip(logp, logp_clip_lo.reshape(1, 1, -1), logp_clip_hi.reshape(1, 1, -1))
    p_draws = np.asarray([_softmax_rows(logp[i]) for i in range(logp.shape[0])], dtype=np.float64)
    p_hat_raw = p_draws.mean(axis=1)
    if active_cols.size != p_true_all.shape[1]:
        p_hat_full = one_shot_base._expand_active_prob_np(
            p_active=p_hat_raw,
            active_cols=active_cols,
            full_dim=int(p_true_all.shape[1]),
        )
    else:
        p_hat_full = p_hat_raw
    p_hat_full = p_hat_full / np.maximum(p_hat_full.sum(axis=1, keepdims=True), 1e-12)

    out = np.zeros((heldout_idx.size, p_true_all.shape[1]), dtype=np.float64)
    for local_pos, idx in enumerate(heldout_idx.tolist()):
        ext_row = {var: np.asarray(ext_marg[var][idx], dtype=np.float64) for var in FULL_VARIABLE_ORDER}
        out[local_pos] = _run_full_ipf(
            seed_joint=p_hat_full[local_pos],
            ext_row=ext_row,
            ipf_iters=int(cfg.ipf_iters),
        )

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _build_hierarchical_vectors(
    *,
    cfg: Any,
    p_true_all: np.ndarray,
    cond_raw: np.ndarray,
    ext_marg: dict[str, np.ndarray],
    heldout_idx: np.ndarray,
    heldout_statefp: str,
    device: str,
) -> np.ndarray:
    _set_all_seeds(int(cfg.seed))
    stage1_checkpoint = pathlib.Path(cfg.stage1_checkpoint).expanduser().resolve()
    stage2_wide_csv = pathlib.Path(cfg.stage2_wide_csv).expanduser().resolve()
    stage2_schema_json = pathlib.Path(cfg.stage2_schema_json).expanduser().resolve()
    stage2_checkpoint = pathlib.Path(cfg.stage2_checkpoint).expanduser().resolve()

    torch = _require_torch()
    stage1_model, _ = _load_stage1_model(
        checkpoint_path=stage1_checkpoint,
        timesteps=int(cfg.stage1_timesteps),
        seed=int(cfg.seed),
    )
    stage1_model.to(device)
    stage2_x_mean, stage2_x_std = _compute_stage2_scaler(
        wide_csv=stage2_wide_csv,
        schema_json=stage2_schema_json,
        heldout_statefp=str(heldout_statefp),
    )
    from tools.model.external_c2f_full_earn_stage2_model import load_stage2_model

    stage2_model, _ = load_stage2_model(checkpoint_path=stage2_checkpoint)
    cond_eval_t = torch.from_numpy(cond_raw[heldout_idx]).to(device=device, dtype=torch.float32)
    coarse_pred_raw = stage1_model.predict_coarse(cond_raw=cond_eval_t).detach().cpu().numpy().astype(np.float64)

    out = np.zeros((heldout_idx.size, p_true_all.shape[1]), dtype=np.float64)
    for local_pos, idx in enumerate(heldout_idx.tolist()):
        ext_row = {var: np.asarray(ext_marg[var][idx], dtype=np.float64) for var in FULL_VARIABLE_ORDER}
        coarse_targets = _coarse_marginals_from_full_ext(ext_row)
        p_coarse_raw = coarse_pred_raw[local_pos]
        p_coarse_raw = p_coarse_raw / max(float(p_coarse_raw.sum()), 1e-12)
        p_coarse_proj = _ipf_nd(
            seed_joint=p_coarse_raw.reshape(COARSE_SHAPE),
            target_marginals=coarse_targets,
            shape=COARSE_SHAPE,
            max_iter=int(cfg.ipf_iters),
        )
        p_coarse_proj = p_coarse_proj / max(float(p_coarse_proj.sum()), 1e-12)
        p_full_from_proj, _ = _combine_from_coarse(
            stage2_model=stage2_model,
            coarse_prob=p_coarse_proj,
            x_mean=stage2_x_mean,
            x_std=stage2_x_std,
            n_draws=int(cfg.stage2_n_eval_joint_samples),
            device=device,
        )
        out[local_pos] = _run_full_ipf(
            seed_joint=p_full_from_proj,
            ext_row=ext_row,
            ipf_iters=int(cfg.ipf_iters),
        )

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _summarize(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        long_df.groupby(["model", "seed", "pair"], as_index=False)["tvd_pair"]
        .agg(tvd_pair_mean="mean", tvd_pair_std="std", n_pumas="count")
        .sort_values(["pair", "model", "seed"])
        .reset_index(drop=True)
    )
    hier = (
        long_df.loc[long_df["model"] == "hierarchical"]
        .groupby("pair", as_index=False)["tvd_pair"]
        .mean()
        .rename(columns={"tvd_pair": "hierarchical_seed_mean_tvd_pair"})
    )
    gap = summary.merge(hier, on="pair", how="left")
    gap["gap_vs_hierarchical_seed_mean"] = gap["tvd_pair_mean"] - gap["hierarchical_seed_mean_tvd_pair"]
    gap = gap.sort_values(["pair", "model", "seed"]).reset_index(drop=True)
    model_mean = (
        summary.groupby(["model", "pair"], as_index=False)
        .agg(
            tvd_pair_mean_across_seeds=("tvd_pair_mean", "mean"),
            tvd_pair_std_across_seeds=("tvd_pair_mean", "std"),
            n_model_seeds=("seed", "nunique"),
            n_pumas=("n_pumas", "min"),
        )
        .merge(hier, on="pair", how="left")
    )
    model_mean["gap_vs_hierarchical_seed_mean"] = (
        model_mean["tvd_pair_mean_across_seeds"] - model_mean["hierarchical_seed_mean_tvd_pair"]
    )
    model_mean = model_mean.sort_values(["pair", "model"]).reset_index(drop=True)
    return summary, gap, model_mean


def main() -> None:
    ap = argparse.ArgumentParser(prog="export_michigan_pairwise_tvd_long")
    ap.add_argument("--heldout_statefp", default="26")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ipf_iters", type=int, default=200)
    ap.add_argument("--co_ipf_iters", type=int, default=500)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--write_vectors", action="store_true")
    args = ap.parse_args()

    heldout_statefp = _canon_statefp(args.heldout_statefp)
    base_cfg = PIPELINE_CONFIGS[0]
    joint_wide_csv = pathlib.Path(base_cfg.stage1_joint_wide_csv).expanduser().resolve()
    schema_json = pathlib.Path(base_cfg.stage1_schema_json).expanduser().resolve()
    condition_csv = pathlib.Path(base_cfg.stage1_condition_csv).expanduser().resolve()
    condition_schema_json = pathlib.Path(base_cfg.stage1_condition_schema_json).expanduser().resolve()

    run_id = f"_paper1_michigan_pairwise_model_comparison_{_utc_now_compact()}"
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else REPO_ROOT / "outputs" / run_id
    metrics_dir = out_dir / "metrics"
    arrays_dir = out_dir / "arrays"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if args.write_vectors:
        arrays_dir.mkdir(parents=True, exist_ok=True)

    df, p_true_all, ids = _load_full_joint_wide(joint_wide_csv=joint_wide_csv, schema_json=schema_json)
    cond_raw, ext_marg = _load_ext_inputs(
        condition_csv=condition_csv,
        condition_schema_json=condition_schema_json,
        ids=ids,
        stage1_schema_json=schema_json,
    )
    is_heldout = (df["statefp"] == heldout_statefp).to_numpy(dtype=bool)
    heldout_idx = np.where(is_heldout)[0]
    train_idx = np.where(~is_heldout)[0]
    if heldout_idx.size == 0:
        raise SystemExit(f"No held-out PUMAs found for statefp={heldout_statefp}.")
    if train_idx.size == 0:
        raise SystemExit(f"No training PUMAs found after holding out statefp={heldout_statefp}.")

    p_true_heldout = np.asarray(p_true_all[heldout_idx], dtype=np.float64)
    device = _select_device(args.device)

    rows: list[dict[str, Any]] = []
    vectors: dict[str, np.ndarray] = {"p_true": p_true_heldout.astype(np.float32)}

    print(f"[load] heldout_statefp={heldout_statefp} n_pumas={heldout_idx.size} device={device}")
    print("[model] ipf")
    p_ipf = _build_ipf_vectors(
        p_true_all=p_true_all,
        ext_marg=ext_marg,
        heldout_idx=heldout_idx,
        train_idx=train_idx,
        ipf_iters=int(args.ipf_iters),
    )
    rows.extend(_pairwise_rows(df=df, heldout_idx=heldout_idx, p_true=p_true_heldout, p_pred=p_ipf, model="ipf", seed=0))
    vectors["ipf_seed0"] = p_ipf.astype(np.float32)

    print("[model] co_national")
    p_co = _build_co_national_vectors(
        df=df,
        p_true_all=p_true_all,
        ext_marg=ext_marg,
        heldout_idx=heldout_idx,
        train_idx=train_idx,
        ipf_iters=int(args.co_ipf_iters),
    )
    rows.extend(_pairwise_rows(df=df, heldout_idx=heldout_idx, p_true=p_true_heldout, p_pred=p_co, model="co_national", seed=0))
    vectors["co_national_seed0"] = p_co.astype(np.float32)

    for cfg in PIPELINE_CONFIGS:
        print(f"[model] hierarchical {cfg.label}")
        p_hier = _build_hierarchical_vectors(
            cfg=cfg,
            p_true_all=p_true_all,
            cond_raw=cond_raw,
            ext_marg=ext_marg,
            heldout_idx=heldout_idx,
            heldout_statefp=heldout_statefp,
            device=device,
        )
        rows.extend(
            _pairwise_rows(
                df=df,
                heldout_idx=heldout_idx,
                p_true=p_true_heldout,
                p_pred=p_hier,
                model="hierarchical",
                seed=int(cfg.seed),
            )
        )
        vectors[f"hierarchical_seed{int(cfg.seed)}"] = p_hier.astype(np.float32)

    for cfg in ONE_SHOT_CONFIGS:
        print(f"[model] one_shot_ddpm {cfg.label}")
        p_one = _build_one_shot_vectors(
            cfg=cfg,
            df=df,
            p_true_all=p_true_all,
            cond_raw=cond_raw,
            ext_marg=ext_marg,
            heldout_idx=heldout_idx,
            train_idx=train_idx,
            device=device,
        )
        rows.extend(
            _pairwise_rows(
                df=df,
                heldout_idx=heldout_idx,
                p_true=p_true_heldout,
                p_pred=p_one,
                model="one_shot_ddpm",
                seed=int(cfg.seed),
            )
        )
        vectors[f"one_shot_ddpm_seed{int(cfg.seed)}"] = p_one.astype(np.float32)

    long_df = pd.DataFrame(rows)
    long_path = metrics_dir / "michigan_pairwise_tvd_long.csv"
    long_df.to_csv(long_path, index=False)

    summary_df, gap_df, model_mean_gap_df = _summarize(long_df)
    summary_path = metrics_dir / "michigan_pairwise_tvd_summary_by_model.csv"
    gap_path = metrics_dir / "michigan_pairwise_tvd_gap_vs_hierarchical.csv"
    model_mean_gap_path = metrics_dir / "michigan_pairwise_tvd_gap_vs_hierarchical_model_mean.csv"
    summary_df.to_csv(summary_path, index=False)
    gap_df.to_csv(gap_path, index=False)
    model_mean_gap_df.to_csv(model_mean_gap_path, index=False)

    puma_meta_path = metrics_dir / "michigan_puma_metadata.csv"
    df.iloc[heldout_idx][["puma_uid", "statefp", "puma5"]].to_csv(puma_meta_path, index=False)
    vector_path = None
    if args.write_vectors:
        vector_path = arrays_dir / "michigan_joint_vectors.npz"
        np.savez_compressed(vector_path, **vectors)

    run_summary = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "experiment": "paper1_michigan_pairwise_tvd_model_comparison",
        "heldout_statefp": heldout_statefp,
        "n_heldout_pumas": int(heldout_idx.size),
        "n_train_pumas": int(train_idx.size),
        "schema": {"variable_order": list(FULL_VARIABLE_ORDER), "shape": list(FULL_SHAPE), "K": int(np.prod(FULL_SHAPE))},
        "pairs": PAIR_NAMES,
        "models": {
            "hierarchical": [int(c.seed) for c in PIPELINE_CONFIGS],
            "one_shot_ddpm": [int(c.seed) for c in ONE_SHOT_CONFIGS],
            "ipf": [0],
            "co_national": [0],
        },
        "inputs": {
            "joint_wide_csv": str(joint_wide_csv),
            "schema_json": str(schema_json),
            "condition_csv": str(condition_csv),
            "condition_schema_json": str(condition_schema_json),
        },
        "device": device,
        "ipf_iters": int(args.ipf_iters),
        "co_ipf_iters": int(args.co_ipf_iters),
        "outputs": {
            "long_table": str(long_path),
            "summary_by_model": str(summary_path),
            "gap_vs_hierarchical": str(gap_path),
            "gap_vs_hierarchical_model_mean": str(model_mean_gap_path),
            "puma_metadata": str(puma_meta_path),
            "joint_vectors_npz": str(vector_path) if vector_path is not None else None,
        },
        "row_count": int(long_df.shape[0]),
    }
    _write_json(out_dir / "run_summary.json", run_summary)
    _write_json(metrics_dir / "run_summary.json", run_summary)

    print(f"[done] wrote {long_path}")
    print(f"[done] rows={long_df.shape[0]}")


if __name__ == "__main__":
    main()
