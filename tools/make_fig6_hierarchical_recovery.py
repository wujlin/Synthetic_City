#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_external_c2f_full_earn_pipeline import (
    _coarse_marginals_from_full_ext,
    _combine_from_coarse,
    _compute_stage2_scaler,
    _load_full_joint_wide,
    _load_stage1_model,
    _run_full_ipf,
)
from tools.external_c2f_full_earn_schema import COARSE_SHAPE, FULL_SHAPE, coarse_from_full_flat
from tools.make_fig4_michigan_regional_validation import (
    FULL_VARIABLE_ORDER,
    ONE_SHOT_CONFIGS,
    PIPELINE_CONFIGS,
    _compute_one_shot_seed_metrics,
    _load_ext_inputs,
    _set_all_seeds,
)
from tools.train_us_puma_5var_diffusion import _ipf_nd, _require_torch, _tvd


def _compute_pipeline_oracle_seed_metrics(
    cfg: Any,
    *,
    representative_puma_uid: str,
    example_seed_label: str,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    _set_all_seeds(cfg.seed)
    stage1_joint_wide_csv = pathlib.Path(cfg.stage1_joint_wide_csv).expanduser().resolve()
    stage1_schema_json = pathlib.Path(cfg.stage1_schema_json).expanduser().resolve()
    stage1_condition_csv = pathlib.Path(cfg.stage1_condition_csv).expanduser().resolve()
    stage1_condition_schema_json = pathlib.Path(cfg.stage1_condition_schema_json).expanduser().resolve()
    stage1_checkpoint = pathlib.Path(cfg.stage1_checkpoint).expanduser().resolve()
    stage2_wide_csv = pathlib.Path(cfg.stage2_wide_csv).expanduser().resolve()
    stage2_schema_json = pathlib.Path(cfg.stage2_schema_json).expanduser().resolve()
    stage2_checkpoint = pathlib.Path(cfg.stage2_checkpoint).expanduser().resolve()

    df, p_true_all, ids = _load_full_joint_wide(joint_wide_csv=stage1_joint_wide_csv, schema_json=stage1_schema_json)
    cond_raw, ext_marg = _load_ext_inputs(
        condition_csv=stage1_condition_csv,
        condition_schema_json=stage1_condition_schema_json,
        ids=ids,
        stage1_schema_json=stage1_schema_json,
    )
    is_mi = (df["statefp"] == "26").to_numpy(dtype=bool)
    mi_idx = np.where(is_mi)[0]

    torch = _require_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stage1_model, _ = _load_stage1_model(
        checkpoint_path=stage1_checkpoint,
        timesteps=int(cfg.stage1_timesteps),
        seed=int(cfg.seed),
    )
    stage1_model.to(device)

    stage2_x_mean, stage2_x_std = _compute_stage2_scaler(
        wide_csv=stage2_wide_csv,
        schema_json=stage2_schema_json,
    )
    from tools.external_c2f_full_earn_stage2_model import load_stage2_model

    stage2_model, _ = load_stage2_model(checkpoint_path=stage2_checkpoint)

    cond_mi_t = torch.from_numpy(cond_raw[mi_idx]).to(device=device, dtype=torch.float32)
    coarse_pred_raw = stage1_model.predict_coarse(cond_raw=cond_mi_t).detach().cpu().numpy().astype(np.float64)

    rows: list[dict[str, Any]] = []
    representative: dict[str, Any] | None = None
    for local_pos, idx in enumerate(mi_idx.tolist()):
        p_true = np.asarray(p_true_all[idx], dtype=np.float64)
        puma_uid = str(df.iloc[idx]["puma_uid"])
        ext_row = {var: np.asarray(ext_marg[var][idx], dtype=np.float64) for var in FULL_VARIABLE_ORDER}
        coarse_targets = _coarse_marginals_from_full_ext(ext_row)

        p_coarse_true = coarse_from_full_flat(p_true)
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
        p_full_oracle, _ = _combine_from_coarse(
            stage2_model=stage2_model,
            coarse_prob=p_coarse_true,
            x_mean=stage2_x_mean,
            x_std=stage2_x_std,
            n_draws=int(cfg.stage2_n_eval_joint_samples),
            device=device,
        )

        p_full_from_proj_ipf = _run_full_ipf(
            seed_joint=p_full_from_proj,
            ext_row=ext_row,
            ipf_iters=int(cfg.ipf_iters),
        )
        p_full_oracle_ipf = _run_full_ipf(
            seed_joint=p_full_oracle,
            ext_row=ext_row,
            ipf_iters=int(cfg.ipf_iters),
        )

        rows.append(
            {
                "puma_uid": puma_uid,
                "statefp": str(df.iloc[idx]["statefp"]),
                "puma5": str(df.iloc[idx]["puma5"]),
                f"stage1_coarse_tvd_{cfg.label}": float(_tvd(p_coarse_proj, p_coarse_true)),
                f"pipeline_tvd_{cfg.label}": float(_tvd(p_full_from_proj_ipf, p_true)),
                f"oracle_tvd_{cfg.label}": float(_tvd(p_full_oracle_ipf, p_true)),
            }
        )

        if cfg.label == example_seed_label and puma_uid == representative_puma_uid:
            representative = {
                "puma_uid": puma_uid,
                "statefp": str(df.iloc[idx]["statefp"]),
                "puma5": str(df.iloc[idx]["puma5"]),
                "coarse_true": p_coarse_true.tolist(),
                "coarse_stage1": np.asarray(p_coarse_proj, dtype=float).reshape(-1).tolist(),
                "full_true": p_true.tolist(),
                "full_pipeline": np.asarray(p_full_from_proj_ipf, dtype=float).reshape(-1).tolist(),
            }

    return pd.DataFrame(rows), representative


def _build_metric_table(
    *,
    representative_puma_uid: str,
    example_seed_label: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    pipeline_df: pd.DataFrame | None = None
    representative: dict[str, Any] | None = None

    for cfg in PIPELINE_CONFIGS:
        pipeline_seed_df, rep = _compute_pipeline_oracle_seed_metrics(
            cfg,
            representative_puma_uid=representative_puma_uid,
            example_seed_label=example_seed_label,
        )
        if pipeline_df is None:
            pipeline_df = pipeline_seed_df
        else:
            pipeline_df = pipeline_df.merge(
                pipeline_seed_df,
                on=["puma_uid", "statefp", "puma5"],
                how="left",
            )
        if rep is not None:
            representative = rep

    if pipeline_df is None:
        raise SystemExit("No pipeline metrics were computed.")

    one_shot_df = pipeline_df[["puma_uid", "statefp", "puma5"]].copy()
    for cfg in ONE_SHOT_CONFIGS:
        one_seed_df = _compute_one_shot_seed_metrics(cfg)
        one_shot_df = one_shot_df.merge(
            one_seed_df[["puma_uid", "statefp", "puma5", f"one_shot_tvd_{cfg.label}"]],
            on=["puma_uid", "statefp", "puma5"],
            how="left",
        )

    df = pipeline_df.merge(one_shot_df, on=["puma_uid", "statefp", "puma5"], how="left")

    stage1_cols = [f"stage1_coarse_tvd_{cfg.label}" for cfg in PIPELINE_CONFIGS]
    pipeline_cols = [f"pipeline_tvd_{cfg.label}" for cfg in PIPELINE_CONFIGS]
    oracle_cols = [f"oracle_tvd_{cfg.label}" for cfg in PIPELINE_CONFIGS]
    one_shot_cols = [f"one_shot_tvd_{cfg.label}" for cfg in ONE_SHOT_CONFIGS]

    df["stage1_coarse_tvd_mean"] = df[stage1_cols].mean(axis=1)
    df["pipeline_tvd_mean"] = df[pipeline_cols].mean(axis=1)
    df["oracle_tvd_mean"] = df[oracle_cols].mean(axis=1)
    df["one_shot_tvd_mean"] = df[one_shot_cols].mean(axis=1)
    df["pipeline_minus_oracle"] = df["pipeline_tvd_mean"] - df["oracle_tvd_mean"]

    stagewise_summary = {
        "stage1_coarse": {
            "mean": float(df[stage1_cols].mean(axis=0).mean()),
            "std": float(df[stage1_cols].mean(axis=0).std(ddof=0)),
        },
        "oracle_stage2": {
            "mean": float(df[oracle_cols].mean(axis=0).mean()),
            "std": float(df[oracle_cols].mean(axis=0).std(ddof=0)),
        },
        "pipeline": {
            "mean": float(df[pipeline_cols].mean(axis=0).mean()),
            "std": float(df[pipeline_cols].mean(axis=0).std(ddof=0)),
        },
        "one_shot": {
            "mean": float(df[one_shot_cols].mean(axis=0).mean()),
            "std": float(df[one_shot_cols].mean(axis=0).std(ddof=0)),
        },
        "pipeline_above_oracle_n": int((df["pipeline_tvd_mean"] > df["oracle_tvd_mean"]).sum()),
        "n_pumas": int(df.shape[0]),
    }
    if representative is None:
        raise SystemExit(f"Representative PUMA {representative_puma_uid} not found in Michigan evaluation.")
    return df.sort_values("puma_uid").reset_index(drop=True), stagewise_summary, representative


def _reshape_coarse_heatmap(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(COARSE_SHAPE).reshape(COARSE_SHAPE[0] * COARSE_SHAPE[1], -1)


def _reshape_full_heatmap(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(FULL_SHAPE).reshape(FULL_SHAPE[0] * FULL_SHAPE[1], -1)


def _draw_group_lines(ax: Any, *, row_breaks: list[int], col_breaks: list[int], color: str = "#ffffff") -> None:
    for r in row_breaks:
        ax.axhline(r - 0.5, color=color, linewidth=0.55, alpha=0.8)
    for c in col_breaks:
        ax.axvline(c - 0.5, color=color, linewidth=0.55, alpha=0.8)


def _plot_stagewise(ax: Any, *, summary: dict[str, Any]) -> None:
    labels = ["Oracle\nStage 2", "Pipeline", "One-shot"]
    keys = ["oracle_stage2", "pipeline", "one_shot"]
    colors = ["#4d908e", "#5b88b2", "#d99b5d"]
    x = np.arange(len(keys), dtype=float)
    means = [summary[k]["mean"] for k in keys]
    ax.bar(
        x,
        means,
        width=0.58,
        color=colors,
        edgecolor="none",
        alpha=0.9,
        zorder=1,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("TVD")
    ax.set_xlim(-0.45, len(keys) - 0.55)
    ax.set_ylim(min(means) - 0.006, max(means) + 0.010)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))


def _plot_pipeline_vs_oracle(ax: Any, *, df: pd.DataFrame) -> None:
    x = df["oracle_tvd_mean"].to_numpy(dtype=float)
    y = df["pipeline_tvd_mean"].to_numpy(dtype=float)
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    ax.scatter(x, y, s=28, color="#5b88b2", alpha=0.8, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#8e8e8e", linewidth=1.0)
    ax.set_xlabel("Oracle TVD")
    ax.set_ylabel("Pipeline TVD")


def _plot_pair_heatmaps(
    fig: Any,
    cell: Any,
    *,
    left: np.ndarray,
    right: np.ndarray,
    left_title: str,
    right_title: str,
    ylabel: str,
    xlabel: str,
    row_breaks: list[int],
    col_breaks: list[int],
    cmap: str = "YlOrBr",
    eps: float = 1e-7,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[Any, Any, Any]:
    sub = cell.subgridspec(1, 2, wspace=0.05)
    ax_left = fig.add_subplot(sub[0, 0])
    ax_right = fig.add_subplot(sub[0, 1])

    left_log = np.log10(np.asarray(left, dtype=np.float64) + float(eps))
    right_log = np.log10(np.asarray(right, dtype=np.float64) + float(eps))
    if vmin is None:
        vmin = float(min(left_log.min(), right_log.min()))
    if vmax is None:
        vmax = float(max(left_log.max(), right_log.max()))

    im_left = ax_left.imshow(left_log, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_right.imshow(right_log, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    for ax, ttl in [(ax_left, left_title), (ax_right, right_title)]:
        ax.set_title(ttl, fontsize=10.5, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        _draw_group_lines(ax, row_breaks=row_breaks, col_breaks=col_breaks)

    ax_left.set_ylabel(ylabel)

    return ax_left, ax_right, im_left


def _plot_panels(
    *,
    df: pd.DataFrame,
    stagewise_summary: dict[str, Any],
    representative: dict[str, Any],
    out_png: pathlib.Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.titlesize": 10.5,
        }
    )

    fig = plt.figure(figsize=(13.2, 10.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.18], hspace=0.34, wspace=0.24)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_stagewise(ax_a, summary=stagewise_summary)
    _plot_pipeline_vs_oracle(ax_b, df=df)

    coarse_true = _reshape_coarse_heatmap(np.asarray(representative["coarse_true"], dtype=np.float64))
    coarse_stage1 = _reshape_coarse_heatmap(np.asarray(representative["coarse_stage1"], dtype=np.float64))
    full_true = _reshape_full_heatmap(np.asarray(representative["full_true"], dtype=np.float64))
    full_pipeline = _reshape_full_heatmap(np.asarray(representative["full_pipeline"], dtype=np.float64))

    eps_shared = 1e-8
    all_logs = [
        np.log10(np.asarray(coarse_true, dtype=np.float64) + eps_shared),
        np.log10(np.asarray(coarse_stage1, dtype=np.float64) + eps_shared),
        np.log10(np.asarray(full_true, dtype=np.float64) + eps_shared),
        np.log10(np.asarray(full_pipeline, dtype=np.float64) + eps_shared),
    ]
    shared_vmin = float(min(arr.min() for arr in all_logs))
    shared_vmax = float(max(arr.max() for arr in all_logs))

    ax_c_left, ax_c_right, im_c = _plot_pair_heatmaps(
        fig,
        gs[1, 0],
        left=coarse_true,
        right=coarse_stage1,
        left_title="True coarse joint",
        right_title="Stage 1 coarse joint",
        ylabel="Coarse age × sex",
        xlabel="Coarse education × employment\n× income groups",
        row_breaks=[2, 4, 6],
        col_breaks=[12, 24],
        eps=eps_shared,
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    ax_d_left, ax_d_right, _ = _plot_pair_heatmaps(
        fig,
        gs[1, 1],
        left=full_true,
        right=full_pipeline,
        left_title="True full joint",
        right_title="Pipeline full joint",
        ylabel="Age × sex",
        xlabel="Education × Employment\n× Income combinations",
        row_breaks=[2, 4, 6, 8, 10, 12, 14, 16, 18],
        col_breaks=[30, 60, 90, 120],
        eps=eps_shared,
        vmin=shared_vmin,
        vmax=shared_vmax,
    )

    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    pos_c = ax_c_left.get_position()
    pos_d = ax_d_left.get_position()
    x_left = min(pos_a.x0, pos_c.x0) - 0.018
    x_right = min(pos_b.x0, pos_d.x0) - 0.018
    fig.text(x_left, pos_a.y1 + 0.008, "a", fontsize=17, fontweight="bold", ha="left", va="bottom")
    fig.text(x_right, pos_b.y1 + 0.008, "b", fontsize=17, fontweight="bold", ha="left", va="bottom")
    fig.text(x_left, pos_c.y1 + 0.008, "c", fontsize=17, fontweight="bold", ha="left", va="bottom")
    fig.text(x_right, pos_d.y1 + 0.008, "d", fontsize=17, fontweight="bold", ha="left", va="bottom")

    cbar = fig.colorbar(
        im_c,
        ax=[ax_c_left, ax_c_right, ax_d_left, ax_d_right],
        orientation="horizontal",
        fraction=0.06,
        pad=0.12,
    )
    cbar.set_label("log10 probability", fontsize=10.5)
    cbar.ax.tick_params(labelsize=9.2)

    pos_c_right = ax_c_right.get_position()
    pos_d_right = ax_d_right.get_position()
    y_xlabel = cbar.ax.get_position().y1 + 0.012
    fig.text((pos_c.x0 + pos_c_right.x1) / 2, y_xlabel, "Coarse education × employment\n× income groups", ha="center", va="bottom", fontsize=11)
    fig.text((pos_d.x0 + pos_d_right.x1) / 2, y_xlabel, "Education × employment\n× income combinations", ha="center", va="bottom", fontsize=11)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--representative_puma_uid", default="2602903")
    ap.add_argument("--example_seed_label", default="seed2")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df, stagewise_summary, representative = _build_metric_table(
        representative_puma_uid=str(args.representative_puma_uid),
        example_seed_label=str(args.example_seed_label),
    )
    (out_dir / "hierarchical_recovery_by_puma.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (out_dir / "hierarchical_recovery_summary.json").write_text(
        json.dumps(
            {
                "stagewise_summary": stagewise_summary,
                "representative_puma_uid": representative["puma_uid"],
                "example_seed_label": str(args.example_seed_label),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _plot_panels(
        df=df,
        stagewise_summary=stagewise_summary,
        representative=representative,
        out_png=out_dir / "fig_06_hierarchical_recovery.png",
    )


if __name__ == "__main__":
    main()
