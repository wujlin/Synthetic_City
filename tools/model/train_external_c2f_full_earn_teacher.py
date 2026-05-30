#!/usr/bin/env python3
from __future__ import annotations

"""
Train the teacher-forced stage-2 coarse-to-fine diffusion model for the 5-way full-earn task.

Each row is a (PUMA, coarse parent cell).
Target:
  local fine conditional distribution over the child cells of that parent,
  represented in a padded slot space of width MAX_CHILDREN.

Condition:
  - full 288-cell coarse table for the region
  - parent one-hot
  - child-slot mask
  - parent mass

This is a learnability probe, not an end-to-end pipeline.
"""

import argparse
import datetime as _dt
import json
import pathlib
import random
import sys
from collections import defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.model.diffusion_tabular import DiffusionTabularModel, TabDDPMConfig
from tools.model.external_c2f_full_earn_stage2_model import (
    SharedConditionStage2Diffusion,
    sample_stage2_local_raw_batch,
)
from tools.model.train_us_puma_5var_diffusion import (
    _canon_statefp,
    _cosine,
    _parse_hidden_dims,
    _require_torch,
    _summ,
    _tvd,
    _utc_now_iso,
    _write_json,
)


def _project_to_child_mask(p_raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    p_raw = np.asarray(p_raw, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    active = mask > 0.5
    out = np.zeros_like(p_raw, dtype=np.float64)
    if int(active.sum()) == 0:
        return out
    active_mass = float(p_raw[active].sum())
    if active_mass > 0:
        out[active] = p_raw[active] / active_mass
    else:
        out[active] = 1.0 / float(active.sum())
    out = out / max(float(out.sum()), 1e-12)
    return out


def _uniform_from_child_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    active = mask > 0.5
    out = np.zeros_like(mask, dtype=np.float64)
    if int(active.sum()) == 0:
        return out
    out[active] = 1.0 / float(active.sum())
    return out


def _structured_cond_rows(
    *,
    row_idx: np.ndarray,
    group_idx: np.ndarray,
    parent_idx: np.ndarray,
    coarse_by_group: np.ndarray,
    child_mask_by_parent: np.ndarray,
    coarse_k: int,
    target_dim: int,
) -> np.ndarray:
    rows = np.asarray(row_idx, dtype=np.int64).reshape(-1)
    parents = np.asarray(parent_idx[rows], dtype=np.int64)
    coarse = np.asarray(coarse_by_group[np.asarray(group_idx[rows], dtype=np.int64)], dtype=np.float32)
    onehot = np.zeros((rows.shape[0], int(coarse_k)), dtype=np.float32)
    onehot[np.arange(rows.shape[0]), parents] = 1.0
    mask = np.asarray(child_mask_by_parent[parents], dtype=np.float32)
    mass = coarse[np.arange(rows.shape[0]), parents].reshape(-1, 1).astype(np.float32)
    return np.concatenate([coarse, onehot, mask, mass], axis=1).astype(np.float32, copy=False)


def _load_structured_stage2_arrays(
    *,
    in_path: pathlib.Path,
    schema: dict[str, Any],
    heldout_statefp: str,
    p_joint_cols: list[str],
    read_chunksize: int,
) -> dict[str, Any]:
    coarse_cols = list(schema["condition_blocks"]["coarse_table"])
    header_cols = list(pd.read_csv(in_path, nrows=0).columns)
    has_condition_source = "condition_source" in header_cols
    meta_cols = ["statefp", "puma_uid", "parent_idx", "child_count"]
    if has_condition_source:
        meta_cols.append("condition_source")
    usecols = meta_cols + coarse_cols + p_joint_cols

    missing = [c for c in usecols if c not in header_cols]
    if missing:
        raise SystemExit(f"wide_csv missing columns for structured_low_memory: {missing[:8]}")

    dtype: dict[str, Any] = {
        "statefp": "string",
        "puma_uid": "string",
        "parent_idx": np.int32,
        "child_count": np.int16,
    }
    if has_condition_source:
        dtype["condition_source"] = "string"
    dtype.update({c: np.float32 for c in coarse_cols})
    dtype.update({c: np.float32 for c in p_joint_cols})

    group_map: dict[str, int] = {}
    coarse_chunks: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    parent_parts: list[np.ndarray] = []
    child_count_parts: list[np.ndarray] = []
    heldout_parts: list[np.ndarray] = []
    p_joint_parts: list[np.ndarray] = []

    n_rows = 0
    chunksize = max(int(read_chunksize), 1)
    print(f"[structured] reading {in_path} with chunksize={chunksize}", file=sys.stderr)
    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            in_path,
            usecols=usecols,
            dtype=dtype,
            chunksize=chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        chunk["statefp"] = chunk["statefp"].map(_canon_statefp)
        if has_condition_source:
            key_series = chunk["puma_uid"].astype(str) + "__" + chunk["condition_source"].astype(str)
        else:
            key_series = chunk["puma_uid"].astype(str) + "__default"

        keys = key_series.to_numpy(dtype=object)
        group_ids = np.empty(len(keys), dtype=np.int32)
        for pos, key in enumerate(keys):
            gid = group_map.get(str(key))
            if gid is None:
                gid = len(group_map)
                group_map[str(key)] = gid
                coarse_row = chunk.iloc[pos][coarse_cols].to_numpy(dtype=np.float32)
                coarse_row = np.clip(coarse_row, 0.0, None)
                coarse_row = coarse_row / max(float(coarse_row.sum()), 1e-12)
                coarse_chunks.append(coarse_row.astype(np.float32, copy=False))
            group_ids[pos] = int(gid)

        p_joint = chunk[p_joint_cols].to_numpy(dtype=np.float32)
        p_joint = np.clip(p_joint, 0.0, None)
        p_joint = p_joint / np.maximum(p_joint.sum(axis=1, keepdims=True), 1e-12)

        group_parts.append(group_ids)
        parent_parts.append(chunk["parent_idx"].to_numpy(dtype=np.int32))
        child_count_parts.append(chunk["child_count"].to_numpy(dtype=np.int16))
        heldout_parts.append((chunk["statefp"].to_numpy(dtype=str) == str(heldout_statefp)))
        p_joint_parts.append(p_joint.astype(np.float32, copy=False))
        n_rows += int(chunk.shape[0])
        if chunk_idx == 1 or chunk_idx % 20 == 0:
            print(
                f"[structured] chunk={chunk_idx} rows={n_rows} groups={len(group_map)}",
                file=sys.stderr,
            )

    if n_rows == 0:
        raise SystemExit("structured_low_memory loaded zero rows.")
    coarse_by_group = np.vstack(coarse_chunks).astype(np.float32, copy=False)
    group_idx = np.concatenate(group_parts).astype(np.int32, copy=False)
    parent_idx = np.concatenate(parent_parts).astype(np.int32, copy=False)
    child_count = np.concatenate(child_count_parts).astype(np.int16, copy=False)
    is_heldout = np.concatenate(heldout_parts).astype(bool, copy=False)
    p_joint = np.vstack(p_joint_parts).astype(np.float32, copy=False)

    return {
        "coarse_by_group": coarse_by_group,
        "group_idx": group_idx,
        "parent_idx": parent_idx,
        "child_count": child_count,
        "is_heldout": is_heldout,
        "p_joint": p_joint,
        "group_count": int(coarse_by_group.shape[0]),
        "n_rows_total": int(n_rows),
    }


def _evaluate_teacher_forced_stage2_structured(
    *,
    model: Any,
    group_idx: np.ndarray,
    parent_idx: np.ndarray,
    coarse_by_group: np.ndarray,
    child_mask_by_parent: np.ndarray,
    p_joint: np.ndarray,
    child_count: np.ndarray,
    eval_idx: np.ndarray,
    n_eval_joint_samples: int,
    device: str | None,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    coarse_k: int,
    target_dim: int,
) -> dict[str, Any]:
    tvd_raw: list[float] = []
    tvd_projected: list[float] = []
    cosine_raw: list[float] = []
    cosine_projected: list[float] = []
    inactive_mass_raw: list[float] = []
    tvd_uniform: list[float] = []
    by_child_count_proj: dict[int, list[float]] = defaultdict(list)
    by_child_count_raw: dict[int, list[float]] = defaultdict(list)

    for idx in eval_idx.tolist():
        c_row = _structured_cond_rows(
            row_idx=np.asarray([idx], dtype=np.int64),
            group_idx=group_idx,
            parent_idx=parent_idx,
            coarse_by_group=coarse_by_group,
            child_mask_by_parent=child_mask_by_parent,
            coarse_k=int(coarse_k),
            target_dim=int(target_dim),
        )[0]
        mask = np.asarray(child_mask_by_parent[int(parent_idx[idx])], dtype=np.float32)
        p_true = np.asarray(p_joint[idx], dtype=np.float32)
        cnt = int(child_count[idx])

        p_hat_raw = sample_stage2_local_raw_batch(
            model=model,
            cond_rows=c_row.reshape(1, -1),
            child_mask_rows=mask.reshape(1, -1),
            n_draws=int(n_eval_joint_samples),
            device=device,
            x_mean=x_mean,
            x_std=x_std,
        )[0]
        p_hat_projected = _project_to_child_mask(p_hat_raw, mask)
        p_uniform = _uniform_from_child_mask(mask)

        tvd_raw.append(_tvd(p_hat_raw, p_true))
        tvd_projected.append(_tvd(p_hat_projected, p_true))
        cosine_raw.append(_cosine(p_hat_raw, p_true))
        cosine_projected.append(_cosine(p_hat_projected, p_true))
        inactive_mass_raw.append(float(p_hat_raw[np.asarray(mask <= 0.5)].sum()))
        tvd_uniform.append(_tvd(p_uniform, p_true))
        by_child_count_proj[cnt].append(_tvd(p_hat_projected, p_true))
        by_child_count_raw[cnt].append(_tvd(p_hat_raw, p_true))

    return {
        "teacher_forced_stage2": {
            "tvd_joint_raw": _summ(tvd_raw),
            "tvd_joint_projected": _summ(tvd_projected),
            "cosine_joint_raw": _summ(cosine_raw),
            "cosine_joint_projected": _summ(cosine_projected),
            "inactive_mass_raw": _summ(inactive_mass_raw),
        },
        "baseline_uniform_active": {
            "tvd_joint": _summ(tvd_uniform),
        },
        "by_child_count": {
            str(k): {
                "n_rows": int(len(v)),
                "tvd_joint_projected": _summ(v),
                "tvd_joint_raw": _summ(by_child_count_raw[k]),
            }
            for k, v in sorted(by_child_count_proj.items())
        },
    }


def _evaluate_teacher_forced_stage2(
    *,
    model: Any,
    cond: np.ndarray,
    p_joint: np.ndarray,
    child_mask: np.ndarray,
    child_count: np.ndarray,
    eval_idx: np.ndarray,
    n_eval_joint_samples: int,
    device: str | None,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> dict[str, Any]:
    tvd_raw: list[float] = []
    tvd_projected: list[float] = []
    cosine_raw: list[float] = []
    cosine_projected: list[float] = []
    inactive_mass_raw: list[float] = []
    tvd_uniform: list[float] = []
    by_child_count_proj: dict[int, list[float]] = defaultdict(list)
    by_child_count_raw: dict[int, list[float]] = defaultdict(list)

    for idx in eval_idx.tolist():
        c_row = cond[idx]
        p_true = p_joint[idx]
        mask = child_mask[idx]
        cnt = int(child_count[idx])

        p_hat_raw = sample_stage2_local_raw_batch(
            model=model,
            cond_rows=c_row.reshape(1, -1),
            child_mask_rows=mask.reshape(1, -1),
            n_draws=int(n_eval_joint_samples),
            device=device,
            x_mean=x_mean,
            x_std=x_std,
        )[0]
        p_hat_projected = _project_to_child_mask(p_hat_raw, mask)
        p_uniform = _uniform_from_child_mask(mask)

        tvd_raw.append(_tvd(p_hat_raw, p_true))
        tvd_projected.append(_tvd(p_hat_projected, p_true))
        cosine_raw.append(_cosine(p_hat_raw, p_true))
        cosine_projected.append(_cosine(p_hat_projected, p_true))
        inactive_mass_raw.append(float(p_hat_raw[np.asarray(mask <= 0.5)].sum()))
        tvd_uniform.append(_tvd(p_uniform, p_true))
        by_child_count_proj[cnt].append(_tvd(p_hat_projected, p_true))
        by_child_count_raw[cnt].append(_tvd(p_hat_raw, p_true))

    return {
        "teacher_forced_stage2": {
            "tvd_joint_raw": _summ(tvd_raw),
            "tvd_joint_projected": _summ(tvd_projected),
            "cosine_joint_raw": _summ(cosine_raw),
            "cosine_joint_projected": _summ(cosine_projected),
            "inactive_mass_raw": _summ(inactive_mass_raw),
        },
        "baseline_uniform_active": {
            "tvd_joint": _summ(tvd_uniform),
        },
        "by_child_count": {
            str(k): {
                "n_rows": int(len(v)),
                "tvd_joint_projected": _summ(v),
                "tvd_joint_raw": _summ(by_child_count_raw[k]),
            }
            for k, v in sorted(by_child_count_proj.items())
        },
    }


def _fit_weighted_stage2_diffusion(
    *,
    model: DiffusionTabularModel,
    x: Any,
    cond: Any,
    child_mask: Any,
    epochs: int,
    batch_size: int,
    device: str | None,
    log_every: int,
    diff_loss_reweight_alpha: float,
    diff_loss_reweight_floor: float,
    diff_loss_reweight_cap: float,
    epoch_end_callback: Callable[[int], None] | None = None,
) -> dict[str, float]:
    torch = _require_torch()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(model.seed)
    torch.manual_seed(model.seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(model.seed)

    x = x.to(device=device, dtype=torch.float32)
    cond = cond.to(device=device, dtype=torch.float32)
    child_mask = child_mask.to(device=device, dtype=torch.float32)
    if x.ndim != 2 or x.shape[1] != model.input_dim:
        raise ValueError(f"x must be (N,{model.input_dim}), got {tuple(x.shape)}")
    if cond.ndim != 2 or cond.shape[1] != model.cond_dim or cond.shape[0] != x.shape[0]:
        raise ValueError(f"cond must be (N,{model.cond_dim}), got {tuple(cond.shape)}")
    if child_mask.shape != x.shape:
        raise ValueError(f"child_mask must match x shape {tuple(x.shape)}, got {tuple(child_mask.shape)}")

    model._init_model(device=device)
    assert model._net is not None
    assert model._schedule is not None
    model._net.train()

    optim = torch.optim.AdamW(model._net.parameters(), lr=model.config.lr, weight_decay=model.config.weight_decay)

    num_rows = int(x.shape[0])
    num_steps = 0
    last_loss = float("nan")
    last_weight_mean = float("nan")
    last_active_weight_mean = float("nan")
    last_inactive_weight_mean = float("nan")

    for epoch_idx in range(int(epochs)):
        indices = torch.randperm(num_rows, device=device)
        for start in range(0, num_rows, int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            batch_x0 = x[batch_idx]
            batch_cond = cond[batch_idx]
            batch_mask = child_mask[batch_idx]

            t = torch.randint(0, model.config.timesteps, (batch_x0.shape[0],), device=device)
            noise = torch.randn_like(batch_x0)
            sqrt_acp = model._schedule["sqrt_alpha_cumprod"][t].unsqueeze(1)
            sqrt_om = model._schedule["sqrt_one_minus_alpha_cumprod"][t].unsqueeze(1)
            x_t = sqrt_acp * batch_x0 + sqrt_om * noise

            eps_pred = model._net(x_t, t, batch_cond)

            active_count = torch.clamp(batch_mask.sum(dim=1, keepdim=True), min=1.0)
            support = batch_mask / active_count
            cell_weight = torch.pow(torch.clamp(support, min=1e-8), float(diff_loss_reweight_alpha))
            cell_weight = cell_weight / torch.clamp(cell_weight.mean(dim=1, keepdim=True), min=1e-8)
            cell_weight = torch.clamp(
                cell_weight,
                min=float(diff_loss_reweight_floor),
                max=float(diff_loss_reweight_cap),
            )
            loss = (cell_weight * torch.square(eps_pred - noise)).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if model.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model._net.parameters(), model.config.grad_clip)
            optim.step()

            last_loss = float(loss.detach().cpu().item())
            last_weight_mean = float(cell_weight.mean().detach().cpu().item())
            active_mask = batch_mask > 0.5
            inactive_mask = ~active_mask
            if bool(active_mask.any()):
                last_active_weight_mean = float(cell_weight[active_mask].mean().detach().cpu().item())
            if bool(inactive_mask.any()):
                last_inactive_weight_mean = float(cell_weight[inactive_mask].mean().detach().cpu().item())
            num_steps += 1
            if log_every > 0 and num_steps % int(log_every) == 0:
                print(
                    f"[train] step={num_steps} loss={last_loss:.6f} "
                    f"w_mean={last_weight_mean:.4f} "
                    f"w_active={last_active_weight_mean:.4f} "
                    f"w_inactive={last_inactive_weight_mean:.4f}"
                )
        if epoch_end_callback is not None:
            epoch_end_callback(int(epoch_idx) + 1)

    return {
        "loss": float(last_loss),
        "cell_weight_mean": float(last_weight_mean),
        "cell_weight_active_mean": float(last_active_weight_mean),
        "cell_weight_inactive_mean": float(last_inactive_weight_mean),
    }


def _fit_shared_stage2_diffusion(
    *,
    model: SharedConditionStage2Diffusion,
    x: Any,
    p_true: Any,
    cond: Any,
    child_mask: Any,
    epochs: int,
    batch_size: int,
    device: str | None,
    log_every: int,
    x_mean: Any,
    x_std: Any,
    diff_loss_reweight_alpha: float,
    diff_loss_reweight_floor: float,
    diff_loss_reweight_cap: float,
    clean_head_weight: float,
    consistency_weight: float,
    income_regime_target: Any | None,
    income_regime_mask: Any | None,
    income_regime_slot: Any | None,
    income_regime_weight: float,
    income_regime_consistency_weight: float,
    aux_t_gate: int,
    epoch_end_callback: Callable[[int], None] | None = None,
) -> dict[str, float]:
    torch = _require_torch()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(model.seed)
    torch.manual_seed(model.seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(model.seed)

    x = x.to(device=device, dtype=torch.float32)
    p_true = p_true.to(device=device, dtype=torch.float32)
    cond = cond.to(device=device, dtype=torch.float32)
    child_mask = child_mask.to(device=device, dtype=torch.float32)
    x_mean = x_mean.to(device=device, dtype=torch.float32)
    x_std = x_std.to(device=device, dtype=torch.float32)
    if income_regime_target is not None:
        income_regime_target = income_regime_target.to(device=device, dtype=torch.float32)
    if income_regime_mask is not None:
        income_regime_mask = income_regime_mask.to(device=device, dtype=torch.float32)
    if income_regime_slot is not None:
        income_regime_slot = income_regime_slot.to(device=device, dtype=torch.long)
    if x.ndim != 2 or x.shape[1] != model.input_dim:
        raise ValueError(f"x must be (N,{model.input_dim}), got {tuple(x.shape)}")
    if p_true.shape != x.shape:
        raise ValueError(f"p_true must match x shape {tuple(x.shape)}, got {tuple(p_true.shape)}")
    if cond.ndim != 2 or cond.shape[1] != model.cond_raw_dim or cond.shape[0] != x.shape[0]:
        raise ValueError(f"cond must be (N,{model.cond_raw_dim}), got {tuple(cond.shape)}")
    if child_mask.shape != x.shape:
        raise ValueError(f"child_mask must match x shape {tuple(x.shape)}, got {tuple(child_mask.shape)}")

    model.to(device)
    model.train()

    num_rows = int(x.shape[0])
    num_steps = 0
    last_metrics: dict[str, float] = {
        "loss": float("nan"),
        "loss_diffusion": float("nan"),
        "loss_clean_head": float("nan"),
        "loss_consistency": float("nan"),
        "loss_income_regime": float("nan"),
        "loss_income_regime_consistency": float("nan"),
        "aux_mask_frac": float("nan"),
        "cell_weight_mean": float("nan"),
        "cell_weight_active_mean": float("nan"),
        "cell_weight_inactive_mean": float("nan"),
    }

    for epoch_idx in range(int(epochs)):
        indices = torch.randperm(num_rows, device=device)
        for start in range(0, num_rows, int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            last_metrics = model.step(
                x0=x[batch_idx],
                cond_raw=cond[batch_idx],
                p_true=p_true[batch_idx],
                child_mask=child_mask[batch_idx],
                x_mean=x_mean,
                x_std=x_std,
                diff_loss_reweight_alpha=float(diff_loss_reweight_alpha),
                diff_loss_reweight_floor=float(diff_loss_reweight_floor),
                diff_loss_reweight_cap=float(diff_loss_reweight_cap),
                clean_head_weight=float(clean_head_weight),
                consistency_weight=float(consistency_weight),
                income_regime_target=None if income_regime_target is None else income_regime_target[batch_idx],
                income_regime_mask=None if income_regime_mask is None else income_regime_mask[batch_idx],
                income_regime_slot=None if income_regime_slot is None else income_regime_slot[batch_idx],
                income_regime_weight=float(income_regime_weight),
                income_regime_consistency_weight=float(income_regime_consistency_weight),
                aux_t_gate=int(aux_t_gate),
            )
            num_steps += 1
            if log_every > 0 and num_steps % int(log_every) == 0:
                print(
                    f"[train] step={num_steps} loss={last_metrics['loss']:.6f} "
                    f"diff={last_metrics['loss_diffusion']:.6f} "
                    f"head={last_metrics['loss_clean_head']:.6f} "
                    f"cons={last_metrics['loss_consistency']:.6f} "
                    f"reg={last_metrics['loss_income_regime']:.6f} "
                    f"reg_cons={last_metrics['loss_income_regime_consistency']:.6f} "
                    f"aux_frac={last_metrics['aux_mask_frac']:.4f} "
                    f"w_mean={last_metrics['cell_weight_mean']:.4f} "
                    f"w_active={last_metrics['cell_weight_active_mean']:.4f} "
                    f"w_inactive={last_metrics['cell_weight_inactive_mean']:.4f}"
                )
        if epoch_end_callback is not None:
            epoch_end_callback(int(epoch_idx) + 1)

    return {k: float(v) for k, v in last_metrics.items()}


def _fit_shared_stage2_diffusion_structured(
    *,
    model: SharedConditionStage2Diffusion,
    x_train_all: np.ndarray,
    p_joint: np.ndarray,
    group_idx: np.ndarray,
    parent_idx: np.ndarray,
    train_idx: np.ndarray,
    coarse_by_group: np.ndarray,
    child_mask_by_parent: np.ndarray,
    coarse_k: int,
    target_dim: int,
    epochs: int,
    batch_size: int,
    device: str | None,
    log_every: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    diff_loss_reweight_alpha: float,
    diff_loss_reweight_floor: float,
    diff_loss_reweight_cap: float,
    clean_head_weight: float,
    consistency_weight: float,
    aux_t_gate: int,
    epoch_end_callback: Callable[[int], None] | None = None,
) -> dict[str, float]:
    torch = _require_torch()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(model.seed)
    np.random.seed(model.seed)
    torch.manual_seed(model.seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(model.seed)

    model.to(device)
    model.train()
    x_mean_t = torch.from_numpy(np.asarray(x_mean, dtype=np.float32).reshape(1, -1)).to(device=device, dtype=torch.float32)
    x_std_t = torch.from_numpy(np.asarray(x_std, dtype=np.float32).reshape(1, -1)).to(device=device, dtype=torch.float32)

    num_rows = int(train_idx.shape[0])
    num_steps = 0
    rng = np.random.default_rng(int(model.seed))
    last_metrics: dict[str, float] = {
        "loss": float("nan"),
        "loss_diffusion": float("nan"),
        "loss_clean_head": float("nan"),
        "loss_consistency": float("nan"),
        "loss_income_regime": float("nan"),
        "loss_income_regime_consistency": float("nan"),
        "aux_mask_frac": float("nan"),
        "cell_weight_mean": float("nan"),
        "cell_weight_active_mean": float("nan"),
        "cell_weight_inactive_mean": float("nan"),
    }

    for epoch_idx in range(int(epochs)):
        order = rng.permutation(num_rows)
        for start in range(0, num_rows, int(batch_size)):
            rows = train_idx[order[start : start + int(batch_size)]]
            parents = np.asarray(parent_idx[rows], dtype=np.int64)
            coarse_np = np.asarray(coarse_by_group[np.asarray(group_idx[rows], dtype=np.int64)], dtype=np.float32)
            mask_np = np.asarray(child_mask_by_parent[parents], dtype=np.float32)

            x0 = torch.from_numpy(np.asarray(x_train_all[rows], dtype=np.float32)).to(device=device, dtype=torch.float32)
            p_true_t = torch.from_numpy(np.asarray(p_joint[rows], dtype=np.float32)).to(device=device, dtype=torch.float32)
            coarse_t = torch.from_numpy(coarse_np).to(device=device, dtype=torch.float32)
            mask_t = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
            parent_t = torch.from_numpy(parents).to(device=device, dtype=torch.long)

            onehot_t = torch.zeros((rows.shape[0], int(coarse_k)), device=device, dtype=torch.float32)
            onehot_t.scatter_(1, parent_t.reshape(-1, 1), 1.0)
            parent_mass_t = torch.gather(coarse_t, 1, parent_t.reshape(-1, 1))
            cond_t = torch.cat([coarse_t, onehot_t, mask_t, parent_mass_t], dim=1)
            if cond_t.shape[1] != model.cond_raw_dim:
                raise ValueError(f"structured cond dim {cond_t.shape[1]} != model cond_raw_dim {model.cond_raw_dim}")
            if x0.shape[1] != int(target_dim):
                raise ValueError(f"x target dim {x0.shape[1]} != target_dim {target_dim}")

            last_metrics = model.step(
                x0=x0,
                cond_raw=cond_t,
                p_true=p_true_t,
                child_mask=mask_t,
                x_mean=x_mean_t,
                x_std=x_std_t,
                diff_loss_reweight_alpha=float(diff_loss_reweight_alpha),
                diff_loss_reweight_floor=float(diff_loss_reweight_floor),
                diff_loss_reweight_cap=float(diff_loss_reweight_cap),
                clean_head_weight=float(clean_head_weight),
                consistency_weight=float(consistency_weight),
                income_regime_target=None,
                income_regime_mask=None,
                income_regime_slot=None,
                income_regime_weight=0.0,
                income_regime_consistency_weight=0.0,
                aux_t_gate=int(aux_t_gate),
            )
            num_steps += 1
            if log_every > 0 and num_steps % int(log_every) == 0:
                print(
                    f"[train] step={num_steps} loss={last_metrics['loss']:.6f} "
                    f"diff={last_metrics['loss_diffusion']:.6f} "
                    f"head={last_metrics['loss_clean_head']:.6f} "
                    f"cons={last_metrics['loss_consistency']:.6f} "
                    f"reg={last_metrics['loss_income_regime']:.6f} "
                    f"reg_cons={last_metrics['loss_income_regime_consistency']:.6f} "
                    f"aux_frac={last_metrics['aux_mask_frac']:.4f} "
                    f"w_mean={last_metrics['cell_weight_mean']:.4f} "
                    f"w_active={last_metrics['cell_weight_active_mean']:.4f} "
                    f"w_inactive={last_metrics['cell_weight_inactive_mean']:.4f}"
                )
        if epoch_end_callback is not None:
            epoch_end_callback(int(epoch_idx) + 1)

    return {k: float(v) for k, v in last_metrics.items()}


def main() -> None:
    ap = argparse.ArgumentParser(prog="train_external_c2f_full_earn_teacher")
    ap.add_argument("--wide_csv", required=True)
    ap.add_argument("--schema_json", required=True)
    ap.add_argument("--eval_mode", choices=["leave_mi_out", "leave_state_out"], default="leave_mi_out")
    ap.add_argument("--heldout_statefp", default="26", help="State FIPS used by leave-state-out evaluation. Default 26 keeps the original Michigan split.")
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--hidden_dims", default="256,256")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--condition_injection", choices=["concat", "film"], default="concat")
    ap.add_argument("--film_hidden_dim", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--encoder_hidden_dims", default="256,256")
    ap.add_argument("--head_hidden_dims", default="256")
    ap.add_argument("--clean_head_weight", type=float, default=0.0)
    ap.add_argument("--consistency_weight", type=float, default=0.0)
    ap.add_argument("--income_regime_weight", type=float, default=0.0)
    ap.add_argument("--income_regime_consistency_weight", type=float, default=0.0)
    ap.add_argument("--aux_t_gate", type=int, default=-1)
    ap.add_argument("--predict_mode", choices=["diffusion", "head", "blend"], default="diffusion")
    ap.add_argument("--blend_alpha", type=float, default=0.0)
    ap.add_argument("--regime_projection_alpha", type=float, default=0.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--n_eval_joint_samples", type=int, default=128)
    ap.add_argument("--diff_loss_reweight_alpha", type=float, default=0.0)
    ap.add_argument("--diff_loss_reweight_floor", type=float, default=0.05)
    ap.add_argument("--diff_loss_reweight_cap", type=float, default=5.0)
    ap.add_argument("--save_final_model", action="store_true")
    ap.add_argument("--save_best_model", action="store_true")
    ap.add_argument("--eval_every_epochs", type=int, default=0)
    ap.add_argument(
        "--structured_low_memory",
        action="store_true",
        help=(
            "Load stage-2 data as structured arrays and build repeated condition "
            "blocks per batch. This avoids materializing the dense wide condition "
            "matrix and is intended for high-K coarse presets."
        ),
    )
    ap.add_argument("--read_chunksize", type=int, default=10000)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    if str(args.predict_mode) != "diffusion" and float(args.clean_head_weight) <= 0.0:
        raise SystemExit("predict_mode=head/blend requires clean_head_weight > 0.")
    if float(args.consistency_weight) > 0.0 and float(args.clean_head_weight) <= 0.0:
        raise SystemExit("consistency_weight > 0 requires clean_head_weight > 0.")
    if not (0.0 <= float(args.blend_alpha) <= 1.0):
        raise SystemExit("blend_alpha must be in [0,1].")
    if not (0.0 <= float(args.regime_projection_alpha) <= 1.0):
        raise SystemExit("regime_projection_alpha must be in [0,1].")
    if float(args.regime_projection_alpha) > 0.0 and float(args.income_regime_weight) <= 0.0:
        raise SystemExit("regime_projection_alpha > 0 requires income_regime_weight > 0.")

    torch = _require_torch()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    in_path = pathlib.Path(args.wide_csv).expanduser().resolve()
    schema_path = pathlib.Path(args.schema_json).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"wide_csv not found: {in_path}")
    if not schema_path.exists():
        raise SystemExit(f"schema_json not found: {schema_path}")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    target_dim = int(schema["target_dim"])
    cond_blocks = dict(schema["condition_blocks"])
    aux_blocks = dict(schema.get("auxiliary_blocks", {}))
    cond_cols = (
        list(cond_blocks["coarse_table"])
        + list(cond_blocks["parent_onehot"])
        + list(cond_blocks["child_mask"])
        + list(cond_blocks["parent_mass"])
    )
    income_regime_target_cols = list(aux_blocks.get("income_regime_target", []))
    income_regime_mask_cols = list(aux_blocks.get("income_regime_mask", []))
    parent_child_income_regime_padded = schema.get("parent_child_income_regime_padded")
    p_joint_cols = [f"p_joint_{i:03d}" for i in range(target_dim)]
    child_mask_cols = list(cond_blocks["child_mask"])

    run_id = f"_us_puma_external_c2f_full_earn_teacher_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else (_REPO_ROOT / "outputs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    heldout_statefp = _canon_statefp(args.heldout_statefp)
    fold_label = "leave_mi_out" if heldout_statefp == "26" else f"leave_state_{heldout_statefp}_out"

    if bool(args.structured_low_memory):
        if (
            float(args.income_regime_weight) > 0.0
            or float(args.income_regime_consistency_weight) > 0.0
            or float(args.regime_projection_alpha) > 0.0
        ):
            raise SystemExit("structured_low_memory currently supports the non-income-regime stage-2 path only.")

        coarse_k = int(schema.get("coarse_K", 0) or len(cond_blocks["coarse_table"]))
        child_mask_by_parent = np.asarray(schema["parent_child_slot_mask"], dtype=np.float32)
        if child_mask_by_parent.shape[0] != coarse_k or child_mask_by_parent.shape[1] != target_dim:
            raise SystemExit(
                "schema parent_child_slot_mask shape does not match "
                f"coarse_K/target_dim: {child_mask_by_parent.shape} vs ({coarse_k}, {target_dim})"
            )
        cond_dim = int(len(cond_cols))
        expected_cond_dim = int(coarse_k + coarse_k + target_dim + 1)
        if cond_dim != expected_cond_dim:
            raise SystemExit(f"structured_low_memory expects cond_dim={expected_cond_dim}, got {cond_dim}")

        structured = _load_structured_stage2_arrays(
            in_path=in_path,
            schema=schema,
            heldout_statefp=heldout_statefp,
            p_joint_cols=p_joint_cols,
            read_chunksize=int(args.read_chunksize),
        )
        p_joint = np.asarray(structured["p_joint"], dtype=np.float32)
        group_idx = np.asarray(structured["group_idx"], dtype=np.int32)
        parent_idx = np.asarray(structured["parent_idx"], dtype=np.int32)
        child_count = np.asarray(structured["child_count"], dtype=np.int16)
        is_heldout_arr = np.asarray(structured["is_heldout"], dtype=bool)
        coarse_by_group = np.asarray(structured["coarse_by_group"], dtype=np.float32)

        train_idx = np.where(~is_heldout_arr)[0]
        test_idx = np.where(is_heldout_arr)[0]
        if train_idx.size == 0 or test_idx.size == 0:
            raise SystemExit("Invalid leave-state-out split in structured_low_memory path.")

        x_log_all = np.log(np.clip(p_joint, 0.0, None) + 1e-6).astype(np.float32)
        x_train_log = x_log_all[train_idx]
        x_mean = x_train_log.mean(axis=0, dtype=np.float64).astype(np.float32)
        x_std = x_train_log.std(axis=0, dtype=np.float64).astype(np.float32)
        x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
        x_train_all = ((x_log_all - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)).astype(np.float32)

        hidden_dims = _parse_hidden_dims(args.hidden_dims)
        encoder_hidden_dims = _parse_hidden_dims(args.encoder_hidden_dims)
        head_hidden_dims = _parse_hidden_dims(args.head_hidden_dims)
        use_clean_head = (
            float(args.clean_head_weight) > 0.0
            or float(args.consistency_weight) > 0.0
            or str(args.predict_mode) != "diffusion"
        )
        if not use_clean_head:
            raise SystemExit("structured_low_memory currently expects the shared clean-head stage-2 model.")
        model_cfg = TabDDPMConfig(
            timesteps=int(args.timesteps),
            hidden_dims=hidden_dims,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            condition_injection=str(args.condition_injection),
            film_hidden_dim=int(args.film_hidden_dim),
        )
        model = SharedConditionStage2Diffusion(
            input_dim=target_dim,
            cond_raw_dim=cond_dim,
            latent_dim=int(args.latent_dim),
            encoder_hidden_dims=encoder_hidden_dims,
            head_hidden_dims=head_hidden_dims,
            diffusion_config=model_cfg,
            seed=int(args.seed),
            enable_clean_head=bool(use_clean_head),
            aux_regime_dim=0,
            predict_mode=str(args.predict_mode),
            blend_alpha=float(args.blend_alpha),
            regime_projection_alpha=0.0,
        )

        saved_checkpoints: list[str] = []
        best_projected_tvd = float("inf")
        best_epoch: int | None = None
        best_ckpt_path: pathlib.Path | None = None
        eval_every_epochs = int(args.eval_every_epochs)

        def _maybe_eval_and_save_structured(epoch_idx: int) -> None:
            nonlocal best_projected_tvd, best_epoch, best_ckpt_path, saved_checkpoints
            if eval_every_epochs <= 0 or epoch_idx % eval_every_epochs != 0:
                return
            ablation = _evaluate_teacher_forced_stage2_structured(
                model=model,
                group_idx=group_idx,
                parent_idx=parent_idx,
                coarse_by_group=coarse_by_group,
                child_mask_by_parent=child_mask_by_parent,
                p_joint=p_joint,
                child_count=child_count,
                eval_idx=test_idx,
                n_eval_joint_samples=int(args.n_eval_joint_samples),
                device=args.device,
                x_mean=x_mean,
                x_std=x_std,
                coarse_k=coarse_k,
                target_dim=target_dim,
            )
            projected = float(ablation["teacher_forced_stage2"]["tvd_joint_projected"]["mean"])
            print(f"[eval] epoch={epoch_idx} tvd_joint_projected={projected:.6f}", file=sys.stderr)
            if bool(args.save_best_model) and projected < best_projected_tvd:
                best_projected_tvd = projected
                best_epoch = int(epoch_idx)
                best_ckpt_path = (
                    out_dir
                    / "checkpoints"
                    / "external_c2f_full_earn_teacher"
                    / fold_label
                    / "best.pt"
                )
                model.save(best_ckpt_path)
                best_str = str(best_ckpt_path)
                if best_str not in saved_checkpoints:
                    saved_checkpoints.append(best_str)
                print(f"[eval] epoch={epoch_idx} new_best_tvd_joint_projected={projected:.6f}", file=sys.stderr)

        train_fit_summary = _fit_shared_stage2_diffusion_structured(
            model=model,
            x_train_all=x_train_all,
            p_joint=p_joint,
            group_idx=group_idx,
            parent_idx=parent_idx,
            train_idx=train_idx,
            coarse_by_group=coarse_by_group,
            child_mask_by_parent=child_mask_by_parent,
            coarse_k=coarse_k,
            target_dim=target_dim,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            device=args.device,
            log_every=int(args.log_every),
            x_mean=x_mean,
            x_std=x_std,
            diff_loss_reweight_alpha=float(args.diff_loss_reweight_alpha),
            diff_loss_reweight_floor=float(args.diff_loss_reweight_floor),
            diff_loss_reweight_cap=float(args.diff_loss_reweight_cap),
            clean_head_weight=float(args.clean_head_weight),
            consistency_weight=float(args.consistency_weight),
            aux_t_gate=int(args.aux_t_gate),
            epoch_end_callback=_maybe_eval_and_save_structured,
        )

        if bool(args.save_final_model):
            ckpt = out_dir / "checkpoints" / "external_c2f_full_earn_teacher" / fold_label / "final.pt"
            model.save(ckpt)
            final_str = str(ckpt)
            if final_str not in saved_checkpoints:
                saved_checkpoints.append(final_str)

        ablation_summary = _evaluate_teacher_forced_stage2_structured(
            model=model,
            group_idx=group_idx,
            parent_idx=parent_idx,
            coarse_by_group=coarse_by_group,
            child_mask_by_parent=child_mask_by_parent,
            p_joint=p_joint,
            child_count=child_count,
            eval_idx=test_idx,
            n_eval_joint_samples=int(args.n_eval_joint_samples),
            device=args.device,
            x_mean=x_mean,
            x_std=x_std,
            coarse_k=coarse_k,
            target_dim=target_dim,
        )
        final_projected_tvd = float(ablation_summary["teacher_forced_stage2"]["tvd_joint_projected"]["mean"])
        if bool(args.save_best_model) and best_epoch is None:
            best_projected_tvd = final_projected_tvd
            best_epoch = int(args.epochs)
            best_ckpt_path = (
                out_dir
                / "checkpoints"
                / "external_c2f_full_earn_teacher"
                / fold_label
                / "best.pt"
            )
            model.save(best_ckpt_path)
            best_str = str(best_ckpt_path)
            if best_str not in saved_checkpoints:
                saved_checkpoints.append(best_str)

        active_count_all = child_mask_by_parent[parent_idx].sum(axis=1, dtype=np.float64)
        inactive_count_all = float(target_dim) - active_count_all
        child_mask_stats = {
            "mean_active_slots": float(np.mean(active_count_all)),
            "mean_inactive_slots": float(np.mean(inactive_count_all)),
            "mean_inactive_frac": float(np.mean(inactive_count_all / max(float(target_dim), 1.0))),
        }
        run_summary = {
            "created_utc": _utc_now_iso(),
            "wide_csv": str(in_path),
            "schema_json": str(schema_path),
            "structured_low_memory": True,
            "read_chunksize": int(args.read_chunksize),
            "structured_group_count": int(structured["group_count"]),
            "coarse_preset": str(schema.get("coarse_preset", "")),
            "coarse_shape": list(schema.get("coarse_shape", [])),
            "coarse_K": int(schema.get("coarse_K", 0) or 0),
            "n_rows_total": int(structured["n_rows_total"]),
            "n_heldout_rows": int(test_idx.size),
            "n_train_rows": int(train_idx.size),
            "heldout_statefp": str(heldout_statefp),
            "target_dim": int(target_dim),
            "cond_dim": int(cond_dim),
            "eval_mode": str(args.eval_mode),
            "fold_label": str(fold_label),
            "timesteps": int(args.timesteps),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "hidden_dims": list(hidden_dims),
            "latent_dim": int(args.latent_dim),
            "encoder_hidden_dims": list(encoder_hidden_dims),
            "head_hidden_dims": list(head_hidden_dims),
            "condition_injection": str(args.condition_injection),
            "film_hidden_dim": int(args.film_hidden_dim),
            "n_eval_joint_samples": int(args.n_eval_joint_samples),
            "diff_loss_reweight_alpha": float(args.diff_loss_reweight_alpha),
            "diff_loss_reweight_floor": float(args.diff_loss_reweight_floor),
            "diff_loss_reweight_cap": float(args.diff_loss_reweight_cap),
            "diff_loss_reweight_basis": "child_mask_uniform_support" if float(args.diff_loss_reweight_alpha) > 0.0 else "none",
            "use_clean_head": bool(use_clean_head),
            "clean_head_weight": float(args.clean_head_weight),
            "consistency_weight": float(args.consistency_weight),
            "income_regime_weight": 0.0,
            "income_regime_consistency_weight": 0.0,
            "use_income_regime_head": False,
            "use_income_regime_structure": False,
            "regime_projection_alpha": 0.0,
            "aux_t_gate": int(args.aux_t_gate),
            "predict_mode": str(args.predict_mode),
            "blend_alpha": float(args.blend_alpha),
            "seed": int(args.seed),
            "device": args.device,
            "save_final_model": bool(args.save_final_model),
            "save_best_model": bool(args.save_best_model),
            "eval_every_epochs": int(args.eval_every_epochs),
            "fit_summary": train_fit_summary,
            "best_projected_tvd_joint": None if best_epoch is None else float(best_projected_tvd),
            "best_epoch": None if best_epoch is None else int(best_epoch),
            "best_checkpoint": None if best_ckpt_path is None else str(best_ckpt_path),
            "final_projected_tvd_joint": float(final_projected_tvd),
            "saved_checkpoints": saved_checkpoints,
            "results": ablation_summary,
            "parent_idx_range": [int(np.min(parent_idx)), int(np.max(parent_idx))],
            "child_count_range": [int(np.min(child_count)), int(np.max(child_count))],
            "child_mask_stats": child_mask_stats,
        }
        _write_json(out_dir / "run_summary.json", run_summary)
        _write_json(out_dir / "metrics" / "ablation_summary.json", ablation_summary)
        print(f"[ok] wrote: {out_dir}", file=sys.stderr)
        return

    df = pd.read_csv(in_path, low_memory=False)
    req = {"statefp", "puma_uid", "parent_idx", "child_count", "parent_mass"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"wide_csv missing columns: {miss}")
    for cols in [cond_cols, p_joint_cols]:
        miss_cols = [c for c in cols if c not in df.columns]
        if miss_cols:
            raise SystemExit(f"wide_csv missing columns: {miss_cols[:5]}")
    if (
        float(args.income_regime_weight) > 0.0
        or float(args.income_regime_consistency_weight) > 0.0
        or float(args.regime_projection_alpha) > 0.0
    ):
        if not income_regime_target_cols or not income_regime_mask_cols:
            raise SystemExit(
                "income regime options require income_regime_target/mask auxiliary blocks in schema."
            )
        for cols in [income_regime_target_cols, income_regime_mask_cols]:
            miss_cols = [c for c in cols if c not in df.columns]
            if miss_cols:
                raise SystemExit(f"wide_csv missing auxiliary columns: {miss_cols[:5]}")
    if float(args.income_regime_consistency_weight) > 0.0:
        if parent_child_income_regime_padded is None:
            raise SystemExit("income_regime_consistency_weight > 0 requires parent_child_income_regime_padded in schema.")

    df["statefp"] = df["statefp"].map(_canon_statefp)
    heldout_statefp = _canon_statefp(args.heldout_statefp)
    is_heldout = df["statefp"] == heldout_statefp
    if int(is_heldout.sum()) == 0:
        raise SystemExit(f"No held-out rows found (statefp=={heldout_statefp}).")
    fold_label = "leave_mi_out" if heldout_statefp == "26" else f"leave_state_{heldout_statefp}_out"

    p_joint = df[p_joint_cols].to_numpy(dtype=np.float32)
    p_joint = np.clip(p_joint, 0.0, None)
    p_joint = p_joint / np.maximum(p_joint.sum(axis=1, keepdims=True), 1e-12)
    child_mask = df[child_mask_cols].to_numpy(dtype=np.float32)
    cond = df[cond_cols].to_numpy(dtype=np.float32)
    income_regime_target = (
        df[income_regime_target_cols].to_numpy(dtype=np.float32) if income_regime_target_cols else None
    )
    income_regime_mask = (
        df[income_regime_mask_cols].to_numpy(dtype=np.float32) if income_regime_mask_cols else None
    )
    child_count = pd.to_numeric(df["child_count"], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
    parent_idx = pd.to_numeric(df["parent_idx"], errors="coerce").fillna(-1).astype(int).to_numpy(dtype=int)
    income_regime_slot = None
    if parent_child_income_regime_padded is not None:
        padded_arr = np.asarray(parent_child_income_regime_padded, dtype=np.int64)
        income_regime_slot = padded_arr[parent_idx]

    x_log_all = np.log(np.clip(p_joint, 0.0, None) + 1e-6).astype(np.float32)
    train_idx = np.where(~is_heldout.to_numpy(dtype=bool))[0]
    test_idx = np.where(is_heldout.to_numpy(dtype=bool))[0]
    if train_idx.size == 0 or test_idx.size == 0:
        raise SystemExit("Invalid leave-state-out split.")

    x_train_log = x_log_all[train_idx]
    x_mean = x_train_log.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = x_train_log.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
    x_train = ((x_train_log - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)).astype(np.float32)

    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    encoder_hidden_dims = _parse_hidden_dims(args.encoder_hidden_dims)
    head_hidden_dims = _parse_hidden_dims(args.head_hidden_dims)
    use_clean_head = (
        float(args.clean_head_weight) > 0.0
        or float(args.consistency_weight) > 0.0
        or str(args.predict_mode) != "diffusion"
    )
    use_income_regime_structure = float(args.income_regime_consistency_weight) > 0.0
    use_income_regime_head = (
        income_regime_target is not None
        and income_regime_mask is not None
        and (
            float(args.income_regime_weight) > 0.0
            or float(args.regime_projection_alpha) > 0.0
        )
    )
    model_cfg = TabDDPMConfig(
        timesteps=int(args.timesteps),
        hidden_dims=hidden_dims,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        condition_injection=str(args.condition_injection),
        film_hidden_dim=int(args.film_hidden_dim),
    )
    if use_clean_head or use_income_regime_head or use_income_regime_structure:
        model: Any = SharedConditionStage2Diffusion(
            input_dim=target_dim,
            cond_raw_dim=int(cond.shape[1]),
            latent_dim=int(args.latent_dim),
            encoder_hidden_dims=encoder_hidden_dims,
            head_hidden_dims=head_hidden_dims,
            diffusion_config=model_cfg,
            seed=int(args.seed),
            enable_clean_head=bool(use_clean_head),
            aux_regime_dim=0 if income_regime_target is None else int(income_regime_target.shape[1]),
            predict_mode=str(args.predict_mode),
            blend_alpha=float(args.blend_alpha),
            regime_projection_alpha=float(args.regime_projection_alpha),
        )
    else:
        model = DiffusionTabularModel(
            input_dim=target_dim,
            cond_dim=int(cond.shape[1]),
            seed=int(args.seed),
            config=model_cfg,
        )
    saved_checkpoints: list[str] = []
    best_projected_tvd = float("inf")
    best_epoch: int | None = None
    best_ckpt_path: pathlib.Path | None = None
    eval_every_epochs = int(args.eval_every_epochs)

    def _maybe_eval_and_save(epoch_idx: int) -> None:
        nonlocal best_projected_tvd, best_epoch, best_ckpt_path, saved_checkpoints
        if eval_every_epochs <= 0 or epoch_idx % eval_every_epochs != 0:
            return
        ablation = _evaluate_teacher_forced_stage2(
            model=model,
            cond=cond,
            p_joint=p_joint,
            child_mask=child_mask,
            child_count=child_count,
            eval_idx=test_idx,
            n_eval_joint_samples=int(args.n_eval_joint_samples),
            device=args.device,
            x_mean=x_mean,
            x_std=x_std,
        )
        projected = float(ablation["teacher_forced_stage2"]["tvd_joint_projected"]["mean"])
        print(
            f"[eval] epoch={epoch_idx} "
            f"tvd_joint_projected={projected:.6f}",
            file=sys.stderr,
        )
        if bool(args.save_best_model) and projected < best_projected_tvd:
            best_projected_tvd = projected
            best_epoch = int(epoch_idx)
            best_ckpt_path = (
                out_dir
                / "checkpoints"
                / "external_c2f_full_earn_teacher"
                / fold_label
                / "best.pt"
            )
            model.save(best_ckpt_path)
            best_str = str(best_ckpt_path)
            if best_str not in saved_checkpoints:
                saved_checkpoints.append(best_str)
            print(
                f"[eval] epoch={epoch_idx} new_best_tvd_joint_projected={projected:.6f}",
                file=sys.stderr,
            )

    if use_clean_head or use_income_regime_head or use_income_regime_structure:
        train_fit_summary = _fit_shared_stage2_diffusion(
            model=model,
            x=torch.from_numpy(x_train),
            p_true=torch.from_numpy(p_joint[train_idx]),
            cond=torch.from_numpy(cond[train_idx]),
            child_mask=torch.from_numpy(child_mask[train_idx]),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            device=args.device,
            log_every=int(args.log_every),
            x_mean=torch.from_numpy(x_mean.reshape(1, -1)),
            x_std=torch.from_numpy(x_std.reshape(1, -1)),
            diff_loss_reweight_alpha=float(args.diff_loss_reweight_alpha),
            diff_loss_reweight_floor=float(args.diff_loss_reweight_floor),
            diff_loss_reweight_cap=float(args.diff_loss_reweight_cap),
            clean_head_weight=float(args.clean_head_weight),
            consistency_weight=float(args.consistency_weight),
            income_regime_target=None if income_regime_target is None else torch.from_numpy(income_regime_target[train_idx]),
            income_regime_mask=None if income_regime_mask is None else torch.from_numpy(income_regime_mask[train_idx]),
            income_regime_slot=None if income_regime_slot is None else torch.from_numpy(income_regime_slot[train_idx]),
            income_regime_weight=float(args.income_regime_weight),
            income_regime_consistency_weight=float(args.income_regime_consistency_weight),
            aux_t_gate=int(args.aux_t_gate),
            epoch_end_callback=_maybe_eval_and_save,
        )
    elif float(args.diff_loss_reweight_alpha) > 0.0:
        train_fit_summary = _fit_weighted_stage2_diffusion(
            model=model,
            x=torch.from_numpy(x_train),
            cond=torch.from_numpy(cond[train_idx]),
            child_mask=torch.from_numpy(child_mask[train_idx]),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            device=args.device,
            log_every=int(args.log_every),
            diff_loss_reweight_alpha=float(args.diff_loss_reweight_alpha),
            diff_loss_reweight_floor=float(args.diff_loss_reweight_floor),
            diff_loss_reweight_cap=float(args.diff_loss_reweight_cap),
            epoch_end_callback=_maybe_eval_and_save,
        )
    else:
        train_fit_summary = model.fit(
            x=torch.from_numpy(x_train),
            cond=torch.from_numpy(cond[train_idx]),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            device=args.device,
            log_every=int(args.log_every),
        )

    if bool(args.save_final_model):
        ckpt = out_dir / "checkpoints" / "external_c2f_full_earn_teacher" / fold_label / "final.pt"
        model.save(ckpt)
        final_str = str(ckpt)
        if final_str not in saved_checkpoints:
            saved_checkpoints.append(final_str)

    ablation_summary = _evaluate_teacher_forced_stage2(
        model=model,
        cond=cond,
        p_joint=p_joint,
        child_mask=child_mask,
        child_count=child_count,
        eval_idx=test_idx,
        n_eval_joint_samples=int(args.n_eval_joint_samples),
        device=args.device,
        x_mean=x_mean,
        x_std=x_std,
    )
    final_projected_tvd = float(ablation_summary["teacher_forced_stage2"]["tvd_joint_projected"]["mean"])
    if bool(args.save_best_model) and best_epoch is None:
        best_projected_tvd = final_projected_tvd
        best_epoch = int(args.epochs)
        best_ckpt_path = (
            out_dir
            / "checkpoints"
            / "external_c2f_full_earn_teacher"
            / fold_label
            / "best.pt"
        )
        model.save(best_ckpt_path)
        best_str = str(best_ckpt_path)
        if best_str not in saved_checkpoints:
            saved_checkpoints.append(best_str)

    active_count_all = child_mask.sum(axis=1, dtype=np.float64)
    inactive_count_all = float(target_dim) - active_count_all
    child_mask_stats = {
        "mean_active_slots": float(np.mean(active_count_all)),
        "mean_inactive_slots": float(np.mean(inactive_count_all)),
        "mean_inactive_frac": float(np.mean(inactive_count_all / max(float(target_dim), 1.0))),
    }

    run_summary = {
        "created_utc": _utc_now_iso(),
        "wide_csv": str(in_path),
        "schema_json": str(schema_path),
        "coarse_preset": str(schema.get("coarse_preset", "")),
        "coarse_shape": list(schema.get("coarse_shape", [])),
        "coarse_K": int(schema.get("coarse_K", 0) or 0),
        "n_rows_total": int(df.shape[0]),
        "n_heldout_rows": int(is_heldout.sum()),
        "n_train_rows": int((~is_heldout).sum()),
        "heldout_statefp": str(heldout_statefp),
        "target_dim": int(target_dim),
        "cond_dim": int(cond.shape[1]),
        "eval_mode": str(args.eval_mode),
        "fold_label": str(fold_label),
        "timesteps": int(args.timesteps),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "hidden_dims": list(hidden_dims),
        "latent_dim": int(args.latent_dim),
        "encoder_hidden_dims": list(encoder_hidden_dims),
        "head_hidden_dims": list(head_hidden_dims),
        "condition_injection": str(args.condition_injection),
        "film_hidden_dim": int(args.film_hidden_dim),
        "n_eval_joint_samples": int(args.n_eval_joint_samples),
        "diff_loss_reweight_alpha": float(args.diff_loss_reweight_alpha),
        "diff_loss_reweight_floor": float(args.diff_loss_reweight_floor),
        "diff_loss_reweight_cap": float(args.diff_loss_reweight_cap),
        "diff_loss_reweight_basis": "child_mask_uniform_support" if float(args.diff_loss_reweight_alpha) > 0.0 else "none",
        "use_clean_head": bool(use_clean_head),
        "clean_head_weight": float(args.clean_head_weight),
        "consistency_weight": float(args.consistency_weight),
        "income_regime_weight": float(args.income_regime_weight),
        "income_regime_consistency_weight": float(args.income_regime_consistency_weight),
        "use_income_regime_head": bool(use_income_regime_head),
        "use_income_regime_structure": bool(use_income_regime_structure),
        "regime_projection_alpha": float(args.regime_projection_alpha),
        "aux_t_gate": int(args.aux_t_gate),
        "predict_mode": str(args.predict_mode),
        "blend_alpha": float(args.blend_alpha),
        "seed": int(args.seed),
        "device": args.device,
        "save_final_model": bool(args.save_final_model),
        "save_best_model": bool(args.save_best_model),
        "eval_every_epochs": int(args.eval_every_epochs),
        "fit_summary": train_fit_summary,
        "best_projected_tvd_joint": None if best_epoch is None else float(best_projected_tvd),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_checkpoint": None if best_ckpt_path is None else str(best_ckpt_path),
        "final_projected_tvd_joint": float(final_projected_tvd),
        "saved_checkpoints": saved_checkpoints,
        "results": ablation_summary,
        "parent_idx_range": [int(np.min(parent_idx)), int(np.max(parent_idx))],
        "child_count_range": [int(np.min(child_count)), int(np.max(child_count))],
        "child_mask_stats": child_mask_stats,
    }

    _write_json(out_dir / "run_summary.json", run_summary)
    _write_json(out_dir / "metrics" / "ablation_summary.json", ablation_summary)
    print(f"[ok] wrote: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
