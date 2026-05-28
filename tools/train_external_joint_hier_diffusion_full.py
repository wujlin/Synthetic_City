#!/usr/bin/env python3
from __future__ import annotations

"""
Train a shared-latent hierarchical diffusion model on the full external v1 schema.

Fine target:
  AGEP_fine(10) x SEX(2) x SCHL_fine(5) x ESR_fine(5) = K=500

Coarse auxiliary target:
  AGEP_lite(4) x SEX(2) x SCHL_lite(3) x ESR_lite(3) = K=72

Core idea:
- Encode the raw external condition into a shared regional latent z.
- Use z as the diffusion condition for the full 4-attribute joint.
- Predict a coarse joint from z with an auxiliary head.
- Encourage consistency between the sampled fine structure and the coarse head.

This keeps diffusion as the main generator while importing the shared
regional-context insight discovered by the diagnostic hierarchical probes.
"""

import argparse
import datetime as _dt
import json
import pathlib
import random
import sys
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.model.diffusion_tabular import (
    TabDDPMConfig,
    _DenoiserMLP,
    _FiLMDenoiserMLP,
    _require_torch,
)
from tools.build_external_target_v1_michigan import AGE_LABELS, ESR_LABELS, SCHL_LABELS, SEX_LABELS
from tools.external_v1_variant_presets import AGE_LITE_LABELS, AGE_TO_LITE, ESR_LITE_LABELS, ESR_TO_LITE, SCHL_LITE_LABELS, SCHL_TO_LITE
from tools.train_us_puma_5var_diffusion import (
    _canon_puma5,
    _canon_statefp,
    _canon_uid,
    _cosine,
    _ipf_nd,
    _marginal_from_joint,
    _nd_independence,
    _parse_hidden_dims,
    _softmax_rows,
    _stable_hash_fold,
    _summ,
    _tvd,
    _write_json,
)
from tools.train_us_puma_external_v1_diffusion import (
    _append_condition_extra_matrix,
    _load_condition_specs_from_schema,
    _load_external_condition_matrix,
    _load_var_specs_from_schema,
)


FINE_VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop"]
COARSE_VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop"]
FINE_CATEGORIES = {
    "AGEP_bin": AGE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LABELS,
    "ESR_allpop": ESR_LABELS,
}
COARSE_CATEGORIES = {
    "AGEP_bin": AGE_LITE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LITE_LABELS,
    "ESR_allpop": ESR_LITE_LABELS,
}
FINE_SHAPE = tuple(len(FINE_CATEGORIES[v]) for v in FINE_VARIABLE_ORDER)
COARSE_SHAPE = tuple(len(COARSE_CATEGORIES[v]) for v in COARSE_VARIABLE_ORDER)
FINE_K = int(np.prod(FINE_SHAPE))
COARSE_K = int(np.prod(COARSE_SHAPE))


def _build_fine_to_coarse_matrix_full() -> np.ndarray:
    age_coarse_idx = {lab: i for i, lab in enumerate(AGE_LITE_LABELS)}
    schl_coarse_idx = {lab: i for i, lab in enumerate(SCHL_LITE_LABELS)}
    esr_coarse_idx = {lab: i for i, lab in enumerate(ESR_LITE_LABELS)}
    out = np.zeros((FINE_K, COARSE_K), dtype=np.float32)
    k = 0
    for age_lab in FINE_CATEGORIES["AGEP_bin"]:
        ac = age_coarse_idx[AGE_TO_LITE[age_lab]]
        for si, _ in enumerate(FINE_CATEGORIES["SEX"]):
            for schl_lab in FINE_CATEGORIES["SCHL_allpop"]:
                qc = schl_coarse_idx[SCHL_TO_LITE[schl_lab]]
                for esr_lab in FINE_CATEGORIES["ESR_allpop"]:
                    ec = esr_coarse_idx[ESR_TO_LITE[esr_lab]]
                    kc = np.ravel_multi_index((ac, si, qc, ec), COARSE_SHAPE)
                    out[k, kc] = 1.0
                    k += 1
    return out


def _aggregate_fine_to_coarse_np(p_fine: np.ndarray, agg_mat: np.ndarray) -> np.ndarray:
    p = np.asarray(p_fine, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    return p @ np.asarray(agg_mat, dtype=np.float64)


def _augment_ext_marginals_from_cross(
    *,
    cond_raw: np.ndarray,
    block_slices: dict[str, slice],
    ext_marg: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    out = dict(ext_marg)
    cross_sl = block_slices.get("AGEP_SEX_cross")
    if cross_sl is None:
        return out

    n_age = len(FINE_CATEGORIES["AGEP_bin"])
    n_sex = len(FINE_CATEGORIES["SEX"])
    cross = np.asarray(cond_raw[:, cross_sl], dtype=np.float64)
    if cross.shape[1] != int(n_age * n_sex):
        raise SystemExit(
            "AGEP_SEX_cross block has unexpected width: "
            f"got {cross.shape[1]}, expected {n_age * n_sex}"
        )
    cross = cross.reshape((-1, n_age, n_sex))
    if "AGEP_bin" not in out:
        age = cross.sum(axis=2)
        age = age / np.maximum(age.sum(axis=1, keepdims=True), 1e-12)
        out["AGEP_bin"] = age.astype(np.float32)
    if "SEX" not in out:
        sex = cross.sum(axis=1)
        sex = sex / np.maximum(sex.sum(axis=1, keepdims=True), 1e-12)
        out["SEX"] = sex.astype(np.float32)
    return out


def _marginals_from_joint_torch(*, p_joint: Any, shape: tuple[int, ...]) -> list[Any]:
    torch = _require_torch()
    ndim = len(shape)
    p_nd = p_joint.reshape((p_joint.shape[0],) + tuple(int(x) for x in shape))
    out: list[Any] = []
    for axis in range(ndim):
        reduce_dims = tuple(i + 1 for i in range(ndim) if i != axis)
        out.append(torch.sum(p_nd, dim=reduce_dims))
    return out


def _expand_active_prob_np(*, p_active: np.ndarray, active_cols: np.ndarray, full_dim: int) -> np.ndarray:
    out = np.zeros((p_active.shape[0], int(full_dim)), dtype=p_active.dtype)
    out[:, np.asarray(active_cols, dtype=np.int64)] = p_active
    return out


def _utc_now_iso_local() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class _ModuleEMA:
    def __init__(self, modules: Any, *, decay: float) -> None:
        torch = _require_torch()
        self.decay = float(decay)
        self.enabled = 0.0 < self.decay < 1.0
        self._shadow: dict[str, Any] | None = None
        if self.enabled:
            self._shadow = {k: v.detach().clone() for k, v in modules.state_dict().items()}
            for v in self._shadow.values():
                if torch.is_floating_point(v):
                    v.requires_grad_(False)

    def update(self, modules: Any) -> None:
        torch = _require_torch()
        if not self.enabled:
            return
        assert self._shadow is not None
        for key, value in modules.state_dict().items():
            shadow = self._shadow[key]
            if torch.is_floating_point(value):
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value.detach())

    def cpu_state_dict(self) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("EMA is disabled")
        assert self._shadow is not None
        return {k: v.detach().cpu().clone() for k, v in self._shadow.items()}

    @contextmanager
    def apply(self, modules: Any) -> Any:
        if not self.enabled:
            yield
            return
        backup = {k: v.detach().clone() for k, v in modules.state_dict().items()}
        assert self._shadow is not None
        modules.load_state_dict(self._shadow, strict=True)
        try:
            yield
        finally:
            modules.load_state_dict(backup, strict=True)


def _split_train_val_indices(
    *,
    train_idx: np.ndarray,
    ids: list[str],
    seed: int,
    fold_name: str,
    val_frac: float,
    val_min_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(train_idx, dtype=int)
    if train_idx.size <= 1 or float(val_frac) <= 0.0:
        return train_idx, np.asarray([], dtype=int)
    val_size = int(round(float(val_frac) * int(train_idx.size)))
    val_size = max(int(val_min_groups), val_size)
    val_size = min(val_size, int(train_idx.size) - 1)
    if val_size <= 0:
        return train_idx, np.asarray([], dtype=int)
    fold_seed = int(seed) + sum(ord(c) for c in str(fold_name))
    rng = np.random.default_rng(fold_seed)
    order = rng.permutation(train_idx.size)
    val_idx = np.sort(train_idx[order[:val_size]])
    train_core_idx = np.sort(train_idx[order[val_size:]])
    return train_core_idx, val_idx


def _evaluate_joint_distribution(
    *,
    model: "SharedLatentHierarchicalDiffusion",
    eval_idx: np.ndarray,
    reference_train_idx: np.ndarray,
    p_fine: np.ndarray,
    p_coarse: np.ndarray,
    cond_raw: np.ndarray,
    ext_marg: dict[str, np.ndarray],
    var_specs: list[tuple[str, int, str]],
    device: Any,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    agg_mat_np: np.ndarray,
    n_eval_joint_samples: int,
    ipf_iters: int,
    logp_clip_lo: np.ndarray | None,
    logp_clip_hi: np.ndarray | None,
    active_cols: np.ndarray | None,
    full_fine_dim: int,
) -> dict[str, Any]:
    torch = _require_torch()
    eval_idx = np.asarray(eval_idx, dtype=int)
    ref_idx = np.asarray(reference_train_idx, dtype=int)
    if eval_idx.size == 0:
        raise ValueError("eval_idx must be non-empty")
    cond_eval_t = torch.from_numpy(cond_raw[eval_idx]).to(device)
    coarse_prob_t = model.predict_coarse(cond_raw=cond_eval_t)
    with torch.inference_mode():
        z_eval = model.encoder(cond_eval_t)
    x_samples = model.sample_latent_conditioned(
        z_cond=z_eval,
        n_draws=int(n_eval_joint_samples),
        device=device,
    ).numpy()

    logp = x_samples.astype(np.float64) * x_std.reshape(1, 1, -1).astype(np.float64) + x_mean.reshape(1, 1, -1).astype(np.float64)
    if logp_clip_lo is not None and logp_clip_hi is not None:
        lo = logp_clip_lo.reshape(1, 1, -1).astype(np.float64)
        hi = logp_clip_hi.reshape(1, 1, -1).astype(np.float64)
        logp = np.clip(logp, lo, hi)
    p_draws = np.asarray([_softmax_rows(logp[i]) for i in range(logp.shape[0])], dtype=np.float64)
    fine_pred_raw = p_draws.mean(axis=1)
    if active_cols is not None:
        fine_pred_raw = _expand_active_prob_np(
            p_active=fine_pred_raw,
            active_cols=np.asarray(active_cols, dtype=np.int64),
            full_dim=int(full_fine_dim),
        )
    fine_pred_raw = fine_pred_raw / np.maximum(fine_pred_raw.sum(axis=1, keepdims=True), 1e-12)

    fine_pred = fine_pred_raw.copy()
    for j, idx in enumerate(eval_idx):
        marginals_ext = [ext_marg[var][idx] for var, _, _ in var_specs]
        fine_pred[j] = _ipf_nd(
            seed_joint=fine_pred_raw[j].reshape(FINE_SHAPE),
            target_marginals=[np.asarray(m, dtype=float) for m in marginals_ext],
            shape=FINE_SHAPE,
            max_iter=int(ipf_iters),
        ).reshape(-1)
        fine_pred[j] = fine_pred[j] / max(float(fine_pred[j].sum()), 1e-12)

    coarse_pred = coarse_prob_t.detach().cpu().numpy().astype(np.float64)
    coarse_from_fine = _aggregate_fine_to_coarse_np(fine_pred, agg_mat_np)
    coarse_from_fine = coarse_from_fine / np.maximum(coarse_from_fine.sum(axis=1, keepdims=True), 1e-12)

    tvd_fine_raw: list[float] = []
    tvd_fine: list[float] = []
    cosine_fine_raw: list[float] = []
    cosine_fine: list[float] = []
    tvd_coarse_head: list[float] = []
    tvd_coarse_from_fine: list[float] = []
    var_eval: dict[str, list[float]] = {var: [] for var, _, _ in var_specs}
    var_raw: dict[str, list[float]] = {var: [] for var, _, _ in var_specs}
    for j, idx in enumerate(eval_idx):
        p_true_f = p_fine[idx]
        p_true_c = p_coarse[idx]
        tvd_fine_raw.append(_tvd(fine_pred_raw[j], p_true_f))
        tvd_fine.append(_tvd(fine_pred[j], p_true_f))
        cosine_fine_raw.append(_cosine(fine_pred_raw[j], p_true_f))
        cosine_fine.append(_cosine(fine_pred[j], p_true_f))
        tvd_coarse_head.append(_tvd(coarse_pred[j], p_true_c))
        tvd_coarse_from_fine.append(_tvd(coarse_from_fine[j], p_true_c))
        for axis, (var, _, _) in enumerate(var_specs):
            mr = _marginal_from_joint(fine_pred_raw[j], shape=FINE_SHAPE, axis=axis)
            me = _marginal_from_joint(fine_pred[j], shape=FINE_SHAPE, axis=axis)
            mt = np.asarray(ext_marg[var][idx], dtype=float)
            var_raw[var].append(_tvd(mr, mt))
            var_eval[var].append(_tvd(me, mt))

    train_seed = np.asarray(p_fine[ref_idx], dtype=np.float64).mean(axis=0)
    train_seed = train_seed / max(float(train_seed.sum()), 1e-12)
    tvd_ipf: list[float] = []
    tvd_ind: list[float] = []
    for idx in eval_idx:
        marginals_ext = [ext_marg[var][idx] for var, _, _ in var_specs]
        p_ipf = _ipf_nd(
            seed_joint=train_seed.reshape(FINE_SHAPE),
            target_marginals=[np.asarray(m, dtype=float) for m in marginals_ext],
            shape=FINE_SHAPE,
            max_iter=int(ipf_iters),
        ).reshape(-1)
        p_ipf = p_ipf / max(float(p_ipf.sum()), 1e-12)
        tvd_ipf.append(_tvd(p_ipf, p_fine[idx]))
        p_ind = _nd_independence(marginals_ext)
        tvd_ind.append(_tvd(p_ind, p_fine[idx]))

    summary: dict[str, Any] = {
        "n_eval": int(eval_idx.size),
        "n_reference_train": int(ref_idx.size),
        "tvd_joint_raw": _summ(tvd_fine_raw),
        "tvd_joint": _summ(tvd_fine),
        "cosine_joint_raw": _summ(cosine_fine_raw),
        "cosine_joint": _summ(cosine_fine),
        "tvd_coarse_head": _summ(tvd_coarse_head),
        "tvd_coarse_from_fine": _summ(tvd_coarse_from_fine),
        "ipf_train_seed_external": {"tvd_joint": _summ(tvd_ipf)},
        "independence_external": {"tvd_joint": _summ(tvd_ind)},
    }
    for var, _, _ in var_specs:
        summary[f"tvd_{var}"] = _summ(var_eval[var])
        summary[f"tvd_{var}_raw"] = _summ(var_raw[var])
    return summary


class SharedLatentHierarchicalDiffusion:
    def __init__(
        self,
        *,
        input_dim: int,
        cond_raw_dim: int,
        latent_dim: int,
        encoder_hidden_dims: tuple[int, ...],
        coarse_hidden_dims: tuple[int, ...],
        diffusion_config: TabDDPMConfig,
        seed: int,
    ) -> None:
        torch = _require_torch()
        nn = torch.nn
        torch.manual_seed(int(seed))

        self.input_dim = int(input_dim)
        self.cond_raw_dim = int(cond_raw_dim)
        self.latent_dim = int(latent_dim)
        self.seed = int(seed)
        self.config = diffusion_config
        self._schedule: dict[str, Any] | None = None

        self.encoder = self._make_mlp(
            in_dim=self.cond_raw_dim,
            hidden_dims=encoder_hidden_dims,
            out_dim=self.latent_dim,
            nn=nn,
        )
        self.coarse_feature = self._make_mlp(
            in_dim=self.latent_dim,
            hidden_dims=coarse_hidden_dims,
            out_dim=self.latent_dim,
            nn=nn,
        )
        self.coarse_out = nn.Linear(self.latent_dim, COARSE_K)

        mode = str(self.config.condition_injection).lower().strip()
        if mode == "concat":
            self.denoiser = _DenoiserMLP(
                input_dim=self.input_dim,
                cond_dim=self.latent_dim,
                hidden_dims=self.config.hidden_dims,
                time_embed_dim=self.config.time_embed_dim,
            )
        elif mode == "film":
            self.denoiser = _FiLMDenoiserMLP(
                input_dim=self.input_dim,
                cond_dim=self.latent_dim,
                hidden_dims=self.config.hidden_dims,
                time_embed_dim=self.config.time_embed_dim,
                film_hidden_dim=int(getattr(self.config, "film_hidden_dim", 128)),
            )
        else:
            raise ValueError("condition_injection must be one of: concat, film")

        self._modules = nn.ModuleList([self.encoder, self.coarse_feature, self.coarse_out, self.denoiser])
        self._opt = torch.optim.AdamW(self._modules.parameters(), lr=float(self.config.lr), weight_decay=float(self.config.weight_decay))

    @staticmethod
    def _make_mlp(*, in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, nn: Any) -> Any:
        layers: list[Any] = []
        dim_in = int(in_dim)
        for dim_out in hidden_dims:
            layers.append(nn.Linear(dim_in, int(dim_out)))
            layers.append(nn.SiLU())
            dim_in = int(dim_out)
        layers.append(nn.Linear(dim_in, int(out_dim)))
        return nn.Sequential(*layers)

    def _init_schedule(self, *, device: Any) -> None:
        torch = _require_torch()
        if self._schedule is not None:
            return
        betas = torch.linspace(self.config.beta_start, self.config.beta_end, self.config.timesteps, device=device)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, device=device), alpha_cumprod[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self._schedule = {
            "betas": betas,
            "alphas": alphas,
            "alpha_cumprod": alpha_cumprod,
            "sqrt_alpha_cumprod": torch.sqrt(alpha_cumprod),
            "sqrt_one_minus_alpha_cumprod": torch.sqrt(1.0 - alpha_cumprod),
            "posterior_variance": posterior_variance,
        }

    def to(self, device: Any) -> None:
        self._modules.to(device)
        self._init_schedule(device=device)

    def train(self) -> None:
        self._modules.train()

    def eval(self) -> None:
        self._modules.eval()

    def step(
        self,
        *,
        x0: Any,
        cond_raw: Any,
        p_coarse_true: Any,
        agg_mat: Any,
        marginal_targets: tuple[Any, ...],
        x_mean: Any,
        x_std: Any,
        logp_clip_lo: Any | None,
        logp_clip_hi: Any | None,
        agg_mat_full: Any,
        full_fine_dim: int,
        active_cols_t: Any | None,
        coarse_weight: float,
        consistency_weight: float,
        marginal_weight: float,
        aux_t_gate: int,
        detach_coarse_encoder: bool,
        diff_loss_reweight_alpha: float,
        diff_loss_reweight_floor: float,
        diff_loss_reweight_cap: float,
    ) -> dict[str, float]:
        torch = _require_torch()
        assert self._schedule is not None
        self.train()

        z = self.encoder(cond_raw)
        z_for_coarse = z.detach() if bool(detach_coarse_encoder) else z
        coarse_feat = self.coarse_feature(z_for_coarse)
        coarse_logits = self.coarse_out(coarse_feat)
        coarse_logp = torch.log_softmax(coarse_logits, dim=1)
        coarse_prob = torch.softmax(coarse_logits, dim=1)

        t = torch.randint(0, self.config.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        sqrt_acp = self._schedule["sqrt_alpha_cumprod"][t].unsqueeze(1)
        sqrt_om = self._schedule["sqrt_one_minus_alpha_cumprod"][t].unsqueeze(1)
        x_t = sqrt_acp * x0 + sqrt_om * noise
        eps_pred = self.denoiser(x_t, t, z)

        if float(diff_loss_reweight_alpha) > 0.0:
            with torch.no_grad():
                logp_true = x0 * x_std + x_mean
                p_true = torch.softmax(logp_true, dim=1)
                cell_weight = torch.pow(torch.clamp(p_true, min=1e-8), float(diff_loss_reweight_alpha))
                cell_weight = cell_weight / torch.clamp(cell_weight.mean(dim=1, keepdim=True), min=1e-8)
                cell_weight = torch.clamp(
                    cell_weight,
                    min=float(diff_loss_reweight_floor),
                    max=float(diff_loss_reweight_cap),
                )
            loss_diff = (cell_weight * torch.square(eps_pred - noise)).mean()
        else:
            loss_diff = torch.nn.functional.mse_loss(eps_pred, noise)

        x0_pred = (x_t - sqrt_om * eps_pred) / torch.clamp(sqrt_acp, min=1e-12)
        logp_pred = x0_pred * x_std + x_mean
        if logp_clip_lo is not None and logp_clip_hi is not None:
            logp_pred = torch.clamp(logp_pred, min=logp_clip_lo, max=logp_clip_hi)
        fine_prob_active = torch.softmax(logp_pred, dim=1)
        if active_cols_t is not None:
            fine_prob = torch.zeros(
                (fine_prob_active.shape[0], int(full_fine_dim)),
                device=fine_prob_active.device,
                dtype=fine_prob_active.dtype,
            )
            fine_prob.scatter_(1, active_cols_t.expand(fine_prob_active.shape[0], -1), fine_prob_active)
        else:
            fine_prob = fine_prob_active
        coarse_from_fine = fine_prob @ agg_mat_full
        fine_marginals = _marginals_from_joint_torch(p_joint=fine_prob, shape=FINE_SHAPE)

        loss_coarse = -(p_coarse_true * coarse_logp).sum(dim=1).mean()
        if int(aux_t_gate) >= 0:
            aux_mask = (t <= int(aux_t_gate)).to(dtype=fine_prob.dtype)
        else:
            aux_mask = torch.ones_like(t, dtype=fine_prob.dtype)
        aux_denom = torch.clamp(aux_mask.sum(), min=1.0)

        loss_cons_per = 0.5 * torch.abs(coarse_from_fine - coarse_prob).sum(dim=1)
        loss_cons = (loss_cons_per * aux_mask).sum() / aux_denom
        if float(marginal_weight) > 0.0:
            loss_marg_per = torch.zeros((x0.shape[0],), device=x0.device)
            for axis, marg_true in enumerate(marginal_targets):
                loss_marg_per = loss_marg_per + 0.5 * torch.abs(fine_marginals[axis] - marg_true).sum(dim=1)
            loss_marg_per = loss_marg_per / max(len(marginal_targets), 1)
            loss_marg = (loss_marg_per * aux_mask).sum() / aux_denom
        else:
            loss_marg = torch.zeros((), device=x0.device)
        loss = (
            loss_diff
            + float(coarse_weight) * loss_coarse
            + float(consistency_weight) * loss_cons
            + float(marginal_weight) * loss_marg
        )

        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._modules.parameters(), float(self.config.grad_clip))
        self._opt.step()
        return {
            "loss": float(loss.detach().cpu()),
            "loss_diffusion": float(loss_diff.detach().cpu()),
            "loss_coarse": float(loss_coarse.detach().cpu()),
            "loss_consistency": float(loss_cons.detach().cpu()),
            "loss_marginal": float(loss_marg.detach().cpu()),
            "aux_mask_frac": float(aux_mask.mean().detach().cpu()),
        }

    def sample_latent_conditioned(
        self,
        *,
        z_cond: Any,
        n_draws: int,
        device: Any,
    ) -> Any:
        torch = _require_torch()
        assert self._schedule is not None
        self.eval()
        n_regions = int(z_cond.shape[0])
        n_total = int(n_regions * n_draws)
        z_rep = z_cond.repeat_interleave(int(n_draws), dim=0)
        with torch.inference_mode():
            x_t = torch.randn((n_total, self.input_dim), device=device)
            betas = self._schedule["betas"]
            alphas = self._schedule["alphas"]
            alpha_cumprod = self._schedule["alpha_cumprod"]
            posterior_variance = self._schedule["posterior_variance"]
            sqrt_one_minus_alpha_cumprod = self._schedule["sqrt_one_minus_alpha_cumprod"]
            for step in reversed(range(int(self.config.timesteps))):
                t = torch.full((n_total,), int(step), device=device, dtype=torch.long)
                eps_pred = self.denoiser(x_t, t, z_rep)
                beta_t = betas[step]
                alpha_t = alphas[step]
                sqrt_om = sqrt_one_minus_alpha_cumprod[step]
                model_mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_om) * eps_pred)
                if step == 0:
                    x_t = model_mean
                else:
                    noise = torch.randn_like(x_t)
                    x_t = model_mean + torch.sqrt(posterior_variance[step]) * noise
            return x_t.reshape(n_regions, int(n_draws), self.input_dim).detach().cpu()

    def predict_coarse(self, *, cond_raw: Any) -> Any:
        torch = _require_torch()
        self.eval()
        with torch.inference_mode():
            z = self.encoder(cond_raw)
            coarse_feat = self.coarse_feature(z)
            coarse_prob = torch.softmax(self.coarse_out(coarse_feat), dim=1)
        return coarse_prob

    def save(self, path: pathlib.Path, *, payload: dict[str, Any], state_dict: dict[str, Any] | None = None) -> None:
        torch = _require_torch()
        path = pathlib.Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "synthpop.external_joint_hier_diffusion_full.v0",
                "state_dict": state_dict if state_dict is not None else self._modules.state_dict(),
                **payload,
            },
            path,
        )


def _load_joint_wide(*, joint_wide_csv: pathlib.Path, schema_json: pathlib.Path | None) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(joint_wide_csv, low_memory=False)
    required = {"statefp", "puma", "puma_uid", "total_person_weight"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"joint_wide_csv missing columns: {missing}")

    if schema_json is not None:
        schema = json.loads(schema_json.read_text(encoding="utf-8"))
        shape = tuple(int(x) for x in schema["shape"])
        if shape != FINE_SHAPE:
            raise SystemExit(f"unexpected fine shape in schema_json: {shape}; expected {FINE_SHAPE}")
    p_joint_cols = [f"p_joint_{i:03d}" for i in range(FINE_K)]
    miss_joint = [c for c in p_joint_cols if c not in df.columns]
    if miss_joint:
        raise SystemExit(f"joint_wide_csv missing joint columns: {miss_joint[:5]}")

    df["statefp"] = df["statefp"].map(_canon_statefp)
    df["puma5"] = df["puma"].map(_canon_puma5)
    df["puma_uid"] = df.apply(lambda r: _canon_uid(r["statefp"], r["puma5"]), axis=1)
    p_joint = df[p_joint_cols].to_numpy(dtype=np.float32)
    p_joint = np.clip(p_joint, 0.0, None)
    p_joint = p_joint / np.maximum(p_joint.sum(axis=1, keepdims=True), 1e-12)
    ids = df["puma_uid"].astype(str).tolist()
    return df, p_joint, ids


def main() -> None:
    ap = argparse.ArgumentParser(prog="train_external_joint_hier_diffusion_full")
    ap.add_argument("--joint_wide_csv", required=True)
    ap.add_argument("--condition_csv", required=True)
    ap.add_argument("--schema_json", default=None)
    ap.add_argument("--condition_schema_json", default=None)
    ap.add_argument(
        "--condition_scale_mode",
        choices=["none", "log10_total", "log10_total_unit"],
        default="none",
        help="Append ACS block-size features to the normalized marginal condition. Default keeps legacy behavior.",
    )
    ap.add_argument("--condition_extra_csv", default=None, help="Optional PUMA-level numeric feature table appended to the stage-1 condition.")
    ap.add_argument(
        "--condition_extra_standardize",
        choices=["none", "zscore"],
        default="none",
        help="Optional global standardization for condition_extra_csv numeric columns.",
    )
    ap.add_argument(
        "--condition_extra_missing_policy",
        choices=["require", "zero"],
        default="require",
        help="How to handle PUMAs missing from condition_extra_csv.",
    )
    ap.add_argument("--eval_mode", choices=["leave_mi_out", "leave_state_out", "mi_kfold", "state_kfold"], default="leave_mi_out")
    ap.add_argument("--heldout_statefp", default="26", help="State FIPS used by leave-state-out/state-kfold evaluation. Default 26 keeps the original Michigan split.")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--encoder_hidden_dims", default="256,256")
    ap.add_argument("--coarse_hidden_dims", default="256")
    ap.add_argument("--diffusion_hidden_dims", default="512,512")
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--condition_injection", choices=["concat", "film"], default="concat")
    ap.add_argument("--film_hidden_dim", type=int, default=128)
    ap.add_argument("--coarse_weight", type=float, default=0.5)
    ap.add_argument("--consistency_weight", type=float, default=1.0)
    ap.add_argument("--marginal_weight", type=float, default=0.0)
    ap.add_argument(
        "--selection_metric",
        choices=[
            "val_tvd_joint",
            "val_tvd_joint_raw",
            "val_tvd_coarse_head",
            "val_tvd_coarse_from_fine",
            "val_combo",
        ],
        default="val_tvd_joint",
    )
    ap.add_argument("--selection_raw_weight", type=float, default=0.25)
    ap.add_argument("--logp_clip_quantile_low", type=float, default=-1.0)
    ap.add_argument("--logp_clip_quantile_high", type=float, default=-1.0)
    ap.add_argument("--aux_t_gate", type=int, default=-1, help="If >=0, apply consistency/marginal losses only to samples with t <= aux_t_gate.")
    ap.add_argument("--detach_coarse_encoder", action="store_true", help="If set, stop gradients from the coarse branch into the shared encoder.")
    ap.add_argument("--diff_loss_reweight_alpha", type=float, default=0.0)
    ap.add_argument("--diff_loss_reweight_floor", type=float, default=0.05)
    ap.add_argument("--diff_loss_reweight_cap", type=float, default=5.0)
    ap.add_argument("--support_mask_mode", choices=["none", "dataset_nonzero"], default="none")
    ap.add_argument("--support_mask_eps", type=float, default=1e-12)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--val_min_groups", type=int, default=96)
    ap.add_argument("--n_val_joint_samples", type=int, default=32)
    ap.add_argument("--val_ipf_iters", type=int, default=200)
    ap.add_argument("--n_eval_joint_samples", type=int, default=128)
    ap.add_argument("--ipf_iters", type=int, default=200)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--save_best_checkpoint", action="store_true")
    ap.add_argument("--save_final_model", action="store_true")
    ap.add_argument(
        "--save_eval_checkpoint_every",
        type=int,
        default=0,
        help="If > 0, save EMA/raw snapshot checkpoints every N epochs on validation-eval steps.",
    )
    ap.add_argument("--run_label", default="external_joint_hier_diffusion_full")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    torch = _require_torch()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    joint_csv = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    condition_csv = pathlib.Path(args.condition_csv).expanduser().resolve()
    condition_extra_csv = pathlib.Path(args.condition_extra_csv).expanduser().resolve() if args.condition_extra_csv else None
    schema_json = pathlib.Path(args.schema_json).expanduser().resolve() if args.schema_json else None
    condition_schema_json = pathlib.Path(args.condition_schema_json).expanduser().resolve() if args.condition_schema_json else None
    for p in [joint_csv, condition_csv]:
        if not p.exists():
            raise SystemExit(f"path not found: {p}")
    if condition_extra_csv is not None and not condition_extra_csv.exists():
        raise SystemExit(f"path not found: {condition_extra_csv}")
    if schema_json is not None and not schema_json.exists():
        raise SystemExit(f"path not found: {schema_json}")
    if condition_schema_json is not None and not condition_schema_json.exists():
        raise SystemExit(f"path not found: {condition_schema_json}")

    run_id = f"_us_puma_{args.run_label}_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else (_REPO_ROOT / "outputs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    df, p_fine_full, ids = _load_joint_wide(joint_wide_csv=joint_csv, schema_json=schema_json)
    heldout_statefp = _canon_statefp(args.heldout_statefp)
    is_heldout = (df["statefp"] == heldout_statefp).to_numpy(dtype=bool)
    if not bool(is_heldout.any()):
        raise SystemExit(f"No held-out rows found for statefp=={heldout_statefp}.")
    if args.eval_mode in {"leave_mi_out", "leave_state_out"}:
        train_idx = np.where(~is_heldout)[0]
        test_idx = np.where(is_heldout)[0]
        fold_label = "leave_mi_out" if heldout_statefp == "26" else f"leave_state_{heldout_statefp}_out"
        folds = [(fold_label, train_idx, test_idx)]
    else:
        heldout_ids = [ids[i] for i in range(len(ids)) if is_heldout[i]]
        fold_map = _stable_hash_fold(sorted(set(heldout_ids)), n_folds=int(args.n_folds), seed=int(args.seed))
        folds = []
        for f in range(int(args.n_folds)):
            te_mask = np.array([(is_heldout[i] and fold_map.get(ids[i], -1) == f) for i in range(len(ids))], dtype=bool)
            tr_mask = ~te_mask
            tr = np.where(tr_mask)[0]
            te = np.where(te_mask)[0]
            if tr.size > 0 and te.size > 0:
                prefix = "mi_fold" if heldout_statefp == "26" else f"state_{heldout_statefp}_fold"
                folds.append((f"{prefix}_{f}", tr, te))
        if not folds:
            raise SystemExit("No valid folds built in state-kfold mode.")

    var_specs = _load_var_specs_from_schema(schema_json=schema_json)
    cond_var_specs = _load_condition_specs_from_schema(
        condition_schema_json=condition_schema_json,
        fallback_var_specs=var_specs,
    )
    cond_raw, block_slices, cond_meta = _load_external_condition_matrix(
        condition_csv=condition_csv,
        ids=ids,
        var_specs=cond_var_specs,
        condition_scale_mode=str(args.condition_scale_mode),
    )
    cond_raw, cond_meta = _append_condition_extra_matrix(
        cond_raw=cond_raw,
        cond_meta=cond_meta,
        extra_csv=condition_extra_csv,
        ids=ids,
        standardize=str(args.condition_extra_standardize),
        missing_policy=str(args.condition_extra_missing_policy),
    )
    cond_raw = cond_raw.astype(np.float32)
    ext_marg = {var: cond_raw[:, sl].copy() for var, sl in block_slices.items()}
    ext_marg = _augment_ext_marginals_from_cross(
        cond_raw=cond_raw,
        block_slices=block_slices,
        ext_marg=ext_marg,
    )
    missing_marg = [var for var in FINE_VARIABLE_ORDER if var not in ext_marg]
    if missing_marg:
        raise SystemExit(f"condition_csv/schema missing marginal target for fine variable(s): {missing_marg}")

    support_mask_mode = str(args.support_mask_mode)
    support_mask_eps = float(args.support_mask_eps)
    if support_mask_mode == "dataset_nonzero":
        active_cols = np.where((p_fine_full > support_mask_eps).any(axis=0))[0].astype(np.int64)
    else:
        active_cols = np.arange(FINE_K, dtype=np.int64)
    if active_cols.size == 0:
        raise SystemExit("support mask removed all fine cells")

    p_fine = p_fine_full[:, active_cols].astype(np.float32)
    p_fine = p_fine / np.maximum(p_fine.sum(axis=1, keepdims=True), 1e-12)

    agg_mat_np = _build_fine_to_coarse_matrix_full()
    p_coarse = _aggregate_fine_to_coarse_np(p_fine_full, agg_mat_np)
    p_coarse = p_coarse / np.maximum(p_coarse.sum(axis=1, keepdims=True), 1e-12)
    x_log_all = np.log(np.clip(p_fine, 0.0, None) + 1e-6).astype(np.float32)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    agg_mat_t = torch.from_numpy(agg_mat_np.astype(np.float32)).to(device)
    active_cols_t = None
    if active_cols.size != FINE_K:
        active_cols_t = torch.from_numpy(active_cols.reshape(1, -1)).to(device=device, dtype=torch.long)

    internal_by_fold: dict[str, Any] = {}
    saved_checkpoints: dict[str, list[str]] = {}
    selection_by_fold: dict[str, Any] = {}

    for fold_name, train_idx, test_idx in folds:
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        train_core_idx, val_idx = _split_train_val_indices(
            train_idx=train_idx,
            ids=ids,
            seed=int(args.seed),
            fold_name=fold_name,
            val_frac=float(args.val_frac),
            val_min_groups=int(args.val_min_groups),
        )
        x_train_log = x_log_all[train_core_idx]
        x_mean = x_train_log.mean(axis=0, dtype=np.float64).astype(np.float32)
        x_std = x_train_log.std(axis=0, dtype=np.float64).astype(np.float32)
        x_std = np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32)
        if 0.0 <= float(args.logp_clip_quantile_low) < float(args.logp_clip_quantile_high) <= 1.0:
            logp_clip_lo = np.quantile(x_train_log, float(args.logp_clip_quantile_low), axis=0).astype(np.float32)
            logp_clip_hi = np.quantile(x_train_log, float(args.logp_clip_quantile_high), axis=0).astype(np.float32)
        else:
            logp_clip_lo = None
            logp_clip_hi = None
        x_train = ((x_train_log - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)).astype(np.float32)

        model = SharedLatentHierarchicalDiffusion(
            input_dim=int(active_cols.size),
            cond_raw_dim=int(cond_raw.shape[1]),
            latent_dim=int(args.latent_dim),
            encoder_hidden_dims=_parse_hidden_dims(args.encoder_hidden_dims),
            coarse_hidden_dims=_parse_hidden_dims(args.coarse_hidden_dims),
            diffusion_config=TabDDPMConfig(
                timesteps=int(args.timesteps),
                hidden_dims=_parse_hidden_dims(args.diffusion_hidden_dims),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                condition_injection=str(args.condition_injection),
                film_hidden_dim=int(args.film_hidden_dim),
            ),
            seed=int(args.seed),
        )
        model.to(device)
        ema = _ModuleEMA(model._modules, decay=float(args.ema_decay))

        x_train_t = torch.from_numpy(x_train).to(device)
        cond_train_t = torch.from_numpy(cond_raw[train_core_idx]).to(device)
        marginal_targets_train_t = tuple(torch.from_numpy(ext_marg[var][train_core_idx].astype(np.float32)).to(device) for var in FINE_VARIABLE_ORDER)
        p_coarse_train_t = torch.from_numpy(p_coarse[train_core_idx].astype(np.float32)).to(device)
        x_mean_t = torch.from_numpy(x_mean.reshape(1, -1)).to(device)
        x_std_t = torch.from_numpy(x_std.reshape(1, -1)).to(device)
        logp_clip_lo_t = torch.from_numpy(logp_clip_lo.reshape(1, -1)).to(device) if logp_clip_lo is not None else None
        logp_clip_hi_t = torch.from_numpy(logp_clip_hi.reshape(1, -1)).to(device) if logp_clip_hi is not None else None

        train_metrics: list[dict[str, float]] = []
        n_train = int(train_core_idx.size)
        bs = int(args.batch_size)
        eval_every = max(1, int(args.eval_every))
        checkpoint_payload = {
            "cond_raw_dim": int(cond_raw.shape[1]),
            "latent_dim": int(args.latent_dim),
            "encoder_hidden_dims": list(_parse_hidden_dims(args.encoder_hidden_dims)),
            "coarse_hidden_dims": list(_parse_hidden_dims(args.coarse_hidden_dims)),
            "diffusion_hidden_dims": list(_parse_hidden_dims(args.diffusion_hidden_dims)),
            "condition_injection": str(args.condition_injection),
            "condition_scale_mode": str(args.condition_scale_mode),
            "condition_meta": cond_meta,
            "fine_shape": list(FINE_SHAPE),
            "active_fine_dim": int(active_cols.size),
            "masked_zero_dim": int(FINE_K - active_cols.size),
            "coarse_shape": list(COARSE_SHAPE),
            "agg_mat": agg_mat_np.tolist(),
            "support_mask_mode": support_mask_mode,
            "support_mask_eps": support_mask_eps,
            "active_cols": active_cols.tolist(),
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "selection_metric": str(args.selection_metric),
            "selection_raw_weight": float(args.selection_raw_weight),
            "logp_clip_quantile_low": float(args.logp_clip_quantile_low),
            "logp_clip_quantile_high": float(args.logp_clip_quantile_high),
            "aux_t_gate": int(args.aux_t_gate),
            "detach_coarse_encoder": bool(args.detach_coarse_encoder),
            "diff_loss_reweight_alpha": float(args.diff_loss_reweight_alpha),
            "diff_loss_reweight_floor": float(args.diff_loss_reweight_floor),
            "diff_loss_reweight_cap": float(args.diff_loss_reweight_cap),
            "save_eval_checkpoint_every": int(args.save_eval_checkpoint_every),
        }
        best_val_metric = float("inf")
        best_epoch: int | None = None
        best_state: dict[str, Any] | None = None
        best_source = "final_raw"
        best_val_summary: dict[str, Any] | None = None
        best_selection_metric_name = str(args.selection_metric)
        for epoch in range(1, int(args.epochs) + 1):
            order = np.random.permutation(n_train)
            last_stats: dict[str, float] | None = None
            for start in range(0, n_train, bs):
                idx = order[start : start + bs]
                idx_t = torch.from_numpy(idx).to(device=device, dtype=torch.long)
                stats = model.step(
                    x0=x_train_t[idx_t],
                    cond_raw=cond_train_t[idx_t],
                    p_coarse_true=p_coarse_train_t[idx_t],
                    agg_mat=agg_mat_t,
                    agg_mat_full=agg_mat_t,
                    full_fine_dim=int(FINE_K),
                    active_cols_t=active_cols_t,
                    marginal_targets=tuple(mt[idx_t] for mt in marginal_targets_train_t),
                    x_mean=x_mean_t,
                    x_std=x_std_t,
                    logp_clip_lo=logp_clip_lo_t,
                    logp_clip_hi=logp_clip_hi_t,
                    coarse_weight=float(args.coarse_weight),
                    consistency_weight=float(args.consistency_weight),
                    marginal_weight=float(args.marginal_weight),
                    aux_t_gate=int(args.aux_t_gate),
                    detach_coarse_encoder=bool(args.detach_coarse_encoder),
                    diff_loss_reweight_alpha=float(args.diff_loss_reweight_alpha),
                    diff_loss_reweight_floor=float(args.diff_loss_reweight_floor),
                    diff_loss_reweight_cap=float(args.diff_loss_reweight_cap),
                )
                last_stats = stats
                ema.update(model._modules)
            should_log = epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(args.epochs)
            should_eval = val_idx.size > 0 and (epoch == 1 or epoch % eval_every == 0 or epoch == int(args.epochs))
            if last_stats is not None and should_log:
                rec = {"epoch": float(epoch), **last_stats}
                if should_eval:
                    with ema.apply(model._modules):
                        val_summary = _evaluate_joint_distribution(
                            model=model,
                            eval_idx=val_idx,
                            reference_train_idx=train_core_idx,
                            p_fine=p_fine_full,
                            p_coarse=p_coarse,
                            cond_raw=cond_raw,
                            ext_marg=ext_marg,
                            var_specs=var_specs,
                            device=device,
                            x_mean=x_mean,
                            x_std=x_std,
                            agg_mat_np=agg_mat_np,
                            n_eval_joint_samples=int(args.n_val_joint_samples),
                            ipf_iters=int(args.val_ipf_iters),
                            logp_clip_lo=logp_clip_lo,
                            logp_clip_hi=logp_clip_hi,
                            active_cols=active_cols if active_cols.size != FINE_K else None,
                            full_fine_dim=int(FINE_K),
                        )
                    rec["val_tvd_joint"] = float(val_summary["tvd_joint"]["mean"])
                    rec["val_tvd_joint_raw"] = float(val_summary["tvd_joint_raw"]["mean"])
                    rec["val_tvd_coarse_head"] = float(val_summary["tvd_coarse_head"]["mean"])
                    rec["val_tvd_coarse_from_fine"] = float(val_summary["tvd_coarse_from_fine"]["mean"])
                    if str(args.selection_metric) == "val_tvd_joint":
                        selection_metric = rec["val_tvd_joint"]
                    elif str(args.selection_metric) == "val_tvd_joint_raw":
                        selection_metric = rec["val_tvd_joint_raw"]
                    elif str(args.selection_metric) == "val_tvd_coarse_head":
                        selection_metric = rec["val_tvd_coarse_head"]
                    elif str(args.selection_metric) == "val_tvd_coarse_from_fine":
                        selection_metric = rec["val_tvd_coarse_from_fine"]
                    elif str(args.selection_metric) == "val_combo":
                        selection_metric = rec["val_tvd_joint"] + float(args.selection_raw_weight) * rec["val_tvd_joint_raw"]
                    else:
                        raise ValueError(f"unsupported selection_metric: {args.selection_metric}")
                    rec["selection_metric"] = float(selection_metric)
                    save_eval_checkpoint_every = int(args.save_eval_checkpoint_every)
                    if save_eval_checkpoint_every > 0 and epoch > 1 and epoch % save_eval_checkpoint_every == 0:
                        snapshot_state = (
                            ema.cpu_state_dict()
                            if ema.enabled
                            else {k: v.detach().cpu().clone() for k, v in model._modules.state_dict().items()}
                        )
                        ckpt = out_dir / "checkpoints" / fold_name / f"epoch_{epoch:04d}.pt"
                        model.save(
                            ckpt,
                            payload={
                                **checkpoint_payload,
                                "snapshot_epoch": int(epoch),
                                "snapshot_source": "ema" if ema.enabled else "raw",
                                "snapshot_val_tvd_joint": float(rec["val_tvd_joint"]),
                                "snapshot_val_tvd_joint_raw": float(rec["val_tvd_joint_raw"]),
                                "snapshot_val_tvd_coarse_head": float(rec["val_tvd_coarse_head"]),
                                "snapshot_val_tvd_coarse_from_fine": float(rec["val_tvd_coarse_from_fine"]),
                            },
                            state_dict=snapshot_state,
                        )
                        saved_checkpoints.setdefault(fold_name, []).append(str(ckpt))
                    if selection_metric < best_val_metric:
                        best_val_metric = float(selection_metric)
                        best_epoch = int(epoch)
                        best_source = "best_val_ema" if ema.enabled else "best_val_raw"
                        best_state = ema.cpu_state_dict() if ema.enabled else {k: v.detach().cpu().clone() for k, v in model._modules.state_dict().items()}
                        best_val_summary = val_summary
                        if bool(args.save_best_checkpoint) and best_state is not None:
                            ckpt = out_dir / "checkpoints" / fold_name / "best.pt"
                            model.save(
                                ckpt,
                                payload={
                                    **checkpoint_payload,
                                    "best_epoch": int(epoch),
                                    "selection_metric": str(args.selection_metric),
                                },
                                state_dict=best_state,
                            )
                            saved_checkpoints.setdefault(fold_name, []).append(str(ckpt))
                train_metrics.append(rec)
                print(
                    f"[train] fold={fold_name} epoch={epoch} "
                    f"loss={rec['loss']:.6f} "
                    f"diff={rec['loss_diffusion']:.6f} "
                    f"coarse={rec['loss_coarse']:.6f} "
                    f"cons={rec['loss_consistency']:.6f} "
                    f"marg={rec['loss_marginal']:.6f}"
                )

        saved_checkpoints.setdefault(fold_name, [])
        if bool(args.save_final_model):
            ckpt = out_dir / "checkpoints" / fold_name / "final.pt"
            model.save(
                ckpt,
                payload=checkpoint_payload,
            )
            saved_checkpoints[fold_name].append(str(ckpt))

        if best_state is not None:
            model._modules.load_state_dict(best_state, strict=True)
        elif ema.enabled:
            model._modules.load_state_dict(ema.cpu_state_dict(), strict=True)
            best_source = "ema_final"

        fold_summary = _evaluate_joint_distribution(
            model=model,
            eval_idx=test_idx,
            reference_train_idx=train_core_idx,
            p_fine=p_fine_full,
            p_coarse=p_coarse,
            cond_raw=cond_raw,
            ext_marg=ext_marg,
            var_specs=var_specs,
            device=device,
            x_mean=x_mean,
            x_std=x_std,
            agg_mat_np=agg_mat_np,
            n_eval_joint_samples=int(args.n_eval_joint_samples),
            ipf_iters=int(args.ipf_iters),
            logp_clip_lo=logp_clip_lo,
            logp_clip_hi=logp_clip_hi,
            active_cols=active_cols if active_cols.size != FINE_K else None,
            full_fine_dim=int(FINE_K),
        )
        fold_summary["n_train"] = int(train_core_idx.size)
        fold_summary["n_val"] = int(val_idx.size)
        fold_summary["n_test"] = int(test_idx.size)
        internal_by_fold[fold_name] = fold_summary
        selection_by_fold[fold_name] = {
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_val_metric": float(best_val_metric) if best_epoch is not None else None,
            "best_val_tvd_joint": float(best_val_summary["tvd_joint"]["mean"]) if best_val_summary is not None else None,
            "best_val_tvd_joint_raw": float(best_val_summary["tvd_joint_raw"]["mean"]) if best_val_summary is not None else None,
            "best_val_tvd_coarse_head": float(best_val_summary["tvd_coarse_head"]["mean"]) if best_val_summary is not None else None,
            "best_val_tvd_coarse_from_fine": float(best_val_summary["tvd_coarse_from_fine"]["mean"]) if best_val_summary is not None else None,
            "selection_metric": best_selection_metric_name,
            "selected_state": best_source,
            "ema_decay": float(args.ema_decay),
            "train_core_size": int(train_core_idx.size),
            "val_size": int(val_idx.size),
        }
        _write_json(out_dir / "metrics" / f"{fold_name}_summary.json", fold_summary)
        _write_json(out_dir / "metrics" / f"{fold_name}_train_metrics.json", train_metrics)
        if best_val_summary is not None:
            _write_json(out_dir / "metrics" / f"{fold_name}_best_val_summary.json", best_val_summary)
        _write_json(out_dir / "metrics" / f"{fold_name}_selection.json", selection_by_fold[fold_name])

    fold_names = sorted(internal_by_fold.keys())
    overall: dict[str, Any] = {
        "tvd_joint_raw": _summ([float(internal_by_fold[f]["tvd_joint_raw"]["mean"]) for f in fold_names]),
        "tvd_joint": _summ([float(internal_by_fold[f]["tvd_joint"]["mean"]) for f in fold_names]),
        "cosine_joint_raw": _summ([float(internal_by_fold[f]["cosine_joint_raw"]["mean"]) for f in fold_names]),
        "cosine_joint": _summ([float(internal_by_fold[f]["cosine_joint"]["mean"]) for f in fold_names]),
        "tvd_coarse_head": _summ([float(internal_by_fold[f]["tvd_coarse_head"]["mean"]) for f in fold_names]),
        "tvd_coarse_from_fine": _summ([float(internal_by_fold[f]["tvd_coarse_from_fine"]["mean"]) for f in fold_names]),
    }
    for var, _, _ in var_specs:
        overall[f"tvd_{var}"] = _summ([float(internal_by_fold[f][f"tvd_{var}"]["mean"]) for f in fold_names])
        overall[f"tvd_{var}_raw"] = _summ([float(internal_by_fold[f][f"tvd_{var}_raw"]["mean"]) for f in fold_names])

    baselines = {
        "ipf_train_seed_external": {
            "tvd_joint": _summ([float(internal_by_fold[f]["ipf_train_seed_external"]["tvd_joint"]["mean"]) for f in fold_names])
        },
        "independence_external": {
            "tvd_joint": _summ([float(internal_by_fold[f]["independence_external"]["tvd_joint"]["mean"]) for f in fold_names])
        },
    }
    summary = {
        "hier_diffusion_joint": overall,
        "baselines": baselines,
        "by_fold": internal_by_fold,
    }
    _write_json(out_dir / "metrics" / "hier_diffusion_summary.json", summary)

    run_summary = {
        "created_utc": _utc_now_iso_local(),
        "joint_wide_csv": str(joint_csv),
        "condition_csv": str(condition_csv),
        "schema_json": str(schema_json) if schema_json is not None else None,
        "condition_schema_json": str(condition_schema_json) if condition_schema_json is not None else None,
        "condition_scale_mode": str(args.condition_scale_mode),
        "condition_meta": cond_meta,
        "n_rows_total": int(df.shape[0]),
        "heldout_statefp": str(heldout_statefp),
        "n_test_heldout": int(sum(int(internal_by_fold[f]["n_test"]) for f in fold_names)),
        "cond_raw_dim": int(cond_raw.shape[1]),
        "latent_dim": int(args.latent_dim),
        "fine_shape": list(FINE_SHAPE),
        "active_fine_dim": int(active_cols.size),
        "masked_zero_dim": int(FINE_K - active_cols.size),
        "coarse_shape": list(COARSE_SHAPE),
        "timesteps": int(args.timesteps),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "encoder_hidden_dims": list(_parse_hidden_dims(args.encoder_hidden_dims)),
        "coarse_hidden_dims": list(_parse_hidden_dims(args.coarse_hidden_dims)),
        "diffusion_hidden_dims": list(_parse_hidden_dims(args.diffusion_hidden_dims)),
        "condition_injection": str(args.condition_injection),
        "coarse_weight": float(args.coarse_weight),
        "consistency_weight": float(args.consistency_weight),
        "marginal_weight": float(args.marginal_weight),
        "selection_metric": str(args.selection_metric),
        "selection_raw_weight": float(args.selection_raw_weight),
        "logp_clip_quantile_low": float(args.logp_clip_quantile_low),
        "logp_clip_quantile_high": float(args.logp_clip_quantile_high),
        "aux_t_gate": int(args.aux_t_gate),
        "detach_coarse_encoder": bool(args.detach_coarse_encoder),
        "diff_loss_reweight_alpha": float(args.diff_loss_reweight_alpha),
        "diff_loss_reweight_floor": float(args.diff_loss_reweight_floor),
        "diff_loss_reweight_cap": float(args.diff_loss_reweight_cap),
        "save_eval_checkpoint_every": int(args.save_eval_checkpoint_every),
        "support_mask_mode": support_mask_mode,
        "support_mask_eps": support_mask_eps,
        "n_eval_joint_samples": int(args.n_eval_joint_samples),
        "ipf_iters": int(args.ipf_iters),
        "eval_every": int(args.eval_every),
        "val_frac": float(args.val_frac),
        "val_min_groups": int(args.val_min_groups),
        "n_val_joint_samples": int(args.n_val_joint_samples),
        "val_ipf_iters": int(args.val_ipf_iters),
        "ema_decay": float(args.ema_decay),
        "saved_checkpoints": saved_checkpoints,
        "selection_by_fold": selection_by_fold,
        "condition_meta": cond_meta,
        "summary": summary,
    }
    _write_json(out_dir / "run_summary.json", run_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
