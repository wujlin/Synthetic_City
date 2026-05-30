#!/usr/bin/env python3
from __future__ import annotations

"""
Train a dedicated Stage-1 coarse diffusion model for the 5-way coarse-to-fine pipeline.

Target:
  288-cell coarse joint
  AGEP_lite(4) x SEX(2) x SCHL_lite(3) x ESR_lite(3) x EARN_lite(4)

Condition:
  external ACS-derived regional conditions, encoded into a shared regional latent.

This model is intentionally coarse-only:
- it preserves the shared regional-context design
- it no longer shares capacity with the 3000-cell full joint task
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


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.model.diffusion_tabular import (
    TabDDPMConfig,
    _DenoiserMLP,
    _FiLMDenoiserMLP,
    _require_torch,
)
import tools.model.train_external_joint_hier_diffusion_full as base
from tools.model.external_c2f_full_earn_schema import (
    AGE_FINE_TO_COARSE,
    COARSE_CATEGORIES,
    COARSE_K,
    COARSE_PRESET,
    COARSE_SHAPE,
    COARSE_VARIABLE_ORDER,
    EARN_FINE_TO_COARSE,
    ESR_FINE_TO_COARSE,
    FULL_K,
    FULL_SHAPE,
    FULL_VARIABLE_ORDER,
    SCHL_FINE_TO_COARSE,
    coarse_from_full_flat,
)
from tools.model.train_us_puma_5var_diffusion import (
    _cosine,
    _ipf_nd,
    _nd_independence,
    _parse_hidden_dims,
    _softmax_rows,
    _stable_hash_fold,
    _summ,
    _tvd,
    _write_json,
)
from tools.model.train_us_puma_external_v1_diffusion import (
    _append_condition_extra_matrix,
    _load_condition_specs_from_schema,
    _load_external_condition_matrix,
    _load_var_specs_from_schema,
)


def _aggregate_prob_to_coarse(*, p_full: np.ndarray, map_idx: np.ndarray, coarse_dim: int) -> np.ndarray:
    out = np.bincount(
        np.asarray(map_idx, dtype=np.int64),
        weights=np.asarray(p_full, dtype=np.float64),
        minlength=int(coarse_dim),
    ).astype(np.float64)
    out = out / max(float(out.sum()), 1e-12)
    return out


def _load_full_joint_wide(*, joint_wide_csv: pathlib.Path, schema_json: pathlib.Path | None) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if schema_json is not None:
        schema = json.loads(schema_json.read_text(encoding="utf-8"))
        shape = tuple(int(x) for x in schema["shape"])
        if shape != FULL_SHAPE:
            raise SystemExit(f"unexpected full shape in schema_json: {shape}; expected {FULL_SHAPE}")

    df = pd.read_csv(joint_wide_csv, low_memory=False)
    req = {"statefp", "puma", "puma_uid"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"joint_wide_csv missing columns: {miss}")

    p_joint_cols = [f"p_joint_{i:03d}" for i in range(FULL_K)]
    miss_joint = [c for c in p_joint_cols if c not in df.columns]
    if miss_joint:
        raise SystemExit(f"joint_wide_csv missing joint columns: {miss_joint[:5]}")

    df["statefp"] = df["statefp"].map(base._canon_statefp)
    df["puma5"] = df["puma"].map(base._canon_puma5)
    df["puma_uid"] = df.apply(lambda r: base._canon_uid(r["statefp"], r["puma5"]), axis=1)
    p_joint = df[p_joint_cols].to_numpy(dtype=np.float32)
    p_joint = np.clip(p_joint, 0.0, None)
    p_joint = p_joint / np.maximum(p_joint.sum(axis=1, keepdims=True), 1e-12)
    ids = df["puma_uid"].astype(str).tolist()
    return df, p_joint, ids


def _coarse_marginals_from_full_ext_row(ext_row: dict[str, np.ndarray]) -> list[np.ndarray]:
    return [
        _aggregate_prob_to_coarse(
            p_full=np.asarray(ext_row["AGEP_bin"], dtype=np.float64),
            map_idx=AGE_FINE_TO_COARSE,
            coarse_dim=COARSE_SHAPE[0],
        ),
        np.asarray(ext_row["SEX"], dtype=np.float64),
        _aggregate_prob_to_coarse(
            p_full=np.asarray(ext_row["SCHL_allpop"], dtype=np.float64),
            map_idx=SCHL_FINE_TO_COARSE,
            coarse_dim=COARSE_SHAPE[2],
        ),
        _aggregate_prob_to_coarse(
            p_full=np.asarray(ext_row["ESR_allpop"], dtype=np.float64),
            map_idx=ESR_FINE_TO_COARSE,
            coarse_dim=COARSE_SHAPE[3],
        ),
        _aggregate_prob_to_coarse(
            p_full=np.asarray(ext_row["EARN_16p_bin"], dtype=np.float64),
            map_idx=EARN_FINE_TO_COARSE,
            coarse_dim=COARSE_SHAPE[4],
        ),
    ]


def _marginals_from_joint_torch(*, p_joint: Any, shape: tuple[int, ...]) -> list[Any]:
    torch = _require_torch()
    ndim = len(shape)
    p_nd = p_joint.reshape((p_joint.shape[0],) + tuple(int(x) for x in shape))
    out: list[Any] = []
    for axis in range(ndim):
        reduce_dims = tuple(i + 1 for i in range(ndim) if i != axis)
        out.append(torch.sum(p_nd, dim=reduce_dims))
    return out


def _load_frozen_full_model_teacher(
    *,
    checkpoint_path: pathlib.Path,
    timesteps: int,
    seed: int,
    device: str,
) -> Any:
    import tools.model.train_external_joint_hier_diffusion_full_earn as full_teacher

    torch = _require_torch()
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != "synthpop.external_joint_hier_diffusion_full.v0":
        raise SystemExit(f"Unsupported teacher checkpoint format: {checkpoint_path}")

    fine_shape = tuple(int(x) for x in payload.get("fine_shape", FULL_SHAPE))
    input_dim = int(payload.get("active_fine_dim", int(np.prod(fine_shape))))
    model = full_teacher.base.SharedLatentHierarchicalDiffusion(
        input_dim=int(input_dim),
        cond_raw_dim=int(payload["cond_raw_dim"]),
        latent_dim=int(payload["latent_dim"]),
        encoder_hidden_dims=tuple(int(x) for x in payload["encoder_hidden_dims"]),
        coarse_hidden_dims=tuple(int(x) for x in payload["coarse_hidden_dims"]),
        diffusion_config=TabDDPMConfig(
            timesteps=int(timesteps),
            hidden_dims=tuple(int(x) for x in payload["diffusion_hidden_dims"]),
            lr=float(payload.get("lr", 1e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            condition_injection=str(payload.get("condition_injection", "concat")),
            film_hidden_dim=int(payload.get("film_hidden_dim", 128)),
        ),
        seed=int(seed),
    )
    model._modules.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    for p in model._modules.parameters():
        p.requires_grad_(False)
    return model


def _predict_teacher_coarse_probs(
    *,
    teacher_model: Any,
    cond_raw: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    torch = _require_torch()
    preds: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, int(cond_raw.shape[0]), int(batch_size)):
            batch = torch.from_numpy(cond_raw[start : start + int(batch_size)]).to(device=device, dtype=torch.float32)
            p = teacher_model.predict_coarse(cond_raw=batch).detach().cpu().numpy().astype(np.float32)
            preds.append(p)
    out = np.concatenate(preds, axis=0)
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out.astype(np.float32)


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


class SharedLatentCoarseDiffusion:
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
        enable_coarse_head: bool,
        coarse_predict_mode: str,
    ) -> None:
        torch = _require_torch()
        nn = torch.nn
        torch.manual_seed(int(seed))

        self.input_dim = int(input_dim)
        self.cond_raw_dim = int(cond_raw_dim)
        self.latent_dim = int(latent_dim)
        self.seed = int(seed)
        self.config = diffusion_config
        self.coarse_hidden_dims = tuple(int(x) for x in coarse_hidden_dims)
        self.enable_coarse_head = bool(enable_coarse_head)
        self.coarse_predict_mode = str(coarse_predict_mode)
        self._schedule: dict[str, Any] | None = None

        self.encoder = self._make_mlp(
            in_dim=self.cond_raw_dim,
            hidden_dims=encoder_hidden_dims,
            out_dim=self.latent_dim,
            nn=nn,
        )
        if self.enable_coarse_head:
            self.coarse_feature = self._make_mlp(
                in_dim=self.latent_dim,
                hidden_dims=self.coarse_hidden_dims,
                out_dim=self.latent_dim,
                nn=nn,
            )
            self.coarse_out = nn.Linear(self.latent_dim, self.input_dim)
        else:
            self.coarse_feature = None
            self.coarse_out = None
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

        modules: list[Any] = [self.encoder]
        if self.enable_coarse_head:
            modules.extend([self.coarse_feature, self.coarse_out])
        modules.append(self.denoiser)
        self._modules = nn.ModuleList(modules)
        self._opt = torch.optim.AdamW(
            self._modules.parameters(),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )

        self._predict_x_mean: np.ndarray | None = None
        self._predict_x_std: np.ndarray | None = None
        self._predict_logp_clip_lo: np.ndarray | None = None
        self._predict_logp_clip_hi: np.ndarray | None = None
        self._predict_n_draws: int = 64

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

    def set_predict_meta(
        self,
        *,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        logp_clip_lo: np.ndarray | None,
        logp_clip_hi: np.ndarray | None,
        n_draws: int,
    ) -> None:
        self._predict_x_mean = np.asarray(x_mean, dtype=np.float32).reshape(1, 1, -1)
        self._predict_x_std = np.asarray(x_std, dtype=np.float32).reshape(1, 1, -1)
        self._predict_logp_clip_lo = None if logp_clip_lo is None else np.asarray(logp_clip_lo, dtype=np.float32).reshape(1, 1, -1)
        self._predict_logp_clip_hi = None if logp_clip_hi is None else np.asarray(logp_clip_hi, dtype=np.float32).reshape(1, 1, -1)
        self._predict_n_draws = int(n_draws)

    def _predict_head_probs(self, *, cond_raw: Any) -> Any:
        torch = _require_torch()
        if not self.enable_coarse_head or self.coarse_feature is None or self.coarse_out is None:
            raise RuntimeError("coarse head is not enabled for this checkpoint")
        self.eval()
        with torch.inference_mode():
            z = self.encoder(cond_raw)
            coarse_feat = self.coarse_feature(z)
            coarse_prob = torch.softmax(self.coarse_out(coarse_feat), dim=1)
        return coarse_prob

    def step(
        self,
        *,
        x0: Any,
        cond_raw: Any,
        p_coarse_true: Any,
        marginal_targets: tuple[Any, ...],
        x_mean: Any,
        x_std: Any,
        logp_clip_lo: Any | None,
        logp_clip_hi: Any | None,
        diffusion_weight: float,
        marginal_weight: float,
        coarse_head_weight: float,
        consistency_weight: float,
        aux_t_gate: int,
        teacher_coarse_prob: Any | None,
        distill_weight: float,
        distill_temperature: float,
    ) -> dict[str, float]:
        torch = _require_torch()
        assert self._schedule is not None
        self.train()

        z = self.encoder(cond_raw)
        if self.enable_coarse_head and self.coarse_feature is not None and self.coarse_out is not None:
            coarse_feat = self.coarse_feature(z)
            coarse_logits = self.coarse_out(coarse_feat)
            coarse_logp = torch.log_softmax(coarse_logits, dim=1)
            coarse_prob = torch.softmax(coarse_logits, dim=1)
            loss_coarse = -(p_coarse_true * coarse_logp).sum(dim=1).mean()
        else:
            coarse_logits = None
            coarse_prob = None
            loss_coarse = torch.zeros((), device=x0.device)
        t = torch.randint(0, self.config.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        sqrt_acp = self._schedule["sqrt_alpha_cumprod"][t].unsqueeze(1)
        sqrt_om = self._schedule["sqrt_one_minus_alpha_cumprod"][t].unsqueeze(1)
        x_t = sqrt_acp * x0 + sqrt_om * noise
        eps_pred = self.denoiser(x_t, t, z)
        loss_diff = torch.nn.functional.mse_loss(eps_pred, noise)

        x0_pred = (x_t - sqrt_om * eps_pred) / torch.clamp(sqrt_acp, min=1e-12)
        logp_pred = x0_pred * x_std + x_mean
        if logp_clip_lo is not None and logp_clip_hi is not None:
            logp_pred = torch.clamp(logp_pred, min=logp_clip_lo, max=logp_clip_hi)
        p_pred = torch.softmax(logp_pred, dim=1)

        if float(marginal_weight) > 0.0:
            pred_marginals = _marginals_from_joint_torch(p_joint=p_pred, shape=COARSE_SHAPE)
            loss_marg_per = torch.zeros((x0.shape[0],), device=x0.device)
            for axis, marg_true in enumerate(marginal_targets):
                loss_marg_per = loss_marg_per + 0.5 * torch.abs(pred_marginals[axis] - marg_true).sum(dim=1)
            loss_marg = loss_marg_per.mean() / max(len(marginal_targets), 1)
        else:
            loss_marg = torch.zeros((), device=x0.device)

        if int(aux_t_gate) >= 0:
            aux_mask = (t <= int(aux_t_gate)).to(dtype=p_pred.dtype)
        else:
            aux_mask = torch.ones_like(t, dtype=p_pred.dtype)
        aux_denom = torch.clamp(aux_mask.sum(), min=1.0)
        if coarse_prob is not None and float(consistency_weight) > 0.0:
            loss_cons_per = 0.5 * torch.abs(p_pred - coarse_prob).sum(dim=1)
            loss_cons = (loss_cons_per * aux_mask).sum() / aux_denom
        else:
            loss_cons = torch.zeros((), device=x0.device)
        if coarse_logits is not None and teacher_coarse_prob is not None and float(distill_weight) > 0.0:
            temp = max(float(distill_temperature), 1e-6)
            teacher_logp = torch.log(torch.clamp(teacher_coarse_prob, min=1e-12))
            teacher_prob_t = torch.softmax(teacher_logp / temp, dim=1)
            student_logp_t = torch.log_softmax(coarse_logits / temp, dim=1)
            loss_distill = -(teacher_prob_t * student_logp_t).sum(dim=1).mean() * (temp * temp)
        else:
            loss_distill = torch.zeros((), device=x0.device)

        loss = (
            float(diffusion_weight) * loss_diff
            + float(marginal_weight) * loss_marg
            + float(coarse_head_weight) * loss_coarse
            + float(consistency_weight) * loss_cons
            + float(distill_weight) * loss_distill
        )

        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._modules.parameters(), float(self.config.grad_clip))
        self._opt.step()
        return {
            "loss": float(loss.detach().cpu()),
            "loss_diffusion": float(loss_diff.detach().cpu()),
            "loss_marginal": float(loss_marg.detach().cpu()),
            "loss_coarse": float(loss_coarse.detach().cpu()),
            "loss_consistency": float(loss_cons.detach().cpu()),
            "loss_distill": float(loss_distill.detach().cpu()),
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
            return x_t.reshape(n_regions, int(n_draws), self.input_dim)

    def predict_coarse(self, *, cond_raw: Any) -> Any:
        torch = _require_torch()
        mode = str(self.coarse_predict_mode).lower().strip()
        if mode == "head":
            return self._predict_head_probs(cond_raw=cond_raw)
        if mode != "diffusion":
            raise ValueError(f"Unsupported coarse_predict_mode: {self.coarse_predict_mode}")
        if self._predict_x_mean is None or self._predict_x_std is None:
            raise RuntimeError("predict metadata is not set")
        self.eval()
        device = cond_raw.device
        with torch.inference_mode():
            z = self.encoder(cond_raw)
            x_samples = self.sample_latent_conditioned(
                z_cond=z,
                n_draws=int(self._predict_n_draws),
                device=device,
            )
            x_mean = torch.from_numpy(self._predict_x_mean).to(device=device, dtype=x_samples.dtype)
            x_std = torch.from_numpy(self._predict_x_std).to(device=device, dtype=x_samples.dtype)
            logp = x_samples * x_std + x_mean
            if self._predict_logp_clip_lo is not None and self._predict_logp_clip_hi is not None:
                clip_lo = torch.from_numpy(self._predict_logp_clip_lo).to(device=device, dtype=x_samples.dtype)
                clip_hi = torch.from_numpy(self._predict_logp_clip_hi).to(device=device, dtype=x_samples.dtype)
                logp = torch.clamp(logp, min=clip_lo, max=clip_hi)
            p_draws = torch.softmax(logp, dim=2)
            p_hat = p_draws.mean(dim=1)
            p_hat = p_hat / torch.clamp(p_hat.sum(dim=1, keepdim=True), min=1e-12)
        return p_hat

    def save(self, path: pathlib.Path, *, payload: dict[str, Any], state_dict: dict[str, Any] | None = None) -> None:
        torch = _require_torch()
        path = pathlib.Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "synthpop.external_c2f_full_earn_stage1_coarse_diffusion.v0",
                "state_dict": state_dict if state_dict is not None else self._modules.state_dict(),
                **payload,
            },
            path,
        )


def load_model_from_checkpoint(*, checkpoint_path: pathlib.Path, timesteps: int, seed: int) -> tuple[SharedLatentCoarseDiffusion, dict[str, Any]]:
    torch = _require_torch()
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != "synthpop.external_c2f_full_earn_stage1_coarse_diffusion.v0":
        raise SystemExit(f"Unsupported stage1 coarse checkpoint format: {checkpoint_path}")
    payload_shape = tuple(int(x) for x in payload.get("coarse_shape", ()))
    if payload_shape and payload_shape != tuple(COARSE_SHAPE):
        raise SystemExit(
            "Stage-1 checkpoint coarse schema mismatch: "
            f"checkpoint coarse_shape={payload_shape}, active coarse_shape={tuple(COARSE_SHAPE)}. "
            "Set SYNTHETIC_CITY_C2F_COARSE_PRESET to the preset used during training."
        )

    coarse_predict_mode = str(payload.get("coarse_predict_mode", "diffusion"))
    coarse_hidden_dims = tuple(int(x) for x in payload.get("coarse_hidden_dims", []))
    enable_coarse_head = bool(payload.get("enable_coarse_head", False) or coarse_predict_mode == "head")
    model = SharedLatentCoarseDiffusion(
        input_dim=int(payload["input_dim"]),
        cond_raw_dim=int(payload["cond_raw_dim"]),
        latent_dim=int(payload["latent_dim"]),
        encoder_hidden_dims=tuple(int(x) for x in payload["encoder_hidden_dims"]),
        coarse_hidden_dims=coarse_hidden_dims,
        diffusion_config=TabDDPMConfig(
            timesteps=int(timesteps),
            hidden_dims=tuple(int(x) for x in payload["diffusion_hidden_dims"]),
            lr=float(payload.get("lr", 1e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            condition_injection=str(payload.get("condition_injection", "concat")),
            film_hidden_dim=int(payload.get("film_hidden_dim", 128)),
        ),
        seed=int(seed),
        enable_coarse_head=enable_coarse_head,
        coarse_predict_mode=coarse_predict_mode,
    )
    model._modules.load_state_dict(payload["state_dict"], strict=True)
    model.set_predict_meta(
        x_mean=np.asarray(payload["x_mean"], dtype=np.float32),
        x_std=np.asarray(payload["x_std"], dtype=np.float32),
        logp_clip_lo=None if payload.get("logp_clip_lo") is None else np.asarray(payload["logp_clip_lo"], dtype=np.float32),
        logp_clip_hi=None if payload.get("logp_clip_hi") is None else np.asarray(payload["logp_clip_hi"], dtype=np.float32),
        n_draws=int(payload.get("predict_n_draws", 64)),
    )
    return model, payload


def _evaluate_coarse_distribution(
    *,
    model: SharedLatentCoarseDiffusion,
    eval_idx: np.ndarray,
    reference_train_idx: np.ndarray,
    p_coarse: np.ndarray,
    cond_raw: np.ndarray,
    ext_coarse_marg: dict[str, np.ndarray],
    device: Any,
    ipf_iters: int,
) -> dict[str, Any]:
    torch = _require_torch()
    eval_idx = np.asarray(eval_idx, dtype=int)
    ref_idx = np.asarray(reference_train_idx, dtype=int)
    cond_eval_t = torch.from_numpy(cond_raw[eval_idx]).to(device=device, dtype=torch.float32)
    coarse_pred_raw = model.predict_coarse(cond_raw=cond_eval_t).detach().cpu().numpy().astype(np.float64)

    coarse_pred = coarse_pred_raw.copy()
    tvd_raw: list[float] = []
    tvd_proj: list[float] = []
    cosine_raw: list[float] = []
    cosine_proj: list[float] = []
    var_eval: dict[str, list[float]] = {var: [] for var in COARSE_VARIABLE_ORDER}
    var_raw: dict[str, list[float]] = {var: [] for var in COARSE_VARIABLE_ORDER}
    for j, idx in enumerate(eval_idx):
        target_marginals = [np.asarray(ext_coarse_marg[var][idx], dtype=np.float64) for var in COARSE_VARIABLE_ORDER]
        coarse_pred[j] = _ipf_nd(
            seed_joint=coarse_pred_raw[j].reshape(COARSE_SHAPE),
            target_marginals=target_marginals,
            shape=COARSE_SHAPE,
            max_iter=int(ipf_iters),
        ).reshape(-1)
        coarse_pred[j] = coarse_pred[j] / max(float(coarse_pred[j].sum()), 1e-12)

        p_true = np.asarray(p_coarse[idx], dtype=np.float64)
        tvd_raw.append(_tvd(coarse_pred_raw[j], p_true))
        tvd_proj.append(_tvd(coarse_pred[j], p_true))
        cosine_raw.append(_cosine(coarse_pred_raw[j], p_true))
        cosine_proj.append(_cosine(coarse_pred[j], p_true))
        for axis, var in enumerate(COARSE_VARIABLE_ORDER):
            mr = base._marginal_from_joint(coarse_pred_raw[j], shape=COARSE_SHAPE, axis=axis)
            me = base._marginal_from_joint(coarse_pred[j], shape=COARSE_SHAPE, axis=axis)
            mt = np.asarray(ext_coarse_marg[var][idx], dtype=np.float64)
            var_raw[var].append(_tvd(mr, mt))
            var_eval[var].append(_tvd(me, mt))

    train_seed = np.asarray(p_coarse[ref_idx], dtype=np.float64).mean(axis=0)
    train_seed = train_seed / max(float(train_seed.sum()), 1e-12)
    tvd_ipf: list[float] = []
    tvd_ind: list[float] = []
    for idx in eval_idx:
        target_marginals = [np.asarray(ext_coarse_marg[var][idx], dtype=np.float64) for var in COARSE_VARIABLE_ORDER]
        p_ipf = _ipf_nd(
            seed_joint=train_seed.reshape(COARSE_SHAPE),
            target_marginals=target_marginals,
            shape=COARSE_SHAPE,
            max_iter=int(ipf_iters),
        ).reshape(-1)
        p_ipf = p_ipf / max(float(p_ipf.sum()), 1e-12)
        tvd_ipf.append(_tvd(p_ipf, p_coarse[idx]))
        p_ind = _nd_independence(target_marginals)
        tvd_ind.append(_tvd(p_ind, p_coarse[idx]))

    summary: dict[str, Any] = {
        "n_eval": int(eval_idx.size),
        "n_reference_train": int(ref_idx.size),
        "tvd_joint_raw": _summ(tvd_raw),
        "tvd_joint": _summ(tvd_proj),
        "cosine_joint_raw": _summ(cosine_raw),
        "cosine_joint": _summ(cosine_proj),
        "ipf_train_seed_external": {"tvd_joint": _summ(tvd_ipf)},
        "independence_external": {"tvd_joint": _summ(tvd_ind)},
    }
    for var in COARSE_VARIABLE_ORDER:
        summary[f"tvd_{var}"] = _summ(var_eval[var])
        summary[f"tvd_{var}_raw"] = _summ(var_raw[var])
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(prog="train_external_c2f_full_earn_stage1_coarse")
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
    ap.add_argument("--condition_extra_csv", default=None)
    ap.add_argument("--condition_extra_standardize", choices=["none", "zscore"], default="none")
    ap.add_argument("--condition_extra_missing_policy", choices=["require", "zero"], default="require")
    ap.add_argument("--eval_mode", choices=["leave_mi_out", "leave_state_out", "mi_kfold", "state_kfold"], default="leave_mi_out")
    ap.add_argument("--heldout_statefp", default="26", help="State FIPS used by leave-state-out/state-kfold evaluation. Default 26 keeps the original Michigan split.")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--encoder_hidden_dims", default="256,256")
    ap.add_argument("--coarse_hidden_dims", default="256")
    ap.add_argument("--diffusion_hidden_dims", default="512,512")
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--condition_injection", choices=["concat", "film"], default="concat")
    ap.add_argument("--film_hidden_dim", type=int, default=128)
    ap.add_argument("--diffusion_weight", type=float, default=1.0)
    ap.add_argument("--marginal_weight", type=float, default=0.0)
    ap.add_argument("--coarse_head_weight", type=float, default=0.0)
    ap.add_argument("--consistency_weight", type=float, default=0.0)
    ap.add_argument("--aux_t_gate", type=int, default=-1)
    ap.add_argument("--teacher_stage1_checkpoint", default=None)
    ap.add_argument("--distill_weight", type=float, default=0.0)
    ap.add_argument("--distill_temperature", type=float, default=1.0)
    ap.add_argument("--teacher_batch_size", type=int, default=4096)
    ap.add_argument("--predict_mode", choices=["diffusion", "head"], default="diffusion")
    ap.add_argument("--selection_metric", choices=["val_tvd_joint", "val_tvd_joint_raw", "val_combo"], default="val_tvd_joint")
    ap.add_argument("--selection_raw_weight", type=float, default=0.25)
    ap.add_argument("--logp_clip_quantile_low", type=float, default=-1.0)
    ap.add_argument("--logp_clip_quantile_high", type=float, default=-1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--val_min_groups", type=int, default=96)
    ap.add_argument("--n_val_joint_samples", type=int, default=64)
    ap.add_argument("--n_eval_joint_samples", type=int, default=64)
    ap.add_argument("--ipf_iters", type=int, default=200)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--save_best_checkpoint", action="store_true")
    ap.add_argument("--save_final_model", action="store_true")
    ap.add_argument("--run_label", default="external_c2f_full_earn_stage1_coarse_diffusion")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    torch = _require_torch()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    joint_csv = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    condition_csv = pathlib.Path(args.condition_csv).expanduser().resolve()
    schema_json = pathlib.Path(args.schema_json).expanduser().resolve() if args.schema_json else None
    condition_schema_json = pathlib.Path(args.condition_schema_json).expanduser().resolve() if args.condition_schema_json else None
    condition_extra_csv = pathlib.Path(args.condition_extra_csv).expanduser().resolve() if args.condition_extra_csv else None
    teacher_stage1_checkpoint = pathlib.Path(args.teacher_stage1_checkpoint).expanduser().resolve() if args.teacher_stage1_checkpoint else None
    for p in [joint_csv, condition_csv]:
        if not p.exists():
            raise SystemExit(f"path not found: {p}")
    if schema_json is not None and not schema_json.exists():
        raise SystemExit(f"path not found: {schema_json}")
    if condition_schema_json is not None and not condition_schema_json.exists():
        raise SystemExit(f"path not found: {condition_schema_json}")
    if condition_extra_csv is not None and not condition_extra_csv.exists():
        raise SystemExit(f"path not found: {condition_extra_csv}")
    if teacher_stage1_checkpoint is not None and not teacher_stage1_checkpoint.exists():
        raise SystemExit(f"path not found: {teacher_stage1_checkpoint}")

    run_id = f"_us_puma_{args.run_label}_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else (_REPO_ROOT / "outputs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    df, p_full, ids = _load_full_joint_wide(joint_wide_csv=joint_csv, schema_json=schema_json)
    p_coarse = np.asarray([coarse_from_full_flat(row) for row in p_full], dtype=np.float32)
    p_coarse = p_coarse / np.maximum(p_coarse.sum(axis=1, keepdims=True), 1e-12)
    x_log_all = np.log(np.clip(p_coarse, 0.0, None) + 1e-6).astype(np.float32)

    heldout_statefp = base._canon_statefp(args.heldout_statefp)
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

    fine_var_specs = _load_var_specs_from_schema(schema_json=schema_json)
    cond_var_specs = _load_condition_specs_from_schema(
        condition_schema_json=condition_schema_json,
        fallback_var_specs=fine_var_specs,
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
    ext_marg = base._augment_ext_marginals_from_cross(
        cond_raw=cond_raw,
        block_slices=block_slices,
        ext_marg=ext_marg,
    )
    missing_marg = [var for var in FULL_VARIABLE_ORDER if var not in ext_marg]
    if missing_marg:
        raise SystemExit(f"condition_csv/schema missing marginal target for fine variable(s): {missing_marg}")

    ext_coarse_marg: dict[str, np.ndarray] = {}
    for var in COARSE_VARIABLE_ORDER:
        rows = []
        for i in range(cond_raw.shape[0]):
            ext_row = {k: np.asarray(ext_marg[k][i], dtype=np.float64) for k in FULL_VARIABLE_ORDER}
            rows.append(_coarse_marginals_from_full_ext_row(ext_row)[COARSE_VARIABLE_ORDER.index(var)])
        ext_coarse_marg[var] = np.asarray(rows, dtype=np.float32)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    internal_by_fold: dict[str, Any] = {}
    saved_checkpoints: dict[str, list[str]] = {}
    selection_by_fold: dict[str, Any] = {}
    coarse_head_enabled = float(args.coarse_head_weight) > 0.0 or str(args.predict_mode) == "head"
    if float(args.distill_weight) > 0.0 and not coarse_head_enabled:
        raise SystemExit("distillation requires coarse head to be enabled")
    teacher_coarse_prob_all: np.ndarray | None = None
    if teacher_stage1_checkpoint is not None and float(args.distill_weight) > 0.0:
        teacher_model = _load_frozen_full_model_teacher(
            checkpoint_path=teacher_stage1_checkpoint,
            timesteps=int(args.timesteps),
            seed=int(args.seed),
            device=str(device),
        )
        teacher_coarse_prob_all = _predict_teacher_coarse_probs(
            teacher_model=teacher_model,
            cond_raw=cond_raw,
            device=str(device),
            batch_size=int(args.teacher_batch_size),
        )
        teacher_model = None

    for fold_name, train_idx, test_idx in folds:
        train_core_idx, val_idx = base._split_train_val_indices(
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

        model = SharedLatentCoarseDiffusion(
            input_dim=int(COARSE_K),
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
            enable_coarse_head=coarse_head_enabled,
            coarse_predict_mode=str(args.predict_mode),
        )
        model.to(device)
        model.set_predict_meta(
            x_mean=x_mean,
            x_std=x_std,
            logp_clip_lo=logp_clip_lo,
            logp_clip_hi=logp_clip_hi,
            n_draws=int(args.n_eval_joint_samples),
        )
        ema = _ModuleEMA(model._modules, decay=float(args.ema_decay))

        x_train_t = torch.from_numpy(x_train).to(device=device, dtype=torch.float32)
        cond_train_t = torch.from_numpy(cond_raw[train_core_idx]).to(device=device, dtype=torch.float32)
        p_coarse_train_t = torch.from_numpy(p_coarse[train_core_idx].astype(np.float32)).to(device=device, dtype=torch.float32)
        teacher_coarse_train_t = (
            torch.from_numpy(teacher_coarse_prob_all[train_core_idx].astype(np.float32)).to(device=device, dtype=torch.float32)
            if teacher_coarse_prob_all is not None
            else None
        )
        marginal_targets_train_t = tuple(
            torch.from_numpy(ext_coarse_marg[var][train_core_idx].astype(np.float32)).to(device=device)
            for var in COARSE_VARIABLE_ORDER
        )
        x_mean_t = torch.from_numpy(x_mean.reshape(1, -1)).to(device=device)
        x_std_t = torch.from_numpy(x_std.reshape(1, -1)).to(device=device)
        logp_clip_lo_t = torch.from_numpy(logp_clip_lo.reshape(1, -1)).to(device=device) if logp_clip_lo is not None else None
        logp_clip_hi_t = torch.from_numpy(logp_clip_hi.reshape(1, -1)).to(device=device) if logp_clip_hi is not None else None

        train_metrics: list[dict[str, float]] = []
        n_train = int(train_core_idx.size)
        bs = int(args.batch_size)
        eval_every = max(1, int(args.eval_every))
        checkpoint_payload = {
            "input_dim": int(COARSE_K),
            "cond_raw_dim": int(cond_raw.shape[1]),
            "latent_dim": int(args.latent_dim),
            "encoder_hidden_dims": list(_parse_hidden_dims(args.encoder_hidden_dims)),
            "coarse_hidden_dims": list(_parse_hidden_dims(args.coarse_hidden_dims)),
            "diffusion_hidden_dims": list(_parse_hidden_dims(args.diffusion_hidden_dims)),
            "condition_injection": str(args.condition_injection),
            "film_hidden_dim": int(args.film_hidden_dim),
            "condition_scale_mode": str(args.condition_scale_mode),
            "condition_extra_csv": str(condition_extra_csv) if condition_extra_csv is not None else None,
            "condition_extra_standardize": str(args.condition_extra_standardize),
            "condition_extra_missing_policy": str(args.condition_extra_missing_policy),
            "condition_meta": cond_meta,
            "diffusion_weight": float(args.diffusion_weight),
            "enable_coarse_head": bool(coarse_head_enabled),
            "coarse_head_weight": float(args.coarse_head_weight),
            "consistency_weight": float(args.consistency_weight),
            "aux_t_gate": int(args.aux_t_gate),
            "teacher_stage1_checkpoint": str(teacher_stage1_checkpoint) if teacher_stage1_checkpoint is not None else None,
            "distill_weight": float(args.distill_weight),
            "distill_temperature": float(args.distill_temperature),
            "teacher_batch_size": int(args.teacher_batch_size),
            "coarse_predict_mode": str(args.predict_mode),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "coarse_preset": str(COARSE_PRESET),
            "coarse_shape": list(COARSE_SHAPE),
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "logp_clip_lo": None if logp_clip_lo is None else logp_clip_lo.tolist(),
            "logp_clip_hi": None if logp_clip_hi is None else logp_clip_hi.tolist(),
            "predict_n_draws": int(args.n_eval_joint_samples),
        }
        best_val_metric = float("inf")
        best_epoch: int | None = None
        best_state: dict[str, Any] | None = None
        best_val_summary: dict[str, Any] | None = None
        best_source = "final_raw"
        for epoch in range(1, int(args.epochs) + 1):
            order = np.random.permutation(n_train)
            last_stats: dict[str, float] | None = None
            for start in range(0, n_train, bs):
                idx = order[start : start + bs]
                idx_t = torch.from_numpy(idx).to(device=device, dtype=torch.long)
                last_stats = model.step(
                    x0=x_train_t[idx_t],
                    cond_raw=cond_train_t[idx_t],
                    p_coarse_true=p_coarse_train_t[idx_t],
                    marginal_targets=tuple(mt[idx_t] for mt in marginal_targets_train_t),
                    x_mean=x_mean_t,
                    x_std=x_std_t,
                    logp_clip_lo=logp_clip_lo_t,
                    logp_clip_hi=logp_clip_hi_t,
                    diffusion_weight=float(args.diffusion_weight),
                    marginal_weight=float(args.marginal_weight),
                    coarse_head_weight=float(args.coarse_head_weight),
                    consistency_weight=float(args.consistency_weight),
                    aux_t_gate=int(args.aux_t_gate),
                    teacher_coarse_prob=teacher_coarse_train_t[idx_t] if teacher_coarse_train_t is not None else None,
                    distill_weight=float(args.distill_weight),
                    distill_temperature=float(args.distill_temperature),
                )
                ema.update(model._modules)

            should_log = epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(args.epochs)
            should_eval = val_idx.size > 0 and (epoch == 1 or epoch % eval_every == 0 or epoch == int(args.epochs))
            if last_stats is not None and should_log:
                rec = {"epoch": float(epoch), **last_stats}
                if should_eval:
                    with ema.apply(model._modules):
                        val_summary = _evaluate_coarse_distribution(
                            model=model,
                            eval_idx=val_idx,
                            reference_train_idx=train_core_idx,
                            p_coarse=p_coarse,
                            cond_raw=cond_raw,
                            ext_coarse_marg=ext_coarse_marg,
                            device=device,
                            ipf_iters=int(args.ipf_iters),
                        )
                    rec["val_tvd_joint"] = float(val_summary["tvd_joint"]["mean"])
                    rec["val_tvd_joint_raw"] = float(val_summary["tvd_joint_raw"]["mean"])
                    if str(args.selection_metric) == "val_tvd_joint":
                        selection_metric = rec["val_tvd_joint"]
                    elif str(args.selection_metric) == "val_tvd_joint_raw":
                        selection_metric = rec["val_tvd_joint_raw"]
                    elif str(args.selection_metric) == "val_combo":
                        selection_metric = rec["val_tvd_joint"] + float(args.selection_raw_weight) * rec["val_tvd_joint_raw"]
                    else:
                        raise ValueError(f"unsupported selection_metric: {args.selection_metric}")
                    rec["selection_metric"] = float(selection_metric)
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
                                payload={**checkpoint_payload, "best_epoch": int(epoch)},
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
                    f"kd={rec['loss_distill']:.6f} "
                    f"marg={rec['loss_marginal']:.6f}"
                )

        saved_checkpoints.setdefault(fold_name, [])
        if bool(args.save_final_model):
            ckpt = out_dir / "checkpoints" / fold_name / "final.pt"
            model.save(ckpt, payload=checkpoint_payload)
            saved_checkpoints[fold_name].append(str(ckpt))

        if best_state is not None:
            model._modules.load_state_dict(best_state, strict=True)
        elif ema.enabled:
            model._modules.load_state_dict(ema.cpu_state_dict(), strict=True)
            best_source = "ema_final"

        fold_summary = _evaluate_coarse_distribution(
            model=model,
            eval_idx=test_idx,
            reference_train_idx=train_core_idx,
            p_coarse=p_coarse,
            cond_raw=cond_raw,
            ext_coarse_marg=ext_coarse_marg,
            device=device,
            ipf_iters=int(args.ipf_iters),
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
            "selection_metric": str(args.selection_metric),
            "selected_state": best_source,
            "ema_decay": float(args.ema_decay),
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
        "ipf_train_seed_external": {
            "tvd_joint": _summ([float(internal_by_fold[f]["ipf_train_seed_external"]["tvd_joint"]["mean"]) for f in fold_names]),
        },
        "independence_external": {
            "tvd_joint": _summ([float(internal_by_fold[f]["independence_external"]["tvd_joint"]["mean"]) for f in fold_names]),
        },
    }
    for var in COARSE_VARIABLE_ORDER:
        overall[f"tvd_{var}"] = _summ([float(internal_by_fold[f][f"tvd_{var}"]["mean"]) for f in fold_names])
        overall[f"tvd_{var}_raw"] = _summ([float(internal_by_fold[f][f"tvd_{var}_raw"]["mean"]) for f in fold_names])

    run_summary = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "joint_wide_csv": str(joint_csv),
        "condition_csv": str(condition_csv),
        "schema_json": str(schema_json) if schema_json else None,
        "condition_schema_json": str(condition_schema_json) if condition_schema_json else None,
        "condition_scale_mode": str(args.condition_scale_mode),
        "condition_extra_csv": str(condition_extra_csv) if condition_extra_csv is not None else None,
        "condition_extra_standardize": str(args.condition_extra_standardize),
        "condition_extra_missing_policy": str(args.condition_extra_missing_policy),
        "condition_meta": cond_meta,
        "run_label": str(args.run_label),
        "eval_mode": str(args.eval_mode),
        "heldout_statefp": str(heldout_statefp),
        "timesteps": int(args.timesteps),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "latent_dim": int(args.latent_dim),
        "encoder_hidden_dims": list(_parse_hidden_dims(args.encoder_hidden_dims)),
        "coarse_hidden_dims": list(_parse_hidden_dims(args.coarse_hidden_dims)),
        "diffusion_hidden_dims": list(_parse_hidden_dims(args.diffusion_hidden_dims)),
        "condition_injection": str(args.condition_injection),
        "film_hidden_dim": int(args.film_hidden_dim),
        "diffusion_weight": float(args.diffusion_weight),
        "marginal_weight": float(args.marginal_weight),
        "coarse_head_weight": float(args.coarse_head_weight),
        "consistency_weight": float(args.consistency_weight),
        "aux_t_gate": int(args.aux_t_gate),
        "teacher_stage1_checkpoint": str(teacher_stage1_checkpoint) if teacher_stage1_checkpoint is not None else None,
        "distill_weight": float(args.distill_weight),
        "distill_temperature": float(args.distill_temperature),
        "teacher_batch_size": int(args.teacher_batch_size),
        "predict_mode": str(args.predict_mode),
        "enable_coarse_head": bool(coarse_head_enabled),
        "selection_metric": str(args.selection_metric),
        "selection_raw_weight": float(args.selection_raw_weight),
        "logp_clip_quantile_low": float(args.logp_clip_quantile_low),
        "logp_clip_quantile_high": float(args.logp_clip_quantile_high),
        "ema_decay": float(args.ema_decay),
        "coarse_preset": str(COARSE_PRESET),
        "coarse_shape": list(COARSE_SHAPE),
        "coarse_k": int(COARSE_K),
        "cond_raw_dim": int(cond_raw.shape[1]),
        "n_regions_total": int(len(ids)),
        "folds": fold_names,
        "selection_by_fold": selection_by_fold,
        "saved_checkpoints": saved_checkpoints,
        "results": overall,
    }
    _write_json(out_dir / "run_summary.json", run_summary)
    print(f"[ok] wrote: {out_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
