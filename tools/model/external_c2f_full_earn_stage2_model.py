#!/usr/bin/env python3
from __future__ import annotations

import pathlib
from dataclasses import asdict
from typing import Any

import numpy as np

from src.synthpop.model.diffusion_tabular import (
    DiffusionTabularModel,
    TabDDPMConfig,
    _DenoiserMLP,
    _FiLMDenoiserMLP,
    _require_torch,
)


def _softmax_rows_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.maximum(ex.sum(axis=1, keepdims=True), 1e-12)


def _masked_log_softmax(logits: Any, mask: Any) -> Any:
    torch = _require_torch()
    large_neg = torch.full_like(logits, -1e9)
    masked_logits = torch.where(mask > 0.5, logits, large_neg)
    logp = torch.log_softmax(masked_logits, dim=1)
    return torch.where(mask > 0.5, logp, large_neg)


def _aggregate_regime_mass_torch(*, local_prob: Any, regime_index: Any, regime_dim: int, regime_mask: Any | None) -> Any:
    torch = _require_torch()
    out = torch.zeros((local_prob.shape[0], int(regime_dim)), device=local_prob.device, dtype=local_prob.dtype)
    valid = regime_index >= 0
    for regime_idx in range(int(regime_dim)):
        slot_mask = valid & (regime_index == int(regime_idx))
        if bool(slot_mask.any()):
            out[:, regime_idx] = (local_prob * slot_mask.to(dtype=local_prob.dtype)).sum(dim=1)
    if regime_mask is not None:
        out = out * regime_mask.to(device=local_prob.device, dtype=local_prob.dtype)
    out = out / torch.clamp(out.sum(dim=1, keepdim=True), min=1e-12)
    return out


def _aggregate_regime_mass_np(*, local_prob: np.ndarray, regime_index: np.ndarray, regime_dim: int, regime_mask: np.ndarray | None) -> np.ndarray:
    local_prob = np.asarray(local_prob, dtype=np.float64)
    regime_index = np.asarray(regime_index, dtype=np.int64)
    out = np.zeros((local_prob.shape[0], int(regime_dim)), dtype=np.float64)
    valid = regime_index >= 0
    for regime_idx in range(int(regime_dim)):
        slot_mask = valid & (regime_index == int(regime_idx))
        if bool(np.any(slot_mask)):
            out[:, regime_idx] = np.where(slot_mask, local_prob, 0.0).sum(axis=1)
    if regime_mask is not None:
        out *= np.asarray(regime_mask, dtype=np.float64)
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out


def _project_local_to_regime_np(
    *,
    local_prob: np.ndarray,
    regime_target: np.ndarray,
    regime_index: np.ndarray,
    child_mask: np.ndarray,
    regime_dim: int,
) -> np.ndarray:
    local_prob = np.asarray(local_prob, dtype=np.float64)
    regime_target = np.asarray(regime_target, dtype=np.float64)
    regime_index = np.asarray(regime_index, dtype=np.int64)
    child_mask = np.asarray(child_mask, dtype=np.float64)
    out = np.zeros_like(local_prob, dtype=np.float64)
    valid = child_mask > 0.5
    for regime_idx in range(int(regime_dim)):
        slot_mask = valid & (regime_index == int(regime_idx))
        if not bool(np.any(slot_mask)):
            continue
        base_mass = np.where(slot_mask, local_prob, 0.0).sum(axis=1, keepdims=True)
        scale = regime_target[:, [regime_idx]] / np.maximum(base_mass, 1e-12)
        out += np.where(slot_mask, local_prob * scale, 0.0)
    out *= valid.astype(np.float64)
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out


class SharedConditionStage2Diffusion:
    def __init__(
        self,
        *,
        input_dim: int,
        cond_raw_dim: int,
        latent_dim: int,
        encoder_hidden_dims: tuple[int, ...],
        head_hidden_dims: tuple[int, ...],
        diffusion_config: TabDDPMConfig,
        seed: int,
        enable_clean_head: bool,
        aux_regime_dim: int,
        predict_mode: str,
        blend_alpha: float,
        regime_projection_alpha: float,
    ) -> None:
        torch = _require_torch()
        nn = torch.nn
        torch.manual_seed(int(seed))

        self.input_dim = int(input_dim)
        self.cond_raw_dim = int(cond_raw_dim)
        self.latent_dim = int(latent_dim)
        self.encoder_hidden_dims = tuple(int(x) for x in encoder_hidden_dims)
        self.head_hidden_dims = tuple(int(x) for x in head_hidden_dims)
        self.config = diffusion_config
        self.seed = int(seed)
        self.enable_clean_head = bool(enable_clean_head)
        self.aux_regime_dim = int(aux_regime_dim)
        self.predict_mode = str(predict_mode)
        self.blend_alpha = float(blend_alpha)
        self.regime_projection_alpha = float(regime_projection_alpha)
        self._schedule: dict[str, Any] | None = None

        self.encoder = self._make_mlp(
            in_dim=self.cond_raw_dim,
            hidden_dims=self.encoder_hidden_dims,
            out_dim=self.latent_dim,
            nn=nn,
        )
        if self.enable_clean_head:
            self.head_feature = self._make_mlp(
                in_dim=self.latent_dim,
                hidden_dims=self.head_hidden_dims,
                out_dim=self.latent_dim,
                nn=nn,
            )
            self.head_out = nn.Linear(self.latent_dim, self.input_dim)
        else:
            self.head_feature = None
            self.head_out = None
        if self.aux_regime_dim > 0:
            self.regime_feature = self._make_mlp(
                in_dim=self.latent_dim,
                hidden_dims=self.head_hidden_dims,
                out_dim=self.latent_dim,
                nn=nn,
            )
            self.regime_out = nn.Linear(self.latent_dim, self.aux_regime_dim)
        else:
            self.regime_feature = None
            self.regime_out = None

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

        modules: list[Any] = [self.encoder, self.denoiser]
        if self.enable_clean_head:
            modules.extend([self.head_feature, self.head_out])
        if self.aux_regime_dim > 0:
            modules.extend([self.regime_feature, self.regime_out])
        self._modules = nn.ModuleList(modules)
        self._opt = torch.optim.AdamW(
            self._modules.parameters(),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )

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

    def save(self, path: pathlib.Path) -> None:
        torch = _require_torch()
        p = pathlib.Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "synthpop.external_c2f_full_earn_teacher.v1",
            "input_dim": int(self.input_dim),
            "cond_raw_dim": int(self.cond_raw_dim),
            "latent_dim": int(self.latent_dim),
            "encoder_hidden_dims": list(self.encoder_hidden_dims),
            "head_hidden_dims": list(self.head_hidden_dims),
            "enable_clean_head": bool(self.enable_clean_head),
            "aux_regime_dim": int(self.aux_regime_dim),
            "predict_mode": str(self.predict_mode),
            "blend_alpha": float(self.blend_alpha),
            "regime_projection_alpha": float(self.regime_projection_alpha),
            "seed": int(self.seed),
            "config": asdict(self.config),
            "state_dict": self._modules.state_dict(),
        }
        torch.save(payload, p)

    def _encode(self, cond_raw: Any) -> Any:
        return self.encoder(cond_raw)

    def _head_probs(self, *, z: Any, child_mask: Any) -> Any:
        torch = _require_torch()
        if not self.enable_clean_head or self.head_feature is None or self.head_out is None:
            return torch.zeros_like(child_mask, dtype=torch.float32)
        head_logits = self.head_out(self.head_feature(z))
        head_logp = _masked_log_softmax(head_logits, child_mask)
        return torch.exp(head_logp)

    def _regime_probs(self, *, z: Any, regime_mask: Any) -> Any:
        torch = _require_torch()
        if self.aux_regime_dim <= 0 or self.regime_feature is None or self.regime_out is None:
            return torch.zeros_like(regime_mask, dtype=torch.float32)
        regime_logits = self.regime_out(self.regime_feature(z))
        regime_logp = _masked_log_softmax(regime_logits, regime_mask)
        return torch.exp(regime_logp)

    def _merge_prediction_probs(self, *, p_diff: Any, head_prob: Any | None) -> Any:
        torch = _require_torch()
        mode = str(self.predict_mode).lower().strip()
        if mode == "diffusion" or head_prob is None:
            return p_diff
        if mode == "head":
            return head_prob
        if mode == "blend":
            alpha = float(np.clip(self.blend_alpha, 0.0, 1.0))
            out = (1.0 - alpha) * p_diff + alpha * head_prob
            out = out / torch.clamp(out.sum(dim=1, keepdim=True), min=1e-12)
            return out
        raise ValueError(f"Unsupported predict_mode: {self.predict_mode}")

    def step(
        self,
        *,
        x0: Any,
        cond_raw: Any,
        p_true: Any,
        child_mask: Any,
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
    ) -> dict[str, float]:
        torch = _require_torch()
        assert self._schedule is not None
        self.train()

        z = self._encode(cond_raw)
        head_prob = None
        if self.enable_clean_head:
            head_prob = self._head_probs(z=z, child_mask=child_mask)
            loss_head = -(p_true * torch.log(torch.clamp(head_prob, min=1e-12))).sum(dim=1).mean()
        else:
            loss_head = torch.zeros((), device=x0.device)

        t = torch.randint(0, self.config.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        sqrt_acp = self._schedule["sqrt_alpha_cumprod"][t].unsqueeze(1)
        sqrt_om = self._schedule["sqrt_one_minus_alpha_cumprod"][t].unsqueeze(1)
        x_t = sqrt_acp * x0 + sqrt_om * noise
        eps_pred = self.denoiser(x_t, t, z)

        if float(diff_loss_reweight_alpha) > 0.0:
            active_count = torch.clamp(child_mask.sum(dim=1, keepdim=True), min=1.0)
            support = child_mask / active_count
            cell_weight = torch.pow(torch.clamp(support, min=1e-8), float(diff_loss_reweight_alpha))
            cell_weight = cell_weight / torch.clamp(cell_weight.mean(dim=1, keepdim=True), min=1e-8)
            cell_weight = torch.clamp(
                cell_weight,
                min=float(diff_loss_reweight_floor),
                max=float(diff_loss_reweight_cap),
            )
            loss_diff = (cell_weight * torch.square(eps_pred - noise)).mean()
            active_mask = child_mask > 0.5
            inactive_mask = ~active_mask
            if bool(active_mask.any()):
                active_weight_mean = float(cell_weight[active_mask].mean().detach().cpu().item())
            else:
                active_weight_mean = float("nan")
            if bool(inactive_mask.any()):
                inactive_weight_mean = float(cell_weight[inactive_mask].mean().detach().cpu().item())
            else:
                inactive_weight_mean = float("nan")
            cell_weight_mean = float(cell_weight.mean().detach().cpu().item())
        else:
            loss_diff = torch.nn.functional.mse_loss(eps_pred, noise)
            cell_weight_mean = float("nan")
            active_weight_mean = float("nan")
            inactive_weight_mean = float("nan")

        x0_pred = (x_t - sqrt_om * eps_pred) / torch.clamp(sqrt_acp, min=1e-12)
        logp_pred = x0_pred * x_std + x_mean
        p_pred = torch.softmax(logp_pred, dim=1)
        p_pred_proj = p_pred * child_mask
        p_pred_proj = p_pred_proj / torch.clamp(p_pred_proj.sum(dim=1, keepdim=True), min=1e-12)

        if int(aux_t_gate) >= 0:
            aux_mask = (t <= int(aux_t_gate)).to(dtype=p_pred.dtype)
        else:
            aux_mask = torch.ones_like(t, dtype=p_pred.dtype)
        aux_denom = torch.clamp(aux_mask.sum(), min=1.0)
        base_prob = self._merge_prediction_probs(p_diff=p_pred_proj, head_prob=head_prob)
        if head_prob is not None and float(consistency_weight) > 0.0:
            loss_cons_per = 0.5 * torch.abs(p_pred_proj - head_prob).sum(dim=1)
            loss_cons = (loss_cons_per * aux_mask).sum() / aux_denom
        else:
            loss_cons = torch.zeros((), device=x0.device)

        if (
            self.aux_regime_dim > 0
            and income_regime_target is not None
            and income_regime_mask is not None
            and float(income_regime_weight) > 0.0
        ):
            regime_prob = self._regime_probs(z=z, regime_mask=income_regime_mask)
            loss_regime = -(income_regime_target * torch.log(torch.clamp(regime_prob, min=1e-12))).sum(dim=1).mean()
        else:
            loss_regime = torch.zeros((), device=x0.device)

        if (
            income_regime_slot is not None
            and income_regime_target is not None
            and income_regime_mask is not None
            and float(income_regime_consistency_weight) > 0.0
        ):
            regime_from_base = _aggregate_regime_mass_torch(
                local_prob=base_prob,
                regime_index=income_regime_slot.to(device=x0.device, dtype=torch.long),
                regime_dim=int(self.aux_regime_dim if self.aux_regime_dim > 0 else income_regime_target.shape[1]),
                regime_mask=income_regime_mask,
            )
            loss_regime_cons_per = -(income_regime_target * torch.log(torch.clamp(regime_from_base, min=1e-12))).sum(dim=1)
            loss_regime_cons = (loss_regime_cons_per * aux_mask).sum() / aux_denom
        else:
            loss_regime_cons = torch.zeros((), device=x0.device)

        loss = (
            loss_diff
            + float(clean_head_weight) * loss_head
            + float(consistency_weight) * loss_cons
            + float(income_regime_weight) * loss_regime
            + float(income_regime_consistency_weight) * loss_regime_cons
        )
        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._modules.parameters(), float(self.config.grad_clip))
        self._opt.step()
        return {
            "loss": float(loss.detach().cpu()),
            "loss_diffusion": float(loss_diff.detach().cpu()),
            "loss_clean_head": float(loss_head.detach().cpu()),
            "loss_consistency": float(loss_cons.detach().cpu()),
            "loss_income_regime": float(loss_regime.detach().cpu()),
            "loss_income_regime_consistency": float(loss_regime_cons.detach().cpu()),
            "aux_mask_frac": float(aux_mask.mean().detach().cpu()),
            "cell_weight_mean": float(cell_weight_mean),
            "cell_weight_active_mean": float(active_weight_mean),
            "cell_weight_inactive_mean": float(inactive_weight_mean),
        }

    def _sample_diffusion_raw(
        self,
        *,
        z_cond: Any,
        n_draws: int,
        device: Any,
        x_mean: np.ndarray,
        x_std: np.ndarray,
    ) -> np.ndarray:
        torch = _require_torch()
        assert self._schedule is not None
        self.eval()
        n_rows = int(z_cond.shape[0])
        n_total = int(n_rows * n_draws)
        z_rep = z_cond.repeat_interleave(int(n_draws), dim=0)
        with torch.inference_mode():
            x_t = torch.randn((n_total, self.input_dim), device=device)
            betas = self._schedule["betas"]
            alphas = self._schedule["alphas"]
            posterior_variance = self._schedule["posterior_variance"]
            sqrt_om_all = self._schedule["sqrt_one_minus_alpha_cumprod"]
            for step in reversed(range(int(self.config.timesteps))):
                t = torch.full((n_total,), int(step), device=device, dtype=torch.long)
                eps_pred = self.denoiser(x_t, t, z_rep)
                beta_t = betas[step]
                alpha_t = alphas[step]
                sqrt_om = sqrt_om_all[step]
                model_mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_om) * eps_pred)
                if step == 0:
                    x_t = model_mean
                    continue
                noise = torch.randn_like(x_t)
                x_t = model_mean + torch.sqrt(posterior_variance[step]) * noise
            z_np = x_t.detach().cpu().numpy().reshape((n_rows, int(n_draws), self.input_dim))

        logp = z_np.astype(np.float64) * x_std.reshape((1, 1, -1)).astype(np.float64) + x_mean.reshape((1, 1, -1)).astype(np.float64)
        p_draws = _softmax_rows_np(logp.reshape((-1, logp.shape[-1]))).reshape((n_rows, int(n_draws), -1))
        p_hat = np.mean(p_draws, axis=1)
        p_hat = p_hat / np.maximum(p_hat.sum(axis=1, keepdims=True), 1e-12)
        return p_hat.astype(np.float64)

    def sample_local_raw(
        self,
        *,
        cond_raw: np.ndarray,
        child_mask: np.ndarray,
        regime_index_rows: np.ndarray | None,
        n_draws: int,
        device: str | None,
        x_mean: np.ndarray,
        x_std: np.ndarray,
    ) -> np.ndarray:
        torch = _require_torch()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        cond_t = torch.from_numpy(np.asarray(cond_raw, dtype=np.float32)).to(device=device, dtype=torch.float32)
        mask_t = torch.from_numpy(np.asarray(child_mask, dtype=np.float32)).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            z = self._encode(cond_t)
            p_diff = self._sample_diffusion_raw(
                z_cond=z,
                n_draws=int(n_draws),
                device=device,
                x_mean=np.asarray(x_mean, dtype=np.float32),
                x_std=np.asarray(x_std, dtype=np.float32),
            )
            if not self.enable_clean_head or self.head_feature is None or self.head_out is None:
                out = p_diff
            else:
                p_head = self._head_probs(z=z, child_mask=mask_t).detach().cpu().numpy().astype(np.float64)
                mode = str(self.predict_mode).lower().strip()
                if mode == "diffusion":
                    out = p_diff
                elif mode == "head":
                    out = p_head
                elif mode == "blend":
                    alpha = float(np.clip(self.blend_alpha, 0.0, 1.0))
                    out = (1.0 - alpha) * p_diff + alpha * p_head
                    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
                else:
                    raise ValueError(f"Unsupported predict_mode: {self.predict_mode}")
            if (
                self.aux_regime_dim > 0
                and regime_index_rows is not None
                and self.regime_feature is not None
                and self.regime_out is not None
                and float(self.regime_projection_alpha) > 0.0
            ):
                regime_mask_np = np.zeros((out.shape[0], self.aux_regime_dim), dtype=np.float64)
                regime_index_np = np.asarray(regime_index_rows, dtype=np.int64)
                valid = regime_index_np >= 0
                for regime_idx in range(int(self.aux_regime_dim)):
                    regime_mask_np[:, regime_idx] = np.any(valid & (regime_index_np == int(regime_idx)), axis=1).astype(np.float64)
                regime_prob = self._regime_probs(
                    z=z,
                    regime_mask=torch.from_numpy(regime_mask_np).to(device=device, dtype=torch.float32),
                ).detach().cpu().numpy().astype(np.float64)
                base_regime = _aggregate_regime_mass_np(
                    local_prob=out,
                    regime_index=regime_index_np,
                    regime_dim=int(self.aux_regime_dim),
                    regime_mask=regime_mask_np,
                )
                alpha = float(np.clip(self.regime_projection_alpha, 0.0, 1.0))
                target_regime = (1.0 - alpha) * base_regime + alpha * regime_prob
                target_regime = target_regime / np.maximum(target_regime.sum(axis=1, keepdims=True), 1e-12)
                out = _project_local_to_regime_np(
                    local_prob=out,
                    regime_target=target_regime,
                    regime_index=regime_index_np,
                    child_mask=np.asarray(child_mask, dtype=np.float64),
                    regime_dim=int(self.aux_regime_dim),
                )
            return out.astype(np.float64)


def load_stage2_model(*, checkpoint_path: pathlib.Path) -> tuple[Any, dict[str, Any]]:
    torch = _require_torch()
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported stage2 checkpoint format: {checkpoint_path}")
    ckpt_format = str(payload.get("format", ""))
    if ckpt_format == "synthpop.tabddpm.v0":
        model = DiffusionTabularModel(
            input_dim=int(payload["input_dim"]),
            cond_dim=int(payload.get("cond_dim", 0)),
            seed=int(payload.get("seed", 0)),
            config=TabDDPMConfig(**dict(payload.get("config", {}))),
        )
        model.load(checkpoint_path)
        return model, payload
    if ckpt_format == "synthpop.external_c2f_full_earn_teacher.v1":
        model = SharedConditionStage2Diffusion(
            input_dim=int(payload["input_dim"]),
            cond_raw_dim=int(payload["cond_raw_dim"]),
            latent_dim=int(payload["latent_dim"]),
            encoder_hidden_dims=tuple(int(x) for x in payload.get("encoder_hidden_dims", (256, 256))),
            head_hidden_dims=tuple(int(x) for x in payload.get("head_hidden_dims", (256,))),
            diffusion_config=TabDDPMConfig(**dict(payload.get("config", {}))),
            seed=int(payload.get("seed", 0)),
            enable_clean_head=bool(payload.get("enable_clean_head", False)),
            aux_regime_dim=int(payload.get("aux_regime_dim", 0)),
            predict_mode=str(payload.get("predict_mode", "diffusion")),
            blend_alpha=float(payload.get("blend_alpha", 0.0)),
            regime_projection_alpha=float(payload.get("regime_projection_alpha", 0.0)),
        )
        model._modules.load_state_dict(payload["state_dict"], strict=True)
        return model, payload
    raise ValueError(f"Unsupported stage2 checkpoint format: {checkpoint_path}")


def sample_stage2_local_raw_batch(
    *,
    model: Any,
    cond_rows: np.ndarray,
    child_mask_rows: np.ndarray,
    regime_index_rows: np.ndarray | None = None,
    n_draws: int,
    device: str | None,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> np.ndarray:
    if hasattr(model, "sample_local_raw"):
        return model.sample_local_raw(
            cond_raw=np.asarray(cond_rows, dtype=np.float32),
            child_mask=np.asarray(child_mask_rows, dtype=np.float32),
            regime_index_rows=None if regime_index_rows is None else np.asarray(regime_index_rows, dtype=np.int64),
            n_draws=int(n_draws),
            device=device,
            x_mean=np.asarray(x_mean, dtype=np.float32),
            x_std=np.asarray(x_std, dtype=np.float32),
        )

    torch = _require_torch()
    cond_rep = np.repeat(np.asarray(cond_rows, dtype=np.float32), repeats=int(n_draws), axis=0)
    z = model.sample(n=int(cond_rep.shape[0]), cond=torch.from_numpy(cond_rep), device=device).numpy()
    n_parent = int(cond_rows.shape[0])
    z = z.reshape((n_parent, int(n_draws), -1))
    logp = z.astype(np.float64) * np.asarray(x_std, dtype=np.float64).reshape((1, 1, -1)) + np.asarray(x_mean, dtype=np.float64).reshape((1, 1, -1))
    p_draws = _softmax_rows_np(logp.reshape((-1, logp.shape[-1]))).reshape((n_parent, int(n_draws), -1))
    p_hat = np.mean(p_draws, axis=1)
    p_hat = p_hat / np.maximum(p_hat.sum(axis=1, keepdims=True), 1e-12)
    return p_hat.astype(np.float64)
