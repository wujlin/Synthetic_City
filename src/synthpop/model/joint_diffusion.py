from __future__ import annotations

"""
Joint latent diffusion model (Scheme C-v2).

KISS implementation strategy (v0):
- Reuse the existing Gaussian TabDDPM implementation on a *concatenated* latent vector:
    x = concat(z_person, z_building)
- Keep the interface explicit so we can later swap in a dedicated joint diffusion + guidance sampler.
"""

from dataclasses import dataclass
from typing import Any

from .diffusion_tabular import DiffusionTabularModel, TabDDPMConfig


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("joint_diffusion.py requires PyTorch.") from e
    return torch


@dataclass(frozen=True)
class JointDiffusionConfig:
    latent_dim: int = 32
    cond_dim: int = 0
    seed: int = 0
    tabddpm: TabDDPMConfig = TabDDPMConfig()


class JointDiffusionModel:
    """
    Joint diffusion over (z_person, z_building).

    Notes:
    - z_person and z_building are assumed to share the same latent dimension.
    - The underlying model operates on a continuous vector and does not enforce semantic constraints.
    """

    def __init__(self, *, config: JointDiffusionConfig) -> None:
        if config.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if config.cond_dim < 0:
            raise ValueError("cond_dim must be >= 0")

        self.config = config
        self.latent_dim = int(config.latent_dim)
        self.cond_dim = int(config.cond_dim)
        self.seed = int(config.seed)

        self._base = DiffusionTabularModel(
            input_dim=2 * self.latent_dim,
            cond_dim=self.cond_dim,
            seed=self.seed,
            config=config.tabddpm,
        )

    def fit(
        self,
        *,
        z_person: Any,
        z_building: Any,
        cond: Any | None = None,
        epochs: int = 5,
        batch_size: int = 2048,
        device: str | None = None,
        log_every: int = 200,
    ) -> dict[str, float]:
        torch = _require_torch()
        x = torch.cat([z_person, z_building], dim=1)
        return self._base.fit(x=x, cond=cond, epochs=epochs, batch_size=batch_size, device=device, log_every=log_every)

    def sample(
        self,
        *,
        n: int,
        cond: Any | None = None,
        device: str | None = None,
        target_marginals: dict[str, Any] | None = None,
        guidance_scale: float = 0.0,
        guidance_schedule: str = "none",
        guidance_recompute_eps: bool = True,
    ) -> tuple[Any, Any]:
        """
        Sample (z_person, z_building).

        v0:
        - When target_marginals is provided and guidance_scale>0, apply distribution guidance at each denoising step.
        - Guidance only affects the *sample trajectory* (x_t); model weights are not updated during sampling.
        """
        torch = _require_torch()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Validate condition shape if used.
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond is required when cond_dim>0")
            cond = torch.as_tensor(cond, device=device, dtype=torch.float32)
            if cond.ndim != 2 or cond.shape[1] != self.cond_dim or cond.shape[0] != int(n):
                raise ValueError(f"cond must be (N,{self.cond_dim}) where N==n, got {tuple(cond.shape)}")
        else:
            cond = None

        use_guidance = target_marginals is not None and float(guidance_scale) > 0.0
        if guidance_schedule not in ("none", "linear"):
            raise ValueError("guidance_schedule must be one of: none, linear")

        # Initialize underlying model/schedule.
        self._base._init_model(device=device)  # noqa: SLF001 (internal use ok within package)
        net = self._base._net
        schedule = self._base._schedule
        if net is None or schedule is None:
            raise RuntimeError("Diffusion model is not initialized.")
        net.eval()

        from ..constraints.soft_guidance import distribution_guidance_step

        n = int(n)
        timesteps = int(self._base.config.timesteps)
        betas = schedule["betas"]
        alphas = schedule["alphas"]
        posterior_variance = schedule["posterior_variance"]
        sqrt_alpha_cumprod = schedule["sqrt_alpha_cumprod"]
        sqrt_one_minus_alpha_cumprod = schedule["sqrt_one_minus_alpha_cumprod"]

        x_t = torch.randn((n, 2 * self.latent_dim), device=device)
        for step in reversed(range(timesteps)):
            t = torch.full((n,), step, device=device, dtype=torch.long)

            with torch.no_grad():
                eps_pred = net(x_t, t, cond)
                # x0 prediction (for guidance computation only)
                x0_pred = (x_t - sqrt_one_minus_alpha_cumprod[step] * eps_pred) / (sqrt_alpha_cumprod[step] + 1e-12)

            if use_guidance:
                if guidance_schedule == "linear" and timesteps > 1:
                    scale_t = float(guidance_scale) * (float(step) / float(timesteps - 1))
                else:
                    scale_t = float(guidance_scale)

                x_t = distribution_guidance_step(
                    x_t=x_t,
                    x_0_pred=x0_pred,
                    target_marginals=target_marginals or {},
                    guidance_scale=scale_t,
                    t=step,
                )

                if guidance_recompute_eps:
                    with torch.no_grad():
                        eps_pred = net(x_t, t, cond)

            with torch.no_grad():
                beta_t = betas[step]
                alpha_t = alphas[step]
                sqrt_om = sqrt_one_minus_alpha_cumprod[step]
                model_mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_om) * eps_pred)

                if step == 0:
                    x_t = model_mean
                else:
                    var_t = posterior_variance[step]
                    noise = torch.randn_like(x_t)
                    x_t = model_mean + torch.sqrt(var_t) * noise

        x = x_t.detach().cpu()
        z_person = x[:, : self.latent_dim]
        z_building = x[:, self.latent_dim :]
        return z_person, z_building
