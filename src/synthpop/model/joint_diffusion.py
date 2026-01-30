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
    ) -> tuple[Any, Any]:
        """
        Sample (z_person, z_building).

        v0:
        - guidance is not integrated into the TabDDPM sampler yet; passing target_marginals is rejected.
        """
        if target_marginals is not None and guidance_scale > 0:
            raise NotImplementedError("TODO(Scheme C-v2): integrate distribution guidance into joint diffusion sampling.")

        x = self._base.sample(n=n, cond=cond, device=device)
        z_person = x[:, : self.latent_dim]
        z_building = x[:, self.latent_dim :]
        return z_person, z_building


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("joint_diffusion.py requires PyTorch.") from e
    return torch

