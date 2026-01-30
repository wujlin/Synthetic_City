from __future__ import annotations

"""
Shared latent space encoders (Scheme C-v2).

PI intent:
- Encode person/device/building features into a shared latent space.
- Use alignment losses (contrastive + distribution match + spatial priors) to learn consistent representations.

KISS note:
- This module provides minimal, importable scaffolding first.
- The concrete alignment objective will be iterated as data formats & supervision signals are finalized.
"""

from dataclasses import dataclass
from typing import Any


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("SharedLatentSpace requires PyTorch. Install via conda/pip (CUDA if available).") from e
    return torch


def _torch_module_base() -> type:
    try:
        import torch  # type: ignore
    except Exception:
        return object
    return torch.nn.Module


class MLPEncoder(_torch_module_base()):
    def __init__(self, *, input_dim: int, latent_dim: int, hidden_dims: tuple[int, ...] = (256, 256)) -> None:
        torch = _require_torch()
        nn = torch.nn
        if hasattr(super(), "__init__"):
            super().__init__()  # type: ignore[misc]

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")

        layers: list[Any] = []
        dim_in = int(input_dim)
        for dim_out in hidden_dims:
            layers.append(nn.Linear(dim_in, int(dim_out)))
            layers.append(nn.SiLU())
            dim_in = int(dim_out)
        layers.append(nn.Linear(dim_in, int(latent_dim)))

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        self.net = nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:  # type: ignore[override]
        return self.net(x)


@dataclass(frozen=True)
class SharedLatentSpaceSpec:
    latent_dim: int = 32
    hidden_dims: tuple[int, ...] = (256, 256)


class SharedLatentSpace:
    """
    Thin wrapper owning three encoders sharing the same latent dimension.

    This class is intentionally minimal; the training loop lives in the pipeline layer.
    """

    def __init__(
        self,
        *,
        person_input_dim: int,
        device_input_dim: int,
        building_input_dim: int,
        spec: SharedLatentSpaceSpec | None = None,
    ) -> None:
        torch = _require_torch()
        self.spec = spec or SharedLatentSpaceSpec()

        self.person_encoder = MLPEncoder(
            input_dim=int(person_input_dim),
            latent_dim=int(self.spec.latent_dim),
            hidden_dims=self.spec.hidden_dims,
        )
        self.device_encoder = MLPEncoder(
            input_dim=int(device_input_dim),
            latent_dim=int(self.spec.latent_dim),
            hidden_dims=self.spec.hidden_dims,
        )
        self.building_encoder = MLPEncoder(
            input_dim=int(building_input_dim),
            latent_dim=int(self.spec.latent_dim),
            hidden_dims=self.spec.hidden_dims,
        )

        # Convenience: allow `.to(device)` on the wrapper.
        self._modules = torch.nn.ModuleList([self.person_encoder, self.device_encoder, self.building_encoder])

    def to(self, device: Any) -> "SharedLatentSpace":
        self._modules.to(device)
        return self

    def encode_person(self, x: Any) -> Any:
        return self.person_encoder(x)

    def encode_device(self, x: Any) -> Any:
        return self.device_encoder(x)

    def encode_building(self, x: Any) -> Any:
        return self.building_encoder(x)

    def alignment_loss(self, *_: Any, **__: Any) -> Any:
        """
        Placeholder for the joint alignment objective.

        Expected components (Scheme C-v2):
        - Device-Building contrastive loss (same CBG positives)
        - Person-Device distribution matching (CBG-level matching)
        - Spatial prior (activity center consistency)
        """
        raise NotImplementedError("TODO(Scheme C-v2): implement alignment losses once training pairs are finalized.")

