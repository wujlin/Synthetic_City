from __future__ import annotations

from dataclasses import dataclass

from .shared_latent import MLPEncoder


@dataclass(frozen=True)
class BuildingEncoderSpec:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: tuple[int, ...] = (256, 256)


def build_building_encoder(*, spec: BuildingEncoderSpec) -> MLPEncoder:
    return MLPEncoder(input_dim=int(spec.input_dim), latent_dim=int(spec.latent_dim), hidden_dims=spec.hidden_dims)

