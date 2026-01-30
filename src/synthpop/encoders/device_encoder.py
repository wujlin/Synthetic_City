from __future__ import annotations

from dataclasses import dataclass

from .shared_latent import MLPEncoder


@dataclass(frozen=True)
class DeviceEncoderSpec:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: tuple[int, ...] = (256, 256)


def build_device_encoder(*, spec: DeviceEncoderSpec) -> MLPEncoder:
    return MLPEncoder(input_dim=int(spec.input_dim), latent_dim=int(spec.latent_dim), hidden_dims=spec.hidden_dims)

