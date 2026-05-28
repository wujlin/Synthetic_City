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

    def alignment_loss(
        self,
        *,
        z_persons: Any,
        z_devices: Any,
        z_buildings: Any,
        device_cbg_ids: Any,
        building_cbg_ids: Any,
        person_cbg_ids: Any | None = None,
        activity_centers: Any | None = None,
        building_locations: Any | None = None,
        weights: dict[str, float] | None = None,
        temperature: float = 0.07,
        mmd_sigma: float = 1.0,
        min_samples: int = 10,
    ) -> Any:
        """
        Total alignment loss = contrastive loss + distribution matching + spatial prior.

        Components:
        1) Device-building contrastive loss with same-CBG positives.
        2) Person-device MMD, globally by default or averaged by CBG when
           person_cbg_ids are provided.
        3) Optional activity-center consistency with paired activity centers
           and building locations.
        """
        torch = _require_torch()

        w = {"contrastive": 1.0, "mmd": 0.1, "spatial": 0.1}
        if weights:
            w.update({k: float(v) for k, v in weights.items()})

        z_persons = torch.as_tensor(z_persons).float()
        z_devices = torch.as_tensor(z_devices).float().to(z_persons.device)
        z_buildings = torch.as_tensor(z_buildings).float().to(z_persons.device)

        from ..alignment.contrastive import infonce_loss_by_group
        from ..alignment.distribution_match import mmd_rbf

        l_contrast = infonce_loss_by_group(
            z_query=z_devices,
            z_key=z_buildings,
            query_group_ids=device_cbg_ids,
            key_group_ids=building_cbg_ids,
            temperature=float(temperature),
        )

        if person_cbg_ids is None:
            l_mmd = mmd_rbf(x=z_persons, y=z_devices, sigma=float(mmd_sigma))
        else:
            pid = _encode_group_ids(person_cbg_ids, device=z_persons.device)
            did = _encode_group_ids(device_cbg_ids, device=z_persons.device)
            losses: list[Any] = []
            for g in torch.unique(pid):
                mp = pid == g
                md = did == g
                if int(mp.sum().item()) < int(min_samples) or int(md.sum().item()) < int(min_samples):
                    continue
                losses.append(mmd_rbf(x=z_persons[mp], y=z_devices[md], sigma=float(mmd_sigma)))
            if losses:
                l_mmd = torch.stack(losses).mean()
            else:
                l_mmd = torch.tensor(0.0, device=z_persons.device)

        l_spatial = torch.tensor(0.0, device=z_persons.device)
        if activity_centers is not None and building_locations is not None:
            from ..alignment.spatial_prior import activity_center_loss

            l_spatial = activity_center_loss(activity_centers=activity_centers, building_locations=building_locations)

        return float(w["contrastive"]) * l_contrast + float(w["mmd"]) * l_mmd + float(w["spatial"]) * l_spatial


def _encode_group_ids(ids: Any, *, device: Any) -> Any:
    torch = _require_torch()
    if isinstance(ids, torch.Tensor):
        return ids.to(device=device).to(dtype=torch.long).reshape(-1)
    if not isinstance(ids, (list, tuple)):
        raise TypeError("group ids must be a torch.Tensor or a list/tuple")
    mapping: dict[str, int] = {}
    out: list[int] = []
    for x in ids:
        k = str(x)
        if k not in mapping:
            mapping[k] = len(mapping)
        out.append(mapping[k])
    return torch.tensor(out, device=device, dtype=torch.long)
