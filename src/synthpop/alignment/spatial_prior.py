from __future__ import annotations

"""
Spatial priors for alignment (Scheme C-v2).

Examples:
- Activity-center consistency between device trajectories and building locations.

Left as a placeholder until we finalize the available spatial signals (geohash5 vs lat/lon vs CBG).
"""

from typing import Any


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("spatial_prior.py requires PyTorch.") from e
    return torch


def activity_center_loss(
    *,
    activity_centers: Any,
    building_locations: Any,
    reduction: str = "mean",
    p: int = 2,
) -> Any:
    """
    Activity-center consistency loss.

    v0 contract (KISS):
    - activity_centers and building_locations are aligned by row index
      (i.e., i-th device paired with i-th building).
    - both tensors are shaped (n, 2) in the same coordinate system (e.g., lon/lat or projected meters).

    Returns:
      scalar torch.Tensor
    """
    torch = _require_torch()
    a = torch.as_tensor(activity_centers).float()
    b = torch.as_tensor(building_locations).float().to(a.device)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 2 or b.shape[1] != 2:
        raise ValueError("activity_centers/building_locations must be shaped (n,2)")
    if a.shape[0] != b.shape[0]:
        raise ValueError("activity_centers and building_locations must have same n (paired by index)")

    d = torch.linalg.vector_norm(a - b, ord=int(p), dim=1)
    if reduction == "mean":
        return d.mean()
    if reduction == "sum":
        return d.sum()
    if reduction == "none":
        return d
    raise ValueError("reduction must be one of: mean, sum, none")
