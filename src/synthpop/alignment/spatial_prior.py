from __future__ import annotations

"""
Spatial priors for alignment (Scheme C-v2).

Examples:
- Activity-center consistency between device trajectories and building locations.

Left as a placeholder until we finalize the available spatial signals (geohash5 vs lat/lon vs CBG).
"""

from typing import Any


def activity_center_loss(*_: Any, **__: Any) -> Any:
    raise NotImplementedError("TODO(Scheme C-v2): implement spatial prior once activity center representation is fixed.")

