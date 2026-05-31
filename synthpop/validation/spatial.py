from __future__ import annotations

"""
Spatial validation (weak supervision via POI/landuse/nightlights, etc.).
"""

from typing import Any


def compute_spatial_metrics(*, assignments: Any, buildings: Any, poi: Any | None) -> dict[str, Any]:
    raise NotImplementedError("TODO(v0): implement spatial metrics.")

