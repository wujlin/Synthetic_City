from __future__ import annotations

"""
Temporal validation (day/night or 24h profiles), if dynamic generation is enabled.
"""

from typing import Any


def compute_temporal_metrics(*, time_profiles: Any, reference: Any) -> dict[str, Any]:
    raise NotImplementedError("TODO(v0): implement temporal metrics.")

