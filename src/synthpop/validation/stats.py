from __future__ import annotations

"""
Statistical validation.

Core acceptance metrics (v0):
- marginal errors at tract/BG
- key 2nd-order associations (e.g., income×occupation, residence×work)
- hard-rule violation rate
"""

from typing import Any


def compute_stats_metrics(*, synthetic: Any, targets: Any) -> dict[str, Any]:
    raise NotImplementedError("TODO(v0): implement stats metrics.")

