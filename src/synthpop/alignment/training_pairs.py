from __future__ import annotations

"""
Utilities to construct training pairs for alignment / joint diffusion (Scheme C-v2).

This module will turn aligned person/device/building embeddings into:
- contrastive positive/negative pairs
- (z_person, z_building, condition) tuples for joint diffusion training
"""

from typing import Any


def build_device_building_pairs(*_: Any, **__: Any) -> Any:
    raise NotImplementedError("TODO(Scheme C-v2): implement training pair construction.")

