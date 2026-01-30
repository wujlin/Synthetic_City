from __future__ import annotations

"""
Contrastive alignment losses (Scheme C-v2).

v0: provide a simple, paired InfoNCE loss where positives are aligned by row index.
"""

from typing import Any


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("contrastive.py requires PyTorch.") from e
    return torch


def infonce_loss_paired(*, z_a: Any, z_b: Any, temperature: float = 0.07) -> Any:
    """
    Paired InfoNCE (SimCLR-style):
    - Inputs: z_a, z_b (batch, d), where i-th row is a positive pair.
    - Output: scalar loss.
    """
    torch = _require_torch()
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    z_a = torch.nn.functional.normalize(z_a, dim=1)
    z_b = torch.nn.functional.normalize(z_b, dim=1)
    logits = (z_a @ z_b.t()) / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)

