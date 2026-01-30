from __future__ import annotations

"""
Distribution matching utilities (Scheme C-v2).

v0: provide an RBF-kernel MMD implementation that can be used for CBG-level distribution alignment.
"""

from typing import Any


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("distribution_match.py requires PyTorch.") from e
    return torch


def mmd_rbf(*, x: Any, y: Any, sigma: float = 1.0) -> Any:
    """
    Maximum Mean Discrepancy with RBF kernel.
    Inputs: x (n,d), y (m,d)
    """
    torch = _require_torch()
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    def _cdist_sq(a: Any, b: Any) -> Any:
        a2 = (a * a).sum(dim=1, keepdim=True)
        b2 = (b * b).sum(dim=1, keepdim=True).t()
        return a2 + b2 - 2.0 * (a @ b.t())

    xx = torch.exp(-_cdist_sq(x, x) / (2.0 * sigma * sigma))
    yy = torch.exp(-_cdist_sq(y, y) / (2.0 * sigma * sigma))
    xy = torch.exp(-_cdist_sq(x, y) / (2.0 * sigma * sigma))
    return xx.mean() + yy.mean() - 2.0 * xy.mean()

