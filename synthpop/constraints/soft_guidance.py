from __future__ import annotations

"""
Soft constraint guidance (macro marginals).

Design intent: during sampling, keep pulling the generated distribution back towards target statistics.
Implementation choices (kept out of this v0 module):
- step-wise guidance vs batched reweighting vs iterative calibration.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SoftGuidanceSpec:
    strength: float = 1.0


def apply_soft_guidance(*, samples: Any, targets: Any, spec: SoftGuidanceSpec) -> Any:
    raise NotImplementedError("TODO(v0): sampling-time soft guidance to match marginals/associations.")


def apply_soft_guidance_v0(
    *,
    samples: Any,
    marginals: Any,
    target_col: str,
    category_col: str = "category",
    target_value_col: str = "target",
    out_weight_col: str = "soft_weight",
    resample: bool = True,
    seed: int = 0,
    eps: float = 1e-12,
) -> Any:
    """
    v0 implementation: post-hoc importance reweighting after batched sampling.

    Design intent:
    - close the simplest loop and verify that marginals can be pulled back;
    - keep a path toward step-wise guidance during denoising.

    Conventions:
    - samples and marginals must be pandas DataFrame objects;
    - marginals must provide a category column and a target-value column;
    - the returned DataFrame includes out_weight_col; if resample=True, it
      returns a same-size resampled DataFrame.
    """
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("apply_soft_guidance_v0 requires pandas and numpy.") from e

    if not isinstance(samples, pd.DataFrame) or not isinstance(marginals, pd.DataFrame):
        raise TypeError("samples/marginals must be pandas DataFrame")
    if target_col not in samples.columns:
        raise ValueError(f"samples missing target_col: {target_col}")

    if category_col in marginals.columns:
        cat_series = marginals[category_col]
    elif target_col in marginals.columns:
        cat_series = marginals[target_col]
    else:
        raise ValueError(f"marginals must have '{category_col}' or '{target_col}' column")

    if target_value_col not in marginals.columns:
        raise ValueError(f"marginals missing target_value_col: {target_value_col}")

    m = marginals.copy()
    m["_cat"] = cat_series.astype(str)
    m["_target"] = pd.to_numeric(m[target_value_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    target = m.groupby("_cat")["_target"].sum()
    total = float(target.sum())
    if total <= 0:
        raise ValueError("marginals target sum must be > 0")
    target_prob = (target / total).to_dict()

    s = samples.copy()
    s["_cat"] = s[target_col].astype(str)
    sample_prob = s["_cat"].value_counts(normalize=True).to_dict()

    def _w(cat: str) -> float:
        tp = float(target_prob.get(cat, 0.0))
        sp = float(sample_prob.get(cat, 0.0))
        if sp <= eps:
            return 0.0
        return tp / sp

    weights = s["_cat"].map(_w).astype(float)
    weights = weights.fillna(0.0).clip(lower=0.0)
    s[out_weight_col] = weights

    if not resample:
        return s.drop(columns=["_cat"])

    w = s[out_weight_col].to_numpy(dtype=float)
    w_sum = float(w.sum())
    if w_sum <= 0:
        # Fallback: cannot reweight; return original with uniform weights.
        s[out_weight_col] = 1.0
        return s.drop(columns=["_cat"])

    p = w / w_sum
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(s), size=len(s), replace=True, p=p)
    out = s.iloc[idx].reset_index(drop=True)
    return out.drop(columns=["_cat"])


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Distribution guidance requires PyTorch.") from e
    return torch


def soft_histogram(*, x: Any, bins: Any, sigma: float = 1.0, eps: float = 1e-12) -> Any:
    """
    Differentiable histogram via Gaussian kernels around bin centers.

    Args:
      x: (n,) or (n,1) tensor
      bins: (k,) tensor of bin centers
    Returns:
      probs: (k,) tensor summing to 1
    """
    torch = _require_torch()
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    x = x.reshape(-1, 1).float()
    bins = bins.reshape(1, -1).float().to(x.device)
    w = torch.exp(-((x - bins) ** 2) / (2.0 * float(sigma) * float(sigma)))
    hist = w.sum(dim=0)
    return hist / (hist.sum() + float(eps))


def tvd(*, current: Any, target: Any) -> Any:
    """Total variation distance between two distributions."""
    torch = _require_torch()
    current = current.float()
    target = target.float().to(current.device)
    return 0.5 * torch.abs(current - target).sum()


def distribution_guidance_step(
    *,
    x_t: Any,
    x_0_pred: Any,
    target_marginals: dict[str, Any],
    guidance_scale: float,
    t: int,
) -> Any:
    """
    Single-step Distribution Guidance (Scheme C-v2 draft).

    This is a minimal, generic implementation that supports two spec types:

    1) Categorical one-hot group:
      target_marginals[name] = {
        "type": "categorical_onehot",
        "indices": [int, ...],      # columns in x_0_pred belonging to the categorical group
        "target": [float, ...],     # target probs (same length as indices)
      }

    2) Continuous scalar histogram:
      target_marginals[name] = {
        "type": "continuous_hist",
        "index": int,              # scalar dimension in x_0_pred
        "bins": [float, ...],      # bin centers
        "target": [float, ...],    # target probs
        "sigma": float,            # optional, defaults to 1.0
      }

    Returns:
      guided_x_t: tensor shaped like x_t

    Notes:
    - This function does not decide *which* marginals to guide; the caller provides the spec dict.
    - For now we compute guidance from x_0_pred and apply it directly on x_t.
    """
    torch = _require_torch()
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be >= 0")
    _ = int(t)  # keep signature stable; schedule-aware scaling can be added later.

    if not isinstance(target_marginals, dict) or not target_marginals:
        return x_t

    x = x_0_pred.detach().clone().requires_grad_(True)
    loss = torch.tensor(0.0, device=x.device)

    for name, spec in target_marginals.items():
        if not isinstance(spec, dict):
            raise TypeError(f"target_marginals['{name}'] must be a dict spec")
        typ = spec.get("type")
        if typ == "categorical_onehot":
            idx = spec.get("indices")
            tgt = spec.get("target")
            if not isinstance(idx, (list, tuple)) or len(idx) == 0:
                raise ValueError(f"{name}: indices must be a non-empty list")
            if tgt is None:
                raise ValueError(f"{name}: target is required")
            logits = x[:, list(idx)]
            p = torch.softmax(logits, dim=1).mean(dim=0)
            target = torch.tensor(tgt, device=x.device, dtype=p.dtype)
            target = target / (target.sum() + 1e-12)
            loss = loss + tvd(current=p, target=target)
        elif typ == "continuous_hist":
            idx = spec.get("index")
            bins = spec.get("bins")
            tgt = spec.get("target")
            sigma = float(spec.get("sigma", 1.0))
            if idx is None:
                raise ValueError(f"{name}: index is required")
            if bins is None or tgt is None:
                raise ValueError(f"{name}: bins/target are required")
            values = x[:, int(idx)]
            probs = soft_histogram(x=values, bins=torch.tensor(bins, device=x.device), sigma=sigma)
            target = torch.tensor(tgt, device=x.device, dtype=probs.dtype)
            target = target / (target.sum() + 1e-12)
            loss = loss + tvd(current=probs, target=target)
        else:
            raise ValueError(f"{name}: unsupported spec type: {typ}")

    if float(guidance_scale) == 0.0:
        return x_t

    (grad,) = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)
    return x_t - float(guidance_scale) * grad
