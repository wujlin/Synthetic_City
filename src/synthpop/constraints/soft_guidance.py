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
    v0 简化版：post-hoc importance reweighting（分批采样/生成后重加权或重采样）。

    设计动机：
    - 先把闭环跑通，证明“边际能拉回”；
    - 后续可迁移到 step-wise guidance（每步去噪后持续拉回）。

    约定（KISS）：
    - samples/marginals 期望为 pandas.DataFrame
    - marginals 至少提供：类别列（默认 category）与目标值列（默认 target）
    - 返回的 DataFrame 会包含 out_weight_col；若 resample=True，则返回重采样后的样本（同大小）。
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
