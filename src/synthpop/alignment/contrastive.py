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


def infonce_loss_by_group(
    *,
    z_query: Any,
    z_key: Any,
    query_group_ids: Any,
    key_group_ids: Any,
    temperature: float = 0.07,
    eps: float = 1e-12,
) -> Any:
    """
    Group-supervised InfoNCE:
    - For each query i, all keys j with the same group id are treated as positives.
    - Loss is averaged across queries with at least one positive.

    This is a simple way to express "same CBG positives" without requiring explicit pair construction.
    """
    torch = _require_torch()
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    z_query = torch.nn.functional.normalize(z_query, dim=1)
    z_key = torch.nn.functional.normalize(z_key, dim=1)
    logits = (z_query @ z_key.t()) / float(temperature)

    qid = _encode_group_ids(query_group_ids, device=logits.device)
    kid = _encode_group_ids(key_group_ids, device=logits.device)
    if qid.ndim != 1 or kid.ndim != 1:
        raise ValueError("group ids must be 1D")
    if qid.shape[0] != logits.shape[0]:
        raise ValueError("query_group_ids length mismatch")
    if kid.shape[0] != logits.shape[1]:
        raise ValueError("key_group_ids length mismatch")

    pos_mask = (qid.reshape(-1, 1) == kid.reshape(1, -1)).to(dtype=logits.dtype)
    exp_logits = torch.exp(logits)
    pos = (exp_logits * pos_mask).sum(dim=1)
    all_sum = exp_logits.sum(dim=1)

    valid = pos > 0
    if valid.sum() == 0:
        raise ValueError("No positive pairs found (check group ids).")
    loss = -torch.log((pos[valid] + float(eps)) / (all_sum[valid] + float(eps)))
    return loss.mean()


def _encode_group_ids(ids: Any, *, device: Any) -> Any:
    """
    Encode group ids into a 1D int64 tensor.

    Supports:
    - torch.Tensor (any dtype convertible to long)
    - list/tuple of hashables (strings/ints)
    """
    torch = _require_torch()
    if isinstance(ids, torch.Tensor):
        return ids.to(device=device).to(dtype=torch.long).reshape(-1)

    if not isinstance(ids, (list, tuple)):
        raise TypeError("group ids must be a torch.Tensor or a list/tuple")

    mapping: dict[str, int] = {}
    out: list[int] = []
    for x in ids:
        k = str(x)
        if k not in mapping:
            mapping[k] = len(mapping)
        out.append(mapping[k])
    return torch.tensor(out, device=device, dtype=torch.long)
