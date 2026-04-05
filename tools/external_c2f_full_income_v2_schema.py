#!/usr/bin/env python3
from __future__ import annotations

"""
Shared schema helpers for the 5-way coarse-to-fine full-income v2 experiment.
"""

from typing import Any

import numpy as np

from tools.build_external_target_v1_michigan import AGE_LABELS, ESR_LABELS, SCHL_LABELS, SEX_LABELS
from tools.external_income_v1_schema import (
    INCOME_LABELS,
    INCOME_LITE_LABELS,
    INCOME_REGIME_LABELS,
    INCOME_TO_LITE,
    INCOME_TO_REGIME,
)
from tools.external_v1_variant_presets import (
    AGE_LITE_LABELS,
    AGE_TO_LITE,
    ESR_LITE_LABELS,
    ESR_TO_LITE,
    SCHL_LITE_LABELS,
    SCHL_TO_LITE,
)


FULL_VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop", "PINCP_allpop_bin"]
FULL_CATEGORIES = {
    "AGEP_bin": AGE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LABELS,
    "ESR_allpop": ESR_LABELS,
    "PINCP_allpop_bin": INCOME_LABELS,
}
FULL_SHAPE = tuple(len(FULL_CATEGORIES[v]) for v in FULL_VARIABLE_ORDER)
FULL_K = int(np.prod(FULL_SHAPE))

COARSE_VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop", "PINCP_allpop_bin"]
COARSE_CATEGORIES = {
    "AGEP_bin": AGE_LITE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LITE_LABELS,
    "ESR_allpop": ESR_LITE_LABELS,
    "PINCP_allpop_bin": INCOME_LITE_LABELS,
}
COARSE_SHAPE = tuple(len(COARSE_CATEGORIES[v]) for v in COARSE_VARIABLE_ORDER)
COARSE_K = int(np.prod(COARSE_SHAPE))

AGE_FINE_TO_COARSE = np.asarray([AGE_LITE_LABELS.index(AGE_TO_LITE[x]) for x in AGE_LABELS], dtype=np.int16)
SCHL_FINE_TO_COARSE = np.asarray([SCHL_LITE_LABELS.index(SCHL_TO_LITE[x]) for x in SCHL_LABELS], dtype=np.int16)
ESR_FINE_TO_COARSE = np.asarray([ESR_LITE_LABELS.index(ESR_TO_LITE[x]) for x in ESR_LABELS], dtype=np.int16)
INCOME_FINE_TO_COARSE = np.asarray([INCOME_LITE_LABELS.index(INCOME_TO_LITE[x]) for x in INCOME_LABELS], dtype=np.int16)
INCOME_FINE_TO_REGIME = np.asarray([INCOME_REGIME_LABELS.index(INCOME_TO_REGIME[x]) for x in INCOME_LABELS], dtype=np.int16)


def _child_parent_index_full() -> np.ndarray:
    out = np.zeros((FULL_K,), dtype=np.int16)
    k = 0
    for ai, _ in enumerate(AGE_LABELS):
        ac = int(AGE_FINE_TO_COARSE[ai])
        for si, _ in enumerate(SEX_LABELS):
            for qi, _ in enumerate(SCHL_LABELS):
                qc = int(SCHL_FINE_TO_COARSE[qi])
                for ei, _ in enumerate(ESR_LABELS):
                    ec = int(ESR_FINE_TO_COARSE[ei])
                    for wi, _ in enumerate(INCOME_LABELS):
                        wc = int(INCOME_FINE_TO_COARSE[wi])
                        out[k] = int(np.ravel_multi_index((ac, si, qc, ec, wc), COARSE_SHAPE))
                        k += 1
    return out


CHILD_PARENT_INDEX_FULL = _child_parent_index_full()


def _parent_to_child_full_indices() -> list[np.ndarray]:
    out: list[list[int]] = [[] for _ in range(COARSE_K)]
    for child_idx, parent_idx in enumerate(CHILD_PARENT_INDEX_FULL.tolist()):
        out[int(parent_idx)].append(int(child_idx))
    return [np.asarray(v, dtype=np.int32) for v in out]


PARENT_TO_CHILD_FULL = _parent_to_child_full_indices()
MAX_CHILDREN = max(len(v) for v in PARENT_TO_CHILD_FULL)


def parent_child_slot_mask() -> np.ndarray:
    out = np.zeros((COARSE_K, MAX_CHILDREN), dtype=np.float32)
    for pid, children in enumerate(PARENT_TO_CHILD_FULL):
        out[pid, : int(children.shape[0])] = 1.0
    return out


PARENT_CHILD_SLOT_MASK = parent_child_slot_mask()


def padded_parent_child_full_indices() -> np.ndarray:
    out = np.full((COARSE_K, MAX_CHILDREN), -1, dtype=np.int32)
    for pid, children in enumerate(PARENT_TO_CHILD_FULL):
        out[pid, : int(children.shape[0])] = children
    return out


PADDED_PARENT_CHILD_FULL = padded_parent_child_full_indices()


def child_income_regime_full() -> np.ndarray:
    out = np.zeros((FULL_K,), dtype=np.int16)
    for idx in range(FULL_K):
        wi = int(np.unravel_index(idx, FULL_SHAPE)[-1])
        out[idx] = int(INCOME_FINE_TO_REGIME[wi])
    return out


CHILD_INCOME_REGIME_FULL = child_income_regime_full()


def child_income_aux_full() -> np.ndarray:
    out = np.zeros((FULL_K,), dtype=np.int16)
    for idx in range(FULL_K):
        wi = int(np.unravel_index(idx, FULL_SHAPE)[-1])
        out[idx] = int(wi)
    return out


CHILD_INCOME_AUX_FULL = child_income_aux_full()


def padded_parent_child_income_regime() -> np.ndarray:
    out = np.full((COARSE_K, MAX_CHILDREN), -1, dtype=np.int16)
    for pid, children in enumerate(PARENT_TO_CHILD_FULL):
        for slot_idx, child_idx in enumerate(children.tolist()):
            # Use fine income bins as the auxiliary grouping. The previous
            # 3-way regime split was vacuous because the coarse income-lite
            # parent already fixed the regime, making auxiliary losses zero.
            out[pid, slot_idx] = int(CHILD_INCOME_AUX_FULL[int(child_idx)])
    return out


PADDED_PARENT_CHILD_INCOME_REGIME = padded_parent_child_income_regime()


def coarse_from_full_flat(p_full: np.ndarray) -> np.ndarray:
    p = np.asarray(p_full, dtype=np.float64).reshape(-1)
    if p.shape[0] != FULL_K:
        raise ValueError(f"Unexpected full joint length={p.shape[0]}, expected={FULL_K}")
    out = np.bincount(CHILD_PARENT_INDEX_FULL.astype(np.int32), weights=p, minlength=COARSE_K).astype(np.float64)
    out = out / max(float(out.sum()), 1e-12)
    return out


def parent_index_labels(parent_idx: int) -> dict[str, Any]:
    ac, si, qc, ec, wc = np.unravel_index(int(parent_idx), COARSE_SHAPE)
    return {
        "parent_idx": int(parent_idx),
        "AGEP_bin_lite": AGE_LITE_LABELS[int(ac)],
        "SEX": SEX_LABELS[int(si)],
        "SCHL_allpop_lite": SCHL_LITE_LABELS[int(qc)],
        "ESR_allpop_lite": ESR_LITE_LABELS[int(ec)],
        "PINCP_allpop_bin_lite": INCOME_LITE_LABELS[int(wc)],
    }


def parent_count_histogram() -> dict[int, int]:
    vals = [int(x.shape[0]) for x in PARENT_TO_CHILD_FULL]
    out: dict[int, int] = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return out
