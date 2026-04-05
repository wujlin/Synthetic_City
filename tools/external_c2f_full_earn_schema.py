#!/usr/bin/env python3
from __future__ import annotations

"""
Shared schema helpers for the 5-way coarse-to-fine full-earn experiment.
"""

from typing import Any

import numpy as np

from tools.build_external_target_v1_michigan import AGE_LABELS, ESR_LABELS, SCHL_LABELS, SEX_LABELS
from tools.external_earn_v1_schema import EARN_LABELS
from tools.external_v1_variant_presets import AGE_LITE_LABELS, AGE_TO_LITE, ESR_LITE_LABELS, ESR_TO_LITE, SCHL_LITE_LABELS, SCHL_TO_LITE


EARN_LITE_LABELS = [
    "not_in_earnings_universe",
    "lt_50k",
    "50k_100k",
    "ge_100k",
]
EARN_TO_LITE = {
    "not_in_earnings_universe": "not_in_earnings_universe",
    "lt_25k": "lt_50k",
    "25k_50k": "lt_50k",
    "50k_75k": "50k_100k",
    "75k_100k": "50k_100k",
    "ge_100k": "ge_100k",
}

FULL_VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop", "EARN_16p_bin"]
FULL_CATEGORIES = {
    "AGEP_bin": AGE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LABELS,
    "ESR_allpop": ESR_LABELS,
    "EARN_16p_bin": EARN_LABELS,
}
FULL_SHAPE = tuple(len(FULL_CATEGORIES[v]) for v in FULL_VARIABLE_ORDER)
FULL_K = int(np.prod(FULL_SHAPE))

COARSE_VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop", "EARN_16p_bin"]
COARSE_CATEGORIES = {
    "AGEP_bin": AGE_LITE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LITE_LABELS,
    "ESR_allpop": ESR_LITE_LABELS,
    "EARN_16p_bin": EARN_LITE_LABELS,
}
COARSE_SHAPE = tuple(len(COARSE_CATEGORIES[v]) for v in COARSE_VARIABLE_ORDER)
COARSE_K = int(np.prod(COARSE_SHAPE))

AGE_FINE_TO_COARSE = np.asarray([AGE_LITE_LABELS.index(AGE_TO_LITE[x]) for x in AGE_LABELS], dtype=np.int16)
SCHL_FINE_TO_COARSE = np.asarray([SCHL_LITE_LABELS.index(SCHL_TO_LITE[x]) for x in SCHL_LABELS], dtype=np.int16)
ESR_FINE_TO_COARSE = np.asarray([ESR_LITE_LABELS.index(ESR_TO_LITE[x]) for x in ESR_LABELS], dtype=np.int16)
EARN_FINE_TO_COARSE = np.asarray([EARN_LITE_LABELS.index(EARN_TO_LITE[x]) for x in EARN_LABELS], dtype=np.int16)


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
                    for wi, _ in enumerate(EARN_LABELS):
                        wc = int(EARN_FINE_TO_COARSE[wi])
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
        "EARN_16p_bin_lite": EARN_LITE_LABELS[int(wc)],
    }


def parent_count_histogram() -> dict[int, int]:
    vals = [int(x.shape[0]) for x in PARENT_TO_CHILD_FULL]
    out: dict[int, int] = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return out
