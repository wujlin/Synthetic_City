#!/usr/bin/env python3
from __future__ import annotations

"""
Shared schema helpers for the 5-way coarse-to-fine full-earn experiment.
"""

import os
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

AGE_COARSER_LABELS = [
    "[0.0, 18.0)",
    "[18.0, 65.0)",
    "[65.0, 1000.0)",
]
AGE_TO_COARSER = {
    "[0.0, 5.0)": "[0.0, 18.0)",
    "[5.0, 18.0)": "[0.0, 18.0)",
    "[18.0, 25.0)": "[18.0, 65.0)",
    "[25.0, 35.0)": "[18.0, 65.0)",
    "[35.0, 45.0)": "[18.0, 65.0)",
    "[45.0, 55.0)": "[18.0, 65.0)",
    "[55.0, 65.0)": "[18.0, 65.0)",
    "[65.0, 75.0)": "[65.0, 1000.0)",
    "[75.0, 85.0)": "[65.0, 1000.0)",
    "[85.0, 1000.0)": "[65.0, 1000.0)",
}

ESR_COARSER_LABELS = [
    "not_16p",
    "16p",
]
ESR_TO_COARSER = {
    "not_16p": "not_16p",
    "employed": "16p",
    "unemployed": "16p",
    "armed_forces": "16p",
    "not_in_labor_force": "16p",
}

EARN_COARSER_LABELS = [
    "not_in_earnings_universe",
    "lt_50k",
    "ge_50k",
]
EARN_TO_COARSER = {
    "not_in_earnings_universe": "not_in_earnings_universe",
    "lt_25k": "lt_50k",
    "25k_50k": "lt_50k",
    "50k_75k": "ge_50k",
    "75k_100k": "ge_50k",
    "ge_100k": "ge_50k",
}

AGE_MID6_LABELS = [
    "[0.0, 18.0)",
    "[18.0, 35.0)",
    "[35.0, 55.0)",
    "[55.0, 65.0)",
    "[65.0, 75.0)",
    "[75.0, 1000.0)",
]
AGE_TO_MID6 = {
    "[0.0, 5.0)": "[0.0, 18.0)",
    "[5.0, 18.0)": "[0.0, 18.0)",
    "[18.0, 25.0)": "[18.0, 35.0)",
    "[25.0, 35.0)": "[18.0, 35.0)",
    "[35.0, 45.0)": "[35.0, 55.0)",
    "[45.0, 55.0)": "[35.0, 55.0)",
    "[55.0, 65.0)": "[55.0, 65.0)",
    "[65.0, 75.0)": "[65.0, 75.0)",
    "[75.0, 85.0)": "[75.0, 1000.0)",
    "[85.0, 1000.0)": "[75.0, 1000.0)",
}

AGE_MID8_LABELS = [
    "[0.0, 18.0)",
    "[18.0, 25.0)",
    "[25.0, 35.0)",
    "[35.0, 45.0)",
    "[45.0, 55.0)",
    "[55.0, 65.0)",
    "[65.0, 75.0)",
    "[75.0, 1000.0)",
]
AGE_TO_MID8 = {
    "[0.0, 5.0)": "[0.0, 18.0)",
    "[5.0, 18.0)": "[0.0, 18.0)",
    "[18.0, 25.0)": "[18.0, 25.0)",
    "[25.0, 35.0)": "[25.0, 35.0)",
    "[35.0, 45.0)": "[35.0, 45.0)",
    "[45.0, 55.0)": "[45.0, 55.0)",
    "[55.0, 65.0)": "[55.0, 65.0)",
    "[65.0, 75.0)": "[65.0, 75.0)",
    "[75.0, 85.0)": "[75.0, 1000.0)",
    "[85.0, 1000.0)": "[75.0, 1000.0)",
}

ESR_MID4_LABELS = [
    "not_16p",
    "employed",
    "unemployed_or_armed_forces",
    "not_in_labor_force",
]
ESR_TO_MID4 = {
    "not_16p": "not_16p",
    "employed": "employed",
    "unemployed": "unemployed_or_armed_forces",
    "armed_forces": "unemployed_or_armed_forces",
    "not_in_labor_force": "not_in_labor_force",
}


def _identity_map(labels: list[str]) -> dict[str, str]:
    return {str(x): str(x) for x in labels}


def _coarse_preset() -> tuple[str, dict[str, list[str]], dict[str, dict[str, str]]]:
    preset = os.environ.get("SYNTHETIC_CITY_C2F_COARSE_PRESET", "main_288").strip().lower()
    if preset in {"main", "main_288", "288"}:
        return (
            "main_288",
            {
                "AGEP_bin": AGE_LITE_LABELS,
                "SEX": SEX_LABELS,
                "SCHL_allpop": SCHL_LITE_LABELS,
                "ESR_allpop": ESR_LITE_LABELS,
                "EARN_16p_bin": EARN_LITE_LABELS,
            },
            {
                "AGEP_bin": AGE_TO_LITE,
                "SEX": _identity_map(SEX_LABELS),
                "SCHL_allpop": SCHL_TO_LITE,
                "ESR_allpop": ESR_TO_LITE,
                "EARN_16p_bin": EARN_TO_LITE,
            },
        )
    if preset in {"coarser", "coarse_108", "108"}:
        return (
            "coarse_108",
            {
                "AGEP_bin": AGE_COARSER_LABELS,
                "SEX": SEX_LABELS,
                "SCHL_allpop": SCHL_LITE_LABELS,
                "ESR_allpop": ESR_COARSER_LABELS,
                "EARN_16p_bin": EARN_COARSER_LABELS,
            },
            {
                "AGEP_bin": AGE_TO_COARSER,
                "SEX": _identity_map(SEX_LABELS),
                "SCHL_allpop": SCHL_TO_LITE,
                "ESR_allpop": ESR_TO_COARSER,
                "EARN_16p_bin": EARN_TO_COARSER,
            },
        )
    if preset in {"finer", "fine_720", "720"}:
        return (
            "fine_720",
            {
                "AGEP_bin": AGE_LITE_LABELS,
                "SEX": SEX_LABELS,
                "SCHL_allpop": SCHL_LABELS,
                "ESR_allpop": ESR_LITE_LABELS,
                "EARN_16p_bin": EARN_LABELS,
            },
            {
                "AGEP_bin": AGE_TO_LITE,
                "SEX": _identity_map(SEX_LABELS),
                "SCHL_allpop": _identity_map(SCHL_LABELS),
                "ESR_allpop": ESR_TO_LITE,
                "EARN_16p_bin": _identity_map(EARN_LABELS),
            },
        )
    if preset in {"mid_480", "480"}:
        return (
            "mid_480",
            {
                "AGEP_bin": AGE_LITE_LABELS,
                "SEX": SEX_LABELS,
                "SCHL_allpop": SCHL_LABELS,
                "ESR_allpop": ESR_LITE_LABELS,
                "EARN_16p_bin": EARN_LITE_LABELS,
            },
            {
                "AGEP_bin": AGE_TO_LITE,
                "SEX": _identity_map(SEX_LABELS),
                "SCHL_allpop": _identity_map(SCHL_LABELS),
                "ESR_allpop": ESR_TO_LITE,
                "EARN_16p_bin": EARN_TO_LITE,
            },
        )
    if preset in {"fine_960", "960"}:
        return (
            "fine_960",
            {
                "AGEP_bin": AGE_LITE_LABELS,
                "SEX": SEX_LABELS,
                "SCHL_allpop": SCHL_LABELS,
                "ESR_allpop": ESR_MID4_LABELS,
                "EARN_16p_bin": EARN_LABELS,
            },
            {
                "AGEP_bin": AGE_TO_LITE,
                "SEX": _identity_map(SEX_LABELS),
                "SCHL_allpop": _identity_map(SCHL_LABELS),
                "ESR_allpop": ESR_TO_MID4,
                "EARN_16p_bin": _identity_map(EARN_LABELS),
            },
        )
    if preset in {"fine_1440", "1440"}:
        return (
            "fine_1440",
            {
                "AGEP_bin": AGE_MID6_LABELS,
                "SEX": SEX_LABELS,
                "SCHL_allpop": SCHL_LABELS,
                "ESR_allpop": ESR_MID4_LABELS,
                "EARN_16p_bin": EARN_LABELS,
            },
            {
                "AGEP_bin": AGE_TO_MID6,
                "SEX": _identity_map(SEX_LABELS),
                "SCHL_allpop": _identity_map(SCHL_LABELS),
                "ESR_allpop": ESR_TO_MID4,
                "EARN_16p_bin": _identity_map(EARN_LABELS),
            },
        )
    if preset in {"fine_1800", "1800"}:
        return (
            "fine_1800",
            {
                "AGEP_bin": AGE_MID6_LABELS,
                "SEX": SEX_LABELS,
                "SCHL_allpop": SCHL_LABELS,
                "ESR_allpop": ESR_LABELS,
                "EARN_16p_bin": EARN_LABELS,
            },
            {
                "AGEP_bin": AGE_TO_MID6,
                "SEX": _identity_map(SEX_LABELS),
                "SCHL_allpop": _identity_map(SCHL_LABELS),
                "ESR_allpop": _identity_map(ESR_LABELS),
                "EARN_16p_bin": _identity_map(EARN_LABELS),
            },
        )
    if preset in {"fine_2400", "2400"}:
        return (
            "fine_2400",
            {
                "AGEP_bin": AGE_LABELS,
                "SEX": SEX_LABELS,
                "SCHL_allpop": SCHL_LABELS,
                "ESR_allpop": ESR_MID4_LABELS,
                "EARN_16p_bin": EARN_LABELS,
            },
            {
                "AGEP_bin": _identity_map(AGE_LABELS),
                "SEX": _identity_map(SEX_LABELS),
                "SCHL_allpop": _identity_map(SCHL_LABELS),
                "ESR_allpop": ESR_TO_MID4,
                "EARN_16p_bin": _identity_map(EARN_LABELS),
            },
        )
    if preset in {"fine_2400_age8", "fine_2400_age8_esrfull", "2400_age8", "age8_2400"}:
        return (
            "fine_2400_age8",
            {
                "AGEP_bin": AGE_MID8_LABELS,
                "SEX": SEX_LABELS,
                "SCHL_allpop": SCHL_LABELS,
                "ESR_allpop": ESR_LABELS,
                "EARN_16p_bin": EARN_LABELS,
            },
            {
                "AGEP_bin": AGE_TO_MID8,
                "SEX": _identity_map(SEX_LABELS),
                "SCHL_allpop": _identity_map(SCHL_LABELS),
                "ESR_allpop": _identity_map(ESR_LABELS),
                "EARN_16p_bin": _identity_map(EARN_LABELS),
            },
        )
    raise ValueError(
        "Unsupported SYNTHETIC_CITY_C2F_COARSE_PRESET="
        f"{preset!r}; expected one of main_288, coarse_108, mid_480, fine_720, fine_960, "
        "fine_1440, fine_1800, fine_2400, fine_2400_age8"
    )

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
COARSE_PRESET, COARSE_CATEGORIES, _COARSE_MAPS = _coarse_preset()
COARSE_SHAPE = tuple(len(COARSE_CATEGORIES[v]) for v in COARSE_VARIABLE_ORDER)
COARSE_K = int(np.prod(COARSE_SHAPE))

AGE_FINE_TO_COARSE = np.asarray(
    [COARSE_CATEGORIES["AGEP_bin"].index(_COARSE_MAPS["AGEP_bin"][x]) for x in AGE_LABELS],
    dtype=np.int16,
)
SCHL_FINE_TO_COARSE = np.asarray(
    [COARSE_CATEGORIES["SCHL_allpop"].index(_COARSE_MAPS["SCHL_allpop"][x]) for x in SCHL_LABELS],
    dtype=np.int16,
)
ESR_FINE_TO_COARSE = np.asarray(
    [COARSE_CATEGORIES["ESR_allpop"].index(_COARSE_MAPS["ESR_allpop"][x]) for x in ESR_LABELS],
    dtype=np.int16,
)
EARN_FINE_TO_COARSE = np.asarray(
    [COARSE_CATEGORIES["EARN_16p_bin"].index(_COARSE_MAPS["EARN_16p_bin"][x]) for x in EARN_LABELS],
    dtype=np.int16,
)


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
        "coarse_preset": COARSE_PRESET,
        "AGEP_bin_lite": COARSE_CATEGORIES["AGEP_bin"][int(ac)],
        "SEX": SEX_LABELS[int(si)],
        "SCHL_allpop_lite": COARSE_CATEGORIES["SCHL_allpop"][int(qc)],
        "ESR_allpop_lite": COARSE_CATEGORIES["ESR_allpop"][int(ec)],
        "EARN_16p_bin_lite": COARSE_CATEGORIES["EARN_16p_bin"][int(wc)],
    }


def parent_count_histogram() -> dict[int, int]:
    vals = [int(x.shape[0]) for x in PARENT_TO_CHILD_FULL]
    out: dict[int, int] = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return out
