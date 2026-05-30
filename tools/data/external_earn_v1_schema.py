#!/usr/bin/env python3
from __future__ import annotations

"""
Shared schema helpers for the Michigan external earnings-proxy experiment.
"""

from typing import Any

import numpy as np


EARN_LABELS = [
    "not_in_earnings_universe",
    "lt_25k",
    "25k_50k",
    "50k_75k",
    "75k_100k",
    "ge_100k",
]

B20001_MALE_COLS = [f"B20001_{i:03d}E" for i in range(3, 23)]
B20001_FEMALE_COLS = [f"B20001_{i:03d}E" for i in range(24, 44)]

# Standard ACS B20001 detailed earnings bins represented by their lower bounds.
B20001_LO_BOUNDS = np.asarray(
    [
        0.0,
        2500.0,
        5000.0,
        7500.0,
        10000.0,
        12500.0,
        15000.0,
        17500.0,
        20000.0,
        22500.0,
        25000.0,
        30000.0,
        35000.0,
        40000.0,
        45000.0,
        50000.0,
        55000.0,
        65000.0,
        75000.0,
        100000.0,
    ],
    dtype=float,
)


def coarse_b20001_groups() -> dict[str, list[int]]:
    return {
        "lt_25k": [int(i) for i, lo in enumerate(B20001_LO_BOUNDS.tolist()) if lo < 25_000.0],
        "25k_50k": [int(i) for i, lo in enumerate(B20001_LO_BOUNDS.tolist()) if 25_000.0 <= lo < 50_000.0],
        "50k_75k": [int(i) for i, lo in enumerate(B20001_LO_BOUNDS.tolist()) if 50_000.0 <= lo < 75_000.0],
        "75k_100k": [int(i) for i, lo in enumerate(B20001_LO_BOUNDS.tolist()) if 75_000.0 <= lo < 100_000.0],
        "ge_100k": [int(i) for i, lo in enumerate(B20001_LO_BOUNDS.tolist()) if lo >= 100_000.0],
    }


def bin_earn_allpop(age: np.ndarray, earn: np.ndarray) -> np.ndarray:
    out = np.zeros(age.shape, dtype=np.int16)
    mask = np.isfinite(age) & (age >= 16.0) & np.isfinite(earn) & (earn > 0.0)
    out[mask & (earn < 25_000.0)] = 1
    out[mask & (earn >= 25_000.0) & (earn < 50_000.0)] = 2
    out[mask & (earn >= 50_000.0) & (earn < 75_000.0)] = 3
    out[mask & (earn >= 75_000.0) & (earn < 100_000.0)] = 4
    out[mask & (earn >= 100_000.0)] = 5
    out[~np.isfinite(age)] = -1
    return out


def b20001_schema_present(df: Any) -> bool:
    cols = set(df.columns.astype(str).tolist())
    return all(c in cols for c in B20001_MALE_COLS + B20001_FEMALE_COLS + ["B20001_001E"])
