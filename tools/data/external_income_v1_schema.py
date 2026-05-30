#!/usr/bin/env python3
from __future__ import annotations

"""
Shared schema helpers for the external full-income experiment.
"""

from typing import Any

import numpy as np


INCOME_LABELS = [
    "not_15p",
    "no_income_15p",
    "lt_10k_or_loss",
    "10k_15k",
    "15k_25k",
    "25k_35k",
    "35k_50k",
    "50k_65k",
    "65k_75k",
    "75k_plus",
]

INCOME_REGIME_LABELS = [
    "not_15p",
    "no_income_15p",
    "positive_income_15p",
]

INCOME_LITE_LABELS = [
    "not_15p",
    "no_income_15p",
    "lt_15k_or_loss",
    "15k_50k",
    "50k_75k",
    "75k_plus",
]

INCOME_TO_LITE = {
    "not_15p": "not_15p",
    "no_income_15p": "no_income_15p",
    "lt_10k_or_loss": "lt_15k_or_loss",
    "10k_15k": "lt_15k_or_loss",
    "15k_25k": "15k_50k",
    "25k_35k": "15k_50k",
    "35k_50k": "15k_50k",
    "50k_65k": "50k_75k",
    "65k_75k": "50k_75k",
    "75k_plus": "75k_plus",
}

INCOME_TO_REGIME = {
    "not_15p": "not_15p",
    "no_income_15p": "no_income_15p",
    "lt_10k_or_loss": "positive_income_15p",
    "10k_15k": "positive_income_15p",
    "15k_25k": "positive_income_15p",
    "25k_35k": "positive_income_15p",
    "35k_50k": "positive_income_15p",
    "50k_65k": "positive_income_15p",
    "65k_75k": "positive_income_15p",
    "75k_plus": "positive_income_15p",
}

B06010_TOTAL_COL = "B06010_001E"
B06010_NO_INCOME_COL = "B06010_002E"
B06010_WITH_INCOME_COL = "B06010_003E"
B06010_INCOME_COLS = {
    "no_income_15p": [B06010_NO_INCOME_COL],
    "lt_10k_or_loss": ["B06010_004E"],
    "10k_15k": ["B06010_005E"],
    "15k_25k": ["B06010_006E"],
    "25k_35k": ["B06010_007E"],
    "35k_50k": ["B06010_008E"],
    "50k_65k": ["B06010_009E"],
    "65k_75k": ["B06010_010E"],
    "75k_plus": ["B06010_011E"],
}


def bin_income_allpop(age: np.ndarray, income: np.ndarray) -> np.ndarray:
    out = np.zeros(age.shape, dtype=np.int16)
    valid_age = np.isfinite(age)
    mask15 = valid_age & (age >= 15.0) & np.isfinite(income)
    out[mask15 & (income == 0.0)] = 1
    out[mask15 & ((income < 0.0) | ((income > 0.0) & (income < 10_000.0)))] = 2
    out[mask15 & (income >= 10_000.0) & (income < 15_000.0)] = 3
    out[mask15 & (income >= 15_000.0) & (income < 25_000.0)] = 4
    out[mask15 & (income >= 25_000.0) & (income < 35_000.0)] = 5
    out[mask15 & (income >= 35_000.0) & (income < 50_000.0)] = 6
    out[mask15 & (income >= 50_000.0) & (income < 65_000.0)] = 7
    out[mask15 & (income >= 65_000.0) & (income < 75_000.0)] = 8
    out[mask15 & (income >= 75_000.0)] = 9
    out[~valid_age] = -1
    return out


def b06010_schema_present(df: Any) -> bool:
    cols = set(df.columns.astype(str).tolist())
    req = {B06010_TOTAL_COL, B06010_NO_INCOME_COL, B06010_WITH_INCOME_COL}
    req |= {c for cols_ in B06010_INCOME_COLS.values() for c in cols_}
    return req <= cols
