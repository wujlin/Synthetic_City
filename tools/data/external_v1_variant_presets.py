#!/usr/bin/env python3
from __future__ import annotations

"""
Preset schemas for external-condition v1 refinement ablations.

All variants are projections from the full v1 schema:
  AGEP_bin(10) x SEX(2) x SCHL_allpop(5) x ESR_allpop(5)
"""

from dataclasses import dataclass

from tools.data.build_external_target_v1_michigan import AGE_LABELS, ESR_LABELS, SCHL_LABELS, SEX_LABELS


AGE_LITE_LABELS = ["[0.0, 18.0)", "[18.0, 35.0)", "[35.0, 65.0)", "[65.0, 1000.0)"]
SCHL_LITE_LABELS = ["not_25p", "non_bachelor", "bachelor_plus"]
ESR_LITE_LABELS = ["not_16p", "employed", "not_employed"]

AGE_TO_LITE = {
    "[0.0, 5.0)": "[0.0, 18.0)",
    "[5.0, 18.0)": "[0.0, 18.0)",
    "[18.0, 25.0)": "[18.0, 35.0)",
    "[25.0, 35.0)": "[18.0, 35.0)",
    "[35.0, 45.0)": "[35.0, 65.0)",
    "[45.0, 55.0)": "[35.0, 65.0)",
    "[55.0, 65.0)": "[35.0, 65.0)",
    "[65.0, 75.0)": "[65.0, 1000.0)",
    "[75.0, 85.0)": "[65.0, 1000.0)",
    "[85.0, 1000.0)": "[65.0, 1000.0)",
}

SCHL_TO_LITE = {
    "not_25p": "not_25p",
    "less_than_high_school": "non_bachelor",
    "high_school_or_ged": "non_bachelor",
    "some_college_or_assoc": "non_bachelor",
    "bachelor_plus": "bachelor_plus",
}

ESR_TO_LITE = {
    "not_16p": "not_16p",
    "employed": "employed",
    "unemployed": "not_employed",
    "armed_forces": "not_employed",
    "not_in_labor_force": "not_employed",
}


def _identity_map(labels: list[str]) -> dict[str, str]:
    return {x: x for x in labels}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    categories: dict[str, list[str]]
    mappings: dict[str, dict[str, str]]

    @property
    def variable_order(self) -> list[str]:
        return ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop"]

    @property
    def shape(self) -> list[int]:
        return [len(self.categories[v]) for v in self.variable_order]

    @property
    def K(self) -> int:
        k = 1
        for n in self.shape:
            k *= int(n)
        return k


VARIANT_SPECS: dict[str, VariantSpec] = {
    "lite": VariantSpec(
        name="lite",
        categories={
            "AGEP_bin": AGE_LITE_LABELS,
            "SEX": SEX_LABELS,
            "SCHL_allpop": SCHL_LITE_LABELS,
            "ESR_allpop": ESR_LITE_LABELS,
        },
        mappings={
            "AGEP_bin": AGE_TO_LITE,
            "SEX": _identity_map(SEX_LABELS),
            "SCHL_allpop": SCHL_TO_LITE,
            "ESR_allpop": ESR_TO_LITE,
        },
    ),
    "age_refine": VariantSpec(
        name="age_refine",
        categories={
            "AGEP_bin": AGE_LABELS,
            "SEX": SEX_LABELS,
            "SCHL_allpop": SCHL_LITE_LABELS,
            "ESR_allpop": ESR_LITE_LABELS,
        },
        mappings={
            "AGEP_bin": _identity_map(AGE_LABELS),
            "SEX": _identity_map(SEX_LABELS),
            "SCHL_allpop": SCHL_TO_LITE,
            "ESR_allpop": ESR_TO_LITE,
        },
    ),
    "schl_refine": VariantSpec(
        name="schl_refine",
        categories={
            "AGEP_bin": AGE_LITE_LABELS,
            "SEX": SEX_LABELS,
            "SCHL_allpop": SCHL_LABELS,
            "ESR_allpop": ESR_LITE_LABELS,
        },
        mappings={
            "AGEP_bin": AGE_TO_LITE,
            "SEX": _identity_map(SEX_LABELS),
            "SCHL_allpop": _identity_map(SCHL_LABELS),
            "ESR_allpop": ESR_TO_LITE,
        },
    ),
    "esr_refine": VariantSpec(
        name="esr_refine",
        categories={
            "AGEP_bin": AGE_LITE_LABELS,
            "SEX": SEX_LABELS,
            "SCHL_allpop": SCHL_LITE_LABELS,
            "ESR_allpop": ESR_LABELS,
        },
        mappings={
            "AGEP_bin": AGE_TO_LITE,
            "SEX": _identity_map(SEX_LABELS),
            "SCHL_allpop": SCHL_TO_LITE,
            "ESR_allpop": _identity_map(ESR_LABELS),
        },
    ),
    "age_schl_refine": VariantSpec(
        name="age_schl_refine",
        categories={
            "AGEP_bin": AGE_LABELS,
            "SEX": SEX_LABELS,
            "SCHL_allpop": SCHL_LABELS,
            "ESR_allpop": ESR_LITE_LABELS,
        },
        mappings={
            "AGEP_bin": _identity_map(AGE_LABELS),
            "SEX": _identity_map(SEX_LABELS),
            "SCHL_allpop": _identity_map(SCHL_LABELS),
            "ESR_allpop": ESR_TO_LITE,
        },
    ),
    "age_esr_refine": VariantSpec(
        name="age_esr_refine",
        categories={
            "AGEP_bin": AGE_LABELS,
            "SEX": SEX_LABELS,
            "SCHL_allpop": SCHL_LITE_LABELS,
            "ESR_allpop": ESR_LABELS,
        },
        mappings={
            "AGEP_bin": _identity_map(AGE_LABELS),
            "SEX": _identity_map(SEX_LABELS),
            "SCHL_allpop": SCHL_TO_LITE,
            "ESR_allpop": _identity_map(ESR_LABELS),
        },
    ),
    "schl_esr_refine": VariantSpec(
        name="schl_esr_refine",
        categories={
            "AGEP_bin": AGE_LITE_LABELS,
            "SEX": SEX_LABELS,
            "SCHL_allpop": SCHL_LABELS,
            "ESR_allpop": ESR_LABELS,
        },
        mappings={
            "AGEP_bin": AGE_TO_LITE,
            "SEX": _identity_map(SEX_LABELS),
            "SCHL_allpop": _identity_map(SCHL_LABELS),
            "ESR_allpop": _identity_map(ESR_LABELS),
        },
    ),
    "full": VariantSpec(
        name="full",
        categories={
            "AGEP_bin": AGE_LABELS,
            "SEX": SEX_LABELS,
            "SCHL_allpop": SCHL_LABELS,
            "ESR_allpop": ESR_LABELS,
        },
        mappings={
            "AGEP_bin": _identity_map(AGE_LABELS),
            "SEX": _identity_map(SEX_LABELS),
            "SCHL_allpop": _identity_map(SCHL_LABELS),
            "ESR_allpop": _identity_map(ESR_LABELS),
        },
    ),
}


def get_variant_spec(name: str) -> VariantSpec:
    if name not in VARIANT_SPECS:
        raise SystemExit(f"Unsupported variant={name}. choices={sorted(VARIANT_SPECS)}")
    return VARIANT_SPECS[name]

