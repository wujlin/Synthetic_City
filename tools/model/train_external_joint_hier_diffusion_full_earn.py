#!/usr/bin/env python3
from __future__ import annotations

"""
Shared-latent hierarchical diffusion trainer for the 5-variable full external schema:

  AGEP_fine(10) x SEX(2) x SCHL_fine(5) x ESR_fine(5) x EARN_fine(6) = K=3000

The coarse auxiliary head is extended to include a lite earnings axis:

  AGEP_lite(4) x SEX(2) x SCHL_lite(3) x ESR_lite(3) x EARN_lite(4) = K=288

This keeps diffusion as the main generator while giving the added earnings axis
an explicit hierarchical anchor during training.
"""

import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.model.train_external_joint_hier_diffusion_full as base
from tools.data.build_external_target_v1_michigan import AGE_LABELS, ESR_LABELS, SCHL_LABELS, SEX_LABELS
from tools.data.external_earn_v1_schema import EARN_LABELS
from tools.data.external_v1_variant_presets import AGE_LITE_LABELS, AGE_TO_LITE, ESR_LITE_LABELS, ESR_TO_LITE, SCHL_LITE_LABELS, SCHL_TO_LITE


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


base.FINE_VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop", "EARN_16p_bin"]
base.COARSE_VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop", "EARN_16p_bin"]
base.FINE_CATEGORIES = {
    "AGEP_bin": AGE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LABELS,
    "ESR_allpop": ESR_LABELS,
    "EARN_16p_bin": EARN_LABELS,
}
base.COARSE_CATEGORIES = {
    "AGEP_bin": AGE_LITE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LITE_LABELS,
    "ESR_allpop": ESR_LITE_LABELS,
    "EARN_16p_bin": EARN_LITE_LABELS,
}
base.FINE_SHAPE = tuple(len(base.FINE_CATEGORIES[v]) for v in base.FINE_VARIABLE_ORDER)
base.COARSE_SHAPE = tuple(len(base.COARSE_CATEGORIES[v]) for v in base.COARSE_VARIABLE_ORDER)
base.FINE_K = int(np.prod(base.FINE_SHAPE))
base.COARSE_K = int(np.prod(base.COARSE_SHAPE))


def _build_fine_to_coarse_matrix_full_earn() -> np.ndarray:
    age_coarse_idx = {lab: i for i, lab in enumerate(AGE_LITE_LABELS)}
    schl_coarse_idx = {lab: i for i, lab in enumerate(SCHL_LITE_LABELS)}
    esr_coarse_idx = {lab: i for i, lab in enumerate(ESR_LITE_LABELS)}
    earn_coarse_idx = {lab: i for i, lab in enumerate(EARN_LITE_LABELS)}
    out = np.zeros((base.FINE_K, base.COARSE_K), dtype=np.float32)
    k = 0
    for age_lab in base.FINE_CATEGORIES["AGEP_bin"]:
        ac = age_coarse_idx[AGE_TO_LITE[age_lab]]
        for si, _ in enumerate(base.FINE_CATEGORIES["SEX"]):
            for schl_lab in base.FINE_CATEGORIES["SCHL_allpop"]:
                qc = schl_coarse_idx[SCHL_TO_LITE[schl_lab]]
                for esr_lab in base.FINE_CATEGORIES["ESR_allpop"]:
                    ec = esr_coarse_idx[ESR_TO_LITE[esr_lab]]
                    for earn_lab in base.FINE_CATEGORIES["EARN_16p_bin"]:
                        wc = earn_coarse_idx[EARN_TO_LITE[earn_lab]]
                        kc = np.ravel_multi_index((ac, si, qc, ec, wc), base.COARSE_SHAPE)
                        out[k, kc] = 1.0
                        k += 1
    return out


base._build_fine_to_coarse_matrix_full = _build_fine_to_coarse_matrix_full_earn


if __name__ == "__main__":
    base.main()
