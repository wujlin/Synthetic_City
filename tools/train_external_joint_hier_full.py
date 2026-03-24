#!/usr/bin/env python3
from __future__ import annotations

"""
Thin wrapper that reuses the validated shared-latent hierarchical trainer
from the age_schl_refine setting, but switches the fine schema to the full
external v1 setting:

  AGEP_fine(10) x SEX(2) x SCHL_fine(5) x ESR_fine(5) = K=500

The coarse head remains the lite schema:

  AGEP_lite(4) x SEX(2) x SCHL_lite(3) x ESR_lite(3) = K=72

This lets us test whether the shared regional latent still helps once the
employment axis is fully refined, without rewriting the trainer from scratch.
"""

import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.train_external_joint_hier_age_schl as base
from tools.build_external_target_v1_michigan import AGE_LABELS, ESR_LABELS, SCHL_LABELS, SEX_LABELS
from tools.external_v1_variant_presets import AGE_LITE_LABELS, AGE_TO_LITE, ESR_LITE_LABELS, ESR_TO_LITE, SCHL_LITE_LABELS, SCHL_TO_LITE


base.FINE_CATEGORIES = {
    "AGEP_bin": AGE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LABELS,
    "ESR_allpop": ESR_LABELS,
}
base.COARSE_CATEGORIES = {
    "AGEP_bin": AGE_LITE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LITE_LABELS,
    "ESR_allpop": ESR_LITE_LABELS,
}
base.FINE_SHAPE = tuple(len(base.FINE_CATEGORIES[v]) for v in base.FINE_VARIABLE_ORDER)
base.COARSE_SHAPE = tuple(len(base.COARSE_CATEGORIES[v]) for v in base.COARSE_VARIABLE_ORDER)
base.FINE_K = int(np.prod(base.FINE_SHAPE))
base.COARSE_K = int(np.prod(base.COARSE_SHAPE))


def _build_fine_to_coarse_matrix_full() -> np.ndarray:
    age_coarse_idx = {lab: i for i, lab in enumerate(AGE_LITE_LABELS)}
    schl_coarse_idx = {lab: i for i, lab in enumerate(SCHL_LITE_LABELS)}
    esr_coarse_idx = {lab: i for i, lab in enumerate(ESR_LITE_LABELS)}
    out = np.zeros((base.FINE_K, base.COARSE_K), dtype=np.float32)
    k = 0
    for age_lab in base.FINE_CATEGORIES["AGEP_bin"]:
        ac = age_coarse_idx[AGE_TO_LITE[age_lab]]
        for si, _ in enumerate(base.FINE_CATEGORIES["SEX"]):
            for schl_lab in base.FINE_CATEGORIES["SCHL_allpop"]:
                qc = schl_coarse_idx[SCHL_TO_LITE[schl_lab]]
                for esr_lab in base.FINE_CATEGORIES["ESR_allpop"]:
                    ec = esr_coarse_idx[ESR_TO_LITE[esr_lab]]
                    kc = np.ravel_multi_index((ac, si, qc, ec), base.COARSE_SHAPE)
                    out[k, kc] = 1.0
                    k += 1
    return out


base._build_fine_to_coarse_matrix = _build_fine_to_coarse_matrix_full


if __name__ == "__main__":
    base.main()
