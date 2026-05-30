#!/usr/bin/env python3
from __future__ import annotations

"""Manuscript Step 2 Stage 1: train the coarse joint-distribution diffusion model."""

import pathlib
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(pathlib.Path(__file__).resolve().parents[1] / "model" / "train_external_c2f_full_earn_stage1_coarse.py"),
        run_name="__main__",
    )
