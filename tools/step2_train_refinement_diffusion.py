#!/usr/bin/env python3
from __future__ import annotations

"""Manuscript Step 2 Stage 2: train the fine refinement diffusion model."""

import pathlib
import runpy


if __name__ == "__main__":
    runpy.run_path(str(pathlib.Path(__file__).resolve().with_name("train_external_c2f_full_earn_teacher.py")), run_name="__main__")
