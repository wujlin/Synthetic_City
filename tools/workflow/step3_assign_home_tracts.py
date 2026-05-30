#!/usr/bin/env python3
from __future__ import annotations

"""Manuscript Step 3: allocate sampled residents to home census tracts."""

import pathlib
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(pathlib.Path(__file__).resolve().parents[1] / "spatial" / "exp_phase2_puma_to_small_area.py"),
        run_name="__main__",
    )
