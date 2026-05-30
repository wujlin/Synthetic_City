#!/usr/bin/env python3
from __future__ import annotations

"""Manuscript Step 3: assign workers to LODES destination tracts."""

import pathlib
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(pathlib.Path(__file__).resolve().parents[1] / "spatial" / "exp_phase3b_assign_work_destinations.py"),
        run_name="__main__",
    )
