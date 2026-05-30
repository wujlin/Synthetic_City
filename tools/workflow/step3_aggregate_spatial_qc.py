#!/usr/bin/env python3
from __future__ import annotations

"""Manuscript Step 3: aggregate national spatial-assignment QC metrics."""

import pathlib
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(pathlib.Path(__file__).resolve().parents[1] / "spatial" / "aggregate_paper1_spatial_national_qc.py"),
        run_name="__main__",
    )
