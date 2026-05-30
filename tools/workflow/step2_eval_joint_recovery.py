#!/usr/bin/env python3
from __future__ import annotations

"""Manuscript Step 2: evaluate recovered five-attribute joint distributions."""

import pathlib
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(pathlib.Path(__file__).resolve().parents[1] / "model" / "eval_external_c2f_full_earn_pipeline.py"),
        run_name="__main__",
    )
