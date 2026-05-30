#!/usr/bin/env python3
from __future__ import annotations

"""Manuscript Step 1: build PUMA-level five-attribute joint targets."""

import pathlib
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(pathlib.Path(__file__).resolve().parents[1] / "data" / "build_external_target_v1_full_earn.py"),
        run_name="__main__",
    )
