#!/usr/bin/env python3
from __future__ import annotations

"""Manuscript Step 1: build POI/LODES PUMA-level spatial representation features."""

import pathlib
import runpy


if __name__ == "__main__":
    runpy.run_path(str(pathlib.Path(__file__).resolve().with_name("build_puma_spatial_features.py")), run_name="__main__")
