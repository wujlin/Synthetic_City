#!/usr/bin/env python3
from __future__ import annotations

"""Manuscript Step 3: place home and workplace coordinates on road-supported points."""

import pathlib
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(pathlib.Path(__file__).resolve().parents[1] / "spatial" / "exp_phase3_road_locations.py"),
        run_name="__main__",
    )
