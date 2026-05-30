#!/usr/bin/env python3
from __future__ import annotations

"""Release: export state-level public CSV files with the manuscript schema."""

import pathlib
import runpy


if __name__ == "__main__":
    runpy.run_path(str(pathlib.Path(__file__).resolve().with_name("export_paper1_release_csv.py")), run_name="__main__")
