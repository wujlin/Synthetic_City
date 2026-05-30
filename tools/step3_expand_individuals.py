#!/usr/bin/env python3
from __future__ import annotations

"""Manuscript Step 3: sample synthetic individuals from predicted PUMA joints."""

import pathlib
import runpy


if __name__ == "__main__":
    runpy.run_path(str(pathlib.Path(__file__).resolve().with_name("exp_phase2_expand_to_persons.py")), run_name="__main__")
