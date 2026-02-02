#!/usr/bin/env python3
from __future__ import annotations

"""
Compatibility wrapper: tract-level conditional diffusion with ACS supervision.

We keep the implementation in a single file:
  - tools/poc_tabddpm_acs_supervised_b01001.py

This wrapper exists because some experiment notes refer to:
  - tools/poc_tabddpm_acs_tract.py

All CLI arguments are passed through unchanged.
"""

import pathlib
import runpy


def main() -> None:
    here = pathlib.Path(__file__).resolve()
    impl = here.with_name("poc_tabddpm_acs_supervised_b01001.py")
    runpy.run_path(str(impl), run_name="__main__")


if __name__ == "__main__":
    main()

