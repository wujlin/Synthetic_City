from __future__ import annotations

"""
US Census / ACS / PUMS processing utilities.

Note: raw downloads are handled by tools/detroit_fetch_public_data.py (KISS, stdlib-only).
This module focuses on parsing and schema normalization (raw → processed).
"""

import pathlib


def prepare_acs_marginals(*, raw_acs_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    raise NotImplementedError("TODO(v0): build processed/marginals/marginals_long.parquet")


def prepare_pums_seed(*, raw_pums_dir: pathlib.Path, out_households: pathlib.Path, out_persons: pathlib.Path) -> None:
    raise NotImplementedError("TODO(v0): clean PUMS household/person tables for training/conditioning.")

