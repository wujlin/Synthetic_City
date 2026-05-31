from __future__ import annotations

"""
Geographic units (TIGER/Line, city boundary) processing.

v0: Only defines intended interfaces; actual conversion to GeoParquet will be implemented incrementally.
"""

import pathlib


def prepare_geo_units(*, tiger_zip_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    raise NotImplementedError("TODO(v0): TIGER zip -> GeoParquet; clip to Detroit study area.")

