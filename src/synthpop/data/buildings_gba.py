from __future__ import annotations

"""
GlobalBuildingAtlas (GBA) LoD1 processing.

We intentionally avoid importing geopandas/shapely at module import time,
so that lightweight CLI utilities keep working even without geo deps.
"""

import pathlib


def prepare_buildings_from_gba_lod1(*, gba_tile_geojson: pathlib.Path, out_path: pathlib.Path) -> None:
    raise NotImplementedError(
        "TODO(v0): clip tile to Detroit boundary, fix/record CRS, compute footprint_area_m2 and height features."
    )

