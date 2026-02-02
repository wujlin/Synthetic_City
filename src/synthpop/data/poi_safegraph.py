from __future__ import annotations

"""
SafeGraph POI processing.

SafeGraph is license-sensitive; we only register existing local shards and extract Detroit subsets
into processed outputs (which may still be restricted depending on agreement).
"""

import pathlib


def extract_detroit_pois(*, safegraph_unzip_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    raise NotImplementedError("TODO(v0): filter POIs to Detroit study area and write processed POI table.")

