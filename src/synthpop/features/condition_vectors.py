from __future__ import annotations

"""
Condition vectors for the diffusion-based hierarchical generation.

Design intent:
- Region conditions summarize macro constraints and built-environment context.
- Building conditions summarize local context (area/height/landuse/POI neighborhood).

v0 (Detroit) encoding policy (PI review aligned):
- Do NOT treat `building_id` as a categorical variable in the diffusion model (too many categories).
- Use a two-stage strategy: generate person attributes + tract/BG, then assign buildings within BG by capacity proxies.
- Building conditions (if used in later versions) should start as a continuous feature vector, not an ID embedding.
"""

import pathlib


def build_region_conditions(*, geo_units_path: pathlib.Path, marginals_path: pathlib.Path, out_path: pathlib.Path) -> None:
    raise NotImplementedError("TODO(v0): build region-level condition vectors.")


def build_building_conditions(*, buildings_path: pathlib.Path, poi_path: pathlib.Path | None, out_path: pathlib.Path) -> None:
    raise NotImplementedError("TODO(v0): build building-level condition vectors.")
