from __future__ import annotations

"""
Condition vectors for the diffusion-based hierarchical generation.

Design intent:
- Region conditions summarize macro constraints and built-environment context.
- Building conditions summarize local context (area/height/landuse/POI neighborhood).

v0 (Detroit) encoding policy (PI review aligned):
- Do NOT treat `building_id` as a categorical variable in the diffusion model (too many categories).
- Building must NOT be a pure post-processing assignment; spatial anchoring should be part of generation.
  Practically, this means: sample with building context, i.e. generate
  P(attrs | macro_condition, building_feature) and carry `bldg_id` with the sample.
- If an approximate staged implementation is needed, it must keep feedback (resample/reweight)
  and be described as an approximation rather than the core innovation.
"""

import pathlib


def build_region_conditions(*, geo_units_path: pathlib.Path, marginals_path: pathlib.Path, out_path: pathlib.Path) -> None:
    raise NotImplementedError("TODO(v0): build region-level condition vectors.")


def build_building_conditions(*, buildings_path: pathlib.Path, poi_path: pathlib.Path | None, out_path: pathlib.Path) -> None:
    raise NotImplementedError("TODO(v0): build building-level condition vectors.")
