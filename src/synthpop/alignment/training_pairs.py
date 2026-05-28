from __future__ import annotations

"""
Utilities to construct training pairs for alignment / joint diffusion (Scheme C-v2).

This module will turn aligned person/device/building embeddings into:
- contrastive positive/negative pairs
- (z_person, z_building, condition) tuples for joint diffusion training
"""

from dataclasses import dataclass
from typing import Any


def _require_numpy_pandas() -> tuple[Any, Any]:
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("training_pairs.py requires numpy and pandas.") from e
    return np, pd


@dataclass(frozen=True)
class TrainingPairsMeta:
    n_persons: int
    n_devices: int
    n_buildings: int
    k_soft_labels: int
    device_fallback_global: int
    building_fallback_global: int
    empty_person_group: int
    empty_device_group: int
    empty_building_group: int


def build_training_pairs(
    *,
    persons: Any,
    devices: Any,
    buildings: Any,
    z_persons: Any,
    z_devices: Any,
    z_buildings: Any,
    person_cbg_col: str = "cbg_geoid",
    device_cbg_col: str = "CENSUS_BLOCK_GROUP",
    building_cbg_col: str = "cbg_geoid",
    k_soft_labels: int = 5,
    seed: int = 0,
    temperature: float = 1.0,
    return_meta: bool = False,
) -> Any:
    """
    Build (z_person, z_building, cbg_id) training pairs using soft labels.

    Logic:
    1) For each person, find the closest device within the same CBG using
       latent L2 distance; fall back to the global nearest device if needed.
    2) For that device, sample k buildings within the same CBG with
       softmax(-dist^2 / temperature); fall back to global buildings if needed.

    Input conventions:
    - persons/devices/buildings are pandas DataFrame objects;
    - z_* are numpy arrays whose row counts match the corresponding frames;
    - CBG columns are strings or castable to strings.

    Default return:
      z_person_out: (n*k, d)
      z_building_out: (n*k, d)
      cbg_ids: (n*k,)

    If return_meta=True, return a 4-tuple with TrainingPairsMeta at the end.
    """
    np, pd = _require_numpy_pandas()

    if k_soft_labels <= 0:
        raise ValueError("k_soft_labels must be > 0")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    if not isinstance(persons, pd.DataFrame) or not isinstance(devices, pd.DataFrame) or not isinstance(buildings, pd.DataFrame):
        raise TypeError("persons/devices/buildings must be pandas DataFrame")
    for df, col, name in [
        (persons, person_cbg_col, "persons"),
        (devices, device_cbg_col, "devices"),
        (buildings, building_cbg_col, "buildings"),
    ]:
        if col not in df.columns:
            raise ValueError(f"{name} missing group col: {col}")

    z_p = np.asarray(z_persons)
    z_d = np.asarray(z_devices)
    z_b = np.asarray(z_buildings)

    if z_p.ndim != 2 or z_d.ndim != 2 or z_b.ndim != 2:
        raise ValueError("z_persons/z_devices/z_buildings must be 2D arrays")
    if z_p.shape[0] != len(persons):
        raise ValueError("len(persons) must match z_persons.shape[0]")
    if z_d.shape[0] != len(devices):
        raise ValueError("len(devices) must match z_devices.shape[0]")
    if z_b.shape[0] != len(buildings):
        raise ValueError("len(buildings) must match z_buildings.shape[0]")
    if z_p.shape[1] != z_d.shape[1] or z_p.shape[1] != z_b.shape[1]:
        raise ValueError("All latent vectors must share the same dimension d")

    n_persons = int(len(persons))
    d_latent = int(z_p.shape[1])

    persons_cbg = persons[person_cbg_col].astype(str).fillna("").to_numpy()
    devices_cbg = devices[device_cbg_col].astype(str).fillna("").to_numpy()
    buildings_cbg = buildings[building_cbg_col].astype(str).fillna("").to_numpy()

    device_by_cbg: dict[str, list[int]] = {}
    for idx, g in enumerate(devices_cbg.tolist()):
        gg = str(g)
        if gg == "" or gg.lower() == "nan":
            continue
        device_by_cbg.setdefault(gg, []).append(int(idx))

    building_by_cbg: dict[str, list[int]] = {}
    for idx, g in enumerate(buildings_cbg.tolist()):
        gg = str(g)
        if gg == "" or gg.lower() == "nan":
            continue
        building_by_cbg.setdefault(gg, []).append(int(idx))

    all_device_idx = np.arange(len(devices), dtype=int)
    all_building_idx = np.arange(len(buildings), dtype=int)

    rng = np.random.default_rng(int(seed))

    z_person_out = np.empty((n_persons * int(k_soft_labels), d_latent), dtype=float)
    z_building_out = np.empty((n_persons * int(k_soft_labels), d_latent), dtype=float)
    cbg_ids_out = np.empty((n_persons * int(k_soft_labels),), dtype=object)

    device_fallback_global = 0
    building_fallback_global = 0
    empty_person_group = 0
    empty_device_group = 0
    empty_building_group = 0

    for i in range(n_persons):
        g = str(persons_cbg[i])
        if g == "" or g.lower() == "nan":
            empty_person_group += 1
            g = ""

        dev_idx_list = device_by_cbg.get(g)
        if not dev_idx_list:
            empty_device_group += 1
            dev_idx_list = all_device_idx.tolist()
            device_fallback_global += 1

        cand_dev = np.asarray(dev_idx_list, dtype=int)
        # Nearest device to person in latent space.
        diff = z_d[cand_dev] - z_p[i]
        d2 = (diff * diff).sum(axis=1)
        j = int(cand_dev[int(d2.argmin())])

        bldg_idx_list = building_by_cbg.get(g)
        if not bldg_idx_list:
            empty_building_group += 1
            bldg_idx_list = all_building_idx.tolist()
            building_fallback_global += 1

        cand_b = np.asarray(bldg_idx_list, dtype=int)
        diff_b = z_b[cand_b] - z_d[j]
        d2_b = (diff_b * diff_b).sum(axis=1)
        # Soft weights from latent distance.
        w = np.exp(-d2_b / float(temperature))
        w_sum = float(w.sum())
        if not (w_sum > 0 and np.isfinite(w_sum)):
            p = None
        else:
            p = w / w_sum

        chosen = rng.choice(len(cand_b), size=int(k_soft_labels), replace=True, p=p)
        base = i * int(k_soft_labels)
        for t, local in enumerate(chosen.tolist()):
            out_idx = base + t
            b_idx = int(cand_b[int(local)])
            z_person_out[out_idx] = z_p[i]
            z_building_out[out_idx] = z_b[b_idx]
            cbg_ids_out[out_idx] = g

    meta = TrainingPairsMeta(
        n_persons=n_persons,
        n_devices=int(len(devices)),
        n_buildings=int(len(buildings)),
        k_soft_labels=int(k_soft_labels),
        device_fallback_global=int(device_fallback_global),
        building_fallback_global=int(building_fallback_global),
        empty_person_group=int(empty_person_group),
        empty_device_group=int(empty_device_group),
        empty_building_group=int(empty_building_group),
    )

    if return_meta:
        return z_person_out, z_building_out, cbg_ids_out, meta
    return z_person_out, z_building_out, cbg_ids_out


def build_device_building_pairs(*args: Any, **kwargs: Any) -> Any:
    """
    Backward-compatible alias (old placeholder name).
    """
    return build_training_pairs(*args, **kwargs)
