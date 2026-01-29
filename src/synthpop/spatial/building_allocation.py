from __future__ import annotations

"""
Explicit spatial allocation (Scheme B):
- The diffusion model generates attributes within a macro geography (PUMA/tract).
- A separate, reviewable allocator maps persons to buildings within the same geography.
"""

from typing import Any


def allocate_to_buildings(
    *,
    persons: Any,
    buildings: Any,
    group_col: str = "puma",
    method: str = "income_price_match",
    income_col: str = "PINCP",
    price_col: str = "price_tier",
    capacity_col: str = "cap_proxy",
    n_tiers: int = 5,
    seed: int = 0,
    return_meta: bool = False,
) -> Any:
    """
    Allocate generated persons to buildings within each group (e.g., PUMA/tract).

    Args:
        persons: pandas DataFrame, must include `group_col` and (for income_price_match) `income_col`.
        buildings: pandas DataFrame, must include `bldg_id`, `group_col`, and
            (for capacity_only/income_price_match) `capacity_col`. For income_price_match also requires `price_col`.
        group_col: grouping column shared by persons/buildings (e.g., "puma" or "tract_geoid").
        method:
            - "random": uniform random within group
            - "capacity_only": sample buildings within group weighted by `capacity_col`
            - "income_price_match": income quantile -> price tier match, then within-tier capacity weighting
        income_col: person income column (used by income_price_match).
        price_col: building tier column (1..n_tiers).
        capacity_col: building capacity proxy column.
        n_tiers: number of tiers used in income quantile mapping.
        seed: RNG seed for reproducibility.
        return_meta: if True, returns (persons_df, meta) where meta records allocation diagnostics
            (e.g., tier fallback frequency).

    Returns:
        persons DataFrame with an added `bldg_id` column.
        If return_meta=True, returns (df, meta).
    """
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("allocate_to_buildings requires pandas and numpy.") from e

    if not isinstance(persons, pd.DataFrame) or not isinstance(buildings, pd.DataFrame):
        raise TypeError("persons/buildings must be pandas DataFrame")

    group_col = str(group_col)
    method = str(method)
    income_col = str(income_col)
    price_col = str(price_col)
    capacity_col = str(capacity_col)
    n_tiers = int(n_tiers)
    if n_tiers <= 0:
        raise ValueError(f"n_tiers must be positive, got: {n_tiers}")

    if group_col not in persons.columns:
        raise ValueError(f"persons missing column: {group_col}")
    if "bldg_id" not in buildings.columns:
        raise ValueError("buildings missing column: bldg_id")
    if group_col not in buildings.columns:
        raise ValueError(f"buildings missing column: {group_col}")

    if method not in {"random", "capacity_only", "income_price_match"}:
        raise ValueError(f"unknown method: {method}")

    if method in {"capacity_only", "income_price_match"} and capacity_col not in buildings.columns:
        raise ValueError(f'buildings missing column required by method="{method}": {capacity_col}')
    if method == "income_price_match" and price_col not in buildings.columns:
        raise ValueError(f'buildings missing column required by method="{method}": {price_col}')
    if method == "income_price_match" and income_col not in persons.columns:
        raise ValueError(f'persons missing column required by method="{method}": {income_col}')

    rng = np.random.default_rng(int(seed))

    persons_g = persons.copy()
    buildings_g = buildings.copy()

    persons_g[group_col] = persons_g[group_col].astype(str)
    buildings_g[group_col] = buildings_g[group_col].astype(str)
    buildings_g["bldg_id"] = buildings_g["bldg_id"].astype(str)

    if capacity_col in buildings_g.columns:
        buildings_g[capacity_col] = pd.to_numeric(buildings_g[capacity_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    if price_col in buildings_g.columns:
        buildings_g[price_col] = pd.to_numeric(buildings_g[price_col], errors="coerce").fillna(0.0).astype(int)
    if income_col in persons_g.columns:
        persons_g[income_col] = pd.to_numeric(persons_g[income_col], errors="coerce").fillna(0.0)

    meta: dict[str, Any] = {
        "method": method,
        "group_col": group_col,
        "seed": int(seed),
        "n_persons": int(persons_g.shape[0]),
        "n_buildings": int(buildings_g.shape[0]),
        "n_groups_persons": int(persons_g[group_col].nunique(dropna=False)),
        "n_groups_buildings": int(buildings_g[group_col].nunique(dropna=False)),
    }
    if method == "income_price_match":
        meta["n_tiers"] = int(n_tiers)

    assigned = np.full((int(persons_g.shape[0]),), None, dtype=object)

    # Pre-group buildings once.
    bldg_by_group: dict[str, Any] = {}
    for g, gb in buildings_g.groupby(group_col, sort=False):
        bldg_by_group[str(g)] = gb

    fallback_assignments = 0
    no_tier_groups = 0
    no_tier_assignments = 0

    for g, gp in persons_g.groupby(group_col, sort=False):
        g = str(g)
        gb = bldg_by_group.get(g)
        if gb is None or gb.shape[0] == 0:
            continue

        b_ids = gb["bldg_id"].to_numpy(dtype=object)

        if method == "random":
            idx = rng.integers(0, len(b_ids), size=int(gp.shape[0]))
            assigned[gp.index.to_numpy(dtype=int)] = b_ids[idx]
            continue

        weights = gb[capacity_col].to_numpy(dtype=float)
        s = float(weights.sum())
        probs = (weights / s) if s > 0 else None

        if method == "capacity_only":
            idx = rng.choice(len(b_ids), size=int(gp.shape[0]), replace=True, p=probs)
            assigned[gp.index.to_numpy(dtype=int)] = b_ids[idx]
            continue

        # income_price_match
        tiers_b = gb[price_col].to_numpy(dtype=int)
        avail = np.array(sorted({int(t) for t in tiers_b.tolist() if int(t) > 0}), dtype=int)
        if avail.size == 0:
            no_tier_groups += 1
            no_tier_assignments += int(gp.shape[0])
            idx = rng.choice(len(b_ids), size=int(gp.shape[0]), replace=True, p=probs)
            assigned[gp.index.to_numpy(dtype=int)] = b_ids[idx]
            continue

        pools: dict[int, tuple[Any, Any]] = {}
        for t in avail.tolist():
            pool_pos = np.where(tiers_b == int(t))[0]
            if pool_pos.size == 0:
                continue
            w = None
            if probs is not None:
                w = probs[pool_pos].astype(float)
                s2 = float(w.sum())
                if s2 > 0:
                    w = w / s2
                else:
                    w = None
            pools[int(t)] = (pool_pos, w)

        gp_sorted = gp.sort_values(income_col, ascending=True)
        n = int(gp_sorted.shape[0])
        q = (np.arange(n) + 0.5) / max(n, 1)
        desired = np.ceil(q * float(n_tiers)).astype(int)
        desired = np.clip(desired, 1, int(n_tiers))

        chosen_pos: list[int] = []
        for d in desired.tolist():
            nearest = int(avail[np.argmin(np.abs(avail - int(d)))])
            if nearest != int(d):
                fallback_assignments += 1
            pool_pos, w = pools.get(nearest, (np.arange(len(b_ids)), None))
            j = int(rng.choice(pool_pos, p=w))
            chosen_pos.append(j)

        person_pos_sorted = gp_sorted.index.to_numpy(dtype=int)
        assigned[person_pos_sorted] = b_ids[np.array(chosen_pos, dtype=int)]

    out = persons.copy()
    out["bldg_id"] = assigned.tolist()

    n_unassigned = int(np.sum(pd.isna(assigned)))
    meta["n_unassigned"] = n_unassigned
    meta["n_assigned"] = int(persons_g.shape[0]) - n_unassigned
    if method == "income_price_match":
        denom = max(int(meta["n_assigned"]), 1)
        meta["tier_fallback"] = {
            "fallback_assignments": int(fallback_assignments),
            "fallback_rate": float(fallback_assignments / denom),
            "no_tier_groups": int(no_tier_groups),
            "no_tier_assignments": int(no_tier_assignments),
        }

    if return_meta:
        return out, meta
    return out
