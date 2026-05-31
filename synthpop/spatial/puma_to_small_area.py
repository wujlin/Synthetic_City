from __future__ import annotations

"""
PUMA -> small-area disaggregation utilities.

Design goal:
- take a PUMA-level joint distribution over person types
- combine it with tract/CBG-level marginals and optional spatial priors
- solve a maximum-entropy style allocation of type counts to small areas

This module keeps the solver explicit and reviewable. It is intentionally
independent of the diffusion trainers.
"""

import itertools
import json
import pathlib
import re
from typing import Any, Mapping


def _require_numpy_pandas() -> tuple[Any, Any]:
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("puma_to_small_area.py requires numpy and pandas.") from e
    return np, pd


def _load_schema(schema: Any) -> dict[str, Any]:
    if isinstance(schema, dict):
        out = dict(schema)
    else:
        path = pathlib.Path(schema).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(str(path))
        out = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(out, dict):
        raise ValueError("schema must resolve to a dict-like JSON object")
    if "variable_order" not in out or "categories" not in out:
        raise ValueError("schema missing required keys: variable_order, categories")
    variable_order = [str(v) for v in out["variable_order"]]
    categories = {str(k): [str(x) for x in list(v)] for k, v in dict(out["categories"]).items()}
    for var in variable_order:
        if var not in categories:
            raise ValueError(f"schema categories missing variable: {var}")
    out["variable_order"] = variable_order
    out["categories"] = categories
    out["shape"] = [len(categories[v]) for v in variable_order]
    return out


def build_type_catalog(
    *,
    schema: Any,
    type_idx_col: str = "type_idx",
) -> Any:
    """
    Expand a joint schema into an explicit type table.
    """
    _, pd = _require_numpy_pandas()
    schema_obj = _load_schema(schema)
    variable_order = list(schema_obj["variable_order"])
    categories = dict(schema_obj["categories"])

    rows: list[dict[str, Any]] = []
    for idx, values in enumerate(itertools.product(*[categories[v] for v in variable_order])):
        row = {str(type_idx_col): int(idx)}
        for var, value in zip(variable_order, values):
            row[str(var)] = str(value)
        rows.append(row)
    out = pd.DataFrame(rows)
    return _augment_type_catalog_with_derived_variables(out)


def _interval_left_edge(label: Any) -> float | None:
    text = str(label).strip()
    m = re.match(r"^[\[\(]\s*([-+]?[0-9]*\.?[0-9]+)\s*,", text)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _augment_type_catalog_with_derived_variables(type_catalog: Any) -> Any:
    _, pd = _require_numpy_pandas()
    if not isinstance(type_catalog, pd.DataFrame):
        raise TypeError("type_catalog must be a pandas DataFrame")

    out = type_catalog.copy()
    cols = set(out.columns.astype(str).tolist())

    age_left = None
    if "AGEP_bin" in cols:
        age_left = out["AGEP_bin"].map(_interval_left_edge)

    if {"AGEP_bin", "SEX"} <= cols and "AGEP_SEX_cross" not in cols:
        out["AGEP_SEX_cross"] = out["AGEP_bin"].astype(str) + "__" + out["SEX"].astype(str)

    if age_left is not None and "ESR_allpop" in cols and "ESR_16p" not in cols:
        src = out["ESR_allpop"].astype(str)
        mask16 = age_left.fillna(-1.0) >= 16.0
        derived = pd.Series(["not_16p"] * int(out.shape[0]), index=out.index, dtype=object)
        derived.loc[mask16] = src.loc[mask16].replace(
            {
                "not_16p": "not_16p",
                "employed": "employed",
                "unemployed": "unemployed",
                "armed_forces": "armed_forces",
                "not_in_labor_force": "not_in_labor_force",
            }
        )
        out["ESR_16p"] = derived.astype(str)

    if age_left is not None and "SCHL_allpop" in cols and "SCHL_25p" not in cols:
        src = out["SCHL_allpop"].astype(str)
        mask25 = age_left.fillna(-1.0) >= 25.0
        derived = pd.Series(["not_25p"] * int(out.shape[0]), index=out.index, dtype=object)
        derived.loc[mask25] = src.loc[mask25]
        out["SCHL_25p"] = derived.astype(str)

    if age_left is not None and "EARN_16p_bin" in cols and "PINCP_16p_bin" not in cols:
        src = out["EARN_16p_bin"].astype(str)
        mask16 = age_left.fillna(-1.0) >= 16.0
        in_universe = mask16 & (~src.isin({"", "nan", "None", "not_in_earnings_universe"}))
        derived = pd.Series(["not_in_pincome_universe"] * int(out.shape[0]), index=out.index, dtype=object)
        derived.loc[in_universe] = src.loc[in_universe]
        out["PINCP_16p_bin"] = derived.astype(str)

    return out


def _joint_prob_cols(df: Any, *, expected_k: int) -> list[str]:
    cols = [str(c) for c in df.columns if str(c).startswith("p_joint_")]
    if len(cols) != int(expected_k):
        raise ValueError(f"expected {expected_k} p_joint_* columns, found {len(cols)}")
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def _largest_remainder(vec: Any, *, total: int) -> Any:
    np, _ = _require_numpy_pandas()
    total = int(total)
    if total < 0:
        raise ValueError(f"total must be non-negative, got: {total}")
    base = np.floor(vec).astype(int)
    rem = int(total - int(base.sum()))
    if rem < 0:
        order = np.argsort(vec - base)
        for i in order[: abs(rem)].tolist():
            if base[i] > 0:
                base[i] -= 1
        return base
    if rem == 0:
        return base
    frac = np.asarray(vec, dtype=float) - np.floor(vec)
    order = np.argsort(-frac)
    base[order[:rem]] += 1
    return base


def joint_wide_to_type_counts(
    *,
    joint_wide: Any,
    schema: Any,
    region_col: str = "puma_uid",
    total_count_col: str = "total_person_weight",
    count_col: str = "count",
    type_idx_col: str = "type_idx",
    integerize: bool = False,
    drop_zero: bool = True,
) -> Any:
    """
    Convert a joint-wide PUMA table into long-format type counts.
    """
    np, pd = _require_numpy_pandas()
    if not isinstance(joint_wide, pd.DataFrame):
        raise TypeError("joint_wide must be a pandas DataFrame")

    schema_obj = _load_schema(schema)
    type_catalog = build_type_catalog(schema=schema_obj, type_idx_col=str(type_idx_col))
    k = int(type_catalog.shape[0])
    prob_cols = _joint_prob_cols(joint_wide, expected_k=k)
    if str(region_col) not in joint_wide.columns:
        raise ValueError(f"joint_wide missing column: {region_col}")
    if str(total_count_col) not in joint_wide.columns:
        raise ValueError(f"joint_wide missing column: {total_count_col}")

    rows: list[Any] = []
    for row in joint_wide.itertuples(index=False):
        region_id = str(getattr(row, str(region_col)))
        total = float(getattr(row, str(total_count_col)))
        probs = np.asarray([float(getattr(row, c)) for c in prob_cols], dtype=float)
        probs = np.clip(probs, 0.0, None)
        mass = float(probs.sum())
        if mass <= 0.0:
            continue
        probs = probs / mass
        counts = probs * max(total, 0.0)
        if bool(integerize):
            counts = _largest_remainder(counts, total=max(int(round(total)), 0))
        block = type_catalog.copy()
        block[str(region_col)] = region_id
        block[str(count_col)] = counts.astype(float)
        rows.append(block)

    if not rows:
        cols = [str(region_col), str(type_idx_col), str(count_col)] + schema_obj["variable_order"]
        return pd.DataFrame(columns=cols)

    out = pd.concat(rows, axis=0, ignore_index=True)
    out[str(count_col)] = pd.to_numeric(out[str(count_col)], errors="coerce").fillna(0.0).astype(float)
    if drop_zero:
        out = out[out[str(count_col)] > 0.0].copy()
    return out.reset_index(drop=True)


def _normalize_targets_long(
    *,
    targets_long: Any,
    group_col: str,
    variable_col: str,
    category_col: str,
    target_col: str,
) -> Any:
    _, pd = _require_numpy_pandas()
    if not isinstance(targets_long, pd.DataFrame):
        raise TypeError("targets_long must be a pandas DataFrame")
    need = [group_col, variable_col, category_col, target_col]
    miss = [c for c in need if c not in targets_long.columns]
    if miss:
        raise ValueError(f"targets_long missing columns: {miss}")
    out = targets_long[need].copy()
    out[group_col] = out[group_col].astype(str)
    out[variable_col] = out[variable_col].astype(str)
    out[category_col] = out[category_col].astype(str)
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = out.groupby([group_col, variable_col, category_col], as_index=False, sort=False)[target_col].sum()
    return out


def build_type_to_group_prior(
    *,
    type_catalog: Any,
    groups: list[str],
    prior_targets_long: Any | None = None,
    residual_targets_long: Any | None = None,
    group_col: str = "tract_geoid",
    variable_col: str = "variable",
    category_col: str = "category",
    target_col: str = "target",
    prior_variables: list[str] | None = None,
    variable_weights: Mapping[str, float] | None = None,
    residual_variables: list[str] | None = None,
    residual_variable_weights: Mapping[str, float] | None = None,
    residual_ratio_clip: float = 5.0,
    group_weights: Any | None = None,
    group_weight_col: str = "weight",
    epsilon: float = 1e-8,
) -> tuple[Any, dict[str, Any]]:
    """
    Build q(g|k)-style prior weights from small-area marginals and optional group weights.
    """
    np, pd = _require_numpy_pandas()
    if not isinstance(type_catalog, pd.DataFrame):
        raise TypeError("type_catalog must be a pandas DataFrame")
    groups = [str(g) for g in groups]
    n_types = int(type_catalog.shape[0])
    n_groups = int(len(groups))
    prior = np.ones((n_types, n_groups), dtype=float)
    meta: dict[str, Any] = {
        "n_types": n_types,
        "n_groups": n_groups,
        "prior_variables": [],
        "residual_variables": [],
        "group_weight_col": None,
        "residual_ratio_clip": float(residual_ratio_clip),
    }

    if group_weights is not None:
        if isinstance(group_weights, Mapping):
            w_map = {str(k): float(v) for k, v in dict(group_weights).items()}
        else:
            if not isinstance(group_weights, pd.DataFrame):
                raise TypeError("group_weights must be a mapping or pandas DataFrame")
            if group_col not in group_weights.columns or str(group_weight_col) not in group_weights.columns:
                raise ValueError(f"group_weights must contain {group_col} and {group_weight_col}")
            w_map = {
                str(r[group_col]): float(r[group_weight_col])
                for r in group_weights[[group_col, str(group_weight_col)]].to_dict(orient="records")
            }
        col_w = np.asarray([max(float(w_map.get(g, 0.0)), float(epsilon)) for g in groups], dtype=float)
        col_w = col_w / max(float(col_w.mean()), float(epsilon))
        prior *= col_w.reshape(1, -1)
        meta["group_weight_col"] = str(group_weight_col)

    tgt = None
    if prior_targets_long is not None:
        tgt = _normalize_targets_long(
            targets_long=prior_targets_long,
            group_col=str(group_col),
            variable_col=str(variable_col),
            category_col=str(category_col),
            target_col=str(target_col),
        )
        available_vars = set(tgt[str(variable_col)].unique().tolist())
        if prior_variables is None:
            prior_variables = [str(v) for v in type_catalog.columns if str(v) in available_vars]
        else:
            prior_variables = [str(v) for v in prior_variables if str(v) in available_vars and str(v) in type_catalog.columns]
        variable_weights = {str(k): float(v) for k, v in dict(variable_weights or {}).items()}

        for var in prior_variables:
            sub = tgt[tgt[str(variable_col)] == str(var)].copy()
            if sub.empty:
                continue
            denom = sub.groupby(str(group_col), sort=False)[str(target_col)].transform("sum").replace(0.0, 1.0)
            sub["_prob"] = sub[str(target_col)] / denom
            lookup: dict[str, dict[str, float]] = {}
            for g, gg in sub.groupby(str(group_col), sort=False):
                lookup[str(g)] = {
                    str(r[str(category_col)]): float(r["_prob"])
                    for _, r in gg[[str(category_col), "_prob"]].iterrows()
                }

            default_vec = np.full((n_groups,), float(epsilon), dtype=float)
            cat_vectors: dict[str, Any] = {}
            for cat in sorted(type_catalog[str(var)].astype(str).unique().tolist()):
                vec = np.asarray(
                    [max(float(lookup.get(g, {}).get(str(cat), 0.0)), float(epsilon)) for g in groups],
                    dtype=float,
                )
                cat_vectors[str(cat)] = vec

            rows = []
            for cat in type_catalog[str(var)].astype(str).tolist():
                rows.append(cat_vectors.get(str(cat), default_vec))
            var_prior = np.vstack(rows)
            weight = float(variable_weights.get(str(var), 1.0))
            prior *= np.power(var_prior, weight)
            meta["prior_variables"].append(str(var))

    if residual_targets_long is not None:
        res = _normalize_targets_long(
            targets_long=residual_targets_long,
            group_col=str(group_col),
            variable_col=str(variable_col),
            category_col=str(category_col),
            target_col=str(target_col),
        )
        available_res_vars = set(res[str(variable_col)].unique().tolist())
        if residual_variables is None:
            residual_variables = [str(v) for v in type_catalog.columns if str(v) in available_res_vars]
        else:
            residual_variables = [
                str(v) for v in residual_variables if str(v) in available_res_vars and str(v) in type_catalog.columns
            ]
        residual_variable_weights = {str(k): float(v) for k, v in dict(residual_variable_weights or {}).items()}

        for var in residual_variables:
            sub = res[res[str(variable_col)] == str(var)].copy()
            if sub.empty:
                continue
            denom = sub.groupby(str(group_col), sort=False)[str(target_col)].transform("sum").replace(0.0, 1.0)
            sub["_prob"] = sub[str(target_col)] / denom
            lookup: dict[str, dict[str, float]] = {}
            for g, gg in sub.groupby(str(group_col), sort=False):
                lookup[str(g)] = {
                    str(r[str(category_col)]): float(r["_prob"])
                    for _, r in gg[[str(category_col), "_prob"]].iterrows()
                }

            if tgt is not None:
                mass_sub = tgt[tgt[str(variable_col)] == str(var)].copy()
                group_mass = (
                    mass_sub.groupby(str(group_col), sort=False)[str(target_col)]
                    .sum()
                    .reindex(groups, fill_value=0.0)
                    .to_numpy(dtype=float)
                )
            else:
                group_mass = np.ones((n_groups,), dtype=float)
            if float(group_mass.sum()) <= 0.0:
                group_mass = np.ones((n_groups,), dtype=float)

            default_vec = np.full((n_groups,), 1.0, dtype=float)
            cat_vectors: dict[str, Any] = {}
            for cat in sorted(type_catalog[str(var)].astype(str).unique().tolist()):
                prob_vec = np.asarray(
                    [max(float(lookup.get(g, {}).get(str(cat), 0.0)), float(epsilon)) for g in groups],
                    dtype=float,
                )
                log_vec = np.log(np.clip(prob_vec, float(epsilon), None))
                center = float(np.average(log_vec, weights=group_mass))
                residual_vec = np.exp(log_vec - center)
                if float(residual_ratio_clip) > 1.0:
                    residual_vec = np.clip(
                        residual_vec,
                        1.0 / float(residual_ratio_clip),
                        float(residual_ratio_clip),
                    )
                cat_vectors[str(cat)] = residual_vec

            rows = []
            for cat in type_catalog[str(var)].astype(str).tolist():
                rows.append(cat_vectors.get(str(cat), default_vec))
            residual_prior = np.vstack(rows)
            weight = float(residual_variable_weights.get(str(var), 1.0))
            prior *= np.power(residual_prior, weight)
            meta["residual_variables"].append(str(var))

    prior = np.clip(prior, float(epsilon), None)
    prior = prior / np.clip(prior.sum(axis=1, keepdims=True), float(epsilon), None)
    return prior, meta


def _constraint_errors_from_matrix(
    *,
    alloc: Any,
    type_counts: Any,
    type_catalog: Any,
    targets_long: Any,
    groups: list[str],
    group_col: str,
    variable_col: str,
    category_col: str,
    target_col: str,
    count_col: str,
    hard_variables: list[str],
    epsilon: float,
) -> dict[str, Any]:
    np, pd = _require_numpy_pandas()

    target = _normalize_targets_long(
        targets_long=targets_long,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    row_target = pd.to_numeric(type_counts[str(count_col)], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    row_now = np.asarray(alloc.sum(axis=1), dtype=float)
    row_abs = np.abs(row_now - row_target)

    by_var: dict[str, Any] = {}
    max_abs = float(row_abs.max()) if row_abs.size else 0.0
    max_rel = float((row_abs / np.clip(row_target, float(epsilon), None)).max()) if row_abs.size else 0.0

    for var in hard_variables:
        sub = target[target[str(variable_col)] == str(var)].copy()
        if sub.empty:
            continue
        masks = {
            str(cat): (type_catalog[str(var)].astype(str).to_numpy() == str(cat))
            for cat in sorted(sub[str(category_col)].astype(str).unique().tolist())
        }
        errs_abs: list[float] = []
        errs_rel: list[float] = []
        for j, g in enumerate(groups):
            gg = sub[sub[str(group_col)] == str(g)]
            for _, r in gg.iterrows():
                cat = str(r[str(category_col)])
                mask = masks.get(cat)
                if mask is None:
                    continue
                pred = float(alloc[mask, j].sum())
                tgt = float(r[str(target_col)])
                errs_abs.append(abs(pred - tgt))
                errs_rel.append(abs(pred - tgt) / max(tgt, float(epsilon)))
        by_var[str(var)] = {
            "max_abs": (float(max(errs_abs)) if errs_abs else 0.0),
            "mean_abs": (float(np.mean(errs_abs)) if errs_abs else 0.0),
            "max_rel": (float(max(errs_rel)) if errs_rel else 0.0),
            "mean_rel": (float(np.mean(errs_rel)) if errs_rel else 0.0),
        }

    return {
        "row_sum": {"max_abs": max_abs, "max_rel": max_rel},
        "hard_constraints": by_var,
    }


def _fit_matrix_to_marginals(
    *,
    init: Any,
    row_sums: Any,
    col_sums: Any,
    epsilon: float,
    max_iters: int = 1000,
    tol: float = 1e-10,
) -> Any:
    np, _ = _require_numpy_pandas()
    x = np.asarray(init, dtype=float)
    x = np.clip(x, float(epsilon), None)
    row_sums = np.asarray(row_sums, dtype=float).reshape(-1)
    col_sums = np.asarray(col_sums, dtype=float).reshape(-1)
    total_r = float(row_sums.sum())
    total_c = float(col_sums.sum())
    if total_r <= 0.0 or total_c <= 0.0:
        raise ValueError("row_sums and col_sums must have positive mass")
    if abs(total_r - total_c) > max(float(tol), 1e-8 * max(total_r, total_c, 1.0)):
        col_sums = col_sums * (total_r / total_c)
    for _ in range(int(max_iters)):
        row_now = x.sum(axis=1)
        x *= (row_sums / np.clip(row_now, float(epsilon), None)).reshape(-1, 1)
        col_now = x.sum(axis=0)
        x *= (col_sums / np.clip(col_now, float(epsilon), None)).reshape(1, -1)
        err = max(
            float(np.abs(x.sum(axis=1) - row_sums).max(initial=0.0)),
            float(np.abs(x.sum(axis=0) - col_sums).max(initial=0.0)),
        )
        if err <= float(tol):
            break
    return x


def reconcile_hard_targets_to_type_counts(
    *,
    type_counts: Any,
    targets_long: Any,
    group_col: str = "tract_geoid",
    variable_col: str = "variable",
    category_col: str = "category",
    target_col: str = "target",
    count_col: str = "count",
    hard_variables: list[str] | None = None,
    epsilon: float = 1e-8,
) -> tuple[Any, dict[str, Any]]:
    """
    Reconcile small-area hard targets with region-level type totals.

    Why:
    - tract/CBG ACS marginals and the PUMA-level joint table may not sum to the
      same region totals because they come from different years / universes
    - without reconciliation, the hard-constraint system can be infeasible
    """
    np, pd = _require_numpy_pandas()
    if not isinstance(type_counts, pd.DataFrame):
        raise TypeError("type_counts must be a pandas DataFrame")

    tgt = _normalize_targets_long(
        targets_long=targets_long,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    groups = sorted(tgt[str(group_col)].astype(str).unique().tolist())
    if not groups:
        return tgt.copy(), {"applied": False, "reason": "no_groups"}

    if hard_variables is None:
        hard_variables = [str(v) for v in tgt[str(variable_col)].astype(str).unique().tolist() if str(v) in type_counts.columns]
    else:
        hard_variables = [str(v) for v in hard_variables if str(v) in type_counts.columns]
    if not hard_variables:
        return tgt.copy(), {"applied": False, "reason": "no_hard_variables"}

    region_total = float(pd.to_numeric(type_counts[str(count_col)], errors="coerce").fillna(0.0).sum())
    if region_total <= 0.0:
        return tgt.copy(), {"applied": False, "reason": "zero_region_total"}

    group_total_vectors: list[Any] = []
    raw_group_total_by_var: dict[str, float] = {}
    for var in hard_variables:
        sub = tgt[tgt[str(variable_col)] == str(var)].copy()
        if sub.empty:
            continue
        vec = (
            sub.groupby(str(group_col), sort=False)[str(target_col)]
            .sum()
            .reindex(groups, fill_value=0.0)
            .to_numpy(dtype=float)
        )
        group_total_vectors.append(vec)
        raw_group_total_by_var[str(var)] = float(vec.sum())
    if not group_total_vectors:
        return tgt.copy(), {"applied": False, "reason": "empty_hard_targets"}

    canonical_group_totals = np.mean(np.vstack(group_total_vectors), axis=0)
    if float(canonical_group_totals.sum()) <= 0.0:
        canonical_group_totals = np.full((len(groups),), region_total / max(len(groups), 1), dtype=float)
    else:
        canonical_group_totals = canonical_group_totals / float(canonical_group_totals.sum()) * region_total

    out_rows: list[dict[str, Any]] = []
    for _, r in tgt[~tgt[str(variable_col)].astype(str).isin(hard_variables)].iterrows():
        out_rows.append(r.to_dict())

    category_total_meta: dict[str, Any] = {}
    for var in hard_variables:
        cats = sorted(type_counts[str(var)].astype(str).unique().tolist())
        col_sums = np.asarray(
            [
                float(
                    pd.to_numeric(type_counts.loc[type_counts[str(var)].astype(str) == str(cat), str(count_col)], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
                for cat in cats
            ],
            dtype=float,
        )
        sub = tgt[tgt[str(variable_col)] == str(var)].copy()
        init = np.full((len(groups), len(cats)), float(epsilon), dtype=float)
        if not sub.empty:
            g_index = {g: i for i, g in enumerate(groups)}
            c_index = {c: i for i, c in enumerate(cats)}
            for _, r in sub.iterrows():
                g = str(r[str(group_col)])
                c = str(r[str(category_col)])
                if g in g_index and c in c_index:
                    init[g_index[g], c_index[c]] = max(float(r[str(target_col)]), float(epsilon))
        fitted = _fit_matrix_to_marginals(
            init=init,
            row_sums=canonical_group_totals,
            col_sums=col_sums,
            epsilon=float(epsilon),
        )
        category_total_meta[str(var)] = {
            "region_target_total": {str(cat): float(col_sums[i]) for i, cat in enumerate(cats)},
            "raw_region_total": float(sub[str(target_col)].sum()) if not sub.empty else 0.0,
        }
        for i, g in enumerate(groups):
            for j, cat in enumerate(cats):
                out_rows.append(
                    {
                        str(group_col): str(g),
                        str(variable_col): str(var),
                        str(category_col): str(cat),
                        str(target_col): float(fitted[i, j]),
                    }
                )

    out = pd.DataFrame(out_rows)
    out = out.groupby([str(group_col), str(variable_col), str(category_col)], as_index=False, sort=False)[str(target_col)].sum()
    meta = {
        "applied": True,
        "hard_variables": [str(v) for v in hard_variables],
        "region_total_from_types": region_total,
        "raw_group_total_by_var": raw_group_total_by_var,
        "canonical_group_total_sum": float(canonical_group_totals.sum()),
        "category_total_meta": category_total_meta,
    }
    return out, meta


def compare_targets_long(
    *,
    predicted_targets_long: Any,
    reference_targets_long: Any,
    group_col: str = "tract_geoid",
    variable_col: str = "variable",
    category_col: str = "category",
    target_col: str = "target",
) -> dict[str, Any]:
    """
    Compare two targets_long tables after normalizing within (group, variable).
    """
    np, pd = _require_numpy_pandas()
    pred = _normalize_targets_long(
        targets_long=predicted_targets_long,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    ref = _normalize_targets_long(
        targets_long=reference_targets_long,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    out: dict[str, Any] = {"variables": {}}
    pred_denom = pred.groupby([str(group_col), str(variable_col)], sort=False)[str(target_col)].transform("sum").replace(0.0, 1.0)
    ref_denom = ref.groupby([str(group_col), str(variable_col)], sort=False)[str(target_col)].transform("sum").replace(0.0, 1.0)
    pred["_share"] = pred[str(target_col)] / pred_denom
    ref["_share"] = ref[str(target_col)] / ref_denom
    variables = sorted(set(pred[str(variable_col)].astype(str).tolist()) & set(ref[str(variable_col)].astype(str).tolist()))
    for var in variables:
        p_var = pred[pred[str(variable_col)] == str(var)][[str(group_col), str(category_col), "_share"]].rename(columns={"_share": "pred"})
        r_var = ref[ref[str(variable_col)] == str(var)][[str(group_col), str(category_col), "_share"]].rename(columns={"_share": "ref"})
        merged = r_var.merge(p_var, on=[str(group_col), str(category_col)], how="outer")
        merged["pred"] = pd.to_numeric(merged["pred"], errors="coerce").fillna(0.0)
        merged["ref"] = pd.to_numeric(merged["ref"], errors="coerce").fillna(0.0)
        merged["abs_err"] = (merged["pred"] - merged["ref"]).abs()
        tvd_vals: list[float] = []
        for _, gg in merged.groupby(str(group_col), sort=False):
            tvd_vals.append(0.5 * float((gg["pred"] - gg["ref"]).abs().sum()))
        out["variables"][str(var)] = {
            "mean_abs_err": float(merged["abs_err"].mean()) if not merged.empty else 0.0,
            "max_abs_err": float(merged["abs_err"].max()) if not merged.empty else 0.0,
            "mean_tvd": float(np.mean(tvd_vals)) if tvd_vals else 0.0,
            "max_tvd": float(np.max(tvd_vals)) if tvd_vals else 0.0,
        }
    return out


def low_rank_project_targets_long(
    *,
    targets_long: Any,
    reference_targets_long: Any | None = None,
    group_col: str = "tract_geoid",
    variable_col: str = "variable",
    category_col: str = "category",
    target_col: str = "target",
    variables: list[str] | None = None,
    rank: int = 2,
    epsilon: float = 1e-8,
) -> tuple[Any, dict[str, Any]]:
    """
    Compress target shares through a shared low-rank compatibility matrix.

    Rows are `(variable, category)` profiles and columns are groups. The matrix is built
    on centered log-shares, truncated by SVD, and projected back to normalized shares.
    """
    np, pd = _require_numpy_pandas()
    tgt = _normalize_targets_long(
        targets_long=targets_long,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    if tgt.empty:
        return tgt.copy(), {"applied": False, "reason": "empty_targets"}

    ref = None
    if reference_targets_long is not None:
        ref = _normalize_targets_long(
            targets_long=reference_targets_long,
            group_col=str(group_col),
            variable_col=str(variable_col),
            category_col=str(category_col),
            target_col=str(target_col),
        )

    if variables is None:
        variables = sorted(tgt[str(variable_col)].astype(str).unique().tolist())
    variables = [str(v) for v in variables]
    tgt = tgt[tgt[str(variable_col)].astype(str).isin(variables)].copy()
    if tgt.empty:
        return tgt.copy(), {"applied": False, "reason": "no_selected_variables"}

    groups = sorted(tgt[str(group_col)].astype(str).unique().tolist())
    rows_meta: list[tuple[str, str, float]] = []
    row_vectors: list[Any] = []
    for var in variables:
        sub = tgt[tgt[str(variable_col)] == str(var)].copy()
        if sub.empty:
            continue
        cats = sorted(sub[str(category_col)].astype(str).unique().tolist())
        denom = sub.groupby(str(group_col), sort=False)[str(target_col)].transform("sum").replace(0.0, 1.0)
        sub["_share"] = sub[str(target_col)] / denom
        if ref is not None:
            ref_sub = ref[ref[str(variable_col)] == str(var)].copy()
            if ref_sub.empty:
                group_mass = np.ones((len(groups),), dtype=float)
            else:
                group_mass = (
                    ref_sub.groupby(str(group_col), sort=False)[str(target_col)]
                    .sum()
                    .reindex(groups, fill_value=0.0)
                    .to_numpy(dtype=float)
                )
        else:
            group_mass = np.ones((len(groups),), dtype=float)
        if float(group_mass.sum()) <= 0.0:
            group_mass = np.ones((len(groups),), dtype=float)
        for cat in cats:
            cat_sub = sub[sub[str(category_col)] == str(cat)].copy()
            share_vec = (
                cat_sub.groupby(str(group_col), sort=False)["_share"]
                .sum()
                .reindex(groups, fill_value=0.0)
                .to_numpy(dtype=float)
            )
            log_vec = np.log(np.clip(share_vec, float(epsilon), None))
            center = float(np.average(log_vec, weights=group_mass))
            row_vectors.append(log_vec - center)
            rows_meta.append((str(var), str(cat), center))

    if not row_vectors:
        return tgt.copy(), {"applied": False, "reason": "no_rows"}

    mat = np.vstack(row_vectors)
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    rank_eff = max(1, min(int(rank), int(min(mat.shape))))
    recon = (u[:, :rank_eff] * s[:rank_eff]) @ vt[:rank_eff, :]
    frob_total = float((s**2).sum())
    frob_kept = float((s[:rank_eff] ** 2).sum())

    out_rows: list[dict[str, Any]] = []
    idx = 0
    for var in variables:
        var_rows: list[tuple[str, Any]] = []
        while idx < len(rows_meta) and rows_meta[idx][0] == str(var):
            _, cat, center = rows_meta[idx]
            vec = np.exp(recon[idx] + float(center))
            var_rows.append((str(cat), vec))
            idx += 1
        if not var_rows:
            continue
        mat_var = np.vstack([vec for _, vec in var_rows])
        denom = np.clip(mat_var.sum(axis=0, keepdims=True), float(epsilon), None)
        mat_var = mat_var / denom
        for row_i, (cat, _) in enumerate(var_rows):
            for col_j, g in enumerate(groups):
                out_rows.append(
                    {
                        str(group_col): str(g),
                        str(variable_col): str(var),
                        str(category_col): str(cat),
                        str(target_col): float(mat_var[row_i, col_j]),
                    }
                )

    out = pd.DataFrame(out_rows)
    meta: dict[str, Any] = {
        "applied": True,
        "rank": int(rank_eff),
        "n_groups": int(len(groups)),
        "n_row_profiles": int(len(rows_meta)),
        "explained_frob_ratio": (frob_kept / frob_total) if frob_total > 0 else 1.0,
        "variables": variables,
    }
    meta["fit_against_input"] = compare_targets_long(
        predicted_targets_long=out,
        reference_targets_long=tgt,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    return out, meta


def low_rank_plus_sparse_project_targets_long(
    *,
    targets_long: Any,
    reference_targets_long: Any | None = None,
    group_col: str = "tract_geoid",
    variable_col: str = "variable",
    category_col: str = "category",
    target_col: str = "target",
    variables: list[str] | None = None,
    rank: int = 2,
    sparse_weight: float = 0.5,
    sparse_threshold: float = 0.0,
    epsilon: float = 1e-8,
) -> tuple[Any, dict[str, Any]]:
    """
    Reconstruct target shares with a low-rank compatibility core plus sparse residual detail.

    The decomposition is performed in centered log-share space:
    - low-rank term captures shared neighborhood archetypes
    - sparse term keeps tract-specific deviations that the low-rank core misses
    """
    np, pd = _require_numpy_pandas()
    tgt = _normalize_targets_long(
        targets_long=targets_long,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    if tgt.empty:
        return tgt.copy(), {"applied": False, "reason": "empty_targets"}

    ref = None
    if reference_targets_long is not None:
        ref = _normalize_targets_long(
            targets_long=reference_targets_long,
            group_col=str(group_col),
            variable_col=str(variable_col),
            category_col=str(category_col),
            target_col=str(target_col),
        )

    if variables is None:
        variables = sorted(tgt[str(variable_col)].astype(str).unique().tolist())
    variables = [str(v) for v in variables]
    tgt = tgt[tgt[str(variable_col)].astype(str).isin(variables)].copy()
    if tgt.empty:
        return tgt.copy(), {"applied": False, "reason": "no_selected_variables"}

    groups = sorted(tgt[str(group_col)].astype(str).unique().tolist())
    rows_meta: list[tuple[str, str, float]] = []
    row_vectors: list[Any] = []
    for var in variables:
        sub = tgt[tgt[str(variable_col)] == str(var)].copy()
        if sub.empty:
            continue
        cats = sorted(sub[str(category_col)].astype(str).unique().tolist())
        denom = sub.groupby(str(group_col), sort=False)[str(target_col)].transform("sum").replace(0.0, 1.0)
        sub["_share"] = sub[str(target_col)] / denom
        if ref is not None:
            ref_sub = ref[ref[str(variable_col)] == str(var)].copy()
            if ref_sub.empty:
                group_mass = np.ones((len(groups),), dtype=float)
            else:
                group_mass = (
                    ref_sub.groupby(str(group_col), sort=False)[str(target_col)]
                    .sum()
                    .reindex(groups, fill_value=0.0)
                    .to_numpy(dtype=float)
                )
        else:
            group_mass = np.ones((len(groups),), dtype=float)
        if float(group_mass.sum()) <= 0.0:
            group_mass = np.ones((len(groups),), dtype=float)
        for cat in cats:
            cat_sub = sub[sub[str(category_col)] == str(cat)].copy()
            share_vec = (
                cat_sub.groupby(str(group_col), sort=False)["_share"]
                .sum()
                .reindex(groups, fill_value=0.0)
                .to_numpy(dtype=float)
            )
            log_vec = np.log(np.clip(share_vec, float(epsilon), None))
            center = float(np.average(log_vec, weights=group_mass))
            row_vectors.append(log_vec - center)
            rows_meta.append((str(var), str(cat), center))

    if not row_vectors:
        return tgt.copy(), {"applied": False, "reason": "no_rows"}

    mat = np.vstack(row_vectors)
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    rank_eff = max(1, min(int(rank), int(min(mat.shape))))
    low_rank = (u[:, :rank_eff] * s[:rank_eff]) @ vt[:rank_eff, :]
    sparse_raw = mat - low_rank
    sparse_threshold = max(float(sparse_threshold), 0.0)
    if sparse_threshold > 0.0:
        sparse_keep = np.sign(sparse_raw) * np.maximum(np.abs(sparse_raw) - sparse_threshold, 0.0)
    else:
        sparse_keep = sparse_raw
    recon = low_rank + float(sparse_weight) * sparse_keep
    frob_total = float((s**2).sum())
    frob_kept = float((s[:rank_eff] ** 2).sum())
    sparse_abs = np.abs(sparse_keep)
    sparse_retained_fraction = float((sparse_abs > 0.0).mean()) if sparse_abs.size else 0.0

    out_rows: list[dict[str, Any]] = []
    idx = 0
    for var in variables:
        var_rows: list[tuple[str, Any]] = []
        while idx < len(rows_meta) and rows_meta[idx][0] == str(var):
            _, cat, center = rows_meta[idx]
            vec = np.exp(recon[idx] + float(center))
            var_rows.append((str(cat), vec))
            idx += 1
        if not var_rows:
            continue
        mat_var = np.vstack([vec for _, vec in var_rows])
        denom = np.clip(mat_var.sum(axis=0, keepdims=True), float(epsilon), None)
        mat_var = mat_var / denom
        for row_i, (cat, _) in enumerate(var_rows):
            for col_j, g in enumerate(groups):
                out_rows.append(
                    {
                        str(group_col): str(g),
                        str(variable_col): str(var),
                        str(category_col): str(cat),
                        str(target_col): float(mat_var[row_i, col_j]),
                    }
                )

    out = pd.DataFrame(out_rows)
    meta: dict[str, Any] = {
        "applied": True,
        "rank": int(rank_eff),
        "n_groups": int(len(groups)),
        "n_row_profiles": int(len(rows_meta)),
        "explained_frob_ratio": (frob_kept / frob_total) if frob_total > 0 else 1.0,
        "sparse_weight": float(sparse_weight),
        "sparse_threshold": float(sparse_threshold),
        "sparse_retained_fraction": sparse_retained_fraction,
        "sparse_mean_abs": float(sparse_abs.mean()) if sparse_abs.size else 0.0,
        "sparse_max_abs": float(sparse_abs.max()) if sparse_abs.size else 0.0,
        "variables": variables,
    }
    meta["fit_against_input"] = compare_targets_long(
        predicted_targets_long=out,
        reference_targets_long=tgt,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    meta["fit_against_low_rank_core"] = compare_targets_long(
        predicted_targets_long=low_rank_project_targets_long(
            targets_long=tgt,
            reference_targets_long=ref,
            group_col=str(group_col),
            variable_col=str(variable_col),
            category_col=str(category_col),
            target_col=str(target_col),
            variables=variables,
            rank=int(rank_eff),
            epsilon=float(epsilon),
        )[0],
        reference_targets_long=out,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    return out, meta


def low_rank_plus_smooth_project_targets_long(
    *,
    targets_long: Any,
    group_features: Any,
    reference_targets_long: Any | None = None,
    group_col: str = "tract_geoid",
    region_col: str | None = "puma_uid",
    variable_col: str = "variable",
    category_col: str = "category",
    target_col: str = "target",
    variables: list[str] | None = None,
    feature_prefixes: tuple[str, ...] = ("cat__", "home_origin_"),
    rank: int = 2,
    smooth_weight: float = 0.5,
    smooth_knn: int = 8,
    smooth_bandwidth: float = 0.0,
    epsilon: float = 1e-8,
) -> tuple[Any, dict[str, Any]]:
    """
    Reconstruct target shares with a low-rank core plus feature-space smoothed residual.

    The residual correction is smoothed over k-nearest groups in the mobility-feature space,
    optionally restricted within each region such as a PUMA.
    """
    np, pd = _require_numpy_pandas()
    tgt = _normalize_targets_long(
        targets_long=targets_long,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    if tgt.empty:
        return tgt.copy(), {"applied": False, "reason": "empty_targets"}
    if not isinstance(group_features, pd.DataFrame):
        raise TypeError("group_features must be a pandas DataFrame")

    ref = None
    if reference_targets_long is not None:
        ref = _normalize_targets_long(
            targets_long=reference_targets_long,
            group_col=str(group_col),
            variable_col=str(variable_col),
            category_col=str(category_col),
            target_col=str(target_col),
        )

    if variables is None:
        variables = sorted(tgt[str(variable_col)].astype(str).unique().tolist())
    variables = [str(v) for v in variables]
    tgt = tgt[tgt[str(variable_col)].astype(str).isin(variables)].copy()
    if tgt.empty:
        return tgt.copy(), {"applied": False, "reason": "no_selected_variables"}

    groups = sorted(tgt[str(group_col)].astype(str).unique().tolist())
    rows_meta: list[tuple[str, str, float]] = []
    row_vectors: list[Any] = []
    for var in variables:
        sub = tgt[tgt[str(variable_col)] == str(var)].copy()
        if sub.empty:
            continue
        cats = sorted(sub[str(category_col)].astype(str).unique().tolist())
        denom = sub.groupby(str(group_col), sort=False)[str(target_col)].transform("sum").replace(0.0, 1.0)
        sub["_share"] = sub[str(target_col)] / denom
        if ref is not None:
            ref_sub = ref[ref[str(variable_col)] == str(var)].copy()
            if ref_sub.empty:
                group_mass = np.ones((len(groups),), dtype=float)
            else:
                group_mass = (
                    ref_sub.groupby(str(group_col), sort=False)[str(target_col)]
                    .sum()
                    .reindex(groups, fill_value=0.0)
                    .to_numpy(dtype=float)
                )
        else:
            group_mass = np.ones((len(groups),), dtype=float)
        if float(group_mass.sum()) <= 0.0:
            group_mass = np.ones((len(groups),), dtype=float)
        for cat in cats:
            cat_sub = sub[sub[str(category_col)] == str(cat)].copy()
            share_vec = (
                cat_sub.groupby(str(group_col), sort=False)["_share"]
                .sum()
                .reindex(groups, fill_value=0.0)
                .to_numpy(dtype=float)
            )
            log_vec = np.log(np.clip(share_vec, float(epsilon), None))
            center = float(np.average(log_vec, weights=group_mass))
            row_vectors.append(log_vec - center)
            rows_meta.append((str(var), str(cat), center))

    if not row_vectors:
        return tgt.copy(), {"applied": False, "reason": "no_rows"}

    feat = group_features.copy()
    if str(group_col) not in feat.columns:
        raise ValueError(f"group_features missing column: {group_col}")
    keep_cols = [str(group_col)]
    if region_col is not None:
        if str(region_col) not in feat.columns:
            raise ValueError(f"group_features missing column: {region_col}")
        keep_cols.append(str(region_col))
    feat_cols = [
        str(c)
        for c in feat.columns
        if str(c) not in set(keep_cols)
        and (
            not feature_prefixes
            or any(str(c).startswith(str(prefix)) for prefix in feature_prefixes)
        )
    ]
    if not feat_cols:
        raise ValueError("group_features has no usable feature columns for smoothing")
    feat = feat[keep_cols + feat_cols].drop_duplicates(subset=[str(group_col)], keep="first").copy()
    feat[str(group_col)] = feat[str(group_col)].astype(str)
    if region_col is not None:
        feat[str(region_col)] = feat[str(region_col)].astype(str)
    feat = feat.set_index(str(group_col)).reindex(groups)
    regions = (
        feat[str(region_col)].fillna("__all__").astype(str).to_numpy()
        if region_col is not None
        else np.asarray(["__all__"] * len(groups), dtype=object)
    )
    feat_mat = feat[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if feat_mat.size <= 0:
        raise ValueError("group_features resolved to an empty feature matrix")
    feat_mean = feat_mat.mean(axis=0, keepdims=True)
    feat_std = feat_mat.std(axis=0, keepdims=True)
    feat_std = np.where(feat_std > float(epsilon), feat_std, 1.0)
    feat_mat = (feat_mat - feat_mean) / feat_std

    mat = np.vstack(row_vectors)
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    rank_eff = max(1, min(int(rank), int(min(mat.shape))))
    low_rank = (u[:, :rank_eff] * s[:rank_eff]) @ vt[:rank_eff, :]
    raw_residual = mat - low_rank

    n_groups = len(groups)
    weight_mat = np.zeros((n_groups, n_groups), dtype=float)
    bandwidth_used: dict[str, float] = {}
    knn_eff = max(1, int(smooth_knn))
    for region in sorted(set(regions.tolist())):
        idx = np.where(regions == region)[0]
        if idx.size == 0:
            continue
        if idx.size == 1:
            weight_mat[idx[0], idx[0]] = 1.0
            bandwidth_used[str(region)] = 1.0
            continue
        x = feat_mat[idx, :]
        diff = x[:, None, :] - x[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        if float(smooth_bandwidth) > 0.0:
            sigma = float(smooth_bandwidth)
        else:
            pos = dist2[dist2 > 0.0]
            sigma = float(np.sqrt(np.median(pos))) if pos.size else 1.0
        sigma = max(float(sigma), float(epsilon))
        bandwidth_used[str(region)] = sigma
        w = np.exp(-dist2 / max(2.0 * sigma * sigma, float(epsilon)))
        k_eff = min(int(knn_eff), int(idx.size))
        order = np.argsort(dist2, axis=1)
        mask = np.zeros_like(w, dtype=bool)
        for i in range(idx.size):
            mask[i, order[i, :k_eff]] = True
        w = w * mask.astype(float)
        row_sum = w.sum(axis=1, keepdims=True)
        zero_mask = (row_sum.reshape(-1) <= float(epsilon))
        if bool(zero_mask.any()):
            w[zero_mask, :] = 0.0
            for i in np.where(zero_mask)[0].tolist():
                w[i, i] = 1.0
            row_sum = w.sum(axis=1, keepdims=True)
        w = w / np.clip(row_sum, float(epsilon), None)
        weight_mat[np.ix_(idx, idx)] = w

    smooth_residual = raw_residual @ weight_mat.T
    recon = low_rank + float(smooth_weight) * smooth_residual
    frob_total = float((s**2).sum())
    frob_kept = float((s[:rank_eff] ** 2).sum())

    out_rows: list[dict[str, Any]] = []
    idx = 0
    for var in variables:
        var_rows: list[tuple[str, Any]] = []
        while idx < len(rows_meta) and rows_meta[idx][0] == str(var):
            _, cat, center = rows_meta[idx]
            vec = np.exp(recon[idx] + float(center))
            var_rows.append((str(cat), vec))
            idx += 1
        if not var_rows:
            continue
        mat_var = np.vstack([vec for _, vec in var_rows])
        denom = np.clip(mat_var.sum(axis=0, keepdims=True), float(epsilon), None)
        mat_var = mat_var / denom
        for row_i, (cat, _) in enumerate(var_rows):
            for col_j, g in enumerate(groups):
                out_rows.append(
                    {
                        str(group_col): str(g),
                        str(variable_col): str(var),
                        str(category_col): str(cat),
                        str(target_col): float(mat_var[row_i, col_j]),
                    }
                )

    out = pd.DataFrame(out_rows)
    meta: dict[str, Any] = {
        "applied": True,
        "rank": int(rank_eff),
        "n_groups": int(n_groups),
        "n_row_profiles": int(len(rows_meta)),
        "explained_frob_ratio": (frob_kept / frob_total) if frob_total > 0 else 1.0,
        "smooth_weight": float(smooth_weight),
        "smooth_knn": int(knn_eff),
        "smooth_bandwidth": (float(smooth_bandwidth) if float(smooth_bandwidth) > 0.0 else None),
        "mean_bandwidth_used": float(np.mean(list(bandwidth_used.values()))) if bandwidth_used else 0.0,
        "residual_raw_mean_abs": float(np.abs(raw_residual).mean()) if raw_residual.size else 0.0,
        "residual_smooth_mean_abs": float(np.abs(smooth_residual).mean()) if smooth_residual.size else 0.0,
        "variables": variables,
        "feature_columns": feat_cols,
    }
    meta["fit_against_input"] = compare_targets_long(
        predicted_targets_long=out,
        reference_targets_long=tgt,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    meta["fit_against_low_rank_core"] = compare_targets_long(
        predicted_targets_long=low_rank_project_targets_long(
            targets_long=tgt,
            reference_targets_long=ref,
            group_col=str(group_col),
            variable_col=str(variable_col),
            category_col=str(category_col),
            target_col=str(target_col),
            variables=variables,
            rank=int(rank_eff),
            epsilon=float(epsilon),
        )[0],
        reference_targets_long=out,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    return out, meta


def predict_targets_from_group_features(
    *,
    group_features: Any,
    reference_targets_long: Any,
    group_col: str = "tract_geoid",
    region_col: str | None = None,
    variable_col: str = "variable",
    category_col: str = "category",
    target_col: str = "target",
    variables: list[str] | None = None,
    feature_cols: list[str] | None = None,
    feature_prefixes: tuple[str, ...] = ("cat__", "home_origin_"),
    ridge_alpha: float = 1.0,
    min_train_groups: int = 64,
    epsilon: float = 1e-8,
) -> tuple[Any, dict[str, Any]]:
    """
    Cross-fitted linear-ridge predictor:
    mobility / group features -> target category shares.

    This provides a mobility-conditioned proxy for q(g|k) without person labels.
    """
    np, pd = _require_numpy_pandas()
    if not isinstance(group_features, pd.DataFrame):
        raise TypeError("group_features must be a pandas DataFrame")

    feat = group_features.copy()
    if str(group_col) not in feat.columns:
        raise ValueError(f"group_features missing column: {group_col}")
    feat[str(group_col)] = feat[str(group_col)].astype(str)
    if region_col is not None and str(region_col) in feat.columns:
        feat[str(region_col)] = feat[str(region_col)].astype(str)

    if feature_cols is None:
        numeric_cols = [c for c in feat.columns if c != str(group_col) and (region_col is None or c != str(region_col))]
        numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(feat[c])]
        prefixes = tuple(str(x) for x in feature_prefixes)
        if prefixes:
            numeric_cols = [c for c in numeric_cols if any(str(c).startswith(pref) for pref in prefixes)]
        feature_cols = numeric_cols
    feature_cols = [str(c) for c in feature_cols if str(c) in feat.columns]
    if not feature_cols:
        raise ValueError("No usable feature columns for mobility-conditioned prior")
    for c in feature_cols:
        feat[str(c)] = pd.to_numeric(feat[str(c)], errors="coerce").fillna(0.0).astype(float)

    ref = _normalize_targets_long(
        targets_long=reference_targets_long,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    if variables is None:
        variables = sorted(ref[str(variable_col)].astype(str).unique().tolist())
    variables = [str(v) for v in variables]

    meta: dict[str, Any] = {
        "feature_cols": feature_cols,
        "ridge_alpha": float(ridge_alpha),
        "min_train_groups": int(min_train_groups),
        "variables": {},
    }

    out_rows: list[dict[str, Any]] = []
    for var in variables:
        sub = ref[ref[str(variable_col)] == str(var)].copy()
        if sub.empty:
            continue
        denom = sub.groupby(str(group_col), sort=False)[str(target_col)].transform("sum").replace(0.0, 1.0)
        sub["_share"] = sub[str(target_col)] / denom
        wide = sub.pivot_table(index=str(group_col), columns=str(category_col), values="_share", fill_value=0.0)
        cats = [str(c) for c in wide.columns.tolist()]
        merged = feat[[str(group_col)] + ([str(region_col)] if (region_col is not None and str(region_col) in feat.columns) else []) + feature_cols].merge(
            wide.reset_index(),
            on=str(group_col),
            how="inner",
        )
        if merged.empty:
            continue

        X = merged[feature_cols].to_numpy(dtype=float)
        Y = merged[cats].to_numpy(dtype=float)
        pred = np.zeros_like(Y, dtype=float)
        if region_col is not None and str(region_col) in merged.columns:
            fold_keys = merged[str(region_col)].astype(str).tolist()
            unique_folds = sorted(set(fold_keys))
        else:
            fold_keys = ["__all__"] * int(merged.shape[0])
            unique_folds = ["__all__"]

        for fold in unique_folds:
            test_mask = np.asarray([fk == str(fold) for fk in fold_keys], dtype=bool)
            train_mask = ~test_mask
            if int(train_mask.sum()) < int(min_train_groups):
                train_mask = np.ones_like(test_mask, dtype=bool)
            Xtr = X[train_mask]
            Ytr = Y[train_mask]
            Xte = X[test_mask]
            if Xte.size == 0:
                continue
            mu = Xtr.mean(axis=0, keepdims=True)
            sigma = Xtr.std(axis=0, keepdims=True)
            sigma = np.where(sigma > 1e-8, sigma, 1.0)
            Xtr_n = (Xtr - mu) / sigma
            Xte_n = (Xte - mu) / sigma
            Xtr_aug = np.concatenate([np.ones((Xtr_n.shape[0], 1), dtype=float), Xtr_n], axis=1)
            Xte_aug = np.concatenate([np.ones((Xte_n.shape[0], 1), dtype=float), Xte_n], axis=1)
            reg = np.eye(Xtr_aug.shape[1], dtype=float) * float(ridge_alpha)
            reg[0, 0] = 0.0
            lhs = Xtr_aug.T @ Xtr_aug + reg
            rhs = Xtr_aug.T @ Ytr
            try:
                coef = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                coef = np.linalg.pinv(lhs) @ rhs
            pred[test_mask] = Xte_aug @ coef

        pred = np.clip(pred, float(epsilon), None)
        pred = pred / np.clip(pred.sum(axis=1, keepdims=True), float(epsilon), None)
        meta["variables"][str(var)] = {
            "n_groups": int(merged.shape[0]),
            "n_categories": int(len(cats)),
        }
        group_ids = merged[str(group_col)].astype(str).tolist()
        for i, gid in enumerate(group_ids):
            for j, cat in enumerate(cats):
                out_rows.append(
                    {
                        str(group_col): str(gid),
                        str(variable_col): str(var),
                        str(category_col): str(cat),
                        str(target_col): float(pred[i, j]),
                    }
                )

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out, meta
    fit = compare_targets_long(
        predicted_targets_long=out,
        reference_targets_long=ref[ref[str(variable_col)].astype(str).isin(variables)].copy(),
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    meta["fit_against_reference"] = fit
    return out, meta


def blend_prior_targets_long(
    *,
    base_targets_long: Any,
    extra_targets_long: Any,
    group_col: str = "tract_geoid",
    variable_col: str = "variable",
    category_col: str = "category",
    target_col: str = "target",
    variables: list[str] | None = None,
    base_weight: float = 1.0,
    extra_weight: float = 1.0,
    epsilon: float = 1e-8,
) -> Any:
    """
    Blend two prior tables by geometric averaging of normalized shares.
    """
    np, pd = _require_numpy_pandas()
    base = _normalize_targets_long(
        targets_long=base_targets_long,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    extra = _normalize_targets_long(
        targets_long=extra_targets_long,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    if variables is None:
        variables = sorted(set(base[str(variable_col)].astype(str).tolist()) & set(extra[str(variable_col)].astype(str).tolist()))
    variables = [str(v) for v in variables]

    base_den = base.groupby([str(group_col), str(variable_col)], sort=False)[str(target_col)].transform("sum").replace(0.0, 1.0)
    extra_den = extra.groupby([str(group_col), str(variable_col)], sort=False)[str(target_col)].transform("sum").replace(0.0, 1.0)
    base["_share"] = base[str(target_col)] / base_den
    extra["_share"] = extra[str(target_col)] / extra_den

    keep_base = base[~base[str(variable_col)].astype(str).isin(variables)][[str(group_col), str(variable_col), str(category_col), "_share"]].rename(columns={"_share": str(target_col)})
    b_sel = base[base[str(variable_col)].astype(str).isin(variables)][[str(group_col), str(variable_col), str(category_col), "_share"]].rename(columns={"_share": "base_share"})
    e_sel = extra[extra[str(variable_col)].astype(str).isin(variables)][[str(group_col), str(variable_col), str(category_col), "_share"]].rename(columns={"_share": "extra_share"})
    merged = b_sel.merge(e_sel, on=[str(group_col), str(variable_col), str(category_col)], how="outer")
    merged["base_share"] = pd.to_numeric(merged["base_share"], errors="coerce").fillna(float(epsilon))
    merged["extra_share"] = pd.to_numeric(merged["extra_share"], errors="coerce").fillna(float(epsilon))
    merged["_blend"] = np.power(np.clip(merged["base_share"], float(epsilon), None), float(base_weight)) * np.power(
        np.clip(merged["extra_share"], float(epsilon), None),
        float(extra_weight),
    )
    denom = merged.groupby([str(group_col), str(variable_col)], sort=False)["_blend"].transform("sum").replace(0.0, 1.0)
    merged[str(target_col)] = merged["_blend"] / denom
    out = pd.concat(
        [
            keep_base[[str(group_col), str(variable_col), str(category_col), str(target_col)]],
            merged[[str(group_col), str(variable_col), str(category_col), str(target_col)]],
        ],
        axis=0,
        ignore_index=True,
    )
    return out


def allocate_region_type_counts(
    *,
    type_counts: Any,
    hard_targets_long: Any,
    type_catalog: Any | None = None,
    group_col: str = "tract_geoid",
    variable_col: str = "variable",
    category_col: str = "category",
    target_col: str = "target",
    count_col: str = "count",
    type_idx_col: str = "type_idx",
    hard_variables: list[str] | None = None,
    prior_targets_long: Any | None = None,
    prior_variables: list[str] | None = None,
    prior_variable_weights: Mapping[str, float] | None = None,
    residual_targets_long: Any | None = None,
    residual_variables: list[str] | None = None,
    residual_variable_weights: Mapping[str, float] | None = None,
    residual_ratio_clip: float = 5.0,
    group_weights: Any | None = None,
    group_weight_col: str = "weight",
    epsilon: float = 1e-8,
    max_iters: int = 200,
    tol: float = 1e-6,
    drop_zero: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """
    Solve a constrained type -> small-area allocation for a single region.
    """
    np, pd = _require_numpy_pandas()
    if not isinstance(type_counts, pd.DataFrame):
        raise TypeError("type_counts must be a pandas DataFrame")
    if type_catalog is None:
        if str(type_idx_col) not in type_counts.columns:
            raise ValueError("type_catalog is required when type_counts lacks type labels")
        type_catalog = type_counts.drop(columns=[str(count_col)], errors="ignore").copy()
    if not isinstance(type_catalog, pd.DataFrame):
        raise TypeError("type_catalog must be a pandas DataFrame")

    if str(type_idx_col) not in type_counts.columns:
        raise ValueError(f"type_counts missing column: {type_idx_col}")
    if str(count_col) not in type_counts.columns:
        raise ValueError(f"type_counts missing column: {count_col}")

    hard_tgt = _normalize_targets_long(
        targets_long=hard_targets_long,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )
    groups = sorted(hard_tgt[str(group_col)].astype(str).unique().tolist())
    if not groups:
        raise ValueError("hard_targets_long resolved to zero groups")

    type_df = type_counts.copy()
    if any(str(c) not in type_df.columns for c in type_catalog.columns):
        join_cols = [c for c in type_catalog.columns if c != str(count_col)]
        type_df = type_df.merge(type_catalog[join_cols], on=str(type_idx_col), how="left")
    type_df[str(type_idx_col)] = pd.to_numeric(type_df[str(type_idx_col)], errors="raise").astype(int)
    type_df = type_df.sort_values(str(type_idx_col), kind="stable").reset_index(drop=True)
    type_df[str(count_col)] = pd.to_numeric(type_df[str(count_col)], errors="coerce").fillna(0.0).clip(lower=0.0)

    available_vars = [str(v) for v in hard_tgt[str(variable_col)].astype(str).unique().tolist() if str(v) in type_df.columns]
    if hard_variables is None:
        hard_variables = list(available_vars)
    else:
        hard_variables = [str(v) for v in hard_variables if str(v) in available_vars]
    if not hard_variables:
        raise ValueError("No hard_variables remain after intersecting with type columns and target variables")

    hard_tgt, reconcile_meta = reconcile_hard_targets_to_type_counts(
        type_counts=type_df,
        targets_long=hard_tgt,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
        count_col=str(count_col),
        hard_variables=hard_variables,
        epsilon=float(epsilon),
    )
    groups = sorted(hard_tgt[str(group_col)].astype(str).unique().tolist())

    prior, prior_meta = build_type_to_group_prior(
        type_catalog=type_df[[c for c in type_df.columns if c != str(count_col)]].copy(),
        groups=groups,
        prior_targets_long=(prior_targets_long if prior_targets_long is not None else hard_tgt),
        residual_targets_long=residual_targets_long,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
        prior_variables=prior_variables,
        variable_weights=prior_variable_weights,
        residual_variables=residual_variables,
        residual_variable_weights=residual_variable_weights,
        residual_ratio_clip=float(residual_ratio_clip),
        group_weights=group_weights,
        group_weight_col=str(group_weight_col),
        epsilon=float(epsilon),
    )

    row_target = type_df[str(count_col)].to_numpy(dtype=float)
    alloc = prior * row_target.reshape(-1, 1)
    alloc = np.clip(alloc, float(epsilon), None)

    masks_by_var: dict[str, dict[str, Any]] = {}
    for var in hard_variables:
        masks_by_var[str(var)] = {}
        for cat in sorted(type_df[str(var)].astype(str).unique().tolist()):
            masks_by_var[str(var)][str(cat)] = (type_df[str(var)].astype(str).to_numpy() == str(cat))

    group_target_map: dict[str, dict[str, dict[str, float]]] = {}
    for var in hard_variables:
        sub = hard_tgt[hard_tgt[str(variable_col)] == str(var)].copy()
        group_target_map[str(var)] = {}
        for g, gg in sub.groupby(str(group_col), sort=False):
            group_target_map[str(var)][str(g)] = {
                str(r[str(category_col)]): float(r[str(target_col)])
                for _, r in gg[[str(category_col), str(target_col)]].iterrows()
            }

    converged = False
    last_err = None
    for _ in range(int(max_iters)):
        row_now = alloc.sum(axis=1)
        row_scale = row_target / np.clip(row_now, float(epsilon), None)
        alloc *= row_scale.reshape(-1, 1)

        for var in hard_variables:
            target_by_group = group_target_map[str(var)]
            masks = masks_by_var[str(var)]
            for j, g in enumerate(groups):
                col = alloc[:, j]
                for cat, tgt in target_by_group.get(str(g), {}).items():
                    mask = masks.get(str(cat))
                    if mask is None:
                        continue
                    current = float(col[mask].sum())
                    if tgt <= 0.0:
                        col[mask] = 0.0
                        continue
                    if current <= 0.0:
                        col[mask] = float(tgt) / max(int(mask.sum()), 1)
                    else:
                        col[mask] *= float(tgt) / current
                alloc[:, j] = col

        errs = _constraint_errors_from_matrix(
            alloc=alloc,
            type_counts=type_df,
            type_catalog=type_df,
            targets_long=hard_tgt,
            groups=groups,
            group_col=str(group_col),
            variable_col=str(variable_col),
            category_col=str(category_col),
            target_col=str(target_col),
            count_col=str(count_col),
            hard_variables=hard_variables,
            epsilon=float(epsilon),
        )
        last_err = errs
        hard_max = max(
            [float(v["max_abs"]) for v in errs["hard_constraints"].values()] + [float(errs["row_sum"]["max_abs"])],
            default=0.0,
        )
        if hard_max <= float(tol):
            converged = True
            break

    records: list[dict[str, Any]] = []
    meta_rows = type_df[[c for c in type_df.columns if c != str(count_col)]].copy()
    for j, g in enumerate(groups):
        vals = alloc[:, j]
        for i, v in enumerate(vals.tolist()):
            if bool(drop_zero) and float(v) <= 0.0:
                continue
            row = meta_rows.iloc[i].to_dict()
            row[str(group_col)] = str(g)
            row[str(count_col)] = float(v)
            records.append(row)

    out = pd.DataFrame(records)
    meta = {
        "n_types": int(type_df.shape[0]),
        "n_groups": int(len(groups)),
        "groups": [str(g) for g in groups],
        "hard_variables": [str(v) for v in hard_variables],
        "reconcile": reconcile_meta,
        "prior": prior_meta,
        "converged": bool(converged),
        "max_iters": int(max_iters),
        "tol": float(tol),
        "errors": last_err,
    }
    return out, meta


def summarize_type_allocation_against_targets(
    *,
    allocation_long: Any,
    targets_long: Any,
    type_catalog: Any | None = None,
    group_col: str = "tract_geoid",
    variable_col: str = "variable",
    category_col: str = "category",
    target_col: str = "target",
    count_col: str = "count",
    type_idx_col: str = "type_idx",
) -> dict[str, Any]:
    """
    Compare allocated type counts against ACS-style targets_long.
    """
    np, pd = _require_numpy_pandas()
    if not isinstance(allocation_long, pd.DataFrame):
        raise TypeError("allocation_long must be a pandas DataFrame")
    if type_catalog is not None and any(str(c) not in allocation_long.columns for c in type_catalog.columns if c != str(count_col)):
        allocation_long = allocation_long.merge(type_catalog, on=str(type_idx_col), how="left")

    tgt = _normalize_targets_long(
        targets_long=targets_long,
        group_col=str(group_col),
        variable_col=str(variable_col),
        category_col=str(category_col),
        target_col=str(target_col),
    )

    out: dict[str, Any] = {"variables": {}}
    for var in sorted(tgt[str(variable_col)].astype(str).unique().tolist()):
        if str(var) not in allocation_long.columns:
            continue
        pred = (
            allocation_long[[str(group_col), str(var), str(count_col)]]
            .rename(columns={str(var): str(category_col)})
            .groupby([str(group_col), str(category_col)], as_index=False, sort=False)[str(count_col)]
            .sum()
            .rename(columns={str(count_col): "pred"})
        )
        ref = tgt[tgt[str(variable_col)] == str(var)][[str(group_col), str(category_col), str(target_col)]].copy()
        merged = ref.merge(pred, on=[str(group_col), str(category_col)], how="left")
        merged["pred"] = pd.to_numeric(merged["pred"], errors="coerce").fillna(0.0)
        merged[str(target_col)] = pd.to_numeric(merged[str(target_col)], errors="coerce").fillna(0.0)
        merged["abs_err"] = (merged["pred"] - merged[str(target_col)]).abs()
        merged["rel_err"] = merged["abs_err"] / merged[str(target_col)].clip(lower=1e-8)

        tvd_by_group: list[float] = []
        for g, gg in merged.groupby(str(group_col), sort=False):
            p = gg["pred"].to_numpy(dtype=float)
            q = gg[str(target_col)].to_numpy(dtype=float)
            p = p / max(float(p.sum()), 1e-12)
            q = q / max(float(q.sum()), 1e-12)
            tvd_by_group.append(0.5 * float(np.abs(p - q).sum()))
        out["variables"][str(var)] = {
            "mean_abs_err": float(merged["abs_err"].mean()) if not merged.empty else 0.0,
            "max_abs_err": float(merged["abs_err"].max()) if not merged.empty else 0.0,
            "mean_rel_err": float(merged["rel_err"].mean()) if not merged.empty else 0.0,
            "max_rel_err": float(merged["rel_err"].max()) if not merged.empty else 0.0,
            "mean_tvd": float(np.mean(tvd_by_group)) if tvd_by_group else 0.0,
            "max_tvd": float(np.max(tvd_by_group)) if tvd_by_group else 0.0,
        }
    return out


def allocate_joint_wide_to_small_areas(
    *,
    joint_wide: Any,
    schema: Any,
    hard_targets_long: Any,
    region_col: str = "puma_uid",
    group_col: str = "tract_geoid",
    count_col: str = "count",
    type_idx_col: str = "type_idx",
    group_to_region: Any | None = None,
    hard_variables: list[str] | None = None,
    prior_targets_long: Any | None = None,
    prior_variables: list[str] | None = None,
    prior_variable_weights: Mapping[str, float] | None = None,
    residual_targets_long: Any | None = None,
    residual_variables: list[str] | None = None,
    residual_variable_weights: Mapping[str, float] | None = None,
    residual_ratio_clip: float = 5.0,
    group_weights: Any | None = None,
    group_weight_col: str = "weight",
    integerize: bool = False,
    epsilon: float = 1e-8,
    max_iters: int = 200,
    tol: float = 1e-6,
) -> tuple[Any, dict[str, Any]]:
    """
    Multi-region wrapper around `allocate_region_type_counts`.
    """
    _, pd = _require_numpy_pandas()
    if not isinstance(joint_wide, pd.DataFrame):
        raise TypeError("joint_wide must be a pandas DataFrame")
    if not isinstance(hard_targets_long, pd.DataFrame):
        raise TypeError("hard_targets_long must be a pandas DataFrame")

    type_catalog = build_type_catalog(schema=schema, type_idx_col=str(type_idx_col))
    type_counts = joint_wide_to_type_counts(
        joint_wide=joint_wide,
        schema=schema,
        region_col=str(region_col),
        count_col=str(count_col),
        type_idx_col=str(type_idx_col),
        integerize=bool(integerize),
    )

    hard_tgt = hard_targets_long.copy()
    if str(region_col) not in hard_tgt.columns:
        if group_to_region is None:
            raise ValueError(f"hard_targets_long missing {region_col}; provide group_to_region mapping")
        if not isinstance(group_to_region, pd.DataFrame):
            raise TypeError("group_to_region must be a pandas DataFrame")
        if str(group_col) not in group_to_region.columns or str(region_col) not in group_to_region.columns:
            raise ValueError(f"group_to_region must contain {group_col} and {region_col}")
        hard_tgt = hard_tgt.merge(
            group_to_region[[str(group_col), str(region_col)]].drop_duplicates(),
            on=str(group_col),
            how="left",
        )

    prior_tgt = prior_targets_long
    if prior_tgt is not None and str(region_col) not in prior_tgt.columns:
        if group_to_region is None:
            raise ValueError(f"prior_targets_long missing {region_col}; provide group_to_region mapping")
        prior_tgt = prior_tgt.merge(
            group_to_region[[str(group_col), str(region_col)]].drop_duplicates(),
            on=str(group_col),
            how="left",
        )

    residual_tgt = residual_targets_long
    if residual_tgt is not None and str(region_col) not in residual_tgt.columns:
        if group_to_region is None:
            raise ValueError(f"residual_targets_long missing {region_col}; provide group_to_region mapping")
        residual_tgt = residual_tgt.merge(
            group_to_region[[str(group_col), str(region_col)]].drop_duplicates(),
            on=str(group_col),
            how="left",
        )

    region_ids = [str(x) for x in joint_wide[str(region_col)].astype(str).tolist()]
    alloc_blocks: list[Any] = []
    region_meta: dict[str, Any] = {}
    for rid in region_ids:
        tc = type_counts[type_counts[str(region_col)].astype(str) == str(rid)].copy()
        ht = hard_tgt[hard_tgt[str(region_col)].astype(str) == str(rid)].copy()
        if tc.empty or ht.empty:
            continue
        pt = None if prior_tgt is None else prior_tgt[prior_tgt[str(region_col)].astype(str) == str(rid)].copy()
        rt = None if residual_tgt is None else residual_tgt[residual_tgt[str(region_col)].astype(str) == str(rid)].copy()
        gw = None
        if group_weights is not None:
            gw = group_weights
            if isinstance(gw, pd.DataFrame) and str(region_col) in gw.columns:
                gw = gw[gw[str(region_col)].astype(str) == str(rid)].copy()

        block, meta = allocate_region_type_counts(
            type_counts=tc.drop(columns=[str(region_col)], errors="ignore"),
            hard_targets_long=ht.drop(columns=[str(region_col)], errors="ignore"),
            type_catalog=type_catalog,
            group_col=str(group_col),
            count_col=str(count_col),
            type_idx_col=str(type_idx_col),
            hard_variables=hard_variables,
            prior_targets_long=(None if pt is None else pt.drop(columns=[str(region_col)], errors="ignore")),
            prior_variables=prior_variables,
            prior_variable_weights=prior_variable_weights,
            residual_targets_long=(None if rt is None else rt.drop(columns=[str(region_col)], errors="ignore")),
            residual_variables=residual_variables,
            residual_variable_weights=residual_variable_weights,
            residual_ratio_clip=float(residual_ratio_clip),
            group_weights=gw,
            group_weight_col=str(group_weight_col),
            epsilon=float(epsilon),
            max_iters=int(max_iters),
            tol=float(tol),
        )
        block[str(region_col)] = str(rid)
        alloc_blocks.append(block)
        region_meta[str(rid)] = meta

    if alloc_blocks:
        alloc_all = pd.concat(alloc_blocks, axis=0, ignore_index=True)
    else:
        alloc_all = pd.DataFrame(columns=[str(region_col), str(group_col), str(type_idx_col), str(count_col)])

    return alloc_all, {"region_meta": region_meta, "n_regions": int(len(region_meta))}
