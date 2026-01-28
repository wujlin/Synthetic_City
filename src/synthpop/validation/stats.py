from __future__ import annotations

"""
Statistical validation.

Core acceptance metrics (v0):
- marginal errors at tract/BG
- key 2nd-order associations (e.g., income×occupation, residence×work)
- hard-rule violation rate
"""

from typing import Any


def compute_stats_metrics(
    *,
    synthetic: Any,
    reference: Any,
    group_col: str = "puma",
    continuous_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    bin_edges: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """
    Minimal statistical validation (P0):
    - tract/PUMA-grouped marginal TVD on binned continuous + categorical variables
    - key association preservation (Pearson corr / Cramér's V)
    - hard-rule violation rate (minimal: child labor)

    Args:
        synthetic: pandas DataFrame (generated samples).
        reference: pandas DataFrame (PUMS holdout or ACS summary-derived microdata).
        group_col: grouping column shared by both frames (e.g., "puma" or "tract_geoid").
        continuous_cols: numeric columns to bin (default: ["AGEP", "PINCP"]).
        categorical_cols: categorical columns (default: ["SEX", "ESR"]).
        bin_edges: optional fixed bin edges per continuous column.

    Returns:
        dict ready to be dumped as JSON for `metrics/stats_metrics.json`.
    """
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("compute_stats_metrics requires pandas and numpy.") from e

    if not isinstance(synthetic, pd.DataFrame) or not isinstance(reference, pd.DataFrame):
        raise TypeError("synthetic/reference must be pandas DataFrame")

    group_col = str(group_col)
    continuous_cols = list(continuous_cols or ["AGEP", "PINCP"])
    categorical_cols = list(categorical_cols or ["SEX", "ESR"])
    bin_edges = dict(bin_edges or {})

    for c in [group_col] + continuous_cols + categorical_cols:
        if c not in synthetic.columns:
            raise ValueError(f"synthetic missing column: {c}")
        if c not in reference.columns:
            raise ValueError(f"reference missing column: {c}")

    syn = synthetic.copy()
    ref = reference.copy()
    syn[group_col] = syn[group_col].astype(str)
    ref[group_col] = ref[group_col].astype(str)

    # --- helpers ---
    def _tvd(p: "Any", q: "Any") -> float:
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        return 0.5 * float(np.abs(p - q).sum())

    def _norm_counts(series: "Any") -> "Any":
        c = series.value_counts(dropna=False, normalize=True)
        return c.astype(float)

    def _marginal_tvd_by_group(*, syn_col: str, ref_col: str) -> dict[str, Any]:
        by_group: dict[str, float] = {}
        groups = sorted(set(syn[group_col].unique().tolist()) & set(ref[group_col].unique().tolist()))
        for g in groups:
            s_g = syn[syn[group_col] == g]
            r_g = ref[ref[group_col] == g]
            if s_g.empty or r_g.empty:
                continue
            s_counts = _norm_counts(s_g[syn_col])
            r_counts = _norm_counts(r_g[ref_col])
            cats = sorted(set(s_counts.index.tolist()) | set(r_counts.index.tolist()))
            p = np.array([float(s_counts.get(k, 0.0)) for k in cats], dtype=float)
            q = np.array([float(r_counts.get(k, 0.0)) for k in cats], dtype=float)
            by_group[str(g)] = _tvd(p, q)

        vals = list(by_group.values())
        if not vals:
            return {"mean": None, "max": None, "worst_group": None, "by_group": by_group}
        worst_group = max(by_group, key=lambda k: by_group[k])
        return {
            "mean": float(np.mean(vals)),
            "max": float(np.max(vals)),
            "worst_group": str(worst_group),
            "by_group": by_group,
        }

    def _cramers_v(x: "Any", y: "Any") -> float | None:
        t = pd.crosstab(x, y, dropna=False)
        n = float(t.to_numpy().sum())
        if n <= 0:
            return None
        obs = t.to_numpy(dtype=float)
        row = obs.sum(axis=1, keepdims=True)
        col = obs.sum(axis=0, keepdims=True)
        expected = row @ col / n
        mask = expected > 0
        chi2 = float(((obs - expected) ** 2 / np.where(mask, expected, 1.0))[mask].sum())
        r, k = obs.shape
        denom = n * float(min(r - 1, k - 1))
        if denom <= 0:
            return 0.0
        v = float(np.sqrt(chi2 / denom))
        return v if np.isfinite(v) else None

    # --- bin continuous ---
    default_edges: dict[str, list[float]] = {
        "AGEP": [0.0, 5.0, 18.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 1000.0],
        "PINCP": [0.0, 10_000.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0, 150_000.0, 250_000.0, 10_000_000.0],
    }
    used_bin_edges: dict[str, list[float]] = {}
    for col in continuous_cols:
        edges = [float(x) for x in bin_edges.get(col, default_edges.get(col, []))]
        if len(edges) < 2:
            raise ValueError(f"bin_edges for {col} is missing/too short; provide at least 2 edges.")
        used_bin_edges[col] = edges

        syn[col] = pd.to_numeric(syn[col], errors="coerce")
        ref[col] = pd.to_numeric(ref[col], errors="coerce")
        if col == "AGEP":
            syn[col] = syn[col].clip(lower=0.0, upper=99.0)
            ref[col] = ref[col].clip(lower=0.0, upper=99.0)
        if col == "PINCP":
            syn[col] = syn[col].clip(lower=0.0)
            ref[col] = ref[col].clip(lower=0.0)

        syn[f"{col}_bin"] = pd.cut(syn[col], bins=edges, include_lowest=True, right=False)
        ref[f"{col}_bin"] = pd.cut(ref[col], bins=edges, include_lowest=True, right=False)

    # --- marginal TVD ---
    marginal_tvd: dict[str, Any] = {}
    for col in continuous_cols:
        key = f"{col}_bin"
        marginal_tvd[key] = _marginal_tvd_by_group(syn_col=key, ref_col=key)
    for col in categorical_cols:
        marginal_tvd[col] = _marginal_tvd_by_group(syn_col=col, ref_col=col)

    # --- association metrics ---
    association: dict[str, Any] = {}
    # Pearson corr for (AGEP, PINCP) if present.
    if "AGEP" in continuous_cols and "PINCP" in continuous_cols:
        s_xy = syn[["AGEP", "PINCP"]].dropna()
        r_xy = ref[["AGEP", "PINCP"]].dropna()
        s_corr = None
        r_corr = None
        if len(s_xy) >= 2:
            v = np.corrcoef(s_xy["AGEP"].to_numpy(dtype=float), s_xy["PINCP"].to_numpy(dtype=float))[0, 1]
            s_corr = float(v) if np.isfinite(v) else None
        if len(r_xy) >= 2:
            v = np.corrcoef(r_xy["AGEP"].to_numpy(dtype=float), r_xy["PINCP"].to_numpy(dtype=float))[0, 1]
            r_corr = float(v) if np.isfinite(v) else None
        association["AGEP_PINCP"] = {
            "synthetic_corr": s_corr,
            "reference_corr": r_corr,
            "diff": (None if (s_corr is None or r_corr is None) else float(s_corr - r_corr)),
        }

    # Cramer's V for (SEX, ESR) if present.
    if "SEX" in categorical_cols and "ESR" in categorical_cols:
        s_v = _cramers_v(syn["SEX"].astype(str), syn["ESR"].astype(str))
        r_v = _cramers_v(ref["SEX"].astype(str), ref["ESR"].astype(str))
        association["SEX_ESR"] = {
            "synthetic_cramers_v": s_v,
            "reference_cramers_v": r_v,
            "diff": (None if (s_v is None or r_v is None) else float(s_v - r_v)),
        }

    # --- hard rule violations (minimal) ---
    def _child_labor_violations(df: "Any") -> tuple[int, float]:
        age = pd.to_numeric(df.get("AGEP"), errors="coerce")
        esr = df.get("ESR")
        if age is None or esr is None:
            return 0, 0.0
        esr_s = esr.astype(str)
        mask = (age < 16) & esr_s.isin({"1", "2", "3"})
        count = int(mask.sum())
        n = int(df.shape[0])
        rate = float(count / n) if n > 0 else 0.0
        return count, rate

    child_count, child_rate = _child_labor_violations(syn)
    hard_rule_violations = {
        "child_labor": {
            "rule": "AGEP<16 and ESR in (1,2,3)",
            "count": child_count,
            "rate": child_rate,
        }
    }

    # --- meta ---
    groups_all = sorted(set(syn[group_col].unique().tolist()) | set(ref[group_col].unique().tolist()))
    meta = {
        "group_col": group_col,
        "n_synthetic": int(syn.shape[0]),
        "n_reference": int(ref.shape[0]),
        "n_groups": int(len(groups_all)),
        "bin_edges": used_bin_edges,
    }

    return {
        "marginal_tvd": marginal_tvd,
        "association": association,
        "hard_rule_violations": hard_rule_violations,
        "meta": meta,
    }


def compute_stats_metrics_against_targets_long(
    *,
    synthetic: Any,
    targets_long: Any,
    group_col: str = "puma",
    variable_col: str = "variable",
    category_col: str = "category",
    target_col: str = "target",
    continuous_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    bin_edges: dict[str, list[float]] | None = None,
    variables: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compare synthetic microdata against external target marginals (ACS-style long format).

    Expected targets_long schema (minimum):
    - group_col: e.g. "puma" or "tract_geoid"
    - variable_col: e.g. "AGEP_bin", "SEX"
    - category_col: category key (string recommended)
    - target_col: target count (non-negative)
    """
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("compute_stats_metrics_against_targets_long requires pandas and numpy.") from e

    if not isinstance(synthetic, pd.DataFrame) or not isinstance(targets_long, pd.DataFrame):
        raise TypeError("synthetic/targets_long must be pandas DataFrame")

    group_col = str(group_col)
    variable_col = str(variable_col)
    category_col = str(category_col)
    target_col = str(target_col)
    continuous_cols = list(continuous_cols or ["AGEP", "PINCP"])
    categorical_cols = list(categorical_cols or ["SEX", "ESR"])
    bin_edges = dict(bin_edges or {})

    for c in [group_col] + continuous_cols + categorical_cols:
        if c not in synthetic.columns:
            raise ValueError(f"synthetic missing column: {c}")
    for c in [group_col, variable_col, category_col, target_col]:
        if c not in targets_long.columns:
            raise ValueError(f"targets_long missing column: {c}")

    syn = synthetic.copy()
    tgt = targets_long.copy()
    syn[group_col] = syn[group_col].astype(str)
    tgt[group_col] = tgt[group_col].astype(str)
    tgt[variable_col] = tgt[variable_col].astype(str)
    tgt[category_col] = tgt[category_col].astype(str)
    tgt[target_col] = pd.to_numeric(tgt[target_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Bin continuous (same defaults as compute_stats_metrics)
    default_edges: dict[str, list[float]] = {
        "AGEP": [0.0, 5.0, 18.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 1000.0],
        "PINCP": [0.0, 10_000.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0, 150_000.0, 250_000.0, 10_000_000.0],
    }
    used_bin_edges: dict[str, list[float]] = {}
    for col in continuous_cols:
        edges = [float(x) for x in bin_edges.get(col, default_edges.get(col, []))]
        if len(edges) < 2:
            raise ValueError(f"bin_edges for {col} is missing/too short; provide at least 2 edges.")
        used_bin_edges[col] = edges

        syn[col] = pd.to_numeric(syn[col], errors="coerce")
        if col == "AGEP":
            syn[col] = syn[col].clip(lower=0.0, upper=99.0)
        if col == "PINCP":
            syn[col] = syn[col].clip(lower=0.0)

        syn[f"{col}_bin"] = pd.cut(syn[col], bins=edges, include_lowest=True, right=False).astype(str)

    for col in categorical_cols:
        syn[col] = syn[col].astype(str)

    def _tvd(p: "Any", q: "Any") -> float:
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        return 0.5 * float(np.abs(p - q).sum())

    # --- derived variables (for ACS tables that don't match PUMS categories 1:1) ---
    target_vars = set(tgt[variable_col].unique().tolist())
    if "ESR_16p" in target_vars and "ESR_16p" not in syn.columns:
        if "AGEP" not in syn.columns or "ESR" not in syn.columns:
            raise ValueError('targets_long requests "ESR_16p" but synthetic lacks AGEP/ESR.')

        age = pd.to_numeric(syn.get("AGEP"), errors="coerce")
        esr = syn.get("ESR").astype(str)
        mask16 = age >= 16

        out = pd.Series([None] * int(syn.shape[0]), index=syn.index, dtype=object)
        out.loc[mask16] = "not_in_labor_force"
        out.loc[mask16 & esr.isin({"1", "2"})] = "employed"
        out.loc[mask16 & esr.isin({"3"})] = "unemployed"
        out.loc[mask16 & esr.isin({"4", "5"})] = "armed_forces"
        syn["ESR_16p"] = out.astype(object)

    # Determine which variables to evaluate.
    if variables is None:
        variables = sorted(set(tgt[variable_col].unique().tolist()))
    else:
        variables = [str(v) for v in variables]

    available_vars = set(syn.columns.tolist())
    used_vars: list[str] = []
    skipped_vars: list[str] = []

    def _tvd_by_group_for_var(var: str) -> dict[str, Any]:
        by_group: dict[str, float] = {}
        t_var = tgt[tgt[variable_col] == var]
        if t_var.empty:
            return {"mean": None, "max": None, "worst_group": None, "by_group": by_group}

        syn_groups = set(syn[group_col].unique().tolist())
        tgt_groups = set(t_var[group_col].unique().tolist())
        groups = sorted(syn_groups & tgt_groups)
        for g in groups:
            s_g = syn[syn[group_col] == g]
            if s_g.empty:
                continue
            if var == "ESR_16p":
                age_g = pd.to_numeric(s_g.get("AGEP"), errors="coerce")
                s_g = s_g[age_g >= 16]
                if s_g.empty:
                    continue
            t_g = t_var[t_var[group_col] == g]
            total = float(t_g[target_col].sum())
            if total <= 0:
                continue

            s_counts = s_g[var].astype(str).value_counts(dropna=False, normalize=True).astype(float)
            t_counts = t_g.groupby(category_col, sort=False)[target_col].sum()
            t_prob = (t_counts / total).astype(float)

            cats = sorted(set(s_counts.index.tolist()) | set(t_prob.index.tolist()))
            p = np.array([float(s_counts.get(k, 0.0)) for k in cats], dtype=float)
            q = np.array([float(t_prob.get(k, 0.0)) for k in cats], dtype=float)
            by_group[str(g)] = _tvd(p, q)

        vals = list(by_group.values())
        if not vals:
            return {"mean": None, "max": None, "worst_group": None, "by_group": by_group}
        worst_group = max(by_group, key=lambda k: by_group[k])
        return {
            "mean": float(np.mean(vals)),
            "max": float(np.max(vals)),
            "worst_group": str(worst_group),
            "by_group": by_group,
        }

    marginal_tvd: dict[str, Any] = {}
    for v in variables:
        if v not in available_vars:
            skipped_vars.append(v)
            continue
        used_vars.append(v)
        marginal_tvd[v] = _tvd_by_group_for_var(v)

    # Hard rule violations (same minimal check)
    def _child_labor_violations(df: "Any") -> tuple[int, float]:
        age = pd.to_numeric(df.get("AGEP"), errors="coerce")
        esr = df.get("ESR")
        if age is None or esr is None:
            return 0, 0.0
        esr_s = esr.astype(str)
        mask = (age < 16) & esr_s.isin({"1", "2", "3"})
        count = int(mask.sum())
        n = int(df.shape[0])
        rate = float(count / n) if n > 0 else 0.0
        return count, rate

    child_count, child_rate = _child_labor_violations(syn)
    hard_rule_violations = {
        "child_labor": {
            "rule": "AGEP<16 and ESR in (1,2,3)",
            "count": child_count,
            "rate": child_rate,
        }
    }

    meta = {
        "group_col": group_col,
        "n_synthetic": int(syn.shape[0]),
        "n_targets_rows": int(tgt.shape[0]),
        "n_groups_synthetic": int(syn[group_col].nunique(dropna=False)),
        "n_groups_targets": int(tgt[group_col].nunique(dropna=False)),
        "variables_used": used_vars,
        "variables_skipped_missing_in_synthetic": skipped_vars,
        "bin_edges": used_bin_edges,
        "reference_source": "targets_long",
        "variable_scopes": {"ESR_16p": "AGEP>=16 (match ACS B23025 population 16+)"} if "ESR_16p" in used_vars else {},
    }

    return {
        "marginal_tvd": marginal_tvd,
        "association": {},
        "hard_rule_violations": hard_rule_violations,
        "meta": meta,
    }
