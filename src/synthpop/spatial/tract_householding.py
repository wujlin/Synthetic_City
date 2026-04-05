from __future__ import annotations

"""
Tract-level household synthesis from person-level synthetic population.

Design goal:
- keep phase2 person marginals intact
- introduce a lightweight, explicit household layer before phase3
- anchor household shells with tract-level ACS B11016
- optionally attach tract-level ACS B19001 household-income bins
"""

from collections import deque
import gzip
import io
import json
import math
import random
import re
import urllib.request
from typing import Any


def _require_numpy_pandas() -> tuple[Any, Any]:
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("tract_householding requires numpy and pandas.") from e
    return np, pd


def _largest_remainder(vec: Any, *, total: int) -> Any:
    np, _ = _require_numpy_pandas()
    arr = np.asarray(vec, dtype=float)
    base = np.floor(arr).astype(int)
    rem = int(total - int(base.sum()))
    if rem < 0:
        frac = arr - np.floor(arr)
        order = np.argsort(frac)
        for i in order[: abs(rem)].tolist():
            if base[i] > 0:
                base[i] -= 1
        return base
    if rem == 0:
        return base
    frac = arr - np.floor(arr)
    order = np.argsort(-frac)
    base[order[:rem]] += 1
    return base


def _make_tract_geoid(df: Any) -> Any:
    _, pd = _require_numpy_pandas()
    state = df["state"].astype(str).str.zfill(2)
    county = df["county"].astype(str).str.zfill(3)
    tract = df["tract"].astype(str).str.zfill(6)
    return (state + county + tract).astype(str)


def _fetch_acs_table(*, table_id: str, year: str = "2022", statefp: str = "26", out_path: Any | None = None) -> Any:
    _, pd = _require_numpy_pandas()
    table_id = str(table_id).upper().strip()
    counts = {
        "B11016": 16,
        "B19001": 17,
    }
    if table_id not in counts:
        raise ValueError(f"unsupported table_id: {table_id}")
    vars_list = [f"{table_id}_{i:03d}E" for i in range(1, counts[table_id] + 1)]
    url = (
        f"https://api.census.gov/data/{year}/acs/acs5?"
        f"get=NAME,{','.join(vars_list)}&for=tract:*&in=state:{str(statefp).zfill(2)}"
    )
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    df = pd.DataFrame(data[1:], columns=data[0])
    for col in vars_list:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["GEOID"] = _make_tract_geoid(df)
    if out_path is not None:
        out_path = str(out_path)
        if out_path.endswith(".gz"):
            with gzip.open(out_path, "wt", encoding="utf-8", newline="") as f:
                df.to_csv(f, index=False)
        else:
            df.to_csv(out_path, index=False)
    return df


def load_or_fetch_acs_table(*, path: Any, table_id: str, year: str = "2022", statefp: str = "26") -> Any:
    _, pd = _require_numpy_pandas()
    from pathlib import Path

    path = Path(path).expanduser().resolve()
    if path.exists():
        if path.suffix.lower() == ".gz":
            return pd.read_csv(path, compression="gzip", low_memory=False)
        return pd.read_csv(path, low_memory=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _fetch_acs_table(table_id=table_id, year=str(year), statefp=str(statefp), out_path=path)


def b11016_to_household_shell_targets(*, b11016_df: Any, group_col: str = "tract_geoid") -> Any:
    _, pd = _require_numpy_pandas()
    df = b11016_df.copy()
    if str(group_col) not in df.columns:
        if "GEOID" in df.columns:
            df[str(group_col)] = df["GEOID"].astype(str)
        elif {"state", "county", "tract"}.issubset(df.columns):
            df[str(group_col)] = _make_tract_geoid(df)
        else:
            raise ValueError(f"b11016_df missing group column: {group_col}")
    df[str(group_col)] = df[str(group_col)].astype(str)
    records: list[dict[str, Any]] = []

    family_cols = {
        2: "B11016_003E",
        3: "B11016_004E",
        4: "B11016_005E",
        5: "B11016_006E",
        6: "B11016_007E",
        7: "B11016_008E",
    }
    nonfamily_cols = {
        1: "B11016_010E",
        2: "B11016_011E",
        3: "B11016_012E",
        4: "B11016_013E",
        5: "B11016_014E",
        6: "B11016_015E",
        7: "B11016_016E",
    }

    for row in df.itertuples(index=False):
        g = str(getattr(row, str(group_col)))
        for size, col in family_cols.items():
            n = int(round(float(getattr(row, col, 0.0) or 0.0)))
            if n > 0:
                records.append({str(group_col): g, "household_type": "family", "household_size": int(size), "n_target": int(n)})
        for size, col in nonfamily_cols.items():
            n = int(round(float(getattr(row, col, 0.0) or 0.0)))
            if n > 0:
                records.append({str(group_col): g, "household_type": "nonfamily", "household_size": int(size), "n_target": int(n)})
    return pd.DataFrame.from_records(records)


def b19001_to_income_targets(*, b19001_df: Any, group_col: str = "tract_geoid") -> Any:
    _, pd = _require_numpy_pandas()
    df = b19001_df.copy()
    if str(group_col) not in df.columns:
        if "GEOID" in df.columns:
            df[str(group_col)] = df["GEOID"].astype(str)
        elif {"state", "county", "tract"}.issubset(df.columns):
            df[str(group_col)] = _make_tract_geoid(df)
        else:
            raise ValueError(f"b19001_df missing group column: {group_col}")
    df[str(group_col)] = df[str(group_col)].astype(str)
    bins = [
        (0.0, 10_000.0, "B19001_002E"),
        (10_000.0, 15_000.0, "B19001_003E"),
        (15_000.0, 20_000.0, "B19001_004E"),
        (20_000.0, 25_000.0, "B19001_005E"),
        (25_000.0, 30_000.0, "B19001_006E"),
        (30_000.0, 35_000.0, "B19001_007E"),
        (35_000.0, 40_000.0, "B19001_008E"),
        (40_000.0, 45_000.0, "B19001_009E"),
        (45_000.0, 50_000.0, "B19001_010E"),
        (50_000.0, 60_000.0, "B19001_011E"),
        (60_000.0, 75_000.0, "B19001_012E"),
        (75_000.0, 100_000.0, "B19001_013E"),
        (100_000.0, 125_000.0, "B19001_014E"),
        (125_000.0, 150_000.0, "B19001_015E"),
        (150_000.0, 200_000.0, "B19001_016E"),
        (200_000.0, float("inf"), "B19001_017E"),
    ]
    rows: list[dict[str, Any]] = []
    for rec in df.itertuples(index=False):
        g = str(getattr(rec, str(group_col)))
        for lo, hi, col in bins:
            n = int(round(float(getattr(rec, col, 0.0) or 0.0)))
            if hi == float("inf"):
                label = f"[{lo}, inf)"
            else:
                label = f"[{lo}, {hi})"
            rows.append({str(group_col): g, "HHINCP_bin": label, "n_target": int(max(n, 0))})
    return pd.DataFrame.from_records(rows)


_INTERVAL_RE = re.compile(r"[\[\(]\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([^)\]]+)\s*[\)\]]")


def _interval_midpoint(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    m = _INTERVAL_RE.search(s)
    if not m:
        return None
    lo = float(m.group(1))
    hi_s = m.group(2).strip().lower()
    if hi_s in {"inf", "+inf", "infinity"}:
        hi = lo + 25.0
    else:
        hi = float(hi_s)
    return 0.5 * (lo + hi)


def _earn_rank(value: Any) -> float:
    mid = _interval_midpoint(value)
    if mid is not None:
        return float(mid)
    s = str(value).strip().lower()
    if s in {"not_16p", "not_16+", "not16p"}:
        return -1.0
    if s in {"no_income", "none"}:
        return 0.0
    return -0.5


def _person_priority_table(persons_g: Any, *, age_col: str, earn_col: str) -> dict[str, list[int]]:
    _, pd = _require_numpy_pandas()
    work = persons_g.copy().reset_index(drop=False).rename(columns={"index": "_row_idx"})
    work["_age_mid"] = work[str(age_col)].map(_interval_midpoint).astype(float)
    work["_age_mid"] = work["_age_mid"].fillna(40.0)
    work["_is_child"] = work["_age_mid"] < 18.0
    work["_is_adult"] = ~work["_is_child"]
    work["_earn_rank"] = work[str(earn_col)].map(_earn_rank).astype(float)
    work["_tie"] = list(range(int(work.shape[0])))

    def _ordered(mask: Any, *, keys: list[str], ascending: list[bool]) -> list[int]:
        part = work.loc[mask].sort_values(keys, ascending=ascending, kind="mergesort")
        return part["_row_idx"].astype(int).tolist()

    return {
        "all": _ordered(work.index == work.index, keys=["_is_adult", "_age_mid", "_earn_rank", "_tie"], ascending=[False, False, False, True]),
        "head": _ordered(work.index == work.index, keys=["_is_adult", "_earn_rank", "_age_mid", "_tie"], ascending=[False, False, False, True]),
        "adult": _ordered(work["_is_adult"], keys=["_age_mid", "_earn_rank", "_tie"], ascending=[False, False, True]),
        "child": _ordered(work["_is_child"], keys=["_age_mid", "_tie"], ascending=[True, True]),
        "adult_alt": _ordered(work["_is_adult"], keys=["_earn_rank", "_age_mid", "_tie"], ascending=[False, False, True]),
        "meta": work.set_index("_row_idx")[["_age_mid", "_is_child", "_is_adult", "_earn_rank"]],
    }


def _pop_next(order: deque[int], *, used: set[int], allow: set[int] | None = None) -> int | None:
    while order:
        idx = int(order.popleft())
        if idx in used:
            continue
        if allow is not None and idx not in allow:
            continue
        used.add(idx)
        return idx
    return None


def _family_min_size(hh_type: str) -> int:
    return 2 if str(hh_type) == "family" else 1


def _build_shells_for_group(shell_targets_g: Any, *, n_persons: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, pd = _require_numpy_pandas()
    shells: list[dict[str, Any]] = []
    if shell_targets_g is not None and int(getattr(shell_targets_g, "shape", [0])[0]) > 0:
        for row in shell_targets_g.itertuples(index=False):
            n_target = int(getattr(row, "n_target"))
            hh_type = str(getattr(row, "household_type"))
            hh_size = int(getattr(row, "household_size"))
            for _ in range(int(n_target)):
                shells.append(
                    {
                        "household_type": hh_type,
                        "size_original": hh_size,
                        "size_final": hh_size,
                        "source_stage": "b11016_primary",
                    }
                )
    if not shells:
        shells = [{"household_type": "nonfamily", "size_original": 1, "size_final": 1, "source_stage": "synthetic_default"} for _ in range(int(n_persons))]

    meta = {
        "n_shells_raw": int(len(shells)),
        "people_capacity_raw": int(sum(int(x["size_final"]) for x in shells)),
        "n_size_reductions": 0,
        "n_size_expansions": 0,
        "n_shells_dropped": 0,
        "n_residual_shells_added": 0,
    }

    total = int(sum(int(x["size_final"]) for x in shells))
    if total < int(n_persons):
        expandable = [i for i, s in enumerate(shells) if int(s["size_final"]) >= 7]
        if not expandable:
            expandable = sorted(
                list(range(len(shells))),
                key=lambda i: (str(shells[i]["household_type"]) == "family", int(shells[i]["size_final"])),
                reverse=True,
            )
        if not expandable:
            expandable = []
        ptr = 0
        while total < int(n_persons) and expandable:
            idx = int(expandable[ptr % len(expandable)])
            shells[idx]["size_final"] = int(shells[idx]["size_final"]) + 1
            shells[idx]["source_stage"] = "b11016_expanded"
            total += 1
            ptr += 1
            meta["n_size_expansions"] += 1
        while total < int(n_persons):
            shells.append({"household_type": "nonfamily", "size_original": 1, "size_final": 1, "source_stage": "residual_added"})
            total += 1
            meta["n_residual_shells_added"] += 1
    elif total > int(n_persons):
        while total > int(n_persons):
            reducible = [i for i, s in enumerate(shells) if int(s["size_final"]) > _family_min_size(str(s["household_type"]))]
            if reducible:
                idx = max(reducible, key=lambda i: int(shells[i]["size_final"]))
                shells[idx]["size_final"] = int(shells[idx]["size_final"]) - 1
                shells[idx]["source_stage"] = "b11016_reduced"
                total -= 1
                meta["n_size_reductions"] += 1
                continue
            nonfamily_size1 = [i for i, s in enumerate(shells) if str(s["household_type"]) == "nonfamily" and int(s["size_final"]) == 1]
            if nonfamily_size1:
                idx = int(nonfamily_size1[-1])
                total -= int(shells[idx]["size_final"])
                shells.pop(idx)
                meta["n_shells_dropped"] += 1
                continue
            family_size2 = [i for i, s in enumerate(shells) if str(s["household_type"]) == "family" and int(s["size_final"]) == 2]
            if family_size2:
                idx = int(family_size2[-1])
                total -= int(shells[idx]["size_final"])
                shells.pop(idx)
                meta["n_shells_dropped"] += 1
                continue
            break
        if total < int(n_persons):
            while total < int(n_persons):
                shells.append({"household_type": "nonfamily", "size_original": 1, "size_final": 1, "source_stage": "residual_added"})
                total += 1
                meta["n_residual_shells_added"] += 1

    shells = [s for s in shells if int(s["size_final"]) > 0]
    meta["n_shells_final"] = int(len(shells))
    meta["people_capacity_final"] = int(sum(int(x["size_final"]) for x in shells))
    return shells, meta


def _assign_shell_members(*, hh_type: str, hh_size: int, orders: dict[str, Any], used: set[int]) -> list[int]:
    head = orders["head_q"]
    adult = orders["adult_q"]
    child = orders["child_q"]
    all_q = orders["all_q"]
    adult_alt = orders["adult_alt_q"]
    meta = orders["meta"]
    members: list[int] = []

    def take(q: deque[int], allow: set[int] | None = None) -> int | None:
        return _pop_next(q, used=used, allow=allow)

    if str(hh_type) == "family":
        anchor = take(head)
        if anchor is None:
            anchor = take(all_q)
        if anchor is not None:
            members.append(anchor)
        if int(hh_size) == 2:
            cand = take(adult_alt)
            if cand is None:
                cand = take(child)
            if cand is None:
                cand = take(all_q)
            if cand is not None:
                members.append(cand)
        else:
            if len(members) < int(hh_size):
                cand = take(adult_alt)
                if cand is not None:
                    members.append(cand)
            while len(members) < int(hh_size):
                cand = take(child)
                if cand is None:
                    cand = take(adult)
                if cand is None:
                    cand = take(all_q)
                if cand is None:
                    break
                members.append(cand)
    else:
        first = take(adult)
        if first is None:
            first = take(all_q)
        if first is not None:
            members.append(first)
        while len(members) < int(hh_size):
            cand = take(adult)
            if cand is None:
                cand = take(all_q)
            if cand is None:
                break
            members.append(cand)

    if len(members) < int(hh_size):
        while len(members) < int(hh_size):
            cand = take(all_q)
            if cand is None:
                break
            members.append(cand)
    if len(members) > int(hh_size):
        members = members[: int(hh_size)]

    members = [int(x) for x in members]
    if str(hh_type) == "family" and members:
        child_count = int(meta.loc[members, "_is_child"].sum())
        if child_count == len(members) and len(members) >= 2:
            # Avoid child-only family households when an adult still exists.
            adult_candidate = take(adult)
            if adult_candidate is not None:
                rep_pos = int(max(range(len(members)), key=lambda i: float(meta.loc[members[i], "_age_mid"])))
                used.discard(int(members[rep_pos]))
                members[rep_pos] = int(adult_candidate)
    return members


def _member_role(*, hh_type: str, rank_in_household: int, is_child: bool) -> str:
    if int(rank_in_household) == 0:
        return "householder"
    if bool(is_child):
        return "child_member"
    if str(hh_type) == "family":
        return "adult_member"
    return "nonfamily_member"


def _assign_income_bins_to_households(*, households_g: Any, income_targets_g: Any, group_col: str) -> Any:
    np, pd = _require_numpy_pandas()
    out = households_g.copy()
    if income_targets_g is None or int(getattr(income_targets_g, "shape", [0])[0]) == 0 or out.empty:
        out["HHINCP_bin"] = pd.Series([None] * int(out.shape[0]), dtype=object)
        return out
    tgt = income_targets_g.copy()
    tgt = tgt[tgt["n_target"] > 0].copy()
    if tgt.empty:
        out["HHINCP_bin"] = pd.Series([None] * int(out.shape[0]), dtype=object)
        return out
    tgt["_lo"] = tgt["HHINCP_bin"].map(lambda x: _interval_midpoint(str(x)) or -1.0)
    tgt = tgt.sort_values(["_lo", "HHINCP_bin"], kind="mergesort").reset_index(drop=True)
    weights = tgt["n_target"].to_numpy(dtype=float)
    weights = weights / max(float(weights.sum()), 1.0) * float(out.shape[0])
    scaled = _largest_remainder(weights, total=int(out.shape[0]))
    labels: list[str] = []
    for label, n in zip(tgt["HHINCP_bin"].tolist(), scaled.tolist()):
        labels.extend([str(label)] * int(n))
    if len(labels) < int(out.shape[0]):
        labels.extend([str(tgt["HHINCP_bin"].iloc[-1])] * (int(out.shape[0]) - len(labels)))
    labels = labels[: int(out.shape[0])]
    out = out.sort_values(["household_income_score", "household_size", "household_id"], ascending=[True, True, True], kind="mergesort").copy()
    out["HHINCP_bin"] = labels
    return out.sort_values("household_id", kind="mergesort").reset_index(drop=True)


def synthesize_households_from_persons(
    *,
    persons: Any,
    shell_targets: Any,
    income_targets: Any | None = None,
    group_col: str = "tract_geoid",
    person_id_col: str = "person_id",
    age_col: str = "AGEP_bin",
    earn_col: str = "EARN_16p_bin",
    household_id_prefix: str = "hh",
    seed: int = 0,
) -> tuple[Any, Any, dict[str, Any]]:
    np, pd = _require_numpy_pandas()
    if not isinstance(persons, pd.DataFrame):
        raise TypeError("persons must be a pandas DataFrame")
    need = [str(group_col), str(person_id_col), str(age_col), str(earn_col)]
    miss = [c for c in need if c not in persons.columns]
    if miss:
        raise ValueError(f"persons missing columns: {miss}")

    out = persons.copy().reset_index(drop=True)
    out[str(group_col)] = out[str(group_col)].astype(str)
    out["household_id"] = pd.Series([None] * int(out.shape[0]), dtype=object)
    out["household_type"] = pd.Series([None] * int(out.shape[0]), dtype=object)
    out["household_size"] = pd.Series([None] * int(out.shape[0]), dtype="float64")
    out["household_role"] = pd.Series([None] * int(out.shape[0]), dtype=object)

    shell_targets = shell_targets.copy() if shell_targets is not None else pd.DataFrame(columns=[str(group_col), "household_type", "household_size", "n_target"])
    if not shell_targets.empty:
        shell_targets[str(group_col)] = shell_targets[str(group_col)].astype(str)
    if income_targets is not None:
        income_targets = income_targets.copy()
        if not income_targets.empty:
            income_targets[str(group_col)] = income_targets[str(group_col)].astype(str)

    household_rows: list[dict[str, Any]] = []
    meta_groups: dict[str, Any] = {}
    rng = random.Random(int(seed))
    hh_seq = 0

    for g, idx in out.groupby(str(group_col), sort=False).groups.items():
        group_idx = list(idx)
        persons_g = out.loc[group_idx, [str(person_id_col), str(group_col), str(age_col), str(earn_col)]].copy()
        persons_g = persons_g.reset_index(drop=True)
        if not persons_g.empty:
            order = list(range(int(persons_g.shape[0])))
            rng.shuffle(order)
            persons_g = persons_g.iloc[order].reset_index(drop=True)
        shell_targets_g = shell_targets[shell_targets[str(group_col)] == str(g)].copy() if not shell_targets.empty else None
        shells, shell_meta = _build_shells_for_group(shell_targets_g, n_persons=int(persons_g.shape[0]))

        orders = _person_priority_table(persons_g, age_col=str(age_col), earn_col=str(earn_col))
        orders["all_q"] = deque(orders["all"])
        orders["head_q"] = deque(orders["head"])
        orders["adult_q"] = deque(orders["adult"])
        orders["child_q"] = deque(orders["child"])
        orders["adult_alt_q"] = deque(orders["adult_alt"])
        used: set[int] = set()
        assignments: list[dict[str, Any]] = []
        group_hh_rows: list[dict[str, Any]] = []

        for shell in shells:
            hh_type = str(shell["household_type"])
            hh_size = int(shell["size_final"])
            members = _assign_shell_members(hh_type=hh_type, hh_size=hh_size, orders=orders, used=used)
            if not members:
                continue
            if len(members) != hh_size:
                shell["source_stage"] = "partially_filled"
            hh_seq += 1
            hh_id = f"{str(household_id_prefix)}_{str(g)}_{hh_seq:07d}"
            ages = orders["meta"].loc[members, "_age_mid"].tolist()
            child_flags = orders["meta"].loc[members, "_is_child"].tolist()
            earn_scores = orders["meta"].loc[members, "_earn_rank"].tolist()
            for rank, local_idx in enumerate(members):
                global_pos = int(group_idx[order[local_idx]])
                assignments.append(
                    {
                        "_global_pos": global_pos,
                        "household_id": hh_id,
                        "household_type": hh_type,
                        "household_size": int(len(members)),
                        "household_role": _member_role(hh_type=hh_type, rank_in_household=int(rank), is_child=bool(child_flags[rank])),
                    }
                )
            group_hh_rows.append(
                {
                    str(group_col): str(g),
                    "household_id": hh_id,
                    "household_type": hh_type,
                    "household_size": int(len(members)),
                    "n_children": int(sum(bool(x) for x in child_flags)),
                    "n_adults": int(len(members) - sum(bool(x) for x in child_flags)),
                    "mean_age_proxy": float(sum(float(x) for x in ages) / max(len(ages), 1)),
                    "household_income_score": float(sum(float(x) for x in earn_scores) / max(len(earn_scores), 1)),
                    "source_stage": str(shell["source_stage"]),
                }
            )

        # Safety: any remaining persons become residual nonfamily households.
        unassigned_local = [i for i in range(int(persons_g.shape[0])) if i not in used]
        for local_idx in unassigned_local:
            hh_seq += 1
            hh_id = f"{str(household_id_prefix)}_{str(g)}_{hh_seq:07d}"
            global_pos = int(group_idx[order[local_idx]])
            assignments.append(
                {
                    "_global_pos": global_pos,
                    "household_id": hh_id,
                    "household_type": "nonfamily",
                    "household_size": 1,
                    "household_role": "householder",
                }
            )
            group_hh_rows.append(
                {
                    str(group_col): str(g),
                    "household_id": hh_id,
                    "household_type": "nonfamily",
                    "household_size": 1,
                    "n_children": int(orders["meta"].loc[local_idx, "_is_child"]),
                    "n_adults": int(not bool(orders["meta"].loc[local_idx, "_is_child"])),
                    "mean_age_proxy": float(orders["meta"].loc[local_idx, "_age_mid"]),
                    "household_income_score": float(orders["meta"].loc[local_idx, "_earn_rank"]),
                    "source_stage": "residual_single",
                }
            )

        assign_df = pd.DataFrame.from_records(assignments)
        for rec in assign_df.to_dict(orient="records"):
            pos = int(rec["_global_pos"])
            out.at[pos, "household_id"] = str(rec["household_id"])
            out.at[pos, "household_type"] = str(rec["household_type"])
            out.at[pos, "household_size"] = int(rec["household_size"])
            out.at[pos, "household_role"] = str(rec["household_role"])

        hh_df = pd.DataFrame.from_records(group_hh_rows)
        income_targets_g = None
        if income_targets is not None and not income_targets.empty:
            income_targets_g = income_targets[income_targets[str(group_col)] == str(g)].copy()
        hh_df = _assign_income_bins_to_households(households_g=hh_df, income_targets_g=income_targets_g, group_col=str(group_col))
        household_rows.extend(hh_df.to_dict(orient="records"))

        meta_groups[str(g)] = {
            "n_persons": int(persons_g.shape[0]),
            "n_households": int(hh_df.shape[0]),
            "avg_household_size": float(float(persons_g.shape[0]) / max(int(hh_df.shape[0]), 1)),
            "n_family_households": int((hh_df["household_type"] == "family").sum()) if not hh_df.empty else 0,
            "n_nonfamily_households": int((hh_df["household_type"] == "nonfamily").sum()) if not hh_df.empty else 0,
            "n_residual_single_households": int((hh_df["source_stage"] == "residual_single").sum()) if not hh_df.empty else 0,
            "shell_meta": shell_meta,
        }

    households = pd.DataFrame.from_records(household_rows)
    if households.empty:
        households = pd.DataFrame(
            columns=[str(group_col), "household_id", "household_type", "household_size", "n_children", "n_adults", "mean_age_proxy", "household_income_score", "source_stage", "HHINCP_bin"]
        )

    out["household_size"] = pd.to_numeric(out["household_size"], errors="coerce").astype("Int64")
    meta = {
        "group_col": str(group_col),
        "person_id_col": str(person_id_col),
        "age_col": str(age_col),
        "earn_col": str(earn_col),
        "n_persons": int(out.shape[0]),
        "n_households": int(households.shape[0]),
        "n_groups": int(out[str(group_col)].nunique()),
        "persons_with_household_id": int(out["household_id"].notna().sum()),
        "household_size_mean": float(pd.to_numeric(households.get("household_size"), errors="coerce").fillna(0).mean() if not households.empty else 0.0),
        "n_income_labeled_households": int(households.get("HHINCP_bin", pd.Series(dtype=object)).notna().sum()) if not households.empty else 0,
        "groups": meta_groups,
    }
    return out, households, meta
