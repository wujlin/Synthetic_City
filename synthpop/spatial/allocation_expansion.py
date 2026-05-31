from __future__ import annotations

"""
Bridge utilities from type-count allocation to individual synthetic persons.

Design goal:
- take phase2 `type_assignment_long` outputs
- integerize floating type-by-group allocation without breaking matrix structure
- expand the integer allocation into person-level records for downstream spatial assignment
"""

from typing import Any


def _require_numpy_pandas() -> tuple[Any, Any]:
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("allocation_expansion requires numpy and pandas.") from e
    return np, pd


def _largest_remainder(vec: Any, *, total: int) -> Any:
    np, _ = _require_numpy_pandas()
    arr = np.asarray(vec, dtype=float)
    total = int(total)
    if total < 0:
        raise ValueError(f"total must be non-negative, got: {total}")
    base = np.floor(arr).astype(int)
    rem = int(total - int(base.sum()))
    if rem < 0:
        order = np.argsort(arr - base)
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


def _canon_geoid_for_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits and len(digits) <= 11 and text.replace(".", "", 1).isdigit():
        return digits.zfill(11)
    return text


class _MaxFlow:
    def __init__(self, n: int) -> None:
        self.n = int(n)
        self.to: list[int] = []
        self.cap: list[int] = []
        self.rev: list[int] = []
        self.adj: list[list[int]] = [[] for _ in range(int(n))]

    def add_edge(self, u: int, v: int, c: int) -> None:
        i = len(self.to)
        self.to.append(int(v))
        self.cap.append(int(c))
        self.rev.append(i + 1)
        self.adj[int(u)].append(i)

        j = len(self.to)
        self.to.append(int(u))
        self.cap.append(0)
        self.rev.append(i)
        self.adj[int(v)].append(j)

    def max_flow(self, s: int, t: int) -> int:
        from collections import deque

        flow = 0
        while True:
            parent = [-1] * self.n
            parent_edge = [-1] * self.n
            q = deque([int(s)])
            parent[int(s)] = int(s)
            while q and parent[int(t)] == -1:
                u = q.popleft()
                for ei in self.adj[u]:
                    if self.cap[ei] <= 0:
                        continue
                    v = self.to[ei]
                    if parent[v] != -1:
                        continue
                    parent[v] = u
                    parent_edge[v] = ei
                    q.append(v)
            if parent[int(t)] == -1:
                break
            aug = 10**9
            v = int(t)
            while v != int(s):
                ei = parent_edge[v]
                aug = min(aug, self.cap[ei])
                v = parent[v]
            v = int(t)
            while v != int(s):
                ei = parent_edge[v]
                self.cap[ei] -= aug
                self.cap[self.rev[ei]] += aug
                v = parent[v]
            flow += aug
        return int(flow)


def _integerize_2d(*, x: Any, row_targets: Any, col_targets: Any) -> Any:
    np, _ = _require_numpy_pandas()
    x = np.asarray(x, dtype=float)
    row = np.asarray(row_targets, dtype=int)
    col = np.asarray(col_targets, dtype=int)
    if x.shape != (row.size, col.size):
        raise ValueError("shape mismatch in integerize")
    if int(row.sum()) != int(col.sum()):
        raise ValueError("row/col totals mismatch")

    base = np.floor(x).astype(int)
    base = np.clip(base, 0, None)
    row_res = row - base.sum(axis=1)
    col_res = col - base.sum(axis=0)
    if (row_res < 0).any() or (col_res < 0).any():
        raise RuntimeError("Negative residuals in integerize.")
    need = int(row_res.sum())
    if need == 0:
        return base

    frac = x - np.floor(x)
    n_row, n_col = base.shape
    s = 0
    row0 = 1
    col0 = row0 + n_row
    t = col0 + n_col
    g = _MaxFlow(t + 1)
    for i in range(n_row):
        g.add_edge(s, row0 + i, int(row_res[i]))
    for j in range(n_col):
        g.add_edge(col0 + j, t, int(col_res[j]))
    for i in range(n_row):
        cols = list(range(n_col))
        cols.sort(key=lambda j: float(frac[i, j]), reverse=True)
        for j in cols:
            if frac[i, j] > 0.0:
                g.add_edge(row0 + i, col0 + j, 1)

    got = g.max_flow(s, t)
    if got != need:
        g = _MaxFlow(t + 1)
        for i in range(n_row):
            g.add_edge(s, row0 + i, int(row_res[i]))
        for j in range(n_col):
            g.add_edge(col0 + j, t, int(col_res[j]))
        for i in range(n_row):
            for j in range(n_col):
                g.add_edge(row0 + i, col0 + j, 1)
        got = g.max_flow(s, t)
        if got != need:
            raise RuntimeError(f"integerize failed: need={need} got={got}")

    for i in range(n_row):
        u = row0 + i
        for ei in g.adj[u]:
            v = g.to[ei]
            if v < col0 or v >= col0 + n_col:
                continue
            rev_ei = g.rev[ei]
            if g.cap[rev_ei] > 0:
                j = v - col0
                base[i, j] += 1

    if not (base.sum(axis=1) == row).all():
        raise RuntimeError("row sums mismatch after integerize")
    if not (base.sum(axis=0) == col).all():
        raise RuntimeError("col sums mismatch after integerize")
    return base


def integerize_type_allocation_long(
    *,
    allocation_long: Any,
    region_col: str = "puma_uid",
    group_col: str = "tract_geoid",
    type_idx_col: str = "type_idx",
    count_col: str = "count",
    out_count_col: str = "count_int",
) -> tuple[Any, dict[str, Any]]:
    np, pd = _require_numpy_pandas()
    if not isinstance(allocation_long, pd.DataFrame):
        raise TypeError("allocation_long must be a pandas DataFrame")
    need = [str(region_col), str(group_col), str(type_idx_col), str(count_col)]
    miss = [c for c in need if c not in allocation_long.columns]
    if miss:
        raise ValueError(f"allocation_long missing columns: {miss}")

    df = allocation_long.copy()
    df[str(region_col)] = df[str(region_col)].astype(str)
    df[str(group_col)] = df[str(group_col)].astype(str)
    df[str(type_idx_col)] = pd.to_numeric(df[str(type_idx_col)], errors="raise").astype(int)
    df[str(count_col)] = pd.to_numeric(df[str(count_col)], errors="coerce").fillna(0.0).clip(lower=0.0)

    key_cols = [str(region_col), str(group_col), str(type_idx_col)]
    extra_cols = [c for c in df.columns if c not in key_cols + [str(count_col)]]
    base_rows = df[key_cols + extra_cols].drop_duplicates(subset=key_cols, keep="first").copy()

    blocks: list[Any] = []
    meta_regions: dict[str, Any] = {}
    total_before = float(df[str(count_col)].sum())
    total_after = 0

    for region_id, sub in df.groupby(str(region_col), sort=False):
        types = sorted(sub[str(type_idx_col)].unique().tolist())
        groups = sorted(sub[str(group_col)].unique().tolist())
        type_to_i = {int(t): i for i, t in enumerate(types)}
        group_to_j = {str(g): j for j, g in enumerate(groups)}
        x = np.zeros((len(types), len(groups)), dtype=float)
        for row in sub[[str(type_idx_col), str(group_col), str(count_col)]].itertuples(index=False):
            x[type_to_i[int(getattr(row, str(type_idx_col)))], group_to_j[str(getattr(row, str(group_col)))]] = float(
                getattr(row, str(count_col))
            )

        total_int = int(round(float(x.sum())))
        row_targets = _largest_remainder(x.sum(axis=1), total=total_int)
        col_targets = _largest_remainder(x.sum(axis=0), total=total_int)
        x_int = _integerize_2d(x=x, row_targets=row_targets, col_targets=col_targets)
        total_after += int(x_int.sum())

        rows: list[dict[str, Any]] = []
        nz_i, nz_j = np.where(x_int > 0)
        for i, j in zip(nz_i.tolist(), nz_j.tolist()):
            rows.append(
                {
                    str(region_col): str(region_id),
                    str(group_col): str(groups[j]),
                    str(type_idx_col): int(types[i]),
                    str(out_count_col): int(x_int[i, j]),
                }
            )
        block = pd.DataFrame(rows)
        if not block.empty:
            block = block.merge(base_rows, on=key_cols, how="left")
            ordered = key_cols + extra_cols + [str(out_count_col)]
            block = block[ordered]
        blocks.append(block)

        meta_regions[str(region_id)] = {
            "n_types": int(len(types)),
            "n_groups": int(len(groups)),
            "total_before": float(x.sum()),
            "total_after": int(x_int.sum()),
            "max_cell_abs_rounding_err": float(np.max(np.abs(x_int - x))) if x.size else 0.0,
        }

    if not blocks:
        cols = key_cols + extra_cols + [str(out_count_col)]
        return pd.DataFrame(columns=cols), {
            "n_regions": 0,
            "total_before": total_before,
            "total_after": 0,
            "regions": {},
        }

    out = pd.concat(blocks, axis=0, ignore_index=True)
    out[str(out_count_col)] = pd.to_numeric(out[str(out_count_col)], errors="raise").astype(int)
    meta = {
        "n_regions": int(len(meta_regions)),
        "total_before": float(total_before),
        "total_after": int(total_after),
        "regions": meta_regions,
    }
    return out, meta


def expand_integer_allocation_to_persons(
    *,
    integer_allocation_long: Any,
    count_col: str = "count_int",
    person_id_col: str = "person_id",
    person_id_prefix: str = "synp",
    esr_col: str = "ESR_allpop",
    employed_values: tuple[str, ...] = ("employed",),
) -> tuple[Any, dict[str, Any]]:
    _, pd = _require_numpy_pandas()
    if not isinstance(integer_allocation_long, pd.DataFrame):
        raise TypeError("integer_allocation_long must be a pandas DataFrame")
    if str(count_col) not in integer_allocation_long.columns:
        raise ValueError(f"integer_allocation_long missing column: {count_col}")

    df = integer_allocation_long.copy()
    df[str(count_col)] = pd.to_numeric(df[str(count_col)], errors="raise").astype(int)
    df = df[df[str(count_col)] > 0].copy().reset_index(drop=True)
    if df.empty:
        out = df.drop(columns=[str(count_col)], errors="ignore").copy()
        out[str(person_id_col)] = pd.Series(dtype=str)
        return out, {"n_persons": 0, "worker_rate": 0.0}

    repeated = df.loc[df.index.repeat(df[str(count_col)])].copy().reset_index(drop=True)
    geo_col = "tract_geoid" if "tract_geoid" in repeated.columns else None
    if geo_col is not None:
        repeated[geo_col] = repeated[geo_col].map(_canon_geoid_for_id)
        repeated["_person_seq"] = repeated.groupby(geo_col, sort=False).cumcount() + 1
        seq = repeated["_person_seq"].astype(int).astype(str).str.zfill(6)
        repeated[str(person_id_col)] = str(person_id_prefix) + "_" + repeated[geo_col].astype(str) + "_" + seq
        person_id_scheme = f"{person_id_prefix}_{{tract_geoid}}_{{seq_within_tract:06d}}"
    else:
        repeated["_person_seq"] = range(int(repeated.shape[0]))
        repeated[str(person_id_col)] = repeated["_person_seq"].map(lambda i: f"{str(person_id_prefix)}_{int(i):010d}")
        person_id_scheme = f"{person_id_prefix}_{{global_seq:010d}}"
    repeated = repeated.drop(columns=["_person_seq", str(count_col)])

    if str(esr_col) in repeated.columns:
        employed = {str(v).strip().lower() for v in employed_values}
        repeated["is_worker"] = repeated[str(esr_col)].astype(str).str.strip().str.lower().isin(employed)

    cols = [str(person_id_col)] + [c for c in repeated.columns if c != str(person_id_col)]
    repeated = repeated[cols].reset_index(drop=True)
    meta = {
        "n_persons": int(repeated.shape[0]),
        "n_unique_person_ids": int(repeated[str(person_id_col)].nunique()),
        "person_id_scheme": str(person_id_scheme),
        "person_id_geo_col": geo_col,
        "worker_rate": (
            float(repeated["is_worker"].mean()) if "is_worker" in repeated.columns and len(repeated) > 0 else 0.0
        ),
    }
    return repeated, meta
