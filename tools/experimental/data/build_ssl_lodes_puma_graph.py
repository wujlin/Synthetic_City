#!/usr/bin/env python3
from __future__ import annotations

"""Build PUMA-level functional-neighborhood graphs from LODES OD files.

The graph is intended for representation learning, not for direct validation.
LODES OD defines a functional relationship: two regions are close if commute
flows connect them, even when they are not geographically adjacent.
"""

import argparse
import datetime as dt
import gzip
import json
import pathlib
from collections.abc import Iterable

import numpy as np
import pandas as pd


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_csv(path: pathlib.Path, **kwargs) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f, **kwargs)
    return pd.read_csv(path, **kwargs)


def _digits(v: object) -> str:
    return "".join(ch for ch in str(v).strip() if ch.isdigit())


def _canon_statefp(v: object) -> str:
    d = _digits(v)
    return str(int(d)).zfill(2) if d else ""


def _canon_puma5(v: object) -> str:
    d = _digits(v)
    return str(int(d)).zfill(5) if d else ""


def _canon_tract(v: object) -> str:
    d = _digits(v)
    if len(d) < 11:
        return ""
    return d[:11]


def _puma_uid(statefp: object, puma: object) -> str:
    s = _canon_statefp(statefp)
    p = _canon_puma5(puma)
    return s + p if s and p else ""


def _load_mapping(path: pathlib.Path, default_statefp: str | None) -> pd.DataFrame:
    m = pd.read_csv(path, dtype=str)
    if "tract_geoid" not in m.columns or "puma" not in m.columns:
        raise SystemExit(f"mapping must contain tract_geoid,puma columns: {path}")
    m = m.loc[:, ["tract_geoid", "puma"] + (["statefp"] if "statefp" in m.columns else [])].copy()
    m["tract_geoid"] = m["tract_geoid"].map(_canon_tract)
    if "statefp" not in m.columns:
        if not default_statefp:
            raise SystemExit("--statefp is required when mapping has no statefp column")
        m["statefp"] = _canon_statefp(default_statefp)
    else:
        m["statefp"] = m["statefp"].map(_canon_statefp)
    m["puma5"] = m["puma"].map(_canon_puma5)
    m["puma_uid"] = m.apply(lambda r: _puma_uid(r["statefp"], r["puma5"]), axis=1)
    m = m[(m["tract_geoid"] != "") & (m["puma_uid"] != "")].copy()
    return m.drop_duplicates("tract_geoid", keep="first")


def _load_lodes(paths: Iterable[pathlib.Path], count_col: str, chunksize: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    usecols = ["w_geocode", "h_geocode", count_col]
    for path in paths:
        if not path.exists():
            raise SystemExit(f"LODES OD file not found: {path}")
        reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype={"w_geocode": str, "h_geocode": str})
        for chunk in reader:
            chunk[count_col] = pd.to_numeric(chunk[count_col], errors="coerce").fillna(0.0)
            chunk = chunk[chunk[count_col] > 0].copy()
            if chunk.empty:
                continue
            chunk["home_tract_geoid"] = chunk["h_geocode"].map(_canon_tract)
            chunk["work_tract_geoid"] = chunk["w_geocode"].map(_canon_tract)
            chunk = chunk[(chunk["home_tract_geoid"] != "") & (chunk["work_tract_geoid"] != "")]
            frames.append(
                chunk.groupby(["home_tract_geoid", "work_tract_geoid"], as_index=False, sort=False)[count_col].sum()
            )
    if not frames:
        return pd.DataFrame(columns=["home_tract_geoid", "work_tract_geoid", count_col])
    out = pd.concat(frames, ignore_index=True)
    return out.groupby(["home_tract_geoid", "work_tract_geoid"], as_index=False, sort=False)[count_col].sum()


def _topk(df: pd.DataFrame, group_col: str, weight_col: str, k: int) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby(group_col, sort=False):
        parts.append(g.sort_values(weight_col, ascending=False).head(k))
    if not parts:
        return df.head(0).copy()
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_ssl_lodes_puma_graph")
    ap.add_argument("--lodes_od", nargs="+", required=True, help="LODES OD csv/csv.gz files.")
    ap.add_argument("--tract_puma_mapping", required=True, help="CSV with tract_geoid,puma[,statefp].")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--statefp", default="", help="Default state FIPS when mapping lacks statefp.")
    ap.add_argument("--count_col", default="S000")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--min_count", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = pathlib.Path(args.tract_puma_mapping).expanduser().resolve()
    lodes_paths = [pathlib.Path(p).expanduser().resolve() for p in args.lodes_od]

    mapping = _load_mapping(mapping_path, default_statefp=args.statefp or None)
    lodes = _load_lodes(lodes_paths, count_col=args.count_col, chunksize=int(args.chunksize))
    if lodes.empty:
        raise SystemExit("No positive LODES OD rows loaded.")

    home_map = mapping.loc[:, ["tract_geoid", "puma_uid"]].rename(
        columns={"tract_geoid": "home_tract_geoid", "puma_uid": "home_puma_uid"}
    )
    work_map = mapping.loc[:, ["tract_geoid", "puma_uid"]].rename(
        columns={"tract_geoid": "work_tract_geoid", "puma_uid": "work_puma_uid"}
    )
    od = lodes.merge(home_map, on="home_tract_geoid", how="inner").merge(work_map, on="work_tract_geoid", how="inner")
    od = od.groupby(["home_puma_uid", "work_puma_uid"], as_index=False, sort=False)[args.count_col].sum()
    od = od.rename(columns={args.count_col: "od_count"})
    od = od[od["od_count"] >= float(args.min_count)].copy()
    if od.empty:
        raise SystemExit("No PUMA OD rows remain after mapping and min_count filter.")

    origin_total = od.groupby("home_puma_uid", sort=False)["od_count"].transform("sum")
    dest_total = od.groupby("work_puma_uid", sort=False)["od_count"].transform("sum")
    od["origin_share"] = od["od_count"] / np.clip(origin_total, 1e-12, None)
    od["destination_share"] = od["od_count"] / np.clip(dest_total, 1e-12, None)
    od["log1p_count"] = np.log1p(od["od_count"].to_numpy(float))

    directed = od.sort_values(["home_puma_uid", "origin_share"], ascending=[True, False]).reset_index(drop=True)

    a = od.loc[:, ["home_puma_uid", "work_puma_uid", "od_count", "origin_share"]].copy()
    b = a.rename(
        columns={
            "home_puma_uid": "work_puma_uid",
            "work_puma_uid": "home_puma_uid",
            "od_count": "reverse_od_count",
            "origin_share": "reverse_origin_share",
        }
    )
    sym = a.merge(b, on=["home_puma_uid", "work_puma_uid"], how="outer").fillna(0.0)
    sym = sym[sym["home_puma_uid"] != sym["work_puma_uid"]].copy()
    sym["sym_count"] = sym["od_count"] + sym["reverse_od_count"]
    sym["sym_share"] = 0.5 * (sym["origin_share"] + sym["reverse_origin_share"])
    sym = sym[sym["sym_count"] >= float(args.min_count)].copy()
    sym = sym.sort_values(["home_puma_uid", "sym_share"], ascending=[True, False]).reset_index(drop=True)
    top_sym = _topk(sym, "home_puma_uid", "sym_share", int(args.top_k)).reset_index(drop=True)

    directed_path = out_dir / "puma_lodes_directed_od_edges.csv"
    sym_path = out_dir / "puma_lodes_symmetric_functional_edges.csv"
    top_path = out_dir / "puma_lodes_top_functional_neighbors.csv"
    directed.to_csv(directed_path, index=False)
    sym.to_csv(sym_path, index=False)
    top_sym.to_csv(top_path, index=False)

    summary = {
        "created_utc": _utc_now(),
        "inputs": {
            "lodes_od": [str(p) for p in lodes_paths],
            "tract_puma_mapping": str(mapping_path),
        },
        "params": {
            "count_col": args.count_col,
            "chunksize": int(args.chunksize),
            "top_k": int(args.top_k),
            "min_count": float(args.min_count),
        },
        "n_mapping_tracts": int(mapping["tract_geoid"].nunique()),
        "n_mapping_pumas": int(mapping["puma_uid"].nunique()),
        "n_tract_od_positive": int(lodes.shape[0]),
        "n_puma_od_edges": int(directed.shape[0]),
        "n_home_pumas": int(directed["home_puma_uid"].nunique()),
        "n_work_pumas": int(directed["work_puma_uid"].nunique()),
        "total_od_count_mapped": float(directed["od_count"].sum()),
        "n_symmetric_edges": int(sym.shape[0]),
        "n_top_functional_edges": int(top_sym.shape[0]),
        "outputs": {
            "directed_od_edges": str(directed_path),
            "symmetric_functional_edges": str(sym_path),
            "top_functional_neighbors": str(top_path),
        },
        "interpretation": {
            "directed_od_edges": "home PUMA -> work PUMA commute-flow graph",
            "symmetric_functional_edges": "functional neighborhood graph for contrastive positive pairs",
            "not_independent_validation": "If LODES is used for representation training, it should not also be described as independent OD validation for the same claim.",
        },
    }
    (out_dir / "lodes_puma_graph_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
