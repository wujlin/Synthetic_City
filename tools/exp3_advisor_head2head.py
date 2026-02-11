#!/usr/bin/env python3
from __future__ import annotations

"""
Exp3 supplement: advisor synthpop head-to-head on BG age×sex.

Goal:
- Compare Exp1 base population against advisor synthetic population on a common,
  auditable target: BG-level age×sex distribution.

Output:
  outputs/<run_id>/advisor_bg_age_sex_comparison.json
"""

import argparse
import datetime as _dt
import json
import os
import pathlib
import sys
import tempfile
import zipfile
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Missing dependency: {pkg}. Install it in your conda env.") from e


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _age_to_p12_idx(age: int) -> int:
    if age < 0:
        age = 0
    if age <= 4:
        return 0
    if age <= 9:
        return 1
    if age <= 14:
        return 2
    if age <= 17:
        return 3
    if age <= 19:
        return 4
    if age == 20:
        return 5
    if age == 21:
        return 6
    if age <= 24:
        return 7
    if age <= 29:
        return 8
    if age <= 34:
        return 9
    if age <= 39:
        return 10
    if age <= 44:
        return 11
    if age <= 49:
        return 12
    if age <= 54:
        return 13
    if age <= 59:
        return 14
    if age <= 61:
        return 15
    if age <= 64:
        return 16
    if age <= 66:
        return 17
    if age <= 69:
        return 18
    if age <= 74:
        return 19
    if age <= 79:
        return 20
    if age <= 84:
        return 21
    return 22


def _tvd(p: list[float], q: list[float]) -> float:
    np = _require("numpy")
    a = np.asarray(p, dtype=float)
    b = np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(a - b).sum())


def _infer_col(df: Any, candidates: list[str]) -> str | None:
    cols = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _clean_bg_geoid(series: Any) -> Any:
    s = series.astype(str).str.replace(r"[^0-9]", "", regex=True)
    # Keep last 12 digits (state+county+tract+bg)
    return s.str[-12:]


def _discover_gpkg_layers(path: pathlib.Path) -> list[str]:
    # Try geopandas first (works with pyogrio engine), then fiona fallback.
    try:
        gpd = _require("geopandas")
        if hasattr(gpd, "list_layers"):
            layers_df = gpd.list_layers(path)  # type: ignore[attr-defined]
            if "name" in layers_df.columns:
                names = [str(x) for x in layers_df["name"].tolist() if str(x)]
                if names:
                    return names
    except Exception:
        pass
    try:
        import fiona  # type: ignore

        return [str(x) for x in fiona.listlayers(path)]
    except Exception:
        return []


def _load_advisor_df(*, advisor_zip: pathlib.Path, advisor_member: str | None, advisor_layer: str | None) -> Any:
    pd = _require("pandas")
    gpd = None
    try:
        gpd = _require("geopandas")
    except Exception:
        gpd = None

    with zipfile.ZipFile(advisor_zip) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/") and not n.startswith("__MACOSX/")]
        if advisor_member:
            if advisor_member not in members:
                raise SystemExit(f"advisor_member not found in zip: {advisor_member}")
            target = advisor_member
        else:
            # Prefer geopackage/parquet/csv in this order.
            def _score(name: str) -> tuple[int, int, int]:
                lo = name.lower()
                if lo.endswith(".gpkg"):
                    # Prefer person/population tables over workplace/network tables.
                    if "population" in lo or lo.endswith("mi_population.gpkg"):
                        return (0, 0, len(name))
                    if "workplace" in lo or "network" in lo:
                        return (0, 2, len(name))
                    return (0, 1, len(name))
                if lo.endswith(".parquet"):
                    return (1, 0, len(name))
                if lo.endswith(".csv"):
                    return (2, 0, len(name))
                return (9, 0, len(name))

            members_sorted = sorted(members, key=_score)
            target = members_sorted[0] if members_sorted else None
            if target is None:
                raise SystemExit("advisor zip has no readable members.")
            print(f"[info] advisor member auto-selected: {target}")

        suffix = pathlib.Path(target).suffix.lower()
        with tempfile.TemporaryDirectory(prefix="advisor_") as td:
            td_path = pathlib.Path(td)
            extracted = pathlib.Path(zf.extract(target, path=td_path))
            if suffix == ".gpkg":
                if gpd is None:
                    raise SystemExit("advisor file is .gpkg but geopandas is not installed.")
                # Handle multi-layer gpkg: pick the layer that contains person-level age/sex fields.
                chosen_layer = advisor_layer
                if chosen_layer is None:
                    layers = _discover_gpkg_layers(extracted)
                    if layers:
                        print(f"[info] advisor gpkg layers: {layers}")
                    if layers:
                        age_cands = {"age", "agep", "age_years", "person_age"}
                        sex_cands = {"sex", "gender", "sex_id", "person_sex"}
                        for lyr in layers:
                            try:
                                probe = gpd.read_file(extracted, layer=lyr, rows=50)
                            except Exception:
                                continue
                            cols_l = {str(c).lower() for c in probe.columns}
                            if cols_l & age_cands and cols_l & sex_cands:
                                chosen_layer = lyr
                                break
                        # fallback: keep first layer if no age/sex match found
                        if chosen_layer is None:
                            chosen_layer = layers[0]
                    if chosen_layer is not None:
                        print(f"[info] advisor gpkg chosen layer: {chosen_layer}")
                try:
                    if chosen_layer is None:
                        return gpd.read_file(extracted)
                    return gpd.read_file(extracted, layer=chosen_layer)
                except Exception as e:
                    raise SystemExit(f"Failed to read advisor gpkg layer={chosen_layer}: {e}") from e
            if suffix == ".parquet":
                return pd.read_parquet(extracted)
            if suffix == ".csv":
                return pd.read_csv(extracted, low_memory=False)
            raise SystemExit(f"Unsupported advisor member type: {target}")


def _auto_find_tiger_bg_zip(*, data_root: pathlib.Path, statefp: str) -> pathlib.Path | None:
    statefp = str(statefp).zfill(2)
    patterns = [
        f"detroit/raw/geo/tiger/**/tl_*_{statefp}_bg.zip",
        f"detroit/raw/census/tiger/**/tl_*_{statefp}_bg.zip",
        f"tl_*_{statefp}_bg.zip",
    ]
    cands: list[pathlib.Path] = []
    for pat in patterns:
        cands.extend([p for p in data_root.glob(pat) if p.is_file()])
    if not cands:
        return None

    # Prefer higher TIGER year if present in filename, then longer path depth.
    def _score(p: pathlib.Path) -> tuple[int, int]:
        name = p.name
        year = -1
        try:
            parts = name.split("_")
            if len(parts) >= 2:
                year = int(parts[1])
        except Exception:
            year = -1
        return (year, len(str(p)))

    cands = sorted(cands, key=_score, reverse=True)
    return cands[0]


def _ensure_bg_geoid(
    *,
    advisor_df: Any,
    tiger_bg_zip: pathlib.Path | None,
) -> Any:
    pd = _require("pandas")
    np = _require("numpy")
    try:
        gpd = _require("geopandas")
    except Exception:
        gpd = None

    d = advisor_df.copy()

    def _with_bg_geoid(bg_df: Any) -> Any:
        out_bg = bg_df.copy()
        if "GEOID" in out_bg.columns:
            out_bg["bg_geoid"] = out_bg["GEOID"].astype(str)
        else:
            req = ["STATEFP", "COUNTYFP", "TRACTCE", "BLKGRPCE"]
            miss = [c for c in req if c not in out_bg.columns]
            if miss:
                raise SystemExit(f"TIGER BG missing columns: {miss}")
            out_bg["bg_geoid"] = (
                out_bg["STATEFP"].astype(str).str.zfill(2)
                + out_bg["COUNTYFP"].astype(str).str.zfill(3)
                + out_bg["TRACTCE"].astype(str).str.zfill(6)
                + out_bg["BLKGRPCE"].astype(str).str.zfill(1)
            )
        return out_bg

    def _spatial_join_bg(points_like: Any, bg_zip: pathlib.Path) -> Any:
        if gpd is None:
            raise SystemExit("Spatial join requires geopandas.")
        bg = _with_bg_geoid(gpd.read_file(f"zip://{pathlib.Path(bg_zip).expanduser().resolve()}"))
        pts = points_like.copy()
        if getattr(pts, "crs", None) is None:
            # Conservative default for external synthetic data exports.
            pts = pts.set_crs("EPSG:4326", allow_override=True)
        if pts.crs != bg.crs:
            pts = pts.to_crs(bg.crs)

        # If advisor geometry is polygon/line, convert to interior points.
        geom_type = pts.geometry.geom_type.astype(str).str.lower()
        if geom_type.isin({"polygon", "multipolygon", "linestring", "multilinestring"}).any():
            pts = pts.copy()
            pts["geometry"] = pts.geometry.representative_point()

        joined = gpd.sjoin(pts, bg[["bg_geoid", "geometry"]], how="left", predicate="within")
        out = pd.DataFrame(joined.drop(columns=["geometry"], errors="ignore"))
        out = out[out["bg_geoid"].notna()].copy()
        out["bg_geoid"] = out["bg_geoid"].astype(str)
        return out

    bg_col = _infer_col(d, ["bg_geoid", "block_group", "blockgroup", "geoid_bg", "bgid", "bgid20", "geoid"])
    if bg_col is not None:
        d["bg_geoid"] = _clean_bg_geoid(d[bg_col])
        d = d[d["bg_geoid"].str.len() == 12].copy()
        return d

    # Prefer geometry if present (common in advisor gpkg exports).
    if gpd is not None and "geometry" in d.columns:
        if tiger_bg_zip is None:
            raise SystemExit("advisor data lacks bg_geoid; please pass --tiger_bg_zip for spatial join.")
        try:
            gdf = d if isinstance(d, gpd.GeoDataFrame) else gpd.GeoDataFrame(d, geometry="geometry")
            gdf = gdf[gdf.geometry.notna()].copy()
            if not gdf.empty:
                return _spatial_join_bg(gdf, tiger_bg_zip)
        except Exception:
            # Fall through to lon/lat branch if geometry branch fails.
            pass

    lon_col = _infer_col(d, ["lon", "lng", "longitude", "x"])
    lat_col = _infer_col(d, ["lat", "latitude", "y"])
    if lon_col is None or lat_col is None:
        raise SystemExit("advisor data missing bg_geoid and missing lon/lat columns.")
    if tiger_bg_zip is None:
        raise SystemExit("advisor data lacks bg_geoid; please pass --tiger_bg_zip for spatial join.")
    if gpd is None:
        raise SystemExit("Spatial join requires geopandas.")

    d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d = d[d[lon_col].notna() & d[lat_col].notna()].copy()
    if d.empty:
        raise SystemExit("advisor data has no valid lon/lat rows.")

    pts = gpd.GeoDataFrame(
        d,
        geometry=gpd.points_from_xy(d[lon_col].to_numpy(dtype=float), d[lat_col].to_numpy(dtype=float)),
        crs="EPSG:4326",
    )
    return _spatial_join_bg(pts, tiger_bg_zip)


def _normalize_sex(series: Any) -> Any:
    pd = _require("pandas")
    s = series.copy()
    s_num = pd.to_numeric(s, errors="coerce")
    out = pd.Series([None] * int(len(s)), index=s.index, dtype=object)
    out.loc[s_num == 1] = "1"
    out.loc[s_num == 2] = "2"
    # String fallback.
    s_str = s.astype(str).str.strip().str.lower()
    out.loc[s_str.isin(["m", "male", "man", "1"])] = "1"
    out.loc[s_str.isin(["f", "female", "woman", "2"])] = "2"
    return out


def main() -> None:
    pd = _require("pandas")
    np = _require("numpy")
    from src.synthpop.pipeline.detroit_v0 import make_run_id
    from src.synthpop.paths import data_root as default_data_root

    ap = argparse.ArgumentParser(prog="exp3_advisor_head2head")
    ap.add_argument("--exp1_counts_path", required=True, help="Exp1 counts table with bg_geoid, age_idx, sex, count.")
    ap.add_argument("--advisor_zip", required=True, help="Advisor synthpop zip, e.g., reference/advisor_synthpop/mi.zip")
    ap.add_argument("--advisor_member", default=None, help="Optional member path inside advisor zip.")
    ap.add_argument("--advisor_layer", default=None, help="Optional layer name for advisor gpkg (if multi-layer).")
    ap.add_argument("--tiger_bg_zip", default=None, help="Optional TIGER BG zip if advisor has only lon/lat.")
    ap.add_argument("--data_root", default=str(default_data_root()))
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--out_dir", default=None, help="Default: outputs/<run_id> under repo.")
    args = ap.parse_args()

    out_dir = (
        pathlib.Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (_REPO_ROOT / "outputs" / make_run_id(prefix="exp3_advisor_head2head"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    exp1_path = pathlib.Path(args.exp1_counts_path).expanduser().resolve()
    if not exp1_path.exists():
        raise SystemExit(f"exp1_counts_path not found: {exp1_path}")
    if exp1_path.suffix.lower() == ".parquet":
        exp1 = pd.read_parquet(exp1_path)
    else:
        exp1 = pd.read_csv(exp1_path, low_memory=False)
    req = ["bg_geoid", "age_idx", "sex", "count"]
    miss = [c for c in req if c not in exp1.columns]
    if miss:
        raise SystemExit(f"exp1_counts missing required columns: {miss}")
    exp1["bg_geoid"] = exp1["bg_geoid"].astype(str)
    exp1["age_idx"] = pd.to_numeric(exp1["age_idx"], errors="coerce").fillna(0).astype(int).clip(lower=0, upper=22)
    exp1["sex"] = pd.to_numeric(exp1["sex"], errors="coerce").fillna(1).astype(int).clip(lower=1, upper=2).astype(str)
    exp1["count"] = pd.to_numeric(exp1["count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    exp1 = exp1[exp1["count"] > 0].copy()

    advisor_zip = pathlib.Path(args.advisor_zip).expanduser().resolve()
    if not advisor_zip.exists():
        raise SystemExit(f"advisor_zip not found: {advisor_zip}")
    adv = _load_advisor_df(advisor_zip=advisor_zip, advisor_member=args.advisor_member, advisor_layer=args.advisor_layer)
    tiger_bg_zip = pathlib.Path(args.tiger_bg_zip).expanduser().resolve() if args.tiger_bg_zip else None
    if tiger_bg_zip is None:
        data_root = pathlib.Path(args.data_root).expanduser().resolve()
        tiger_bg_zip = _auto_find_tiger_bg_zip(data_root=data_root, statefp=str(args.statefp))
    adv = _ensure_bg_geoid(advisor_df=adv, tiger_bg_zip=tiger_bg_zip)

    age_col = _infer_col(adv, ["age", "AGEP", "Age", "AGE"])
    sex_col = _infer_col(adv, ["sex", "SEX", "gender", "Gender"])
    if age_col is None or sex_col is None:
        raise SystemExit(
            "advisor data missing required age/sex columns. "
            f"Detected columns={list(adv.columns)}; use --advisor_member/--advisor_layer to pick person-level table."
        )
    adv["age_num"] = pd.to_numeric(adv[age_col], errors="coerce")
    adv["sex_std"] = _normalize_sex(adv[sex_col])
    adv = adv[adv["age_num"].notna() & adv["sex_std"].notna()].copy()
    if adv.empty:
        raise SystemExit("advisor data has no valid rows after age/sex normalization.")
    adv["age_idx"] = adv["age_num"].astype(int).map(_age_to_p12_idx).astype(int).clip(lower=0, upper=22)
    adv["bg_geoid"] = adv["bg_geoid"].astype(str)

    # Aggregate distributions by BG.
    exp_bg = exp1.groupby(["bg_geoid", "sex", "age_idx"], as_index=False)["count"].sum()
    adv_bg = adv.groupby(["bg_geoid", "sex_std", "age_idx"], as_index=False).size().rename(columns={"size": "count"})
    adv_bg = adv_bg.rename(columns={"sex_std": "sex"})

    exp_groups = set(exp_bg["bg_geoid"].unique().tolist())
    adv_groups = set(adv_bg["bg_geoid"].unique().tolist())
    overlap = sorted(exp_groups & adv_groups)
    if not overlap:
        raise SystemExit("No overlapping BG GEOIDs between Exp1 and advisor data.")

    by_bg: dict[str, float] = {}
    for bg in overlap:
        e = exp_bg[exp_bg["bg_geoid"] == bg]
        a = adv_bg[adv_bg["bg_geoid"] == bg]
        cells = sorted(set(zip(e["sex"], e["age_idx"])) | set(zip(a["sex"], a["age_idx"])))
        e_sum = float(e["count"].sum())
        a_sum = float(a["count"].sum())
        if e_sum <= 0 or a_sum <= 0:
            continue
        p = []
        q = []
        for sx, ag in cells:
            p.append(float(e.loc[(e["sex"] == sx) & (e["age_idx"] == ag), "count"].sum()) / e_sum)
            q.append(float(a.loc[(a["sex"] == sx) & (a["age_idx"] == ag), "count"].sum()) / a_sum)
        by_bg[str(bg)] = _tvd(p, q)

    vals = np.asarray(list(by_bg.values()), dtype=float) if by_bg else np.asarray([], dtype=float)
    if vals.size == 0:
        raise SystemExit("No valid BG-level TVD computed.")
    sorted_bg = sorted(by_bg.items(), key=lambda kv: kv[1], reverse=True)

    summary = {
        "mean_tvd_bg_age_sex": float(vals.mean()),
        "p50_tvd_bg_age_sex": float(np.quantile(vals, 0.5)),
        "p90_tvd_bg_age_sex": float(np.quantile(vals, 0.9)),
        "max_tvd_bg_age_sex": float(vals.max()),
        "n_bg_overlap": int(vals.size),
        "worst_bg_top10": [{"bg_geoid": b, "tvd": float(v)} for b, v in sorted_bg[:10]],
    }

    out = {
        "created_utc": _utc_now_iso(),
        "inputs": {
            "exp1_counts_path": str(exp1_path),
            "advisor_zip": str(advisor_zip),
            "advisor_member": args.advisor_member,
            "advisor_layer": args.advisor_layer,
            "tiger_bg_zip": (str(tiger_bg_zip) if tiger_bg_zip else None),
            "data_root": str(pathlib.Path(args.data_root).expanduser().resolve()),
            "statefp": str(args.statefp),
        },
        "summary": summary,
        "note": "Head-to-head compares normalized BG age×sex distributions (not absolute counts).",
    }
    _write_json(out_dir / "advisor_bg_age_sex_comparison.json", out)
    _write_json(
        out_dir / "run.metadata.json",
        {
            "created_utc": _utc_now_iso(),
            "argv": sys.argv,
            "env": {"RAW_ROOT": os.environ.get("RAW_ROOT"), "SYNTHCITY_DATA_ROOT": os.environ.get("SYNTHCITY_DATA_ROOT")},
        },
    )
    print(f"[ok] wrote: {out_dir / 'advisor_bg_age_sex_comparison.json'}")


if __name__ == "__main__":
    main()
