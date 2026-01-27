#!/usr/bin/env python3
from __future__ import annotations

"""
建筑物尺度人口画像可视化（Detroit PoC）

目标（KISS）：
- 直接把 building_portrait.csv（以及其中的 price_tier/price_per_sqft）做成可 review 的图与可上 GIS 的点数据。
- 不引入复杂 Web 前端；默认输出 PNG + GeoJSON。

输入（最小）：
- building_portrait.csv（来自 tools/poc_tabddpm_pums_buildingcond.py）

输出：
- figures/*.png：income/age/pop_count 的建筑点图 + price_tier 分档分布图
- building_portrait_points.geojson：便于 QGIS/Kepler 直接加载
- viz_summary.json：核心统计（相关性、分档中位数等）
"""

import argparse
import importlib
import json
import math
import pathlib
from typing import Any


def _require(pkg: str) -> Any:
    try:
        return importlib.import_module(pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency: {pkg}\n"
            "Recommended (conda): conda install -c conda-forge pandas numpy matplotlib\n"
            "Or (pip): pip install pandas numpy matplotlib"
        ) from e


def _finite_series(pd, s):
    s = pd.to_numeric(s, errors="coerce")
    return s.replace([math.inf, -math.inf], pd.NA).dropna()


def _log1p_series(pd, s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.clip(lower=0.0)
    return (s + 1.0).apply(math.log)


def _scatter_map(*, plt, df, x: str, y: str, color: str, out_png: pathlib.Path, title: str, cmap: str) -> None:
    fig = plt.figure(figsize=(8, 7), dpi=200)
    ax = fig.add_subplot(111)

    xs = df[x].to_numpy()
    ys = df[y].to_numpy()
    cs = df[color].to_numpy()

    sc = ax.scatter(xs, ys, c=cs, s=1.5, cmap=cmap, alpha=0.6, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_aspect("equal", adjustable="box")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(color)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _boxplot_by_tier(*, plt, df, tier_col: str, value_col: str, out_png: pathlib.Path, title: str) -> None:
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = fig.add_subplot(111)

    tiers = sorted([t for t in df[tier_col].dropna().unique().tolist() if int(t) > 0])
    data = []
    labels = []
    for t in tiers:
        sub = df[df[tier_col] == t][value_col]
        sub = sub.replace([math.inf, -math.inf], float("nan")).dropna()
        if len(sub) == 0:
            continue
        data.append(sub.to_numpy())
        labels.append(str(int(t)))

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(f"{tier_col} (Q1..Qn)")
    ax.set_ylabel(value_col)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _write_geojson_points(*, df, out_path: pathlib.Path, lon_col: str, lat_col: str, props: list[str]) -> None:
    feats = []
    for _, row in df.iterrows():
        lon = row.get(lon_col)
        lat = row.get(lat_col)
        if lon is None or lat is None:
            continue
        try:
            lon_f = float(lon)
            lat_f = float(lat)
        except Exception:
            continue

        p = {}
        for k in props:
            v = row.get(k)
            if v is None:
                continue
            if isinstance(v, (int, float)) and (math.isnan(v) if isinstance(v, float) else False):
                continue
            # pandas scalar / numpy scalar
            try:
                if hasattr(v, "item"):
                    v = v.item()
            except Exception:
                pass
            p[k] = v

        feats.append(
            {
                "type": "Feature",
                "properties": p,
                "geometry": {"type": "Point", "coordinates": [lon_f, lat_f]},
            }
        )

    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(prog="viz_building_portrait")
    p.add_argument("--portrait_csv", required=True, help="Path to building_portrait.csv")
    p.add_argument("--out_dir", required=True, help="Output directory (figures + geojson + summary)")
    p.add_argument("--max_points", type=int, default=0, help="Optional: subsample points for maps (0=no subsample).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pd = _require("pandas")
    np = _require("numpy")
    mpl = _require("matplotlib")
    mpl.use("Agg")  # headless
    plt = _require("matplotlib.pyplot")

    portrait_csv = pathlib.Path(args.portrait_csv).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(portrait_csv, low_memory=False)
    needed = ["bldg_id", "centroid_lon", "centroid_lat", "pop_count", "age_p50", "income_p50", "price_tier", "price_per_sqft"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"portrait_csv missing columns: {missing}")

    df["centroid_lon"] = pd.to_numeric(df["centroid_lon"], errors="coerce")
    df["centroid_lat"] = pd.to_numeric(df["centroid_lat"], errors="coerce")
    df["pop_count"] = pd.to_numeric(df["pop_count"], errors="coerce")
    df["age_p50"] = pd.to_numeric(df["age_p50"], errors="coerce")
    df["income_p50"] = pd.to_numeric(df["income_p50"], errors="coerce")
    df["price_tier"] = pd.to_numeric(df["price_tier"], errors="coerce")
    df["price_per_sqft"] = pd.to_numeric(df["price_per_sqft"], errors="coerce")

    df = df.dropna(subset=["centroid_lon", "centroid_lat"]).copy()

    # Keep a copy for GIS export (no subsample)
    geojson_props = [
        "bldg_id",
        "pop_count",
        "age_p50",
        "income_p50",
        "price_tier",
        "price_per_sqft",
        "dist_cbd_km",
        "puma",
    ]
    _write_geojson_points(
        df=df,
        out_path=out_dir / "building_portrait_points.geojson",
        lon_col="centroid_lon",
        lat_col="centroid_lat",
        props=[c for c in geojson_props if c in df.columns],
    )

    # Subsample for scatter maps (optional)
    if int(args.max_points) > 0 and df.shape[0] > int(args.max_points):
        df_map = df.sample(n=int(args.max_points), random_state=int(args.seed)).copy()
    else:
        df_map = df

    # Map colors (log-scale where needed)
    df_map = df_map.copy()
    df_map["log_income_p50"] = _log1p_series(pd, df_map["income_p50"])
    df_map["log_pop_count"] = _log1p_series(pd, df_map["pop_count"])
    df_map["age_p50_clean"] = _finite_series(pd, df_map["age_p50"])

    _scatter_map(
        plt=plt,
        df=df_map.dropna(subset=["log_income_p50"]),
        x="centroid_lon",
        y="centroid_lat",
        color="log_income_p50",
        out_png=fig_dir / "map_income_p50_log1p.png",
        title="Building portrait: log1p(income_p50)",
        cmap="viridis",
    )
    _scatter_map(
        plt=plt,
        df=df_map.dropna(subset=["age_p50"]),
        x="centroid_lon",
        y="centroid_lat",
        color="age_p50",
        out_png=fig_dir / "map_age_p50.png",
        title="Building portrait: age_p50",
        cmap="plasma",
    )
    _scatter_map(
        plt=plt,
        df=df_map.dropna(subset=["log_pop_count"]),
        x="centroid_lon",
        y="centroid_lat",
        color="log_pop_count",
        out_png=fig_dir / "map_pop_count_log1p.png",
        title="Building portrait: log1p(pop_count)",
        cmap="magma",
    )

    # Tier-wise distributions
    _boxplot_by_tier(
        plt=plt,
        df=df,
        tier_col="price_tier",
        value_col="income_p50",
        out_png=fig_dir / "box_income_p50_by_price_tier.png",
        title="income_p50 by price_tier (tract quantiles)",
    )
    _boxplot_by_tier(
        plt=plt,
        df=df,
        tier_col="price_tier",
        value_col="age_p50",
        out_png=fig_dir / "box_age_p50_by_price_tier.png",
        title="age_p50 by price_tier (tract quantiles)",
    )

    # Summary
    def _pearson(a, b) -> float:
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        m = a.notna() & b.notna()
        if int(m.sum()) < 3:
            return float("nan")
        return float(a[m].corr(b[m], method="pearson"))

    summary = {
        "portrait_csv": str(portrait_csv),
        "n_buildings": int(df.shape[0]),
        "corr": {
            "pearson_income_p50__price_tier": _pearson(df["income_p50"], df["price_tier"]),
            "pearson_income_p50__price_per_sqft": _pearson(_log1p_series(pd, df["income_p50"]), _log1p_series(pd, df["price_per_sqft"])),
            "pearson_age_p50__price_tier": _pearson(df["age_p50"], df["price_tier"]),
        },
        "income_p50_by_price_tier": {
            str(int(t)): {
                "count": int(g.shape[0]),
                "median": float(g["income_p50"].median()),
                "mean": float(g["income_p50"].mean()),
            }
            for t, g in df.dropna(subset=["price_tier"]).groupby("price_tier", sort=True)
            if int(t) > 0
        },
        "artifacts": {
            "geojson": str(out_dir / "building_portrait_points.geojson"),
            "fig_dir": str(fig_dir),
        },
    }
    (out_dir / "viz_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_dir}/viz_summary.json")
    print(f"[ok] wrote: {out_dir}/building_portrait_points.geojson")
    print(f"[ok] wrote figures under: {fig_dir}")


if __name__ == "__main__":
    main()
