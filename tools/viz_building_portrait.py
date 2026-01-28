#!/usr/bin/env python3
from __future__ import annotations

"""
建筑物尺度人口画像可视化（Detroit PoC）

目标：
- 直接把 building_portrait.csv 做成“可 review / 可进 GIS / 可入稿”的最小产物。
- 对齐论文级风格规范：`docs/visual_style_guide.md` 与 `src/plot_style.py`。

输入：
- building_portrait.csv（来自 tools/poc_tabddpm_pums_buildingcond.py）

输出（out_dir 下）：
- figures/*.png + figures/*.pdf：地图、分档箱线图、分布直方图
- building_portrait_points.geojson：建筑点图层（QGIS/Kepler 直接加载）
- viz_summary.json：关键相关性与分档统计
"""

import argparse
import importlib
import json
import math
import pathlib
import sys
from typing import Any

# Allow running as a plain script without installing the repo.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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


def _robust_vmin_vmax(pd, series, *, q_lo: float = 0.02, q_hi: float = 0.98) -> tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").replace([math.inf, -math.inf], pd.NA).dropna()
    if s.shape[0] == 0:
        return (0.0, 1.0)
    lo = float(s.quantile(q_lo))
    hi = float(s.quantile(q_hi))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(s.min())
        hi = float(s.max())
        if hi <= lo:
            hi = lo + 1.0
    return (lo, hi)


def _save_both(save_figure, fig, out_base: pathlib.Path) -> None:
    save_figure(fig, out_base.with_suffix(".png"))
    save_figure(fig, out_base.with_suffix(".pdf"))


def _scatter_map(
    *,
    plt,
    save_figure,
    despine,
    figsize: tuple[float, float],
    df,
    x: str,
    y: str,
    color: str,
    out_base: pathlib.Path,
    title: str,
    cmap: str,
    colorbar_label: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    s: float = 1.2,
    alpha: float = 0.55,
) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    xs = df[x].to_numpy()
    ys = df[y].to_numpy()
    cs = df[color].to_numpy()

    sc = ax.scatter(xs, ys, c=cs, s=s, cmap=cmap, alpha=alpha, linewidths=0, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(colorbar_label)
    despine(ax)

    fig.tight_layout()
    _save_both(save_figure, fig, out_base)
    plt.close(fig)


def _boxplot_by_tier(
    *,
    plt,
    save_figure,
    despine,
    figsize: tuple[float, float],
    df,
    tier_col: str,
    value_col: str,
    out_base: pathlib.Path,
    title: str,
    y_label: str,
    log1p: bool = False,
) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    tiers = sorted([t for t in df[tier_col].dropna().unique().tolist() if int(t) > 0])
    data = []
    labels = []
    for t in tiers:
        sub = df[df[tier_col] == t][value_col]
        sub = sub.replace([math.inf, -math.inf], float("nan")).dropna()
        if len(sub) == 0:
            continue
        if log1p:
            sub = sub.clip(lower=0.0).apply(lambda v: math.log1p(float(v)))
        data.append(sub.to_numpy())
        labels.append(str(int(t)))

    ax.boxplot(
        data,
        labels=labels,
        showfliers=False,
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        medianprops={"linewidth": 1.6, "color": "black"},
    )
    ax.set_title(title)
    ax.set_xlabel(f"{tier_col} (tract quantiles)")
    ax.set_ylabel(y_label)
    despine(ax)

    fig.tight_layout()
    _save_both(save_figure, fig, out_base)
    plt.close(fig)


def _hist(
    *,
    plt,
    save_figure,
    despine,
    figsize: tuple[float, float],
    pd,
    series,
    out_base: pathlib.Path,
    title: str,
    x_label: str,
    bins: int = 60,
    color: str = "#0072B2",
) -> None:
    s = pd.to_numeric(series, errors="coerce").replace([math.inf, -math.inf], pd.NA).dropna()
    if s.shape[0] == 0:
        return

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.hist(s.to_numpy(), bins=bins, color=color, alpha=0.85, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    despine(ax)
    fig.tight_layout()
    _save_both(save_figure, fig, out_base)
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
            try:
                if hasattr(v, "item"):
                    v = v.item()
            except Exception:
                pass
            p[k] = v

        feats.append({"type": "Feature", "properties": p, "geometry": {"type": "Point", "coordinates": [lon_f, lat_f]}})

    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(prog="viz_building_portrait")
    p.add_argument("--portrait_csv", required=True, help="Path to building_portrait.csv")
    p.add_argument("--out_dir", required=True, help="Output directory (figures + geojson + summary)")
    p.add_argument("--max_points", type=int, default=0, help="Optional: subsample points for maps (0=no subsample).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pd = _require("pandas")
    _require("numpy")
    mpl = _require("matplotlib")
    mpl.use("Agg")  # headless; must be set before pyplot
    plt = _require("matplotlib.pyplot")

    plot_style = _require("src.plot_style")
    paper_style = plot_style.paper_style
    save_figure = plot_style.save_figure
    despine = plot_style.despine
    OKABE_ITO = plot_style.OKABE_ITO
    FIGSIZE_FULL = plot_style.FIGSIZE_FULL
    FIGSIZE_HALF = plot_style.FIGSIZE_HALF

    portrait_csv = pathlib.Path(args.portrait_csv).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(portrait_csv, low_memory=False)
    needed = ["bldg_id", "centroid_lon", "centroid_lat", "pop_count", "age_p50", "income_p50", "price_tier", "price_per_sqft"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"portrait_csv missing columns: {missing}")

    for c in ["centroid_lon", "centroid_lat", "pop_count", "age_p50", "income_p50", "price_tier", "price_per_sqft"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["centroid_lon", "centroid_lat"]).copy()

    # GeoJSON export (no subsample)
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

    # Subsample for maps (optional)
    if int(args.max_points) > 0 and df.shape[0] > int(args.max_points):
        df_map = df.sample(n=int(args.max_points), random_state=int(args.seed)).copy()
    else:
        df_map = df.copy()

    df_map["log_income_p50"] = _log1p_series(pd, df_map["income_p50"])
    df_map["log_pop_count"] = _log1p_series(pd, df_map["pop_count"])

    xlim = (float(df_map["centroid_lon"].min()), float(df_map["centroid_lon"].max()))
    ylim = (float(df_map["centroid_lat"].min()), float(df_map["centroid_lat"].max()))

    with paper_style():
        # Maps (robust color scaling)
        vmin_i, vmax_i = _robust_vmin_vmax(pd, df_map["log_income_p50"])
        vmin_p, vmax_p = _robust_vmin_vmax(pd, df_map["log_pop_count"])
        vmin_a, vmax_a = _robust_vmin_vmax(pd, df_map["age_p50"])

        _scatter_map(
            plt=plt,
            save_figure=save_figure,
            despine=despine,
            figsize=FIGSIZE_FULL,
            df=df_map.dropna(subset=["log_income_p50"]),
            x="centroid_lon",
            y="centroid_lat",
            color="log_income_p50",
            out_base=fig_dir / "map_income_p50_log1p",
            title="Building portrait: log1p(income_p50)",
            cmap="viridis",
            colorbar_label="log1p(income_p50)",
            xlim=xlim,
            ylim=ylim,
            vmin=vmin_i,
            vmax=vmax_i,
        )
        _scatter_map(
            plt=plt,
            save_figure=save_figure,
            despine=despine,
            figsize=FIGSIZE_FULL,
            df=df_map.dropna(subset=["age_p50"]),
            x="centroid_lon",
            y="centroid_lat",
            color="age_p50",
            out_base=fig_dir / "map_age_p50",
            title="Building portrait: age_p50",
            cmap="plasma",
            colorbar_label="age_p50",
            xlim=xlim,
            ylim=ylim,
            vmin=vmin_a,
            vmax=vmax_a,
        )
        _scatter_map(
            plt=plt,
            save_figure=save_figure,
            despine=despine,
            figsize=FIGSIZE_FULL,
            df=df_map.dropna(subset=["log_pop_count"]),
            x="centroid_lon",
            y="centroid_lat",
            color="log_pop_count",
            out_base=fig_dir / "map_pop_count_log1p",
            title="Building portrait: log1p(pop_count)",
            cmap="magma",
            colorbar_label="log1p(pop_count)",
            xlim=xlim,
            ylim=ylim,
            vmin=vmin_p,
            vmax=vmax_p,
        )

        # Tier-wise distributions
        _boxplot_by_tier(
            plt=plt,
            save_figure=save_figure,
            despine=despine,
            figsize=FIGSIZE_FULL,
            df=df,
            tier_col="price_tier",
            value_col="income_p50",
            out_base=fig_dir / "box_income_p50_by_price_tier",
            title="income_p50 by price_tier (tract quantiles)",
            y_label="log1p(income_p50)",
            log1p=True,
        )
        _boxplot_by_tier(
            plt=plt,
            save_figure=save_figure,
            despine=despine,
            figsize=FIGSIZE_FULL,
            df=df,
            tier_col="price_tier",
            value_col="age_p50",
            out_base=fig_dir / "box_age_p50_by_price_tier",
            title="age_p50 by price_tier (tract quantiles)",
            y_label="age_p50",
            log1p=False,
        )

        # Distribution diagnostics (building-level)
        _hist(
            plt=plt,
            save_figure=save_figure,
            despine=despine,
            figsize=FIGSIZE_HALF,
            pd=pd,
            series=_log1p_series(pd, df["income_p50"]),
            out_base=fig_dir / "hist_log_income_p50",
            title="Distribution: log1p(income_p50) across buildings",
            x_label="log1p(income_p50)",
            color=OKABE_ITO["blue"],
        )
        _hist(
            plt=plt,
            save_figure=save_figure,
            despine=despine,
            figsize=FIGSIZE_HALF,
            pd=pd,
            series=_log1p_series(pd, df["pop_count"]),
            out_base=fig_dir / "hist_log_pop_count",
            title="Distribution: log1p(pop_count) across buildings",
            x_label="log1p(pop_count)",
            color=OKABE_ITO["bluish_green"],
        )
        _hist(
            plt=plt,
            save_figure=save_figure,
            despine=despine,
            figsize=FIGSIZE_HALF,
            pd=pd,
            series=_finite_series(pd, df["age_p50"]),
            out_base=fig_dir / "hist_age_p50",
            title="Distribution: age_p50 across buildings",
            x_label="age_p50",
            color=OKABE_ITO["vermillion"],
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
        "notes": {
            "style_source_of_truth": "src/plot_style.py",
            "no_bbox_inches_tight": True,
        },
    }
    (out_dir / "viz_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_dir}/viz_summary.json")
    print(f"[ok] wrote: {out_dir}/building_portrait_points.geojson")
    print(f"[ok] wrote figures under: {fig_dir} (png+pdf)")


if __name__ == "__main__":
    main()

