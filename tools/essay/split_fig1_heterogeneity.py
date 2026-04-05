#!/usr/bin/env python3
from __future__ import annotations

"""
Split original heterogeneity figure into:
- fig_01_map.pdf (main text)
- fig_s02_heterogeneity_stats.pdf (SI: distribution stats)
"""

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.plot_style import OKABE_ITO, add_panel_label, despine, paper_style, save_figure


def _load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(prog="split_fig1_heterogeneity")
    ap.add_argument("--puma_zip", required=True, help="cb_2020_us_puma20_500k.zip")
    ap.add_argument("--us_heterogeneity_json", required=True)
    ap.add_argument("--mi_heterogeneity_json", required=True)
    ap.add_argument("--out_map_pdf", required=True)
    ap.add_argument("--out_stats_pdf", required=True)
    ap.add_argument("--out_map_png", default="")
    ap.add_argument("--out_stats_png", default="")
    args = ap.parse_args()

    import geopandas as gpd
    import pandas as pd
    from matplotlib.colors import Normalize

    puma_zip = pathlib.Path(args.puma_zip).expanduser().resolve()
    us_json = pathlib.Path(args.us_heterogeneity_json).expanduser().resolve()
    mi_json = pathlib.Path(args.mi_heterogeneity_json).expanduser().resolve()
    if not puma_zip.exists():
        raise SystemExit(f"puma_zip not found: {puma_zip}")
    if not us_json.exists():
        raise SystemExit(f"us_heterogeneity_json not found: {us_json}")
    if not mi_json.exists():
        raise SystemExit(f"mi_heterogeneity_json not found: {mi_json}")

    us = _load_json(us_json)
    mi = _load_json(mi_json)
    us_rows = us["by_puma"]
    mi_rows = mi["by_puma"]
    us_tvds = np.array([float(r["tvd_to_global"]) for r in us_rows], dtype=float)
    mi_tvds = np.array([float(r["tvd_to_global"]) for r in mi_rows], dtype=float)

    # --- map ---
    gdf = gpd.read_file(f"zip://{puma_zip}")
    df = pd.DataFrame(us_rows)
    df["GEOID20"] = df["statefp"].astype(str).str.zfill(2) + df["puma"].astype(str).str.zfill(5)
    if "GEOID20" not in gdf.columns:
        if "STATEFP20" in gdf.columns and "PUMACE20" in gdf.columns:
            gdf["GEOID20"] = gdf["STATEFP20"].astype(str) + gdf["PUMACE20"].astype(str)
        else:
            raise SystemExit("Cannot build GEOID20 from puma shapefile.")

    gdf = gdf.merge(df[["GEOID20", "tvd_to_global"]], on="GEOID20", how="left")
    TERR = {"60", "66", "69", "72", "78"}
    sfp_col = "STATEFP20" if "STATEFP20" in gdf.columns else "STATEFP"
    gdf = gdf[~gdf[sfp_col].astype(str).isin(TERR)].copy()

    conus = gdf[(gdf[sfp_col] != "02") & (gdf[sfp_col] != "15")]
    alaska = gdf[gdf[sfp_col] == "02"]
    hawaii = gdf[gdf[sfp_col] == "15"]
    michigan = gdf[gdf[sfp_col].astype(str) == "26"].copy()
    cmap = plt.cm.YlOrRd
    norm = Normalize(vmin=0.05, vmax=0.40)
    plot_kw = dict(
        column="tvd_to_global",
        cmap=cmap,
        norm=norm,
        linewidth=0.05,
        edgecolor="0.6",
        missing_kwds={"color": "white", "edgecolor": "0.85", "linewidth": 0.05},
    )

    with paper_style():
        fig = plt.figure(figsize=(7.0, 3.9))
        ax = fig.add_axes([0.03, 0.08, 0.82, 0.84])
        conus.plot(ax=ax, **plot_kw)
        if not michigan.empty:
            michigan.dissolve().boundary.plot(ax=ax, color="#2B6CB0", linewidth=1.4, zorder=5)
            ax.annotate(
                "Michigan",
                xy=(-85.6, 44.8),
                xytext=(-79.8, 48.7),
                color="#2B6CB0",
                fontsize=9.5,
                ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", color="#2B6CB0", lw=0.8, shrinkA=2, shrinkB=2),
            )
        ax.set_xlim(-128, -65)
        ax.set_ylim(23, 52)
        ax.axis("off")

        ax_ak = fig.add_axes([0.03, 0.12, 0.16, 0.24])
        if not alaska.empty:
            alaska.plot(ax=ax_ak, **plot_kw)
            ax_ak.set_xlim(-180, -130)
            ax_ak.set_ylim(51, 72)
        ax_ak.axis("off")

        ax_hi = fig.add_axes([0.17, 0.12, 0.11, 0.14])
        if not hawaii.empty:
            hawaii.plot(ax=ax_hi, **plot_kw)
            ax_hi.set_xlim(-160.5, -154.5)
            ax_hi.set_ylim(18.8, 22.5)
        ax_hi.axis("off")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.33, 0.08, 0.47, 0.03])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("TVD to national-average copula", fontsize=9.5, labelpad=3)
        cb.ax.tick_params(labelsize=8.5)

        out_map_pdf = pathlib.Path(args.out_map_pdf).expanduser().resolve()
        save_figure(fig, out_map_pdf)
        if str(args.out_map_png).strip():
            fig.savefig(pathlib.Path(args.out_map_png).expanduser().resolve(), dpi=220)
        plt.close(fig)

    # --- SI stats ---
    C1 = OKABE_ITO["blue"]
    C2 = OKABE_ITO["vermillion"]
    C3 = OKABE_ITO["bluish_green"]
    with paper_style():
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3))
        fig.subplots_adjust(wspace=0.30, bottom=0.24)

        ax = axes[0]
        ax.hist(us_tvds, bins=40, color=C1, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(float(np.mean(us_tvds)), color=C2, linestyle="--", linewidth=1.5, label=f"Mean={np.mean(us_tvds):.3f}")
        ax.set_xlabel("TVD to global")
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=8.2)
        despine(ax)
        add_panel_label(ax, "a", dx=-18)

        ax = axes[1]
        ax.scatter(np.arange(us_tvds.size), np.sort(us_tvds), s=4, color=C1, alpha=0.45, rasterized=True)
        ax.axhline(float(np.mean(us_tvds)), color=C2, linestyle="--", linewidth=1.2)
        ax.set_xlabel("PUMA rank")
        ax.set_ylabel("TVD to global")
        despine(ax)
        add_panel_label(ax, "b", dx=-18)

        ax = axes[2]
        ax.hist(mi_tvds, bins=20, color=C3, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(float(np.mean(mi_tvds)), color=C2, linestyle="--", linewidth=1.5, label=f"MI mean={np.mean(mi_tvds):.3f}")
        ax.set_xlabel("TVD to global")
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=8.2)
        despine(ax)
        add_panel_label(ax, "c", dx=-18)

        out_stats_pdf = pathlib.Path(args.out_stats_pdf).expanduser().resolve()
        save_figure(fig, out_stats_pdf)
        if str(args.out_stats_png).strip():
            fig.savefig(pathlib.Path(args.out_stats_png).expanduser().resolve(), dpi=220)
        plt.close(fig)

    print(f"[ok] wrote: {pathlib.Path(args.out_map_pdf).expanduser().resolve()}")
    print(f"[ok] wrote: {pathlib.Path(args.out_stats_pdf).expanduser().resolve()}")
    if str(args.out_map_png).strip():
        print(f"[ok] wrote: {pathlib.Path(args.out_map_png).expanduser().resolve()}")
    if str(args.out_stats_png).strip():
        print(f"[ok] wrote: {pathlib.Path(args.out_stats_png).expanduser().resolve()}")


if __name__ == "__main__":
    main()
