#!/usr/bin/env python3
from __future__ import annotations

"""
Standalone PUMA-level copula heterogeneity choropleth.

Unlike the older version, this script does not download any geometry.
The caller must provide a local Census PUMA shapefile ZIP and a local
heterogeneity JSON.
"""

import argparse
import json
import pathlib
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from plot_style import paper_style, save_figure

TERRITORIES = {"60", "66", "69", "72", "78"}
AK_FP = "02"
HI_FP = "15"


def _load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tvd_df(path: pathlib.Path) -> pd.DataFrame:
    data = _load_json(path)
    rows = data["by_puma"]
    df = pd.DataFrame(rows)
    df["GEOID20"] = df["statefp"].astype(str).str.zfill(2) + df["puma"].astype(str).str.zfill(5)
    return df[["GEOID20", "tvd_to_global"]].copy()


def _read_puma_gdf(puma_zip: pathlib.Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(f"zip://{puma_zip}")
    if "GEOID20" not in gdf.columns:
        if "STATEFP20" in gdf.columns and "PUMACE20" in gdf.columns:
            gdf["GEOID20"] = gdf["STATEFP20"].astype(str) + gdf["PUMACE20"].astype(str)
        elif "STATEFP" in gdf.columns and "PUMACE" in gdf.columns:
            gdf["GEOID20"] = gdf["STATEFP"].astype(str) + gdf["PUMACE"].astype(str)
        else:
            raise SystemExit(f"Cannot build GEOID20 from shapefile columns: {list(gdf.columns)}")
    return gdf


def main() -> None:
    ap = argparse.ArgumentParser(prog="plot_choropleth")
    ap.add_argument("--puma_zip", required=True, help="Local cb_2020_us_puma20_500k.zip")
    ap.add_argument("--heterogeneity_json", required=True, help="Local heterogeneity_diagnostic.json")
    ap.add_argument("--out_pdf", required=True)
    ap.add_argument("--out_png", default="")
    args = ap.parse_args()

    puma_zip = pathlib.Path(args.puma_zip).expanduser().resolve()
    het_json = pathlib.Path(args.heterogeneity_json).expanduser().resolve()
    out_pdf = pathlib.Path(args.out_pdf).expanduser().resolve()
    out_png = pathlib.Path(args.out_png).expanduser().resolve() if str(args.out_png).strip() else None

    if not puma_zip.exists():
        raise SystemExit(f"puma_zip not found: {puma_zip}")
    if not het_json.exists():
        raise SystemExit(f"heterogeneity_json not found: {het_json}")

    print(f"[info] puma_zip: {puma_zip}")
    print(f"[info] heterogeneity_json: {het_json}")

    gdf = _read_puma_gdf(puma_zip)
    tvd_df = _load_tvd_df(het_json)
    gdf = gdf.merge(tvd_df, on="GEOID20", how="left")

    sfp_col = "STATEFP20" if "STATEFP20" in gdf.columns else "STATEFP"
    gdf[sfp_col] = gdf[sfp_col].astype(str).str.zfill(2)
    gdf = gdf[~gdf[sfp_col].isin(TERRITORIES)].copy()

    conus = gdf[(gdf[sfp_col] != AK_FP) & (gdf[sfp_col] != HI_FP)].copy()
    alaska = gdf[gdf[sfp_col] == AK_FP].copy()
    hawaii = gdf[gdf[sfp_col] == HI_FP].copy()

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

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with paper_style():
        fig = plt.figure(figsize=(7.0, 4.8))

        ax_main = fig.add_axes([0.02, 0.18, 0.82, 0.80])
        conus.plot(ax=ax_main, **plot_kw)
        ax_main.set_xlim(-128, -65)
        ax_main.set_ylim(23, 52)
        ax_main.axis("off")

        ax_ak = fig.add_axes([0.0, -0.02, 0.25, 0.28])
        if not alaska.empty:
            alaska.plot(ax=ax_ak, **plot_kw)
            ax_ak.set_xlim(-180, -130)
            ax_ak.set_ylim(51, 72)
        ax_ak.axis("off")

        ax_hi = fig.add_axes([0.24, 0.0, 0.18, 0.18])
        if not hawaii.empty:
            hawaii.plot(ax=ax_hi, **plot_kw)
            ax_hi.set_xlim(-160.5, -154.5)
            ax_hi.set_ylim(18.8, 22.5)
        ax_hi.axis("off")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.45, 0.08, 0.38, 0.025])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("TVD to national-average copula", fontsize=9, labelpad=4)
        cb.ax.tick_params(labelsize=8)

        save_figure(fig, out_pdf)
        if out_png is not None:
            fig.savefig(out_png, dpi=200)
        plt.close(fig)

    print(f"[ok] wrote: {out_pdf}")
    if out_png is not None:
        print(f"[ok] wrote: {out_png}")


if __name__ == "__main__":
    main()
