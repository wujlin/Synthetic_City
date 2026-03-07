"""
PUMA-level copula heterogeneity choropleth map.

Downloads Census TIGER cartographic boundaries (2020 PUMA, 500k resolution)
and joins with TVD-to-global data to produce a US-wide choropleth.

Usage:
    python tools/essay/plot_choropleth.py

Output:
    Essay/figures/fig_choropleth.pdf
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# inset_axes not needed; we use fig.add_axes for manual insets

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from plot_style import paper_style, save_figure, OKABE_ITO

OUT_DIR = REPO / "Essay" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = REPO / "data" / "geo_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CB_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_puma20_500k.zip"
CB_ZIP = CACHE_DIR / "cb_2020_us_puma20_500k.zip"
CB_DIR = CACHE_DIR / "cb_2020_us_puma20_500k"

# Exclude non-CONUS (AK=02, HI=15 handled via insets; territories excluded)
TERRITORIES = {"60", "66", "69", "72", "78"}
AK_FP, HI_FP = "02", "15"


def download_puma_boundaries() -> gpd.GeoDataFrame:
    """Download and cache Census cartographic boundary file for 2020 PUMAs."""
    if not CB_ZIP.exists():
        print(f"  Downloading PUMA boundaries from {CB_URL} ...")
        urlretrieve(CB_URL, CB_ZIP)
        print(f"  Saved to {CB_ZIP}")
    if not CB_DIR.exists():
        CB_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(CB_ZIP, "r") as zf:
            zf.extractall(CB_DIR)
    shp_files = list(CB_DIR.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp found in {CB_DIR}")
    gdf = gpd.read_file(shp_files[0])
    return gdf


def load_tvd_data() -> pd.DataFrame:
    """Load per-PUMA TVD-to-global from heterogeneity diagnostic."""
    p = REPO / "outputs" / "_tmp_puma5var_us_smoke" / "heterogeneity_diagnostic.json"
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    records = data["by_puma"]
    df = pd.DataFrame(records)
    # Build GEOID to match Census format: STATEFP(2) + PUMA(5)
    df["GEOID20"] = df["statefp"].str.zfill(2) + df["puma"].str.zfill(5)
    return df[["GEOID20", "tvd_to_global"]]


def _get_statefp(gdf: gpd.GeoDataFrame) -> str:
    """Return the column name that holds state FIPS codes."""
    for c in ("STATEFP20", "STATEFP"):
        if c in gdf.columns:
            return c
    return "__statefp__"


def plot_choropleth():
    """Create CONUS + AK/HI inset choropleth of copula heterogeneity."""
    gdf = download_puma_boundaries()
    tvd_df = load_tvd_data()

    if "GEOID20" not in gdf.columns:
        if "STATEFP20" in gdf.columns and "PUMACE20" in gdf.columns:
            gdf["GEOID20"] = gdf["STATEFP20"] + gdf["PUMACE20"]
        elif "STATEFP" in gdf.columns and "PUMACE" in gdf.columns:
            gdf["GEOID20"] = gdf["STATEFP"] + gdf["PUMACE"]
        else:
            raise KeyError(f"Cannot build GEOID. Columns: {list(gdf.columns)}")

    gdf = gdf.merge(tvd_df, on="GEOID20", how="left")
    sfp = _get_statefp(gdf)
    if sfp == "__statefp__":
        gdf[sfp] = gdf["GEOID20"].str[:2]
    gdf = gdf[~gdf[sfp].isin(TERRITORIES)]

    conus = gdf[(gdf[sfp] != AK_FP) & (gdf[sfp] != HI_FP)].copy()
    alaska = gdf[gdf[sfp] == AK_FP].copy()
    hawaii = gdf[gdf[sfp] == HI_FP].copy()

    vmin, vmax = 0.05, 0.40
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.YlOrRd

    plot_kw = dict(
        column="tvd_to_global", cmap=cmap, norm=norm,
        linewidth=0.05, edgecolor="0.6",
        missing_kwds={"color": "white", "edgecolor": "0.85", "linewidth": 0.05},
    )

    with paper_style():
        fig = plt.figure(figsize=(7.0, 4.8))

        # CONUS — main axes, leave room at bottom for AK/HI and colorbar
        ax_main = fig.add_axes([0.02, 0.18, 0.82, 0.80])
        conus.plot(ax=ax_main, **plot_kw)
        ax_main.set_xlim(-128, -65)
        ax_main.set_ylim(23, 52)
        ax_main.axis("off")

        # Alaska inset — bottom-left, clip Aleutians for better framing
        ax_ak = fig.add_axes([0.0, -0.02, 0.25, 0.28])
        if not alaska.empty:
            alaska.plot(ax=ax_ak, **plot_kw)
            ax_ak.set_xlim(-180, -130)
            ax_ak.set_ylim(51, 72)
        ax_ak.axis("off")

        # Hawaii inset — to the right of Alaska, zoomed to main islands
        ax_hi = fig.add_axes([0.24, 0.0, 0.18, 0.18])
        if not hawaii.empty:
            hawaii.plot(ax=ax_hi, **plot_kw)
            ax_hi.set_xlim(-160.5, -154.5)
            ax_hi.set_ylim(18.8, 22.5)
        ax_hi.axis("off")

        # Horizontal colorbar at bottom-right
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.45, 0.08, 0.38, 0.025])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("TVD to national-average copula", fontsize=9, labelpad=4)
        cb.ax.tick_params(labelsize=8)

        save_figure(fig, OUT_DIR / "fig_choropleth.pdf")
        fig.savefig(OUT_DIR / "fig_choropleth.png", dpi=200)
        print(f"  → {OUT_DIR / 'fig_choropleth.pdf'}")
        print(f"  → {OUT_DIR / 'fig_choropleth.png'}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating choropleth...")
    plot_choropleth()
    print("Done.")
