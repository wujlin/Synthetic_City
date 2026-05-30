#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, FixedLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthpop.plot_style import add_panel_label, paper_style, save_figure


PROJECT_ROOT = Path("/Users/jinlin/Desktop/Project/Synthetic_City")
DEFAULT_ATTR_CSV = (
    PROJECT_ROOT / "outputs" / "_sync_phase3_detroit_hcenter_20260330" / "attribute_spatial_home_shares.csv"
)
DEFAULT_TRACT_ZIP = PROJECT_ROOT / "dataset" / "cache" / "geo" / "tl_2023_26_tract.zip"


def _pct_fmt(x: float, _pos=None) -> str:
    return f"{100.0 * x:.0f}%"


def _despine_map(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")


def _small_horizontal_colorbar(
    ax,
    sm,
    label: str,
    ticks: list[float],
    *,
    box: tuple[float, float, float, float] = (0.18, -0.095, 0.50, 0.040),
) -> None:
    cax = ax.inset_axes(list(box))
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.55)
    cb.outline.set_edgecolor("#8E8678")
    cb.ax.tick_params(length=1.8, width=0.55, pad=1.0, labelsize=8.6, colors="#6F675B")
    cb.locator = FixedLocator(ticks)
    cb.update_ticks()
    cb.ax.xaxis.set_major_formatter(FuncFormatter(_pct_fmt))
    cb.set_label(label, labelpad=1.4, size=9.2, color="#5B5449")


def _attribute_panel(
    ax,
    gdf: gpd.GeoDataFrame,
    value_col: str,
    *,
    cmap: str = "YlGnBu",
    vmin_quantile: float | None = None,
    vmax_quantile: float | None = 0.995,
    center_tick: float | None = None,
    center_value: float | None = None,
) -> None:
    values = pd.to_numeric(gdf[value_col], errors="coerce")
    finite = values[np.isfinite(values)]
    if len(finite):
        if vmin_quantile is None:
            vmin = 0.0
        else:
            vmin = float(np.quantile(finite, vmin_quantile))
        if vmax_quantile is None:
            vmax = float(finite.max())
        else:
            vmax = float(np.quantile(finite, vmax_quantile))
        vmax = max(vmax, vmin + 1e-6)
    else:
        vmin = 0.0
        vmax = 1.0
    if center_value is not None and vmin < center_value < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center_value, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    gdf.plot(
        ax=ax,
        column=value_col,
        cmap=cmap,
        norm=norm,
        linewidth=0.14,
        edgecolor="#E4DED2",
        legend=False,
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    if center_tick is not None and vmin < center_tick < vmax:
        ticks = [vmin, center_tick, vmax]
    else:
        ticks = [vmin, (vmin + vmax) / 2.0, vmax]
    _small_horizontal_colorbar(ax, sm, "Share of tract residents", ticks)
    _despine_map(ax)


def _blank_panel(ax) -> None:
    ax.set_axis_off()


def build_figure(attr_csv: Path, tract_zip: Path, out_pdf: Path, out_png: Path) -> None:
    attr_df = pd.read_csv(attr_csv, dtype={"tract_geoid": str})
    tracts = gpd.read_file(tract_zip)
    tracts["tract_geoid"] = tracts["GEOID"].astype(str)
    gdf = tracts.loc[tracts["tract_geoid"].isin(set(attr_df["tract_geoid"]))].copy()
    gdf = gdf.merge(attr_df, on="tract_geoid", how="left")

    panels = [
        ("child_share", {}),
        (
            "female_share",
            {
                "cmap": "RdBu_r",
                "vmin_quantile": 0.01,
                "vmax_quantile": 0.99,
                "center_tick": 0.50,
                "center_value": 0.50,
            },
        ),
        ("employed_share", {}),
        ("bachelor_plus_share", {}),
        ("high_income_share", {}),
    ]

    with paper_style():
        fig, axes = plt.subplots(2, 3, figsize=(10.15, 6.95))
        axes = axes.reshape(2, 3)
        map_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
        for idx, (ax, (col, kwargs)) in enumerate(zip(map_axes, panels)):
            _attribute_panel(ax, gdf, col, **kwargs)
            add_panel_label(ax, chr(ord("a") + idx), dx=-24.0, dy=3.0)
        _blank_panel(axes[1, 2])

        fig.subplots_adjust(left=0.045, right=0.995, top=0.965, bottom=0.09, wspace=0.08, hspace=0.24)
        save_figure(fig, out_pdf)
        save_figure(fig, out_png, dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attr_csv", type=Path, default=DEFAULT_ATTR_CSV)
    parser.add_argument("--tract_zip", type=Path, default=DEFAULT_TRACT_ZIP)
    parser.add_argument("--out_pdf", type=Path, required=True)
    parser.add_argument("--out_png", type=Path, required=True)
    args = parser.parse_args()
    build_figure(args.attr_csv, args.tract_zip, args.out_pdf, args.out_png)


if __name__ == "__main__":
    main()
