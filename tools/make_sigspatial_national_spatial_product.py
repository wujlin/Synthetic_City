#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plot_style import OKABE_ITO, PaperStyle, add_panel_label, despine, paper_style, save_figure


def _conus(df: pd.DataFrame, lon_col: str, lat_col: str) -> pd.DataFrame:
    out = df[[lon_col, lat_col, "statefp"]].copy()
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out["statefp"] = out["statefp"].astype(str).str.zfill(2)
    out = out.dropna(subset=[lon_col, lat_col])
    out = out[~out["statefp"].isin(["02", "15"])].copy()
    out = out[(out[lon_col].between(-125.5, -66.0)) & (out[lat_col].between(24.0, 50.5))]
    return out


def _plot_points(ax, df: pd.DataFrame, lon_col: str, lat_col: str, color: str) -> None:
    ax.scatter(
        df[lon_col],
        df[lat_col],
        s=0.9,
        alpha=0.18,
        linewidths=0,
        color=color,
        rasterized=True,
    )
    ax.set_xlim(-125.5, -66.0)
    ax.set_ylim(24.0, 50.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    radius_km = 6371.0088
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _concentration_curve(counts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(counts, errors="coerce").fillna(0.0)
    values = values[values > 0].sort_values(ascending=False).to_numpy(dtype=float)
    if values.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    x = np.arange(1, values.size + 1, dtype=float) / float(values.size)
    y = np.cumsum(values) / float(values.sum())
    return np.r_[0.0, x], np.r_[0.0, y]


def _top_share(counts: pd.Series, frac: float = 0.10) -> float:
    values = pd.to_numeric(counts, errors="coerce").fillna(0.0)
    values = values[values > 0].sort_values(ascending=False).to_numpy(dtype=float)
    if values.size == 0:
        return 0.0
    n = max(1, int(np.ceil(float(frac) * values.size)))
    return float(values[:n].sum() / values.sum())


def _add_counts(base: pd.Series | None, new_counts: pd.Series) -> pd.Series:
    new_counts = pd.to_numeric(new_counts, errors="coerce").fillna(0.0)
    if base is None:
        return new_counts.copy()
    return base.add(new_counts, fill_value=0.0)


def _counts_from_full_outputs(state_qc_csv: Path) -> tuple[pd.Series, pd.Series]:
    state_qc = pd.read_csv(state_qc_csv)
    if "output_parquet" not in state_qc.columns:
        raise ValueError(f"state QC missing output_parquet column: {state_qc_csv}")

    home_counts: pd.Series | None = None
    work_counts: pd.Series | None = None
    cols = ["tract_geoid", "work_tract_geoid", "is_worker"]
    for path_raw in state_qc["output_parquet"].dropna().astype(str).tolist():
        path = Path(path_raw)
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=cols)
        home_counts = _add_counts(home_counts, df["tract_geoid"].dropna().astype(str).value_counts())
        worker = df["is_worker"].astype(bool) & df["work_tract_geoid"].notna()
        work_counts = _add_counts(work_counts, df.loc[worker, "work_tract_geoid"].astype(str).value_counts())
    if home_counts is None or work_counts is None:
        raise RuntimeError(f"could not aggregate tract counts from full outputs listed in: {state_qc_csv}")
    return home_counts, work_counts


def _plot_concentration(ax, home_counts: pd.Series, work_counts: pd.Series) -> None:
    hx, hy = _concentration_curve(home_counts)
    wx, wy = _concentration_curve(work_counts)
    ax.plot(hx, hy, color=OKABE_ITO["bluish_green"], linewidth=2.0, label="Home locations")
    ax.plot(wx, wy, color=OKABE_ITO["orange"], linewidth=2.0, label="Workplace locations")
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.8, linestyle=":", label="Equal concentration")
    home_top = _top_share(home_counts, 0.10)
    work_top = _top_share(work_counts, 0.10)
    ax.axvline(0.10, color="#666666", linewidth=0.8, linestyle="--")
    ax.scatter([0.10, 0.10], [home_top, work_top], s=22, color=[OKABE_ITO["bluish_green"], OKABE_ITO["orange"]], zorder=4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Cumulative share of tracts")
    ax.set_ylabel("Cumulative share of locations")
    ax.legend(frameon=False, loc="lower right")
    despine(ax)


def _plot_commute_distance(ax, distances: np.ndarray) -> None:
    distances = np.asarray(distances, dtype=float)
    distances = distances[np.isfinite(distances) & (distances >= 0)]
    if distances.size == 0:
        ax.text(0.5, 0.5, "No valid home--work pairs", transform=ax.transAxes, ha="center", va="center")
        return
    median = float(np.median(distances))
    p90 = float(np.percentile(distances, 90))
    bins = np.linspace(0, 100, 41)
    ax.hist(
        np.clip(distances, 0, 100),
        bins=bins,
        density=True,
        color=OKABE_ITO["orange"],
        alpha=0.62,
        edgecolor="white",
        linewidth=0.25,
    )
    ax.axvline(median, color="#222222", linewidth=1.2)
    ax.axvline(min(p90, 100.0), color="#555555", linewidth=1.0, linestyle="--")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Home to workplace distance (km)")
    ax.set_ylabel("Density")
    despine(ax)


def build_figure(
    *,
    sample_parquet: Path,
    state_qc_csv: Path | None,
    out_pdf: Path,
    out_png: Path,
) -> None:
    sample = pd.read_parquet(sample_parquet)

    home = _conus(sample, "home_lon", "home_lat")
    work = _conus(sample.loc[sample.get("is_worker", False).astype(bool)].copy(), "work_lon", "work_lat")

    if state_qc_csv is not None:
        home_counts, work_counts = _counts_from_full_outputs(state_qc_csv)
    else:
        home_counts = sample["tract_geoid"].dropna().astype(str).value_counts()
        work_counts = sample.loc[
            sample.get("is_worker", False).astype(bool) & sample["work_tract_geoid"].notna(),
            "work_tract_geoid",
        ].astype(str).value_counts()
    valid_workers = sample.loc[
        sample.get("is_worker", False).astype(bool)
        & sample["home_lon"].notna()
        & sample["home_lat"].notna()
        & sample["work_lon"].notna()
        & sample["work_lat"].notna()
    ].copy()
    distances = _haversine_km(
        valid_workers["home_lon"].to_numpy(dtype=float),
        valid_workers["home_lat"].to_numpy(dtype=float),
        valid_workers["work_lon"].to_numpy(dtype=float),
        valid_workers["work_lat"].to_numpy(dtype=float),
    )

    style = PaperStyle(font_size=8.5, axes_labelsize=9.0, axes_titlesize=9.8, tick_labelsize=8.0, legend_fontsize=7.8)
    with paper_style(style):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), gridspec_kw={"height_ratios": [1.05, 0.95]})
        ax_a, ax_b, ax_c, ax_d = axes.ravel()

        _plot_points(ax_a, home, "home_lon", "home_lat", OKABE_ITO["bluish_green"])
        add_panel_label(ax_a, "a", dx=-20)
        _plot_points(ax_b, work, "work_lon", "work_lat", OKABE_ITO["orange"])
        add_panel_label(ax_b, "b", dx=-20)

        _plot_concentration(ax_c, home_counts, work_counts)
        add_panel_label(ax_c, "c", dx=-34)

        _plot_commute_distance(ax_d, distances)
        add_panel_label(ax_d, "d", dx=-22)

        fig.subplots_adjust(left=0.105, right=0.985, top=0.92, bottom=0.10, wspace=0.28, hspace=0.33)
        save_figure(fig, out_pdf)
        save_figure(fig, out_png, dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-parquet", type=Path, required=True)
    parser.add_argument("--state-qc-csv", type=Path, default=None)
    parser.add_argument("--out-pdf", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args()
    build_figure(
        sample_parquet=args.sample_parquet,
        state_qc_csv=args.state_qc_csv,
        out_pdf=args.out_pdf,
        out_png=args.out_png,
    )


if __name__ == "__main__":
    main()
