#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from synthpop.paths import data_root
from tools.data.build_external_condition_v1_acs_puma import _parse_states, _scope_tag
from synthpop.data.state_codes import STATEFP_TO_POSTAL as _STATEFP_TO_POSTAL_50


TABLE_VAR_COUNTS: dict[str, int] = {
    "B01001": 49,
    "B15003": 25,
    "B23025": 7,
    "B20001": 43,
}


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _var_names(table_id: str, suffix: str) -> list[str]:
    n = int(TABLE_VAR_COUNTS[table_id])
    return [f"{table_id}_{i:03d}{suffix}" for i in range(1, n + 1)]


def _fetch_acs_puma_table(
    *,
    acs_year: int,
    table_id: str,
    suffix: str,
    statefp: str,
    api_key: str | None,
    retries: int,
) -> pd.DataFrame:
    base = f"https://api.census.gov/data/{int(acs_year)}/acs/acs5"
    params = {
        "get": ",".join(["NAME"] + _var_names(table_id, suffix)),
        "for": "public use microdata area:*",
        "in": f"state:{str(statefp).zfill(2)}",
    }
    if api_key:
        params["key"] = api_key
    url = base + "?" + urllib.parse.urlencode(params)
    last_err: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                rows = json.loads(resp.read().decode("utf-8"))
            if not rows or len(rows) < 2:
                raise RuntimeError(f"empty ACS response for {table_id}{suffix}, state={statefp}")
            df = pd.DataFrame(rows[1:], columns=rows[0])
            df["statefp"] = str(statefp).zfill(2)
            df["puma"] = df["public use microdata area"].astype(str).str.zfill(5)
            df["puma_uid"] = df["statefp"] + df["puma"]
            return df
        except Exception as exc:  # pragma: no cover - network resilience
            last_err = exc
            time.sleep(2.0 + attempt)
    raise RuntimeError(f"failed ACS fetch for {table_id}{suffix}, state={statefp}: {last_err}") from last_err


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    out = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return out


def _table_uncertainty_features(*, est: pd.DataFrame, moe: pd.DataFrame, table_id: str) -> pd.DataFrame:
    e_cols = _var_names(table_id, "E")
    m_cols = _var_names(table_id, "M")
    e = _safe_numeric(est, e_cols)
    m = _safe_numeric(moe, m_cols)

    valid = np.isfinite(e) & np.isfinite(m) & (m >= 0)
    se = np.where(valid, np.abs(m) / 1.645, np.nan)
    denom = np.maximum(np.abs(e), 1.0)
    rse = se / denom
    log_rse = np.log1p(rse)

    def nan_stat(fn: Any, arr: np.ndarray, default: float = 0.0) -> np.ndarray:
        with np.errstate(all="ignore"):
            vals = fn(arr, axis=1)
        vals = np.asarray(vals, dtype=np.float64)
        vals = np.where(np.isfinite(vals), vals, default)
        return vals

    valid_count = valid.sum(axis=1).astype(np.float64)
    high_03 = np.where(valid_count > 0, np.nansum(rse > 0.30, axis=1) / np.maximum(valid_count, 1.0), 0.0)
    high_05 = np.where(valid_count > 0, np.nansum(rse > 0.50, axis=1) / np.maximum(valid_count, 1.0), 0.0)

    out = pd.DataFrame(
        {
            "statefp": est["statefp"].astype(str).str.zfill(2),
            "puma": est["puma"].astype(str).str.zfill(5),
            "puma_uid": est["puma_uid"].astype(str).str.zfill(7),
            f"unc__{table_id}__valid_cell_share": valid_count / max(len(e_cols), 1),
            f"unc__{table_id}__mean_log_rse": nan_stat(np.nanmean, log_rse),
            f"unc__{table_id}__median_log_rse": nan_stat(np.nanmedian, log_rse),
            f"unc__{table_id}__p90_log_rse": nan_stat(lambda x, axis: np.nanpercentile(x, 90, axis=axis), log_rse),
            f"unc__{table_id}__max_log_rse": nan_stat(np.nanmax, log_rse),
            f"unc__{table_id}__share_rse_gt_0p30": high_03,
            f"unc__{table_id}__share_rse_gt_0p50": high_05,
            f"unc__{table_id}__mean_log1p_moe": nan_stat(np.nanmean, np.log1p(np.where(valid, np.abs(m), np.nan))),
        }
    )
    return out


def _build_state_features(*, acs_year: int, statefp: str, api_key: str | None, retries: int) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for table_id in TABLE_VAR_COUNTS:
        est = _fetch_acs_puma_table(
            acs_year=int(acs_year),
            table_id=table_id,
            suffix="E",
            statefp=statefp,
            api_key=api_key,
            retries=int(retries),
        )
        moe = _fetch_acs_puma_table(
            acs_year=int(acs_year),
            table_id=table_id,
            suffix="M",
            statefp=statefp,
            api_key=api_key,
            retries=int(retries),
        )
        tables.append(_table_uncertainty_features(est=est, moe=moe, table_id=table_id))

    out = tables[0]
    for feat in tables[1:]:
        out = out.merge(feat.drop(columns=["statefp", "puma"]), on="puma_uid", how="outer")
        out["statefp"] = out["puma_uid"].astype(str).str[:2]
        out["puma"] = out["puma_uid"].astype(str).str[2:]

    feature_cols = [c for c in out.columns if c.startswith("unc__")]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Compact cross-table summary features.
    mean_cols = [c for c in feature_cols if c.endswith("__mean_log_rse")]
    p90_cols = [c for c in feature_cols if c.endswith("__p90_log_rse")]
    high_cols = [c for c in feature_cols if c.endswith("__share_rse_gt_0p30")]
    out["unc__all_tables__mean_log_rse"] = out[mean_cols].mean(axis=1)
    out["unc__all_tables__max_p90_log_rse"] = out[p90_cols].max(axis=1)
    out["unc__all_tables__mean_share_rse_gt_0p30"] = out[high_cols].mean(axis=1)
    return out.sort_values(["statefp", "puma"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(prog="build_acs_puma_uncertainty_features")
    ap.add_argument("--acs_year", type=int, default=2022)
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--statefps", default="")
    ap.add_argument("--all_states", action="store_true")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--out_path", type=pathlib.Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    states = _parse_states(statefp=args.statefp, statefps=args.statefps, all_states=bool(args.all_states))
    bad = [s for s in states if s not in _STATEFP_TO_POSTAL_50]
    if bad:
        raise SystemExit(f"unsupported statefps: {bad}")

    scope = _scope_tag(states)
    root = data_root()
    out_path = (
        pathlib.Path(args.out_path).expanduser().resolve()
        if args.out_path is not None
        else root / "us" / "processed" / "features" / f"acs5_{int(args.acs_year)}_puma_uncertainty_{scope}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not bool(args.overwrite):
        raise SystemExit(f"out_path exists: {out_path} (use --overwrite)")

    api_key = args.api_key or os.environ.get("CENSUS_API_KEY")
    frames: list[pd.DataFrame] = []
    for statefp in states:
        frame = _build_state_features(acs_year=int(args.acs_year), statefp=statefp, api_key=api_key, retries=int(args.retries))
        frames.append(frame)
        print(f"[ok] state={statefp} pumas={frame.shape[0]}", file=sys.stderr)

    out = pd.concat(frames, ignore_index=True).sort_values(["statefp", "puma"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    feature_cols = [c for c in out.columns if c.startswith("unc__")]
    meta = {
        "dataset": "ACS PUMA survey-uncertainty features",
        "created_utc": _utc_now(),
        "acs_year": int(args.acs_year),
        "scope": scope,
        "statefps": states,
        "n_states": len(states),
        "n_pumas": int(out.shape[0]),
        "tables": list(TABLE_VAR_COUNTS),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "definition": "RSE = ACS MOE / 1.645 / max(abs(estimate), 1); negative ACS MOE sentinel values are treated as missing.",
        "api_key_used": bool(api_key),
        "out_path": str(out_path),
    }
    out_path.with_suffix(out_path.suffix + ".metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
