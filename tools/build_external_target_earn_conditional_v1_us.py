#!/usr/bin/env python3
from __future__ import annotations

"""
Build US-wide PUMS-derived conditional earnings targets at the
PUMA x 4-attribute-cell level.

Target object:
- p(EARN_16p_bin | AGEP_bin, SEX, SCHL_allpop, ESR_allpop, PUMA)

This artifact is used to turn the current region-level earnings proxy into a
person-level assignment step: once a synthetic person is placed into a
4-attribute cell, the model can draw an earnings bin conditionally from that
cell and the region.
"""

import argparse
import json
import pathlib
import sys
import zipfile
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.synthpop.paths import data_root
from tools.build_external_target_v1_michigan import (
    AGE_LABELS,
    ESR_LABELS,
    SCHL_LABELS,
    SEX_LABELS,
    SHAPE,
    _bin_age,
    _bin_esr_allpop,
    _bin_schl_allpop,
    _bin_sex,
    _normalize_puma,
    _resolve_person_zip,
    _utc_now_iso,
)
from src.synthpop.data.state_codes import STATEFP_TO_POSTAL as _STATEFP_TO_POSTAL_50
from tools.external_earn_v1_schema import EARN_LABELS, bin_earn_allpop
from tools.train_us_puma_5var_diffusion import _canon_puma5, _canon_statefp, _canon_uid


FINE_K = int(np.prod(SHAPE))


def _parse_states(*, statefp: str, statefps: str, all_states: bool) -> list[str]:
    if bool(all_states):
        return sorted(_STATEFP_TO_POSTAL_50.keys())
    if str(statefps).strip():
        vals = [str(x).strip().zfill(2) for x in str(statefps).split(",") if str(x).strip()]
        if not vals:
            raise SystemExit("--statefps provided but empty.")
        bad = [s for s in vals if s not in _STATEFP_TO_POSTAL_50]
        if bad:
            raise SystemExit(f"Unsupported statefps: {bad}")
        return vals
    s = str(statefp).zfill(2)
    if s not in _STATEFP_TO_POSTAL_50:
        raise SystemExit(f"Unsupported statefp={s}")
    return [s]


def _scope_tag(states: list[str]) -> str:
    if len(states) == len(_STATEFP_TO_POSTAL_50):
        return "us"
    return "state" + "_".join(states)


def _find_csv_member(path: pathlib.Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise SystemExit(f"No CSV member found in zip: {path}")
        names = sorted(names, key=lambda n: zf.getinfo(n).file_size, reverse=True)
        return names[0]


def _load_person_df(person_path: pathlib.Path) -> pd.DataFrame:
    usecols = ["PUMA", "PUMA20", "PWGTP", "AGEP", "SEX", "SCHL", "ESR", "PERNP"]
    if person_path.suffix.lower() == ".zip":
        member = _find_csv_member(person_path)
        with zipfile.ZipFile(person_path) as zf, zf.open(member) as f:
            return pd.read_csv(f, usecols=lambda c: c in set(usecols), low_memory=False)
    return pd.read_csv(person_path, usecols=lambda c: c in set(usecols), low_memory=False)


def _aggregate_state_conditional(*, statefp: str, person_path: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = _load_person_df(person_path)
    required = ["PWGTP", "AGEP", "SEX", "PERNP"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"person file missing required columns {missing}: {person_path}")

    puma_col = "PUMA20" if "PUMA20" in df.columns else "PUMA" if "PUMA" in df.columns else None
    if puma_col is None:
        raise SystemExit(f"person file missing PUMA/PUMA20: {person_path}")

    puma = df[puma_col].map(_normalize_puma).astype(object)
    w = pd.to_numeric(df["PWGTP"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    age = pd.to_numeric(df["AGEP"], errors="coerce").to_numpy(dtype=float)
    sex = pd.to_numeric(df["SEX"], errors="coerce").to_numpy(dtype=float)
    schl = pd.to_numeric(df["SCHL"], errors="coerce").to_numpy(dtype=float) if "SCHL" in df.columns else np.full(age.shape, np.nan)
    esr = pd.to_numeric(df["ESR"], errors="coerce").to_numpy(dtype=float) if "ESR" in df.columns else np.full(age.shape, np.nan)
    earn = pd.to_numeric(df["PERNP"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)

    age_b = _bin_age(age)
    sex_b = _bin_sex(sex)
    schl_b = _bin_schl_allpop(age, schl)
    esr_b = _bin_esr_allpop(age, esr)
    earn_b = bin_earn_allpop(age, earn)

    valid = (
        puma.notna().to_numpy(dtype=bool)
        & np.isfinite(w)
        & (w > 0)
        & np.isfinite(age)
        & (age_b >= 0)
        & (age_b < SHAPE[0])
        & (sex_b >= 0)
        & (sex_b < SHAPE[1])
        & (schl_b >= 0)
        & (schl_b < SHAPE[2])
        & (esr_b >= 0)
        & (esr_b < SHAPE[3])
        & (earn_b >= 0)
        & (earn_b < len(EARN_LABELS))
    )

    puma_v = puma.to_numpy(dtype=object)[valid].astype(str)
    w_v = w[valid]
    age_v = age_b[valid]
    sex_v = sex_b[valid]
    schl_v = schl_b[valid]
    esr_v = esr_b[valid]
    earn_v = earn_b[valid]

    rows: list[dict[str, Any]] = []
    for pu in sorted(set(puma_v.tolist())):
        mask = puma_v == pu
        if not bool(mask.any()):
            continue

        cell_idx = np.ravel_multi_index((age_v[mask], sex_v[mask], schl_v[mask], esr_v[mask]), dims=SHAPE)
        cell_counts = np.zeros((FINE_K,), dtype=float)
        joint_counts = np.zeros((FINE_K, len(EARN_LABELS)), dtype=float)
        np.add.at(cell_counts, cell_idx, w_v[mask])
        np.add.at(joint_counts, (cell_idx, earn_v[mask]), w_v[mask])

        total = float(cell_counts.sum())
        if total <= 0:
            continue

        puma5 = str(int(pu)).zfill(5)
        puma_uid = _canon_uid(statefp, puma5)
        nonempty = np.flatnonzero(cell_counts > 0.0)
        for ci in nonempty.tolist():
            age_i, sex_i, schl_i, esr_i = np.unravel_index(int(ci), SHAPE)
            earn_counts = joint_counts[int(ci)]
            cell_weight = float(cell_counts[int(ci)])
            p_earn = earn_counts / max(float(earn_counts.sum()), 1e-12)
            row = {
                "statefp": _canon_statefp(statefp),
                "puma": str(int(pu)),
                "puma5": puma5,
                "puma_uid": puma_uid,
                "cell_idx": int(ci),
                "age_idx": int(age_i),
                "sex_idx": int(sex_i),
                "schl_idx": int(schl_i),
                "esr_idx": int(esr_i),
                "age_label": AGE_LABELS[int(age_i)],
                "sex_label": SEX_LABELS[int(sex_i)],
                "schl_label": SCHL_LABELS[int(schl_i)],
                "esr_label": ESR_LABELS[int(esr_i)],
                "cell_weight": cell_weight,
                "cell_prob": cell_weight / total,
                "total_person_weight": total,
            }
            for j, v in enumerate(p_earn.tolist()):
                row[f"p_earn_{j:02d}"] = float(v)
            for j, v in enumerate(earn_counts.tolist()):
                row[f"count_earn_{j:02d}"] = float(v)
            rows.append(row)

    info = {
        "statefp": _canon_statefp(statefp),
        "person_path": str(person_path),
        "n_rows_raw": int(df.shape[0]),
        "n_rows_valid": int(valid.sum()),
        "n_regions_with_rows": int(len(sorted(set(puma_v.tolist())))),
        "n_conditional_rows": int(len(rows)),
    }
    return rows, info


def main() -> None:
    default_root = data_root()
    ap = argparse.ArgumentParser(prog="build_external_target_earn_conditional_v1_us")
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--statefps", default="")
    ap.add_argument("--all_states", action="store_true")
    ap.add_argument("--pums_year", type=int, default=2023)
    ap.add_argument("--pums_period", default="5-Year")
    ap.add_argument(
        "--pums_dir",
        default=str(default_root / "us" / "raw" / "pums" / "pums_2023_5-Year"),
        help="Directory containing state-level PUMS person zips.",
    )
    ap.add_argument(
        "--out_dir",
        default=str(default_root / "us" / "processed" / "external_targets"),
        help="Output directory.",
    )
    args = ap.parse_args()

    states = _parse_states(statefp=args.statefp, statefps=args.statefps, all_states=bool(args.all_states))
    pums_dir = pathlib.Path(args.pums_dir).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pums_dir.exists():
        raise SystemExit(f"pums_dir not found: {pums_dir}")

    rows: list[dict[str, Any]] = []
    state_infos: list[dict[str, Any]] = []
    for statefp in states:
        person_zip = _resolve_person_zip(pums_dir=pums_dir, statefp=statefp)
        st_rows, st_info = _aggregate_state_conditional(statefp=statefp, person_path=person_zip)
        rows.extend(st_rows)
        state_infos.append(st_info)
        print(
            f"[ok] state={statefp} n_conditional_rows={st_info['n_conditional_rows']} n_rows_valid={st_info['n_rows_valid']}",
            file=sys.stderr,
        )

    if not rows:
        raise SystemExit("No conditional rows were produced.")

    df = pd.DataFrame(rows)
    scope_tag = _scope_tag(states)
    stem = f"exttarget_earn_cond_v1_pums_{int(args.pums_year)}_puma_{scope_tag}"
    csv_path = out_dir / f"{stem}.csv"
    schema_json = out_dir / f"{stem}.schema.json"
    metadata_json = out_dir / f"{stem}.metadata.json"

    df.to_csv(csv_path, index=False)
    schema_json.write_text(
        json.dumps(
            {
                "schema": "external_target_earn_conditional_v1",
                "conditioning_variables": ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop"],
                "conditioning_shape": list(SHAPE),
                "conditioning_K": int(FINE_K),
                "target_variable": "EARN_16p_bin",
                "target_categories": EARN_LABELS,
                "conditioning_categories": {
                    "AGEP_bin": AGE_LABELS,
                    "SEX": SEX_LABELS,
                    "SCHL_allpop": SCHL_LABELS,
                    "ESR_allpop": ESR_LABELS,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    nonempty_by_puma = df.groupby("puma_uid", sort=False)["cell_idx"].count().astype(int)
    meta = {
        "schema": "external_target_earn_conditional_v1",
        "created_at": _utc_now_iso(),
        "scope": scope_tag,
        "statefps": states,
        "n_states": int(len(states)),
        "pums_year": int(args.pums_year),
        "pums_period": str(args.pums_period),
        "pums_dir": str(pums_dir),
        "outputs": {
            "csv": str(csv_path),
            "schema_json": str(schema_json),
        },
        "conditioning": {
            "variables": ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop"],
            "shape": list(SHAPE),
            "K": int(FINE_K),
        },
        "target": {
            "variable": "EARN_16p_bin",
            "categories": EARN_LABELS,
            "source_variable": "PERNP",
            "note": "Conditional earnings proxy by non-empty 4-attribute cells; age<16 or PERNP<=0 map to not_in_earnings_universe.",
        },
        "info": {
            "n_conditional_rows": int(df.shape[0]),
            "n_pumas": int(df["puma_uid"].nunique()),
            "mean_nonempty_cells_per_puma": float(nonempty_by_puma.mean()),
            "median_nonempty_cells_per_puma": float(nonempty_by_puma.median()),
            "state_summaries": state_infos,
        },
    }
    metadata_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
