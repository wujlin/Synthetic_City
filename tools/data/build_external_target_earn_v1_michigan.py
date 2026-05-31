#!/usr/bin/env python3
from __future__ import annotations

"""
Build a Michigan PUMA-level PUMS target for the external earnings proxy experiment.
"""

import argparse
import json
import pathlib
import sys
import zipfile
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.build_external_target_v1_michigan import _normalize_puma, _resolve_person_zip, _utc_now_iso
from tools.data.external_earn_v1_schema import EARN_LABELS, bin_earn_allpop
from tools.model.train_us_puma_5var_diffusion import _canon_puma5, _canon_statefp, _canon_uid, _canon_uid_loose


def _find_csv_member(path: pathlib.Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise SystemExit(f"No CSV member found in zip: {path}")
        names = sorted(names, key=lambda n: zf.getinfo(n).file_size, reverse=True)
        return names[0]


def _load_person_df(person_path: pathlib.Path) -> pd.DataFrame:
    usecols = ["PUMA", "PUMA20", "PWGTP", "AGEP", "PERNP"]
    if person_path.suffix.lower() == ".zip":
        member = _find_csv_member(person_path)
        with zipfile.ZipFile(person_path) as zf, zf.open(member) as f:
            return pd.read_csv(f, usecols=lambda c: c in set(usecols), low_memory=False)
    return pd.read_csv(person_path, usecols=lambda c: c in set(usecols), low_memory=False)


def _aggregate_state(*, statefp: str, person_path: pathlib.Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = _load_person_df(person_path)
    required = ["PWGTP", "AGEP", "PERNP"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"person file missing required columns {missing}: {person_path}")

    puma_col = "PUMA20" if "PUMA20" in df.columns else "PUMA" if "PUMA" in df.columns else None
    if puma_col is None:
        raise SystemExit(f"person file missing PUMA/PUMA20: {person_path}")

    puma = df[puma_col].map(_normalize_puma).astype(object)
    w = pd.to_numeric(df["PWGTP"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    age = pd.to_numeric(df["AGEP"], errors="coerce").to_numpy(dtype=float)
    earn = pd.to_numeric(df["PERNP"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    earn_b = bin_earn_allpop(age, earn)

    valid = (
        puma.notna().to_numpy(dtype=bool)
        & np.isfinite(w)
        & (w > 0)
        & np.isfinite(age)
        & (earn_b >= 0)
        & (earn_b < len(EARN_LABELS))
    )

    puma_v = puma.to_numpy(dtype=object)[valid].astype(str)
    w_v = w[valid]
    earn_v = earn_b[valid]

    rows: list[dict[str, Any]] = []
    for pu in sorted(set(puma_v.tolist())):
        mask = puma_v == pu
        counts = np.zeros((len(EARN_LABELS),), dtype=float)
        np.add.at(counts, earn_v[mask], w_v[mask])
        total = float(counts.sum())
        if total <= 0:
            continue
        probs = counts / total
        puma5 = str(int(pu)).zfill(5)
        puma_uid = _canon_uid(statefp, puma5)
        rows.append(
            {
                "statefp": _canon_statefp(statefp),
                "puma": str(int(pu)),
                "puma5": puma5,
                "puma_uid": puma_uid,
                "total_person_weight": total,
                "n_persons_unweighted": int(mask.sum()),
                "p_earn": probs.astype(float),
                "count_earn": counts.astype(float),
            }
        )

    info = {
        "statefp": _canon_statefp(statefp),
        "person_path": str(person_path),
        "n_rows_raw": int(df.shape[0]),
        "n_rows_valid": int(valid.sum()),
        "n_pumas": int(len(rows)),
    }
    return rows, info


def _condition_alignment(*, rows: list[dict[str, Any]], condition_csv: pathlib.Path) -> dict[str, Any]:
    cond = pd.read_csv(condition_csv, low_memory=False)
    if "puma_uid" not in cond.columns:
        raise SystemExit(f"condition_csv missing puma_uid: {condition_csv}")
    cond["puma_uid"] = cond["puma_uid"].map(_canon_uid_loose)
    cond["target"] = pd.to_numeric(cond["target"], errors="coerce").fillna(0.0)
    cond = cond[cond["variable"].astype(str) == "EARN_16p_bin"].copy()

    target_records = []
    for r in rows:
        for i, cat in enumerate(EARN_LABELS):
            target_records.append({"puma_uid": r["puma_uid"], "category": cat, "prob": float(r["p_earn"][i])})
    tgt = pd.DataFrame(target_records)

    groups = sorted(set(tgt["puma_uid"]) & set(cond["puma_uid"]))
    by_group: dict[str, float] = {}
    for g in groups:
        tt = tgt[tgt["puma_uid"] == g].groupby("category", sort=False)["prob"].sum()
        cc = cond[cond["puma_uid"] == g].groupby("category", sort=False)["target"].sum()
        cc = cc / float(cc.sum()) if float(cc.sum()) > 0 else cc
        cats = sorted(set(tt.index.tolist()) | set(cc.index.tolist()))
        p = np.asarray([float(tt.get(k, 0.0)) for k in cats], dtype=float)
        q = np.asarray([float(cc.get(k, 0.0)) for k in cats], dtype=float)
        by_group[g] = 0.5 * float(np.abs(p - q).sum())
    vals = list(by_group.values())
    return {
        "condition_csv": str(condition_csv),
        "geography_key": "puma_uid",
        "variables": {
            "EARN_16p_bin": {
                "mean_tvd": float(np.mean(vals)) if vals else None,
                "max_tvd": float(np.max(vals)) if vals else None,
                "n_groups": int(len(vals)),
                "by_group": by_group,
            }
        },
    }


def main() -> None:
    from synthpop.paths import data_root as default_data_root

    default_data = pathlib.Path(default_data_root()).expanduser().resolve()
    ap = argparse.ArgumentParser(prog="build_external_target_earn_v1_michigan")
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--pums_year", type=int, default=2023)
    ap.add_argument("--pums_period", default="5-Year")
    ap.add_argument("--pums_dir", default=str(default_data / "detroit" / "raw" / "pums" / "pums_2023_5-Year"))
    ap.add_argument("--person_path", default=None, help="Optional direct path to csv_pmi.zip or psam_p26.csv.")
    ap.add_argument("--condition_csv", default=None)
    ap.add_argument("--out_dir", default=str(default_data / "detroit" / "processed" / "external_targets"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    statefp = _canon_statefp(args.statefp)
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    person_path = pathlib.Path(args.person_path).expanduser().resolve() if args.person_path else _resolve_person_zip(
        pums_dir=pathlib.Path(args.pums_dir).expanduser().resolve(),
        statefp=statefp,
    )
    if not person_path.exists():
        raise SystemExit(f"person_path not found: {person_path}")

    rows, info = _aggregate_state(statefp=statefp, person_path=person_path)
    if not rows:
        raise SystemExit("No PUMA rows were produced.")

    stem = f"exttarget_earn_v1_pums_{int(args.pums_year)}_puma_state{statefp}_michigan"
    wide_csv = out_dir / f"{stem}.csv"
    long_csv = out_dir / f"{stem}_long.csv"
    schema_json = out_dir / f"{stem}.schema.json"
    metadata_json = out_dir / f"{stem}.metadata.json"
    for p in [wide_csv, long_csv, schema_json, metadata_json]:
        if p.exists() and not args.overwrite:
            raise SystemExit(f"output exists: {p} (use --overwrite)")

    wide_rows = []
    long_rows = []
    for r in rows:
        row = {
            "statefp": r["statefp"],
            "puma": r["puma"],
            "puma5": r["puma5"],
            "puma_uid": r["puma_uid"],
            "total_person_weight": float(r["total_person_weight"]),
            "n_persons_unweighted": int(r["n_persons_unweighted"]),
        }
        for i, v in enumerate(np.asarray(r["p_earn"], dtype=float).reshape(-1)):
            row[f"p_earn_{i:02d}"] = float(v)
        for i, v in enumerate(np.asarray(r["count_earn"], dtype=float).reshape(-1)):
            row[f"count_earn_{i:02d}"] = float(v)
            long_rows.append(
                {
                    "statefp": r["statefp"],
                    "puma": r["puma"],
                    "puma_uid": r["puma_uid"],
                    "variable": "EARN_16p_bin",
                    "category": EARN_LABELS[i],
                    "prob": float(r["p_earn"][i]),
                    "count_weighted": float(v),
                }
            )
        wide_rows.append(row)

    pd.DataFrame(wide_rows).to_csv(wide_csv, index=False)
    pd.DataFrame(long_rows).to_csv(long_csv, index=False)
    schema_json.write_text(
        json.dumps(
            {
                "schema": "external_target_earn_v1",
                "variable_order": ["EARN_16p_bin"],
                "shape": [len(EARN_LABELS)],
                "K": len(EARN_LABELS),
                "categories": {"EARN_16p_bin": EARN_LABELS},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    meta: dict[str, Any] = {
        "schema": "external_target_earn_v1",
        "created_at": _utc_now_iso(),
        "statefp": statefp,
        "pums_year": int(args.pums_year),
        "pums_period": str(args.pums_period),
        "person_path": str(person_path),
        "outputs": {
            "wide_csv": str(wide_csv),
            "long_csv": str(long_csv),
            "schema_json": str(schema_json),
        },
        "variable": {
            "EARN_16p_bin": {
                "source_variable": "PERNP",
                "categories": EARN_LABELS,
                "note": "All-population earnings proxy: age<16 or PERNP<=0 map to not_in_earnings_universe.",
            }
        },
        "info": info,
    }
    if args.condition_csv:
        cond_path = pathlib.Path(args.condition_csv).expanduser().resolve()
        if not cond_path.exists():
            raise SystemExit(f"condition_csv not found: {cond_path}")
        meta["condition_alignment"] = _condition_alignment(rows=rows, condition_csv=cond_path)
    metadata_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
