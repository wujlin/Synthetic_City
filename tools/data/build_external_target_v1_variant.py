#!/usr/bin/env python3
from __future__ import annotations

"""
Project external-target v1 PUMS-derived joints into a named refinement-ablation variant schema.
"""

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.build_external_target_v1_michigan import (
    AGE_LABELS,
    ESR_LABELS,
    SCHL_LABELS,
    SEX_LABELS,
    SHAPE,
    _utc_now_iso,
)
from tools.data.external_v1_variant_presets import get_variant_spec
from tools.model.train_us_puma_5var_diffusion import _canon_puma5, _canon_statefp, _canon_uid, _canon_uid_loose


FULL_VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop"]
FULL_CATEGORIES = {
    "AGEP_bin": AGE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LABELS,
    "ESR_allpop": ESR_LABELS,
}


def _build_index_maps(spec_name: str) -> dict[str, np.ndarray]:
    spec = get_variant_spec(spec_name)
    out: dict[str, np.ndarray] = {}
    for var in FULL_VARIABLE_ORDER:
        dst_labels = spec.categories[var]
        dst_index = {label: i for i, label in enumerate(dst_labels)}
        mapping = spec.mappings[var]
        out[var] = np.asarray([dst_index[mapping[src]] for src in FULL_CATEGORIES[var]], dtype=np.int16)
    return out


def _aggregate_joint(p_old: np.ndarray, *, spec_name: str) -> np.ndarray:
    spec = get_variant_spec(spec_name)
    maps = _build_index_maps(spec_name)
    old_tab = np.asarray(p_old, dtype=np.float64).reshape(SHAPE)
    new_shape = tuple(spec.shape)
    new_tab = np.zeros(new_shape, dtype=np.float64)
    for ai in range(SHAPE[0]):
        an = int(maps["AGEP_bin"][ai])
        for si in range(SHAPE[1]):
            sn = int(maps["SEX"][si])
            for qi in range(SHAPE[2]):
                qn = int(maps["SCHL_allpop"][qi])
                for ei in range(SHAPE[3]):
                    en = int(maps["ESR_allpop"][ei])
                    new_tab[an, sn, qn, en] += old_tab[ai, si, qi, ei]
    new_tab = new_tab / max(float(new_tab.sum()), 1e-12)
    return new_tab.reshape(-1)


def _condition_alignment(*, wide: pd.DataFrame, condition_csv: pathlib.Path, spec_name: str) -> dict[str, Any]:
    spec = get_variant_spec(spec_name)
    cond = pd.read_csv(condition_csv, low_memory=False)
    if "puma_uid" not in cond.columns:
        raise SystemExit(f"condition_csv missing puma_uid: {condition_csv}")
    cond["puma_uid"] = cond["puma_uid"].map(_canon_uid_loose)
    cond["target"] = pd.to_numeric(cond["target"], errors="coerce").fillna(0.0)

    target_records: list[dict[str, Any]] = []
    for r in wide.to_dict(orient="records"):
        uid = _canon_uid(r["statefp"], r.get("puma5", r["puma"]))
        for var in spec.variable_order:
            labels = spec.categories[var]
            prefix = {
                "AGEP_bin": "p_age",
                "SEX": "p_sex",
                "SCHL_allpop": "p_schl",
                "ESR_allpop": "p_esr",
            }[var]
            for i, cat in enumerate(labels):
                target_records.append({"puma_uid": uid, "variable": var, "category": cat, "prob": float(r[f"{prefix}_{i:02d}"])})
    tgt = pd.DataFrame(target_records)

    out: dict[str, Any] = {"condition_csv": str(condition_csv), "geography_key": "puma_uid", "variables": {}}
    for var in spec.variable_order:
        t_var = tgt[tgt["variable"] == var].copy()
        c_var = cond[cond["variable"] == var].copy()
        groups = sorted(set(t_var["puma_uid"]) & set(c_var["puma_uid"]))
        vals: list[float] = []
        by_group: dict[str, float] = {}
        for g in groups:
            tt = t_var[t_var["puma_uid"] == g].groupby("category", sort=False)["prob"].sum()
            cc = c_var[c_var["puma_uid"] == g].groupby("category", sort=False)["target"].sum()
            cc = cc / float(cc.sum()) if float(cc.sum()) > 0 else cc
            cats = sorted(set(tt.index.tolist()) | set(cc.index.tolist()))
            p = np.asarray([float(tt.get(k, 0.0)) for k in cats], dtype=float)
            q = np.asarray([float(cc.get(k, 0.0)) for k in cats], dtype=float)
            tvd = 0.5 * float(np.abs(p - q).sum())
            vals.append(tvd)
            by_group[g] = tvd
        out["variables"][var] = {
            "mean_tvd": float(np.mean(vals)) if vals else None,
            "max_tvd": float(np.max(vals)) if vals else None,
            "n_groups": int(len(vals)),
            "by_group": by_group,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_external_target_v1_variant")
    ap.add_argument("--joint_wide_csv", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--condition_csv", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    spec = get_variant_spec(str(args.variant))
    in_path = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"joint_wide_csv not found: {in_path}")

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = in_path.name.replace("exttarget_v1_", f"exttarget_v1_{spec.name}_").replace("_joint_wide.csv", "")
    wide_csv = out_dir / f"{stem}_joint_wide.csv"
    long_csv = out_dir / f"{stem}_joint_long.csv"
    schema_json = out_dir / f"{stem}.schema.json"
    metadata_json = out_dir / f"{stem}.metadata.json"
    if any(p.exists() for p in [wide_csv, long_csv, schema_json, metadata_json]) and not args.overwrite:
        raise SystemExit(f"output exists under {out_dir} (use --overwrite)")

    df = pd.read_csv(in_path, low_memory=False)
    req = {"statefp", "puma", "puma_uid", "total_person_weight", "n_persons_unweighted"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"joint_wide_csv missing columns: {miss}")

    p_joint_cols = [f"p_joint_{i:03d}" for i in range(int(np.prod(SHAPE)))]
    missing_joint = [c for c in p_joint_cols if c not in df.columns]
    if missing_joint:
        raise SystemExit(f"joint_wide_csv missing joint columns: {missing_joint[:5]}")

    out_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    new_shape = tuple(spec.shape)
    var_prefix = {
        "AGEP_bin": "p_age",
        "SEX": "p_sex",
        "SCHL_allpop": "p_schl",
        "ESR_allpop": "p_esr",
    }
    for r in df.to_dict(orient="records"):
        p_old = np.asarray([float(r[c]) for c in p_joint_cols], dtype=np.float64)
        p_new = _aggregate_joint(p_old, spec_name=spec.name)
        tab = p_new.reshape(new_shape)
        statefp = _canon_statefp(r["statefp"])
        puma5 = _canon_puma5(r.get("puma5", r["puma"]))
        puma_uid = _canon_uid(statefp, puma5)
        row = {
            "statefp": statefp,
            "puma": str(int(puma5)) if puma5 else "",
            "puma5": puma5,
            "puma_uid": puma_uid,
            "total_person_weight": float(r["total_person_weight"]),
            "n_persons_unweighted": int(r["n_persons_unweighted"]),
        }
        marginals = {
            "AGEP_bin": tab.sum(axis=(1, 2, 3)),
            "SEX": tab.sum(axis=(0, 2, 3)),
            "SCHL_allpop": tab.sum(axis=(0, 1, 3)),
            "ESR_allpop": tab.sum(axis=(0, 1, 2)),
        }
        for var in spec.variable_order:
            for i, v in enumerate(marginals[var]):
                row[f"{var_prefix[var]}_{i:02d}"] = float(v)
        for i, v in enumerate(p_new):
            row[f"p_joint_{i:03d}"] = float(v)
        out_rows.append(row)

        for ai, age_cat in enumerate(spec.categories["AGEP_bin"]):
            for si, sex_cat in enumerate(spec.categories["SEX"]):
                for qi, schl_cat in enumerate(spec.categories["SCHL_allpop"]):
                    for ei, esr_cat in enumerate(spec.categories["ESR_allpop"]):
                        long_rows.append(
                            {
                                "statefp": statefp,
                                "puma": str(int(puma5)) if puma5 else "",
                                "puma_uid": puma_uid,
                                "AGEP_bin": age_cat,
                                "SEX": sex_cat,
                                "SCHL_allpop": schl_cat,
                                "ESR_allpop": esr_cat,
                                "prob": float(tab[ai, si, qi, ei]),
                            }
                        )

    wide = pd.DataFrame(out_rows)
    long = pd.DataFrame(long_rows)
    wide.to_csv(wide_csv, index=False)
    long.to_csv(long_csv, index=False)

    schema = {
        "schema": f"external_target_v1_{spec.name}",
        "variant": spec.name,
        "variable_order": spec.variable_order,
        "shape": spec.shape,
        "K": spec.K,
        "categories": spec.categories,
    }
    schema_json.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta: dict[str, Any] = {
        "schema": f"external_target_v1_{spec.name}",
        "variant": spec.name,
        "created_at": _utc_now_iso(),
        "source_joint_wide_csv": str(in_path),
        "outputs": {
            "joint_wide_csv": str(wide_csv),
            "joint_long_csv": str(long_csv),
            "schema_json": str(schema_json),
        },
        "shape": spec.shape,
        "K": spec.K,
        "variables": spec.categories,
    }
    if args.condition_csv:
        cond_path = pathlib.Path(args.condition_csv).expanduser().resolve()
        if not cond_path.exists():
            raise SystemExit(f"condition_csv not found: {cond_path}")
        meta["condition_alignment"] = _condition_alignment(wide=wide, condition_csv=cond_path, spec_name=spec.name)
    metadata_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {wide_csv}")


if __name__ == "__main__":
    main()

