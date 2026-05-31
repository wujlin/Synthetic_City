#!/usr/bin/env python3
from __future__ import annotations

"""
Build US-wide PUMA-level 5-variable person-joint distributions from PUMS.

Variables (person-level):
- age: 2 or 4 bins (configurable)
- sex: 2 bins [male, female]
- income (PINCP): 2 or 4 bins (configurable)
- education (SCHL): 2 or 4 bins (configurable)
- employment (ESR): 2 or 4 bins (configurable)

Joint size: age_bins * 2 * income_bins * schl_bins * esr_bins
Example:
- 4*2*4*4*4 = 512 (default)
- 2*2*2*2*2 = 32 (P5 coarse)
- 4*2*4*2*2 = 128 (P5 mid)

Outputs:
- puma_5var_joint_wide.csv
- puma_5var_joint_long.csv
- schema_5var.json
- heterogeneity_diagnostic.json
- run.metadata.json
"""

import argparse
import datetime as _dt
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

from synthpop.data.state_codes import STATEFP_TO_POSTAL as _STATEFP_TO_POSTAL_50


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_puma(value: Any) -> str | None:
    if value is None:
        return None
    try:
        s = str(value).strip()
        if not s:
            return None
        return str(int(float(s)))
    except Exception:
        return None


def _tvd(p: Any, q: Any) -> float:
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    return 0.5 * float(np.abs(p - q).sum())


def _find_csv_member(path: pathlib.Path) -> str:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise SystemExit(f"No CSV member found in zip: {path}")
        names = sorted(names, key=lambda n: zf.getinfo(n).file_size, reverse=True)
        return names[0]


def _resolve_person_zip(*, pums_dir: pathlib.Path, statefp: str) -> pathlib.Path:
    if statefp not in _STATEFP_TO_POSTAL_50:
        raise SystemExit(f"Unsupported statefp={statefp}")
    postal = _STATEFP_TO_POSTAL_50[statefp]
    candidates = [
        pums_dir / f"csv_p{postal}.zip",
        pums_dir / f"psam_p{statefp}.zip",
    ]
    out = next((p for p in candidates if p.exists()), None)
    if out is None:
        raise SystemExit(f"Missing person zip for state={statefp}. tried={candidates}")
    return out


def _age_edges_labels(n_bins: int) -> tuple[np.ndarray, list[str]]:
    if int(n_bins) == 2:
        return np.asarray([45.0], dtype=float), ["0-44", "45+"]
    if int(n_bins) == 4:
        return np.asarray([25.0, 45.0, 65.0], dtype=float), ["0-24", "25-44", "45-64", "65+"]
    raise ValueError(f"Unsupported age_bins={n_bins}; expected 2 or 4.")


def _bin_age(age: np.ndarray, *, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, age, side="right").astype(np.int16)


def _bin_sex(sex: np.ndarray) -> np.ndarray:
    out = np.full(sex.shape, -1, dtype=np.int16)
    out[sex == 1] = 0
    out[sex == 2] = 1
    return out


def _income_edges_labels(n_bins: int) -> tuple[np.ndarray, list[str]]:
    if int(n_bins) == 2:
        return np.asarray([50000.0], dtype=float), ["<50k", "50k+"]
    if int(n_bins) == 4:
        return np.asarray([25000.0, 50000.0, 100000.0], dtype=float), ["<25k", "25k-50k", "50k-100k", "100k+"]
    raise ValueError(f"Unsupported income_bins={n_bins}; expected 2 or 4.")


def _bin_income(inc: np.ndarray, *, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, inc, side="right").astype(np.int16)


def _schl_labels(n_bins: int) -> list[str]:
    if int(n_bins) == 2:
        return ["<SomeCollege", "SomeCollege+"]
    if int(n_bins) == 4:
        return ["<HS", "HS/GED", "SomeCollege", "BA+"]
    raise ValueError(f"Unsupported schl_bins={n_bins}; expected 2 or 4.")


def _bin_schl(schl: np.ndarray, age: np.ndarray, *, n_bins: int) -> np.ndarray:
    # PUMS SCHL rough mapping:
    # 4-bin: <HS(<=15), HS/GED(16-17), SomeCollege(18-20), BA+(>=21)
    # 2-bin: <SomeCollege(<18), SomeCollege+(>=18)
    out = np.full(schl.shape, -1, dtype=np.int16)
    m = np.isfinite(schl)
    s = schl[m]
    if int(n_bins) == 4:
        out_m = np.where(s <= 15, 0, np.where(s <= 17, 1, np.where(s <= 20, 2, 3))).astype(np.int16)
    elif int(n_bins) == 2:
        out_m = np.where(s >= 18, 1, 0).astype(np.int16)
    else:
        raise ValueError(f"Unsupported schl_bins={n_bins}; expected 2 or 4.")
    out[m] = out_m
    out[~m] = 0
    out[~np.isfinite(age)] = -1
    return out


def _esr_labels(n_bins: int) -> list[str]:
    if int(n_bins) == 2:
        return ["LaborForce", "NILF"]
    if int(n_bins) == 4:
        return ["Employed", "Unemployed", "Armed", "NILF"]
    raise ValueError(f"Unsupported esr_bins={n_bins}; expected 2 or 4.")


def _bin_esr(esr: np.ndarray, age: np.ndarray, *, n_bins: int) -> np.ndarray:
    if int(n_bins) == 4:
        # employed: 1,2 -> 0
        # unemployed: 3 -> 1
        # armed: 4,5 -> 2
        # NILF: 6 or missing/other -> 3
        out = np.full(esr.shape, 3, dtype=np.int16)
        out[(esr == 1) | (esr == 2)] = 0
        out[esr == 3] = 1
        out[(esr == 4) | (esr == 5)] = 2
        out[esr == 6] = 3
        out[np.isfinite(age) & (age < 16)] = 3
        return out
    if int(n_bins) == 2:
        # LaborForce: ESR in 1..5 ; NILF: ESR=6 or missing/other
        out = np.full(esr.shape, 1, dtype=np.int16)
        labor = (esr == 1) | (esr == 2) | (esr == 3) | (esr == 4) | (esr == 5)
        out[labor] = 0
        out[np.isfinite(age) & (age < 16)] = 1
        return out
    raise ValueError(f"Unsupported esr_bins={n_bins}; expected 2 or 4.")


def _aggregate_state(
    *,
    statefp: str,
    person_zip: pathlib.Path,
    alpha: float,
    shape: tuple[int, int, int, int, int],
    age_edges: np.ndarray,
    income_edges: np.ndarray,
    schl_bins: int,
    esr_bins: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    member = _find_csv_member(person_zip)
    usecols = ["PUMA", "PUMA20", "PWGTP", "AGEP", "SEX", "PINCP", "SCHL", "ESR"]
    with zipfile.ZipFile(person_zip) as zf, zf.open(member) as f:
        df = pd.read_csv(f, usecols=lambda c: c in set(usecols), low_memory=False)

    required = ["PWGTP", "AGEP", "SEX"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"person zip missing required columns {missing}: {person_zip}")

    puma_col = "PUMA20" if "PUMA20" in df.columns else "PUMA" if "PUMA" in df.columns else None
    if puma_col is None:
        raise SystemExit(f"person zip missing PUMA/PUMA20: {person_zip}")

    puma = df[puma_col].map(_normalize_puma).astype(object)
    w = pd.to_numeric(df["PWGTP"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    age = pd.to_numeric(df["AGEP"], errors="coerce").to_numpy(dtype=float)
    sex = pd.to_numeric(df["SEX"], errors="coerce").to_numpy(dtype=float)
    inc = pd.to_numeric(df["PINCP"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    schl = pd.to_numeric(df["SCHL"], errors="coerce").to_numpy(dtype=float) if "SCHL" in df.columns else np.full(age.shape, np.nan)
    esr = pd.to_numeric(df["ESR"], errors="coerce").to_numpy(dtype=float) if "ESR" in df.columns else np.full(age.shape, np.nan)

    age_b = _bin_age(age, edges=age_edges)
    sex_b = _bin_sex(sex)
    inc_b = _bin_income(inc, edges=income_edges)
    schl_b = _bin_schl(schl, age, n_bins=int(schl_bins))
    esr_b = _bin_esr(esr, age, n_bins=int(esr_bins))

    valid = (
        puma.notna().to_numpy(dtype=bool)
        & np.isfinite(w)
        & (w > 0)
        & np.isfinite(age)
        & (sex_b >= 0)
        & (sex_b < shape[1])
        & (age_b >= 0)
        & (age_b < shape[0])
        & (inc_b >= 0)
        & (inc_b < shape[2])
        & (schl_b >= 0)
        & (schl_b < shape[3])
        & (esr_b >= 0)
        & (esr_b < shape[4])
    )

    puma_v = puma.to_numpy(dtype=object)[valid].astype(str)
    w_v = w[valid]
    age_v = age_b[valid]
    sex_v = sex_b[valid]
    inc_v = inc_b[valid]
    schl_v = schl_b[valid]
    esr_v = esr_b[valid]

    rows: list[dict[str, Any]] = []
    for pu in sorted(set(puma_v.tolist())):
        m = puma_v == pu
        if not bool(m.any()):
            continue
        idx = np.ravel_multi_index(
            (age_v[m], sex_v[m], inc_v[m], schl_v[m], esr_v[m]),
            dims=shape,
        )
        counts = np.zeros((int(np.prod(shape)),), dtype=float)
        np.add.at(counts, idx, w_v[m])
        total = float(counts.sum())
        if total <= 0:
            continue
        # Laplace smoothing.
        sm = counts + float(alpha)
        p_joint = sm / float(sm.sum())

        tab = p_joint.reshape(shape)
        p_age = tab.sum(axis=(1, 2, 3, 4))
        p_sex = tab.sum(axis=(0, 2, 3, 4))
        p_inc = tab.sum(axis=(0, 1, 3, 4))
        p_schl = tab.sum(axis=(0, 1, 2, 4))
        p_esr = tab.sum(axis=(0, 1, 2, 3))

        puma5 = str(int(pu)).zfill(5)
        puma_uid = f"{str(statefp).zfill(2)}{puma5}"
        rows.append(
            {
                "statefp": str(statefp).zfill(2),
                "puma": str(int(pu)),
                "puma5": puma5,
                "puma_uid": puma_uid,
                "total_person_weight": total,
                "n_persons_unweighted": int(m.sum()),
                "p_joint": p_joint.astype(float),
                "p_age": p_age.astype(float),
                "p_sex": p_sex.astype(float),
                "p_income": p_inc.astype(float),
                "p_schl": p_schl.astype(float),
                "p_esr": p_esr.astype(float),
            }
        )

    info = {
        "statefp": str(statefp).zfill(2),
        "person_zip": str(person_zip),
        "n_rows_raw": int(df.shape[0]),
        "n_rows_valid": int(valid.sum()),
        "n_pumas": int(len(rows)),
    }
    return rows, info


def _to_wide_df(rows: list[dict[str, Any]], shape: tuple[int, int, int, int, int]) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    K = int(np.prod(shape))
    for r in rows:
        row = {
            "statefp": r["statefp"],
            "puma": r["puma"],
            "puma5": r["puma5"],
            "puma_uid": r["puma_uid"],
            "total_person_weight": float(r["total_person_weight"]),
            "n_persons_unweighted": int(r["n_persons_unweighted"]),
        }
        for i, v in enumerate(np.asarray(r["p_age"], dtype=float).reshape(-1)):
            row[f"p_age_{i:02d}"] = float(v)
        for i, v in enumerate(np.asarray(r["p_sex"], dtype=float).reshape(-1)):
            row[f"p_sex_{i:02d}"] = float(v)
        for i, v in enumerate(np.asarray(r["p_income"], dtype=float).reshape(-1)):
            row[f"p_income_{i:02d}"] = float(v)
        for i, v in enumerate(np.asarray(r["p_schl"], dtype=float).reshape(-1)):
            row[f"p_schl_{i:02d}"] = float(v)
        for i, v in enumerate(np.asarray(r["p_esr"], dtype=float).reshape(-1)):
            row[f"p_esr_{i:02d}"] = float(v)
        pj = np.asarray(r["p_joint"], dtype=float).reshape(-1)
        if pj.size != K:
            raise ValueError(f"p_joint size mismatch: {pj.size} vs {K}")
        for k, v in enumerate(pj):
            row[f"p_joint_{k:03d}"] = float(v)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _to_long_df(rows: list[dict[str, Any]], shape: tuple[int, int, int, int, int]) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for r in rows:
        pj = np.asarray(r["p_joint"], dtype=float).reshape(shape)
        cnt = pj * float(r["total_person_weight"])
        for a in range(shape[0]):
            for s in range(shape[1]):
                for i in range(shape[2]):
                    for e in range(shape[3]):
                        for w in range(shape[4]):
                            out.append(
                                {
                                    "statefp": r["statefp"],
                                    "puma": r["puma"],
                                    "puma_uid": r["puma_uid"],
                                    "age_bin_idx": a,
                                    "sex_bin_idx": s,
                                    "income_bin_idx": i,
                                    "schl_bin_idx": e,
                                    "esr_bin_idx": w,
                                    "prob_joint": float(pj[a, s, i, e, w]),
                                    "count_weighted": float(cnt[a, s, i, e, w]),
                                }
                            )
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(prog="build_us_puma_5var_joint")
    ap.add_argument(
        "--pums_dir",
        default="dataset/wsA_staging/us/raw/pums/pums_2023_5-Year",
        help="Directory containing US PUMS person zips (csv_p??.zip).",
    )
    ap.add_argument("--statefps", default="all", help='Comma-separated state FIPS or "all".')
    ap.add_argument("--age_bins", type=int, choices=[2, 4], default=4, help="Age bins: 2 or 4.")
    ap.add_argument("--income_bins", type=int, choices=[2, 4], default=4, help="Income bins: 2 or 4.")
    ap.add_argument("--schl_bins", type=int, choices=[2, 4], default=4, help="Education bins: 2 or 4.")
    ap.add_argument("--esr_bins", type=int, choices=[2, 4], default=4, help="Employment bins: 2 or 4.")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing alpha per joint cell.")
    ap.add_argument(
        "--heterogeneity_warn_threshold",
        type=float,
        default=0.10,
        help="Warn if mean TVD to global is below this threshold.",
    )
    ap.add_argument(
        "--out_dir",
        default="dataset/wsA_staging/us/processed/pums/puma_5var_joint_2023_5-Year",
        help="Output directory.",
    )
    args = ap.parse_args()

    if float(args.alpha) < 0:
        raise SystemExit("--alpha must be >= 0")

    pums_dir = pathlib.Path(args.pums_dir).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not pums_dir.exists():
        raise SystemExit(f"pums_dir not found: {pums_dir}")

    if str(args.statefps).lower() == "all":
        statefps = sorted(_STATEFP_TO_POSTAL_50.keys())
    else:
        raw = [x.strip() for x in str(args.statefps).split(",") if x.strip()]
        statefps = [str(x).zfill(2) for x in raw]
    bad = [s for s in statefps if s not in _STATEFP_TO_POSTAL_50]
    if bad:
        raise SystemExit(f"Unsupported statefps: {bad}")

    age_edges, age_labels = _age_edges_labels(int(args.age_bins))
    income_edges, income_labels = _income_edges_labels(int(args.income_bins))
    schl_labels = _schl_labels(int(args.schl_bins))
    esr_labels = _esr_labels(int(args.esr_bins))

    shape = (int(args.age_bins), 2, int(args.income_bins), int(args.schl_bins), int(args.esr_bins))
    all_rows: list[dict[str, Any]] = []
    by_state: list[dict[str, Any]] = []

    for sf in statefps:
        person_zip = _resolve_person_zip(pums_dir=pums_dir, statefp=sf)
        rows, info = _aggregate_state(
            statefp=sf,
            person_zip=person_zip,
            alpha=float(args.alpha),
            shape=shape,
            age_edges=age_edges,
            income_edges=income_edges,
            schl_bins=int(args.schl_bins),
            esr_bins=int(args.esr_bins),
        )
        all_rows.extend(rows)
        by_state.append(info)
        print(
            f"[ok] state={sf} pumas={info['n_pumas']} valid={info['n_rows_valid']}/{info['n_rows_raw']}",
            file=sys.stderr,
        )

    if not all_rows:
        raise SystemExit("No PUMA rows produced.")

    wide = _to_wide_df(all_rows, shape=shape).sort_values(["statefp", "puma5"]).reset_index(drop=True)
    long = _to_long_df(all_rows, shape=shape).sort_values(
        ["statefp", "puma_uid", "age_bin_idx", "sex_bin_idx", "income_bin_idx", "schl_bin_idx", "esr_bin_idx"]
    ).reset_index(drop=True)

    # Heterogeneity diagnostic.
    p_joint_cols = sorted([c for c in wide.columns if c.startswith("p_joint_")], key=lambda x: int(x.split("_")[-1]))
    weights = pd.to_numeric(wide["total_person_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pj = wide[p_joint_cols].to_numpy(dtype=float)
    wsum = float(weights.sum())
    if wsum <= 0:
        global_joint = np.mean(pj, axis=0)
    else:
        global_joint = (pj * weights.reshape(-1, 1)).sum(axis=0) / wsum
    global_joint = global_joint / max(float(global_joint.sum()), 1e-12)

    tvd_vals = []
    by_puma = []
    for i, r in wide.iterrows():
        p = pj[i]
        v = float(_tvd(p, global_joint))
        tvd_vals.append(v)
        by_puma.append(
            {
                "statefp": str(r["statefp"]).zfill(2),
                "puma": str(r["puma"]),
                "puma_uid": str(r["puma_uid"]),
                "tvd_to_global": v,
            }
        )
    arr = np.asarray(tvd_vals, dtype=float)
    hetero = {
        "mean_tvd_to_global": float(np.mean(arr)),
        "std_tvd_to_global": float(np.std(arr, ddof=0)),
        "median_tvd_to_global": float(np.median(arr)),
        "p90_tvd_to_global": float(np.quantile(arr, 0.9)),
        "max_tvd_to_global": float(np.max(arr)),
        "n_pumas": int(arr.size),
        "threshold": float(args.heterogeneity_warn_threshold),
        "pass_threshold": bool(float(np.mean(arr)) >= float(args.heterogeneity_warn_threshold)),
        "by_puma": by_puma,
    }

    wide_path = out_dir / "puma_5var_joint_wide.csv"
    long_path = out_dir / "puma_5var_joint_long.csv"
    schema_path = out_dir / "schema_5var.json"
    hetero_path = out_dir / "heterogeneity_diagnostic.json"
    meta_path = out_dir / "run.metadata.json"

    wide.to_csv(wide_path, index=False)
    long.to_csv(long_path, index=False)
    _write_json(
        schema_path,
        {
            "shape": {"age": shape[0], "sex": shape[1], "income": shape[2], "schl": shape[3], "esr": shape[4]},
            "age_bins": age_labels,
            "sex_bins": ["male", "female"],
            "income_bins": income_labels,
            "schl_bins": schl_labels,
            "esr_bins": esr_labels,
            "joint_dim": int(np.prod(shape)),
            "laplace_alpha": float(args.alpha),
            "definitions": {
                "income_var": "PINCP",
                "education_var": "SCHL",
                "employment_var": "ESR",
            },
        },
    )
    _write_json(hetero_path, hetero)
    _write_json(
        meta_path,
        {
            "created_utc": _utc_now_iso(),
            "pums_dir": str(pums_dir),
            "statefps": statefps,
            "n_states": int(len(statefps)),
            "n_pumas": int(wide.shape[0]),
            "n_long_rows": int(long.shape[0]),
            "bin_config": {
                "age_bins": int(args.age_bins),
                "sex_bins": 2,
                "income_bins": int(args.income_bins),
                "schl_bins": int(args.schl_bins),
                "esr_bins": int(args.esr_bins),
            },
            "outputs": {
                "wide_csv": str(wide_path),
                "long_csv": str(long_path),
                "schema_json": str(schema_path),
                "heterogeneity_json": str(hetero_path),
            },
            "by_state": by_state,
        },
    )

    if not bool(hetero["pass_threshold"]):
        print(
            f"[warn] mean TVD to global = {hetero['mean_tvd_to_global']:.4f} < threshold={args.heterogeneity_warn_threshold:.4f}",
            file=sys.stderr,
        )

    print(f"[ok] wrote: {wide_path}", file=sys.stderr)
    print(f"[ok] wrote: {long_path}", file=sys.stderr)
    print(f"[ok] wrote: {schema_path}", file=sys.stderr)
    print(f"[ok] wrote: {hetero_path}", file=sys.stderr)
    print(f"[ok] wrote: {meta_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
