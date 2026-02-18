#!/usr/bin/env python3
"""
Exp2 Baselines: Independence + IPF-only for age×income joint (PUMA holdout).

Why this exists:
- Exp0 shows age×income dependence (copula) varies across areas.
- Exp2 uses conditional diffusion to learn these joint structures.
- For PI review / Table 1, we also want simple baselines:
  1) Independence: p(age, inc)=p(age)p(inc)
  2) IPF-only: fit a train-seed joint to target marginals via IPF (oracle marginals from holdout PUMS).

This script reproduces the Exp2 fold split (stable hash by PUMA, seed) and reports:
- TVD(independence_joint, reference_joint)
- TVD(ipf_joint, reference_joint)
- (optional) TVD(seed_joint, reference_joint) as a sanity baseline

All computations are weighted by PWGTP.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import pathlib
import sys
import zipfile
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Missing dependency: {pkg}. Install it in your conda env.") from e


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _find_first_csv_in_zip(zip_path: pathlib.Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No .csv found inside: {zip_path}")
        return names[0]


def _resolve_pums_person_zip(*, data_root: pathlib.Path, pums_year: int, pums_period: str, statefp: str) -> pathlib.Path:
    statefp = str(statefp).zfill(2)
    state_postal_lower = "mi" if statefp == "26" else None
    raw_dir = data_root / "detroit" / "raw" / "pums" / f"pums_{pums_year}_{pums_period}"
    candidates: list[pathlib.Path] = [raw_dir / f"psam_p{statefp}.zip"]
    if state_postal_lower is not None:
        candidates.append(raw_dir / f"csv_p{state_postal_lower}i.zip")  # csv_pmi.zip
        candidates.append(raw_dir / f"csv_p{state_postal_lower}.zip")
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit(f"PUMS person zip not found. Tried: {candidates}")


def _age_to_p12_idx(age: int) -> int:
    if age < 0:
        age = 0
    if age <= 4:
        return 0
    if age <= 9:
        return 1
    if age <= 14:
        return 2
    if age <= 17:
        return 3
    if age <= 19:
        return 4
    if age == 20:
        return 5
    if age == 21:
        return 6
    if age <= 24:
        return 7
    if age <= 29:
        return 8
    if age <= 34:
        return 9
    if age <= 39:
        return 10
    if age <= 44:
        return 11
    if age <= 49:
        return 12
    if age <= 54:
        return 13
    if age <= 59:
        return 14
    if age <= 61:
        return 15
    if age <= 64:
        return 16
    if age <= 66:
        return 17
    if age <= 69:
        return 18
    if age <= 74:
        return 19
    if age <= 79:
        return 20
    if age <= 84:
        return 21
    return 22


def _stable_hash_fold(values: list[str], *, n_folds: int, seed: int) -> dict[str, int]:
    import hashlib

    out: dict[str, int] = {}
    for v in values:
        h = hashlib.sha1((str(seed) + "::" + str(v)).encode("utf-8")).hexdigest()
        out[str(v)] = int(h[:8], 16) % int(n_folds)
    return out


def _tvd(p: Any, q: Any) -> float:
    np = _require("numpy")
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p - q).sum())


def _bin_income_idx(*, income: Any, edges: list[float]) -> Any:
    np = _require("numpy")
    x = np.asarray(income, dtype=float)
    # right=False bins: [e[i], e[i+1])
    idx = np.searchsorted(np.asarray(edges, dtype=float), x, side="right") - 1
    idx = np.clip(idx, 0, int(len(edges) - 2)).astype(int)
    return idx


def _weighted_joint_age_income(
    *, df: Any, puma: str | None, puma_col: str, wcol: str, income_edges: list[float]
) -> Any:
    pd = _require("pandas")
    np = _require("numpy")

    d = df if puma is None else df[df[puma_col].astype(str) == str(puma)]
    if getattr(d, "empty", False):
        return None

    w = pd.to_numeric(d[wcol], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    age = pd.to_numeric(d["AGEP"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=99.0).astype(int)
    inc = pd.to_numeric(d["PINCP"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    age_idx = age.map(_age_to_p12_idx).to_numpy(dtype=int)
    inc_idx = _bin_income_idx(income=inc, edges=income_edges)
    n_age = 23
    n_inc = int(len(income_edges) - 1)
    out = np.zeros((n_age, n_inc), dtype=float)
    mask = (w > 0) & np.isfinite(w)
    if not bool(mask.any()):
        return None
    np.add.at(out, (age_idx[mask], inc_idx[mask]), w[mask])
    s = float(out.sum())
    if s <= 0 or not math.isfinite(s):
        return None
    return out / s


def _ipf_2d(*, seed_joint: Any, target_row: Any, target_col: Any, iters: int = 50, eps: float = 1e-12) -> Any:
    np = _require("numpy")
    x = np.asarray(seed_joint, dtype=float)
    r = np.asarray(target_row, dtype=float)
    c = np.asarray(target_col, dtype=float)
    if x.ndim != 2:
        raise ValueError("seed_joint must be 2D")
    if r.ndim != 1 or c.ndim != 1:
        raise ValueError("target marginals must be 1D")
    if x.shape != (r.size, c.size):
        raise ValueError(f"shape mismatch: seed {x.shape} vs row {r.size} col {c.size}")
    if r.sum() <= 0 or c.sum() <= 0:
        raise ValueError("target marginals must be non-empty")

    x = x.copy()
    x = x + float(eps)
    x = x / float(x.sum())

    r = r / float(r.sum())
    c = c / float(c.sum())

    for _ in range(int(iters)):
        rs = x.sum(axis=1)
        x = x * (r / (rs + eps)).reshape(-1, 1)
        cs = x.sum(axis=0)
        x = x * (c / (cs + eps)).reshape(1, -1)

    s = float(x.sum())
    if s <= 0 or not math.isfinite(s):
        return (r.reshape(-1, 1) * c.reshape(1, -1)).astype(float)
    return (x / s).astype(float)


def main() -> None:
    pd = _require("pandas")

    from src.synthpop.paths import data_root as default_data_root
    from src.synthpop.pipeline.detroit_v0 import make_run_id

    ap = argparse.ArgumentParser(prog="exp2_baselines_age_income")
    ap.add_argument("--data_root", default=str(default_data_root()))
    ap.add_argument("--pums_year", type=int, default=2022)
    ap.add_argument("--pums_period", default="5-Year")
    ap.add_argument("--statefp", default="26")
    ap.add_argument("--pums_person_zip", default=None, help="Optional override for PUMS person zip.")
    ap.add_argument("--n_rows", type=int, default=None, help="Optional cap for faster iteration.")
    ap.add_argument("--seed", type=int, default=0, help="Fold split seed (must match Exp2).")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--ipf_iters", type=int, default=50)
    ap.add_argument(
        "--income_edges",
        default="0,10000,25000,50000,75000,100000,150000,250000,10000000",
        help="Comma-separated income bin edges (right-open).",
    )
    ap.add_argument("--out_dir", default=None, help="Default: outputs/<run_id> under repo.")
    args = ap.parse_args()

    data_root = pathlib.Path(args.data_root).expanduser().resolve()
    out_dir = (
        pathlib.Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (_REPO_ROOT / "outputs" / make_run_id(prefix="exp2_baselines_age_income"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    income_edges = [float(x) for x in str(args.income_edges).split(",") if str(x).strip()]
    if len(income_edges) < 3:
        raise SystemExit("--income_edges must have at least 3 numbers.")

    pums_zip = (
        pathlib.Path(str(args.pums_person_zip)).expanduser().resolve()
        if args.pums_person_zip
        else _resolve_pums_person_zip(
            data_root=data_root, pums_year=int(args.pums_year), pums_period=str(args.pums_period), statefp=str(args.statefp)
        )
    )
    if not pums_zip.exists():
        raise SystemExit(f"pums_person_zip not found: {pums_zip}")
    member = _find_first_csv_in_zip(pums_zip)
    usecols = ["PUMA", "PUMA20", "PWGTP", "AGEP", "PINCP"]
    with zipfile.ZipFile(pums_zip) as zf, zf.open(member) as f:
        df = pd.read_csv(f, usecols=lambda c: c in set(usecols), low_memory=False)

    # Keep geography boundary-consistent with selected PUMS release.
    if int(args.pums_year) >= 2022:
        if "PUMA20" not in df.columns:
            raise SystemExit(f"PUMS {int(args.pums_year)} requires PUMA20, but it is missing.")
        df["PUMA"] = df["PUMA20"]
    elif "PUMA" not in df.columns:
        raise SystemExit("Legacy PUMS requires PUMA, but it is missing.")

    # Clean.
    puma_num = pd.to_numeric(df["PUMA"], errors="coerce")
    df = df[puma_num.notna() & (puma_num != -9)].copy()
    df["PUMA"] = df["PUMA"].astype(str)
    df["PWGTP"] = pd.to_numeric(df["PWGTP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["AGEP"] = pd.to_numeric(df["AGEP"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=99.0)
    df["PINCP"] = pd.to_numeric(df["PINCP"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df[df["PWGTP"] > 0].copy()
    if df.empty:
        raise SystemExit("No valid PUMS rows after cleaning.")

    if args.n_rows is not None and int(args.n_rows) > 0 and int(df.shape[0]) > int(args.n_rows):
        df = df.sample(n=int(args.n_rows), random_state=int(args.seed)).reset_index(drop=True)

    pumas = sorted(df["PUMA"].unique().tolist())
    fold_of = _stable_hash_fold(pumas, n_folds=int(args.n_folds), seed=int(args.seed))

    by_fold = {}
    for fold in range(int(args.n_folds)):
        test_pumas = {p for p, f in fold_of.items() if f == fold}
        train_df = df[~df["PUMA"].isin(test_pumas)].copy()
        test_df = df[df["PUMA"].isin(test_pumas)].copy()
        seed_joint = _weighted_joint_age_income(
            df=train_df, puma=None, puma_col="PUMA", wcol="PWGTP", income_edges=income_edges
        )
        if seed_joint is None:
            raise SystemExit(f"Failed to compute seed_joint for fold={fold} (train_df empty?)")

        per_puma = {}
        for puma in sorted(test_pumas):
            ref_joint = _weighted_joint_age_income(
                df=test_df, puma=str(puma), puma_col="PUMA", wcol="PWGTP", income_edges=income_edges
            )
            if ref_joint is None:
                continue
            p_age = ref_joint.sum(axis=1)
            p_inc = ref_joint.sum(axis=0)
            indep_joint = (p_age.reshape(-1, 1) * p_inc.reshape(1, -1)).astype(float)
            ipf_joint = _ipf_2d(seed_joint=seed_joint, target_row=p_age, target_col=p_inc, iters=int(args.ipf_iters))

            per_puma[str(puma)] = {
                "tvd_joint_independence": _tvd(indep_joint, ref_joint),
                "tvd_joint_seed": _tvd(seed_joint, ref_joint),
                "tvd_joint_ipf": _tvd(ipf_joint, ref_joint),
            }

        def _agg(metric: str) -> dict[str, float] | None:
            vals = [per_puma[p][metric] for p in per_puma]
            if not vals:
                return None
            np = _require("numpy")
            arr = np.asarray(vals, dtype=float)
            return {"mean": float(arr.mean()), "max": float(arr.max())}

        by_fold[str(fold)] = {
            "n_test_rows": int(test_df.shape[0]),
            "n_test_pumas": int(len(test_pumas)),
            "by_puma": per_puma,
            "summary": {
                "tvd_joint_independence": _agg("tvd_joint_independence"),
                "tvd_joint_seed": _agg("tvd_joint_seed"),
                "tvd_joint_ipf": _agg("tvd_joint_ipf"),
            },
        }

    payload = {
        "created_utc": _utc_now_iso(),
        "argv": sys.argv,
        "inputs": {
            "pums_zip": str(pums_zip),
            "member": str(member),
            "pums_year": int(args.pums_year),
            "pums_period": str(args.pums_period),
            "statefp": str(args.statefp),
            "n_rows": (int(args.n_rows) if args.n_rows is not None else None),
            "seed": int(args.seed),
            "n_folds": int(args.n_folds),
            "ipf_iters": int(args.ipf_iters),
            "income_edges": income_edges,
        },
        "by_fold": by_fold,
        "note": "IPF baseline uses oracle marginals from holdout PUMS; intended as a lower-bound baseline for joint recovery given marginals.",
    }

    out_path = out_dir / "baselines_age_income_joint.json"
    _write_json(out_path, payload)
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()
