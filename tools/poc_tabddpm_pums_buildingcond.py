#!/usr/bin/env python3
from __future__ import annotations

"""
PoC (v0.1, Scheme B): TabDDPM-style conditional diffusion on PUMS + explicit building allocation.

Core intent (PI review aligned, Scheme B):
- The diffusion model learns attribute joint structure from PUMS only:
  P(attrs | macro_geo_context) (macro context = PUMA for this PoC).
- Spatial anchoring is implemented as an explicit, parameterizable allocator:
  f(attrs, group) -> building, enabling clean ablations and reviewable assumptions.

This PoC is a mechanism probe:
- Macro condition: PUMA one-hot (already validated in earlier PoC).
- Meso anchoring: explicit building allocation within the same PUMA (not trained).
- Micro constraints: minimal rule checks are reported in metrics (no projection in this PoC).

Outputs:
- model.pt / encoder.json / train_summary.json
- samples_building.csv: per-person samples with bldg_id
- building_portrait.csv: building-level aggregates (pop_count, age/income stats)
- sample_summary.json
- metrics/stats_metrics.json: marginal TVD + key associations + rule violations
"""

import argparse
import datetime as _dt
import json
import os
import pathlib
import random
import sys
import zipfile
from typing import Any

# Allow running as a plain script without installing the repo.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as e:
        raise RuntimeError(f"Missing dependency: {pkg}. Install it in your conda env.") from e


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _runs_dir(data_root_path: pathlib.Path) -> pathlib.Path:
    p = data_root_path / "detroit" / "outputs" / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _latest_pointer_path(runs_dir: pathlib.Path) -> pathlib.Path:
    return runs_dir / "_latest_poc_tabddpm_pums_buildingcond.json"


def _write_latest_pointer(*, runs_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    path = _latest_pointer_path(runs_dir)
    path.write_text(
        json.dumps({"out_dir": str(out_dir), "updated_utc": _utc_now_iso()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_latest_pointer(*, runs_dir: pathlib.Path) -> pathlib.Path | None:
    path = _latest_pointer_path(runs_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        out_dir = payload.get("out_dir")
        if out_dir:
            return pathlib.Path(out_dir).expanduser().resolve()
    except Exception:
        return None
    return None


def _write_run_metadata(*, out_dir: pathlib.Path, args: argparse.Namespace, extra: dict[str, Any]) -> None:
    meta = {
        "created_utc": _utc_now_iso(),
        "argv": sys.argv,
        "script": pathlib.Path(__file__).name,
        "mode": args.mode,
        "data_root": str(pathlib.Path(args.data_root).expanduser().resolve()),
        "env": {
            "RAW_ROOT": os.environ.get("RAW_ROOT"),
            "SYNTHCITY_DATA_ROOT": os.environ.get("SYNTHCITY_DATA_ROOT"),
        },
        "args": vars(args),
        **extra,
    }
    (out_dir / "run.metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _find_first_csv_in_zip(zip_path: pathlib.Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No .csv found inside: {zip_path}")
        return names[0]


def _one_hot(series) -> tuple["Any", list[str]]:
    pd = _require("pandas")
    np = _require("numpy")
    cat = pd.Categorical(series)
    if (cat.codes < 0).any():
        raise RuntimeError("Unexpected NaNs in categorical series; clean data before one-hot.")
    depth = len(cat.categories)
    mat = np.eye(depth, dtype=np.float32)[cat.codes]
    return mat, [str(x) for x in cat.categories.tolist()]


def _softmax(x, axis: int = 1):
    np = _require("numpy")
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _summary_stats(x) -> dict[str, float]:
    np = _require("numpy")
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(x.mean()),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
    }


def _normalize_puma(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
        return str(int(float(value)))
    except Exception:
        return None


def _load_pums_zip(
    *,
    data_root: pathlib.Path,
    pums_year: int,
    pums_period: str,
    statefp: str,
) -> tuple[pathlib.Path, str]:
    statefp = str(statefp).zfill(2)
    state_postal_lower = "mi" if statefp == "26" else None
    if state_postal_lower is None:
        raise SystemExit(f"Unsupported --statefp={statefp} for Detroit PoC. v0 only supports MI (26).")

    raw_dir = data_root / "detroit" / "raw" / "pums" / f"pums_{pums_year}_{pums_period}"
    candidates = [
        raw_dir / f"psam_p{statefp}.zip",  # older naming
        raw_dir / f"csv_p{state_postal_lower}.zip",  # newer naming
    ]
    zip_path = next((p for p in candidates if p.exists()), candidates[0])
    if zip_path.exists():
        return zip_path, _find_first_csv_in_zip(zip_path)

    search_root = data_root / "detroit" / "raw" / "pums"
    patterns = [f"psam_p{statefp}.zip", f"csv_p{state_postal_lower}.zip"]
    found: list[pathlib.Path] = []
    for pat in patterns:
        found.extend(sorted(search_root.glob(f"**/{pat}")))
    found = list(dict.fromkeys([str(p) for p in found]))  # de-dup
    if len(found) == 1:
        zip_path = pathlib.Path(found[0])
        return zip_path, _find_first_csv_in_zip(zip_path)
    if len(found) > 1:
        msg = "\n".join(found[:10])
        raise SystemExit(
            "PUMS zip not found at default path and multiple candidates exist.\n"
            f"default candidates:\n  - {candidates[0]}\n  - {candidates[1]}\n"
            f"candidates (first 10):\n{msg}\n"
        )
    raise SystemExit(
        "PUMS zip not found.\n"
        f"Tried:\n  - {candidates[0]}\n  - {candidates[1]}\n"
        "Hint: download it first, or run stage-pums to copy it into detroit/raw/pums/.\n"
    )


def main() -> None:
    pd = _require("pandas")
    np = _require("numpy")
    torch = _require("torch")

    from src.synthpop.model.diffusion_tabular import DiffusionTabularModel, TabDDPMConfig
    from src.synthpop.pipeline.detroit_v0 import make_run_id
    from src.synthpop.paths import data_root as default_data_root

    p = argparse.ArgumentParser(prog="poc_tabddpm_pums_buildingcond")
    p.add_argument("--mode", choices=["train", "sample", "train-sample"], default="train-sample")
    p.add_argument("--data_root", default=str(default_data_root()))
    p.add_argument("--buildings_csv", required=True, help="Prepared buildings feature CSV (see prepare_detroit_buildings_gba.py).")
    p.add_argument("--pums_year", type=int, default=2023)
    p.add_argument("--pums_period", default="5-Year")
    p.add_argument("--statefp", default="26")
    p.add_argument("--n_rows", type=int, default=200_000)
    p.add_argument("--n_samples", type=int, default=50_000)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_frac", type=float, default=0.8, help="Train split fraction within each PUMA (rest used as holdout reference).")
    p.add_argument(
        "--allocation_method",
        choices=["random", "capacity_only", "income_price_match"],
        default="income_price_match",
        help="Explicit spatial allocation method (Scheme B).",
    )
    p.add_argument("--n_tiers", type=int, default=5, help="Number of tiers for income_price_match (default: 5).")
    p.add_argument(
        "--acs_marginals_long",
        default=None,
        help="Optional: path to ACS-derived targets_long CSV (see tools/build_acs_marginals_long.py). "
        "If provided, writes metrics/stats_metrics_acs.json.",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Default: data_root/detroit/outputs/runs/<run_id> (train) or latest run (sample).",
    )
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--encoder_path", default=None)
    p.add_argument("--log_every", type=int, default=200)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root_path = pathlib.Path(args.data_root).expanduser().resolve()
    runs_dir = _runs_dir(data_root_path)

    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    else:
        if args.mode in {"train", "train-sample"}:
            out_dir = runs_dir / make_run_id(prefix="poc_tabddpm_pums_buildingcond")
        else:
            latest = _read_latest_pointer(runs_dir=runs_dir)
            if latest is None:
                raise SystemExit(f"out_dir not provided and no latest run found at: {_latest_pointer_path(runs_dir)}")
            out_dir = latest

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = pathlib.Path(args.ckpt_path).expanduser().resolve() if args.ckpt_path else (out_dir / "model.pt")
    encoder_path = pathlib.Path(args.encoder_path).expanduser().resolve() if args.encoder_path else (out_dir / "encoder.json")

    encoder = None
    train_metrics = None

    # Load buildings
    buildings_csv = pathlib.Path(args.buildings_csv).expanduser().resolve()
    buildings = pd.read_csv(buildings_csv, low_memory=False)
    required_bcols = [
        "bldg_id",
        "puma",
        "footprint_area_m2",
        "height_m",
        "cap_proxy",
        "dist_cbd_km",
        "centroid_lon",
        "centroid_lat",
    ]
    missing_b = [c for c in required_bcols if c not in buildings.columns]
    if missing_b:
        raise SystemExit(f"buildings_csv missing columns: {missing_b}")
    buildings["puma"] = buildings["puma"].map(_normalize_puma)
    if str(args.allocation_method) == "income_price_match" and "price_tier" not in buildings.columns:
        raise SystemExit(
            'allocation_method="income_price_match" requires buildings_csv to include "price_tier". '
            "Hint: run tools/join_detroit_buildings_parcel_assessment.py to derive price_tier."
        )

    if args.mode in {"train", "train-sample"}:
        data_root_path = pathlib.Path(args.data_root).expanduser().resolve()
        zip_path, member = _load_pums_zip(
            data_root=data_root_path,
            pums_year=int(args.pums_year),
            pums_period=str(args.pums_period),
            statefp=str(args.statefp),
        )

        with zipfile.ZipFile(zip_path) as zf, zf.open(member) as f:
            df = pd.read_csv(f, nrows=int(args.n_rows), low_memory=False)

        cols = ["AGEP", "SEX", "ESR", "PINCP", "PUMA"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"Missing expected columns in PUMS person file: {missing}")
        df = df[cols].copy()

        df["AGEP"] = pd.to_numeric(df["AGEP"], errors="coerce")
        df["PINCP"] = pd.to_numeric(df["PINCP"], errors="coerce").clip(lower=0.0)
        df["PINCP_log"] = np.log1p(df["PINCP"].to_numpy(dtype=np.float32))
        df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
        df["ESR"] = pd.to_numeric(df["ESR"], errors="coerce")
        df["PUMA"] = pd.to_numeric(df["PUMA"], errors="coerce")
        df = df.dropna()

        # Detroit Scheme B PoC: only keep PUMAs that exist in the Detroit building subset.
        valid_pumas = set(buildings["puma"].dropna().astype(str).unique().tolist())
        df["PUMA_STR"] = df["PUMA"].astype(int).astype(str)
        df = df[df["PUMA_STR"].isin(valid_pumas)].copy()
        if df.empty:
            raise SystemExit(
                "After filtering to Detroit PUMAs, no PUMS rows remain. "
                "Check that buildings_csv has correct puma codes and the city boundary/clip is correct."
            )

        df = df.reset_index(drop=True)
        train_frac = float(args.train_frac)
        if not (0.0 < train_frac < 1.0):
            raise SystemExit(f"--train_frac must be in (0,1), got: {train_frac}")

        rng = np.random.default_rng(int(args.seed))
        is_train = np.zeros((int(df.shape[0]),), dtype=bool)
        for _puma, pos in df.groupby("PUMA_STR", sort=False).indices.items():
            pos = np.array(list(pos), dtype=int)
            perm = rng.permutation(pos)
            n = int(perm.size)
            if n <= 0:
                continue
            n_train = int(np.floor(train_frac * n))
            if n >= 2:
                n_train = max(1, min(n_train, n - 1))
            else:
                n_train = n
            is_train[perm[:n_train]] = True

        df_train = df[is_train].copy()
        df_ref = df[~is_train].copy()  # PUMS holdout reference
        if df_ref.empty:
            raise SystemExit("Holdout reference split is empty. Lower --train_frac or increase --n_rows.")

        age = df_train["AGEP"].to_numpy(dtype=np.float32)
        income_log = df_train["PINCP_log"].to_numpy(dtype=np.float32)

        cont = np.stack([age, income_log], axis=1)
        cont_mean = cont.mean(axis=0)
        cont_std = cont.std(axis=0)
        cont_std = np.where(cont_std <= 1e-6, 1.0, cont_std)
        cont_z = ((cont - cont_mean) / cont_std).astype(np.float32)

        sex_oh, sex_cats = _one_hot(df_train["SEX"].astype(int).astype(str))
        esr_oh, esr_cats = _one_hot(df_train["ESR"].astype(int).astype(str))
        puma_oh, puma_cats = _one_hot(df_train["PUMA_STR"].astype(str))

        # x: attributes only (no PUMA in x)
        x_np = np.concatenate([cont_z, sex_oh, esr_oh], axis=1).astype(np.float32)

        # cond: macro PUMA one-hot only (Scheme B; no building features in training)
        cond_np = puma_oh.astype(np.float32)

        encoder = {
            "format": "poc.tabddpm_pums_schemeb.v0",
            "continuous_mean": cont_mean.tolist(),
            "continuous_std": cont_std.tolist(),
            "categorical": {
                "SEX": {"categories": sex_cats},
                "ESR": {"categories": esr_cats},
            },
            "condition": {"PUMA": {"categories": puma_cats}},
            "split": {"train_frac": train_frac, "seed": int(args.seed)},
            "reference": {
                "source": "pums",
                "pums_zip": str(zip_path),
                "pums_member": member,
                "n_rows": int(args.n_rows),
                "pums_year": int(args.pums_year),
                "pums_period": str(args.pums_period),
                "statefp": str(args.statefp),
            },
            "puma_prob": {str(k): float(v) for k, v in df_train["PUMA_STR"].astype(str).value_counts(normalize=True).items()},
            "buildings_csv": str(buildings_csv),
        }
        encoder_path.write_text(json.dumps(encoder, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        x = torch.from_numpy(x_np)
        cond = torch.from_numpy(cond_np)

        cfg = TabDDPMConfig(timesteps=int(args.timesteps))
        model = DiffusionTabularModel(input_dim=x.shape[1], cond_dim=cond.shape[1], seed=int(args.seed), config=cfg)

        train_metrics = model.fit(
            x=x,
            cond=cond,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            device=args.device,
            log_every=int(args.log_every),
        )
        model.save(ckpt_path)

        (out_dir / "train_summary.json").write_text(
            json.dumps(
                {
                    "train_rows": int(df_train.shape[0]),
                    "holdout_rows": int(df_ref.shape[0]),
                    "train_frac": float(args.train_frac),
                    "train_metrics": train_metrics,
                    "ckpt_path": str(ckpt_path),
                    "encoder_path": str(encoder_path),
                    "pums_zip": str(zip_path),
                    "pums_member": member,
                    "buildings_csv": str(buildings_csv),
                    "scheme": "B",
                    "allocation_method": str(args.allocation_method),
                    "n_tiers": int(args.n_tiers),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        _write_latest_pointer(runs_dir=runs_dir, out_dir=out_dir)
        _write_run_metadata(
            out_dir=out_dir,
            args=args,
            extra={
                "run_id": out_dir.name,
                "run_dir": str(out_dir),
                "ckpt_path": str(ckpt_path),
                "encoder_path": str(encoder_path),
                "pums_zip": str(zip_path),
                "pums_member": member,
                "buildings_csv": str(buildings_csv),
            },
        )

    if args.mode == "train":
        print(f"[ok] trained and saved checkpoint: {ckpt_path}")
        return

    # ---- sample ----
    if encoder is None:
        if not encoder_path.exists():
            raise SystemExit(f"encoder_path not found: {encoder_path} (run --mode train first)")
        encoder = json.loads(encoder_path.read_text(encoding="utf-8"))

    model = DiffusionTabularModel(input_dim=1, cond_dim=0, seed=int(args.seed))  # overwritten by load()
    model.load(ckpt_path)

    sex_cats = list(encoder["categorical"]["SEX"]["categories"])
    esr_cats = list(encoder["categorical"]["ESR"]["categories"])
    cont_mean = np.array(encoder["continuous_mean"], dtype=np.float32)
    cont_std = np.array(encoder["continuous_std"], dtype=np.float32)

    puma_cats = list(encoder["condition"]["PUMA"]["categories"])
    puma_prob_map = {str(k): float(v) for k, v in dict(encoder.get("puma_prob", {})).items()}
    puma_probs = np.array([puma_prob_map.get(str(c), 0.0) for c in puma_cats], dtype=float)
    if float(puma_probs.sum()) > 0:
        puma_probs = puma_probs / float(puma_probs.sum())
    else:
        puma_probs = None

    # Keep only buildings that match the model's PUMA categories.
    buildings = buildings[buildings["puma"].isin(set(puma_cats))].copy()
    if buildings.empty:
        raise SystemExit("No buildings match PUMA categories. Check buildings_csv / Detroit subset.")

    rng = np.random.default_rng(int(args.seed))
    puma_idx = rng.choice(len(puma_cats), size=int(args.n_samples), replace=True, p=puma_probs)
    puma_s = np.array([puma_cats[i] for i in puma_idx.tolist()], dtype=object)
    cond_puma_oh = np.eye(len(puma_cats), dtype=np.float32)[puma_idx]
    cond_s = torch.from_numpy(cond_puma_oh.astype(np.float32))

    # Step 1: sample attributes (Scheme B; building not used as a training condition)
    x_s = model.sample(n=int(args.n_samples), cond=cond_s, device=args.device).numpy()

    cont_s = x_s[:, :2] * cont_std + cont_mean
    age_s = np.clip(cont_s[:, 0], 0.0, 99.0)
    income_s = np.expm1(np.clip(cont_s[:, 1], 0.0, None))

    offset = 2
    sex_dim = len(sex_cats)
    sex_logits = x_s[:, offset : offset + sex_dim]
    sex_prob = _softmax(sex_logits, axis=1)
    sex_conf = sex_prob.max(axis=1)
    sex_s = sex_prob.argmax(axis=1)
    offset += sex_dim

    esr_dim = len(esr_cats)
    esr_logits = x_s[:, offset : offset + esr_dim]
    esr_prob = _softmax(esr_logits, axis=1)
    esr_conf = esr_prob.max(axis=1)
    esr_s = esr_prob.argmax(axis=1)

    out_df = pd.DataFrame(
        {
            "person_id": np.arange(int(args.n_samples), dtype=int).tolist(),
            "puma": puma_s.tolist(),
            "AGEP": age_s.astype(np.float32),
            "PINCP": income_s.astype(np.float32),
            "SEX": [sex_cats[i] for i in sex_s.tolist()],
            "ESR": [esr_cats[i] for i in esr_s.tolist()],
        }
    )

    # Step 2: explicit building allocation (reviewable, parameterizable)
    from src.synthpop.spatial.building_allocation import allocate_to_buildings

    out_df, alloc_meta = allocate_to_buildings(
        persons=out_df,
        buildings=buildings,
        group_col="puma",
        method=str(args.allocation_method),
        income_col="PINCP",
        price_col="price_tier",
        capacity_col="cap_proxy",
        n_tiers=int(args.n_tiers),
        seed=int(args.seed),
        return_meta=True,
    )
    if "price_tier" in buildings.columns:
        out_df = out_df.merge(buildings[["bldg_id", "price_tier"]].copy(), on="bldg_id", how="left")
    out_df.to_csv(out_dir / "samples_building.csv", index=False)

    # Building-level portrait aggregates
    def _q(series, q: float) -> float:
        return float(series.quantile(q))

    portrait = (
        out_df.groupby("bldg_id", sort=False)
        .agg(
            pop_count=("bldg_id", "size"),
            age_mean=("AGEP", "mean"),
            age_p50=("AGEP", lambda s: _q(s, 0.5)),
            age_p90=("AGEP", lambda s: _q(s, 0.9)),
            income_mean=("PINCP", "mean"),
            income_p50=("PINCP", lambda s: _q(s, 0.5)),
            income_p90=("PINCP", lambda s: _q(s, 0.9)),
        )
        .reset_index()
    )
    base_cols = ["bldg_id", "centroid_lon", "centroid_lat", "footprint_area_m2", "height_m", "cap_proxy", "dist_cbd_km", "puma"]
    opt_cols = [c for c in ["price_per_sqft", "price_tier"] if c in buildings.columns]
    keep_cols = base_cols + opt_cols
    b_meta = buildings[keep_cols].copy()
    b_meta["bldg_id"] = b_meta["bldg_id"].astype(str)
    portrait = portrait.merge(b_meta, on="bldg_id", how="left")
    portrait.to_csv(out_dir / "building_portrait.csv", index=False)

    def _freq(series) -> dict[str, float]:
        counts = series.value_counts(normalize=True)
        return {str(k): float(v) for k, v in counts.items()}

    # Stats metrics (P0): synthetic vs PUMS holdout (split recreated from encoder).
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stats_metrics_path = metrics_dir / "stats_metrics.json"
    stats_metrics_acs_path = metrics_dir / "stats_metrics_acs.json"
    stats_error = None
    stats_acs_error = None
    try:
        from src.synthpop.validation.stats import compute_stats_metrics

        ref_info = dict(encoder.get("reference", {}))
        split_info = dict(encoder.get("split", {}))
        if str(ref_info.get("source", "")) != "pums":
            raise RuntimeError(f"unsupported reference source: {ref_info.get('source')}")

        pums_zip = pathlib.Path(str(ref_info["pums_zip"])).expanduser().resolve()
        pums_member = str(ref_info["pums_member"])
        nrows_ref = int(ref_info.get("n_rows", 0))
        if nrows_ref <= 0:
            nrows_ref = None  # type: ignore[assignment]

        with zipfile.ZipFile(pums_zip) as zf, zf.open(pums_member) as f:
            ref_df = pd.read_csv(f, nrows=nrows_ref, low_memory=False)

        cols = ["AGEP", "SEX", "ESR", "PINCP", "PUMA"]
        ref_df = ref_df[cols].copy()
        ref_df["AGEP"] = pd.to_numeric(ref_df["AGEP"], errors="coerce")
        ref_df["PINCP"] = pd.to_numeric(ref_df["PINCP"], errors="coerce").clip(lower=0.0)
        ref_df["SEX"] = pd.to_numeric(ref_df["SEX"], errors="coerce")
        ref_df["ESR"] = pd.to_numeric(ref_df["ESR"], errors="coerce")
        ref_df["PUMA"] = pd.to_numeric(ref_df["PUMA"], errors="coerce")
        ref_df = ref_df.dropna().reset_index(drop=True)
        ref_df["puma"] = ref_df["PUMA"].astype(int).astype(str)

        valid_pumas = set(buildings["puma"].dropna().astype(str).unique().tolist())
        ref_df = ref_df[ref_df["puma"].isin(valid_pumas)].copy().reset_index(drop=True)
        if ref_df.empty:
            raise RuntimeError("reference became empty after filtering to building PUMAs")

        train_frac = float(split_info.get("train_frac", 0.8))
        split_seed = int(split_info.get("seed", 0))
        rng_ref = np.random.default_rng(split_seed)
        is_train = np.zeros((int(ref_df.shape[0]),), dtype=bool)
        for _puma, pos in ref_df.groupby("puma", sort=False).indices.items():
            pos = np.array(list(pos), dtype=int)
            perm = rng_ref.permutation(pos)
            n = int(perm.size)
            if n <= 0:
                continue
            n_train = int(np.floor(train_frac * n))
            if n >= 2:
                n_train = max(1, min(n_train, n - 1))
            else:
                n_train = n
            is_train[perm[:n_train]] = True
        ref_holdout = ref_df[~is_train].copy()
        if ref_holdout.empty:
            raise RuntimeError("holdout reference split is empty (sample stage)")

        ref_holdout["SEX"] = ref_holdout["SEX"].astype(int).astype(str)
        ref_holdout["ESR"] = ref_holdout["ESR"].astype(int).astype(str)
        ref_holdout = ref_holdout[["puma", "AGEP", "PINCP", "SEX", "ESR"]].copy()

        syn_for_stats = out_df[["puma", "AGEP", "PINCP", "SEX", "ESR"]].copy()
        stats_metrics = compute_stats_metrics(synthetic=syn_for_stats, reference=ref_holdout, group_col="puma")
        stats_metrics_path.write_text(json.dumps(stats_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as e:
        stats_error = str(e)

    if args.acs_marginals_long:
        try:
            from src.synthpop.validation.stats import compute_stats_metrics_against_targets_long

            acs_path = pathlib.Path(args.acs_marginals_long).expanduser().resolve()
            if not acs_path.exists():
                raise RuntimeError(f"acs_marginals_long not found: {acs_path}")
            targets_long = pd.read_csv(acs_path, low_memory=False)
            syn_for_stats = out_df[["puma", "AGEP", "PINCP", "SEX", "ESR"]].copy()
            stats_acs = compute_stats_metrics_against_targets_long(
                synthetic=syn_for_stats,
                targets_long=targets_long,
                group_col="puma",
            )
            stats_metrics_acs_path.write_text(json.dumps(stats_acs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as e:
            stats_acs_error = str(e)

    sample_summary = {
        "mode": "sample" if args.mode == "sample" else "train-sample",
        "scheme": "B",
        "n_samples": int(args.n_samples),
        "ckpt_path": str(ckpt_path),
        "encoder_path": str(encoder_path),
        "allocation": {"method": str(args.allocation_method), "n_tiers": int(args.n_tiers), "meta": alloc_meta},
        "metrics": {
            "stats_metrics_path": str(stats_metrics_path),
            "stats_error": stats_error,
            "stats_metrics_acs_path": (str(stats_metrics_acs_path) if args.acs_marginals_long else None),
            "stats_acs_error": stats_acs_error,
        },
        "decode": {
            "note": "Known simplification: Gaussian DDPM outputs unconstrained logits; decoding uses argmax after softmax.",
            "confidence": {
                "SEX": _summary_stats(sex_conf),
                "ESR": _summary_stats(esr_conf),
            },
        },
        "sample_freq": {
            "SEX": _freq(out_df["SEX"].astype(str)),
            "ESR": _freq(out_df["ESR"].astype(str)),
        },
        "building_portrait": {
            "n_buildings_hit": int(portrait.shape[0]),
            "pop_count": _summary_stats(portrait["pop_count"].to_numpy()),
            "age_mean": _summary_stats(portrait["age_mean"].to_numpy()),
            "income_p50": _summary_stats(portrait["income_p50"].to_numpy()),
        },
    }
    if "price_tier" in out_df.columns:
        sample_summary["sample_freq"]["price_tier"] = _freq(out_df["price_tier"].astype(str))
    (out_dir / "sample_summary.json").write_text(json.dumps(sample_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[ok] wrote: {out_dir}/samples_building.csv, {out_dir}/building_portrait.csv, "
        f"{out_dir}/sample_summary.json, {stats_metrics_path}"
    )


if __name__ == "__main__":
    main()
