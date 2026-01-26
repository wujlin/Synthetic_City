#!/usr/bin/env python3
from __future__ import annotations

"""
PoC (v0.1): TabDDPM-style conditional diffusion with *building features* injected as spatial conditions.

Core intent (PI-aligned):
- Do NOT "generate then assign buildings" as a pure post-processing step.
- Instead: treat buildings as part of the generation context and sample
  P(attrs | macro_condition, building_feature).

This PoC is a mechanism probe:
- Macro condition: PUMA one-hot (already validated in earlier PoC).
- Meso condition: building features (continuous) concatenated into condition vector.
- Micro constraints (hard rules) are not enforced here; handled in later stages.

Outputs:
- model.pt / encoder.json / train_summary.json
- samples_building.csv: per-person samples with bldg_id
- building_portrait.csv: building-level aggregates (pop_count, age/income stats)
- sample_summary.json
"""

import argparse
import json
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


def _assign_buildings_to_persons(
    *,
    persons: "Any",
    buildings: "Any",
    pairing: str,
    seed: int,
    n_tiers: int = 5,
    puma_col_person: str = "PUMA",
    puma_col_bldg: str = "puma",
    weight_col: str = "cap_proxy",
    income_col: str = "PINCP_log",
) -> "Any":
    pd = _require("pandas")
    np = _require("numpy")

    persons = persons.copy()
    buildings = buildings.copy()
    buildings[puma_col_bldg] = buildings[puma_col_bldg].map(_normalize_puma)
    persons[puma_col_person] = persons[puma_col_person].map(_normalize_puma)

    pairing = str(pairing)
    n_tiers = int(n_tiers)
    if n_tiers <= 0:
        raise ValueError(f"n_tiers must be positive, got: {n_tiers}")

    # Prepare building index per PUMA
    bldg_by_puma: dict[str, "Any"] = {}
    for puma, g in buildings.groupby(puma_col_bldg, sort=False):
        g = g.copy()
        g[weight_col] = pd.to_numeric(g[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        if pairing == "price_tier":
            if "price_tier" not in g.columns:
                raise ValueError('pairing="price_tier" requires buildings to have column "price_tier".')
            g["price_tier"] = pd.to_numeric(g["price_tier"], errors="coerce").fillna(0.0).astype(int)
        elif pairing == "quantile":
            g = g.sort_values(weight_col, ascending=True)
        bldg_by_puma[str(puma)] = g

    rng = np.random.default_rng(int(seed))
    assigned_bldg_id = np.empty((int(persons.shape[0]),), dtype=object)
    assigned_bldg_idx = np.full((int(persons.shape[0]),), -1, dtype=int)

    for puma, g_person in persons.groupby(puma_col_person, sort=False):
        puma = str(puma)
        if puma not in bldg_by_puma:
            # No buildings for this PUMA in the city subset; mark as missing.
            assigned_bldg_id[g_person.index.to_numpy(dtype=int)] = None
            assigned_bldg_idx[g_person.index.to_numpy(dtype=int)] = -1
            continue

        b = bldg_by_puma[puma]
        if b.shape[0] == 0:
            assigned_bldg_id[g_person.index.to_numpy(dtype=int)] = None
            assigned_bldg_idx[g_person.index.to_numpy(dtype=int)] = -1
            continue

        weights = b[weight_col].to_numpy(dtype=float)
        s = float(weights.sum())
        if s > 0:
            weights = weights / s
        else:
            weights = None

        b_ids = b["bldg_id"].astype(str).tolist()
        b_index = b.index.to_numpy(dtype=int)

        if pairing == "random":
            idx = rng.choice(len(b_ids), size=int(g_person.shape[0]), replace=True, p=weights)
            person_pos = g_person.index.to_numpy(dtype=int)
            assigned_bldg_id[person_pos] = [b_ids[int(j)] for j in idx.tolist()]
            assigned_bldg_idx[person_pos] = [int(b_index[int(j)]) for j in idx.tolist()]
        elif pairing == "quantile":
            # Sort persons by income, map to building cumulative capacity distribution.
            gp = g_person.sort_values(income_col, ascending=True)
            n = int(gp.shape[0])
            q = (np.arange(n) + 0.5) / max(n, 1)
            if weights is None:
                cum = np.linspace(1.0 / len(b_ids), 1.0, num=len(b_ids))
            else:
                cum = np.cumsum(weights)
            j_idx = np.searchsorted(cum, q, side="left").clip(0, len(b_ids) - 1)
            person_pos_sorted = gp.index.to_numpy(dtype=int)
            assigned_bldg_id[person_pos_sorted] = [b_ids[int(j)] for j in j_idx.tolist()]
            assigned_bldg_idx[person_pos_sorted] = [int(b_index[int(j)]) for j in j_idx.tolist()]
        elif pairing == "price_tier":
            # Affordability-aligned pairing:
            #   quantile(income)  <->  building price_tier (Q1..Qn)
            # within each tier: sample buildings weighted by capacity proxy
            tiers_b = pd.to_numeric(b["price_tier"], errors="coerce").fillna(0.0).astype(int).to_numpy(dtype=int)
            avail = np.array(sorted({int(t) for t in tiers_b.tolist() if int(t) > 0}), dtype=int)
            if avail.size == 0:
                # fallback: no tiers (should not happen); behave like random
                idx = rng.choice(len(b_ids), size=int(g_person.shape[0]), replace=True, p=weights)
                person_pos = g_person.index.to_numpy(dtype=int)
                assigned_bldg_id[person_pos] = [b_ids[int(j)] for j in idx.tolist()]
                assigned_bldg_idx[person_pos] = [int(b_index[int(j)]) for j in idx.tolist()]
            else:
                pools: dict[int, tuple["Any", "Any"]] = {}
                for t in avail.tolist():
                    pool_pos = np.where(tiers_b == int(t))[0]
                    if pool_pos.size == 0:
                        continue
                    w = None
                    if weights is not None:
                        w = weights[pool_pos].astype(float)
                        s2 = float(w.sum())
                        if s2 > 0:
                            w = w / s2
                        else:
                            w = None
                    pools[int(t)] = (pool_pos, w)

                gp = g_person.sort_values(income_col, ascending=True)
                n = int(gp.shape[0])
                q = (np.arange(n) + 0.5) / max(n, 1)
                desired = np.ceil(q * float(n_tiers)).astype(int)
                desired = np.clip(desired, 1, int(n_tiers))

                chosen_pos: list[int] = []
                for d in desired.tolist():
                    # Map to nearest available tier within this PUMA
                    nearest = int(avail[np.argmin(np.abs(avail - int(d)))])
                    pool_pos, w = pools.get(nearest, (np.arange(len(b_ids)), None))
                    j = int(rng.choice(pool_pos, p=w))
                    chosen_pos.append(j)

                person_pos_sorted = gp.index.to_numpy(dtype=int)
                assigned_bldg_id[person_pos_sorted] = [b_ids[int(j)] for j in chosen_pos]
                assigned_bldg_idx[person_pos_sorted] = [int(b_index[int(j)]) for j in chosen_pos]
        else:
            raise ValueError(f"unknown pairing: {pairing}")

    persons = persons.copy()
    persons["bldg_id"] = assigned_bldg_id.tolist()
    persons["_bldg_rowidx"] = assigned_bldg_idx.tolist()
    return persons


def main() -> None:
    pd = _require("pandas")
    np = _require("numpy")
    torch = _require("torch")

    from src.synthpop.model.diffusion_tabular import DiffusionTabularModel, TabDDPMConfig
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
    p.add_argument("--pairing", choices=["random", "quantile", "price_tier"], default="price_tier", help="How to pair PUMS persons with buildings for conditional training.")
    p.add_argument("--n_tiers", type=int, default=5, help="Number of price tiers (Q1..Qn). Used by pairing=price_tier.")
    p.add_argument("--out_dir", default="outputs/_poc_tabddpm_pums_buildingcond")
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--encoder_path", default=None)
    p.add_argument("--log_every", type=int, default=200)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
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
        "price_per_sqft",
        "price_tier",
    ]
    missing_b = [c for c in required_bcols if c not in buildings.columns]
    if missing_b:
        raise SystemExit(f"buildings_csv missing columns: {missing_b}")
    buildings["puma"] = buildings["puma"].map(_normalize_puma)

    # Prepare building feature matrix (continuous)
    feat_cols = ["footprint_area_m2", "height_m", "cap_proxy", "dist_cbd_km", "price_tier", "price_per_sqft"]
    feat = buildings[feat_cols].copy()
    feat["footprint_area_m2"] = pd.to_numeric(feat["footprint_area_m2"], errors="coerce").fillna(0.0).clip(lower=0.0)
    feat["height_m"] = pd.to_numeric(feat["height_m"], errors="coerce").fillna(0.0).clip(lower=0.0)
    feat["cap_proxy"] = pd.to_numeric(feat["cap_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)
    feat["dist_cbd_km"] = pd.to_numeric(feat["dist_cbd_km"], errors="coerce").fillna(0.0).clip(lower=0.0)
    feat["price_tier"] = pd.to_numeric(feat["price_tier"], errors="coerce").fillna(0.0).clip(lower=0.0)
    feat["price_per_sqft"] = pd.to_numeric(feat["price_per_sqft"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # log-transform scale-sensitive features
    feat_transformed = np.stack(
        [
            np.log1p(feat["footprint_area_m2"].to_numpy(dtype=np.float32)),
            feat["height_m"].to_numpy(dtype=np.float32),
            np.log1p(feat["cap_proxy"].to_numpy(dtype=np.float32)),
            feat["dist_cbd_km"].to_numpy(dtype=np.float32),
            feat["price_tier"].to_numpy(dtype=np.float32),
            np.log1p(feat["price_per_sqft"].to_numpy(dtype=np.float32)),
        ],
        axis=1,
    )
    bfeat_mean = feat_transformed.mean(axis=0)
    bfeat_std = feat_transformed.std(axis=0)
    bfeat_std = np.where(bfeat_std <= 1e-6, 1.0, bfeat_std)
    bfeat_z_all = ((feat_transformed - bfeat_mean) / bfeat_std).astype(np.float32)

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

        # Detroit building-conditioned PoC: only keep PUMAs that exist in the Detroit building subset.
        valid_pumas = set(buildings["puma"].dropna().astype(str).unique().tolist())
        df["PUMA_STR"] = df["PUMA"].astype(int).astype(str)
        df = df[df["PUMA_STR"].isin(valid_pumas)].copy()
        if df.empty:
            raise SystemExit(
                "After filtering to Detroit PUMAs, no PUMS rows remain. "
                "Check that buildings_csv has correct puma codes and the city boundary/clip is correct."
            )

        age = df["AGEP"].to_numpy(dtype=np.float32)
        income_log = df["PINCP_log"].to_numpy(dtype=np.float32)

        cont = np.stack([age, income_log], axis=1)
        cont_mean = cont.mean(axis=0)
        cont_std = cont.std(axis=0)
        cont_std = np.where(cont_std <= 1e-6, 1.0, cont_std)
        cont_z = ((cont - cont_mean) / cont_std).astype(np.float32)

        sex_oh, sex_cats = _one_hot(df["SEX"].astype(int).astype(str))
        esr_oh, esr_cats = _one_hot(df["ESR"].astype(int).astype(str))
        puma_oh, puma_cats = _one_hot(df["PUMA_STR"].astype(str))

        # x: attributes only (no PUMA in x)
        x_np = np.concatenate([cont_z, sex_oh, esr_oh], axis=1).astype(np.float32)

        # Pair PUMS persons with Detroit buildings in the same PUMA (mechanism probe).
        persons_for_pair = df[["PUMA", "PINCP_log"]].copy()
        persons_for_pair["PUMA"] = df["PUMA_STR"].astype(str).to_numpy()
        buildings_sub = buildings[buildings["puma"].isin(set(puma_cats))].copy()
        paired = _assign_buildings_to_persons(
            persons=persons_for_pair.rename(columns={"PUMA": "PUMA"}),
            buildings=buildings_sub,
            pairing=str(args.pairing),
            seed=int(args.seed),
            n_tiers=int(args.n_tiers),
            puma_col_person="PUMA",
            puma_col_bldg="puma",
            weight_col="cap_proxy",
            income_col="PINCP_log",
        )

        b_idx = paired["_bldg_rowidx"].to_numpy(dtype=int)
        if (b_idx < 0).any():
            raise SystemExit("Some persons could not be paired with buildings (missing PUMA coverage).")
        bfeat_z = bfeat_z_all[b_idx]

        # cond: macro PUMA one-hot + building feature vector
        cond_np = np.concatenate([puma_oh, bfeat_z], axis=1).astype(np.float32)

        encoder = {
            "continuous_mean": cont_mean.tolist(),
            "continuous_std": cont_std.tolist(),
            "categorical": {
                "SEX": {"categories": sex_cats},
                "ESR": {"categories": esr_cats},
            },
            "condition": {
                "PUMA": {"categories": puma_cats},
                "building_features": {
                    "columns": ["log1p_area", "height_m", "log1p_cap", "dist_cbd_km", "price_tier", "log1p_price_per_sqft"],
                    "mean": bfeat_mean.tolist(),
                    "std": bfeat_std.tolist(),
                },
                "pairing": str(args.pairing),
                "n_tiers": int(args.n_tiers),
            },
            "puma_prob": {str(k): float(v) for k, v in df["PUMA_STR"].astype(str).value_counts(normalize=True).items()},
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
                    "train_rows": int(df.shape[0]),
                    "train_metrics": train_metrics,
                    "ckpt_path": str(ckpt_path),
                    "encoder_path": str(encoder_path),
                    "pums_zip": str(zip_path),
                    "pums_member": member,
                    "buildings_csv": str(buildings_csv),
                    "pairing": str(args.pairing),
                    "n_tiers": int(args.n_tiers),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
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

    bfeat_mean = np.array(encoder["condition"]["building_features"]["mean"], dtype=np.float32)
    bfeat_std = np.array(encoder["condition"]["building_features"]["std"], dtype=np.float32)
    n_tiers = int(encoder["condition"].get("n_tiers", 5))

    # Reload buildings and build per-PUMA sampling pools
    buildings = pd.read_csv(buildings_csv, low_memory=False)
    buildings["puma"] = buildings["puma"].map(_normalize_puma)
    buildings = buildings[buildings["puma"].isin(set(puma_cats))].copy()
    if buildings.empty:
        raise SystemExit("No buildings match PUMA categories. Check buildings_csv / Detroit subset.")

    feat = buildings[feat_cols].copy()
    feat["footprint_area_m2"] = pd.to_numeric(feat["footprint_area_m2"], errors="coerce").fillna(0.0).clip(lower=0.0)
    feat["height_m"] = pd.to_numeric(feat["height_m"], errors="coerce").fillna(0.0).clip(lower=0.0)
    feat["cap_proxy"] = pd.to_numeric(feat["cap_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)
    feat["dist_cbd_km"] = pd.to_numeric(feat["dist_cbd_km"], errors="coerce").fillna(0.0).clip(lower=0.0)
    feat["price_tier"] = pd.to_numeric(feat["price_tier"], errors="coerce").fillna(0.0).clip(lower=0.0)
    feat["price_per_sqft"] = pd.to_numeric(feat["price_per_sqft"], errors="coerce").fillna(0.0).clip(lower=0.0)
    feat_transformed = np.stack(
        [
            np.log1p(feat["footprint_area_m2"].to_numpy(dtype=np.float32)),
            feat["height_m"].to_numpy(dtype=np.float32),
            np.log1p(feat["cap_proxy"].to_numpy(dtype=np.float32)),
            feat["dist_cbd_km"].to_numpy(dtype=np.float32),
            feat["price_tier"].to_numpy(dtype=np.float32),
            np.log1p(feat["price_per_sqft"].to_numpy(dtype=np.float32)),
        ],
        axis=1,
    )
    bfeat_z_all = ((feat_transformed - bfeat_mean) / bfeat_std).astype(np.float32)

    bldg_ids = buildings["bldg_id"].astype(str).to_numpy()
    bldg_puma = buildings["puma"].astype(str).to_numpy()
    bldg_weight = pd.to_numeric(buildings["cap_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    bldg_price_tier = pd.to_numeric(buildings["price_tier"], errors="coerce").fillna(0.0).astype(int).to_numpy(dtype=int)

    rng = np.random.default_rng(int(args.seed))
    puma_idx = rng.choice(len(puma_cats), size=int(args.n_samples), replace=True, p=puma_probs)
    puma_s = np.array([puma_cats[i] for i in puma_idx.tolist()], dtype=object)
    cond_puma_oh = np.eye(len(puma_cats), dtype=np.float32)[puma_idx]

    # For each PUMA, sample buildings weighted by capacity proxy
    chosen_bldg_row = np.full((int(args.n_samples),), -1, dtype=int)
    for i, puma_code in enumerate(puma_cats):
        mask = puma_s == puma_code
        if not bool(mask.any()):
            continue
        pool_idx = np.where(bldg_puma == puma_code)[0]
        if pool_idx.size == 0:
            continue
        w = bldg_weight[pool_idx].astype(float)
        s = float(w.sum())
        if s > 0:
            w = w / s
        else:
            w = None
        chosen = rng.choice(pool_idx, size=int(mask.sum()), replace=True, p=w)
        chosen_bldg_row[np.where(mask)[0]] = chosen

    if (chosen_bldg_row < 0).any():
        raise SystemExit("Some samples could not find a building in the chosen PUMA. Check PUMA coverage in buildings.")

    cond_bfeat = bfeat_z_all[chosen_bldg_row]
    cond_s_np = np.concatenate([cond_puma_oh, cond_bfeat], axis=1).astype(np.float32)
    cond_s = torch.from_numpy(cond_s_np)

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
            "bldg_id": bldg_ids[chosen_bldg_row].tolist(),
            "PUMA": puma_s.tolist(),
            "AGEP": age_s.astype(np.float32),
            "PINCP": income_s.astype(np.float32),
            "SEX": [sex_cats[i] for i in sex_s.tolist()],
            "ESR": [esr_cats[i] for i in esr_s.tolist()],
            "price_tier": bldg_price_tier[chosen_bldg_row].tolist(),
        }
    )
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
    keep_cols = ["bldg_id", "centroid_lon", "centroid_lat", "footprint_area_m2", "height_m", "cap_proxy", "dist_cbd_km", "puma", "price_per_sqft", "price_tier"]
    b_meta = buildings[keep_cols].copy()
    b_meta["bldg_id"] = b_meta["bldg_id"].astype(str)
    portrait = portrait.merge(b_meta, on="bldg_id", how="left")
    portrait.to_csv(out_dir / "building_portrait.csv", index=False)

    def _freq(series) -> dict[str, float]:
        counts = series.value_counts(normalize=True)
        return {str(k): float(v) for k, v in counts.items()}

    sample_summary = {
        "mode": "sample" if args.mode == "sample" else "train-sample",
        "n_samples": int(args.n_samples),
        "ckpt_path": str(ckpt_path),
        "encoder_path": str(encoder_path),
        "condition": {"n_tiers": int(n_tiers)},
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
            "price_tier": _freq(out_df["price_tier"].astype(str)),
        },
        "building_portrait": {
            "n_buildings_hit": int(portrait.shape[0]),
            "pop_count": _summary_stats(portrait["pop_count"].to_numpy()),
            "age_mean": _summary_stats(portrait["age_mean"].to_numpy()),
            "income_p50": _summary_stats(portrait["income_p50"].to_numpy()),
        },
    }
    (out_dir / "sample_summary.json").write_text(json.dumps(sample_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_dir}/samples_building.csv, {out_dir}/building_portrait.csv, {out_dir}/sample_summary.json")


if __name__ == "__main__":
    main()
