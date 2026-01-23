#!/usr/bin/env python3
"""
Minimal proof-of-concept: TabDDPM-style mixed-type tabular diffusion on ACS PUMS (person file).

Scope (KISS):
- Train on a small subset (n_rows) to validate "mixed-type diffusion + conditioning" is runnable.
- Uses continuous diffusion on a continuous vector representation:
  - continuous cols: standardized
  - categorical cols: one-hot, decoded via argmax after sampling
- Conditioning uses a one-hot vector (default: PUMA).

This is NOT the final Detroit synthesis pipeline; it is a technical feasibility probe.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import zipfile
from typing import Any


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


def main() -> None:
    pd = _require("pandas")
    np = _require("numpy")
    torch = _require("torch")

    from src.synthpop.model.diffusion_tabular import DiffusionTabularModel, TabDDPMConfig
    from src.synthpop.paths import data_root as default_data_root

    p = argparse.ArgumentParser(prog="poc_tabddpm_pums")
    p.add_argument(
        "--mode",
        choices=["train", "sample", "train-sample"],
        default="train-sample",
        help="train: only train+save ckpt; sample: only load+sample; train-sample: do both.",
    )
    p.add_argument("--data_root", default=str(default_data_root()), help="Resolved data root (Detroit layout).")
    p.add_argument("--pums_year", type=int, default=2023)
    p.add_argument("--pums_period", default="5-Year")
    p.add_argument("--statefp", default="26")
    p.add_argument("--n_rows", type=int, default=200_000)
    p.add_argument("--n_samples", type=int, default=50_000)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--device", default=None, help='e.g., "cuda", "cuda:0", "cpu" (default: auto)')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="outputs/_poc_tabddpm_pums")
    p.add_argument("--ckpt_path", default=None, help="Checkpoint path (default: <out_dir>/model.pt).")
    p.add_argument("--encoder_path", default=None, help="Encoder metadata path (default: <out_dir>/encoder.json).")
    p.add_argument("--log_every", type=int, default=200)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = pathlib.Path(args.ckpt_path).expanduser().resolve() if args.ckpt_path else (out_dir / "model.pt")
    encoder_path = (
        pathlib.Path(args.encoder_path).expanduser().resolve() if args.encoder_path else (out_dir / "encoder.json")
    )

    encoder = None
    train_metrics = None
    zip_path = None
    member = None

    if args.mode in {"train", "train-sample"}:
        data_root_path = pathlib.Path(args.data_root).expanduser().resolve()
        raw_dir = (
            data_root_path
            / "detroit"
            / "raw"
            / "pums"
            / f"pums_{args.pums_year}_{args.pums_period}"
        )
        zip_path = raw_dir / f"psam_p{args.statefp}.zip"
        if not zip_path.exists():
            raise SystemExit(f"PUMS zip not found: {zip_path}")

        member = _find_first_csv_in_zip(zip_path)
        with zipfile.ZipFile(zip_path) as zf, zf.open(member) as f:
            df = pd.read_csv(f, nrows=int(args.n_rows), low_memory=False)

        cols = ["AGEP", "SEX", "ESR", "PINCP", "PUMA"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"Missing expected columns in PUMS person file: {missing}")

        df = df[cols].copy()
        df["AGEP"] = pd.to_numeric(df["AGEP"], errors="coerce")
        df["PINCP"] = pd.to_numeric(df["PINCP"], errors="coerce")
        df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
        df["ESR"] = pd.to_numeric(df["ESR"], errors="coerce")
        df["PUMA"] = pd.to_numeric(df["PUMA"], errors="coerce")
        df = df.dropna()

        age = df["AGEP"].to_numpy(dtype=np.float32)
        income = df["PINCP"].to_numpy(dtype=np.float32)
        income = np.maximum(income, 0.0)
        income_log = np.log1p(income).astype(np.float32)

        cont = np.stack([age, income_log], axis=1)
        cont_mean = cont.mean(axis=0)
        cont_std = cont.std(axis=0)
        cont_std = np.where(cont_std <= 1e-6, 1.0, cont_std)
        cont_z = ((cont - cont_mean) / cont_std).astype(np.float32)

        sex_oh, sex_cats = _one_hot(df["SEX"].astype(int).astype(str))
        esr_oh, esr_cats = _one_hot(df["ESR"].astype(int).astype(str))
        puma_oh, puma_cats = _one_hot(df["PUMA"].astype(int).astype(str))

        x_np = np.concatenate([cont_z, sex_oh, esr_oh], axis=1).astype(np.float32)
        cond_np = puma_oh.astype(np.float32)

        encoder = {
            "continuous_mean": cont_mean.tolist(),
            "continuous_std": cont_std.tolist(),
            "categorical": {
                "SEX": {"categories": sex_cats},
                "ESR": {"categories": esr_cats},
                "PUMA": {"categories": puma_cats},
            },
            "puma_prob": {str(k): float(v) for k, v in df["PUMA"].astype(int).astype(str).value_counts(normalize=True).items()},
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
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if args.mode in {"sample", "train-sample"}:
        if encoder is None:
            if not encoder_path.exists():
                raise SystemExit(f"encoder_path not found: {encoder_path} (run --mode train first)")
            encoder = json.loads(encoder_path.read_text(encoding="utf-8"))

        model = DiffusionTabularModel(input_dim=1, cond_dim=0, seed=int(args.seed))  # overwritten by load()
        model.load(ckpt_path)

        sex_cats = list(encoder["categorical"]["SEX"]["categories"])
        esr_cats = list(encoder["categorical"]["ESR"]["categories"])
        puma_cats = list(encoder["categorical"]["PUMA"]["categories"])
        cont_mean = np.array(encoder["continuous_mean"], dtype=np.float32)
        cont_std = np.array(encoder["continuous_std"], dtype=np.float32)

        puma_prob_map = {str(k): float(v) for k, v in dict(encoder.get("puma_prob", {})).items()}
        puma_probs = np.array([puma_prob_map.get(str(c), 0.0) for c in puma_cats], dtype=float)
        if float(puma_probs.sum()) > 0:
            puma_probs = puma_probs / float(puma_probs.sum())
        else:
            puma_probs = None

        rng = np.random.default_rng(int(args.seed))
        puma_idx = rng.choice(len(puma_cats), size=int(args.n_samples), replace=True, p=puma_probs)
        cond_s_np = np.eye(len(puma_cats), dtype=np.float32)[puma_idx]
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
                "AGEP": age_s.astype(np.float32),
                "PINCP": income_s.astype(np.float32),
                "SEX": [sex_cats[i] for i in sex_s.tolist()],
                "ESR": [esr_cats[i] for i in esr_s.tolist()],
                "PUMA": [puma_cats[i] for i in puma_idx.tolist()],
            }
        )

        out_df.to_csv(out_dir / "samples.csv", index=False)

    def _freq(series) -> dict[str, float]:
        counts = series.value_counts(normalize=True)
        return {str(k): float(v) for k, v in counts.items()}

    if args.mode == "train":
        print(f"[ok] trained and saved checkpoint: {ckpt_path}")
        return

    if args.mode == "sample":
        sample_summary = {
            "mode": "sample",
            "n_samples": int(args.n_samples),
            "ckpt_path": str(ckpt_path),
            "encoder_path": str(encoder_path),
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
        }
        (out_dir / "sample_summary.json").write_text(
            json.dumps(sample_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[ok] wrote: {out_dir}/samples.csv and {out_dir}/sample_summary.json")
        return

    # train-sample
    combined = {
        "mode": "train-sample",
        "n_samples": int(args.n_samples),
        "ckpt_path": str(ckpt_path),
        "encoder_path": str(encoder_path),
        "train_metrics": train_metrics,
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
    }
    (out_dir / "poc_summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_dir}/samples.csv and {out_dir}/poc_summary.json")


if __name__ == "__main__":
    main()
