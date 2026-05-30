#!/usr/bin/env python3
from __future__ import annotations

"""
Train a latent-bottleneck regional-context model to predict EARN_16p_bin from
the existing four-attribute external full condition.

Question answered by this run:
- Does the current external demographic condition carry enough regional context
  to predict a tractable earnings proxy without directly observing earnings?
"""

import argparse
import datetime as _dt
import json
import pathlib
import random
import sys
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.model.train_us_puma_5var_diffusion import (
    _canon_puma5,
    _canon_statefp,
    _canon_uid,
    _cosine,
    _parse_hidden_dims,
    _require_torch,
    _summ,
    _tvd,
    _utc_now_iso,
    _write_json,
)
from tools.model.train_us_puma_external_v1_diffusion import _default_var_specs, _load_external_condition_matrix


def _load_earn_target(*, target_csv: pathlib.Path) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    df = pd.read_csv(target_csv, low_memory=False)
    req = {"statefp", "puma", "puma_uid", "total_person_weight"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"target_csv missing columns: {miss}")
    df["statefp"] = df["statefp"].map(_canon_statefp)
    df["puma5"] = df["puma"].map(_canon_puma5)
    df["puma"] = df["puma5"].map(lambda x: str(int(x)) if x else "")
    df["puma_uid"] = df.apply(lambda r: _canon_uid(r["statefp"], r["puma5"]), axis=1)
    p_cols = sorted([c for c in df.columns if c.startswith("p_earn_")], key=lambda x: int(x.split("_")[-1]))
    if not p_cols:
        raise SystemExit(f"target_csv has no p_earn_* columns: {target_csv}")
    p = df[p_cols].to_numpy(dtype=np.float32)
    p = np.clip(p, 0.0, None)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    ids = df["puma_uid"].astype(str).tolist()
    return df, p, ids, p_cols


class RegionalContextEarnModel:
    def __init__(
        self,
        *,
        cond_dim: int,
        latent_dim: int,
        encoder_hidden_dims: tuple[int, ...],
        head_hidden_dims: tuple[int, ...],
        out_dim: int,
        lr: float,
        weight_decay: float,
        seed: int,
    ) -> None:
        torch = _require_torch()
        nn = torch.nn
        torch.manual_seed(int(seed))
        self.encoder = self._make_mlp(in_dim=int(cond_dim), hidden_dims=encoder_hidden_dims, out_dim=int(latent_dim), nn=nn)
        self.head = self._make_mlp(in_dim=int(latent_dim), hidden_dims=head_hidden_dims, out_dim=int(out_dim), nn=nn)
        self._modules = nn.ModuleList([self.encoder, self.head])
        self._opt = torch.optim.AdamW(self._modules.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    @staticmethod
    def _make_mlp(*, in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, nn: Any) -> Any:
        layers: list[Any] = []
        dim_in = int(in_dim)
        for dim_out in hidden_dims:
            layers.append(nn.Linear(dim_in, int(dim_out)))
            layers.append(nn.SiLU())
            dim_in = int(dim_out)
        layers.append(nn.Linear(dim_in, int(out_dim)))
        return nn.Sequential(*layers)

    def to(self, device: Any) -> None:
        self._modules.to(device)

    def train(self) -> None:
        self._modules.train()

    def eval(self) -> None:
        self._modules.eval()

    def step(self, *, cond: Any, p_true: Any) -> dict[str, float]:
        torch = _require_torch()
        self.train()
        z = self.encoder(cond)
        logits = self.head(z)
        logp = torch.log_softmax(logits, dim=1)
        loss = -(p_true * logp).sum(dim=1).mean()
        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        self._opt.step()
        return {"loss": float(loss.detach().cpu())}

    def predict(self, *, cond: Any) -> tuple[Any, Any]:
        torch = _require_torch()
        self.eval()
        with torch.no_grad():
            z = self.encoder(cond)
            logits = self.head(z)
            p = torch.softmax(logits, dim=1)
        return z, p

    def save(self, path: pathlib.Path, *, payload: dict[str, Any]) -> None:
        torch = _require_torch()
        path = pathlib.Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "synthpop.external_earn_context.v0", "state_dict": self._modules.state_dict(), **payload}, path)


def main() -> None:
    ap = argparse.ArgumentParser(prog="train_external_earn_from_context")
    ap.add_argument("--target_csv", required=True)
    ap.add_argument("--condition_csv", required=True)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--encoder_hidden_dims", default="256,256")
    ap.add_argument("--head_hidden_dims", default="128,128")
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_final_model", action="store_true")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    torch = _require_torch()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    target_csv = pathlib.Path(args.target_csv).expanduser().resolve()
    condition_csv = pathlib.Path(args.condition_csv).expanduser().resolve()
    for p in [target_csv, condition_csv]:
        if not p.exists():
            raise SystemExit(f"path not found: {p}")

    run_id = f"_us_puma_external_earn_from_context_{_dt.datetime.now(_dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else (_REPO_ROOT / "outputs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    df, p_true, ids, p_cols = _load_earn_target(target_csv=target_csv)
    is_mi = (df["statefp"] == "26").to_numpy(dtype=bool)
    train_idx = np.where(~is_mi)[0]
    test_idx = np.where(is_mi)[0]
    if train_idx.size == 0 or test_idx.size == 0:
        raise SystemExit("invalid leave_mi_out split")

    var_specs = _default_var_specs()
    cond, _, cond_meta = _load_external_condition_matrix(condition_csv=condition_csv, ids=ids, var_specs=var_specs)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model = RegionalContextEarnModel(
        cond_dim=int(cond.shape[1]),
        latent_dim=int(args.latent_dim),
        encoder_hidden_dims=_parse_hidden_dims(args.encoder_hidden_dims),
        head_hidden_dims=_parse_hidden_dims(args.head_hidden_dims),
        out_dim=int(p_true.shape[1]),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    model.to(device)

    cond_train = torch.from_numpy(cond[train_idx].astype(np.float32)).to(device)
    p_train = torch.from_numpy(p_true[train_idx].astype(np.float32)).to(device)

    train_metrics: list[dict[str, float]] = []
    n_train = int(train_idx.size)
    bs = int(args.batch_size)
    for epoch in range(1, int(args.epochs) + 1):
        order = np.random.permutation(n_train)
        last_stats: dict[str, float] | None = None
        for start in range(0, n_train, bs):
            idx = order[start : start + bs]
            idx_t = torch.from_numpy(idx).to(device=device, dtype=torch.long)
            last_stats = model.step(cond=cond_train[idx_t], p_true=p_train[idx_t])
        if last_stats is not None and (epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(args.epochs)):
            rec = {"epoch": float(epoch), **last_stats}
            train_metrics.append(rec)
            print(f"[train] epoch={epoch} loss={rec['loss']:.6f}")

    cond_test_t = torch.from_numpy(cond[test_idx].astype(np.float32)).to(device)
    z_test, p_pred_t = model.predict(cond=cond_test_t)
    p_pred = p_pred_t.detach().cpu().numpy().astype(np.float64)
    z_norm = np.linalg.norm(z_test.detach().cpu().numpy(), axis=1)

    tvd_vals = []
    cos_vals = []
    mae_vals = []
    train_mean = np.asarray(p_true[train_idx], dtype=np.float64).mean(axis=0)
    train_mean = train_mean / max(float(train_mean.sum()), 1e-12)
    baseline_tvd_vals = []
    for j, idx in enumerate(test_idx):
        p_t = np.asarray(p_true[idx], dtype=np.float64)
        p_h = np.asarray(p_pred[j], dtype=np.float64)
        tvd_vals.append(_tvd(p_h, p_t))
        cos_vals.append(_cosine(p_h, p_t))
        mae_vals.append(float(np.abs(p_h - p_t).mean()))
        baseline_tvd_vals.append(_tvd(train_mean, p_t))

    summary = {
        "earn_from_context": {
            "tvd_earn": _summ(tvd_vals),
            "cosine_earn": _summ(cos_vals),
            "mae_earn": _summ(mae_vals),
            "latent_norm": _summ(z_norm.tolist()),
        },
        "baselines": {
            "train_mean_earn": {
                "tvd_earn": _summ(baseline_tvd_vals),
            }
        },
    }

    saved_checkpoints: list[str] = []
    if bool(args.save_final_model):
        ckpt = out_dir / "checkpoints" / "external_earn_from_context" / "leave_mi_out" / "final.pt"
        model.save(
            ckpt,
            payload={
                "cond_dim": int(cond.shape[1]),
                "latent_dim": int(args.latent_dim),
                "encoder_hidden_dims": list(_parse_hidden_dims(args.encoder_hidden_dims)),
                "head_hidden_dims": list(_parse_hidden_dims(args.head_hidden_dims)),
                "out_dim": int(p_true.shape[1]),
                "target_columns": p_cols,
            },
        )
        saved_checkpoints.append(str(ckpt))

    run_summary = {
        "created_utc": _utc_now_iso(),
        "target_csv": str(target_csv),
        "condition_csv": str(condition_csv),
        "n_rows_total": int(df.shape[0]),
        "n_train": int(train_idx.size),
        "n_test_mi": int(test_idx.size),
        "cond_dim": int(cond.shape[1]),
        "latent_dim": int(args.latent_dim),
        "target_dim": int(p_true.shape[1]),
        "encoder_hidden_dims": list(_parse_hidden_dims(args.encoder_hidden_dims)),
        "head_hidden_dims": list(_parse_hidden_dims(args.head_hidden_dims)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "device": str(device),
        "condition_meta": cond_meta,
        "saved_checkpoints": saved_checkpoints,
        "results": summary,
    }
    _write_json(out_dir / "run_summary.json", run_summary)
    _write_json(out_dir / "metrics" / "earn_from_context_summary.json", summary)
    _write_json(out_dir / "metrics" / "training_curve.json", {"train_metrics": train_metrics})
    print(f"[ok] wrote: {out_dir}")


if __name__ == "__main__":
    main()
