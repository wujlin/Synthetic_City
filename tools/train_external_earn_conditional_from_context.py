#!/usr/bin/env python3
from __future__ import annotations

"""
Train a person-level conditional earnings head:

    p(EARN_16p_bin | AGEP_bin, SEX, SCHL_allpop, ESR_allpop, regional context)

The regional context comes from external PUMA-level conditions. The target comes
from PUMS-derived conditional earnings distributions at the non-empty 4-attribute
cell level.
"""

import argparse
import datetime as _dt
import pathlib
import random
import sys
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.build_external_target_v1_michigan import AGE_LABELS, ESR_LABELS, SCHL_LABELS, SEX_LABELS
from tools.external_earn_v1_schema import EARN_LABELS
from tools.train_us_puma_5var_diffusion import (
    _canon_puma5,
    _canon_statefp,
    _canon_uid,
    _canon_uid_loose,
    _cosine,
    _parse_hidden_dims,
    _require_torch,
    _summ,
    _tvd,
    _utc_now_iso,
    _write_json,
)
from tools.train_us_puma_external_v1_diffusion import _default_var_specs, _load_external_condition_matrix


ATTR_DIMS = {
    "age_idx": len(AGE_LABELS),
    "sex_idx": len(SEX_LABELS),
    "schl_idx": len(SCHL_LABELS),
    "esr_idx": len(ESR_LABELS),
}


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denom = float(weights.sum())
    if denom <= 0:
        return float("nan")
    return float(np.sum(values * weights) / denom)


def _load_conditional_target(*, target_csv: pathlib.Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(target_csv, low_memory=False)
    required = {"statefp", "cell_idx", "age_idx", "sex_idx", "schl_idx", "esr_idx", "cell_weight", "cell_prob"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"target_csv missing columns: {missing}")
    cols = set(df.columns.astype(str).tolist())
    if "puma_uid" in cols:
        df["puma_uid"] = df["puma_uid"].map(_canon_uid_loose)
    elif {"statefp", "puma"} <= cols:
        df["statefp"] = df["statefp"].map(_canon_statefp)
        df["puma"] = df["puma"].map(_canon_puma5)
        df["puma_uid"] = df.apply(lambda r: _canon_uid(r["statefp"], r["puma"]), axis=1)
    else:
        raise SystemExit("target_csv missing puma_uid and cannot reconstruct it from (statefp, puma)")
    p_cols = sorted([c for c in df.columns if c.startswith("p_earn_")], key=lambda x: int(x.split("_")[-1]))
    if len(p_cols) != len(EARN_LABELS):
        raise SystemExit(f"target_csv expected {len(EARN_LABELS)} p_earn_* columns, got {len(p_cols)}")

    for col, dim in ATTR_DIMS.items():
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.any(~np.isfinite(vals)) or np.any(vals < 0) or np.any(vals >= dim):
            raise SystemExit(f"Invalid indices in {col}: expected within [0, {dim})")
        df[col] = vals.astype(int)

    probs = df[p_cols].to_numpy(dtype=np.float32)
    probs = np.clip(probs, 0.0, None)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    cell_weight = pd.to_numeric(df["cell_weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32)
    cell_prob = pd.to_numeric(df["cell_prob"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32)
    ids = df["puma_uid"].astype(str).tolist()
    return df, probs, cell_weight, cell_prob, ids


def _build_attr_features(df: pd.DataFrame) -> np.ndarray:
    n = int(df.shape[0])
    dim = sum(ATTR_DIMS.values())
    out = np.zeros((n, dim), dtype=np.float32)
    offset = 0
    for col, size in ATTR_DIMS.items():
        idx = df[col].to_numpy(dtype=int)
        out[np.arange(n), offset + idx] = 1.0
        offset += int(size)
    return out


class RegionalConditionalEarnModel:
    def __init__(
        self,
        *,
        cond_dim: int,
        attr_dim: int,
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
        self.head = self._make_mlp(in_dim=int(latent_dim + attr_dim), hidden_dims=head_hidden_dims, out_dim=int(out_dim), nn=nn)
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

    def step(self, *, cond: Any, attr: Any, p_true: Any, weight: Any) -> dict[str, float]:
        torch = _require_torch()
        self.train()
        z = self.encoder(cond)
        logits = self.head(torch.cat([z, attr], dim=1))
        logp = torch.log_softmax(logits, dim=1)
        row_loss = -(p_true * logp).sum(dim=1)
        loss = torch.sum(row_loss * weight) / torch.clamp(torch.sum(weight), min=1e-12)
        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        self._opt.step()
        return {"loss": float(loss.detach().cpu())}

    def predict(self, *, cond: Any, attr: Any) -> tuple[Any, Any]:
        torch = _require_torch()
        self.eval()
        with torch.no_grad():
            z = self.encoder(cond)
            logits = self.head(torch.cat([z, attr], dim=1))
            p = torch.softmax(logits, dim=1)
        return z, p

    def save(self, path: pathlib.Path, *, payload: dict[str, Any]) -> None:
        torch = _require_torch()
        path = pathlib.Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "synthpop.external_earn_conditional.v0", "state_dict": self._modules.state_dict(), **payload}, path)


def _train_cell_mean(df: pd.DataFrame, p_true: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(ATTR_DIMS) and p_true.shape[0], p_true.shape[1]), dtype=np.float64)
    # placeholder shape will be replaced below
    out = np.zeros((int(np.prod([v for v in ATTR_DIMS.values()])), p_true.shape[1]), dtype=np.float64)
    weight_out = np.zeros((out.shape[0],), dtype=np.float64)
    cell_idx = df["cell_idx"].to_numpy(dtype=int)
    for j, ci in enumerate(cell_idx.tolist()):
        out[ci] += float(weights[j]) * np.asarray(p_true[j], dtype=np.float64)
        weight_out[ci] += float(weights[j])
    for ci in range(out.shape[0]):
        if weight_out[ci] > 0:
            out[ci] = out[ci] / weight_out[ci]
    return out, weight_out


def main() -> None:
    ap = argparse.ArgumentParser(prog="train_external_earn_conditional_from_context")
    ap.add_argument("--target_csv", required=True)
    ap.add_argument("--condition_csv", required=True)
    ap.add_argument("--condition_mode", choices=["base4", "merged5"], default="merged5")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--encoder_hidden_dims", default="256,256")
    ap.add_argument("--head_hidden_dims", default="256,256")
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

    run_id = f"_us_puma_external_earn_conditional_{_dt.datetime.now(_dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else (_REPO_ROOT / "outputs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    df, p_true, cell_weight, cell_prob, ids = _load_conditional_target(target_csv=target_csv)
    attr_feat = _build_attr_features(df)

    is_mi = (df["statefp"].astype(str) == "26").to_numpy(dtype=bool)
    train_idx = np.where(~is_mi)[0]
    test_idx = np.where(is_mi)[0]
    if train_idx.size == 0 or test_idx.size == 0:
        raise SystemExit("invalid leave_mi_out split")

    var_specs = _default_var_specs()
    if str(args.condition_mode) == "merged5":
        var_specs = list(var_specs) + [("EARN_16p_bin", "p_earn_", EARN_LABELS)]
    cond, _, cond_meta = _load_external_condition_matrix(condition_csv=condition_csv, ids=ids, var_specs=var_specs)
    cond = cond.astype(np.float32)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model = RegionalConditionalEarnModel(
        cond_dim=int(cond.shape[1]),
        attr_dim=int(attr_feat.shape[1]),
        latent_dim=int(args.latent_dim),
        encoder_hidden_dims=_parse_hidden_dims(args.encoder_hidden_dims),
        head_hidden_dims=_parse_hidden_dims(args.head_hidden_dims),
        out_dim=int(p_true.shape[1]),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    model.to(device)

    cond_train = torch.from_numpy(cond[train_idx]).to(device)
    attr_train = torch.from_numpy(attr_feat[train_idx]).to(device)
    p_train = torch.from_numpy(p_true[train_idx]).to(device)
    w_train = torch.from_numpy(cell_weight[train_idx].astype(np.float32)).to(device)

    train_metrics: list[dict[str, float]] = []
    n_train = int(train_idx.size)
    bs = int(args.batch_size)
    for epoch in range(1, int(args.epochs) + 1):
        order = np.random.permutation(n_train)
        last_stats: dict[str, float] | None = None
        for start in range(0, n_train, bs):
            idx = order[start : start + bs]
            idx_t = torch.from_numpy(idx).to(device=device, dtype=torch.long)
            last_stats = model.step(
                cond=cond_train[idx_t],
                attr=attr_train[idx_t],
                p_true=p_train[idx_t],
                weight=w_train[idx_t],
            )
        if last_stats is not None and (epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(args.epochs)):
            rec = {"epoch": float(epoch), **last_stats}
            train_metrics.append(rec)
            print(f"[train] epoch={epoch} loss={rec['loss']:.6f}")

    cond_test_t = torch.from_numpy(cond[test_idx]).to(device)
    attr_test_t = torch.from_numpy(attr_feat[test_idx]).to(device)
    z_test, p_pred_t = model.predict(cond=cond_test_t, attr=attr_test_t)
    p_pred = p_pred_t.detach().cpu().numpy().astype(np.float64)
    z_norm = np.linalg.norm(z_test.detach().cpu().numpy(), axis=1)

    test_weight = np.asarray(cell_weight[test_idx], dtype=np.float64)
    test_rows = df.iloc[test_idx].copy()
    true_test = np.asarray(p_true[test_idx], dtype=np.float64)

    train_global_mean = np.average(np.asarray(p_true[train_idx], dtype=np.float64), axis=0, weights=np.asarray(cell_weight[train_idx], dtype=np.float64))
    train_global_mean = train_global_mean / max(float(train_global_mean.sum()), 1e-12)

    cell_mean, cell_mean_w = _train_cell_mean(df.iloc[train_idx].copy(), p_true[train_idx], cell_weight[train_idx])
    baseline_cell = np.zeros_like(true_test)
    test_cell_idx = test_rows["cell_idx"].to_numpy(dtype=int)
    for j, ci in enumerate(test_cell_idx.tolist()):
        if ci < cell_mean.shape[0] and cell_mean_w[ci] > 0:
            baseline_cell[j] = cell_mean[ci]
        else:
            baseline_cell[j] = train_global_mean
        baseline_cell[j] = baseline_cell[j] / max(float(baseline_cell[j].sum()), 1e-12)

    tvd_model = np.asarray([_tvd(p_pred[j], true_test[j]) for j in range(true_test.shape[0])], dtype=np.float64)
    cos_model = np.asarray([_cosine(p_pred[j], true_test[j]) for j in range(true_test.shape[0])], dtype=np.float64)
    mae_model = np.asarray([float(np.abs(p_pred[j] - true_test[j]).mean()) for j in range(true_test.shape[0])], dtype=np.float64)
    tvd_global = np.asarray([_tvd(train_global_mean, true_test[j]) for j in range(true_test.shape[0])], dtype=np.float64)
    tvd_cell = np.asarray([_tvd(baseline_cell[j], true_test[j]) for j in range(true_test.shape[0])], dtype=np.float64)

    region_tvd_model: list[float] = []
    region_tvd_global: list[float] = []
    region_tvd_cell: list[float] = []
    for uid, sub in test_rows.groupby("puma_uid", sort=False):
        idx = sub.index.to_numpy(dtype=int)
        local = np.searchsorted(test_idx, idx)
        weights_region = np.asarray(sub["cell_weight"], dtype=np.float64)
        true_region = np.average(true_test[local], axis=0, weights=weights_region)
        pred_region = np.average(p_pred[local], axis=0, weights=weights_region)
        cell_region = np.average(baseline_cell[local], axis=0, weights=weights_region)
        pred_region = pred_region / max(float(pred_region.sum()), 1e-12)
        cell_region = cell_region / max(float(cell_region.sum()), 1e-12)
        true_region = true_region / max(float(true_region.sum()), 1e-12)
        region_tvd_model.append(_tvd(pred_region, true_region))
        region_tvd_global.append(_tvd(train_global_mean, true_region))
        region_tvd_cell.append(_tvd(cell_region, true_region))

    summary = {
        "conditional_earn": {
            "weighted_tvd_earn": _weighted_mean(tvd_model, test_weight),
            "weighted_cosine_earn": _weighted_mean(cos_model, test_weight),
            "weighted_mae_earn": _weighted_mean(mae_model, test_weight),
            "latent_norm": _summ(z_norm.tolist()),
            "n_test_rows": int(test_idx.size),
            "n_test_pumas": int(test_rows["puma_uid"].nunique()),
        },
        "aggregated_region_earn": {
            "tvd_earn": _summ(region_tvd_model),
        },
        "baselines": {
            "train_mean_earn": {
                "weighted_tvd_earn": _weighted_mean(tvd_global, test_weight),
                "aggregated_region_tvd_earn": _summ(region_tvd_global),
            },
            "train_cell_mean_earn": {
                "weighted_tvd_earn": _weighted_mean(tvd_cell, test_weight),
                "aggregated_region_tvd_earn": _summ(region_tvd_cell),
            },
        },
    }

    saved_checkpoints: list[str] = []
    if bool(args.save_final_model):
        ckpt = out_dir / "checkpoints" / "external_earn_conditional_from_context" / "leave_mi_out" / "final.pt"
        model.save(
            ckpt,
            payload={
                "cond_dim": int(cond.shape[1]),
                "attr_dim": int(attr_feat.shape[1]),
                "latent_dim": int(args.latent_dim),
                "encoder_hidden_dims": list(_parse_hidden_dims(args.encoder_hidden_dims)),
                "head_hidden_dims": list(_parse_hidden_dims(args.head_hidden_dims)),
                "out_dim": int(p_true.shape[1]),
                "condition_mode": str(args.condition_mode),
                "earn_labels": EARN_LABELS,
            },
        )
        saved_checkpoints.append(str(ckpt))

    run_summary = {
        "created_utc": _utc_now_iso(),
        "target_csv": str(target_csv),
        "condition_csv": str(condition_csv),
        "condition_mode": str(args.condition_mode),
        "n_rows_total": int(df.shape[0]),
        "n_train_rows": int(train_idx.size),
        "n_test_rows": int(test_idx.size),
        "latent_dim": int(args.latent_dim),
        "attr_dim": int(attr_feat.shape[1]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "encoder_hidden_dims": list(_parse_hidden_dims(args.encoder_hidden_dims)),
        "head_hidden_dims": list(_parse_hidden_dims(args.head_hidden_dims)),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "device": str(device),
        "condition_meta": cond_meta,
        "saved_checkpoints": saved_checkpoints,
        "results": summary,
    }

    _write_json(out_dir / "run_summary.json", run_summary)
    _write_json(out_dir / "metrics" / "earn_conditional_summary.json", summary)
    _write_json(out_dir / "metrics" / "training_curve.json", {"train_metrics": train_metrics})
    print(f"[ok] wrote: {out_dir}")


if __name__ == "__main__":
    main()
