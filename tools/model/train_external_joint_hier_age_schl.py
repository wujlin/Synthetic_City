#!/usr/bin/env python3
from __future__ import annotations

"""
Train a shared-latent hierarchical model for the external age+education refinement task.

Input:
- external condition under the official age_schl_refine schema

Output heads:
- coarse joint: AGEP_lite x SEX x SCHL_lite x ESR_lite  (K=72)
- fine joint:   AGEP_fine x SEX x SCHL_fine x ESR_lite  (K=300)

Core idea:
- A single encoder extracts a region-level latent state z from the external condition.
- A coarse head predicts the coarse joint.
- A fine head predicts the fine joint from z and the coarse prediction.
- The model is trained jointly with a consistency constraint:
    aggregate(fine) ~= coarse

This is intended as a minimal scientific probe of whether preserving a shared
regional latent across scales can reduce the information loss observed in the
naive coarse-to-fine pipeline.
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

from tools.data.external_v1_variant_presets import (
    AGE_LITE_LABELS,
    AGE_TO_LITE,
    ESR_LITE_LABELS,
    SCHL_LITE_LABELS,
    SCHL_TO_LITE,
)
from tools.model.train_us_puma_5var_diffusion import (
    _canon_puma5,
    _canon_statefp,
    _canon_uid,
    _cosine,
    _ipf_nd,
    _parse_hidden_dims,
    _require_torch,
    _summ,
    _tvd,
    _utc_now_iso,
    _write_json,
)
from tools.model.train_us_puma_external_v1_diffusion import _load_external_condition_matrix, _load_var_specs_from_schema


FINE_VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop"]
COARSE_VARIABLE_ORDER = ["AGEP_bin", "SEX", "SCHL_allpop", "ESR_allpop"]
FINE_CATEGORIES = {
    "AGEP_bin": [
        "[0.0, 5.0)",
        "[5.0, 18.0)",
        "[18.0, 25.0)",
        "[25.0, 35.0)",
        "[35.0, 45.0)",
        "[45.0, 55.0)",
        "[55.0, 65.0)",
        "[65.0, 75.0)",
        "[75.0, 85.0)",
        "[85.0, 1000.0)",
    ],
    "SEX": ["1", "2"],
    "SCHL_allpop": [
        "not_25p",
        "less_than_high_school",
        "high_school_or_ged",
        "some_college_or_assoc",
        "bachelor_plus",
    ],
    "ESR_allpop": ESR_LITE_LABELS,
}
COARSE_CATEGORIES = {
    "AGEP_bin": AGE_LITE_LABELS,
    "SEX": ["1", "2"],
    "SCHL_allpop": SCHL_LITE_LABELS,
    "ESR_allpop": ESR_LITE_LABELS,
}
FINE_SHAPE = tuple(len(FINE_CATEGORIES[v]) for v in FINE_VARIABLE_ORDER)
COARSE_SHAPE = tuple(len(COARSE_CATEGORIES[v]) for v in COARSE_VARIABLE_ORDER)
FINE_K = int(np.prod(FINE_SHAPE))
COARSE_K = int(np.prod(COARSE_SHAPE))


def _default_run_label() -> str:
    stem = pathlib.Path(sys.argv[0]).stem
    if stem.startswith("train_"):
        stem = stem[len("train_") :]
    return stem or "external_joint_hier"


def _load_joint_wide(*, joint_wide_csv: pathlib.Path, schema_json: pathlib.Path) -> tuple[pd.DataFrame, np.ndarray, list[str], tuple[int, ...]]:
    schema = json.loads(schema_json.read_text(encoding="utf-8"))
    shape = tuple(int(x) for x in schema["shape"])
    K = int(np.prod(shape))
    df = pd.read_csv(joint_wide_csv, low_memory=False)
    req = {"statefp", "puma", "puma_uid"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise SystemExit(f"joint_wide_csv missing columns: {miss}")
    df["statefp"] = df["statefp"].map(_canon_statefp)
    df["puma5"] = df["puma"].map(_canon_puma5)
    df["puma_uid"] = df.apply(lambda r: _canon_uid(r["statefp"], r["puma5"]), axis=1)
    p_joint_cols = [f"p_joint_{i:03d}" for i in range(K)]
    miss_joint = [c for c in p_joint_cols if c not in df.columns]
    if miss_joint:
        raise SystemExit(f"joint_wide_csv missing joint columns: {miss_joint[:5]}")
    p_joint = df[p_joint_cols].to_numpy(dtype=np.float32)
    p_joint = np.clip(p_joint, 0.0, None)
    p_joint = p_joint / np.maximum(p_joint.sum(axis=1, keepdims=True), 1e-12)
    ids = df["puma_uid"].astype(str).tolist()
    return df, p_joint, ids, shape


def _build_fine_to_coarse_matrix() -> np.ndarray:
    age_coarse_idx = {lab: i for i, lab in enumerate(AGE_LITE_LABELS)}
    schl_coarse_idx = {lab: i for i, lab in enumerate(SCHL_LITE_LABELS)}
    out = np.zeros((FINE_K, COARSE_K), dtype=np.float32)
    k = 0
    for ai, age_lab in enumerate(FINE_CATEGORIES["AGEP_bin"]):
        ac = age_coarse_idx[AGE_TO_LITE[age_lab]]
        for si, _ in enumerate(FINE_CATEGORIES["SEX"]):
            for qi, schl_lab in enumerate(FINE_CATEGORIES["SCHL_allpop"]):
                qc = schl_coarse_idx[SCHL_TO_LITE[schl_lab]]
                for ei, _ in enumerate(FINE_CATEGORIES["ESR_allpop"]):
                    kc = np.ravel_multi_index((ac, si, qc, ei), COARSE_SHAPE)
                    out[k, kc] = 1.0
                    k += 1
    return out


def _aggregate_fine_to_coarse_np(p_fine: np.ndarray, agg_mat: np.ndarray) -> np.ndarray:
    return np.asarray(p_fine, dtype=np.float64) @ np.asarray(agg_mat, dtype=np.float64)


class JointHierarchicalModel:
    def __init__(
        self,
        *,
        cond_dim: int,
        latent_dim: int,
        encoder_hidden_dims: tuple[int, ...],
        fine_hidden_dims: tuple[int, ...],
        coarse_hidden_dims: tuple[int, ...],
        fine_input_mode: str,
        lr: float,
        weight_decay: float,
        seed: int,
    ) -> None:
        torch = _require_torch()
        nn = torch.nn
        torch.manual_seed(int(seed))
        self.cond_dim = int(cond_dim)
        self.latent_dim = int(latent_dim)
        self.fine_input_mode = str(fine_input_mode)
        if self.fine_input_mode not in {"z_only", "z_coarse_prob", "z_coarse_latent"}:
            raise ValueError(f"unsupported fine_input_mode: {self.fine_input_mode}")
        self.encoder = self._make_mlp(
            in_dim=self.cond_dim,
            hidden_dims=encoder_hidden_dims,
            out_dim=self.latent_dim,
            nn=nn,
        )
        self.coarse_feature = self._make_mlp(
            in_dim=self.latent_dim,
            hidden_dims=coarse_hidden_dims,
            out_dim=self.latent_dim,
            nn=nn,
        )
        self.coarse_out = nn.Linear(self.latent_dim, COARSE_K)
        if self.fine_input_mode == "z_only":
            fine_in_dim = self.latent_dim
        elif self.fine_input_mode == "z_coarse_prob":
            fine_in_dim = self.latent_dim + COARSE_K
        else:
            fine_in_dim = self.latent_dim + self.latent_dim
        self.fine_head = self._make_mlp(
            in_dim=fine_in_dim,
            hidden_dims=fine_hidden_dims,
            out_dim=FINE_K,
            nn=nn,
        )
        self._modules = nn.ModuleList([self.encoder, self.coarse_feature, self.coarse_out, self.fine_head])
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

    def parameters(self) -> Any:
        return self._modules.parameters()

    def _fine_input(self, *, z: Any, coarse_prob: Any, coarse_feat: Any) -> Any:
        torch = _require_torch()
        if self.fine_input_mode == "z_only":
            return z
        if self.fine_input_mode == "z_coarse_prob":
            return torch.cat([z, coarse_prob], dim=1)
        return torch.cat([z, coarse_feat], dim=1)

    def step(
        self,
        *,
        cond: Any,
        p_coarse_true: Any,
        p_fine_true: Any,
        agg_mat: Any,
        coarse_weight: float,
        consistency_weight: float,
    ) -> dict[str, float]:
        torch = _require_torch()
        self.train()
        z = self.encoder(cond)
        coarse_feat = self.coarse_feature(z)
        coarse_logits = self.coarse_out(coarse_feat)
        coarse_logp = torch.log_softmax(coarse_logits, dim=1)
        coarse_prob = torch.softmax(coarse_logits, dim=1)

        fine_logits = self.fine_head(self._fine_input(z=z, coarse_prob=coarse_prob, coarse_feat=coarse_feat))
        fine_logp = torch.log_softmax(fine_logits, dim=1)
        fine_prob = torch.softmax(fine_logits, dim=1)
        coarse_from_fine = fine_prob @ agg_mat

        loss_coarse = -(p_coarse_true * coarse_logp).sum(dim=1).mean()
        loss_fine = -(p_fine_true * fine_logp).sum(dim=1).mean()
        loss_cons = 0.5 * torch.abs(coarse_from_fine - coarse_prob).sum(dim=1).mean()
        loss = loss_fine + float(coarse_weight) * loss_coarse + float(consistency_weight) * loss_cons

        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        self._opt.step()
        return {
            "loss": float(loss.detach().cpu()),
            "loss_fine": float(loss_fine.detach().cpu()),
            "loss_coarse": float(loss_coarse.detach().cpu()),
            "loss_consistency": float(loss_cons.detach().cpu()),
        }

    def predict_prob(self, *, cond: Any) -> tuple[Any, Any]:
        torch = _require_torch()
        self.eval()
        with torch.no_grad():
            z = self.encoder(cond)
            coarse_feat = self.coarse_feature(z)
            coarse_logits = self.coarse_out(coarse_feat)
            coarse_prob = torch.softmax(coarse_logits, dim=1)
            fine_logits = self.fine_head(self._fine_input(z=z, coarse_prob=coarse_prob, coarse_feat=coarse_feat))
            fine_prob = torch.softmax(fine_logits, dim=1)
        return coarse_prob, fine_prob

    def save(self, path: pathlib.Path, *, payload: dict[str, Any]) -> None:
        torch = _require_torch()
        path = pathlib.Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "synthpop.external_joint_hier.v0",
                "state_dict": self._modules.state_dict(),
                **payload,
            },
            path,
        )


def main() -> None:
    ap = argparse.ArgumentParser(prog="train_external_joint_hier_age_schl")
    ap.add_argument("--joint_wide_csv", required=True)
    ap.add_argument("--condition_csv", required=True)
    ap.add_argument("--schema_json", required=True)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--encoder_hidden_dims", default="256,256")
    ap.add_argument("--coarse_hidden_dims", default="256")
    ap.add_argument("--fine_hidden_dims", default="512,512")
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--fine_input_mode", choices=["z_only", "z_coarse_prob", "z_coarse_latent"], default="z_coarse_prob")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--coarse_weight", type=float, default=0.5)
    ap.add_argument("--consistency_weight", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--ipf_iters", type=int, default=200)
    ap.add_argument("--save_final_model", action="store_true")
    ap.add_argument("--run_label", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    torch = _require_torch()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    joint_csv = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    condition_csv = pathlib.Path(args.condition_csv).expanduser().resolve()
    schema_json = pathlib.Path(args.schema_json).expanduser().resolve()
    for p in [joint_csv, condition_csv, schema_json]:
        if not p.exists():
            raise SystemExit(f"path not found: {p}")

    run_label = str(args.run_label or _default_run_label())
    run_id = f"_us_puma_{run_label}_{args.fine_input_mode}_{_dt.datetime.now(_dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else (_REPO_ROOT / "outputs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    df, p_fine, ids, fine_shape = _load_joint_wide(joint_wide_csv=joint_csv, schema_json=schema_json)
    if tuple(fine_shape) != FINE_SHAPE:
        raise SystemExit(f"unexpected fine shape: {fine_shape}, expected {FINE_SHAPE}")
    is_mi = (df["statefp"] == "26").to_numpy(dtype=bool)
    train_idx = np.where(~is_mi)[0]
    test_idx = np.where(is_mi)[0]
    if train_idx.size == 0 or test_idx.size == 0:
        raise SystemExit("invalid leave_mi_out split")

    var_specs = _load_var_specs_from_schema(schema_json=schema_json)
    cond, block_slices, _ = _load_external_condition_matrix(
        condition_csv=condition_csv,
        ids=ids,
        var_specs=var_specs,
    )
    cond = cond.astype(np.float32)
    ext_marg = {var: cond[:, sl].copy() for var, sl in block_slices.items()}

    agg_mat_np = _build_fine_to_coarse_matrix()
    p_coarse = _aggregate_fine_to_coarse_np(p_fine, agg_mat_np)
    p_coarse = p_coarse / np.maximum(p_coarse.sum(axis=1, keepdims=True), 1e-12)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    agg_mat = torch.from_numpy(agg_mat_np).to(device)

    model = JointHierarchicalModel(
        cond_dim=int(cond.shape[1]),
        latent_dim=int(args.latent_dim),
        encoder_hidden_dims=_parse_hidden_dims(args.encoder_hidden_dims),
        coarse_hidden_dims=_parse_hidden_dims(args.coarse_hidden_dims),
        fine_hidden_dims=_parse_hidden_dims(args.fine_hidden_dims),
        fine_input_mode=str(args.fine_input_mode),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    model.to(device)

    cond_train = torch.from_numpy(cond[train_idx]).to(device)
    p_coarse_train = torch.from_numpy(p_coarse[train_idx].astype(np.float32)).to(device)
    p_fine_train = torch.from_numpy(p_fine[train_idx].astype(np.float32)).to(device)

    train_metrics: list[dict[str, float]] = []
    n_train = int(train_idx.size)
    bs = int(args.batch_size)
    for epoch in range(1, int(args.epochs) + 1):
        order = np.random.permutation(n_train)
        last_stats: dict[str, float] | None = None
        for start in range(0, n_train, bs):
            idx = order[start : start + bs]
            idx_t = torch.from_numpy(idx).to(device=device, dtype=torch.long)
            stats = model.step(
                cond=cond_train[idx_t],
                p_coarse_true=p_coarse_train[idx_t],
                p_fine_true=p_fine_train[idx_t],
                agg_mat=agg_mat,
                coarse_weight=float(args.coarse_weight),
                consistency_weight=float(args.consistency_weight),
            )
            last_stats = stats
        if last_stats is not None and (epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(args.epochs)):
            rec = {"epoch": float(epoch), **last_stats}
            train_metrics.append(rec)
            print(
                f"[train] epoch={epoch} "
                f"loss={rec['loss']:.6f} "
                f"fine={rec['loss_fine']:.6f} "
                f"coarse={rec['loss_coarse']:.6f} "
                f"cons={rec['loss_consistency']:.6f}"
            )

    cond_test_t = torch.from_numpy(cond[test_idx]).to(device)
    coarse_prob_t, fine_prob_t = model.predict_prob(cond=cond_test_t)
    coarse_pred = coarse_prob_t.detach().cpu().numpy().astype(np.float64)
    fine_pred_raw = fine_prob_t.detach().cpu().numpy().astype(np.float64)
    fine_pred_raw = fine_pred_raw / np.maximum(fine_pred_raw.sum(axis=1, keepdims=True), 1e-12)
    fine_pred = fine_pred_raw.copy()
    for j, idx in enumerate(test_idx):
        marginals_ext = [ext_marg[var][idx] for var, _, _ in var_specs]
        fine_pred[j] = _ipf_nd(
            seed_joint=fine_pred_raw[j].reshape(FINE_SHAPE),
            target_marginals=[np.asarray(m, dtype=float) for m in marginals_ext],
            shape=FINE_SHAPE,
            max_iter=int(args.ipf_iters),
        ).reshape(-1)
        fine_pred[j] = fine_pred[j] / max(float(fine_pred[j].sum()), 1e-12)
    coarse_from_fine_raw = _aggregate_fine_to_coarse_np(fine_pred_raw, agg_mat_np)
    coarse_from_fine_raw = coarse_from_fine_raw / np.maximum(coarse_from_fine_raw.sum(axis=1, keepdims=True), 1e-12)
    coarse_from_fine = _aggregate_fine_to_coarse_np(fine_pred, agg_mat_np)
    coarse_from_fine = coarse_from_fine / np.maximum(coarse_from_fine.sum(axis=1, keepdims=True), 1e-12)

    tvd_fine_raw = []
    tvd_fine = []
    cosine_fine_raw = []
    cosine_fine = []
    tvd_coarse_head = []
    tvd_coarse_from_fine = []
    for j, idx in enumerate(test_idx):
        p_true_f = p_fine[idx]
        p_true_c = p_coarse[idx]
        tvd_fine_raw.append(_tvd(fine_pred_raw[j], p_true_f))
        tvd_fine.append(_tvd(fine_pred[j], p_true_f))
        cosine_fine_raw.append(_cosine(fine_pred_raw[j], p_true_f))
        cosine_fine.append(_cosine(fine_pred[j], p_true_f))
        tvd_coarse_head.append(_tvd(coarse_pred[j], p_true_c))
        tvd_coarse_from_fine.append(_tvd(coarse_from_fine[j], p_true_c))

    train_seed = np.asarray(p_fine[train_idx], dtype=np.float64).mean(axis=0)
    train_seed = train_seed / max(float(train_seed.sum()), 1e-12)
    tvd_ipf = []
    tvd_ind = []
    for idx in test_idx:
        marginals_ext = [ext_marg[var][idx] for var, _, _ in var_specs]
        p_ipf = _ipf_nd(
            seed_joint=train_seed.reshape(FINE_SHAPE),
            target_marginals=[np.asarray(m, dtype=float) for m in marginals_ext],
            shape=FINE_SHAPE,
            max_iter=int(args.ipf_iters),
        ).reshape(-1)
        p_ipf = p_ipf / max(float(p_ipf.sum()), 1e-12)
        tvd_ipf.append(_tvd(p_ipf, p_fine[idx]))

        p_ind = np.ones(FINE_SHAPE, dtype=np.float64)
        for axis, m in enumerate(marginals_ext):
            shape = [1] * len(FINE_SHAPE)
            shape[axis] = len(m)
            p_ind *= np.asarray(m, dtype=np.float64).reshape(shape)
        p_ind = p_ind.reshape(-1)
        p_ind = p_ind / max(float(p_ind.sum()), 1e-12)
        tvd_ind.append(_tvd(p_ind, p_fine[idx]))

    summary = {
        "hierarchical_joint": {
            "tvd_joint_raw": _summ(tvd_fine_raw),
            "tvd_joint": _summ(tvd_fine),
            "cosine_joint_raw": _summ(cosine_fine_raw),
            "cosine_joint": _summ(cosine_fine),
            "tvd_coarse_head": _summ(tvd_coarse_head),
            "tvd_coarse_from_fine": _summ(tvd_coarse_from_fine),
        },
        "baselines": {
            "ipf_train_seed_external": {"tvd_joint": _summ(tvd_ipf)},
            "independence_external": {"tvd_joint": _summ(tvd_ind)},
        },
    }

    saved_checkpoints: list[str] = []
    if bool(args.save_final_model):
        ckpt = out_dir / "checkpoints" / run_label / "leave_mi_out" / "final.pt"
        model.save(
            ckpt,
            payload={
                "cond_dim": int(cond.shape[1]),
                "latent_dim": int(args.latent_dim),
                "encoder_hidden_dims": list(_parse_hidden_dims(args.encoder_hidden_dims)),
                "coarse_hidden_dims": list(_parse_hidden_dims(args.coarse_hidden_dims)),
                "fine_hidden_dims": list(_parse_hidden_dims(args.fine_hidden_dims)),
                "fine_input_mode": str(args.fine_input_mode),
                "fine_shape": list(FINE_SHAPE),
                "coarse_shape": list(COARSE_SHAPE),
                "agg_mat": agg_mat_np.tolist(),
            },
        )
        saved_checkpoints.append(str(ckpt))

    run_summary = {
        "created_utc": _utc_now_iso(),
        "joint_wide_csv": str(joint_csv),
        "condition_csv": str(condition_csv),
        "schema_json": str(schema_json),
        "n_rows_total": int(df.shape[0]),
        "n_train": int(train_idx.size),
        "n_test_mi": int(test_idx.size),
        "cond_dim": int(cond.shape[1]),
        "latent_dim": int(args.latent_dim),
        "fine_shape": list(FINE_SHAPE),
        "coarse_shape": list(COARSE_SHAPE),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "encoder_hidden_dims": list(_parse_hidden_dims(args.encoder_hidden_dims)),
        "coarse_hidden_dims": list(_parse_hidden_dims(args.coarse_hidden_dims)),
        "fine_hidden_dims": list(_parse_hidden_dims(args.fine_hidden_dims)),
        "fine_input_mode": str(args.fine_input_mode),
        "run_label": run_label,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "coarse_weight": float(args.coarse_weight),
        "consistency_weight": float(args.consistency_weight),
        "ipf_iters": int(args.ipf_iters),
        "seed": int(args.seed),
        "device": str(device),
        "saved_checkpoints": saved_checkpoints,
        "results": summary,
    }

    _write_json(out_dir / "run_summary.json", run_summary)
    _write_json(out_dir / "metrics" / "hierarchical_summary.json", summary)
    _write_json(out_dir / "metrics" / "training_curve.json", {"train_metrics": train_metrics})
    print(f"[ok] wrote: {out_dir}")


if __name__ == "__main__":
    main()
