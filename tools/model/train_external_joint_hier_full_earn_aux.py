#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import random
import sys
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.model.train_external_joint_hier_age_schl as base
from tools.data.build_external_target_v1_michigan import AGE_LABELS, ESR_LABELS, SCHL_LABELS, SEX_LABELS
from tools.data.external_earn_v1_schema import EARN_LABELS
from tools.data.external_v1_variant_presets import AGE_LITE_LABELS, AGE_TO_LITE, ESR_LITE_LABELS, ESR_TO_LITE, SCHL_LITE_LABELS, SCHL_TO_LITE
from tools.model.train_us_puma_5var_diffusion import _canon_puma5, _canon_statefp, _canon_uid, _canon_uid_loose, _cosine, _parse_hidden_dims, _require_torch, _summ, _tvd, _utc_now_iso, _write_json
from tools.model.train_us_puma_external_v1_diffusion import _load_external_condition_matrix, _load_var_specs_from_schema


base.FINE_CATEGORIES = {
    "AGEP_bin": AGE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LABELS,
    "ESR_allpop": ESR_LABELS,
}
base.COARSE_CATEGORIES = {
    "AGEP_bin": AGE_LITE_LABELS,
    "SEX": SEX_LABELS,
    "SCHL_allpop": SCHL_LITE_LABELS,
    "ESR_allpop": ESR_LITE_LABELS,
}
base.FINE_SHAPE = tuple(len(base.FINE_CATEGORIES[v]) for v in base.FINE_VARIABLE_ORDER)
base.COARSE_SHAPE = tuple(len(base.COARSE_CATEGORIES[v]) for v in base.COARSE_VARIABLE_ORDER)
base.FINE_K = int(np.prod(base.FINE_SHAPE))
base.COARSE_K = int(np.prod(base.COARSE_SHAPE))


def _build_fine_to_coarse_matrix_full() -> np.ndarray:
    age_coarse_idx = {lab: i for i, lab in enumerate(AGE_LITE_LABELS)}
    schl_coarse_idx = {lab: i for i, lab in enumerate(SCHL_LITE_LABELS)}
    esr_coarse_idx = {lab: i for i, lab in enumerate(ESR_LITE_LABELS)}
    out = np.zeros((base.FINE_K, base.COARSE_K), dtype=np.float32)
    k = 0
    for age_lab in base.FINE_CATEGORIES["AGEP_bin"]:
        ac = age_coarse_idx[AGE_TO_LITE[age_lab]]
        for si, _ in enumerate(base.FINE_CATEGORIES["SEX"]):
            for schl_lab in base.FINE_CATEGORIES["SCHL_allpop"]:
                qc = schl_coarse_idx[SCHL_TO_LITE[schl_lab]]
                for esr_lab in base.FINE_CATEGORIES["ESR_allpop"]:
                    ec = esr_coarse_idx[ESR_TO_LITE[esr_lab]]
                    kc = np.ravel_multi_index((ac, si, qc, ec), base.COARSE_SHAPE)
                    out[k, kc] = 1.0
                    k += 1
    return out


base._build_fine_to_coarse_matrix = _build_fine_to_coarse_matrix_full


def _load_earn_target(*, earn_target_csv: pathlib.Path, ids: list[str]) -> np.ndarray:
    df = pd.read_csv(earn_target_csv, low_memory=False)
    need = {"puma_uid"}
    cols = set(df.columns.astype(str).tolist())
    if "puma_uid" in cols:
        df["puma_uid"] = df["puma_uid"].map(_canon_uid_loose)
    elif {"statefp", "puma"} <= cols:
        df["statefp"] = df["statefp"].map(_canon_statefp)
        df["puma"] = df["puma"].map(_canon_puma5)
        df["puma_uid"] = df.apply(lambda r: _canon_uid(r["statefp"], r["puma"]), axis=1)
    else:
        miss = [c for c in need if c not in cols]
        raise SystemExit(f"earn_target_csv missing columns: {miss}; available={sorted(cols)}")
    p_cols = sorted([c for c in df.columns if c.startswith("p_earn_")], key=lambda x: int(x.split("_")[-1]))
    if len(p_cols) != len(EARN_LABELS):
        raise SystemExit(f"earn_target_csv expected {len(EARN_LABELS)} p_earn_* cols, got {len(p_cols)}")
    lookup = {str(r["puma_uid"]): np.asarray([float(r[c]) for c in p_cols], dtype=np.float32) for _, r in df.iterrows()}
    missing_ids = [uid for uid in ids if uid not in lookup]
    if missing_ids:
        raise SystemExit(f"earn_target_csv missing {len(missing_ids)} ids. Example={missing_ids[:5]}")
    out = np.stack([lookup[uid] for uid in ids], axis=0)
    out = np.clip(out, 0.0, None)
    out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out


class JointHierarchicalEarnAuxModel(base.JointHierarchicalModel):
    def __init__(
        self,
        *,
        cond_dim: int,
        latent_dim: int,
        encoder_hidden_dims: tuple[int, ...],
        fine_hidden_dims: tuple[int, ...],
        coarse_hidden_dims: tuple[int, ...],
        earn_hidden_dims: tuple[int, ...],
        fine_input_mode: str,
        lr: float,
        weight_decay: float,
        seed: int,
    ) -> None:
        super().__init__(
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            fine_hidden_dims=fine_hidden_dims,
            coarse_hidden_dims=coarse_hidden_dims,
            fine_input_mode=fine_input_mode,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
        )
        torch = _require_torch()
        nn = torch.nn
        self.earn_head = self._make_mlp(
            in_dim=self.latent_dim,
            hidden_dims=earn_hidden_dims,
            out_dim=len(EARN_LABELS),
            nn=nn,
        )
        self._modules.append(self.earn_head)
        self._opt = torch.optim.AdamW(self._modules.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    def step(
        self,
        *,
        cond: Any,
        p_coarse_true: Any,
        p_fine_true: Any,
        p_earn_true: Any,
        agg_mat: Any,
        coarse_weight: float,
        consistency_weight: float,
        earn_weight: float,
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

        earn_logits = self.earn_head(z)
        earn_logp = torch.log_softmax(earn_logits, dim=1)

        loss_coarse = -(p_coarse_true * coarse_logp).sum(dim=1).mean()
        loss_fine = -(p_fine_true * fine_logp).sum(dim=1).mean()
        loss_cons = 0.5 * torch.abs(coarse_from_fine - coarse_prob).sum(dim=1).mean()
        loss_earn = -(p_earn_true * earn_logp).sum(dim=1).mean()
        loss = loss_fine + float(coarse_weight) * loss_coarse + float(consistency_weight) * loss_cons + float(earn_weight) * loss_earn

        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        self._opt.step()
        return {
            "loss": float(loss.detach().cpu()),
            "loss_fine": float(loss_fine.detach().cpu()),
            "loss_coarse": float(loss_coarse.detach().cpu()),
            "loss_consistency": float(loss_cons.detach().cpu()),
            "loss_earn": float(loss_earn.detach().cpu()),
        }

    def predict_prob(self, *, cond: Any) -> tuple[Any, Any, Any]:
        torch = _require_torch()
        self.eval()
        with torch.no_grad():
            z = self.encoder(cond)
            coarse_feat = self.coarse_feature(z)
            coarse_logits = self.coarse_out(coarse_feat)
            coarse_prob = torch.softmax(coarse_logits, dim=1)
            fine_logits = self.fine_head(self._fine_input(z=z, coarse_prob=coarse_prob, coarse_feat=coarse_feat))
            fine_prob = torch.softmax(fine_logits, dim=1)
            earn_prob = torch.softmax(self.earn_head(z), dim=1)
        return coarse_prob, fine_prob, earn_prob


def main() -> None:
    ap = argparse.ArgumentParser(prog="train_external_joint_hier_full_earn_aux")
    ap.add_argument("--joint_wide_csv", required=True)
    ap.add_argument("--condition_csv", required=True)
    ap.add_argument("--schema_json", required=True)
    ap.add_argument("--earn_target_csv", required=True)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--encoder_hidden_dims", default="256,256")
    ap.add_argument("--coarse_hidden_dims", default="256")
    ap.add_argument("--fine_hidden_dims", default="512,512")
    ap.add_argument("--earn_hidden_dims", default="128,128")
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--fine_input_mode", choices=["z_only", "z_coarse_prob", "z_coarse_latent"], default="z_only")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--coarse_weight", type=float, default=0.5)
    ap.add_argument("--consistency_weight", type=float, default=1.0)
    ap.add_argument("--earn_weight", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--ipf_iters", type=int, default=200)
    ap.add_argument("--save_final_model", action="store_true")
    ap.add_argument("--run_label", default="external_joint_hier_full_earn_aux")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    torch = _require_torch()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    joint_csv = pathlib.Path(args.joint_wide_csv).expanduser().resolve()
    condition_csv = pathlib.Path(args.condition_csv).expanduser().resolve()
    schema_json = pathlib.Path(args.schema_json).expanduser().resolve()
    earn_target_csv = pathlib.Path(args.earn_target_csv).expanduser().resolve()
    for p in [joint_csv, condition_csv, schema_json, earn_target_csv]:
        if not p.exists():
            raise SystemExit(f"path not found: {p}")

    run_id = f"_us_puma_{args.run_label}_{args.fine_input_mode}_{_dt.datetime.now(_dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve() if args.out_dir else (_REPO_ROOT / "outputs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    df, p_fine, ids, fine_shape = base._load_joint_wide(joint_wide_csv=joint_csv, schema_json=schema_json)
    if tuple(fine_shape) != base.FINE_SHAPE:
        raise SystemExit(f"unexpected fine shape: {fine_shape}, expected {base.FINE_SHAPE}")
    p_earn = _load_earn_target(earn_target_csv=earn_target_csv, ids=ids)

    is_mi = (df["statefp"] == "26").to_numpy(dtype=bool)
    train_idx = np.where(~is_mi)[0]
    test_idx = np.where(is_mi)[0]
    if train_idx.size == 0 or test_idx.size == 0:
        raise SystemExit("invalid leave_mi_out split")

    joint_var_specs = _load_var_specs_from_schema(schema_json=schema_json)
    cond_var_specs = list(joint_var_specs) + [("EARN_16p_bin", "p_earn_", EARN_LABELS)]
    cond, block_slices, cond_meta = _load_external_condition_matrix(condition_csv=condition_csv, ids=ids, var_specs=cond_var_specs)
    cond = cond.astype(np.float32)
    ext_marg = {var: cond[:, block_slices[var]].copy() for var, _, _ in joint_var_specs}

    agg_mat_np = base._build_fine_to_coarse_matrix()
    p_coarse = base._aggregate_fine_to_coarse_np(p_fine, agg_mat_np)
    p_coarse = p_coarse / np.maximum(p_coarse.sum(axis=1, keepdims=True), 1e-12)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    agg_mat = torch.from_numpy(agg_mat_np).to(device)

    model = JointHierarchicalEarnAuxModel(
        cond_dim=int(cond.shape[1]),
        latent_dim=int(args.latent_dim),
        encoder_hidden_dims=_parse_hidden_dims(args.encoder_hidden_dims),
        coarse_hidden_dims=_parse_hidden_dims(args.coarse_hidden_dims),
        fine_hidden_dims=_parse_hidden_dims(args.fine_hidden_dims),
        earn_hidden_dims=_parse_hidden_dims(args.earn_hidden_dims),
        fine_input_mode=str(args.fine_input_mode),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    model.to(device)

    cond_train = torch.from_numpy(cond[train_idx]).to(device)
    p_coarse_train = torch.from_numpy(p_coarse[train_idx].astype(np.float32)).to(device)
    p_fine_train = torch.from_numpy(p_fine[train_idx].astype(np.float32)).to(device)
    p_earn_train = torch.from_numpy(p_earn[train_idx].astype(np.float32)).to(device)

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
                p_coarse_true=p_coarse_train[idx_t],
                p_fine_true=p_fine_train[idx_t],
                p_earn_true=p_earn_train[idx_t],
                agg_mat=agg_mat,
                coarse_weight=float(args.coarse_weight),
                consistency_weight=float(args.consistency_weight),
                earn_weight=float(args.earn_weight),
            )
        if last_stats is not None and (epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(args.epochs)):
            rec = {"epoch": float(epoch), **last_stats}
            train_metrics.append(rec)
            print(
                f"[train] epoch={epoch} "
                f"loss={rec['loss']:.6f} "
                f"fine={rec['loss_fine']:.6f} "
                f"coarse={rec['loss_coarse']:.6f} "
                f"cons={rec['loss_consistency']:.6f} "
                f"earn={rec['loss_earn']:.6f}"
            )

    cond_test_t = torch.from_numpy(cond[test_idx]).to(device)
    coarse_prob_t, fine_prob_t, earn_prob_t = model.predict_prob(cond=cond_test_t)
    coarse_pred = coarse_prob_t.detach().cpu().numpy().astype(np.float64)
    fine_pred_raw = fine_prob_t.detach().cpu().numpy().astype(np.float64)
    earn_pred = earn_prob_t.detach().cpu().numpy().astype(np.float64)
    fine_pred_raw = fine_pred_raw / np.maximum(fine_pred_raw.sum(axis=1, keepdims=True), 1e-12)
    earn_pred = earn_pred / np.maximum(earn_pred.sum(axis=1, keepdims=True), 1e-12)
    fine_pred = fine_pred_raw.copy()
    for j, idx in enumerate(test_idx):
        marginals_ext = [ext_marg[var][idx] for var, _, _ in joint_var_specs]
        fine_pred[j] = base._ipf_nd(
            seed_joint=fine_pred_raw[j].reshape(base.FINE_SHAPE),
            target_marginals=[np.asarray(m, dtype=float) for m in marginals_ext],
            shape=base.FINE_SHAPE,
            max_iter=int(args.ipf_iters),
        ).reshape(-1)
        fine_pred[j] = fine_pred[j] / max(float(fine_pred[j].sum()), 1e-12)
    coarse_from_fine = base._aggregate_fine_to_coarse_np(fine_pred, agg_mat_np)
    coarse_from_fine = coarse_from_fine / np.maximum(coarse_from_fine.sum(axis=1, keepdims=True), 1e-12)

    tvd_fine = []
    cosine_fine = []
    tvd_coarse_head = []
    tvd_coarse_from_fine = []
    tvd_earn = []
    cosine_earn = []
    mae_earn = []
    for j, idx in enumerate(test_idx):
        p_true_f = p_fine[idx]
        p_true_c = p_coarse[idx]
        p_true_e = p_earn[idx]
        tvd_fine.append(_tvd(fine_pred[j], p_true_f))
        cosine_fine.append(_cosine(fine_pred[j], p_true_f))
        tvd_coarse_head.append(_tvd(coarse_pred[j], p_true_c))
        tvd_coarse_from_fine.append(_tvd(coarse_from_fine[j], p_true_c))
        tvd_earn.append(_tvd(earn_pred[j], p_true_e))
        cosine_earn.append(_cosine(earn_pred[j], p_true_e))
        mae_earn.append(float(np.abs(earn_pred[j] - p_true_e).mean()))

    train_seed = np.asarray(p_fine[train_idx], dtype=np.float64).mean(axis=0)
    train_seed = train_seed / max(float(train_seed.sum()), 1e-12)
    tvd_ipf = []
    tvd_ind = []
    for idx in test_idx:
        marginals_ext = [ext_marg[var][idx] for var, _, _ in joint_var_specs]
        p_ipf = base._ipf_nd(
            seed_joint=train_seed.reshape(base.FINE_SHAPE),
            target_marginals=[np.asarray(m, dtype=float) for m in marginals_ext],
            shape=base.FINE_SHAPE,
            max_iter=int(args.ipf_iters),
        ).reshape(-1)
        p_ipf = p_ipf / max(float(p_ipf.sum()), 1e-12)
        tvd_ipf.append(_tvd(p_ipf, p_fine[idx]))

        p_ind = np.ones(base.FINE_SHAPE, dtype=np.float64)
        for axis, m in enumerate(marginals_ext):
            shape = [1] * len(base.FINE_SHAPE)
            shape[axis] = len(m)
            p_ind *= np.asarray(m, dtype=np.float64).reshape(shape)
        p_ind = p_ind.reshape(-1)
        p_ind = p_ind / max(float(p_ind.sum()), 1e-12)
        tvd_ind.append(_tvd(p_ind, p_fine[idx]))

    train_mean_earn = np.asarray(p_earn[train_idx], dtype=np.float64).mean(axis=0)
    train_mean_earn = train_mean_earn / max(float(train_mean_earn.sum()), 1e-12)
    tvd_earn_baseline = [_tvd(train_mean_earn, p_earn[idx]) for idx in test_idx]

    summary = {
        "hierarchical_joint": {
            "tvd_joint": _summ(tvd_fine),
            "cosine_joint": _summ(cosine_fine),
            "tvd_coarse_head": _summ(tvd_coarse_head),
            "tvd_coarse_from_fine": _summ(tvd_coarse_from_fine),
        },
        "earn_aux": {
            "tvd_earn": _summ(tvd_earn),
            "cosine_earn": _summ(cosine_earn),
            "mae_earn": _summ(mae_earn),
        },
        "baselines": {
            "ipf_train_seed_external": {"tvd_joint": _summ(tvd_ipf)},
            "independence_external": {"tvd_joint": _summ(tvd_ind)},
            "train_mean_earn": {"tvd_earn": _summ(tvd_earn_baseline)},
        },
    }

    saved_checkpoints: list[str] = []
    if bool(args.save_final_model):
        ckpt = out_dir / "checkpoints" / str(args.run_label) / "leave_mi_out" / "final.pt"
        model.save(
            ckpt,
            payload={
                "cond_dim": int(cond.shape[1]),
                "latent_dim": int(args.latent_dim),
                "encoder_hidden_dims": list(_parse_hidden_dims(args.encoder_hidden_dims)),
                "coarse_hidden_dims": list(_parse_hidden_dims(args.coarse_hidden_dims)),
                "fine_hidden_dims": list(_parse_hidden_dims(args.fine_hidden_dims)),
                "earn_hidden_dims": list(_parse_hidden_dims(args.earn_hidden_dims)),
                "fine_input_mode": str(args.fine_input_mode),
                "fine_shape": list(base.FINE_SHAPE),
                "coarse_shape": list(base.COARSE_SHAPE),
                "agg_mat": agg_mat_np.tolist(),
                "earn_labels": EARN_LABELS,
            },
        )
        saved_checkpoints.append(str(ckpt))

    run_summary = {
        "created_utc": _utc_now_iso(),
        "joint_wide_csv": str(joint_csv),
        "condition_csv": str(condition_csv),
        "schema_json": str(schema_json),
        "earn_target_csv": str(earn_target_csv),
        "n_rows_total": int(df.shape[0]),
        "n_train": int(train_idx.size),
        "n_test_mi": int(test_idx.size),
        "cond_dim": int(cond.shape[1]),
        "latent_dim": int(args.latent_dim),
        "fine_shape": list(base.FINE_SHAPE),
        "coarse_shape": list(base.COARSE_SHAPE),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "encoder_hidden_dims": list(_parse_hidden_dims(args.encoder_hidden_dims)),
        "coarse_hidden_dims": list(_parse_hidden_dims(args.coarse_hidden_dims)),
        "fine_hidden_dims": list(_parse_hidden_dims(args.fine_hidden_dims)),
        "earn_hidden_dims": list(_parse_hidden_dims(args.earn_hidden_dims)),
        "fine_input_mode": str(args.fine_input_mode),
        "run_label": str(args.run_label),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "coarse_weight": float(args.coarse_weight),
        "consistency_weight": float(args.consistency_weight),
        "earn_weight": float(args.earn_weight),
        "ipf_iters": int(args.ipf_iters),
        "seed": int(args.seed),
        "device": str(device),
        "condition_meta": cond_meta,
        "saved_checkpoints": saved_checkpoints,
        "results": summary,
    }
    _write_json(out_dir / "run_summary.json", run_summary)
    _write_json(out_dir / "metrics" / "hierarchical_summary.json", summary)
    _write_json(out_dir / "metrics" / "training_curve.json", {"train_metrics": train_metrics})
    print(f"[ok] wrote: {out_dir}")


if __name__ == "__main__":
    main()
