#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.experimental.workflows.export_michigan_pairwise_tvd_long import (  # noqa: E402
    _build_hierarchical_vectors,
    _load_ext_inputs,
    _select_device,
)
from tools.model.eval_external_c2f_full_earn_pipeline import _load_full_joint_wide  # noqa: E402
from tools.figures.make_fig4_michigan_regional_validation import PIPELINE_CONFIGS  # noqa: E402


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Export hierarchical full-earn joint vectors for all PUMAs using an existing "
            "leave-Michigan-out hierarchical checkpoint. This provides a base distribution "
            "for projection-aware residual correction experiments."
        )
    )
    ap.add_argument("--seed", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--heldout_statefp", default="26", help="State used by the existing leave-out checkpoint/scaler.")
    ap.add_argument("--n_eval_joint_samples", type=int, default=None)
    ap.add_argument("--ipf_iters", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    cfg = PIPELINE_CONFIGS[int(args.seed)]
    if args.n_eval_joint_samples is not None:
        cfg = dataclasses.replace(cfg, stage2_n_eval_joint_samples=int(args.n_eval_joint_samples))
    if args.ipf_iters is not None:
        cfg = dataclasses.replace(cfg, ipf_iters=int(args.ipf_iters))

    out_dir = args.output_dir or pathlib.Path(f"outputs/_hierarchical_full_earn_base_vectors_seed{int(args.seed)}_{_utc_ts()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_wide_csv = pathlib.Path(cfg.stage1_joint_wide_csv).expanduser().resolve()
    schema_json = pathlib.Path(cfg.stage1_schema_json).expanduser().resolve()
    condition_csv = pathlib.Path(cfg.stage1_condition_csv).expanduser().resolve()
    condition_schema_json = pathlib.Path(cfg.stage1_condition_schema_json).expanduser().resolve()

    df, p_true_all, ids = _load_full_joint_wide(joint_wide_csv=joint_wide_csv, schema_json=schema_json)
    cond_raw, ext_marg = _load_ext_inputs(
        condition_csv=condition_csv,
        condition_schema_json=condition_schema_json,
        ids=ids,
        stage1_schema_json=schema_json,
    )
    all_idx = np.arange(p_true_all.shape[0], dtype=np.int64)
    device = _select_device(str(args.device))
    p_hier = _build_hierarchical_vectors(
        cfg=cfg,
        p_true_all=p_true_all,
        cond_raw=cond_raw,
        ext_marg=ext_marg,
        heldout_idx=all_idx,
        heldout_statefp=str(args.heldout_statefp).zfill(2),
        device=device,
    )
    p_hier = p_hier / np.maximum(p_hier.sum(axis=1, keepdims=True), 1e-12)

    key = f"hierarchical_seed{int(args.seed)}"
    out_npz = out_dir / "hierarchical_full_earn_base_vectors_all_pumas.npz"
    np.savez_compressed(
        out_npz,
        p_true=p_true_all.astype(np.float32),
        **{key: p_hier.astype(np.float32)},
        puma_uid=df["puma_uid"].astype(str).to_numpy(),
        statefp=df["statefp"].astype(str).to_numpy(),
        puma5=df["puma5"].astype(str).to_numpy(),
    )

    tvd = 0.5 * np.abs(p_hier - p_true_all).sum(axis=1)
    summary = {
        "output_dir": str(out_dir),
        "npz": str(out_npz),
        "key": key,
        "seed": int(args.seed),
        "heldout_statefp": str(args.heldout_statefp).zfill(2),
        "n_eval_joint_samples": int(cfg.stage2_n_eval_joint_samples),
        "ipf_iters": int(cfg.ipf_iters),
        "device": device,
        "n_pumas": int(p_hier.shape[0]),
        "tvd_mean_all_pumas": float(np.mean(tvd)),
        "tvd_median_all_pumas": float(np.median(tvd)),
        "tvd_mean_heldout_state": float(np.mean(tvd[df["statefp"].astype(str).to_numpy() == str(args.heldout_statefp).zfill(2)])),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
