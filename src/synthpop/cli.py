from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

from .detroit.constants import DEFAULT_TIGER_YEAR, STATEFP_MI
from .detroit.paths import tiger_dir
from .detroit.stage_tiger import stage_tiger_zips
from .pipeline.detroit_v0 import init_dirs as detroit_init_dirs
from .pipeline.detroit_v0 import print_status as detroit_print_status
from .paths import data_root, project_root


def _cmd_paths(_: argparse.Namespace) -> None:
    root = project_root()
    droot = data_root()
    info = {
        "project_root": str(root),
        "data_root": str(droot),
        "env": {
            "RAW_ROOT": os.environ.get("RAW_ROOT"),
            "SYNTHCITY_DATA_ROOT": os.environ.get("SYNTHCITY_DATA_ROOT"),
        },
        "detroit": {
            "safegraph_symlink": str(droot / "detroit" / "raw" / "poi" / "safegraph" / "safegraph_unzip"),
            "safegraph_meta": str(droot / "detroit" / "raw" / "poi" / "safegraph" / "safegraph.metadata.json"),
            "tiger_2023_dir": str(droot / "detroit" / "raw" / "geo" / "tiger" / "TIGER2023"),
        },
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))


def _cmd_detroit_stage_tiger(args: argparse.Namespace) -> None:
    """
    Stage the 4 TIGER zip files into the canonical folder layout.
    Default behavior is non-destructive: copy (not move).
    """
    year = int(args.tiger_year)
    statefp = args.statefp
    src_dir = pathlib.Path(args.src_dir).expanduser().resolve()
    dst_dir = pathlib.Path(args.dst_dir).expanduser().resolve()
    mode = args.mode
    overwrite = args.overwrite

    staged = stage_tiger_zips(
        tiger_year=year,
        statefp=statefp,
        src_dir=src_dir,
        dst_dir=dst_dir,
        mode=mode,
        overwrite=overwrite,
    )

    if len(staged) < 4:
        print("[warn] Some TIGER zips are missing in src_dir (expected 4).", file=sys.stderr)
        print(f"[hint] expected names: tl_{year}_{statefp}_{{place,tract,bg,puma20}}.zip", file=sys.stderr)
    print(f"[ok] staged {len(staged)}/4 TIGER zip(s) to: {dst_dir}", file=sys.stderr)


def _cmd_detroit_status(args: argparse.Namespace) -> None:
    droot = pathlib.Path(args.data_root).expanduser().resolve()
    detroit_print_status(data_root=droot)


def _cmd_detroit_init_dirs(args: argparse.Namespace) -> None:
    droot = pathlib.Path(args.data_root).expanduser().resolve()
    created = detroit_init_dirs(data_root=droot)
    print(json.dumps({"created": created}, ensure_ascii=False, indent=2))


def _run_poc_tabddpm_pums(*, mode: str, args: argparse.Namespace) -> None:
    script = project_root() / "tools" / "poc_tabddpm_pums.py"
    if not script.exists():
        raise SystemExit(f"PoC script not found: {script}")

    env = os.environ.copy()
    repo_root = str(project_root())
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(script),
        "--mode",
        mode,
        "--data_root",
        str(pathlib.Path(args.data_root).expanduser().resolve()),
        "--pums_year",
        str(args.pums_year),
        "--pums_period",
        str(args.pums_period),
        "--statefp",
        str(args.statefp),
        "--n_rows",
        str(args.n_rows),
        "--n_samples",
        str(args.n_samples),
        "--timesteps",
        str(args.timesteps),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--out_dir",
        str(args.out_dir),
        "--log_every",
        str(args.log_every),
    ]

    if args.device:
        cmd += ["--device", str(args.device)]
    if args.ckpt_path:
        cmd += ["--ckpt_path", str(args.ckpt_path)]
    if args.encoder_path:
        cmd += ["--encoder_path", str(args.encoder_path)]

    subprocess.run(cmd, check=True, cwd=repo_root, env=env)


def _cmd_detroit_poc_train(args: argparse.Namespace) -> None:
    _run_poc_tabddpm_pums(mode="train", args=args)


def _cmd_detroit_poc_sample(args: argparse.Namespace) -> None:
    _run_poc_tabddpm_pums(mode="sample", args=args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="synthpop")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_paths = sub.add_parser("paths", help="Print resolved project/data paths (JSON).")
    p_paths.set_defaults(func=_cmd_paths)

    p_detroit = sub.add_parser("detroit", help="Detroit helper commands (KISS).")
    det_sub = p_detroit.add_subparsers(dest="det_cmd", required=True)

    p_stage = det_sub.add_parser("stage-tiger", help="Copy/symlink TIGER zips into canonical folder.")
    p_stage.add_argument("--tiger_year", default=str(DEFAULT_TIGER_YEAR))
    p_stage.add_argument("--statefp", default=STATEFP_MI, help="State FIPS (MI=26).")
    p_stage.add_argument(
        "--src_dir",
        default=str(data_root()),
        help="Source dir containing downloaded tl_<year>_26_*.zip files.",
    )
    p_stage.add_argument(
        "--dst_dir",
        default=str(tiger_dir(data_root(), tiger_year=DEFAULT_TIGER_YEAR)),
        help="Destination TIGER directory (canonical layout).",
    )
    p_stage.add_argument("--mode", choices=["copy", "symlink"], default="copy", help="Default: copy (non-destructive).")
    p_stage.add_argument("--overwrite", action="store_true", help="Overwrite existing staged files.")
    p_stage.set_defaults(func=_cmd_detroit_stage_tiger)

    p_status = det_sub.add_parser("status", help="Print Detroit raw data status (JSON).")
    p_status.add_argument(
        "--data_root",
        default=str(data_root()),
        help="Data root (default resolved from SYNTHCITY_DATA_ROOT/RAW_ROOT/<repo>/data).",
    )
    p_status.set_defaults(func=_cmd_detroit_status)

    p_init = det_sub.add_parser("init-dirs", help="Create canonical Detroit folders (non-destructive).")
    p_init.add_argument(
        "--data_root",
        default=str(data_root()),
        help="Data root (default resolved from SYNTHCITY_DATA_ROOT/RAW_ROOT/<repo>/data).",
    )
    p_init.set_defaults(func=_cmd_detroit_init_dirs)

    p_poc_train = det_sub.add_parser("poc-train", help="PoC train (TabDDPM-style) on PUMS person subset.")
    p_poc_train.add_argument("--data_root", default=str(data_root()))
    p_poc_train.add_argument("--pums_year", default="2023")
    p_poc_train.add_argument("--pums_period", default="5-Year")
    p_poc_train.add_argument("--statefp", default=STATEFP_MI)
    p_poc_train.add_argument("--n_rows", default="200000")
    p_poc_train.add_argument("--n_samples", default="50000", help="Ignored in train-only; kept for shared wrapper.")
    p_poc_train.add_argument("--timesteps", default="200")
    p_poc_train.add_argument("--epochs", default="5")
    p_poc_train.add_argument("--batch_size", default="2048")
    p_poc_train.add_argument("--device", default=None)
    p_poc_train.add_argument("--seed", default="0")
    p_poc_train.add_argument("--out_dir", default="outputs/_poc_tabddpm_pums")
    p_poc_train.add_argument("--ckpt_path", default=None)
    p_poc_train.add_argument("--encoder_path", default=None)
    p_poc_train.add_argument("--log_every", default="200")
    p_poc_train.set_defaults(func=_cmd_detroit_poc_train)

    p_poc_sample = det_sub.add_parser("poc-sample", help="PoC sample: load checkpoint and generate samples.")
    p_poc_sample.add_argument("--data_root", default=str(data_root()), help="Kept for wrapper parity (not used in sample).")
    p_poc_sample.add_argument("--pums_year", default="2023", help="Kept for wrapper parity (not used in sample).")
    p_poc_sample.add_argument("--pums_period", default="5-Year", help="Kept for wrapper parity (not used in sample).")
    p_poc_sample.add_argument("--statefp", default=STATEFP_MI, help="Kept for wrapper parity (not used in sample).")
    p_poc_sample.add_argument("--n_rows", default="200000", help="Kept for wrapper parity (not used in sample).")
    p_poc_sample.add_argument("--n_samples", default="50000")
    p_poc_sample.add_argument("--timesteps", default="200", help="Kept for wrapper parity (not used in sample).")
    p_poc_sample.add_argument("--epochs", default="5", help="Kept for wrapper parity (not used in sample).")
    p_poc_sample.add_argument("--batch_size", default="2048", help="Kept for wrapper parity (not used in sample).")
    p_poc_sample.add_argument("--device", default=None)
    p_poc_sample.add_argument("--seed", default="0")
    p_poc_sample.add_argument("--out_dir", default="outputs/_poc_tabddpm_pums")
    p_poc_sample.add_argument("--ckpt_path", default=None, help="Default: <out_dir>/model.pt")
    p_poc_sample.add_argument("--encoder_path", default=None, help="Default: <out_dir>/encoder.json")
    p_poc_sample.add_argument("--log_every", default="200", help="Kept for wrapper parity (not used in sample).")
    p_poc_sample.set_defaults(func=_cmd_detroit_poc_sample)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
