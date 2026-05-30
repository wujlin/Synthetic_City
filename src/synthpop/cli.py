from __future__ import annotations

import argparse
import json
import os

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
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthpop")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_paths = sub.add_parser("paths", help="Print resolved project/data paths as JSON.")
    p_paths.set_defaults(func=_cmd_paths)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
