from __future__ import annotations

import os
import pathlib


def project_root() -> pathlib.Path:
    # src/synthpop/paths.py -> repo root is 2 parents up.
    return pathlib.Path(__file__).resolve().parents[2]


def data_root() -> pathlib.Path:
    """
    Data root resolution (KISS):
    1) SYNTHCITY_DATA_ROOT (explicit)
    2) RAW_ROOT/synthetic_city/data (wsA convention)
    3) <repo>/data (local convention; usually a symlink)
    """
    explicit = os.environ.get("SYNTHCITY_DATA_ROOT")
    if explicit:
        return pathlib.Path(explicit).expanduser().resolve()

    raw_root = os.environ.get("RAW_ROOT")
    if raw_root:
        return (pathlib.Path(raw_root).expanduser() / "synthetic_city" / "data").resolve()

    return project_root() / "data"


def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

