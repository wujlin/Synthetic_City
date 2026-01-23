from __future__ import annotations

import pathlib


def detroit_root(data_root: pathlib.Path) -> pathlib.Path:
    return data_root / "detroit"


def tiger_dir(data_root: pathlib.Path, *, tiger_year: int) -> pathlib.Path:
    return detroit_root(data_root) / "raw" / "geo" / "tiger" / f"TIGER{tiger_year}"


def safegraph_dir(data_root: pathlib.Path) -> pathlib.Path:
    return detroit_root(data_root) / "raw" / "poi" / "safegraph"

