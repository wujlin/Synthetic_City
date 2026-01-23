from __future__ import annotations

import pathlib
import shutil


def stage_tiger_zips(
    *,
    tiger_year: int,
    statefp: str,
    src_dir: pathlib.Path,
    dst_dir: pathlib.Path,
    mode: str = "copy",
    overwrite: bool = False,
) -> list[pathlib.Path]:
    """
    Stage the 4 TIGER zip files into a canonical folder.

    Returns the list of files successfully staged.
    """
    names = [
        f"tl_{tiger_year}_{statefp}_place.zip",
        f"tl_{tiger_year}_{statefp}_tract.zip",
        f"tl_{tiger_year}_{statefp}_bg.zip",
        f"tl_{tiger_year}_{statefp}_puma20.zip",
    ]

    dst_dir.mkdir(parents=True, exist_ok=True)

    staged: list[pathlib.Path] = []
    for fname in names:
        src = src_dir / fname
        if not src.exists():
            continue

        dst = dst_dir / fname
        if dst.exists() or dst.is_symlink():
            if not overwrite:
                staged.append(dst)
                continue
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                raise RuntimeError(f"refuse to overwrite non-file: {dst}")

        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "symlink":
            dst.symlink_to(src)
        else:
            raise ValueError(f"unknown mode: {mode}")

        staged.append(dst)

    return staged

