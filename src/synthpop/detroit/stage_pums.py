from __future__ import annotations

import pathlib
import shutil


def stage_pums_zips(
    *,
    pums_year: int,
    pums_period: str,
    statefp: str,
    src_dir: pathlib.Path,
    dst_dir: pathlib.Path,
    mode: str = "copy",
    overwrite: bool = False,
) -> list[pathlib.Path]:
    """
    Stage PUMS zips into a canonical folder.

    Detroit v0 only supports MI (26) for now.
    Returns the list of files successfully staged.
    """
    statefp = str(statefp).zfill(2)
    state_postal_lower = "mi" if statefp == "26" else None
    if state_postal_lower is None:
        raise ValueError(f"Unsupported statefp for Detroit v0: {statefp}")

    names = [
        # Older naming
        f"psam_h{statefp}.zip",
        f"psam_p{statefp}.zip",
        # Newer naming (postal abbreviation)
        f"csv_h{state_postal_lower}.zip",
        f"csv_p{state_postal_lower}.zip",
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

