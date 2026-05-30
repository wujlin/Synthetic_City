#!/usr/bin/env python3
from __future__ import annotations

"""Release: upload release-format state files to OSF."""

import pathlib
import runpy


if __name__ == "__main__":
    runpy.run_path(str(pathlib.Path(__file__).resolve().with_name("upload_osf_release_incremental.py")), run_name="__main__")
