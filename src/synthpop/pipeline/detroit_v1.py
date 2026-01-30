from __future__ import annotations

"""
Detroit v1 pipeline (Scheme C-v2 draft).

This file is intentionally a *thin orchestration layer*.
It does not hard-code training details; instead it wires together:
- Veraset loaders -> device features
- encoders/alignment -> shared latent space
- joint diffusion -> sampling
- validation -> metrics

For now we keep this as a dry-run/status helper until the data contract is finalized.
"""

import json
import pathlib
from typing import Any

from ..detroit.paths import detroit_root


def status_v1(*, data_root: pathlib.Path) -> dict[str, Any]:
    """
    Lightweight "do we have the expected inputs" snapshot for Scheme C-v2.
    Only checks paths exist (no heavy validation by design).
    """
    det = detroit_root(data_root)
    mobility = det / "raw" / "mobility" / "veraset"

    def _exists(p: pathlib.Path) -> bool:
        try:
            return p.exists()
        except OSError:
            return False

    return {
        "data_root": str(data_root),
        "detroit_root": str(det),
        "raw": {
            "mobility_veraset": {
                "root": str(mobility),
                "exists": _exists(mobility),
                "home_dir": {"path": str(mobility / "home"), "exists": _exists(mobility / "home")},
                "work_dir": {"path": str(mobility / "work"), "exists": _exists(mobility / "work")},
                "visits_dir": {"path": str(mobility / "visits"), "exists": _exists(mobility / "visits")},
                "meta": {"path": str(mobility / "dewey.metadata.json"), "exists": _exists(mobility / "dewey.metadata.json")},
            }
        },
    }


def print_status_v1(*, data_root: pathlib.Path) -> None:
    print(json.dumps(status_v1(data_root=data_root), ensure_ascii=False, indent=2))


def run_detroit_v1(*, data_root: pathlib.Path, config: dict[str, Any] | None = None, dry_run: bool = True) -> dict[str, Any]:
    """
    Draft entrypoint for Scheme C-v2.

    When dry_run=True, returns a status snapshot.
    When dry_run=False, this will later execute:
    1) encoder pretraining
    2) alignment learning
    3) training pair construction
    4) joint diffusion training
    5) guided sampling + assignment
    6) validation
    """
    _ = config or {}
    if dry_run:
        return status_v1(data_root=data_root)
    raise NotImplementedError("TODO(Scheme C-v2): implement full Detroit v1 pipeline after data contract is locked.")

