from __future__ import annotations

import json
import pathlib
from typing import Any


def load_config(path: str | pathlib.Path) -> dict[str, Any]:
    """
    Load a small experiment config.

    KISS policy:
    - JSON is always supported.
    - YAML is optional (only if PyYAML is installed).
    """
    p = pathlib.Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))

    if p.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("YAML config requires PyYAML (pip install pyyaml).") from e
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    raise ValueError(f"Unsupported config extension: {p.suffix} (use .json/.yaml)")

