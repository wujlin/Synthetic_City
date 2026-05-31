from __future__ import annotations

"""
Projection / local resampling to enforce hard constraints during sampling.

Key idea: minimal necessary edits to bring samples back to the feasible set,
repeated inside the sampling loop (not post-hoc filtering).
"""

from typing import Any


def project_to_feasible(*, households: Any | None, persons: Any, rules: Any) -> tuple[Any | None, Any]:
    raise NotImplementedError("TODO(v0): projection or local resampling for constraint violations.")

