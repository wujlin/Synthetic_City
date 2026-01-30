from __future__ import annotations

"""
ACS cross-tab helpers (Scheme C-v2).

This is reserved for v1+ where we may need joint constraints beyond simple marginals.
Example: income x tenure, age x sex, etc.
"""

from typing import Any


def load_acs_crosstab(*_: Any, **__: Any) -> Any:
    raise NotImplementedError("TODO(Scheme C-v2): implement ACS cross-tab loading once target tables are finalized.")

