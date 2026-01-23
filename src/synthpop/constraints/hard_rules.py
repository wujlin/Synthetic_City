from __future__ import annotations

"""
Hard rules / structural zeros / hierarchical consistency.

v0: define a minimal, reviewable interface; implement rules incrementally.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuleViolation:
    rule_id: str
    message: str
    count: int


def check_rules(*, households: Any | None, persons: Any, rules: Any) -> list[RuleViolation]:
    raise NotImplementedError("TODO(v0): rule engine (age-employment, household size consistency, etc.).")

