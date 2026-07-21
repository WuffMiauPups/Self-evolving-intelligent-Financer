from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    SHORT = 0
    FLAT = 1
    LONG = 2


ACTION_TO_SIGN: dict[Action, int] = {
    Action.SHORT: -1,
    Action.FLAT: 0,
    Action.LONG: 1,
}

DEFAULT_FEATURE_DTYPE = "float32"
ACCOUNTING_DTYPE = "float64"

# Which quantity a proposal's ``confidence`` carries (2026-07-21 fix,
# CONFIDENCE_AUDIT.md): softmax P(resolved action class) computed in
# BrainPolicy._build_proposal. Ledger rows and Platt artifacts carry this
# tag so consumers can refuse pre-fix quantities (legacy sigmoid head).
CONFIDENCE_SEMANTICS = "p_resolved_class"

