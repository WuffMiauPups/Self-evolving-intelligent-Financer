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

