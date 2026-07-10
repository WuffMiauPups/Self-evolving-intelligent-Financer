"""Per-decision observation replay + RL experience scaffolding.

``DecisionObservation`` / ``ObservationBuffer`` are the retrain loop's
outcome stream (MASTER_PLAN §4.5.3): one record per executed decision,
persisted as append-friendly JSONL in the style of ``src.risk.ledger``.
The execution layer appends records in a parallel workstream;
``scripts/run_train.py --check-retrain-trigger`` reads them back through
``load_observations``.

``Experience`` / ``ReplayBuffer`` remain the RL fine-tune scaffold
(reserved; consumed only by ``rl_scaffold``).
"""
from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionObservation:
    """One executed decision and its realized outcome.

    Records the EXECUTED (risk-clamped) action, not the proposal — the
    loop learns from what actually happened (standing rule 4).
    """

    decision_id: str
    symbol: str
    timestamp: datetime
    features_ref: str  # feature-snapshot fingerprint or artifact path
    executed_action: int  # -1 short / 0 flat / +1 long, as executed
    realized_pnl: float  # net of fees
    clamped: bool  # risk layer resized/vetoed the proposal

    def to_json_line(self) -> str:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return json.dumps(payload, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict) -> DecisionObservation:
        return cls(
            decision_id=str(payload["decision_id"]),
            symbol=str(payload["symbol"]),
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            features_ref=str(payload.get("features_ref", "")),
            executed_action=int(payload["executed_action"]),
            realized_pnl=float(payload.get("realized_pnl", 0.0)),
            clamped=bool(payload.get("clamped", False)),
        )


class ObservationBuffer:
    """Bounded, persistable buffer of per-decision observations."""

    def __init__(self, capacity: int = 10_000) -> None:
        if capacity <= 0:
            raise ValueError("ObservationBuffer capacity must be positive")
        self.capacity = capacity
        self._items: deque[DecisionObservation] = deque(maxlen=capacity)

    def add(self, observation: DecisionObservation) -> None:
        self._items.append(observation)

    def extend(self, observations: Iterable[DecisionObservation]) -> None:
        for observation in observations:
            self.add(observation)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def observations(self) -> list[DecisionObservation]:
        return list(self._items)

    def save(self, path: str | Path) -> int:
        """Persist the buffered observations as JSONL (snapshot rewrite)."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as fp:
            for observation in self._items:
                fp.write(observation.to_json_line() + "\n")
        logger.info(
            "observation_buffer_saved path=%s count=%d", str(file_path), len(self._items)
        )
        return len(self._items)

    @classmethod
    def load(cls, path: str | Path, *, capacity: int = 10_000) -> ObservationBuffer:
        """Rehydrate a buffer from JSONL (torn/corrupt lines skipped)."""
        buffer = cls(capacity=capacity)
        buffer.extend(load_observations(path))
        return buffer


def load_observations(path: str | Path) -> list[DecisionObservation]:
    """Read a JSONL observation stream (ledger-style: torn lines skipped)."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    observations: list[DecisionObservation] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            observations.append(DecisionObservation.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return observations


def append_observation(path: str | Path, observation: DecisionObservation) -> None:
    """Append one observation to the JSONL stream (best-effort, ledger-style)."""
    try:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("a", encoding="utf-8") as fp:
            fp.write(observation.to_json_line() + "\n")
    except Exception as exc:  # noqa: BLE001 - persistence must not kill trading
        logger.warning("observation_persist_failed path=%s error=%s", path, exc)


# ---------------------------------------------------------------------------
# RL experience scaffold (reserved; consumed only by rl_scaffold)
# ---------------------------------------------------------------------------


@dataclass
class Experience:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: deque[Experience] = deque(maxlen=capacity)

    def push(self, experience: Experience) -> None:
        self._items.append(experience)

    def __len__(self) -> int:
        return len(self._items)

    def sample(self, batch_size: int, *, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        if batch_size > len(self._items):
            raise ValueError("Not enough items in replay buffer")
        indices = torch.randperm(len(self._items))[:batch_size].tolist()
        items = [self._items[index] for index in indices]
        batch = {
            "observation": torch.stack([item.observation for item in items]),
            "action": torch.stack([item.action for item in items]),
            "reward": torch.stack([item.reward for item in items]),
            "next_observation": torch.stack([item.next_observation for item in items]),
            "done": torch.stack([item.done for item in items]),
        }
        if device is not None:
            batch = {key: value.to(device) for key, value in batch.items()}
        return batch
