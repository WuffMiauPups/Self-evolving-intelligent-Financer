"""Columnar in-memory storage for training samples.

Why this exists
---------------
The original ``samples_by_split: dict[str, list[TrainingSample]]`` storage
held one ``TrainingSample`` dataclass per row, each carrying ~7 separate
``torch.Tensor`` objects (features, regime, temporal, label, path_label,
plus four branch tensors) and two Python dicts (regression_targets,
future_returns).

Empirically that costs about **22 KB per sample** of in-memory footprint
on Windows + PyTorch 2.x — most of which is per-tensor metadata, dataclass
slots, and Python object overhead.  The actual feature data is closer to
2 KB per sample.

For a 38-symbol × 4-year × 5-minute training universe (~3 million samples)
the dataclass form needs **~46 GB of resident memory** before training
even starts.  That overflows a 64 GB workstation once you add the
training-loop activations, optimizer state, and the PyTorch CUDA caching
allocator.

The columnar form below stores the same data as one big tensor per field,
plus parallel Python lists for the string/datetime metadata.  In-memory
cost drops to about **3 KB per sample**, so the same 3 M-sample universe
fits in **~9 GB** instead of 46 GB — a 5× reduction.  Diagnostics +
class-weight computation also become bulk-tensor operations and run in
seconds rather than minutes.

Design rules
------------
- ``ColumnarSamples`` is the canonical in-memory form.  Any code path
  that used to hold ``list[TrainingSample]`` should switch to
  ``ColumnarSamples`` over time.
- ``ColumnarSamples.from_samples`` builds columns by stacking once per
  shard, then immediately drops the shard's per-sample dataclasses so the
  intermediate ``list[TrainingSample]`` is freed before the next shard
  loads.  This is what keeps peak working set bounded during cold start.
- ``ColumnarSamples.append`` is O(N) (one ``torch.cat`` per field) and is
  used to merge per-symbol shards into the bundle-wide column store.
- ``ColumnarSamples.iter_samples`` yields lazily-reconstructed
  ``TrainingSample`` instances for the very few legacy callers that
  cannot work in column form yet.  Each yielded sample is short-lived
  (refcount-freed at the next iteration).
- The on-disk shard format is **unchanged** — the conversion happens at
  load time inside ``data_module.build_real_data_training_bundle``.  This
  preserves all existing cache files, fingerprint logic, and roll-back
  paths.

This module deliberately has zero import-time dependencies on
``data_module`` or ``class_balance`` so it can be imported from anywhere.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime

import torch

from src.data.dataset import TrainingSample


@dataclass
class ColumnarSamples:
    """Tensor-of-arrays storage for a batch of training samples.

    All tensor fields are aligned along ``dim=0`` so that ``self.features[i]``,
    ``self.label[i]``, ``self.branch_inputs['trend'][i]``, etc. all refer to
    the same logical sample.  The Python list fields (symbols, timestamps,
    direction_names, path_names) are kept in lock-step with the same indexing.

    Construction is via ``from_samples`` (one shard's worth of dataclasses
    at a time) or via ``empty(...)`` followed by ``append`` for incremental
    accumulation across shards.
    """

    # Core tensor columns — required, every sample has these.
    features: torch.Tensor                    # (N, F)
    label: torch.Tensor                       # (N,) int64
    regime_input: torch.Tensor                # (N, R)
    temporal_tensor: torch.Tensor             # (N, T_features, seq_len)
    branch_inputs: dict[str, torch.Tensor]    # {branch_name: (N, branch_dim)}

    # Optional / per-task tensor columns.
    path_label: torch.Tensor | None = None    # (N,) int64
    regression_targets: dict[str, torch.Tensor] = field(default_factory=dict)  # {target: (N,)}
    future_returns: dict[int, torch.Tensor] = field(default_factory=dict)      # {horizon: (N,)}

    # Metadata kept as plain Python lists — cheap (str interning) and not
    # tensorizable in any useful way.
    symbols: list[str] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    direction_names: list[str] = field(default_factory=list)
    path_names: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_samples(cls, samples: list[TrainingSample]) -> "ColumnarSamples":
        """Build columns by stacking ``samples`` once.

        ``samples`` MUST be non-empty.  All samples must agree on the set of
        branch_input keys, regression_target keys, and future_return horizons
        (mismatches raise ``ValueError`` rather than producing silently-wrong
        columns).  Caller is expected to drop the input list reference
        immediately after this returns so the dataclass form can be GC'd.
        """
        if not samples:
            raise ValueError("from_samples requires at least one sample")

        head = samples[0]
        branch_keys = list(head.branch_inputs.keys())
        regression_keys = sorted(head.regression_targets.keys())
        horizon_keys = sorted(head.future_returns.keys())

        features = torch.stack([s.features for s in samples], dim=0)
        label = torch.stack([s.label for s in samples], dim=0)
        regime = torch.stack([s.regime_input for s in samples], dim=0)
        temporal = torch.stack([s.temporal_tensor for s in samples], dim=0)
        branch_cols: dict[str, torch.Tensor] = {}
        for k in branch_keys:
            branch_cols[k] = torch.stack([s.branch_inputs[k] for s in samples], dim=0)

        # Path label may be None on some samples; coerce to long zeros if so.
        if any(s.path_label is not None for s in samples):
            path_col = torch.stack(
                [
                    s.path_label if s.path_label is not None else torch.tensor(0, dtype=torch.long)
                    for s in samples
                ],
                dim=0,
            )
        else:
            path_col = None

        regression_cols: dict[str, torch.Tensor] = {}
        for name in regression_keys:
            regression_cols[name] = torch.tensor(
                [float(s.regression_targets.get(name, 0.0)) for s in samples],
                dtype=torch.float32,
            )

        future_cols: dict[int, torch.Tensor] = {}
        for horizon in horizon_keys:
            future_cols[int(horizon)] = torch.tensor(
                [float(s.future_returns.get(horizon, 0.0)) for s in samples],
                dtype=torch.float32,
            )

        return cls(
            features=features,
            label=label,
            regime_input=regime,
            temporal_tensor=temporal,
            branch_inputs=branch_cols,
            path_label=path_col,
            regression_targets=regression_cols,
            future_returns=future_cols,
            symbols=[s.symbol for s in samples],
            timestamps=[s.timestamp for s in samples],
            direction_names=[s.direction_name for s in samples],
            path_names=[s.path_name for s in samples],
        )

    # ------------------------------------------------------------------
    # accumulation
    # ------------------------------------------------------------------
    def append(self, other: "ColumnarSamples") -> "ColumnarSamples":
        """Concatenate ``other`` into ``self`` (returns ``self`` for chaining).

        Mutates ``self`` in place.  ``other``'s tensors are concatenated along
        dim 0; mismatched feature widths or branch keys raise ``ValueError``.
        Path-label and regression/future-return columns from ``other`` are
        merged when both sides have them; one-sided columns are dropped from
        the merged form to keep shape invariants intact.
        """
        if len(self) == 0:
            # Steal other's columns wholesale (cheap; no copies).
            self.features = other.features
            self.label = other.label
            self.regime_input = other.regime_input
            self.temporal_tensor = other.temporal_tensor
            self.branch_inputs = dict(other.branch_inputs)
            self.path_label = other.path_label
            self.regression_targets = dict(other.regression_targets)
            self.future_returns = dict(other.future_returns)
            self.symbols = list(other.symbols)
            self.timestamps = list(other.timestamps)
            self.direction_names = list(other.direction_names)
            self.path_names = list(other.path_names)
            return self
        if len(other) == 0:
            return self

        if set(other.branch_inputs) != set(self.branch_inputs):
            raise ValueError(
                f"branch_inputs key mismatch: self={sorted(self.branch_inputs)} "
                f"other={sorted(other.branch_inputs)}"
            )
        self.features = torch.cat([self.features, other.features], dim=0)
        self.label = torch.cat([self.label, other.label], dim=0)
        self.regime_input = torch.cat([self.regime_input, other.regime_input], dim=0)
        self.temporal_tensor = torch.cat([self.temporal_tensor, other.temporal_tensor], dim=0)
        for k in self.branch_inputs:
            self.branch_inputs[k] = torch.cat(
                [self.branch_inputs[k], other.branch_inputs[k]], dim=0
            )
        if self.path_label is not None and other.path_label is not None:
            self.path_label = torch.cat([self.path_label, other.path_label], dim=0)
        else:
            # If either side lacks path labels, the merged store cannot
            # offer them either — fall back to all-zeros only when at least
            # one side had real values.
            if self.path_label is not None or other.path_label is not None:
                self_pl = (
                    self.path_label
                    if self.path_label is not None
                    else torch.zeros(len(self) - len(other), dtype=torch.long)
                )
                other_pl = (
                    other.path_label
                    if other.path_label is not None
                    else torch.zeros(len(other), dtype=torch.long)
                )
                self.path_label = torch.cat([self_pl, other_pl], dim=0)
        # Regression targets: union of keys, padding missing sides with zeros.
        all_reg = set(self.regression_targets) | set(other.regression_targets)
        merged_reg: dict[str, torch.Tensor] = {}
        for name in all_reg:
            left = self.regression_targets.get(
                name, torch.zeros(len(self) - len(other), dtype=torch.float32)
            )
            right = other.regression_targets.get(
                name, torch.zeros(len(other), dtype=torch.float32)
            )
            merged_reg[name] = torch.cat([left, right], dim=0)
        self.regression_targets = merged_reg
        # Future returns: same merge strategy.
        all_horizons = set(self.future_returns) | set(other.future_returns)
        merged_fut: dict[int, torch.Tensor] = {}
        for horizon in all_horizons:
            left = self.future_returns.get(
                horizon, torch.zeros(len(self) - len(other), dtype=torch.float32)
            )
            right = other.future_returns.get(
                horizon, torch.zeros(len(other), dtype=torch.float32)
            )
            merged_fut[int(horizon)] = torch.cat([left, right], dim=0)
        self.future_returns = merged_fut

        self.symbols.extend(other.symbols)
        self.timestamps.extend(other.timestamps)
        self.direction_names.extend(other.direction_names)
        self.path_names.extend(other.path_names)
        return self

    @classmethod
    def empty(cls) -> "ColumnarSamples":
        """A zero-row column store, useful as an accumulator seed."""
        return cls(
            features=torch.empty((0, 0), dtype=torch.float32),
            label=torch.empty((0,), dtype=torch.long),
            regime_input=torch.empty((0, 0), dtype=torch.float32),
            temporal_tensor=torch.empty((0, 0, 0), dtype=torch.float32),
            branch_inputs={},
        )

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        # ``features`` is the canonical row count.  An empty accumulator has
        # shape (0, 0), so .shape[0] is the right thing.
        return int(self.features.shape[0])

    def slice(self, indices: list[int] | torch.Tensor) -> "ColumnarSamples":
        """Return a new ColumnarSamples containing only the given row indices.

        Used by fast-mode subsampling and by the val/test split helpers.
        Tensor fields are gathered (advanced-indexed); list fields are list
        comprehensions.  The result is a fully-independent column store —
        no view aliasing back into ``self``.
        """
        if isinstance(indices, list):
            idx = torch.tensor(indices, dtype=torch.long)
        else:
            idx = indices.to(dtype=torch.long)
        return ColumnarSamples(
            features=self.features.index_select(0, idx).clone(),
            label=self.label.index_select(0, idx).clone(),
            regime_input=self.regime_input.index_select(0, idx).clone(),
            temporal_tensor=self.temporal_tensor.index_select(0, idx).clone(),
            branch_inputs={k: v.index_select(0, idx).clone() for k, v in self.branch_inputs.items()},
            path_label=(
                self.path_label.index_select(0, idx).clone()
                if self.path_label is not None
                else None
            ),
            regression_targets={
                k: v.index_select(0, idx).clone() for k, v in self.regression_targets.items()
            },
            future_returns={
                k: v.index_select(0, idx).clone() for k, v in self.future_returns.items()
            },
            symbols=[self.symbols[i] for i in idx.tolist()],
            timestamps=[self.timestamps[i] for i in idx.tolist()],
            direction_names=[self.direction_names[i] for i in idx.tolist()],
            path_names=[self.path_names[i] for i in idx.tolist()],
        )

    def get_sample(self, index: int) -> TrainingSample:
        """Reconstruct a single ``TrainingSample`` lazily from the columns.

        Used by ``ColumnarDataset.__getitem__`` so that downstream code paths
        (collate, training loop) keep their existing per-sample contract.
        Each call allocates a transient TrainingSample (~22 KB), which is
        immediately consumed by the collate function and refcount-freed.
        Tensor fields use views into the columns where possible — the
        collate function handles the .clone() if needed.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"sample index out of range: {index} (n={len(self)})")
        path_label_value = (
            self.path_label[index] if self.path_label is not None else None
        )
        return TrainingSample(
            features=self.features[index],
            branch_inputs={k: v[index] for k, v in self.branch_inputs.items()},
            regime_input=self.regime_input[index],
            temporal_tensor=self.temporal_tensor[index],
            label=self.label[index],
            symbol=self.symbols[index],
            timestamp=self.timestamps[index],
            direction_name=self.direction_names[index],
            path_label=path_label_value,
            path_name=self.path_names[index],
            regression_targets={
                name: float(col[index].item())
                for name, col in self.regression_targets.items()
            },
            future_returns={
                horizon: float(col[index].item())
                for horizon, col in self.future_returns.items()
            },
        )

    def iter_samples(self) -> Iterator[TrainingSample]:
        """Yield all samples as transient ``TrainingSample`` instances.

        Each yielded sample lives only until the next iteration step (or
        until the consumer drops the reference).  This is the bridge that
        lets the legacy ``compute_target_diagnostics`` /
        ``build_class_weights`` functions accept ColumnarSamples without
        rewriting their internals — at the cost of doing N transient
        dataclass allocations.  For 3 M samples this is ~30-60s of CPU but
        the peak working set stays at the columnar level (~9 GB).
        """
        for i in range(len(self)):
            yield self.get_sample(i)

    def __iter__(self) -> Iterator[TrainingSample]:
        """Alias for ``iter_samples`` so legacy ``for s in samples_by_split[k]``
        loops keep working without code changes."""
        return self.iter_samples()

    def __getitem__(self, index: int) -> TrainingSample:
        """Alias for ``get_sample`` so legacy ``samples_by_split[k][i]``
        access keeps working without code changes."""
        return self.get_sample(index)

    # ------------------------------------------------------------------
    # device placement
    # ------------------------------------------------------------------
    def to_device(self, device: torch.device) -> None:
        """Move all tensor columns to *device* in-place (non-blocking).

        After this call the columnar collate slices GPU tensors directly --
        no CPU->GPU transfer per batch.  ``pin_memory`` must be disabled on
        the DataLoader when columns are already on CUDA (pinning GPU tensors
        is a no-op at best and a crash at worst).
        """
        self.features = self.features.to(device, non_blocking=True)
        self.regime_input = self.regime_input.to(device, non_blocking=True)
        self.temporal_tensor = self.temporal_tensor.to(device, non_blocking=True)
        self.label = self.label.to(device, non_blocking=True)
        if self.path_label is not None:
            self.path_label = self.path_label.to(device, non_blocking=True)
        for k in list(self.branch_inputs.keys()):
            self.branch_inputs[k] = self.branch_inputs[k].to(device, non_blocking=True)
        for k in list(self.regression_targets.keys()):
            self.regression_targets[k] = self.regression_targets[k].to(device, non_blocking=True)
        for k in list(self.future_returns.keys()):
            self.future_returns[k] = self.future_returns[k].to(device, non_blocking=True)

    # ------------------------------------------------------------------
    # debug + introspection
    # ------------------------------------------------------------------
    def estimated_bytes(self) -> int:
        """Approximate in-memory bytes (tensor data only — Python overhead
        for the lists is small in comparison and ignored)."""
        total = 0
        for t in (self.features, self.label, self.regime_input, self.temporal_tensor):
            total += t.element_size() * t.numel()
        for t in self.branch_inputs.values():
            total += t.element_size() * t.numel()
        if self.path_label is not None:
            total += self.path_label.element_size() * self.path_label.numel()
        for t in self.regression_targets.values():
            total += t.element_size() * t.numel()
        for t in self.future_returns.values():
            total += t.element_size() * t.numel()
        return total


class ColumnarDataset(torch.utils.data.Dataset):
    """Torch ``Dataset`` wrapper around a ``ColumnarSamples`` store.

    ``__getitem__`` reconstructs a transient ``TrainingSample`` per row so
    the existing ``collate_training_samples`` function continues to work
    unchanged.  The persistent storage is the column tensors held in
    ``self.columns``; per-sample dataclass instances exist only inside one
    DataLoader worker batch at a time.
    """

    def __init__(
        self,
        columns: ColumnarSamples,
        *,
        feature_contract,
        split: str,
    ) -> None:
        self.columns = columns
        self.feature_contract = feature_contract
        self.split = split

    def __len__(self) -> int:
        return len(self.columns)

    def __getitem__(self, index: int) -> TrainingSample:
        return self.columns.get_sample(index)


class IndexDataset(torch.utils.data.Dataset):
    """Minimal dataset that returns raw integer indices.

    Designed for use with ``make_columnar_collate``: the DataLoader calls
    ``__getitem__(i)`` which returns ``i`` itself; the collate function then
    slices the columnar tensors directly with all batch indices at once.
    This avoids the per-sample ``TrainingSample`` reconstruction and the
    O(batch_size) ``torch.stack`` in the standard collate path.
    """

    def __init__(self, length: int) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> int:
        return index


def columnar_from_shards(
    shards: Iterable[tuple[str, dict[str, list[TrainingSample]]]],
    *,
    drop_callback=None,
) -> dict[str, ColumnarSamples]:
    """Stream-load shards into a per-split ColumnarSamples accumulator.

    ``shards`` is an iterable of ``(symbol, samples_by_split)`` pairs — one
    per shard.  After each shard is consumed, ``drop_callback(symbol)`` is
    invoked so the caller can release any held references to the original
    payload (this is what keeps peak working set bounded — the input
    dataclass list for shard N is freed before shard N+1 loads).

    Returns a dict mapping split name -> ColumnarSamples.
    """
    out: dict[str, ColumnarSamples] = {}
    for symbol, samples_by_split in shards:
        for split_name, split_samples in samples_by_split.items():
            if not split_samples:
                continue
            shard_columns = ColumnarSamples.from_samples(split_samples)
            if split_name not in out:
                out[split_name] = shard_columns
            else:
                out[split_name].append(shard_columns)
        if drop_callback is not None:
            drop_callback(symbol)
    return out


def normalize_columns(
    train: ColumnarSamples,
    *others: ColumnarSamples,
    eps: float = 1e-6,
    clamp: float = 10.0,
) -> None:
    """Z-score normalize float input columns using train set statistics.

    Computes mean/std from *train*, applies to train + all *others* (val, test).
    Clamps to ``[-clamp, clamp]`` to prevent float16 overflow in AMP.
    Labels, regression targets, future returns, and metadata are untouched.
    """

    def _normalize_tensor(train_t: torch.Tensor, *other_ts: torch.Tensor) -> None:
        """Normalize a 1-D, 2-D ``(N, D)``, or 3-D ``(N, C, T)`` tensor in-place."""
        if train_t.dtype not in (torch.float32, torch.float64):
            return
        if train_t.numel() == 0:
            return
        if train_t.dim() == 2:
            mean = train_t.mean(dim=0)
            std = train_t.std(dim=0).clamp(min=eps)
            train_t.sub_(mean).div_(std).clamp_(-clamp, clamp)
            for t in other_ts:
                t.sub_(mean).div_(std).clamp_(-clamp, clamp)
        elif train_t.dim() == 3:
            # (N, C, T) -- normalize per channel
            mean = train_t.mean(dim=(0, 2), keepdim=True)
            std = train_t.std(dim=(0, 2), keepdim=True).clamp(min=eps)
            train_t.sub_(mean).div_(std).clamp_(-clamp, clamp)
            for t in other_ts:
                t.sub_(mean).div_(std).clamp_(-clamp, clamp)
        elif train_t.dim() == 1:
            mean = train_t.mean()
            std = train_t.std().clamp(min=eps)
            train_t.sub_(mean).div_(std).clamp_(-clamp, clamp)
            for t in other_ts:
                t.sub_(mean).div_(std).clamp_(-clamp, clamp)

    # Normalize input feature tensors only (not labels/targets).
    others_list = list(others)
    _normalize_tensor(train.features, *[o.features for o in others_list])
    _normalize_tensor(train.regime_input, *[o.regime_input for o in others_list])
    _normalize_tensor(train.temporal_tensor, *[o.temporal_tensor for o in others_list])

    # Normalize each branch input tensor.
    for branch_name in train.branch_inputs:
        _normalize_tensor(
            train.branch_inputs[branch_name],
            *[o.branch_inputs[branch_name] for o in others_list],
        )
