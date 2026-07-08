from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import torch
from torch.utils.data import WeightedRandomSampler


@dataclass(frozen=True)
class RegressionSummary:
    mean: float
    std: float
    min: float
    max: float


@dataclass
class TargetDiagnostics:
    split: str
    total_samples: int
    class_counts: dict[str, int]
    class_ratios: dict[str, float]
    flat_rate: float
    symbol_counts: dict[str, int]
    flat_rate_by_symbol: dict[str, float]
    regression_summaries: dict[str, RegressionSummary] = field(default_factory=dict)
    primary_return_quantiles: dict[str, float] = field(default_factory=dict)
    near_flat_count: int = 0
    near_flat_rate: float = 0.0
    near_boundary_count: int = 0
    near_boundary_rate: float = 0.0
    # State + future-path diagnostics
    path_label_counts: dict[str, int] = field(default_factory=dict)
    path_label_ratios: dict[str, float] = field(default_factory=dict)
    per_horizon_return_quantiles: dict[str, dict[str, float]] = field(default_factory=dict)
    risk_path_summaries: dict[str, RegressionSummary] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


_DIRECTION_NAMES = ("short", "flat", "long")  # index = label int (0/1/2)


def _compute_target_diagnostics_columnar(
    columns,
    *,
    split: str,
    minority_class_floor_warning: float,
    label_epsilon: float,
    inspect_target_boundaries: bool,
    near_flat_multiplier: float = 1.25,
    epsilon_reduction_factor: float = 0.5,
    flat_dominance_threshold: float = 0.80,
) -> TargetDiagnostics:
    """Vectorized diagnostics computed straight from a ColumnarSamples store.

    All per-sample work runs as bulk torch ops on the underlying column
    tensors.  For ~3 M samples this completes in ~1-2 seconds versus the
    ~30-60 seconds the lazy-iterator path would take, AND uses zero
    transient ``TrainingSample`` allocations.
    """
    n = len(columns)
    label_tensor = columns.label
    # Class counter via bincount on the label tensor.
    bins = torch.bincount(label_tensor.to(torch.long), minlength=3).tolist()
    class_counter: Counter[str] = Counter()
    for cls_idx, count in enumerate(bins):
        if count > 0:
            class_counter[_DIRECTION_NAMES[cls_idx]] = int(count)

    # Symbol counts + per-symbol flat counts via Python iteration over the
    # parallel symbols list.  We can't easily tensorize string keys, but
    # the bookkeeping is O(N) C-level work because the comparison against
    # ``flat_idx`` happens once per row in a list comprehension.
    symbol_counter: Counter[str] = Counter(columns.symbols)
    flat_idx = 1  # _DIRECTION_NAMES.index("flat")
    flat_mask_list = (label_tensor == flat_idx).tolist()
    flat_counter_by_symbol: Counter[str] = Counter()
    for sym, is_flat in zip(columns.symbols, flat_mask_list):
        if is_flat:
            flat_counter_by_symbol[sym] += 1

    # Path-label counter from the parallel ``path_names`` list.
    path_label_counter: Counter[str] = Counter(
        name for name in columns.path_names if name
    )

    # Future returns: each horizon column is already a (N,) tensor.
    primary_returns: list[float] = []
    per_horizon_returns: dict[int, list[float]] = {}
    if columns.future_returns:
        primary_horizon = max(columns.future_returns.keys())
        primary_returns = columns.future_returns[primary_horizon].tolist()
        for horizon, col in columns.future_returns.items():
            per_horizon_returns[int(horizon)] = col.tolist()

    # Regression targets: vectorized summary stats per target.
    regression_summaries: dict[str, RegressionSummary] = {}
    risk_summaries: dict[str, RegressionSummary] = {}
    for name, col in columns.regression_targets.items():
        if col.numel() == 0:
            continue
        summary = RegressionSummary(
            mean=float(col.mean().item()),
            std=float(col.std(unbiased=False).item()) if col.numel() > 1 else 0.0,
            min=float(col.min().item()),
            max=float(col.max().item()),
        )
        regression_summaries[name] = summary
        if name in ("mfe", "mae", "volatility", "path_drawdown"):
            risk_summaries[name] = summary

    flat_rate = class_counter["flat"] / n if n else 0.0
    class_ratios = {k: v / n for k, v in class_counter.items()} if n else {}
    path_label_ratios = {k: v / n for k, v in path_label_counter.items()} if n else {}
    flat_rate_by_symbol = {
        sym: flat_counter_by_symbol[sym] / count
        for sym, count in symbol_counter.items()
        if count > 0
    }
    primary_return_quantiles = _summarize_quantiles(primary_returns)
    per_horizon_return_quantiles = {
        f"h{horizon}": _summarize_quantiles(returns)
        for horizon, returns in sorted(per_horizon_returns.items())
    }

    near_flat_count = 0
    near_boundary_count = 0
    if inspect_target_boundaries and primary_returns and label_epsilon > 0.0:
        epsilon = abs(label_epsilon)
        primary_tensor = torch.tensor(primary_returns, dtype=torch.float32)
        abs_returns = primary_tensor.abs()
        near_flat_count = int((abs_returns <= epsilon * near_flat_multiplier).sum().item())
        near_boundary_count = int(
            ((abs_returns - epsilon).abs() <= max(epsilon * epsilon_reduction_factor, 1.0e-6)).sum().item()
        )

    warnings: list[str] = []
    for class_name, count in class_counter.items():
        share = count / n if n else 0.0
        if share < minority_class_floor_warning:
            warnings.append(f"{split}:{class_name}_share_low={share:.4f}")
    if flat_rate > flat_dominance_threshold:
        warnings.append(f"{split}:flat_dominance={flat_rate:.4f}")
    if inspect_target_boundaries and n > 0 and near_boundary_count / n > 0.25:
        warnings.append(f"{split}:boundary_heavy={near_boundary_count / n:.4f}")

    return TargetDiagnostics(
        split=split,
        total_samples=n,
        class_counts=dict(class_counter),
        class_ratios=class_ratios,
        flat_rate=flat_rate,
        symbol_counts=dict(symbol_counter),
        flat_rate_by_symbol=flat_rate_by_symbol,
        regression_summaries=regression_summaries,
        primary_return_quantiles=primary_return_quantiles,
        near_flat_count=near_flat_count,
        near_flat_rate=near_flat_count / n if n else 0.0,
        near_boundary_count=near_boundary_count,
        near_boundary_rate=near_boundary_count / n if n else 0.0,
        path_label_counts=dict(path_label_counter),
        path_label_ratios=path_label_ratios,
        per_horizon_return_quantiles=per_horizon_return_quantiles,
        risk_path_summaries=risk_summaries,
        warnings=warnings,
    )


def compute_target_diagnostics(
    samples,
    *,
    split: str,
    minority_class_floor_warning: float,
    label_epsilon: float = 0.0,
    inspect_target_boundaries: bool = True,
    near_flat_multiplier: float = 1.25,
    epsilon_reduction_factor: float = 0.5,
    flat_dominance_threshold: float = 0.80,
) -> TargetDiagnostics:
    # Fast columnar path: detect ColumnarSamples and dispatch to the
    # vectorized helper.  Avoids ~30-60 s of lazy-iterator wrapping for
    # multi-million sample splits.  Lazy import to avoid an import cycle
    # with ``data_module``.
    try:
        from src.training.sample_columns import ColumnarSamples
    except ImportError:
        ColumnarSamples = None  # type: ignore[assignment]
    if ColumnarSamples is not None and isinstance(samples, ColumnarSamples):
        return _compute_target_diagnostics_columnar(
            samples,
            split=split,
            minority_class_floor_warning=minority_class_floor_warning,
            label_epsilon=label_epsilon,
            inspect_target_boundaries=inspect_target_boundaries,
            near_flat_multiplier=near_flat_multiplier,
            epsilon_reduction_factor=epsilon_reduction_factor,
            flat_dominance_threshold=flat_dominance_threshold,
        )
    class_counter: Counter[str] = Counter()
    path_label_counter: Counter[str] = Counter()
    symbol_counter: Counter[str] = Counter()
    flat_counter_by_symbol: Counter[str] = Counter()
    regression_values: dict[str, list[float]] = defaultdict(list)
    primary_returns: list[float] = []
    per_horizon_returns: dict[int, list[float]] = defaultdict(list)
    risk_values: dict[str, list[float]] = defaultdict(list)
    total_samples = len(samples)
    for sample in samples:
        direction_name = str(getattr(sample, "direction_name"))
        symbol = str(getattr(sample, "symbol"))
        path_name = str(getattr(sample, "path_name", ""))
        class_counter[direction_name] += 1
        symbol_counter[symbol] += 1
        if path_name:
            path_label_counter[path_name] += 1
        if direction_name == "flat":
            flat_counter_by_symbol[symbol] += 1
        future_returns = dict(getattr(sample, "future_returns", {}))
        if future_returns:
            primary_returns.append(float(future_returns[max(future_returns)]))
            for horizon, ret in future_returns.items():
                per_horizon_returns[int(horizon)].append(float(ret))
        reg_targets = dict(getattr(sample, "regression_targets", {}))
        for target_name, value in reg_targets.items():
            regression_values[target_name].append(float(value))
        # Collect risk/path stats from regression targets or direct attributes
        for risk_name in ("mfe", "mae", "volatility", "path_drawdown"):
            if risk_name in reg_targets:
                risk_values[risk_name].append(float(reg_targets[risk_name]))
    flat_rate = class_counter["flat"] / total_samples if total_samples else 0.0
    class_ratios = {
        class_name: count / total_samples
        for class_name, count in class_counter.items()
    }
    path_label_ratios = {
        name: count / total_samples
        for name, count in path_label_counter.items()
    }
    flat_rate_by_symbol = {
        symbol: flat_counter_by_symbol[symbol] / count
        for symbol, count in symbol_counter.items()
        if count > 0
    }
    primary_return_quantiles = _summarize_quantiles(primary_returns)
    per_horizon_return_quantiles = {
        f"h{horizon}": _summarize_quantiles(returns)
        for horizon, returns in sorted(per_horizon_returns.items())
    }
    risk_path_summaries = {
        name: _summarize(values)
        for name, values in risk_values.items()
    }
    near_flat_count = 0
    near_boundary_count = 0
    if inspect_target_boundaries and primary_returns and label_epsilon > 0.0:
        epsilon = abs(label_epsilon)
        near_flat_count = sum(1 for value in primary_returns if abs(value) <= epsilon * near_flat_multiplier)
        near_boundary_count = sum(
            1
            for value in primary_returns
            if abs(abs(value) - epsilon) <= max(epsilon * epsilon_reduction_factor, 1.0e-6)
        )
    warnings: list[str] = []
    for class_name, count in class_counter.items():
        share = count / total_samples if total_samples else 0.0
        if share < minority_class_floor_warning:
            warnings.append(f"{split}:{class_name}_share_low={share:.4f}")
    if flat_rate > flat_dominance_threshold:
        warnings.append(f"{split}:flat_dominance={flat_rate:.4f}")
    if inspect_target_boundaries and total_samples > 0 and near_boundary_count / total_samples > 0.25:
        warnings.append(f"{split}:boundary_heavy={near_boundary_count / total_samples:.4f}")
    return TargetDiagnostics(
        split=split,
        total_samples=total_samples,
        class_counts=dict(class_counter),
        class_ratios=class_ratios,
        flat_rate=flat_rate,
        symbol_counts=dict(symbol_counter),
        flat_rate_by_symbol=flat_rate_by_symbol,
        regression_summaries={
            name: _summarize(values)
            for name, values in regression_values.items()
        },
        primary_return_quantiles=primary_return_quantiles,
        near_flat_count=near_flat_count,
        near_flat_rate=near_flat_count / total_samples if total_samples else 0.0,
        near_boundary_count=near_boundary_count,
        near_boundary_rate=near_boundary_count / total_samples if total_samples else 0.0,
        path_label_counts=dict(path_label_counter),
        path_label_ratios=path_label_ratios,
        per_horizon_return_quantiles=per_horizon_return_quantiles,
        risk_path_summaries=risk_path_summaries,
        warnings=warnings,
    )


def _label_tensor_from_samples(samples) -> torch.Tensor | None:
    """Fast path: pull the integer label tensor directly out of a
    ``ColumnarSamples`` instance instead of iterating 3 M lazy dataclasses
    just to call ``.item()`` on each scalar tensor.

    Returns ``None`` when the input is not a ColumnarSamples (so the caller
    can fall back to the legacy per-sample iteration path).  We import the
    type lazily to avoid an import cycle with ``data_module``.
    """
    try:
        from src.training.sample_columns import ColumnarSamples
    except ImportError:  # pragma: no cover - defensive
        return None
    if isinstance(samples, ColumnarSamples):
        return samples.label
    return None


def build_class_weights(
    samples,
    *,
    mode: str,
    cap: float | None = None,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    label_tensor = _label_tensor_from_samples(samples)
    if label_tensor is not None:
        # Columnar fast path: bincount on the integer label tensor.  3 classes
        # → constant work; for 2 M samples this is ~1 ms vs ~30-60 s through
        # the lazy-iterator path that wraps each sample in a TrainingSample
        # and calls .item() on the scalar label tensor.
        if label_tensor.numel() == 0:
            return None
        bins = torch.bincount(label_tensor, minlength=3).tolist()
        counts: Counter[int] = Counter()
        for cls, c in enumerate(bins):
            if c > 0:
                counts[cls] = int(c)
    else:
        if not samples:
            return None
        counts = Counter(int(getattr(sample, "label").item()) for sample in samples)
    if not counts:
        return None
    weights = _compute_class_weights(counts, mode=mode)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if cap is not None:
        weight_tensor = torch.clamp(weight_tensor, max=cap)
    return weight_tensor / weight_tensor.mean()


def build_weighted_sampler(
    samples,
    *,
    mode: str = "inverse_frequency",
) -> WeightedRandomSampler | None:
    label_tensor = _label_tensor_from_samples(samples)
    if label_tensor is not None:
        if label_tensor.numel() == 0:
            return None
        bins = torch.bincount(label_tensor, minlength=3).tolist()
        counts: Counter[int] = Counter()
        for cls, c in enumerate(bins):
            if c > 0:
                counts[cls] = int(c)
        class_weights = _compute_class_weights(counts, mode=mode)
        # Vectorized per-sample weight lookup: build a (3,) weight table
        # and gather by label.  This is O(N) bytes of work in C instead of
        # an N-step Python loop calling .item() per sample.
        weight_table = torch.tensor(class_weights, dtype=torch.double)
        sample_weights_tensor = weight_table[label_tensor.to(torch.long)]
        return WeightedRandomSampler(
            weights=sample_weights_tensor.tolist(),
            num_samples=int(label_tensor.numel()),
            replacement=True,
        )
    if not samples:
        return None
    counts = Counter(int(getattr(sample, "label").item()) for sample in samples)
    class_weights = _compute_class_weights(counts, mode=mode)
    sample_weights = [
        class_weights[int(getattr(sample, "label").item())]
        for sample in samples
    ]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def _summarize(values: list[float]) -> RegressionSummary:
    tensor = torch.tensor(values, dtype=torch.float32)
    return RegressionSummary(
        mean=float(tensor.mean().item()),
        std=float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
        min=float(tensor.min().item()),
        max=float(tensor.max().item()),
    )


def _summarize_quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    tensor = torch.tensor(values, dtype=torch.float32)
    quantiles = torch.quantile(tensor, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float32))
    return {
        "p10": float(quantiles[0].item()),
        "p25": float(quantiles[1].item()),
        "p50": float(quantiles[2].item()),
        "p75": float(quantiles[3].item()),
        "p90": float(quantiles[4].item()),
    }


def _compute_class_weights(counts: Counter[int], *, mode: str) -> list[float]:
    total = sum(counts.values())
    weights: list[float] = []
    for label in range(3):
        count = counts.get(label, 1)
        if mode == "balanced":
            weights.append(total / max(count, 1))
        elif mode == "inverse_frequency":
            weights.append(1.0 / max(count, 1))
        elif mode == "effective_number":
            beta = (total - 1.0) / max(total, 1.0)
            effective_num = (1.0 - (beta ** count)) / max(1.0 - beta, 1.0e-6)
            weights.append(1.0 / max(effective_num, 1.0e-6))
        else:
            raise ValueError(f"Unsupported class weight mode: {mode}")
    return weights
