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


def compute_target_diagnostics(
    samples: list[object],
    *,
    split: str,
    minority_class_floor_warning: float,
    label_epsilon: float = 0.0,
    inspect_target_boundaries: bool = True,
) -> TargetDiagnostics:
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
        near_flat_count = sum(1 for value in primary_returns if abs(value) <= epsilon * 1.25)
        near_boundary_count = sum(
            1
            for value in primary_returns
            if abs(abs(value) - epsilon) <= max(epsilon * 0.5, 1.0e-6)
        )
    warnings: list[str] = []
    for class_name, count in class_counter.items():
        share = count / total_samples if total_samples else 0.0
        if share < minority_class_floor_warning:
            warnings.append(f"{split}:{class_name}_share_low={share:.4f}")
    if flat_rate > 0.80:
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


def build_class_weights(
    samples: list[object],
    *,
    mode: str,
    cap: float | None = None,
) -> torch.Tensor | None:
    if mode == "none" or not samples:
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
    samples: list[object],
    *,
    mode: str = "inverse_frequency",
) -> WeightedRandomSampler | None:
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
