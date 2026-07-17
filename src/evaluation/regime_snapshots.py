"""Regime snapshots — offline half (MASTER_PLAN §7.2.2).

Per-session K-means centroids over the regime feature family, plus
Mahalanobis matching for novelty scoring.  Deliberately NOT alpha-shape
IoU (do-NOT #6) and never a model-swap signal (do-NOT #9).

A "session" is one trading date; its feature vector is the cross-symbol
mean of ``compute_regime_features`` evaluated at session close (the same
regime family the model's regime branch consumes — trend strength,
volatility ratio, volume ratio, return alignment).  Regular-hours US
bars stay within one UTC calendar date, so sessions group by the bar
timestamp's UTC date.

First consumer: real regime labels for the §4.5.7 backward-transfer
slices (``slice_decision_log_by_regime``), replacing the chronological
cut.  Second consumer (novelty event tightening risk caps, shrink-only)
is the ONLINE half and is not wired here.

K-means is implemented locally (numpy, seeded kmeans++, restarts) — the
project deliberately has no sklearn dependency, and the clustering must
be deterministic for a given input + seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np
import torch

from src.core.logging import get_logger
from src.data.schemas import MarketBar, bars_to_tensor
from src.features.regime_features import compute_regime_features

SESSION_FEATURE_NAMES: tuple[str, ...] = (
    "regime_trend_strength",
    "regime_volatility_ratio",
    "regime_volume_ratio",
    "regime_return_alignment",
)

# A session needs at least this many 5-min bars to yield a meaningful
# end-of-session regime vector (a full regular session has ~78).
_MIN_SESSION_BARS = 12

# Novelty threshold: a session farther from every centroid than this
# percentile of the training distances is "novel".
_NOVELTY_PERCENTILE = 95.0

_STD_FLOOR = 1.0e-8


def build_session_features(
    bars_by_symbol: dict[str, list[MarketBar]],
    *,
    short_window: int,
    medium_window: int,
    long_window: int,
    epsilon: float,
) -> tuple[list[date], np.ndarray]:
    """(sorted session dates, [n_sessions, 4] raw feature matrix).

    Per symbol and date, ``compute_regime_features`` runs on that
    session's bars (windows clip to short sessions, matching the scalar
    path's semantics); the session vector is the cross-symbol mean.
    Sessions with fewer than ``_MIN_SESSION_BARS`` bars for every symbol
    are dropped.
    """
    per_date: dict[date, list[np.ndarray]] = {}
    for bars in bars_by_symbol.values():
        by_day: dict[date, list[MarketBar]] = {}
        for bar in bars:
            by_day.setdefault(bar.timestamp.date(), []).append(bar)
        for day, day_bars in by_day.items():
            if len(day_bars) < _MIN_SESSION_BARS:
                continue
            tensor = bars_to_tensor(sorted(day_bars, key=lambda b: b.timestamp))
            features = compute_regime_features(
                tensor,
                short_window=short_window,
                medium_window=medium_window,
                long_window=long_window,
                epsilon=epsilon,
            )
            vector = np.array(
                [float(features[name].item()) for name in SESSION_FEATURE_NAMES],
                dtype=np.float64,
            )
            if not np.isfinite(vector).all():
                continue
            per_date.setdefault(day, []).append(vector)
    dates = sorted(per_date)
    if not dates:
        return [], np.zeros((0, len(SESSION_FEATURE_NAMES)), dtype=np.float64)
    matrix = np.stack([np.mean(per_date[day], axis=0) for day in dates])
    return dates, matrix


def _kmeans_fit(
    features: np.ndarray, k: int, *, seed: int, n_init: int = 10, max_iter: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded kmeans++ with restarts; returns (centroids [k, d], labels)."""
    rng = np.random.default_rng(seed)
    best_inertia, best = np.inf, None
    n = features.shape[0]
    for _ in range(n_init):
        # kmeans++ seeding
        centroids = [features[rng.integers(n)]]
        for _ in range(1, k):
            dist_sq = np.min(
                ((features[:, None, :] - np.array(centroids)[None, :, :]) ** 2).sum(-1),
                axis=1,
            )
            total = dist_sq.sum()
            if total <= 0:
                centroids.append(features[rng.integers(n)])
                continue
            centroids.append(features[rng.choice(n, p=dist_sq / total)])
        centers = np.array(centroids)
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(max_iter):
            dists = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            new_labels = dists.argmin(axis=1)
            if (new_labels == labels).all() and _ > 0:
                break
            labels = new_labels
            for cluster in range(k):
                members = features[labels == cluster]
                if len(members):
                    centers[cluster] = members.mean(axis=0)
        inertia = float(
            ((features - centers[labels]) ** 2).sum()
        )
        if inertia < best_inertia:
            best_inertia, best = inertia, (centers.copy(), labels.copy())
    assert best is not None
    return best


@dataclass
class ScoredSession:
    cluster: int
    distance: float
    novel: bool


@dataclass
class RegimeSnapshot:
    """Fitted per-session regime clustering + novelty scorer.

    All geometry lives in z-space (per-dimension standardization over the
    training sessions); ``score`` applies the stored transform.  Distance
    is Mahalanobis with the assigned cluster's diagonal covariance
    (floored), matching the §7.2.2 "MMD/Mahalanobis, not alpha-shape"
    decision.
    """

    feature_names: tuple[str, ...]
    k: int
    mean: np.ndarray
    std: np.ndarray
    centroids: np.ndarray  # [k, d], z-space
    cluster_std: np.ndarray  # [k, d], z-space diagonal std per cluster (floored)
    labels_by_date: dict[str, int] = field(default_factory=dict)  # iso date -> cluster
    novelty_threshold: float = 0.0  # p95 of training distances

    def _z(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.mean) / self.std

    def _distance(self, z: np.ndarray, cluster: int) -> float:
        delta = (z - self.centroids[cluster]) / self.cluster_std[cluster]
        return float(np.sqrt((delta**2).sum()))

    def score(self, raw_features: np.ndarray) -> ScoredSession:
        z = self._z(np.asarray(raw_features, dtype=np.float64))
        distances = [self._distance(z, cluster) for cluster in range(self.k)]
        cluster = int(np.argmin(distances))
        distance = distances[cluster]
        return ScoredSession(
            cluster=cluster, distance=distance, novel=distance > self.novelty_threshold
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "k": self.k,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "centroids": self.centroids.tolist(),
            "cluster_std": self.cluster_std.tolist(),
            "labels_by_date": dict(self.labels_by_date),
            "novelty_threshold": self.novelty_threshold,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RegimeSnapshot":
        return cls(
            feature_names=tuple(payload["feature_names"]),
            k=int(payload["k"]),
            mean=np.array(payload["mean"], dtype=np.float64),
            std=np.array(payload["std"], dtype=np.float64),
            centroids=np.array(payload["centroids"], dtype=np.float64),
            cluster_std=np.array(payload["cluster_std"], dtype=np.float64),
            labels_by_date=dict(payload["labels_by_date"]),
            novelty_threshold=float(payload["novelty_threshold"]),
        )


def build_regime_snapshot(
    bars_by_symbol: dict[str, list[MarketBar]],
    *,
    k: int,
    seed: int,
    short_window: int,
    medium_window: int,
    long_window: int,
    epsilon: float,
) -> RegimeSnapshot | None:
    """Fit a snapshot over every session in the given bars.

    Returns None when there are fewer sessions than clusters (the caller
    decides the fallback — the slice gate falls back to chronological).
    Cluster ids are relabeled by first chronological occurrence so ids
    are stable and readable for a given input.
    """
    logger = get_logger("regime_snapshots")
    dates, raw = build_session_features(
        bars_by_symbol,
        short_window=short_window,
        medium_window=medium_window,
        long_window=long_window,
        epsilon=epsilon,
    )
    if len(dates) < k:
        logger.warning(
            "regime_snapshot_too_few_sessions", sessions=len(dates), k=k
        )
        return None
    mean = raw.mean(axis=0)
    std = np.maximum(raw.std(axis=0), _STD_FLOOR)
    z = (raw - mean) / std
    centroids, labels = _kmeans_fit(z, k, seed=seed)

    # Relabel clusters by first chronological occurrence (stable ids).
    remap: dict[int, int] = {}
    for label in labels:
        if int(label) not in remap:
            remap[int(label)] = len(remap)
    # Clusters that never won a session (possible on degenerate data)
    # keep their geometry at the end of the id range.
    for cluster in range(k):
        if cluster not in remap:
            remap[cluster] = len(remap)
    labels = np.array([remap[int(label)] for label in labels])
    ordered = np.argsort([remap[cluster] for cluster in range(k)])
    centroids = centroids[ordered]

    cluster_std = np.ones_like(centroids)
    for cluster in range(k):
        members = z[labels == cluster]
        if len(members) >= 2:
            cluster_std[cluster] = np.maximum(members.std(axis=0), _STD_FLOOR)

    snapshot = RegimeSnapshot(
        feature_names=SESSION_FEATURE_NAMES,
        k=k,
        mean=mean,
        std=std,
        centroids=centroids,
        cluster_std=cluster_std,
        labels_by_date={day.isoformat(): int(label) for day, label in zip(dates, labels)},
    )
    train_distances = np.array(
        [snapshot._distance(z[i], int(labels[i])) for i in range(len(dates))]
    )
    snapshot.novelty_threshold = float(np.percentile(train_distances, _NOVELTY_PERCENTILE))
    logger.info(
        "regime_snapshot_built",
        sessions=len(dates),
        k=k,
        cluster_sizes={int(c): int((labels == c).sum()) for c in range(k)},
        novelty_threshold=round(snapshot.novelty_threshold, 4),
    )
    return snapshot


def slice_decision_log_by_regime(
    decision_log: Sequence[dict[str, object]],
    labels_by_date: dict[str, int],
) -> tuple[list[tuple[str, list[dict[str, object]]]], int]:
    """Bucket decisions by their session's regime cluster.

    Returns (slices, dropped): slices are ``("regime<id>", records)`` in
    cluster-id order; decisions on unlabeled dates are dropped and
    counted.  Two decision logs over the same window share the SAME
    labels_by_date, so bucketing is identical — the pairing the
    backward-transfer gate needs.
    """
    buckets: dict[int, list[dict[str, object]]] = {}
    dropped = 0
    for record in decision_log:
        label = labels_by_date.get(record["timestamp"].date().isoformat())
        if label is None:
            dropped += 1
            continue
        buckets.setdefault(int(label), []).append(record)
    slices = [
        (f"regime{cluster}", buckets[cluster]) for cluster in sorted(buckets)
    ]
    return slices, dropped
