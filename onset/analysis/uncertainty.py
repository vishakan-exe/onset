"""Calibration, conformal prediction and bootstrap confidence intervals.

Everything here is label-agnostic and dataset-agnostic: it is used on the phantom instrument-calibration runs
(``scripts/calibrate_engine.py``) and, unchanged, on the held-out folds of real datasets by the learning harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Equal-width reliability bins for binary probabilities ``p`` and outcomes ``y`` (0/1)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= edges[i + 1]))
        if m.any():
            out.append(dict(lo=float(edges[i]), hi=float(edges[i + 1]), mean_p=float(p[m].mean()),
                            frac=float(y[m].mean()), n=int(m.sum())))
    return out


def ece_binary(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = reliability_curve(p, y, n_bins)
    n = sum(b["n"] for b in bins)
    return float(sum(b["n"] * abs(b["mean_p"] - b["frac"]) for b in bins) / max(n, 1))


def ece_multiclass(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Top-label expected calibration error."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    return ece_binary(conf, (pred == y).astype(float), n_bins)


def brier_binary(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss_binary(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(logits: np.ndarray, y: np.ndarray, grid: Optional[np.ndarray] = None) -> float:
    """Temperature T > 0 minimising the NLL of softmax(logits / T) on held-out data (1-D grid + refinement)."""
    grid = np.exp(np.linspace(np.log(0.2), np.log(8.0), 60)) if grid is None else grid

    def nll(T):
        z = logits / T
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean()
    vals = [nll(T) for T in grid]
    i = int(np.argmin(vals))
    lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    fine = np.linspace(lo, hi, 25)
    return float(fine[int(np.argmin([nll(T) for T in fine]))])


def fit_scalar_temper(evaluate: Callable[[float], float], grid: Sequence[float]) -> tuple[float, list]:
    """Generic 1-D calibration: ``evaluate(lam)`` returns a loss (lower is better); returns (best, curve)."""
    curve = [(float(g), float(evaluate(g))) for g in grid]
    return min(curve, key=lambda t: t[1])[0], curve


@dataclass
class LACConformal:
    """Split-conformal prediction sets with the LAC (least-ambiguous) score  s = 1 - p_true.

    Marginal coverage >= 1 - alpha holds if calibration and test items are exchangeable.  In micro-expression
    work exchangeability must be assessed at the *subject* level: calibrate on held-out subjects only."""
    alpha: float = 0.1
    qhat: float = 1.0
    n_cal: int = 0

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "LACConformal":
        s = 1.0 - probs[np.arange(len(y)), y]
        n = len(s)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        self.qhat = float(np.sort(s)[min(k, n) - 1]) if n else 1.0
        self.n_cal = n
        return self

    def predict_set(self, p: np.ndarray, labels: Sequence[str]) -> list[str]:
        keep = [labels[i] for i in range(len(labels)) if p[i] >= 1.0 - self.qhat]
        return keep or [labels[int(np.argmax(p))]]

    def coverage(self, probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        hit, size = 0, 0
        for p, t in zip(probs, y):
            keep = np.flatnonzero(p >= 1.0 - self.qhat)
            if len(keep) == 0:
                keep = np.array([int(np.argmax(p))])
            hit += int(t in keep)
            size += len(keep)
        return hit / len(y), size / len(y)


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, stat: Callable[[np.ndarray], float] = np.mean,
                      n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Subject-level (cluster) bootstrap: resample *subjects*, not clips, so repeated clips of one person do not
    manufacture precision.  Returns (estimate, lo, hi)."""
    rng = np.random.default_rng(seed)
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    ids = np.unique(clusters)
    groups = [values[clusters == c] for c in ids]
    est = float(stat(values))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        boots[b] = stat(np.concatenate([groups[i] for i in pick]))
    return est, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(p), float(max(0.0, c - h)), float(min(1.0, c + h))


def permutation_pvalue(a: np.ndarray, b: np.ndarray, stat: Callable = lambda x, y: np.mean(x) - np.mean(y),
                       n_perm: int = 5000, seed: int = 0) -> float:
    """Two-sided permutation test for a difference in ``stat`` between two groups (exchangeable under H0)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = stat(a, b)
    pool = np.concatenate([a, b])
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        cnt += abs(stat(pool[: len(a)], pool[len(a):])) >= abs(obs) - 1e-12
    return float((cnt + 1) / (n_perm + 1))
