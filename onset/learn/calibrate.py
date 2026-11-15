"""Calibration and prediction sets for the learned models, fitted on held-out *subjects* only.

A model's raw probabilities are rarely honest.  Temperature scaling (one number) fixes the commonest failure, over-
confidence; split-conformal prediction sets then give a set of labels that contains the true one with a stated
probability (``1 - alpha``) *if* calibration and test subjects are exchangeable.  Both are fitted on predictions the
model made for subjects it was not trained on (inner subject-disjoint folds, or the validation split that each ensemble member holds out of
subjects) - never on the outer test subjects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..analysis import uncertainty as UNC


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, float) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


@dataclass
class Calibrator:
    temperature: float = 1.0
    qhat: float = 1.0
    alpha: float = 0.1
    n_cal: int = 0
    fitted: bool = False

    def fit(self, logits: np.ndarray, y: np.ndarray, alpha: float = 0.1, min_items: int = 20) -> "Calibrator":
        self.alpha = alpha
        self.n_cal = int(len(y))
        if len(y) < min_items:                      # too little held-out data to calibrate anything: leave identity + empty sets
            return self
        self.temperature = UNC.fit_temperature(np.asarray(logits, float), np.asarray(y, int))
        probs = softmax(logits, self.temperature)
        self.qhat = UNC.LACConformal(alpha=alpha).fit(probs, np.asarray(y, int)).qhat
        self.fitted = True
        return self

    def probs(self, logits: np.ndarray) -> np.ndarray:
        return softmax(logits, self.temperature)

    def sets(self, probs: np.ndarray) -> list[list[int]]:
        """Prediction sets as lists of class indices; the whole label set when the calibrator was not fitted (an honest 'don't know')."""
        out = []
        for p in probs:
            if not self.fitted:
                out.append(list(range(len(p))))
                continue
            keep = np.flatnonzero(p >= 1.0 - self.qhat)
            out.append(keep.tolist() if len(keep) else [int(np.argmax(p))])
        return out


def set_stats(sets: list, y: np.ndarray, n_classes: int) -> dict:
    y = np.asarray(y, int)
    cover = np.array([int(t in s) for s, t in zip(sets, y)], float)
    size = np.array([len(s) for s in sets], float)
    singleton = size == 1
    acc_single = float(np.mean([s[0] == t for s, t, one in zip(sets, y, singleton) if one])) if singleton.any() else None
    return dict(coverage=float(cover.mean()) if len(y) else None, mean_size=float(size.mean()) if len(y) else None,
                singleton_rate=float(singleton.mean()) if len(y) else None, accuracy_when_singleton=acc_single,
                full_set_rate=float((size == n_classes).mean()) if len(y) else None)


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Top-label reliability: expected calibration error and the binned curve."""
    probs, y = np.asarray(probs), np.asarray(y, int)
    conf, pred = probs.max(1), probs.argmax(1)
    hit = (pred == y).astype(float)
    return dict(ece=UNC.ece_binary(conf, hit, n_bins), bins=UNC.reliability_curve(conf, hit, n_bins),
                nll=float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1.0)))), brier=float(np.mean(((probs - np.eye(probs.shape[1])[y]) ** 2).sum(1))))
