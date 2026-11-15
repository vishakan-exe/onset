"""Metrics for imbalanced, subject-grouped micro-expression evaluation.

Headline numbers follow the MEGC convention: **UF1** (unweighted F1: per-class F1 from true/false positives and
negatives pooled over all folds, then averaged over classes) and **UAR** (unweighted average recall).  Accuracy is
reported too, but on data where one class holds half the clips it mostly measures the majority class.

Intervals are *subject-level* bootstraps: the unit that is resampled is the person, because clips of one person are
not independent and resampling clips would manufacture precision.  Model comparisons use paired subject-level
permutation tests (swap the two models' predictions for a random half of the subjects).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def confusion(y: np.ndarray, p: np.ndarray, n_classes: int) -> np.ndarray:
    m = np.zeros((n_classes, n_classes), int)
    np.add.at(m, (np.asarray(y, int), np.asarray(p, int)), 1)
    return m


def per_class(y: np.ndarray, p: np.ndarray, n_classes: int) -> dict:
    cm = confusion(y, p, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros(n_classes), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(n_classes), where=(tp + fn) > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(n_classes), where=(2 * tp + fp + fn) > 0)
    return dict(precision=prec, recall=rec, f1=f1, support=cm.sum(1), predicted=cm.sum(0), confusion=cm)


def uf1(y, p, n_classes: int) -> float:
    pc = per_class(y, p, n_classes)
    present = (pc["support"] + pc["predicted"]) > 0
    return float(pc["f1"][present].mean()) if present.any() else float("nan")


def uar(y, p, n_classes: int) -> float:
    pc = per_class(y, p, n_classes)
    present = pc["support"] > 0
    return float(pc["recall"][present].mean()) if present.any() else float("nan")


def accuracy(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float((y == p).mean()) if len(y) else float("nan")


def chance_uf1_uniform(y, n_classes: int) -> float:
    """Expected UF1 of a guesser that picks every class with equal probability, given the class counts of ``y``.
    On imbalanced data this is not 1/C (rare classes get a low precision), so it is the right floor for 'beats chance'."""
    n = np.bincount(np.asarray(y, int), minlength=n_classes).astype(float)
    tot = n.sum()
    if tot == 0:
        return float("nan")
    tp = n / n_classes
    fp = (tot - n) / n_classes
    fn = n - tp
    present = n > 0
    f1 = 2 * tp[present] / (2 * tp[present] + fp[present] + fn[present])
    return float(f1.mean())


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone step-down), returned in the input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def summary(y, p, n_classes: int, classes: Optional[Sequence[str]] = None) -> dict:
    pc = per_class(y, p, n_classes)
    out = dict(n=int(len(y)), uf1=uf1(y, p, n_classes), uar=uar(y, p, n_classes), accuracy=accuracy(y, p),
               chance_uf1=chance_uf1_uniform(y, n_classes), chance_uar=1.0 / n_classes,
               majority_accuracy=float(np.bincount(np.asarray(y, int), minlength=n_classes).max() / max(len(y), 1)))
    names = list(classes) if classes is not None else [str(i) for i in range(n_classes)]
    out["per_class"] = [dict(label=names[i], support=int(pc["support"][i]), predicted=int(pc["predicted"][i]),
                             precision=float(pc["precision"][i]), recall=float(pc["recall"][i]), f1=float(pc["f1"][i])) for i in range(n_classes)]
    out["confusion"] = pc["confusion"].tolist()
    return out


def subject_bootstrap(fn: Callable[[np.ndarray, np.ndarray], float], y, p, groups, n_boot: int = 2000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float, float]:
    """``fn(y, p)`` on the pooled predictions with a percentile interval from resampling *subjects* with replacement."""
    y, p, g = np.asarray(y), np.asarray(p), np.asarray(groups)
    subs = sorted(set(g.tolist()))
    idx = {s: np.flatnonzero(g == s) for s in subs}
    rng = np.random.default_rng(seed)
    est = float(fn(y, p))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(subs), len(subs))
        ii = np.concatenate([idx[subs[i]] for i in pick])
        vals[b] = fn(y[ii], p[ii])
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return est, float("nan"), float("nan")
    return est, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_permutation(fn: Callable[[np.ndarray, np.ndarray], float], y, pa, pb, groups, n_perm: int = 5000, seed: int = 0) -> dict:
    """Two-sided test of ``fn(y, pa) - fn(y, pb)`` against zero.  Under the null the two models are exchangeable, so for each
    subject the two prediction vectors are swapped with probability one half."""
    y, pa, pb, g = (np.asarray(v) for v in (y, pa, pb, groups))
    subs = sorted(set(g.tolist()))
    masks = [g == s for s in subs]
    obs = float(fn(y, pa) - fn(y, pb))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        flip = rng.random(len(subs)) < 0.5
        a, b = pa.copy(), pb.copy()
        for f, m in zip(flip, masks):
            if f:
                a[m], b[m] = pb[m], pa[m]
        hits += abs(fn(y, a) - fn(y, b)) >= abs(obs) - 1e-12
    return dict(difference=obs, p=float((hits + 1) / (n_perm + 1)), n_subjects=len(subs))


def per_subject_scores(fn: Callable[[np.ndarray, np.ndarray], float], y, p, groups) -> dict:
    g = np.asarray(groups)
    return {s: float(fn(np.asarray(y)[g == s], np.asarray(p)[g == s])) for s in sorted(set(g.tolist()))}


def risk_coverage(probs: np.ndarray, y: np.ndarray, n_points: int = 11) -> list[dict]:
    """Selective prediction: answer only when the top probability is at least a threshold; accuracy on what was answered."""
    probs, y = np.asarray(probs), np.asarray(y)
    conf = probs.max(1)
    pred = probs.argmax(1)
    out = []
    for q in np.linspace(0.0, 0.9, n_points):
        thr = float(np.quantile(conf, q)) if q > 0 else 0.0
        keep = conf >= thr
        out.append(dict(threshold=thr, coverage=float(keep.mean()), accuracy=float((pred[keep] == y[keep]).mean()) if keep.any() else None))
    return out
