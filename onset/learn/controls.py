"""Negative controls.  A result is only worth reporting if these behave.

``shuffled_labels``  Training labels are permuted inside every fold; the held-out score must fall to chance.  If it does not,
                     the pipeline is leaking information about the test clips (or the labels) somewhere.
``identity_probe``   Can *who the person is* be read from the features?  If yes, clip-level splits would let a model recognise
                     faces instead of expressions - which is why all folds here are subject-disjoint.
``leakage_gap``      The same model and features scored twice: once with subject-disjoint folds (valid) and once with clip-level
                     folds (invalid, shown only to quantify how much a leaky protocol would have flattered the number).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import evaluate as EV
from . import metrics as MET
from . import protocols as P
from .features import ClipSet, HAND_BLOCKS

SUSPICIOUS_MARGIN = 0.08      # shuffled-label UF1 more than this above chance is reported as a failed control


def shuffled_labels(factory: Callable, X: ClipSet, y: np.ndarray, groups: np.ndarray, folds, n_classes: int, seeds=(0, 1, 2),
                    alpha: float = 0.1, real_uf1: float | None = None) -> dict:
    chance = MET.chance_uf1_uniform(y, n_classes)
    scores = []
    for s in seeds:
        cp = EV.cross_predict(factory, X, y, groups, folds, n_classes, seed=s, alpha=alpha, shuffle_train=True)
        scores.append(MET.uf1(y, cp.pred, n_classes))
    m, sd = float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else None
    ok = (m - chance) <= SUSPICIOUS_MARGIN
    out = dict(name="shuffled_labels", uf1_runs=[float(v) for v in scores], uf1_mean=m, uf1_sd=sd, chance_uf1=chance,
               passed=bool(ok), threshold=chance + SUSPICIOUS_MARGIN)
    if real_uf1 is not None:
        out["real_uf1"] = float(real_uf1)
        out["margin_over_shuffled"] = float(real_uf1 - m)
    out["reading"] = ("training labels were permuted; the held-out score should sit at chance (%.2f). It is %.2f - %s."
                      % (chance, m, "as expected" if ok else "TOO HIGH: look for leakage or a labelling artefact before trusting any number"))
    return out


def identity_probe(X: ClipSet, groups: np.ndarray, k: int = 3, seed: int = 0, min_clips: int = 4) -> dict:
    """Decode the subject from the features with clip-level folds.  Chance is 1/(number of subjects)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from .models import Standardizer
    subs, counts = np.unique(groups, return_counts=True)
    keep_s = set(subs[counts >= max(min_clips, k)].tolist())
    keep = np.flatnonzero([g in keep_s for g in groups])
    if len(keep_s) < 3 or len(keep) < 3 * k:
        return dict(name="identity_probe", skipped=True, reason="too few subjects with several clips to attempt it")
    a = np.concatenate([X.hand[:, HAND_BLOCKS["pattern"]], X.hand[:, HAND_BLOCKS["timing"]]], axis=1)[keep]
    g = groups[keep]
    pred = np.empty(len(keep), dtype=object)
    for tr, te in StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(a, g):
        std = Standardizer().fit(a[tr])
        clf = LogisticRegression(C=0.3, max_iter=400, class_weight="balanced").fit(std(a[tr]), g[tr])
        pred[te] = clf.predict(std(a[te]))
    acc = float(np.mean(pred == g))
    chance = 1.0 / len(keep_s)
    share = float(max(np.unique(g, return_counts=True)[1]) / len(g))
    above = acc > max(2.0 * chance, chance + 0.10)
    reading = ("subject identity is decoded from the features at %.0f%% against %.0f%% by chance (%d subjects): %s"
               % (100 * acc, 100 * chance, len(keep_s),
                  "the features carry identity, so a clip-level split would partly reward recognising faces. The folds here are "
                  "subject-disjoint, which is what keeps this from inflating the reported numbers" if above else
                  "identity is only weakly present in the features"))
    return dict(name="identity_probe", accuracy=acc, chance=chance, majority_share=share, n_subjects=len(keep_s), n_clips=int(len(keep)),
                reading=reading)


def leakage_gap(factory: Callable, X: ClipSet, y: np.ndarray, groups: np.ndarray, n_classes: int, k: int = 5, seed: int = 0,
                alpha: float = 0.1) -> dict:
    """Same model, same features: subject-disjoint folds vs clip-level folds."""
    from sklearn.model_selection import KFold, StratifiedKFold
    ok_folds = P.kfold(groups, y, k=k, seed=seed)
    P.audit([_R(g, i) for i, g in enumerate(groups)], ok_folds)
    valid = EV.cross_predict(factory, X, y, groups, ok_folds, n_classes, seed=seed, alpha=alpha)
    min_class = int(np.bincount(y, minlength=n_classes)[np.bincount(y, minlength=n_classes) > 0].min())
    kk = max(2, min(k, min_class))
    splitter = StratifiedKFold(n_splits=kk, shuffle=True, random_state=seed) if min_class >= 2 else KFold(n_splits=kk, shuffle=True, random_state=seed)
    leaky = [P.Fold(f"clip:{i + 1}/{kk}", tr, te) for i, (tr, te) in enumerate(splitter.split(np.zeros(len(y)), y))]
    inv = EV.cross_predict(factory, X, y, groups, leaky, n_classes, seed=seed, alpha=alpha)
    u_valid, u_leaky = MET.uf1(y, valid.pred, n_classes), MET.uf1(y, inv.pred, n_classes)
    gap = float(u_leaky - u_valid)
    if gap > 0.02:
        reading = ("with clip-level folds the same model scores UF1 %.2f instead of %.2f: a leaky split would have added %+.2f. "
                   "Only the subject-disjoint number says anything about new people." % (u_leaky, u_valid, gap))
    else:
        reading = ("with clip-level folds the same model scores UF1 %.2f against %.2f with subject-disjoint folds (difference %+.2f): no "
                   "flattering showed up here, which on a small or synthetic set can be noise and does not make a clip-level split safe. "
                   "Only the subject-disjoint number says anything about new people." % (u_leaky, u_valid, gap))
    return dict(name="leakage_gap", uf1_subject_disjoint=float(u_valid), uf1_clip_level_INVALID=float(u_leaky),
                inflation=gap, chance_uf1=MET.chance_uf1_uniform(y, n_classes), k=k, reading=reading)


class _R:
    """Minimal record for ``protocols.audit`` when only groups are known."""

    def __init__(self, group, i):
        self.group, self.uid, self.folder, self.dataset, self.subject = group, f"clip{i}", f"/clip{i}", "x", group
        self.extra, self.onset, self.offset = {}, 0, 1
