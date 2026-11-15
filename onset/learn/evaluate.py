"""The fold loop shared by the run command and the controls: fit on training subjects, predict held-out subjects.

For every fold the model is fitted on the training clips only; its predictions for subjects it did not train on
(``model.cal_logits``) fit the temperature and the conformal threshold; only then are the held-out subjects scored.
Nothing about a test subject reaches fitting, calibration or model selection.  Each clip is predicted exactly once per
seed, by the fold in which its subject was held out.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from . import calibrate as CAL
from . import metrics as MET
from .features import ClipSet
from .protocols import Fold


@dataclass
class CrossPrediction:
    logits: np.ndarray                 # (n, C) raw model outputs
    probs: np.ndarray                  # (n, C) calibrated probabilities
    pred: np.ndarray                   # (n,) predicted class
    sets: list                         # conformal prediction sets (lists of class indices)
    fold: np.ndarray                   # (n,) index of the fold that predicted each clip
    temperatures: list = field(default_factory=list)
    set_alpha: float = 0.1
    fit_seconds: list = field(default_factory=list)
    infer_ms: float = 0.0              # mean model-inference time per clip
    extras: dict = field(default_factory=dict)
    seed: int = 0


def cross_predict(factory: Callable[[], object], X: ClipSet, y: np.ndarray, groups: np.ndarray, folds: Sequence[Fold], n_classes: int,
                  seed: int = 0, alpha: float = 0.1, shuffle_train: bool = False,
                  progress: Optional[Callable[[int, int], None]] = None) -> CrossPrediction:
    """Predict every clip with a model that never saw its subject.  ``shuffle_train`` permutes the training labels inside each
    fold (a negative control: the held-out score must then fall to chance)."""
    n = len(y)
    logits = np.zeros((n, n_classes))
    probs = np.zeros((n, n_classes))
    pred = np.full(n, -1, int)
    sets: list = [None] * n
    fold_of = np.full(n, -1, int)
    temps, fit_s, infer_ms = [], [], []
    extras: dict = {}
    rng = np.random.default_rng(seed + 7919)
    for fi, f in enumerate(folds):
        ytr = y[f.train].copy()
        if shuffle_train:
            ytr = ytr[rng.permutation(len(ytr))]
        model = factory()
        model.fit(X.take(f.train), ytr, groups[f.train], n_classes, seed)
        Xte = X.take(f.test)
        t0 = time.perf_counter()
        lg = model.logits(Xte)
        infer_ms.append((time.perf_counter() - t0) * 1000.0 / max(len(f.test), 1))
        cal = CAL.Calibrator()
        if model.cal_logits is not None and model.cal_y is not None:
            cal.fit(model.cal_logits, model.cal_y, alpha=alpha)
        pr = cal.probs(lg)
        pd_ = model.predict(Xte) if model.name == "prior" else pr.argmax(1)
        logits[f.test], probs[f.test], pred[f.test] = lg, pr, pd_
        for i, s in zip(f.test, cal.sets(pr)):
            sets[i] = s
        fold_of[f.test] = fi
        temps.append(float(cal.temperature) if cal.fitted else None)
        fit_s.append(float(model.fit_seconds))
        if hasattr(model, "unclear"):
            extras.setdefault("unclear", np.zeros(n))[f.test] = model.unclear(Xte)
        if hasattr(model, "member_disagreement"):
            extras.setdefault("disagreement", np.zeros(n))[f.test] = model.member_disagreement(Xte)
        if hasattr(model, "n_parameters"):
            extras["n_parameters"] = model.n_parameters()
        if progress:
            progress(fi + 1, len(folds))
    return CrossPrediction(logits, probs, pred, sets, fold_of, temps, alpha, fit_s, float(np.mean(infer_ms)), extras, seed)


# --------------------------------------------------------------------------------------------------------------
def _ci(fn, y, p, g, n_boot, seed):
    est, lo, hi = MET.subject_bootstrap(fn, y, p, g, n_boot=n_boot, seed=seed)
    return dict(value=est, lo=lo, hi=hi)


def stack_seeds(cps: Sequence[CrossPrediction], y: np.ndarray, groups: np.ndarray):
    """Concatenate seeds so that a subject-level resample takes all of a subject's clips from every seed together."""
    S = len(cps)
    return (np.tile(y, S), np.concatenate([c.pred for c in cps]), np.tile(groups, S))


def model_report(cps: Sequence[CrossPrediction], y: np.ndarray, groups: np.ndarray, classes: Sequence[str], n_boot: int = 1000,
                 seed: int = 0) -> dict:
    """Everything reported for one model over one or more seeds."""
    C = len(classes)
    ya, pa, ga = stack_seeds(cps, y, groups)
    fns = dict(uf1=lambda a, b: MET.uf1(a, b, C), uar=lambda a, b: MET.uar(a, b, C), accuracy=MET.accuracy)
    head = {k: _ci(f, ya, pa, ga, n_boot, seed) for k, f in fns.items()}
    per_seed = {k: [float(f(y, c.pred)) for c in cps] for k, f in fns.items()}
    for k in head:
        head[k]["seed_mean"] = float(np.mean(per_seed[k]))
        head[k]["seed_sd"] = float(np.std(per_seed[k], ddof=1)) if len(cps) > 1 else None
    summ = MET.summary(ya, pa, C, classes)
    probs = np.concatenate([c.probs for c in cps])
    sets = [s for c in cps for s in c.sets]
    rel = CAL.reliability(probs, ya)
    temps = [t for c in cps for t in c.temperatures if t is not None]
    out = dict(
        n_clips=int(len(y)), n_subjects=int(len(set(groups.tolist()))), n_seeds=len(cps), headline=head,
        chance=dict(uf1=summ["chance_uf1"], uar=summ["chance_uar"], majority_accuracy=summ["majority_accuracy"]),
        per_class=summ["per_class"], confusion=summ["confusion"],
        calibration=dict(ece=rel["ece"], nll=rel["nll"], brier=rel["brier"], bins=rel["bins"],
                         temperature=float(np.mean(temps)) if temps else None),
        sets=dict(alpha=cps[0].set_alpha, **CAL.set_stats(sets, ya, C)),
        risk_coverage=MET.risk_coverage(probs, ya),
        latency=dict(inference_ms_per_clip=float(np.mean([c.infer_ms for c in cps])),
                     fit_seconds_per_fold=float(np.mean([s for c in cps for s in c.fit_seconds])) if any(c.fit_seconds for c in cps) else None),
    )
    if "n_parameters" in cps[0].extras:
        out["n_parameters"] = int(cps[0].extras["n_parameters"])
    if "unclear" in cps[0].extras:
        u = np.concatenate([c.extras["unclear"] for c in cps])
        out["abstention_mass"] = dict(mean_unclear=float(u.mean()), share_unclear_over_half=float((u > 0.5).mean()))
    if "disagreement" in cps[0].extras:
        d = np.concatenate([c.extras["disagreement"] for c in cps])
        wrong = np.concatenate([c.pred != y for c in cps])
        out["member_disagreement"] = dict(mean=float(d.mean()), mean_when_wrong=float(d[wrong].mean()) if wrong.any() else None,
                                          mean_when_right=float(d[~wrong].mean()) if (~wrong).any() else None)
    return out


def pairwise(preds: dict, y: np.ndarray, groups: np.ndarray, classes: Sequence[str], pairs: Sequence[tuple], n_perm: int = 2000,
             seed: int = 0) -> list[dict]:
    """Paired subject-level permutation tests of UF1 differences (Holm-adjusted over the pairs given)."""
    C = len(classes)
    fn = lambda a, b: MET.uf1(a, b, C)
    rows = []
    for a, b in pairs:
        ya, pa, ga = stack_seeds(preds[a], y, groups)
        _, pb, _ = stack_seeds(preds[b], y, groups)
        r = MET.paired_permutation(fn, ya, pa, pb, ga, n_perm=n_perm, seed=seed)
        rows.append(dict(a=a, b=b, metric="UF1", difference=r["difference"], p=r["p"], n_subjects=r["n_subjects"]))
    adj = MET.holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = q
    return rows
