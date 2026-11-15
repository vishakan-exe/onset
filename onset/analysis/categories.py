"""From AU evidence to category *hypotheses* (never a verdict).

Micro-expression datasets label clips with emotion categories that were assigned from the AUs, the video context
and (sometimes) the participant's report, so the AU -> category map is loose and differs between datasets.  This
module keeps that map explicit and inspectable:

* a weight table ``W[c, k]`` (log-odds contribution of AU k to category c) plus a bias per category,
* an explicit ``unclear`` option whose log-odds is constant, so weakly-supported patterns abstain,
* ``P(c | y) = sum_S P(S | y) P(c | S)`` - the AU posterior is propagated, not collapsed to a point estimate.

The shipped weights are an *expert prior* written from FACS/EMFACS prototypes.  They are not fitted to any data.
``fit_weights`` fits the same functional form to AU/category annotations (e.g. the CASME II / SAMM / MMEW
spreadsheets that come with the licensed datasets) with cross-validation, which is what the learning harness does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import templates as TPL

BASIC7 = ["happiness", "surprise", "fear", "anger", "disgust", "sadness", "contempt"]
NEGATIVE = {"fear", "anger", "disgust", "sadness", "contempt"}

# log-odds weights per AU, per basic emotion (expert prior from FACS / EMFACS prototypes)
_W7: dict[str, dict[int, float]] = {
    "happiness": {12: 3.0, 6: 1.6, 10: -0.3, 15: -1.0, 4: -1.0, 14: -0.4},
    "surprise": {1: 1.6, 2: 1.6, 5: 2.0, 4: -1.0, 12: -0.5},
    "fear": {1: 1.0, 2: 1.0, 4: 1.0, 5: 1.6, 7: 0.3},
    "anger": {4: 2.0, 5: 1.0, 7: 1.6, 17: 0.6, 10: 0.3},
    "disgust": {9: 2.8, 10: 2.2, 4: 0.4, 7: 0.3, 15: 0.6, 17: 0.5},
    "sadness": {1: 1.0, 4: 1.2, 15: 2.2, 17: 1.2},
    "contempt": {14: 2.8, 12: 0.5, 10: 0.3},
}
_B7 = {"happiness": 0.0, "surprise": -0.3, "fear": -0.8, "anger": -0.3, "disgust": -0.3, "sadness": -0.3, "contempt": -0.5}
_WCASME: dict[str, dict[int, float]] = {
    "happiness": _W7["happiness"], "surprise": _W7["surprise"], "disgust": _W7["disgust"],
    "repression": {14: 1.8, 17: 1.8, 15: 1.4, 4: 0.3},
}
_BCASME = {"happiness": 0.0, "surprise": -0.3, "disgust": -0.3, "repression": -0.4}
UNCLEAR_LOGIT = 1.0


@dataclass
class Scheme:
    name: str
    labels: list
    W: np.ndarray                 # (C, K)
    bias: np.ndarray              # (C,)
    unclear: float = UNCLEAR_LOGIT
    note: str = ""
    groups: Optional[dict] = None  # label -> list of basic7 labels it sums (for derived schemes)

    def logits(self, present: Sequence[int]) -> np.ndarray:
        """(C+1,) logits for a set of AU numbers; the last entry is 'unclear'."""
        z = self.bias.copy()
        for a in present:
            z = z + self.W[:, TPL.AUS.index(a)]
        return np.concatenate([z, [self.unclear]])

    def probs_given_set(self, present: Sequence[int]) -> np.ndarray:
        z = self.logits(present)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()


def _table(spec: dict, bias: dict, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    W = np.zeros((len(labels), len(TPL.AUS)))
    for i, lab in enumerate(labels):
        for au, w in spec.get(lab, {}).items():
            W[i, TPL.AUS.index(au)] = w
    return W, np.array([bias.get(lab, 0.0) for lab in labels])


def basic7() -> Scheme:
    W, b = _table(_W7, _B7, BASIC7)
    return Scheme("basic7", BASIC7, W, b, note="Seven basic emotions (EMFACS-style AU prototypes; expert prior).")


def casme2_5() -> Scheme:
    labs = ["happiness", "disgust", "surprise", "repression"]
    W, b = _table(_WCASME, _BCASME, labs)
    # CASME II's fifth class ('others') is what the explicit 'unclear' option is for
    return Scheme("casme2_5", labs + ["others"], np.vstack([W, np.zeros((1, W.shape[1]))]),
                  np.concatenate([b, [0.6]]), unclear=UNCLEAR_LOGIT,
                  note="CASME II-style classes; 'others' absorbs patterns outside the four prototypes.")


def megc3() -> Scheme:
    s = basic7()
    s.name = "megc3"
    s.note = "MEGC composite classes: positive = happiness; negative = anger, disgust, fear, sadness, contempt; surprise."
    s.groups = {"positive": ["happiness"], "negative": sorted(NEGATIVE), "surprise": ["surprise"]}
    s.labels = list(s.groups)
    return s


SCHEMES = {"megc3": megc3, "casme2_5": casme2_5, "basic7": basic7}


@dataclass
class CategoryResult:
    scheme: str
    labels: list
    probs: np.ndarray            # (C,) over the scheme's labels, summing with `unclear` (and `artefact`) to 1
    unclear: float
    artefact: float              # probability that the movement is a blink / speech, not an expression
    top: str
    top_p: float
    margin: float                # top - runner-up
    abstain: bool
    set: list = field(default_factory=list)   # conformal prediction set (labels), if a calibrator was supplied
    note: str = ""


def category_posterior(set_probs: dict, p_artefact: float, scheme: Scheme, p_null: float = 0.0,
                       abstain_below: float = 0.5, calibrator=None) -> CategoryResult:
    """Propagate the posterior over AU sets (``set_probs``: frozenset(AU) -> P) into category probabilities.

    ``p_artefact`` (blink or speech) is carved out first: those events are not expressions.  ``p_null`` (no
    AU-like pattern) is part of ``unclear``."""
    base = SCHEMES["basic7"]() if scheme.groups else scheme
    C = len(base.labels)
    acc = np.zeros(C + 1)
    total = sum(set_probs.values()) or 1.0
    for S, p in set_probs.items():
        acc += (p / total) * base.probs_given_set(sorted(S))
    if scheme.groups:
        out = np.array([sum(acc[base.labels.index(l)] for l in scheme.groups[lab]) for lab in scheme.labels] + [acc[-1]])
    else:
        out = acc
    out = out * (1.0 - p_artefact)
    labels = list(scheme.labels)
    probs, unclear = out[:-1], float(out[-1])
    order = np.argsort(-probs)
    top = labels[int(order[0])]
    top_p = float(probs[order[0]])
    margin = float(top_p - (probs[order[1]] if len(order) > 1 else 0.0))
    abstain = bool(top_p < abstain_below or unclear > top_p or p_artefact > 0.5)
    pset = calibrator.predict_set(np.concatenate([probs, [unclear]]), labels + ["unclear"]) if calibrator else []
    return CategoryResult(scheme.name, labels, probs, unclear, float(p_artefact), top, top_p, margin, abstain, pset,
                          scheme.note)


def fit_weights(au_sets: Sequence[Sequence[int]], labels: Sequence[str], classes: Sequence[str], l2: float = 1.0,
                iters: int = 400, lr: float = 0.1, seed: int = 0) -> Scheme:
    """Fit the same softmax form to annotated (AU set, category) pairs by L2-regularised multinomial logistic
    regression (full-batch gradient descent).  Features are AU presence/absence; no video is needed, so this runs
    on the annotation spreadsheets alone.  Returns a Scheme (without an 'unclear' class: abstention is set by the
    conformal/threshold layer on top)."""
    C, K = len(classes), len(TPL.AUS)
    X = np.zeros((len(au_sets), K))
    for i, s in enumerate(au_sets):
        for a in s:
            if a in TPL.AUS:
                X[i, TPL.AUS.index(a)] = 1.0
    y = np.array([list(classes).index(l) for l in labels])
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, (C, K))
    b = np.zeros(C)
    Y = np.eye(C)[y]
    for _ in range(iters):
        z = X @ W.T + b
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        g = (p - Y)
        W -= lr * (g.T @ X / len(X) + l2 * W / len(X))
        b -= lr * g.mean(0)
    return Scheme("fitted", list(classes), W, b, unclear=-50.0, note="Fitted to annotated AU sets.")
