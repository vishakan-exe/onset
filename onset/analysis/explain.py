"""Explanations that can be checked.

* Evidence ledger: for every ROI the measured displacement, what the leading hypothesis explains, what it leaves
  over, and how much that ROI contributes to the case (reduction in whitened squared error versus the
  no-expression model).
* Deletion test: ROIs are removed in order of ledger evidence and the posterior is recomputed; the same number of
  *random* ROIs is removed as a control.  If the ledger is faithful, removing its top ROIs must hurt the
  conclusion clearly more than removing random ones.  Support is measured on the log-odds scale (a probability
  saturates at 1 long before the evidence is exhausted, which made the first version of this test blind exactly
  where it matters) and the number of deletions needed to flip the conclusion is reported next to the curves.
  An explanation that fails the test is flagged, not hidden.
* Contrast: why hypothesis A and not B - the ROIs where their fits differ most.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..vision import geometry as G
from . import templates as TPL
from .au_engine import AUEngine, AUPosterior

DIRS = [(0, -1, "up"), (0, 1, "down")]


def direction_word(dx: float, dy: float, side: str) -> str:
    """Plain-language direction of an ROI displacement (image y down; 'out' = away from the face midline)."""
    mag = np.hypot(dx, dy)
    if mag < 1e-6:
        return "still"
    ang = np.degrees(np.arctan2(-dy, dx if side != "L" else -dx))       # 0 = outward, 90 = up
    labels = [(-22.5, 22.5, "outward"), (22.5, 67.5, "up and outward"), (67.5, 112.5, "up"),
              (112.5, 157.5, "up and inward"), (157.5, 180.1, "inward"), (-180.1, -157.5, "inward"),
              (-157.5, -112.5, "down and inward"), (-112.5, -67.5, "down"), (-67.5, -22.5, "down and outward")]
    for lo, hi, w in labels:
        if lo <= ang < hi:
            return w
    return "outward"


def _side(name: str) -> str:
    return "L" if name.endswith("_L") else "R" if name.endswith("_R") else "C"


def evidence_ledger(engine: AUEngine, post: AUPosterior, y: np.ndarray, sigma: np.ndarray, mask: Optional[np.ndarray],
                    iod_canon: float, map_hyp: Optional[dict] = None) -> list[dict]:
    """One row per ROI, sorted by evidence (descending).

    A region's *evidence* is how much of its movement the named AUs account for: with head motion, blink and speech
    components removed (as fitted under the leading hypothesis), the whitened squared movement of the ROI before and
    after subtracting the AU part of the fit.  A region the AUs do not touch therefore has no evidence, however the
    nuisance terms had to shift to accommodate them."""
    h = map_hyp or post.hypotheses[0]
    fit = engine.fit_hypothesis(h["aus"], y, sigma, mask, h["blink"], h["speech"])
    w = fit["weights"]
    y = np.asarray(y, float)
    after_nuisance = y - fit["nuis"]                       # what is left once head motion / blink / speech are removed
    rows = []
    for r, name in enumerate(G.ROI_NAMES):
        if mask is not None and mask[r] <= 0:
            rows.append(dict(roi=name, group=G.GROUP_OF[name], used=False, evidence=0.0))
            continue
        e_before = float(((after_nuisance[r] * w[r]) ** 2).sum())
        e_after = float(((fit["resid"][r] * w[r]) ** 2).sum())
        dx, dy = float(y[r, 0]), float(y[r, 1])
        z = float(np.sqrt(((y[r] / np.maximum(sigma[r], 1e-6)) ** 2).sum()))
        rows.append(dict(
            roi=name, group=G.GROUP_OF[name], used=True,
            dx_px=dx, dy_px=dy, mag_px=float(np.hypot(dx, dy)), mag_pct_iod=float(np.hypot(dx, dy) / iod_canon * 100),
            direction=direction_word(dx, dy, _side(name)), z=z,
            explained_dx=float(fit["au"][r, 0]), explained_dy=float(fit["au"][r, 1]),
            resid_dx=float(fit["resid"][r, 0]), resid_dy=float(fit["resid"][r, 1]),
            evidence=e_before - e_after,
            aus=[a for a in h["aus"] if np.linalg.norm(TPL.au_templates()[TPL.AUS.index(a)][r]) > 0.15]))
    rows.sort(key=lambda d: -d.get("evidence", 0.0))
    return rows


LOGIT_EPS = 1e-6


def _logit(p: float, eps: float = LOGIT_EPS) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return float(np.log(p) - np.log1p(-p))


def _sigmoid(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def _support(post: AUPosterior, target_aus: list[int]) -> float:
    """Posterior support for the original conclusion, in log-odds: the *lowest* inclusion probability over the AUs it
    named (one lost AU is a changed conclusion); odds of 'some AU-like pattern' when it named none."""
    if not target_aus:
        return _logit(1.0 - post.p_null)
    return _logit(min(post.pip[TPL.AUS.index(a)] for a in target_aus))


def _first_flip(curve: list[float]) -> Optional[int]:
    """Number of deletions after which the conclusion is no longer the more probable one (support < 0)."""
    for n, v in enumerate(curve):
        if n and v < 0.0:
            return n
    return None


def deletion_test(engine: AUEngine, y: np.ndarray, sigma: np.ndarray, mask: Optional[np.ndarray], ledger: list[dict],
                  target_aus: list[int], p_blink: float, p_speech: float, n_max: int = 8, n_random: int = 8,
                  seed: int = 0) -> dict:
    """Faithfulness of the ledger: support for the conclusion (log-odds) as the ledger's top ROIs - or the same number
    of random ROIs - are deleted.  Returns both curves, the number of deletions that flips the conclusion, and a
    verdict:

    * ``faithful``  - deleting the ledger's ROIs hurts the conclusion clearly more than deleting random ones
                      (mean gap >= 1 log-odds unit and the ranked curve is below at least 80 % of the random orderings);
    * ``robust``    - the conclusion stays above 95 % even with the top ROIs gone (the test cannot discriminate, but the
                      finding does not hinge on a few regions);
    * ``diffuse``   - the conclusion falls as fast for random ROIs as for the ledger's: the evidence is spread out and the
                      ranking does not single out special regions;
    * ``inverted``  - random deletions hurt more than the ledger's: the ranking misleads;
    * ``na``        - nothing to test (the conclusion was not above 50 % to begin with)."""
    rng = np.random.default_rng(seed)
    R = len(G.ROI_NAMES)
    used = [r for r in range(R) if mask is None or mask[r] > 0]
    ranked = [G.ROI_NAMES.index(d["roi"]) for d in ledger if d.get("used") and d.get("evidence", 0) > 0]
    n_max = int(min(n_max, len(ranked), max(1, len(used) // 3)))
    base_mask = np.ones(R) if mask is None else np.asarray(mask, float).copy()
    metric = ("log-odds of the lowest inclusion probability among the named AUs" if target_aus
              else "log-odds of 'some AU-like pattern'")

    def run(removed):
        m = base_mask.copy()
        m[list(removed)] = 0.0
        post = engine.infer(y, sigma, m, p_blink, p_speech, n_samples=8, rng=np.random.default_rng(1))
        return _support(post, target_aus)

    base = run([])
    out = dict(n=list(range(0, n_max + 1)), metric=metric, unit="log-odds", baseline=base,
               removed_order=[G.ROI_NAMES[r] for r in ranked[:n_max]])
    if base <= 0.0 or n_max < 1:
        out.update(ranked=[base], random_mean=[base], random_sd=[0.0], ranked_p=[float(_sigmoid(base))],
                   random_mean_p=[float(_sigmoid(base))], gap=0.0, p_empirical=1.0, faithful=None, verdict="na",
                   flips=dict(ranked=None, random_median=None, random_never=0, n_random=0))
        return out
    top_curve = [base] + [run(ranked[:n]) for n in range(1, n_max + 1)]
    rand = np.zeros((n_random, n_max + 1))
    for i in range(n_random):
        order = rng.permutation(used)
        rand[i, 0] = base
        for n in range(1, n_max + 1):
            rand[i, n] = run(order[:n])
    rand_mean, rand_sd = rand.mean(0), rand.std(0)
    gap = float(np.mean(rand_mean[1:] - np.array(top_curve[1:])))
    # one-sided empirical p: fraction of random orderings that hurt the conclusion at least as much (mean over n)
    top_auc = float(np.mean(top_curve[1:]))
    p_emp = float((1 + (rand[:, 1:].mean(1) <= top_auc).sum()) / (n_random + 1))
    flips_rand = [_first_flip(list(r)) for r in rand]
    seen = sorted(f for f in flips_rand if f is not None)
    never = int(sum(f is None for f in flips_rand))
    med = None if never > n_random / 2 else float(np.median(seen + [n_max + 1] * never))
    robust = min(top_curve) > _logit(0.95)
    if robust:
        verdict = "robust"
    elif gap >= 1.0 and p_emp <= 0.2:
        verdict = "faithful"
    elif gap <= -1.0:
        verdict = "inverted"
    else:
        verdict = "diffuse"
    out.update(ranked=top_curve, random_mean=rand_mean.tolist(), random_sd=rand_sd.tolist(),
               ranked_p=_sigmoid(top_curve).tolist(), random_mean_p=_sigmoid(rand_mean).tolist(), gap=gap, p_empirical=p_emp,
               faithful=True if verdict == "faithful" else None if verdict == "robust" else False, verdict=verdict,
               flips=dict(ranked=_first_flip(top_curve), random_median=med, random_never=never, n_random=n_random))
    return out


def contrast(engine: AUEngine, y, sigma, mask, a: dict, b: dict, top: int = 6) -> list[dict]:
    """ROIs that separate hypothesis ``a`` from ``b`` (each a dict(aus, blink, speech)): positive = favours a."""
    fa = engine.fit_hypothesis(a["aus"], y, sigma, mask, a.get("blink", False), a.get("speech", False))
    fb = engine.fit_hypothesis(b["aus"], y, sigma, mask, b.get("blink", False), b.get("speech", False))
    w = fa["weights"]
    rows = []
    for r, name in enumerate(G.ROI_NAMES):
        ea = float(((fa["resid"][r] * w[r]) ** 2).sum())
        eb = float(((fb["resid"][r] * w[r]) ** 2).sum())
        rows.append(dict(roi=name, favours_a=eb - ea, a_vec=fa["fit"][r].tolist(), b_vec=fb["fit"][r].tolist(),
                         measured=np.asarray(y[r]).tolist()))
    rows.sort(key=lambda d: -abs(d["favours_a"]))
    return rows[:top]


def narrative(post: AUPosterior, ledger: list[dict], iod_canon: float, duration: Optional[float]) -> str:
    """One plain-language paragraph; every number in it is traceable to the ledger."""
    named = [(a, p) for a, p in zip(post.aus, post.pip) if p >= 0.5]
    if post.p_blink > 0.5:
        return ("This movement coincides with a blink: the eyelid region is excluded from the analysis because the "
                "lids moving across the eye dominates anything the skin does.")
    if post.p_speech > 0.5:
        return "Mouth movement from talking or chewing is the most likely cause; the lower face is not interpretable here."
    if not named:
        return ("The movement is real but does not match any single AU pattern clearly enough to name one "
                f"(probability of 'no AU-like pattern' {post.p_null:.0%}).")
    top = [d for d in ledger if d.get("used") and d.get("evidence", 0) > 0][:3]
    regs = ", ".join(f"{d['roi'].replace('_', ' ')} ({d['direction']}, {d['mag_pct_iod']:.1f}% IOD)" for d in top)
    names = " + ".join(f"AU{a} ({TPL.AU_NAMES[a].lower()})" for a, _ in named)
    dur = f" lasting about {duration * 1000:.0f} ms" if duration else ""
    return f"Pattern consistent with {names}{dur}. Strongest evidence: {regs}."
