#!/usr/bin/env python
"""Calibrate and evaluate the instrument on the phantom corpus (see ``build_phantom_eval.py``).

This is *instrument calibration*: the footage is synthetic, the faces are not people, and every number it produces
describes how the measurement chain behaves on scripted movements with known ground truth.  It is not evidence about
accuracy on human micro-expressions.  The learning harness (``onset.learn``) is where licensed datasets are evaluated.

Ground truth is expressed in the instrument's own units.  Every scripted movement carries ``exact`` (written by
``build_phantom_eval.py``): the true canonical-space displacement between the two frames the measurement uses,
projected on the same nuisance-free AU templates the engine fits, so "true amplitude of AU 12 = 0.55 % of the eye
distance" means exactly what the report's AU amplitude means.  Without it the generator's own amplitude would be
compared with a template coefficient and the comparison would be about units, not about the instrument.

What it does
  1. Fits the AU engine's calibration knobs (model-error ``eta``, posterior temper, AU-count prior) on the
     *calibration* identities (A-C) by held-out log-loss of the AU inclusion probabilities - a local search that keeps
     walking until the optimum is inside the explored range - and evaluates on the *test* identities (D-F) that none of
     these settings ever saw.
  2. Sweeps the spotting threshold: recall (MEGC-style IoU >= 0.5 and a looser apex match) against false alarms per
     10 s of footage, with where the false alarms come from (blink, speech, head, noise).
  3. Draws the sensitivity envelope: detection probability against true movement size for each camera (descriptive, all
     clips), and checks the clip-level "smallest detectable movement" the report prints against what was actually detected
     (-> ``mda_scale``, ``mda90_scale``).  These two factors are the one thing fitted on all six faces (they only rescale a
     printed quantity); they are checked by refitting on A-C and on D-F alone and by applying the A-C curve to D-F.
  4. Checks the amplitudes: bias and 90 % interval coverage against the true engine-unit amplitude, with a
     split-conformal widening factor fitted on the calibration identities (-> ``amp_interval_k``).
  5. Measures how faithfully the measurement chain reproduces the true movement (``pair_gain``) and how much of the
     true pattern the AU templates can explain at all (``fit_r2``).
  6. Evaluates category hypotheses with abstention and split-conformal prediction sets (propagation of AU
     uncertainty only: the category truth is the expert map applied to the true AU set).
  7. Runs control experiments (shuffled truth, prior-only guess, noise-only patterns) so the numbers cannot be
     mistaken for something the data does not support.

Outputs: ``data/calibration.json`` (parameters read by the pipeline) and ``data/evidence/phantom_eval.json``
(everything the Evidence station draws).

    python scripts/calibrate_engine.py --corpus DIR --out data --cache DIR2 --workers 2
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import pickle
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from onset.analysis import categories as CAT                      # noqa: E402
from onset.analysis import quality as QUAL                        # noqa: E402
from onset.analysis import templates as TPL                       # noqa: E402
from onset.analysis import uncertainty as UNC                     # noqa: E402
from onset.analysis.au_engine import AUEngine                     # noqa: E402
from onset.analysis.pipeline import _event_status                 # noqa: E402
from onset.analysis.scoring import EXPRESSION_STATUSES, match_events  # noqa: E402
from onset.jsonsafe import clean                                  # noqa: E402
from onset.vision.spotting import whiten_templates                # noqa: E402

AUS = list(TPL.AUS)
CAL_PHANTOMS = "ABC"
T_OP_DEFAULT = 5.0               # operating threshold used while tuning the engine (re-run at the chosen one)
PRIORS = [(0.12, 0.38, 0.32, 0.18), (0.10, 0.46, 0.32, 0.12), (0.08, 0.52, 0.30, 0.10), (0.15, 0.30, 0.32, 0.23)]
ETAS = [0.05, 0.10, 0.15, 0.22, 0.30, 0.40, 0.55, 0.75]
TEMPERS = [0.15, 0.25, 0.35, 0.50, 0.70, 1.00]
ETA_BOUNDS, TEMPER_BOUNDS = (0.02, 1.6), (0.05, 2.0)
THRESHOLDS = [round(x, 2) for x in np.arange(3.5, 9.01, 0.25)]
CAMERAS = ["webcam30", "phone30", "webcam24_lowlight", "webcam15", "highspeed100"]
AMP_FLOOR_PCT = 0.15             # true AU amplitudes below this (% IOD) are noise-level: not used to judge amplitude read-outs
AMP_ALPHA = 0.10


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------------------------------------------
def load_paths(corpus: str) -> list[str]:
    return sorted(glob.glob(os.path.join(corpus, "clip_*.pkl")))


def load(path: str) -> dict:
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    d["split"] = "cal" if d["spec"]["phantom"] in CAL_PHANTOMS else "test"
    d["path"] = path
    return d


def exact_of(d: dict) -> list:
    """Engine-unit truth of every scripted movement (``None`` entries when the corpus predates it)."""
    ev, ex = d["truth"]["events"], d.get("exact")
    if ex is None or len(ex) != len(ev):
        return [None] * len(ev)
    return ex


def det_dict(row: dict, i: int, status: str = "micro") -> dict:
    return dict(id=i, status=status, t_apex=float(row["apex"]), t_onset=float(row["onset"]),
                t_offset=None if row["offset"] is None else float(row["offset"]), aus=[], row=i)


def truth_sets(truth_events):
    return [sorted(int(a) for a in e["aus"]) for e in truth_events]


def _mda(q: dict, a: int) -> float:
    m = q["mda50_pct"]
    return float(m[a] if a in m else m[str(a)])


# --------------------------------------------------------------------------------------------------------------
# 1. the engine's calibration knobs
def grid_worker(args):
    path, grid, t_op = args
    d = load(path)
    truth = d["truth"]["events"]
    rows = [r for r in d["events"] if r["cand"]["score"] >= t_op]
    out = {}
    if not truth or not rows:
        return path, d["split"], out
    m = match_events(truth, [det_dict(r, i) for i, r in enumerate(rows)])
    pairs = [(x["truth"], m["detections"][x["det"]]["row"]) for x in m["matches"] if x["det"] is not None]
    tsets = truth_sets(truth)
    eng, cur_prior = None, None
    for gi, (eta, temper, pi) in enumerate(grid):
        if pi != cur_prior:
            eng = AUEngine(d["cent"], d["centre"], d["iod_canon"], size_prior=PRIORS[pi])
            cur_prior = pi
        eng.eta, eng.temper = eta, temper
        res = []
        for ti, ri in pairs:
            r = rows[ri]
            post = eng.infer(r["pattern"], r["sigma"], r["mask"], r["p_blink"], r["p_speech"], n_samples=8)
            res.append((post.pip.copy(), tsets[ti]))
        out[gi] = res
    return path, d["split"], out


def au_scores(pips, truths, thr: float = 0.5) -> dict:
    P = np.array(pips, float).reshape(len(pips), len(AUS))
    Y = np.zeros_like(P)
    for i, t in enumerate(truths):
        for a in t:
            Y[i, AUS.index(a)] = 1.0
    pred = P >= thr
    tp, fp, fn = int((pred & (Y > 0)).sum()), int((pred & (Y == 0)).sum()), int((~pred & (Y > 0)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    exact = float(np.mean([set(AUS[j] for j in np.flatnonzero(pred[i])) == set(truths[i]) for i in range(len(P))])) if len(P) else float("nan")
    per = {}
    for j, a in enumerate(AUS):
        t_, f_, n_ = int((pred[:, j] & (Y[:, j] > 0)).sum()), int((pred[:, j] & (Y[:, j] == 0)).sum()), int((~pred[:, j] & (Y[:, j] > 0)).sum())
        per[str(a)] = dict(n_true=int(Y[:, j].sum()), tp=t_, fp=f_, fn=n_, precision=t_ / max(t_ + f_, 1), recall=t_ / max(t_ + n_, 1),
                           f1=2 * t_ / max(2 * t_ + f_ + n_, 1))
    return dict(n=int(len(P)), log_loss=UNC.log_loss_binary(P.ravel(), Y.ravel()), brier=UNC.brier_binary(P.ravel(), Y.ravel()),
                ece=UNC.ece_binary(P.ravel(), Y.ravel()), precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9),
                exact_set=exact, per_au=per, reliability=UNC.reliability_curve(P.ravel(), Y.ravel(), 10))


def run_grid(paths, workers, cache, t_op) -> dict:
    """Coarse grid over (eta, temper), then a local search that walks until the best setting has all its neighbours
    evaluated (so the optimum is never an artefact of the grid's edge), then the AU-count priors, then the search again."""
    cpath = os.path.join(cache, f"grid_T{t_op:.2f}.pkl")
    results: dict = {}
    if os.path.exists(cpath):
        with open(cpath, "rb") as fh:
            results = pickle.load(fh)
    pool = mp.get_context("spawn").Pool(workers)

    def sweep(grid):
        todo = [g for g in dict.fromkeys(grid) if g not in results]
        if not todo:
            return
        log(f"grid: {len(todo)} settings x {len(paths)} clips")
        agg = {g: {"cal": [], "test": []} for g in todo}
        for p, split, out in pool.imap_unordered(grid_worker, [(p, todo, t_op) for p in paths]):
            for gi, res in out.items():
                agg[todo[gi]][split].extend(res)
        for g in todo:
            results[g] = {s: au_scores([r[0] for r in agg[g][s]], [r[1] for r in agg[g][s]]) for s in ("cal", "test") if agg[g][s]}
        with open(cpath, "wb") as fh:
            pickle.dump(results, fh)

    def best():
        return min((g for g in results if "cal" in results[g]), key=lambda g: results[g]["cal"]["log_loss"])

    def clip(v, lo_hi):
        return round(float(min(max(v, lo_hi[0]), lo_hi[1])), 3)

    def local_search(max_iter=8):
        for _ in range(max_iter):
            b = best()
            nb = [(clip(b[0] * f, ETA_BOUNDS), b[1], b[2]) for f in (0.8, 1.25)] + \
                 [(b[0], clip(b[1] * f, TEMPER_BOUNDS), b[2]) for f in (0.7, 1.4)]
            if all(g in results for g in nb):
                return
            sweep(nb)
            if best() == b:
                return

    try:
        sweep([(e, t, 0) for e in ETAS for t in TEMPERS])
        local_search()
        b = best()
        log("eta/temper optimum", b, round(results[b]["cal"]["log_loss"], 4))
        sweep([(b[0], b[1], pi) for pi in range(1, len(PRIORS))])
        local_search()
        b = best()
        log("optimum with prior", b, round(results[b]["cal"]["log_loss"], 4))
    finally:
        pool.close()
        pool.join()
    return results


# --------------------------------------------------------------------------------------------------------------
# 2. final posteriors at the chosen parameters
def final_worker(args):
    path, eta, temper, pi = args
    d = load(path)
    eng = AUEngine(d["cent"], d["centre"], d["iod_canon"], eta=eta, temper=temper, size_prior=PRIORS[pi])
    out = []
    for i, r in enumerate(d["events"]):
        post = eng.infer(r["pattern"], r["sigma"], r["mask"], r["p_blink"], r["p_speech"], n_samples=600)
        dur = None if r["offset"] is None else r["offset"] - r["onset"]
        em = SimpleNamespace(head_excursion=r["head_excursion"], blink_cue=r["blink_cue"], speech_cue=r["speech_cue"],
                             sustained=r["sustained"], duration=dur)
        status, _, _ = _event_status(SimpleNamespace(kind=r["cand"]["kind"], masked=r["cand"].get("masked", "")), post, em)
        out.append(dict(i=i, status=status, pip=post.pip, amp_given=post.amp_given, amp_lo=post.amp_lo, amp_hi=post.amp_hi,
                        amp_g_lo=post.amp_given_lo, amp_g_hi=post.amp_given_hi,
                        p_null=post.p_null, p_blink=post.p_blink, p_speech=post.p_speech,
                        set_probs={tuple(sorted(k)): float(v) for k, v in post.set_probs.items() if v > 1e-5},
                        score=r["cand"]["score"], apex=r["apex"], onset=r["onset"], offset=r["offset"],
                        template=r["template"], kind=r["cand"]["kind"]))
    return path, out


def run_final(paths, workers, params, cache) -> dict:
    cpath = os.path.join(cache, "final.pkl")
    if os.path.exists(cpath):
        with open(cpath, "rb") as fh:
            blob = pickle.load(fh)
        if blob["params"] == params and set(blob["finals"]) == set(paths):
            return blob["finals"]
    log("final posteriors at", params)
    finals = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for p, out in pool.imap_unordered(final_worker, [(p, *params) for p in paths]):
            finals[p] = out
    with open(cpath, "wb") as fh:
        pickle.dump(dict(params=params, finals=finals), fh)
    return finals


# --------------------------------------------------------------------------------------------------------------
# 3. spotting
def _in_windows(t, wins, lo=0.3, hi=0.4):
    return any(w["t_on"] - lo <= t <= w["t_off"] + hi for w in wins)


def detections(fin, T):
    return [dict(id=f["i"], status=f["status"], t_apex=f["apex"], t_onset=f["onset"], t_offset=f["offset"], aus=[], row=f["i"])
            for f in fin if f["score"] >= T and f["status"] in EXPRESSION_STATUSES]


def spotting_eval(clips, finals, T: float, subset=None) -> dict:
    tp_l = tp_s = n_truth = fa = 0
    secs = 0.0
    fa_src = dict(blink=0, speech=0, head=0, noise=0)
    apex_err = []
    for d in clips:
        if subset is not None and not subset(d):
            continue
        truth = d["truth"]["events"]
        m = match_events(truth, detections(finals[d["path"]], T))
        n_truth += len(truth)
        tp_l += sum(1 for x in m["matches"] if x["det"] is not None)
        tp_s += sum(1 for x in m["matches"] if x.get("strict"))
        apex_err += [abs(x["apex_err"]) for x in m["matches"] if x["det"] is not None]
        secs += d["spec"]["duration"]
        for f in m["false_alarms"]:
            fa += 1
            t = f["t_apex"]
            if _in_windows(t, d["truth"]["blinks"]):
                fa_src["blink"] += 1
            elif _in_windows(t, d["truth"]["speech"], 0.2, 0.2):
                fa_src["speech"] += 1
            elif _in_windows(t, [dict(t_on=g["t"], t_off=g["t"]) for g in d["truth"]["gaze"]], 0.3, 0.3):
                fa_src["head"] += 1       # gaze darts move the skin around the eyes
            else:
                fa_src["noise"] += 1
    p = tp_s / max(tp_s + fa, 1)
    r_s = tp_s / max(n_truth, 1)
    return dict(T=T, n_truth=n_truth, tp_loose=tp_l, tp_strict=tp_s, fp=fa, seconds=secs, recall_loose=tp_l / max(n_truth, 1),
                recall_strict=r_s, precision=p, f1=2 * p * r_s / max(p + r_s, 1e-9), fa_per_10s=10 * fa / max(secs, 1e-9),
                fa_sources=fa_src, apex_err_median=float(np.median(apex_err)) if apex_err else None)


# --------------------------------------------------------------------------------------------------------------
# 4. sensitivity: detection against true size, and the clip-level "smallest detectable movement"
def fit_logistic(x, y, slope_bounds=(0.3, 40.0), start=None, ceiling=False, c_bounds=(0.3, 1.0)):
    """P(detected) = c * expit(a + b x) with x = log size and c = 1 unless ``ceiling``.  Bounded slope (a perfectly separable
    sample would send it to infinity), light ridge; three starting slopes unless ``start`` is given (bootstrap refits).
    Returns (a, b), or (a, b, c) with ``ceiling``: the largest movements are not found every time either (a blink, speech or a
    head movement can hide them), and a curve forced to reach 1 would flatten its slope to make room for that."""
    x, y = np.asarray(x, float), np.asarray(y, float)

    def nll(p):
        z = np.clip(p[0] + p[1] * x, -40.0, 40.0)
        if not ceiling:
            return float(np.sum(np.logaddexp(0.0, z) - y * z) + 1e-3 * p[1] ** 2)
        q = np.clip(p[2] / (1.0 + np.exp(-z)), 1e-9, 1 - 1e-9)
        return float(-np.sum(y * np.log(q) + (1 - y) * np.log1p(-q)) + 1e-3 * p[1] ** 2)
    x0 = float(np.median(x))
    if start is not None:
        starts = [list(start)]
    elif ceiling:
        starts = [[-b0 * x0, b0, c0] for b0 in (1.0, 3.0, 8.0) for c0 in (0.7, 0.95)]
    else:
        starts = [[-b0 * x0, b0] for b0 in (1.0, 3.0, 8.0)]
    bounds = [(-400.0, 400.0), slope_bounds] + ([c_bounds] if ceiling else [])
    best = None
    for s0 in starts:
        r = minimize(nll, s0, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    return tuple(float(v) for v in best.x)


def at_prob(a, b, p, c=1.0):
    """Size (not log size) at which the fitted detection probability is ``p`` (``p`` is a fraction of the ceiling ``c``)."""
    q = min(max(p / c, 1e-9), 1 - 1e-9)
    return float(np.exp(np.clip((np.log(q / (1 - q)) - a) / b, -20.0, 20.0)))


def curve_pts(a, b, lo, hi, n=40, c=1.0):
    xs = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    return [dict(x=float(x), p=float(c / (1 + np.exp(-np.clip(a + b * np.log(x), -40, 40))))) for x in xs]


def truth_records(clips, finals, T) -> list[dict]:
    """One record per scripted movement: true size in engine units, whether it was found, the clip's predicted
    detectability (``snr`` = predicted z of the true pattern / threshold constant), grade."""
    out = []
    for d in clips:
        truth = d["truth"]["events"]
        ex = exact_of(d)
        m = match_events(truth, detections(finals[d["path"]], T))
        q = d["quality"]
        Ap, _, _ = whiten_templates(d["sigma_pair"], d["cent"], d["centre"])
        for k, (x, e) in enumerate(zip(m["matches"], truth)):
            aus = sorted(int(a) for a in e["aus"])
            rec = dict(clip=d["spec"]["idx"], camera=d["spec"]["camera"], split=d["split"], aus=aus, n_au=len(aus),
                       kind=e.get("kind", "micro"), detected=x["det"] is not None, strict=bool(x.get("strict")),
                       grade=q["grade"], mda_med=float(q["mda_median_pct"]), nuisance=d["spec"]["nuisance"],
                       amp_gen=100 * float(e["amp_iod"]), mda50=float(min(_mda(q, a) for a in aus)))
            xe = ex[k]
            if xe is not None:
                v = np.zeros(Ap.shape[1])
                for a, px in xe["a_true_px"].items():
                    v += float(px) * Ap[AUS.index(int(a))]
                rec.update(size=float(xe["size_pct"]), a_true={int(a): float(v_) for a, v_ in xe["a_true_pct"].items()},
                           pair_gain=float(xe["pair_gain"]), fit_r2=float(xe["fit_r2"]),
                           snr=float(np.linalg.norm(v) / QUAL.Z_CRIT))
            out.append(rec)
    return out


def binned(recs, key, lo, hi, n=8):
    edges = np.exp(np.linspace(np.log(lo), np.log(hi), n + 1))
    rows = []
    for i in range(n):
        sel = [r for r in recs if edges[i] <= r[key] < edges[i + 1]]
        k = sum(r["detected"] for r in sel)
        p, a, b = UNC.wilson(k, len(sel)) if sel else (float("nan"),) * 3
        rows.append(dict(lo=float(edges[i]), hi=float(edges[i + 1]), n=len(sel), detected=int(k), rate=p, ci_lo=a, ci_hi=b))
    return rows


BOOT = 300


def logistic_summary(sel, key, boot=None, seed=0, lo=None, hi=None, ceiling=False):
    """Detection probability against ``key`` (log-logistic): the 50 % and 90 % points with clip-level bootstrap intervals.
    With ``ceiling`` the curve tops out at its own maximum ``c`` and the points are where it reaches c/2 and 0.9 c."""
    rng = np.random.default_rng(seed)
    boot = BOOT if boot is None else boot
    x = np.log([r[key] for r in sel])
    y = [float(r["detected"]) for r in sel]
    fit = fit_logistic(x, y, ceiling=ceiling)
    a, b = fit[0], fit[1]
    c = fit[2] if ceiling else 1.0
    by_clip: dict = {}
    for r in sel:
        by_clip.setdefault(r["clip"], []).append(r)
    clips = sorted(by_clip)
    a50s, a90s, cs = [], [], []
    for _ in range(boot):
        pick = rng.choice(clips, len(clips))
        rs = [r for c_ in pick for r in by_clip[c_]]
        if len({r["detected"] for r in rs}) < 2:
            continue
        f = fit_logistic(np.log([r[key] for r in rs]), [float(r["detected"]) for r in rs], start=fit, ceiling=ceiling)
        cc = f[2] if ceiling else 1.0
        a50s.append(at_prob(f[0], f[1], 0.5 * cc, cc))
        a90s.append(at_prob(f[0], f[1], 0.9 * cc, cc))
        cs.append(cc)
    lo = lo or float(np.exp(x.min()))
    hi = hi or float(np.exp(x.max()))
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) > 20 else None      # noqa: E731
    out = dict(n=len(sel), detected=int(sum(y)), a50=at_prob(a, b, 0.5 * c, c), a90=at_prob(a, b, 0.9 * c, c), slope=b,
               a50_ci=q(a50s), a90_ci=q(a90s), curve=curve_pts(a, b, lo, hi, c=c), bins=binned(sel, key, lo, hi), _fit=fit)
    if ceiling:
        out.update(ceiling=c, ceiling_ci=q(cs))
    return out


def envelope(recs) -> dict:
    """Detection against the true size of the movement (largest regional displacement, % of the eye distance)."""
    out = {}
    pool = [r for r in recs if "size" in r]
    if not pool:
        return out
    lo, hi = float(np.quantile([r["size"] for r in pool], 0.01)) * 0.9, float(np.quantile([r["size"] for r in pool], 0.99)) * 1.1
    for cam in CAMERAS + ["all"]:
        sel = [r for r in pool if cam == "all" or r["camera"] == cam]
        if len(sel) < 8 or len({r["detected"] for r in sel}) < 2:
            continue
        s = logistic_summary(sel, "size", lo=lo, hi=hi, ceiling=True)
        s.pop("_fit")
        out[cam] = s
    return out


def fidelity(recs) -> dict:
    """How faithfully the chain measures the true movement: ``pair_gain`` = measured / true projection on the AU
    templates (1 = perfect), ``fit_r2`` = share of the true pattern the AU templates can explain at all."""
    def q(v):
        v = np.asarray(v, float)
        return dict(q10=float(np.quantile(v, 0.1)), q25=float(np.quantile(v, 0.25)), q50=float(np.quantile(v, 0.5)),
                    q75=float(np.quantile(v, 0.75)), q90=float(np.quantile(v, 0.9)))
    out = {}
    pool = [r for r in recs if "pair_gain" in r and np.isfinite(r["pair_gain"])]
    for cam in CAMERAS + ["all"]:
        sel = [r for r in pool if cam == "all" or r["camera"] == cam]
        if len(sel) >= 6:
            out[cam] = dict(n=len(sel), pair_gain=q([r["pair_gain"] for r in sel]), fit_r2=q([r["fit_r2"] for r in sel]),
                            size=q([r["size"] for r in sel]))
    for name, sel in (("single_au", [r for r in pool if r["n_au"] == 1]), ("multi_au", [r for r in pool if r["n_au"] > 1]),
                      ("found", [r for r in pool if r["detected"]]), ("missed", [r for r in pool if not r["detected"]])):
        if len(sel) >= 6:
            out[name] = dict(n=len(sel), pair_gain=q([r["pair_gain"] for r in sel]), fit_r2=q([r["fit_r2"] for r in sel]),
                             size=q([r["size"] for r in sel]))
    return out


def mda_check(recs) -> dict:
    """The clip's own "smallest detectable movement" (50 % point, predicted from its noise) against what was detected."""
    singles = [r for r in recs if "snr" in r and r["n_au"] == 1 and r["snr"] > 0]
    allr = [r for r in recs if "snr" in r and r["snr"] > 0]
    out = dict(n_single=len(singles), n_all=len(allr))
    if len(singles) < 20 or len({r["detected"] for r in singles}) < 2:
        return out
    s = logistic_summary(singles, "snr", lo=0.15, hi=8.0)
    a, b = s.pop("_fit")
    out.update(single_au_50pct=s["a50"], single_au_90pct=s["a90"], slope=s["slope"], a50_ci=s["a50_ci"], a90_ci=s["a90_ci"],
               curve=s["curve"], bins=s["bins"], fit=[float(a), float(b)])
    if len({r["detected"] for r in allr}) > 1:
        a2, b2 = fit_logistic(np.log([r["snr"] for r in allr]), [float(r["detected"]) for r in allr])
        out["all_events_50pct"] = at_prob(a2, b2, 0.5)
    by_cam = {}
    for cam in CAMERAS:
        sel = [r for r in singles if r["camera"] == cam]
        if len(sel) >= 15 and len({r["detected"] for r in sel}) > 1:
            ac, bc = fit_logistic(np.log([r["snr"] for r in sel]), [float(r["detected"]) for r in sel])
            by_cam[cam] = dict(n=len(sel), a50=at_prob(ac, bc, 0.5), a90=at_prob(ac, bc, 0.9))
    out["by_camera"] = by_cam
    return out


def slim_mda(mc: dict) -> dict:
    """The part of an ``mda_check`` the Evidence station needs when it is shown next to the pooled fit."""
    return {k: mc.get(k) for k in ("n_single", "n_all", "single_au_50pct", "single_au_90pct", "a50_ci", "a90_ci", "bins")}


def mda_transfer(fit, recs) -> dict:
    """The detection curve fitted on one group of faces, applied to the single-action-unit movements of the other: the share the
    curve predicts would be found against the share that was, and how many of the movements it calls "nine in ten" sizes were found."""
    sel = [r for r in recs if "snr" in r and r["n_au"] == 1 and r["snr"] > 0]
    if not fit or len(sel) < 10:
        return {}
    a, b = fit
    p = 1.0 / (1.0 + np.exp(-np.clip(a + b * np.log([r["snr"] for r in sel]), -40, 40)))
    s90 = at_prob(a, b, 0.9)
    big = [r for r in sel if r["snr"] >= s90]
    k = int(sum(r["detected"] for r in big))
    lo, hi = (UNC.wilson(k, len(big))[1:]) if big else (None, None)
    return dict(n=len(sel), predicted=float(p.mean()), observed=float(np.mean([float(r["detected"]) for r in sel])), at_90_point=float(s90),
                n_at_or_above=len(big), found_at_or_above=k, found_ci_lo=lo, found_ci_hi=hi)


def regrade(rec, scale) -> str:
    """Grade of a clip once its predicted smallest movement is multiplied by the calibrated ``mda_scale``."""
    old_i = QUAL.GRADES.index(rec["grade"])
    base_old = int(np.searchsorted(QUAL.MDA_EDGES, rec["mda_med"]))
    pen = max(0, old_i - base_old)
    return QUAL.GRADES[min(len(QUAL.GRADES) - 1, int(np.searchsorted(QUAL.MDA_EDGES, rec["mda_med"] * scale)) + pen)]


# --------------------------------------------------------------------------------------------------------------
# 5. amplitudes
def amplitude_rows(clips, finals, T) -> list[dict]:
    """One row per correctly named AU of a found movement: engine-unit truth against the read-out and its interval (% IOD)."""
    rows = []
    for d in clips:
        truth = d["truth"]["events"]
        ex = exact_of(d)
        m = match_events(truth, detections(finals[d["path"]], T))
        iod = d["iod_canon"]
        for k, x in enumerate(m["matches"]):
            if x["det"] is None or ex[k] is None:
                continue
            f = finals[d["path"]][m["detections"][x["det"]]["row"]]
            for a, tru in ex[k]["a_true_pct"].items():
                a = int(a)
                j = AUS.index(a)
                if tru < AMP_FLOOR_PCT or f["pip"][j] < 0.5 or f["amp_given"][j] <= 0:
                    continue
                f100 = 100.0 / iod
                rows.append(dict(camera=d["spec"]["camera"], split=d["split"], au=a, n_au=len(ex[k]["aus"]), true=float(tru),
                                 est=float(f["amp_given"][j] * f100), lo=float(f["amp_g_lo"][j] * f100), hi=float(f["amp_g_hi"][j] * f100),
                                 gain=float(ex[k]["pair_gain"])))
    return rows


def interval_k(rows, alpha=AMP_ALPHA) -> float:
    """Split-conformal widening factor for the amplitude intervals: the (1-alpha) quantile of the distance from the
    truth to the estimate in units of the reported half-widths (lower or upper, whichever side the truth is on)."""
    s = []
    for r in rows:
        w = (r["hi"] - r["est"]) if r["true"] >= r["est"] else (r["est"] - r["lo"])
        if w > 1e-6:
            s.append(abs(r["true"] - r["est"]) / w)
    if len(s) < 20:
        return 1.0
    n = len(s)
    kk = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(s)[min(kk, n) - 1])


def apply_k(r, k):
    return r["est"] - k * (r["est"] - r["lo"]), r["est"] + k * (r["hi"] - r["est"])


def coverage(rows, k=1.0):
    if not rows:
        return None
    hits = []
    for r in rows:
        lo, hi = apply_k(r, k)
        hits.append(lo <= r["true"] <= hi)
    return float(np.mean(hits))


def amplitude_summary(rows, k) -> dict:
    cal = [r for r in rows if r["split"] == "cal"]
    test = [r for r in rows if r["split"] == "test"]
    ratio = np.array([r["est"] / r["true"] for r in rows]) if rows else np.array([])
    out = dict(n=len(rows), n_cal=len(cal), n_test=len(test), floor_pct=AMP_FLOOR_PCT, k=k,
               coverage_90=coverage(rows), coverage_90_test=coverage(test), coverage_90_cal=coverage(cal),
               coverage_90_widened=coverage(rows, k), coverage_90_widened_test=coverage(test, k),
               coverage_90_widened_cal=coverage(cal, k))
    if len(ratio):
        out.update(ratio_median=float(np.median(ratio)), ratio_q25=float(np.quantile(ratio, 0.25)), ratio_q75=float(np.quantile(ratio, 0.75)))
    by_cam = {}
    for cam in CAMERAS:
        sel = [r for r in rows if r["camera"] == cam]
        if len(sel) >= 10:
            by_cam[cam] = dict(n=len(sel), coverage_90_widened=coverage(sel, k), ratio_median=float(np.median([r["est"] / r["true"] for r in sel])))
    out["by_camera"] = by_cam
    single = [r for r in rows if r["n_au"] == 1]
    multi = [r for r in rows if r["n_au"] > 1]
    out["single_au"] = dict(n=len(single), coverage_90_widened=coverage(single, k), ratio_median=float(np.median([r["est"] / r["true"] for r in single])) if single else None)
    out["multi_au"] = dict(n=len(multi), coverage_90_widened=coverage(multi, k), ratio_median=float(np.median([r["est"] / r["true"] for r in multi])) if multi else None)
    # a subsample of the points for a scatter on the Evidence page
    rng = np.random.default_rng(0)
    pick = rng.permutation(len(rows))[:260]
    out["points"] = [dict(true=round(rows[i]["true"], 3), est=round(rows[i]["est"], 3), lo=round(apply_k(rows[i], k)[0], 3),
                          hi=round(apply_k(rows[i], k)[1], 3), cam=rows[i]["camera"], split=rows[i]["split"]) for i in pick]
    return out


# --------------------------------------------------------------------------------------------------------------
# 6. categories
def scheme_probs(scheme, set_probs, p_art, p_null):
    res = CAT.category_posterior({frozenset(k): v for k, v in set_probs.items()}, p_art, scheme, p_null)
    return np.concatenate([res.probs, [res.unclear]]), res


def truth_label(scheme, aus):
    res = CAT.category_posterior({frozenset(aus): 1.0}, 0.0, scheme, 0.0)
    v = np.concatenate([res.probs, [res.unclear]])
    return int(np.argmax(v)), res.labels + ["unclear"]


def category_eval(clips, finals, T, alpha=0.1) -> dict:
    out = {}
    recs = {s: dict(cal=[], test=[]) for s in CAT.SCHEMES}
    for d in clips:
        fin = finals[d["path"]]
        truth = d["truth"]["events"]
        m = match_events(truth, detections(fin, T))
        for x in m["matches"]:
            if x["det"] is None:
                continue
            f = fin[m["detections"][x["det"]]["row"]]
            aus = sorted(int(a) for a in truth[x["truth"]]["aus"])
            art = 1.0 - (1.0 - f["p_blink"]) * (1.0 - f["p_speech"])
            for name, fn in CAT.SCHEMES.items():
                sch = fn()
                v, res = scheme_probs(sch, f["set_probs"], art, f["p_null"])
                y, labels = truth_label(sch, aus)
                recs[name][d["split"]].append((v, y, res.abstain, res.top_p))
    for name in CAT.SCHEMES:
        cal, test = recs[name]["cal"], recs[name]["test"]
        if len(cal) < 10 or len(test) < 5:
            continue
        Vc, yc = np.array([r[0] for r in cal]), np.array([r[1] for r in cal])
        Vt, yt = np.array([r[0] for r in test]), np.array([r[1] for r in test])
        conf = UNC.LACConformal(alpha=alpha).fit(Vc, yc)
        cov, size = conf.coverage(Vt, yt)
        pred = Vt.argmax(1)
        ab = np.array([r[2] for r in test])
        acc_all = float((pred == yt).mean())
        acc_sel = float((pred[~ab] == yt[~ab]).mean()) if (~ab).any() else float("nan")
        out[name] = dict(n_cal=len(cal), n_test=len(test), qhat=float(conf.qhat), alpha=alpha, coverage_test=float(cov), mean_set_size=float(size),
                         accuracy_all=acc_all, accuracy_when_committing=acc_sel, commit_rate=float((~ab).mean()),
                         majority_baseline=float(np.bincount(yt).max() / len(yt)),
                         labels=CAT.SCHEMES[name]().labels + ["unclear"])
    return out


# --------------------------------------------------------------------------------------------------------------
T_FLOOR = 4.5                    # never operate below this (the null distribution of the scan statistic has a heavy tail there)


def choose_threshold(table, budget):
    """Spotting threshold: the one with the best spotting F1 on the *calibration* split among the thresholds that stay within the
    false-alarm budget (alarms per 10 s of footage).  Returns (threshold, budget_met).  When no tested threshold stays within the
    budget the least trigger-happy one is returned and ``budget_met`` is False: the caller must say so instead of reporting the budget
    as if it had been honoured."""
    cand = [r for r in table if r["T"] >= T_FLOOR]
    ok = [r for r in cand if r["cal"]["fa_per_10s"] <= budget]
    if not ok:
        return max(r["T"] for r in cand), False
    best = max(ok, key=lambda r: (round(r["cal"]["f1"], 6), r["T"]))
    return best["T"], True


def tune(paths, clips, workers, cache, t_op, target_fa):
    """Engine parameters at the operating threshold ``t_op``, then the spotting threshold that parameter set implies."""
    grid = run_grid(paths, workers, cache, t_op)
    best = min((g for g in grid if "cal" in grid[g]), key=lambda g: grid[g]["cal"]["log_loss"])
    eta, temper, pi = best
    log("chosen", dict(eta=eta, temper=temper, size_prior=PRIORS[pi]), "cal", {k: round(grid[best]["cal"][k], 4) for k in ("log_loss", "ece", "f1", "exact_set")},
        "test", {k: round(grid[best]["test"][k], 4) for k in ("log_loss", "ece", "f1", "exact_set")})
    finals = run_final(paths, workers, (eta, temper, pi), cache)
    cal_sub = lambda d: d["split"] == "cal"          # noqa: E731
    test_sub = lambda d: d["split"] == "test"        # noqa: E731
    table = [dict(T=T, cal=spotting_eval(clips, finals, T, cal_sub), test=spotting_eval(clips, finals, T, test_sub),
                  all=spotting_eval(clips, finals, T)) for T in THRESHOLDS]
    T_star, met = choose_threshold(table, target_fa)
    return grid, best, finals, table, T_star, met


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--target-fa", type=float, default=0.5,
                    help="false-alarm budget: alarms per 10 s of footage allowed on the calibration split (0.5 = one per 20 s); the "
                         "threshold is the best-F1 one within it")
    ap.add_argument("--t-op", type=float, default=T_OP_DEFAULT, help="threshold at which the engine is first tuned")
    ap.add_argument("--quick", action="store_true", help="smoke test: tiny grid, few bootstrap draws (numbers are not the calibration)")
    a = ap.parse_args()
    if a.quick:
        global ETAS, TEMPERS, BOOT
        ETAS, TEMPERS, BOOT = [0.15, 0.4], [0.35, 1.0], 40
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(os.path.join(a.out, "evidence"), exist_ok=True)
    paths = load_paths(a.corpus)
    log(f"{len(paths)} clips")
    clips = [load(p) for p in paths]
    have_exact = sum(1 for c in clips if c.get("exact") is not None)
    if have_exact < len(clips):
        log(f"WARNING: {len(clips) - have_exact} clips carry no engine-unit truth; sensitivity and amplitude checks use the rest")

    # ---- 1-2. engine parameters and spotting threshold (tuned at the threshold it ends up with)
    t_op = a.t_op
    for rnd in range(3):
        t_used = t_op
        grid, best, finals, table, T_star, met = tune(paths, clips, a.workers, a.cache, t_used, a.target_fa)
        log(f"round {rnd}: tuned at T={t_used}, spotting threshold now {T_star}")
        if abs(T_star - t_used) < 0.4:
            break
        t_op = T_star
    eta, temper, pi = best
    star = next(r for r in table if r["T"] == T_star)
    log("spotting threshold chosen:", T_star, "cal FA/10s", round(star["cal"]["fa_per_10s"], 3), "test", round(star["test"]["fa_per_10s"], 3),
        f"(budget {a.target_fa}/10 s {'met' if met else 'NOT MET by any tested threshold'})")
    cal_sub = lambda d: d["split"] == "cal"          # noqa: E731
    test_sub = lambda d: d["split"] == "test"        # noqa: E731
    by_cam = {cam: spotting_eval(clips, finals, T_star, lambda d, c=cam: d["spec"]["camera"] == c) for cam in CAMERAS}
    by_nuis = {n: spotting_eval(clips, finals, T_star, lambda d, n_=n: d["spec"]["nuisance"] == n_) for n in ("still", "normal", "busy")}
    null_only = spotting_eval(clips, finals, T_star, lambda d: d["spec"]["null"])

    # ---- 3. sensitivity envelope and the clip-level smallest-detectable-movement
    recs = truth_records(clips, finals, T_star)
    env = envelope(recs)                                                  # descriptive: every clip, both identity groups
    # The bench rescales each recording's "smallest detectable movement" by one factor (two points of one curve).  Unlike everything
    # else fitted here, that factor is fitted on all six faces: it only rescales a quantity the report prints, and no held-out score on
    # the Evidence page depends on it.  It is then checked the way a held-out fit would be: refitted on each identity group alone, and
    # the curve fitted on A-C is applied to D-F.  The two groups disagree by more than their pooled fit suggests, and the Evidence
    # page says so.
    mc = mda_check(recs)
    recs_cal = [r for r in recs if r["split"] == "cal"]
    recs_test = [r for r in recs if r["split"] == "test"]
    mc_cal, mc_test = mda_check(recs_cal), mda_check(recs_test)
    transfer = mda_transfer(mc_cal.get("fit"), recs_test)
    mda_scale = float(mc.get("single_au_50pct", 1.0))
    mda90_scale = float(mc.get("single_au_90pct", 1.0))
    log(f"MDA check (all faces, n={mc.get('n_single')}): single-AU events are found half the time at {mda_scale:.2f}× the predicted "
        f"smallest movement, 90 % at {mda90_scale:.2f}")
    for name, m_ in (("A-C alone", mc_cal), ("D-F alone", mc_test)):
        log(f"  {name} (n={m_.get('n_single')}): 50 % at {m_.get('single_au_50pct', float('nan')):.2f} {m_.get('a50_ci')}, "
            f"90 % at {m_.get('single_au_90pct', float('nan')):.2f} {m_.get('a90_ci')}")
    log(f"  A-C curve applied to D-F: {transfer}")
    for r in recs:
        r["grade_cal"] = regrade(r, mda_scale)
    grade_rows = []
    for g in QUAL.GRADES:
        sel = [r for r in recs if r["grade_cal"] == g and r.get("size", r["amp_gen"]) >= 1.0]
        k = sum(r["detected"] for r in sel)
        p, lo, hi = UNC.wilson(k, len(sel)) if sel else (float("nan"),) * 3
        grade_rows.append(dict(grade=g, clips=len({r["clip"] for r in recs if r["grade_cal"] == g}), events_ge_1pct=len(sel), detected=int(k), rate=p,
                               ci_lo=lo, ci_hi=hi))
    fid = fidelity(recs)

    # ---- 4. AU engine at T_star on matched events, and by SNR / camera
    recs_by_clip: dict = {}
    for r in recs:
        recs_by_clip.setdefault(r["clip"], []).append(r)

    def matched(subset):
        pips, tru, meta = [], [], []
        for d in clips:
            if not subset(d):
                continue
            fin = finals[d["path"]]
            truth = d["truth"]["events"]
            m = match_events(truth, detections(fin, T_star))
            clip_recs = recs_by_clip.get(d["spec"]["idx"], [])        # truth order, as written by truth_records
            for x_ in m["matches"]:
                if x_["det"] is None:
                    continue
                f = fin[m["detections"][x_["det"]]["row"]]
                e = truth[x_["truth"]]
                aus = sorted(int(a_) for a_ in e["aus"])
                pips.append(f["pip"])
                tru.append(aus)
                meta.append(dict(camera=d["spec"]["camera"], snr=clip_recs[x_["truth"]].get("snr"), aus=aus))
        return pips, tru, meta

    pips_t, tru_t, meta_t = matched(test_sub)
    pips_c, tru_c, meta_c = matched(cal_sub)
    au_test, au_cal = au_scores(pips_t, tru_t), au_scores(pips_c, tru_c)
    allp, allt, allm = pips_t + pips_c, tru_t + tru_c, meta_t + meta_c
    by_snr = {}
    for name, lo, hi in (("below 1.5 × the smallest detectable", 0, 1.5), ("1.5 to 3 ×", 1.5, 3), ("above 3 ×", 3, 1e9)):
        sel = [i for i, m_ in enumerate(allm) if m_["snr"] is not None and lo <= m_["snr"] / mda_scale < hi]
        if len(sel) >= 4:
            sc = au_scores([allp[i] for i in sel], [allt[i] for i in sel])
            by_snr[name] = {k: sc[k] for k in ("n", "precision", "recall", "f1", "exact_set", "log_loss", "ece")}
    by_camera_au = {}
    for cam in CAMERAS:
        sel = [i for i, m_ in enumerate(allm) if m_["camera"] == cam]
        if len(sel) >= 5:
            sc = au_scores([allp[i] for i in sel], [allt[i] for i in sel])
            by_camera_au[cam] = {k: sc[k] for k in ("n", "precision", "recall", "f1", "exact_set")}

    # ---- 5. amplitudes: bias, interval coverage, conformal widening fitted on the calibration identities only
    arows = amplitude_rows(clips, finals, T_star)
    k_amp = interval_k([r for r in arows if r["split"] == "cal"])
    amp = amplitude_summary(arows, k_amp)
    log(f"amplitude read-outs: {amp['n']} named AUs, median est/true {amp.get('ratio_median', float('nan')):.2f}, "
        f"raw 90 % intervals cover {amp['coverage_90']}, widened x{k_amp:.2f} cover {amp['coverage_90_widened_test']} on the test identities")

    # ---- 6. categories
    cats = category_eval(clips, finals, T_star)

    # ---- 7. controls (test split)
    rng = np.random.default_rng(0)
    f1_shuf = []
    for _ in range(300):
        perm = rng.permutation(len(tru_t))
        f1_shuf.append(au_scores(pips_t, [tru_t[i] for i in perm])["f1"])
    freq = np.zeros(len(AUS))
    for t in tru_c:
        for a_ in t:
            freq[AUS.index(a_)] += 1
    top2 = np.argsort(-freq)[:2]
    prior_pips = [np.where(np.isin(np.arange(len(AUS)), top2), 1.0, 0.0) for _ in tru_t]
    f1_prior = au_scores(prior_pips, tru_t)["f1"]
    noise_named = []
    for d in clips:
        fin = finals[d["path"]]
        m = match_events(d["truth"]["events"], detections(fin, T_star))
        for fa in m["false_alarms"]:
            noise_named.append(int((fin[fa["row"]]["pip"] >= 0.5).sum()))
    controls = dict(
        shuffled_truth_f1=dict(mean=float(np.mean(f1_shuf)), q95=float(np.quantile(f1_shuf, 0.95)), observed=au_test["f1"]),
        prior_only_f1=float(f1_prior), prior_only_aus=[AUS[i] for i in top2],
        false_alarm_named_aus=dict(n_false_alarms=len(noise_named), mean_named=float(np.mean(noise_named)) if noise_named else None,
                                   share_with_any=float(np.mean([n > 0 for n in noise_named])) if noise_named else None))

    # ---- write
    grid_rows = [dict(eta=g[0], temper=g[1], size_prior=list(PRIORS[g[2]]),
                      cal={k: v[k] for k in ("log_loss", "brier", "ece", "f1", "exact_set", "n")} if (v := grid[g].get("cal")) else None,
                      test={k: v[k] for k in ("log_loss", "brier", "ece", "f1", "exact_set", "n")} if (v := grid[g].get("test")) else None) for g in sorted(grid)]
    default = (0.15, 1.0, 0)
    n_mov = sum(len(c["truth"]["events"]) for c in clips)
    summary = dict(
        corpus=f"{len(clips)} phantom clips, {n_mov} scripted movements, "
               f"{sum(1 for c in clips if c['spec']['null'])} movement-free clips; 6 face identities (A-C calibration, D-F test)",
        spot_threshold=T_star, recall_test=star["test"]["recall_strict"], recall_loose_test=star["test"]["recall_loose"], fa_per_10s_test=star["test"]["fa_per_10s"],
        precision_test=star["test"]["precision"], au_f1_test=au_test["f1"], au_exact_set_test=au_test["exact_set"], au_ece_test=au_test["ece"],
        au_ece_default_test=grid.get(default, {}).get("test", {}).get("ece"),
        mda_scale=mda_scale, mda_scale_cal=mc_cal.get("single_au_50pct"), mda_scale_test=mc_test.get("single_au_50pct"), amp_coverage_test=amp.get("coverage_90_widened_test"),
        amp_ratio_median=amp.get("ratio_median"),
        note="synthetic phantoms: instrument calibration, not accuracy on human faces")
    created = time.strftime("%Y-%m-%dT%H:%M:%S")
    calibration = dict(version=2, created=created, eta=eta, temper=temper, size_prior=list(PRIORS[pi]),
                       spot_threshold=T_star, t_op=t_used, mda_scale=float(mda_scale), mda90_scale=float(mda90_scale),
                       mda_edges=list(QUAL.MDA_EDGES), amp_interval_k=float(k_amp),
                       conformal={k: dict(alpha=v["alpha"], qhat=v["qhat"], n_cal=v["n_cal"]) for k, v in cats.items()},
                       provenance=dict(kind="synthetic phantom corpus", clips=len(clips), seed=1, split=dict(calibration="A-C", test="D-F")),
                       summary=summary)
    evidence = dict(
        version=2, created=created, summary=summary,
        engine=dict(chosen=dict(eta=eta, temper=temper, size_prior=list(PRIORS[pi])), default=dict(eta=0.15, temper=1.0, size_prior=list(PRIORS[0])),
                    grid=grid_rows, test=au_test, cal=au_cal, default_test=grid.get(default, {}).get("test"), amplitude_interval=amp,
                    by_snr=by_snr, by_camera=by_camera_au),
        spotting=dict(chosen=T_star, budget_fa_per_10s=a.target_fa, budget_met=bool(met), floor=T_FLOOR,
                      rule="the threshold with the best spotting F1 on the calibration split among those within the false-alarm budget",
                      table=table, by_camera=by_cam, by_nuisance=by_nuis, movement_free=null_only),
        sensitivity=dict(envelope=env, mda_check=mc, mda_split=dict(calibration=slim_mda(mc_cal), test=slim_mda(mc_test), transfer=transfer), grades=grade_rows, fidelity=fid,
                         x_axis="true size of the movement: its largest regional displacement, as % of the distance between the eyes"),
        categories=cats, controls=controls,
        scope=dict(
            fitted_on_calibration_faces="engine settings (eta, temper, size prior), spotting threshold, amplitude-interval widening, "
                                        "category conformal thresholds",
            fitted_on_all_faces="the two smallest-detectable-movement factors (sensitivity.mda_check), because they only rescale a reported quantity; "
                                "checked by refitting on each identity group (sensitivity.mda_split)",
            held_out_test_faces="summary.*_test, spotting.table[*].test, engine.test, engine.amplitude_interval (coverage), "
                                "categories.*.coverage_test, controls",
            all_faces_descriptive="spotting.by_camera, spotting.by_nuisance, spotting.movement_free, sensitivity.envelope, sensitivity.grades, "
                                  "sensitivity.fidelity, engine.by_snr, engine.by_camera: these describe the corpus as a whole and were not used to fit anything"),
        reproducibility=dict(corpus_seed=1, clips=len(clips), python=platform.python_version(), numpy=np.__version__,
                             command=f"python scripts/calibrate_engine.py --corpus <corpus> --out data --cache <cache> --workers {a.workers}"),
        caveat="Synthetic phantoms with scripted movements. These numbers calibrate and characterise the instrument; they say nothing about accuracy on human faces.")
    with open(os.path.join(a.out, "calibration.json"), "w") as fh:
        json.dump(clean(calibration), fh, indent=1, allow_nan=False)
    with open(os.path.join(a.out, "evidence", "phantom_eval.json"), "w") as fh:
        json.dump(clean(evidence), fh, separators=(",", ":"), allow_nan=False)
    log("wrote calibration.json and evidence/phantom_eval.json")
    log(json.dumps(clean(summary), indent=1))


if __name__ == "__main__":
    main()
