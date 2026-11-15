"""Action-unit inference: Bayesian model comparison over small AU sets, with nuisance motion in the model.

The measurement is the ROI displacement pattern ``y`` (27 ROIs x 2 components, canonical px) at the apex of a
spotted movement, with per-component noise ``sigma`` measured from the clip itself.  Every hypothesis is a
small set S of AUs (|S| <= 3) plus optional nuisance components (blink, speech); residual rigid motion and a
conjugate-gaze term are *always* in the model, so a head-motion error cannot masquerade as an AU.

    y = sum_{k in S} a_k t_k  +  b_blink n_blink + b_speech n_speech  +  rigid(c) + gaze g  +  eps

Amplitudes have half-normal priors (an AU only pulls one way), nuisance parameters Gaussian priors.  The
marginal likelihood of every hypothesis is closed-form (Gaussian linear model, Woodbury form), so ~1,200
hypotheses are scored in a few milliseconds.  The posterior over hypotheses gives posterior inclusion
probabilities, amplitudes with credible intervals, and the probability that the movement is a blink, speech
or "not AU-like" (the null).  A tempering exponent (fitted on the calibration faces, not the test faces) calibrates the probabilities.

``eta`` adds a multiplicative model-error term (the templates are ROI-level approximations of real
anatomy).  *Which* AUs are present is decided with sigma_eff^2 = sigma^2 + (eta * |y|)^2 per ROI, so large,
slightly-off patterns are not "explained" by inventing extra AUs.  *How big* each AU is comes from refitting the
plausible hypotheses with the model error proportional to the signal each one predicts, sigma_eff^2 = sigma^2 +
(eta * |s_h|)^2: weights that follow the observed magnitude down-weight exactly the regions that moved most and
read every amplitude about a third too small.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.special import ndtr

from ..vision import geometry as G
from . import templates as TPL

# FACS / EMFACS-style frequent co-occurrences get a mild prior boost (documented expert prior, not fitted)
COOCCUR = {(1, 2), (6, 12), (4, 7), (9, 10), (4, 5), (1, 4), (5, 7), (12, 14), (15, 17), (4, 9), (7, 12), (2, 5), (1, 5), (10, 12)}
ANTAGONIST = {(12, 15)}


@dataclass
class AUPosterior:
    aus: list                          # AU numbers, order of the arrays below
    pip: np.ndarray                    # (K,) posterior inclusion probability
    amp_mean: np.ndarray               # (K,) model-averaged amplitude, px (0 when absent)
    amp_lo: np.ndarray                 # (K,) 5% quantile
    amp_hi: np.ndarray                 # (K,) 95% quantile
    amp_given: np.ndarray              # (K,) mean amplitude given the AU is present
    p_null: float                      # P(no AU) = the pattern is noise or not AU-like
    p_blink: float
    p_speech: float
    hypotheses: list                   # top hypotheses: dict(aus, blink, speech, p, amp)
    log_bf_null: float                 # log Bayes factor of the best hypothesis against "no AU, no blink, no speech"
    fit: dict                          # arrays (R,2): data, aus, nuisance, residual of the MAP hypothesis
    chi2: float                        # whitened residual of the MAP hypothesis per dof
    rigid: np.ndarray                  # (5,) posterior-mean nuisance coefficients under the MAP hypothesis
    set_probs: dict = field(default_factory=dict)   # frozenset(AUs) -> probability (marginalised over nuisance)
    amp_given_lo: Optional[np.ndarray] = None       # (K,) 5% quantile of the amplitude *given the AU is present* (px)
    amp_given_hi: Optional[np.ndarray] = None       # (K,) 95% quantile of the amplitude given the AU is present (px)

    def top_aus(self, thr: float = 0.5):
        return [a for a, p in zip(self.aus, self.pip) if p >= thr]


class AUEngine:
    def __init__(self, centroids: np.ndarray, centre, iod_canon: float, max_active: int = 3, temper: float = 1.0,
                 eta: float = 0.15, tau_a_pct=(0.5, 1.5, 4.0), tau_nuis: float = 0.6, tau_events: float = 2.5,
                 size_prior=(0.12, 0.38, 0.32, 0.18)):
        self.iod = float(iod_canon)
        self.K = len(TPL.AUS)
        self.max_active = max_active
        self.temper = temper
        self.eta = eta
        self.tau_a = np.asarray(tau_a_pct) / 100.0 * self.iod
        self.tau_events = tau_events
        R = len(G.ROI_NAMES)
        self.R = R
        au = TPL.au_templates().reshape(self.K, -1)
        blink = TPL.blink_template().reshape(1, -1)
        speech = TPL.speech_template().reshape(1, -1)
        nuis = TPL.nuisance_columns(centroids, centre).reshape(-1, 2 * R)       # 6 affine + gaze
        self.cols = np.concatenate([au, blink, speech, nuis], axis=0)           # K + 2 + 7
        self.i_blink, self.i_speech = self.K, self.K + 1
        self.i_nuis = list(range(self.K + 2, self.K + 2 + nuis.shape[0]))
        self.tau_nuis = tau_nuis
        # hypothesis lattice
        self.subsets = [()]
        for m in range(1, max_active + 1):
            self.subsets += list(itertools.combinations(range(self.K), m))
        sizes = np.array([len(s) for s in self.subsets])
        sp = np.asarray(size_prior[: max_active + 1], float)
        sp = sp / sp.sum()
        counts = np.bincount(sizes, minlength=max_active + 1)
        prior = np.array([sp[m] / counts[m] for m in sizes])
        for i, s in enumerate(self.subsets):
            aus = [TPL.AUS[j] for j in s]
            for a, b in itertools.combinations(aus, 2):
                key = (min(a, b), max(a, b))
                if key in COOCCUR:
                    prior[i] *= 2.0
                if key in ANTAGONIST:
                    prior[i] *= 0.3
        self.log_prior_sets = np.log(prior / prior.sum())
        self.groups = {}
        for i, s in enumerate(self.subsets):
            self.groups.setdefault(len(s), []).append(i)

    # ------------------------------------------------------------------------------------------------
    def fit_hypothesis(self, aus, y: np.ndarray, sigma: np.ndarray, mask: Optional[np.ndarray] = None,
                       blink: bool = False, speech: bool = False, tau_idx: int = 1,
                       signal: Optional[np.ndarray] = None) -> dict:
        """Posterior-mean fit of one explicit hypothesis (AU numbers + optional blink/speech) with the always-on
        nuisance terms.  Used by the explanation layer (contrasts, deletion tests).

        The model-error term scales with the *signal this hypothesis predicts* (``signal``, (R, 2) px; when not given,
        the hypothesis' own fit at the measured noise), never with the raw observation: weights that follow the
        observed magnitude down-weight exactly the regions that moved most and bias every amplitude low."""
        y = np.asarray(y, float)
        sigma = np.asarray(sigma, float)
        if self.eta > 0:
            if signal is None:
                f0 = self._fit(aus, y, sigma, mask, blink, speech, tau_idx)
                signal = f0["au"] + f0["events"]
            sig = self._sigma_model(sigma, signal)
        else:
            sig = sigma
        return self._fit(aus, y, sig, mask, blink, speech, tau_idx)

    def _fit(self, aus, y: np.ndarray, sig: np.ndarray, mask, blink: bool, speech: bool, tau_idx: int) -> dict:
        R = self.R
        w = 1.0 / sig.reshape(-1)
        if mask is not None:
            w = w * np.repeat(np.asarray(mask, float), 2)
        ks = [TPL.AUS.index(a) for a in aus]
        ev = ([self.i_blink] if blink else []) + ([self.i_speech] if speech else [])
        cols = ks + ev + self.i_nuis
        V = self.cols[cols] * w[None, :]
        yw = y.reshape(-1) * w
        tv = np.concatenate([np.full(len(ks), self.tau_a[tau_idx] ** 2), np.full(len(ev), self.tau_events ** 2),
                             np.full(len(self.i_nuis), self.tau_nuis ** 2)])
        A = V @ V.T + np.diag(1.0 / tv)
        mu = np.linalg.solve(A, V @ yw)
        sd = np.sqrt(np.diag(np.linalg.inv(A)))
        npos = len(ks) + len(ev)
        mu[:npos] = np.maximum(mu[:npos], 0.0)
        coef = np.zeros(self.cols.shape[0])
        coef[cols] = mu
        fit = self.cols.T @ coef
        au = self.cols[ks].T @ coef[ks] if ks else np.zeros(2 * R)
        evp = self.cols[ev].T @ coef[ev] if ev else np.zeros(2 * R)
        nz = self.cols[ev + self.i_nuis].T @ coef[ev + self.i_nuis]
        resid = y.reshape(-1) - fit
        chi2 = float(((resid * w) ** 2).sum())
        return dict(fit=fit.reshape(R, 2), au=au.reshape(R, 2), events=evp.reshape(R, 2), nuis=nz.reshape(R, 2),
                    resid=resid.reshape(R, 2), coef=coef, sd=sd, chi2=chi2, weights=(w.reshape(R, 2)))

    # ------------------------------------------------------------------------------------------------
    def _sigma_obs(self, y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Noise for deciding *which* AUs are present: model error proportional to the observed magnitude.  One set
        of weights for every hypothesis (so the Gram matrix is computed once) and robust to template mismatch anywhere
        the face moved, but biased for amplitudes - those come from refits with ``_sigma_model``."""
        return np.sqrt(sigma ** 2 + (self.eta * np.abs(y)) ** 2)

    def _sigma_model(self, sigma: np.ndarray, signal: np.ndarray) -> np.ndarray:
        return np.sqrt(sigma ** 2 + (self.eta * np.abs(signal)) ** 2)

    def infer(self, y: np.ndarray, sigma: np.ndarray, mask: Optional[np.ndarray] = None, p_blink: float = 0.05,
              p_speech: float = 0.05, n_samples: int = 600, rng: Optional[np.random.Generator] = None,
              top: int = 8) -> AUPosterior:
        """y, sigma: (R, 2) px.  mask: (R,) 1 = ROI usable.  p_blink / p_speech: prior plausibility from landmarks.

        Two kinds of weights.  *Which AUs* are present is decided with model error proportional to the observed
        magnitude (a robust likelihood: a large, slightly-off pattern is not "explained" by inventing AUs).  *How big*
        each AU is comes from refitting the plausible hypotheses with the model error proportional to the signal each
        hypothesis predicts: weights that follow the observation down-weight the regions that moved most and read every
        amplitude about a third too small."""
        R, K = self.R, self.K
        y = np.asarray(y, float)
        sigma = np.asarray(sigma, float)
        sig = self._sigma_obs(y, sigma) if self.eta > 0 else sigma
        w = 1.0 / sig.reshape(-1)
        if mask is not None:
            w = w * np.repeat(np.asarray(mask, float), 2)
        yw = y.reshape(-1) * w
        Vw = self.cols * w[None, :]
        G_all = Vw @ Vw.T
        b_all = Vw @ yw
        yy = float(yw @ yw)
        dof = max(1.0, float((w > 0).sum()) - len(self.i_nuis))
        rng = rng or np.random.default_rng(0)

        nuis_combos = [(), (self.i_blink,), (self.i_speech,), (self.i_blink, self.i_speech)]
        log_pn = []
        for c in nuis_combos:
            lp = 0.0
            lp += np.log(p_blink if self.i_blink in c else 1 - p_blink)
            lp += np.log(p_speech if self.i_speech in c else 1 - p_speech)
            log_pn.append(lp)

        recs = []            # (log_post, set_index, nuis_index, mean(d), sd(d), cols(d))
        logs = []
        stored = []
        for nci, nc in enumerate(nuis_combos):
            for m, idxs in self.groups.items():
                S = np.array([self.subsets[i] for i in idxs], dtype=int).reshape(len(idxs), m)
                cols = np.concatenate([S, np.tile(np.array(list(nc) + self.i_nuis, dtype=int), (len(idxs), 1))], axis=1)
                d = cols.shape[1]
                Gs = G_all[cols[:, :, None], cols[:, None, :]]
                bs = b_all[cols]
                # per mixture component of the AU amplitude prior
                comp_logs = []
                comp_means = []
                comp_sds = []
                for tau in self.tau_a:
                    tv = np.concatenate([np.full(m, tau ** 2), np.full(len(nc), self.tau_events ** 2),
                                         np.full(len(self.i_nuis), self.tau_nuis ** 2)])
                    A = Gs + np.diag(1.0 / tv)[None]
                    L = np.linalg.cholesky(A)
                    mu = np.linalg.solve(A, bs[..., None])[..., 0]
                    logdetA = 2.0 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)
                    lml = -0.5 * (logdetA + np.log(tv).sum() - (bs * mu).sum(1))
                    Ainv = np.linalg.inv(A)
                    sd = np.sqrt(np.maximum(np.diagonal(Ainv, axis1=1, axis2=2), 1e-12))
                    # positivity of the AU and event amplitudes (half-normal priors)
                    npos = m + len(nc)
                    if npos:
                        z = mu[:, :npos] / sd[:, :npos]
                        lml = lml + npos * np.log(2.0) + np.log(np.maximum(ndtr(z), 1e-300)).sum(1)
                    comp_logs.append(lml)
                    comp_means.append(mu)
                    comp_sds.append(sd)
                comp_logs = np.stack(comp_logs, 0)                       # (T, nS)
                mx = comp_logs.max(0)
                lml = mx + np.log(np.exp(comp_logs - mx).mean(0))        # equal-weight scale mixture
                # keep component means/sds of the best component for amplitude summaries
                best = comp_logs.argmax(0)
                mus = np.stack(comp_means, 0)[best, np.arange(len(idxs))]
                sds = np.stack(comp_sds, 0)[best, np.arange(len(idxs))]
                post = self.temper * lml + self.log_prior_sets[idxs] + log_pn[nci]
                logs.append(post)
                stored.append((nci, idxs, cols, mus, sds, lml))
        allp = np.concatenate(logs)
        allp -= allp.max()
        pr = np.exp(allp)
        pr /= pr.sum()

        # --- marginals
        pip = np.zeros(K)
        p_null = 0.0
        p_bl = 0.0
        p_sp = 0.0
        set_probs: dict = {}
        offs = 0
        flat = []                               # (prob, nci, set_idx, mus, sds, cols)
        for (nci, idxs, cols, mus, sds, lml), lg in zip(stored, logs):
            p_here = pr[offs: offs + len(idxs)]
            offs += len(idxs)
            for j, i in enumerate(idxs):
                p = p_here[j]
                s = self.subsets[i]
                for k in s:
                    pip[k] += p
                if len(s) == 0:
                    p_null += p
                if self.i_blink in nuis_combos[nci]:
                    p_bl += p
                if self.i_speech in nuis_combos[nci]:
                    p_sp += p
                key = frozenset(TPL.AUS[k] for k in s)
                set_probs[key] = set_probs.get(key, 0.0) + p
                flat.append((p, nci, i, mus[j], sds[j], cols[j]))
        flat.sort(key=lambda r: -r[0])
        probs = np.array([f[0] for f in flat])

        # --- amplitudes: refit the plausible hypotheses without the observation-dependent weights, then sample the mixture
        cum_all = np.cumsum(probs) / probs.sum()
        n_h = int(min(len(flat), max(10, int(np.searchsorted(cum_all, 0.995)) + 1), 60))
        refit = []
        for (p, nci, i, mu, sd, cols) in flat[:n_h]:
            s_ = self.subsets[i]
            f = self.fit_hypothesis([TPL.AUS[k] for k in s_], y, sigma, mask, blink=self.i_blink in nuis_combos[nci],
                                    speech=self.i_speech in nuis_combos[nci])
            refit.append(f)
        amp_mean = np.zeros(K)
        draws = np.zeros((max(n_samples, 0), K))
        if n_samples > 0:
            pr_h = probs[:n_h] / probs[:n_h].sum()
            picks = rng.choice(n_h, size=n_samples, p=pr_h)
            for si, h in enumerate(picks):
                s_ = self.subsets[flat[h][2]]
                if not s_:
                    continue
                f = refit[h]
                a = np.maximum(f["coef"][list(s_)] + rng.standard_normal(len(s_)) * f["sd"][: len(s_)], 0.0)
                for k, v in zip(s_, a):
                    draws[si, k] = v
            amp_mean = draws.mean(0)
            amp_lo, amp_hi = np.quantile(draws, 0.05, axis=0), np.quantile(draws, 0.95, axis=0)
        else:
            amp_lo, amp_hi = np.zeros(K), np.zeros(K)
        pres = (draws > 0).sum(0)
        amp_given = np.where(pres > 0, draws.sum(0) / np.maximum(pres, 1), 0.0)
        g_lo, g_hi = np.zeros(K), np.zeros(K)
        for k in range(K):
            pos = draws[:, k][draws[:, k] > 0]
            if len(pos) >= 8:
                g_lo[k], g_hi[k] = np.quantile(pos, 0.05), np.quantile(pos, 0.95)
            elif len(pos):                                   # too few draws to quote quantiles: a wide, honest default
                g_lo[k], g_hi[k] = 0.5 * amp_given[k], 1.8 * amp_given[k]

        # --- MAP hypothesis fit, for explanation
        p0, nci, i, mu, sd, cols = flat[0]
        s0 = self.subsets[i]
        f0 = refit[0]
        resid = y - f0["fit"]
        wmap = f0["weights"]
        chi2 = float(((resid * wmap) ** 2).sum() / dof)
        fit = dict(data=y, aus=f0["au"], nuisance=f0["nuis"], residual=resid)
        # Bayes factor of the best hypothesis vs the empty hypothesis (both with the always-on nuisance terms)
        null_pos = next((r for r in range(len(flat)) if len(self.subsets[flat[r][2]]) == 0 and flat[r][1] == 0), None)
        log_bf = float(np.log(max(p0, 1e-300)) - np.log(max(flat[null_pos][0] if null_pos is not None else 1e-300, 1e-300)))
        hyps = []
        for h, (p, nci_, i_, mu_, sd_, cols_) in enumerate(flat[:top]):
            ss = self.subsets[i_]
            f = refit[h] if h < len(refit) else self.fit_hypothesis([TPL.AUS[k] for k in ss], y, sigma, mask, blink=self.i_blink in nuis_combos[nci_],
                                                                    speech=self.i_speech in nuis_combos[nci_])
            hyps.append(dict(aus=[TPL.AUS[k] for k in ss], blink=self.i_blink in nuis_combos[nci_],
                             speech=self.i_speech in nuis_combos[nci_], p=float(p),
                             amp={TPL.AUS[k]: float(max(f["coef"][k], 0.0)) for k in ss}))
        rigid = f0["coef"][self.i_nuis]
        return AUPosterior(aus=list(TPL.AUS), pip=pip, amp_mean=amp_mean, amp_lo=amp_lo, amp_hi=amp_hi,
                           amp_given=amp_given, p_null=float(p_null), p_blink=float(p_bl), p_speech=float(p_sp),
                           hypotheses=hyps, log_bf_null=log_bf, fit=fit, chi2=chi2, rigid=rigid, set_probs=set_probs,
                           amp_given_lo=g_lo, amp_given_hi=g_hi)
