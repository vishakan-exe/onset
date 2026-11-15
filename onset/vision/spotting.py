"""Micro-movement spotting: a calibrated scan statistic over time.

For every time scale s the transient  T_i = (F[i-k] - F[i]) / 2  (k = scale in frames) is turned into a
whitened 54-vector (27 ROIs x 2) using a robust per-clip noise estimate.  Residual rigid motion is
projected out, then the vector is matched against every AU template; the spotting score at time i is the
largest one-sided z-score over AUs and scales.  Under the null (noise only) each z is ~N(0, 1), the
maximum is calibrated on null phantom clips (scripts/calibrate_engine.py) and the threshold is a
documented constant, not a hand-tuned magic number per clip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.signal import find_peaks

from ..analysis import templates as TPL
from . import geometry as G
from .measure import LagSeries, Measurer

TAUS = (0.07, 0.13, 0.25, 0.60)           # seconds: the time scales (apex-to-baseline distance)
SIGMA_FLOOR = 0.004                       # px, protects against exactly-static footage
SPOT_THRESHOLD = 5.25                     # Studentised z; the calibrated value (data/calibration.json) overrides it: best spotting F1 within 0.5 false alarms per 10 s on the phantom calibration split (scripts/calibrate_engine.py)


@dataclass
class ScaleSpec:
    tau: float
    k: int
    stride: int


def scale_specs(fps: float, n: int, taus=TAUS, pair_budget: int = 1400) -> list[ScaleSpec]:
    """Scales (lag k frames) and strides.  The stride divides k so that both pair flows of a transient exist
    on the grid; it stays small enough that every scale has >= ~8 samples per lag for the noise estimate,
    and grows only when the clip is long (``pair_budget`` caps the number of flow pairs)."""
    ks, seen = [], set()
    for tau in taus:
        k = int(max(1, round(tau * fps)))
        if k in seen or 2 * k + 2 > n:
            continue
        seen.add(k)
        ks.append((tau, k))
    mult = 1
    while True:
        out = []
        for tau, k in ks:
            limit = max(1, k // 8) * mult
            stride = max(d for d in range(1, min(limit, k) + 1) if k % d == 0)
            out.append(ScaleSpec(tau, k, stride))
        pairs = sum(max(0, (n - sp.k) // sp.stride) for sp in out)
        if pairs <= pair_budget or mult > 64:
            return out
        mult += 1


@dataclass
class ScaleResult:
    spec: ScaleSpec
    centre: np.ndarray        # (m,) frame index of each transient
    T: np.ndarray             # (m, R, 2)
    sigma: np.ndarray         # (R, 2) robust *region-specific* noise of T, after the common-mode (rigid) part is projected out
    sigma_pair: np.ndarray    # (R, 2) the same for the raw pair flows (displacement noise at lag k)
    zeta: np.ndarray          # (m, K+2) signed z-scores of the AU, blink and speech templates (nuisance-projected)
    tstat: np.ndarray         # (m, K+2) the same Studentised: divided by the rms of what the template leaves unexplained
    amp: np.ndarray           # (m, K+2) template amplitude estimates (px)
    energy: np.ndarray        # (m,) mean squared whitened residual per dof (nuisance-projected)
    glob: np.ndarray          # (m, 4) mean removed global motion over the pair centred here
    sigma_raw: float = 0.0    # median raw noise of T including the common-mode part (px): the stabilisation jitter
    masked: list = field(default_factory=list)    # MaskedScan per nuisance family (blink, speech): the AU scan with those regions left out


@dataclass
class Candidate:
    index: int                # frame index of the peak
    time: float
    score: float              # largest z over AUs and scales
    scale: ScaleSpec
    au: int                   # best-matching single AU number, or 0 when blink / speech matched best
    kind: str                 # "au" | "blink" | "speech": what the best-matching template is (diagnostic only)
    zeta_by_au: dict          # template label -> z at the peak (best scale)
    amp_px: float             # amplitude of the best single template, px (canonical)
    t_start: float            # rough extent (half-maximum of the best scale's score)
    t_end: float
    energy: float
    note: str = ""
    z_peak: float = 0.0       # plain (not Studentised) z of the winning template at the peak
    masked: str = ""          # "blink" / "speech": found by the scan that leaves that family's regions out


@dataclass
class SpottingResult:
    scales: list
    frame_score: np.ndarray       # (N,) max over scales and AUs of the z-scores (linear interpolation onto frames)
    frame_energy: np.ndarray      # (N,) max over scales of the unexplained+explained energy
    frame_scale: np.ndarray       # (N,) index of the winning scale
    candidates: list
    times: np.ndarray
    fps: float
    threshold: float
    noise_summary: dict = field(default_factory=dict)


def robust_sigma(x: np.ndarray, axis=0) -> np.ndarray:
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def whiten_templates(sigma: np.ndarray, centroids, centre, mask: Optional[np.ndarray] = None):
    """-> (A' (K, 2R) whitened + nuisance-projected AU templates, Q (2R, P) nuisance basis, w (2R,) weights)."""
    R = sigma.shape[0]
    w = 1.0 / sigma.reshape(-1)
    if mask is not None:
        w = w * np.repeat(mask.astype(float), 2)
    A = TPL.scan_templates().reshape(len(TPL.SCAN_KINDS), -1) * w
    N = TPL.nuisance_columns(centroids, centre).reshape(-1, 2 * R) * w
    Q, _ = np.linalg.qr(N.T)
    Ap = A - (A @ Q) @ Q.T
    return Ap, Q, w


def _floor(sig: np.ndarray) -> np.ndarray:
    """Floors: absolute (exactly static footage) and relative to the median ROI (a textureless ROI is noisy, not quiet)."""
    sig = np.maximum(sig, SIGMA_FLOOR)
    return np.maximum(sig, 0.35 * np.median(sig))


def noise_after_projection(X: np.ndarray, centroids, centre, iters: int = 3) -> tuple[np.ndarray, float]:
    """Robust per-ROI-component noise of a series X (m, R, 2) of ROI displacements, *after* the seven nuisance
    directions (residual head motion as a full 2-D affine map: 6, conjugate gaze: 1) have been projected out.

    The raw spread of the series is dominated by the stabilisation error, which moves all regions together and
    is fitted away when a template is matched (on phantoms it is >95 % of the variance); counting it as
    per-region noise makes every z-score several times too small.  The estimate is therefore iterated: whiten with
    the current noise, project the nuisance directions out, measure the spread of what is left (MAD), correct for
    the degrees of freedom the projection removed.  Returns (sigma (R, 2), raw median sigma)."""
    m, R, _ = X.shape
    Xf = X.reshape(m, -1)
    sig_raw = _floor(robust_sigma(Xf).reshape(R, 2))
    N = TPL.nuisance_columns(centroids, centre).reshape(-1, 2 * R)
    sig = sig_raw
    for _ in range(iters):
        w = 1.0 / sig.reshape(-1)
        Q, _ = np.linalg.qr((N * w).T)
        Z = Xf * w
        Zp = Z - (Z @ Q) @ Q.T
        corr = 1.0 / np.sqrt(max(1e-3, 1.0 - Q.shape[1] / Zp.shape[1]))
        sig = _floor(robust_sigma(Zp / w).reshape(R, 2) * corr)
    return sig, float(np.median(sig_raw))


@dataclass
class MaskedScan:
    """The AU scan of one scale with the regions that a nuisance family (blinks: the eyelids; talking: the lower face)
    drags along left out, evaluated only for the transients that overlap one of that family's windows.  It lets a
    movement of the *other* regions be found while the person blinks or talks.  Arrays are (m, K) over the scan
    templates, -inf where a transient is not near a window or a template keeps too little of its energy."""
    name: str
    zeta: np.ndarray
    tstat: np.ndarray
    amp: np.ndarray
    zbest: np.ndarray         # (m,) best AU z, 0 where not evaluated
    tbest: np.ndarray


EYE_ROIS = [G.ROI_NAMES.index(n_) for n_ in G.ROI_GROUPS["eye"]]
LOWER_ROIS = [G.ROI_NAMES.index(n_) for n_ in G.ROI_GROUPS["mouth"] + G.ROI_GROUPS["cheek"]]
MASK_KEEP = 0.30             # a template must keep this share of its energy outside the left-out regions
NEAR_PAD_S = 0.10
MASKED_SCAN = True


def _near_windows(t_centre: np.ndarray, half: float, windows, pad: float = NEAR_PAD_S) -> np.ndarray:
    near = np.zeros(len(t_centre), bool)
    for a, b in windows or []:
        near |= (t_centre + half + pad >= a) & (t_centre - half - pad <= b)
    return near


BLINK_ANCHOR_S = 0.30        # a blink candidate this close to a landmark-detected blink counts as evidenced
STUDENT_TERMS = 3          # templates that may explain a transient before what is left counts as "unexplained"
STUDENT_ADD_Z = 3.0        # a further template must itself be this significant to be added


def greedy_rss(tot: np.ndarray, dots: np.ndarray, G: np.ndarray, nrm2: np.ndarray, max_terms: int = STUDENT_TERMS,
               add_z: float = STUDENT_ADD_Z) -> tuple[np.ndarray, np.ndarray]:
    """Energy left unexplained at every transient after the best greedy fit of up to ``max_terms`` templates (the
    first is the best match; each further one must be significant on its own).  A movement made of two or three
    AUs is thus explained by its parts, while a codec glitch or a pattern that is no combination of AUs still
    leaves most of its energy behind.  Returns (rss (m,), number of templates used (m,))."""
    m, K = dots.shape
    rss = tot.copy()
    used = np.zeros(m, int)
    z = dots / np.sqrt(np.maximum(nrm2, 1e-12))
    diag = np.diag(G).copy()
    for i in range(m):
        k = int(np.argmax(z[i]))
        if z[i, k] <= 0.0:
            continue
        S = [k]
        expl = dots[i, k] ** 2 / max(nrm2[k], 1e-12)
        for _ in range(max_terms - 1):
            GS = G[np.ix_(S, S)]
            c = np.linalg.solve(GS, dots[i, S])
            Gk = G[:, S]
            num = dots[i] - Gk @ c
            den = diag - np.einsum("ks,sk->k", Gk, np.linalg.solve(GS, Gk.T))
            zadd = num / np.sqrt(np.maximum(den, 1e-9))
            zadd[S] = -np.inf
            kn = int(np.argmax(zadd))
            if zadd[kn] < add_z:
                break
            S.append(kn)
            expl += num[kn] ** 2 / max(den[kn], 1e-9)
        rss[i] = max(tot[i] - expl, 0.0)
        used[i] = len(S)
    return rss, used


STUDENT_MODE = "greedy"     # "single": residual after the best template alone; "greedy": after up to STUDENT_TERMS templates


def masked_scan(name: str, rois: list, near: np.ndarray, T: np.ndarray, sig: np.ndarray, centroids, centre,
                nrm2_full: np.ndarray) -> MaskedScan:
    m, R = len(T), T.shape[1]
    K = len(TPL.SCAN_KINDS)
    zeta = np.full((m, K), -np.inf)
    tstat = np.full((m, K), -np.inf)
    amp = np.zeros((m, K))
    zbest = np.zeros(m)
    tbest = np.zeros(m)
    if near.any():
        mask = np.ones(R)
        mask[rois] = 0.0
        Ap, Q, w = whiten_templates(sig, centroids, centre, mask)
        valid = (Ap ** 2).sum(1) >= MASK_KEEP * nrm2_full
        Ap = Ap * valid[:, None]
        nrm2 = (Ap ** 2).sum(1)
        Z = T[near].reshape(int(near.sum()), -1) * w
        Zp = Z - (Z @ Q) @ Q.T
        dots = Zp @ Ap.T
        z = dots / np.sqrt(np.maximum(nrm2, 1e-12))
        dof = max(2, int((w > 0).sum()) - Q.shape[1])
        tot = (Zp ** 2).sum(1)
        rss, used = greedy_rss(tot, dots, Ap @ Ap.T, nrm2)
        t = z / np.maximum(np.sqrt(rss / np.maximum(dof - used, 1))[:, None], 1.0)
        z[:, ~valid] = -np.inf
        t[:, ~valid] = -np.inf
        zeta[near], tstat[near], amp[near] = z, t, dots / np.maximum(nrm2, 1e-12)
        k_au = len(TPL.AUS)
        zb = np.where(np.isfinite(z[:, :k_au]), z[:, :k_au], -1e9).max(1)
        tb = np.where(np.isfinite(t[:, :k_au]), t[:, :k_au], -1e9).max(1)
        zbest[near], tbest[near] = np.maximum(zb, 0.0), np.maximum(tb, 0.0)
    return MaskedScan(name, zeta, tstat, amp, zbest, tbest)


def analyse_scale(spec: ScaleSpec, ls: LagSeries, centroids, centre, times: Optional[np.ndarray] = None,
                  fps: float = 0.0, blink_windows: Optional[list] = None, speech_windows: Optional[list] = None) -> ScaleResult:
    idx, T, W = Measurer.transient(ls)
    R = T.shape[1]
    sig, sig_raw = noise_after_projection(T, centroids, centre)
    sig_pair, _ = noise_after_projection(ls.u, centroids, centre)
    Ap, Q, w = whiten_templates(sig, centroids, centre)
    Z = T.reshape(len(T), -1) * w
    Zp = Z - (Z @ Q) @ Q.T
    nrm2 = (Ap ** 2).sum(1)
    dots = Zp @ Ap.T
    zeta = dots / np.sqrt(np.maximum(nrm2, 1e-12))
    amp = dots / np.maximum(nrm2, 1e-12)
    dof = max(2, Zp.shape[1] - Q.shape[1])
    tot = (Zp ** 2).sum(1)
    energy = tot / dof
    # Studentised score: the template's z divided by the rms of what the template leaves unexplained at this very
    # frame (never boosted: floor 1).  A single-frame codec/exposure glitch, or a pattern that is not an AU at all,
    # leaves a lot of unexplained energy and is discounted; a weak genuine movement leaves only noise.  On null
    # phantoms this removes the heavy tails of the plain z (still clips: p99.9 11.4 -> 4.8).  The plain z is still
    # what ranks the lobes of one movement against each other (the central lobe is the strongest in z but not
    # necessarily in t, because a big movement is never exactly template-shaped).
    if STUDENT_MODE == "greedy":
        rss_g, used = greedy_rss(tot, dots, Ap @ Ap.T, nrm2)
        tstat = zeta / np.maximum(np.sqrt(rss_g / np.maximum(dof - used, 1))[:, None], 1.0)
    else:
        rss = np.maximum(tot[:, None] - zeta ** 2, 0.0)
        tstat = zeta / np.maximum(np.sqrt(rss / (dof - 1)), 1.0)
    # global motion that was removed around each transient centre (mean of the two pair estimates)
    g = np.zeros((len(idx), 4), np.float32)
    index = {int(j): ii for ii, j in enumerate(ls.pos)}
    for m, i in enumerate(idx):
        g[m] = 0.5 * (ls.glob[index[int(i) - ls.k]] + ls.glob[index[int(i)]])
    masked = []
    if MASKED_SCAN and times is not None and fps > 0:
        half = spec.k / fps
        tc = np.asarray(times, float)[idx]
        for name, rois, wins in (("blink", EYE_ROIS, blink_windows), ("speech", LOWER_ROIS, speech_windows)):
            if wins:
                masked.append(masked_scan(name, rois, _near_windows(tc, half, wins), T, sig, centroids, centre, nrm2))
    return ScaleResult(spec, idx, T, sig, sig_pair, zeta, tstat, amp, energy, g, sig_raw, masked)


def _is_blink_lobe(c: "Candidate", blink_windows: Optional[list]) -> bool:
    """An AU candidate that is the mirror image of the blink template (the lids opening) and sits where a blink's
    transient would put it: within 0.2 s of a blink the landmarks saw, or - at the long scales - one lag away from
    one of its edges.  Elsewhere an upper-lid raise is left alone: it may be a real AU5."""
    if not blink_windows or c.kind != "au" or c.zeta_by_au.get("blink", 0.0) > -0.5 * c.z_peak:
        return False
    d_edge = min(max(a - c.time, c.time - b, 0.0) for a, b in blink_windows)
    if d_edge <= 0.2:
        return True
    tau = c.scale.tau
    return tau >= 0.25 and abs(d_edge - tau) <= 0.4 * tau


def suppress_sidelobes(cands: list, results: list, frames: np.ndarray, min_separation_s: float = 0.15,
                       blink_windows: Optional[list] = None) -> list:
    """Reduce the raw peaks to one candidate per movement.

    The transient  T_i = x_i - (x_{i-k} + x_{i+k}) / 2  of a bump of the skin position x leaves opposite-signed
    side lobes a distance k before and after the bump (they look like the antagonist AU, and the reopening of the
    lids after a blink looks like a raised upper lid).  Candidates are visited strongest first, blinks and speech
    before AUs, so a side lobe can never outrank the movement that causes it.  A candidate is dropped when
      * it is a side lobe: an already-kept candidate sits about one lag k away, is at least as strong in plain z, and
        its template fitted at this position and scale comes out clearly *negative* (z < -2).  (Amplitudes are not
        compared: flow under-measures large movements such as blinks, so the textbook ratio of one half does not hold.)
      * it duplicates a kept candidate of the same family (AU / blink / speech) within ``min_separation_s``;
      * it is an AU candidate within ``min_separation_s`` of a stronger kept blink or speech candidate (the AU
        template merely correlates with the non-AU pattern);
      * it is the mirror image of a blink (its blink-template z is at most -0.5 of its own z: the lids opening) where
        a blink the landmarks saw would put it (``_is_blink_lobe``) - a blink is not a point event, so the lobe geometry
        alone cannot catch every lid-opening transient.
    A genuinely independent opposite movement inside a side-lobe window and weaker than its neighbour would be
    absorbed; that is accepted."""
    def anchored(c: "Candidate") -> bool:
        """A non-AU candidate whose cause is independently evidenced: speech, or a blink the landmarks saw nearby.
        Only such candidates may claim the lobes around them first.  A *blink-like* candidate without a landmark blink
        is as likely to be the echo of a raised upper lid (AU5: its side lobes have the blink template's sign)."""
        if c.kind == "speech":
            return True
        if c.kind != "blink" or not blink_windows:
            return False
        return min(max(a - c.time, c.time - b, 0.0) for a, b in blink_windows) <= BLINK_ANCHOR_S

    order = sorted(cands, key=lambda c: (not anchored(c), -c.z_peak))
    kept: list = []
    for c in order:
        drop = False
        r = next(r for r in results if r.spec.k == c.scale.k)
        j = int(np.clip(np.searchsorted(r.centre, c.index), 0, len(r.centre) - 1))
        if j > 0 and abs(r.centre[j - 1] - c.index) < abs(r.centre[j] - c.index):
            j -= 1
        if c.masked != "blink" and _is_blink_lobe(c, blink_windows):
            c.note = "eyelids opening or closing around a blink"
            continue
        for o in kept:
            dt = abs(o.time - c.time)
            # a candidate found with a nuisance family's regions left out cannot be that family's movement
            by_family = bool(c.masked) and o.kind == c.masked
            if dt <= min_separation_s and (o.kind == c.kind or (c.kind == "au" and o.z_peak >= c.z_peak and not by_family)):
                drop = True
                c.note = f"same movement as the {o.kind if o.kind != 'au' else 'AU' + str(o.au)} candidate at {o.time:.2f}s"
                break
            if not (0.5 * c.scale.tau <= dt <= 1.6 * c.scale.tau) or by_family:
                continue
            ko = TPL.SCAN_KINDS.index((o.kind, o.au))
            # a step-like movement leaves lobes of equal strength on both sides, so a blink / speech cause need only be
            # half as strong as its lobe; an AU cause must be at least as strong
            need = 0.5 if anchored(o) else 1.0
            if r.zeta[j, ko] < -2.0 and o.z_peak >= need * c.z_peak:
                drop = True
                c.note = f"side lobe of the {o.kind if o.kind != 'au' else 'AU' + str(o.au)} movement at {o.time:.2f}s"
                break
        if not drop:
            kept.append(c)
    return sorted(kept, key=lambda c: c.time)


def compute_lagseries(meas: Measurer, fps: float, taus=TAUS,
                      progress: Optional[Callable[[float], None]] = None) -> tuple[list, list]:
    """The expensive part: dense flows at every scale -> (specs, lag series).  Can be cached and re-scanned."""
    n = meas.reg.gray.shape[0]
    specs = scale_specs(fps, n, taus)
    if not specs:
        raise ValueError("clip too short for the spotting time scales")
    lagseries = []
    for si, spec in enumerate(specs):
        sub = None if progress is None else (lambda p, si=si: progress((si + p) / len(specs)))
        lagseries.append(meas.lag_series(spec.k, spec.stride, sub))
    return specs, lagseries


def scan(specs: list, lagseries: list, n: int, times: np.ndarray, fps: float, centroids, centre,
         threshold: float = SPOT_THRESHOLD, min_separation_s: float = 0.15, blink_windows: Optional[list] = None,
         speech_windows: Optional[list] = None) -> SpottingResult:
    """The cheap part: noise model, template statistics, peak picking, side-lobe suppression.

    Peaks are located on the plain z (the central lobe of a movement is its strongest lobe in z), but a peak only
    becomes a candidate when its *Studentised* score reaches ``threshold`` (glitch-robust)."""
    results = [analyse_scale(spec, ls, centroids, centre, times, fps, blink_windows, speech_windows)
               for spec, ls in zip(specs, lagseries)]
    frames = np.arange(n)
    z_scales = np.full((len(results), n), -np.inf)
    t_scales = np.full((len(results), n), -np.inf)
    energy_scales = np.zeros((len(results), n))
    for si, r in enumerate(results):
        if len(r.centre) == 0:
            continue
        z_scales[si] = np.interp(frames, r.centre, r.zeta.max(1), left=-np.inf, right=-np.inf)
        t_scales[si] = np.interp(frames, r.centre, r.tstat.max(1), left=-np.inf, right=-np.inf)
        energy_scales[si] = np.interp(frames, r.centre, r.energy, left=0, right=0)
    z_frame = np.max(z_scales, axis=0)
    z_frame[~np.isfinite(z_frame)] = 0.0
    frame_score = np.max(t_scales, axis=0)                      # what the interface plots
    frame_score[~np.isfinite(frame_score)] = 0.0
    frame_scale = np.argmax(z_scales, axis=0)
    frame_energy = energy_scales.max(0)
    # --- candidates
    # peaks are picked with a short minimum distance (the lobes of a movement at the shortest scale are only two
    # frames apart); one-event-per-cluster is enforced afterwards by ``suppress_sidelobes``
    peaks, _ = find_peaks(z_frame, height=threshold, distance=max(1, int(round(0.05 * fps))))
    cands = []
    for p in peaks:
        si = int(frame_scale[p])
        r = results[si]
        j = int(np.clip(np.searchsorted(r.centre, p), 0, len(r.centre) - 1))
        if j > 0 and abs(r.centre[j - 1] - p) < abs(r.centre[j] - p):
            j -= 1
        zbest = r.zeta[j]
        k_best = int(np.argmax(zbest))
        # Studentised score of the winning template, best of the neighbouring transient samples (a glitch-free event
        # may peak one sample away in t from where it peaks in z)
        lo, hi = max(0, j - 1), min(len(r.centre), j + 2)
        t_pk = float(r.tstat[lo:hi, k_best].max())
        if t_pk < threshold:
            continue
        # extent: where this scale's best z stays above half of the peak
        zc = np.interp(frames, r.centre, r.zeta.max(1), left=0, right=0)
        half = 0.5 * z_frame[p]
        a = p
        while a > 0 and zc[a - 1] >= half:
            a -= 1
        b = p
        while b < n - 1 and zc[b + 1] >= half:
            b += 1
        kind, au_n = TPL.SCAN_KINDS[k_best]
        cands.append(Candidate(index=int(p), time=float(times[p]), score=t_pk, scale=r.spec,
                               au=int(au_n), kind=kind,
                               zeta_by_au={TPL.scan_label(i_): float(z) for i_, z in enumerate(zbest)},
                               amp_px=float(r.amp[j, k_best]), t_start=float(times[a]), t_end=float(times[b]),
                               energy=float(frame_energy[p]), z_peak=float(zbest[k_best])))
    # --- candidates from the masked scans: movements of the regions a blink / talking leaves alone
    k_au = len(TPL.AUS)
    for name in ("blink", "speech"):
        zm_scales = np.zeros((len(results), n))
        tm_scales = np.zeros((len(results), n))
        have = False
        for si, r in enumerate(results):
            ms = next((x for x in r.masked if x.name == name), None)
            if ms is None or len(r.centre) == 0 or not ms.zbest.any():
                continue
            have = True
            zm_scales[si] = np.interp(frames, r.centre, ms.zbest, left=0, right=0)
            tm_scales[si] = np.interp(frames, r.centre, ms.tbest, left=0, right=0)
        if not have:
            continue
        zm_frame = zm_scales.max(0)
        frame_score = np.maximum(frame_score, tm_scales.max(0))
        mscale = np.argmax(zm_scales, axis=0)
        mpeaks, _ = find_peaks(zm_frame, height=threshold, distance=max(1, int(round(0.05 * fps))))
        for p in mpeaks:
            r = results[int(mscale[p])]
            ms = next(x for x in r.masked if x.name == name)
            j = int(np.clip(np.searchsorted(r.centre, p), 0, len(r.centre) - 1))
            if j > 0 and abs(r.centre[j - 1] - p) < abs(r.centre[j] - p):
                j -= 1
            zrow = ms.zeta[j, :k_au]
            if not np.isfinite(zrow).any():
                continue
            k_best = int(np.argmax(np.where(np.isfinite(zrow), zrow, -np.inf)))
            lo, hi = max(0, j - 1), min(len(r.centre), j + 2)
            t_pk = float(ms.tstat[lo:hi, k_best].max())
            if not np.isfinite(t_pk) or t_pk < threshold:
                continue
            zc = np.interp(frames, r.centre, ms.zbest, left=0, right=0)
            half = 0.5 * zm_frame[p]
            a = b = p
            while a > 0 and zc[a - 1] >= half:
                a -= 1
            while b < n - 1 and zc[b + 1] >= half:
                b += 1
            kind, au_n = TPL.SCAN_KINDS[k_best]
            twin = next((c for c in cands if c.kind == "au" and abs(c.time - times[p]) <= min_separation_s), None)
            if twin is not None and twin.score >= t_pk:
                if twin.au == au_n:
                    twin.masked = twin.masked or name  # the ordinary scan has it, and the other regions confirm it
                continue
            cand = Candidate(index=int(p), time=float(times[p]), score=t_pk, scale=r.spec, au=int(au_n), kind=kind,
                             zeta_by_au={TPL.scan_label(i_): (float(z) if np.isfinite(z) else 0.0) for i_, z in enumerate(r.zeta[j])},
                             amp_px=float(ms.amp[j, k_best]), t_start=float(times[a]), t_end=float(times[b]),
                             energy=float(frame_energy[p]), z_peak=float(zrow[k_best]), masked=name,
                             note=("eyelids left out: a blink is nearby" if name == "blink" else "lower face left out: mouth movement nearby"))
            if twin is not None:
                cands[cands.index(twin)] = cand
            else:
                cands.append(cand)
    cands = suppress_sidelobes(cands, results, frames, min_separation_s, blink_windows)
    summary = {f"tau_{r.spec.tau:.2f}": dict(k=r.spec.k, stride=r.spec.stride, sigma_median_px=float(np.median(r.sigma)),
                                           sigma_pair_median_px=float(np.median(r.sigma_pair)),
                                           sigma_raw_median_px=float(r.sigma_raw)) for r in results}
    return SpottingResult(results, frame_score, frame_energy, frame_scale, cands, times, fps, threshold, summary)


def spot(meas: Measurer, times: np.ndarray, fps: float, centroids, centre, threshold: float = SPOT_THRESHOLD,
         taus=TAUS, min_separation_s: float = 0.15,
         progress: Optional[Callable[[float], None]] = None, return_lagseries: bool = False,
         blink_windows: Optional[list] = None, speech_windows: Optional[list] = None):
    n = meas.reg.gray.shape[0]
    specs, lagseries = compute_lagseries(meas, fps, taus, progress)
    out = scan(specs, lagseries, n, times, fps, centroids, centre, threshold, min_separation_s, blink_windows, speech_windows)
    return (out, lagseries) if return_lagseries else out
