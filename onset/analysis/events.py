"""Measurement pass for one spotted movement: quiet reference -> displacement traces -> apex pattern -> timing.

The spotting stage only says *when* something moved.  Here the movement is measured properly: every ROI's
displacement relative to a quiet frame just before the event is tracked through the event (direct flow from
the reference, so errors do not accumulate), the apex pattern is averaged over the apex window, and onset /
apex / offset are read from the amplitude trace of the best-matching template.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..vision.measure import Measurer
from ..vision.nuisance import NuisanceCues
from ..vision.spotting import Candidate, SpottingResult, whiten_templates
from . import templates as TPL

MICRO_MAX_S = 0.5            # a micro-expression lasts no longer than ~0.5 s from onset to offset (Yan et al. 2013)


@dataclass
class EventMeasurement:
    cand: Candidate
    ref: int
    frames: np.ndarray
    t: np.ndarray
    D: np.ndarray                 # (nf, R, 2) displacement vs reference (px, canonical)
    W: np.ndarray                 # (nf, R) texture weight
    glob: np.ndarray              # (nf, 4) rigid residual removed from each frame
    apex_i: int                   # index into ``frames``
    window: np.ndarray            # indices into ``frames`` averaged for the apex pattern
    pattern: np.ndarray           # (R, 2)
    sigma: np.ndarray             # (R, 2) noise of the pattern
    template: str                 # label of the template used for the timing trace (e.g. 'AU12', 'blink')
    amp_trace: np.ndarray         # (nf,) amplitude of that template along time, px
    amp_sd: float                 # noise of one amplitude sample, px
    onset_t: float
    apex_t: float
    offset_t: Optional[float]     # None when the displacement never returned to baseline inside the clip
    sustained: bool
    head_excursion: tuple
    blink_cue: float
    speech_cue: float
    ref_t: float
    notes: list = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        return None if self.offset_t is None else self.offset_t - self.onset_t


def choose_reference(spot: SpottingResult, i_peak: int, fps: float, n: int,
                     cues: Optional[NuisanceCues] = None, times: Optional[np.ndarray] = None) -> tuple[int, str]:
    """A quiet frame 0.2-1.2 s before the peak, as close to the event as quietness allows.

    Quiet = low spotting energy, no blink and no mouth movement.  Near the clip start the baseline is taken
    after the event instead."""
    lo = int(round(0.20 * fps))
    hi = int(round(1.20 * fps))
    w = max(1, int(round(0.05 * fps)))
    e = np.convolve(spot.frame_energy, np.ones(w) / w, mode="same")
    bad = np.zeros(n, bool)
    if cues is not None:
        bad |= cues.blink > 0.05
        bad |= cues.speech > 0.3
        k = max(1, int(round(0.10 * fps)))
        bad = np.convolve(bad.astype(float), np.ones(2 * k + 1), mode="same") > 0
    thr = np.percentile(e, 35) * 1.5 + 1e-9

    def pick(pool: np.ndarray, closest_first: bool):
        pool = pool[~bad[pool]] if (~bad[pool]).any() else pool
        order = pool[np.argsort(np.abs(pool - i_peak))] if closest_first else pool
        for i in order:
            if e[i] <= thr:
                return int(i)
        return int(pool[np.argmin(e[pool])])

    a, b = max(0, i_peak - hi), i_peak - lo
    if b - a >= max(2, int(0.06 * fps)):
        return pick(np.arange(a, b + 1), True), "before"
    a2, b2 = i_peak + int(round(0.45 * fps)), min(n - 1, i_peak + int(round(1.4 * fps)))
    if b2 - a2 >= 2:
        return pick(np.arange(a2, b2 + 1), True), "after"
    if i_peak - int(round(0.15 * fps)) >= 1:
        return pick(np.arange(0, i_peak - int(round(0.15 * fps)) + 1), True), "before"
    return 0, "none"


def measure_event(meas: Measurer, spot: SpottingResult, cand: Candidate, cues: Optional[NuisanceCues], times: np.ndarray,
                  fps: float, centroids: np.ndarray, centre, max_frames: int = 90) -> EventMeasurement:
    n = meas.reg.gray.shape[0]
    ref, side = choose_reference(spot, cand.index, fps, n, cues, times)
    notes = []
    if side == "after":
        notes.append("baseline taken after the movement (it starts too close to the clip start)")
    if side == "none":
        notes.append("no quiet baseline available")
    post = int(round(0.75 * fps))
    if side == "after":
        f0, f1 = max(0, cand.index - int(round(0.6 * fps))), ref
    else:
        f0, f1 = ref, min(n - 1, cand.index + post)
    frames = np.arange(f0, f1 + 1)
    stride = int(np.ceil(len(frames) / max_frames))
    frames = frames[::stride] if stride > 1 else frames
    if cand.index not in frames:
        frames = np.sort(np.append(frames, cand.index))
    D, W, glob = meas.trace(ref, list(frames))
    t = times[frames]
    lag = max(1, abs(int(cand.index) - ref))
    # noise of a displacement measured against a single reference frame: the pair-flow noise at a comparable lag
    sr = min(spot.scales, key=lambda r: abs(r.spec.k - lag))
    sigma_pair = sr.sigma_pair
    Ap, Q, w = whiten_templates(sigma_pair, centroids, centre)
    Z = D.reshape(len(frames), -1) * w
    Zp = Z - (Z @ Q) @ Q.T
    nrm = np.sqrt((Ap ** 2).sum(1))
    zeta = (Zp @ Ap.T) / np.maximum(nrm, 1e-9)                  # (nf, K)
    # which template matches this movement best, judged on a lightly smoothed trace around the peak
    sm = max(1, int(round(0.03 * fps / stride)))
    zs = np.apply_along_axis(lambda v: np.convolve(v, np.ones(2 * sm + 1) / (2 * sm + 1), mode="same"), 0, zeta)
    K = len(TPL.AUS)
    near = np.abs(t - cand.time) <= 0.25
    if cand.kind == "au":
        zz = zs[:, :K]
        kbest = int(np.argmax(zz[near].max(0))) if near.any() else int(np.argmax(zz.max(0)))
    else:
        kbest = K + (0 if cand.kind == "blink" else 1)
    ztr = zs[:, kbest]
    lim = np.abs(t - cand.time) <= 0.30
    apex_i = int(np.argmax(np.where(lim, ztr, -np.inf)))
    zmax = ztr[apex_i]
    win = np.flatnonzero((ztr >= 0.6 * zmax) & (np.abs(t - t[apex_i]) <= 0.12))
    if len(win) == 0:
        win = np.array([apex_i])
    pattern = D[win].mean(0)
    n_avg = max(1, len(win))
    sigma = sigma_pair * np.sqrt((1.0 + 1.0 / n_avg) / 2.0)
    amp = (Zp @ Ap[kbest]) / max(float(nrm[kbest] ** 2), 1e-12)
    amp_s = np.convolve(amp, np.ones(2 * sm + 1) / (2 * sm + 1), mode="same")
    amp_sd = float(1.0 / max(nrm[kbest], 1e-9))
    a_max = float(amp_s[apex_i])
    level = max(0.15 * a_max, 1.5 * amp_sd)
    # onset: last sample before the apex below the level; offset: first sample after the apex below it
    onset_i = apex_i
    while onset_i > 0 and amp_s[onset_i - 1] > level:
        onset_i -= 1
    onset_t = float(t[onset_i]) if onset_i > 0 else float(t[0])
    if onset_i == 0:
        notes.append("onset precedes the measured window")
    offset_i = apex_i
    while offset_i < len(frames) - 1 and amp_s[offset_i + 1] > level:
        offset_i += 1
    offset_t = float(t[offset_i]) if offset_i < len(frames) - 1 else None
    tail = amp_s[-max(2, len(frames) // 12):]
    sustained = bool(offset_t is None and a_max > 0 and tail.mean() > 0.4 * a_max)
    if sustained:
        notes.append("displacement persists to the end of the measured window (not a brief movement)")
    t_lo, t_hi = onset_t - 0.1, (offset_t if offset_t is not None else t[-1]) + 0.1
    exc = cues.head_excursion(t_lo, t_hi, times) if cues is not None else (0.0, 0.0)
    bl = cues.blink_at(t_lo, t_hi, times) if cues is not None else 0.0
    spc = cues.speech_at(t_lo, t_hi, times) if cues is not None else 0.0
    return EventMeasurement(cand, ref, frames, t, D, W, glob, apex_i, win, pattern, sigma, TPL.scan_label(kbest), amp_s, amp_sd,
                            onset_t, float(t[apex_i]), offset_t, sustained, exc, bl, spc, float(times[ref]), notes)
