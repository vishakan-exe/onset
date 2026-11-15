"""Non-expression motion cues taken from the landmarks: blinks, mouth movement (speech), head motion.

These are *priors for the inference engine*, not filters: the engine fits blink/speech templates as
optional components next to the AUs, and these series tell it how plausible each one is at a given time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from .register import LandmarkTrack, Registration

# MediaPipe face-mesh indices.  L/R are image sides (L = the eye on the left of the picture).
EYE = {"L": dict(outer=33, inner=133, up=(160, 159, 158), lo=(144, 145, 153)),
       "R": dict(outer=263, inner=362, up=(387, 386, 385), lo=(373, 374, 380))}
MOUTH = dict(upper=13, lower=14, left=78, right=308)


def eye_aspect(track: LandmarkTrack) -> np.ndarray:
    """(N, 2) eye aspect ratio (aperture / width) for the L and R eye."""
    out = np.zeros((track.n, 2))
    for j, side in enumerate(("L", "R")):
        e = EYE[side]
        width = np.linalg.norm(track.xy[:, e["outer"]] - track.xy[:, e["inner"]], axis=1) + 1e-6
        ap = sum(np.linalg.norm(track.xy[:, u] - track.xy[:, l], axis=1) for u, l in zip(e["up"], e["lo"])) / 3.0
        out[:, j] = ap / width
    return out


def mouth_aperture(track: LandmarkTrack) -> np.ndarray:
    width = np.linalg.norm(track.xy[:, MOUTH["left"]] - track.xy[:, MOUTH["right"]], axis=1) + 1e-6
    return np.linalg.norm(track.xy[:, MOUTH["upper"]] - track.xy[:, MOUTH["lower"]], axis=1) / width


@dataclass
class NuisanceCues:
    ear: np.ndarray               # (N,) mean eye aspect ratio
    ear_rel: np.ndarray           # (N,) EAR / clip baseline (1 = fully open)
    blink: np.ndarray             # (N,) 0..1 plausibility that the lids are closing/closed
    mouth: np.ndarray             # (N,) mouth aperture
    speech: np.ndarray            # (N,) 0..1 plausibility of talking/mouth movement
    head_pos: np.ndarray          # (N, 2) face-centre shift vs the reference frame, IOD, lightly smoothed
    head_roll: np.ndarray         # (N,) roll vs the reference frame, deg, lightly smoothed
    blink_windows: list           # [(t0, t1)] seconds
    speech_windows: list
    blink_mask: np.ndarray = None  # (N,) bool: frames inside a (widened) blink window

    def blink_at(self, t0: float, t1: float, times: np.ndarray) -> float:
        m = (times >= t0) & (times <= t1)
        return float(self.blink[m].max()) if m.any() else 0.0

    def speech_at(self, t0: float, t1: float, times: np.ndarray) -> float:
        m = (times >= t0) & (times <= t1)
        return float(self.speech[m].max()) if m.any() else 0.0

    def head_excursion(self, t0: float, t1: float, times: np.ndarray) -> tuple[float, float]:
        """(translation range in IOD, roll range in deg) of the head between t0 and t1."""
        m = (times >= t0) & (times <= t1)
        if m.sum() < 2:
            return 0.0, 0.0
        tr = np.ptp(self.head_pos[m], axis=0)
        return float(np.hypot(*tr)), float(np.ptp(self.head_roll[m]))


def _windows(flag: np.ndarray, times: np.ndarray, min_len_s: float = 0.0) -> list:
    out, start = [], None
    for i, f in enumerate(flag):
        if f and start is None:
            start = i
        if (not f or i == len(flag) - 1) and start is not None:
            end = i if not f else i
            if times[end] - times[start] >= min_len_s:
                out.append((float(times[start]), float(times[end])))
            start = None
    return out


def cues(track: LandmarkTrack, reg: Registration, times: np.ndarray, fps: float) -> NuisanceCues:
    n = track.n
    ear = eye_aspect(track).mean(1)
    base = np.percentile(ear, 85)
    rel = ear / max(base, 1e-6)
    # a blink is a dip well below the open-eye baseline; the landmark net is noisy, so use a soft score
    blink = np.clip((0.85 - rel) / 0.35, 0.0, 1.0)
    blink_flag = blink > 0.5
    # widen (60 ms before, 150 ms after): the skin around the eye keeps moving after the lids pass the aperture edge
    pre, post = max(1, int(round(0.06 * fps))), max(1, int(round(0.15 * fps)))
    wide = np.zeros_like(blink_flag)
    for i in np.flatnonzero(blink_flag):
        wide[max(0, i - pre): i + post + 1] = True
    # mouth movement: variability of the aperture inside a 0.5 s window relative to its clip-wide spread
    mo = mouth_aperture(track)
    w = max(3, int(round(0.5 * fps)))
    mean = uniform_filter1d(mo, w)
    var = uniform_filter1d((mo - mean) ** 2, w)
    sd = np.sqrt(var)
    ref = np.median(sd) + 1e-6
    thr = max(4.0 * ref, 0.04)
    speech = np.clip((sd - 0.5 * thr) / (0.5 * thr), 0.0, 1.0)
    speech_flag = speech > 0.5
    # head pose from the registration transform (what the landmark stage had to undo); differentiating
    # landmark jitter is noise, so keep positions (lightly smoothed) and report excursions over windows
    sm = max(1, int(round(0.10 * fps)))
    head_pos = uniform_filter1d(reg.head[:, :2], sm, axis=0, mode="nearest")
    head_roll = uniform_filter1d(reg.head[:, 2], sm, mode="nearest")
    return NuisanceCues(ear=ear, ear_rel=rel, blink=blink, mouth=mo, speech=speech, head_pos=head_pos,
                        head_roll=head_roll, blink_windows=_windows(wide, times),
                        speech_windows=_windows(speech_flag, times, 0.25), blink_mask=wide)
