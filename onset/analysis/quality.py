"""Capture quality and the clip's own sensitivity: what is the smallest movement this footage can show?

The headline number is the *minimum detectable amplitude* (MDA): the template amplitude that reaches the
spotting threshold given the noise actually measured in this clip (robust spread of ROI displacement between
frames that should show nothing).  It is expressed in percent of the inter-ocular distance (%IOD) so that it is
comparable between cameras and distances.  The grade is a bucketed MDA, lowered one step by hard capture
problems (too few pixels on the face, heavy blur, clipped exposure, dropped frames).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm

from ..vision.register import LandmarkTrack, Registration
from ..vision.spotting import SpottingResult, whiten_templates
from . import templates as TPL

GRADES = ["A", "B", "C", "D", "E"]
MDA_EDGES = [0.6, 1.0, 1.6, 2.5]        # %IOD (50 % detection point) separating A|B|C|D|E
Z_CRIT = 5.0


@dataclass
class Quality:
    grade: str
    mda_median_pct: float
    mda50_pct: dict               # AU -> amplitude (%IOD) detected 50 % of the time
    mda90_pct: dict               # AU -> amplitude (%IOD) detected 90 % of the time
    min_duration_ms: float
    items: list = field(default_factory=list)
    advice: list = field(default_factory=list)
    summary: str = ""
    iod_px: float = 0.0           # distance between the eyes in the recording's own pixels (to turn % of it into pixels)

    def to_dict(self) -> dict:
        return dict(grade=self.grade, mda_median_pct=self.mda_median_pct, mda50_pct=self.mda50_pct,
                    mda90_pct=self.mda90_pct, min_duration_ms=self.min_duration_ms, items=self.items,
                    advice=self.advice, summary=self.summary, iod_px=self.iod_px)


def mda_from_sigma(sigma_pair: np.ndarray, centroids, centre, iod_canon: float, z_crit: float = Z_CRIT,
                   mask: Optional[np.ndarray] = None) -> tuple[dict, dict]:
    """Per-AU amplitude (%IOD) at which a template movement is detected 50 % / 90 % of the time."""
    Ap, _, _ = whiten_templates(sigma_pair, centroids, centre, mask)
    nrm = np.sqrt((Ap[: len(TPL.AUS)] ** 2).sum(1))
    m50 = z_crit / np.maximum(nrm, 1e-9)
    m90 = (z_crit + norm.ppf(0.90)) / np.maximum(nrm, 1e-9)
    f = 100.0 / iod_canon
    return ({a: float(v * f) for a, v in zip(TPL.AUS, m50)}, {a: float(v * f) for a, v in zip(TPL.AUS, m90)})


def _status(v: float, good: float, ok: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "good" if v >= good else "ok" if v >= ok else "poor"
    return "good" if v <= good else "ok" if v <= ok else "poor"


def calibrate_mda(mda50: dict, mda90: dict, mda_scale: float = 1.0, mda90_scale: Optional[float] = None) -> tuple[dict, dict]:
    """Bring the analytic smallest-detectable amplitudes (z = Z_CRIT against the clip's own noise) onto what the phantom
    corpus showed was actually found: ``mda_scale`` multiplies the 50 % point; the 90 % point is the measured
    ``mda90_scale`` times the *uncorrected* 50 % point (when the corpus never reached 90 %, the analytic gap between
    the two points carries over instead)."""
    c50 = {a: v * mda_scale for a, v in mda50.items()}
    if mda90_scale is not None and np.isfinite(mda90_scale):
        c90 = {a: max(v * mda90_scale, c50[a]) for a, v in mda50.items()}
    else:
        c90 = {a: v * mda_scale for a, v in mda90.items()}
    return c50, c90


def assess(info: dict, track: LandmarkTrack, reg: Registration, sigma_pair: np.ndarray, centroids, centre,
           face_found_fraction: float = 1.0, mda_scale: float = 1.0, mda90_scale: Optional[float] = None) -> Quality:
    fps = info.get("fps", 0.0)
    jitter = info.get("fps_jitter", 0.0)
    iod_src = track.iod_px
    mda50, mda90 = calibrate_mda(*mda_from_sigma(sigma_pair, centroids, centre, reg.canon.iod), mda_scale, mda90_scale)
    med = float(np.median(list(mda50.values())))
    grade_i = int(np.searchsorted(MDA_EDGES, med))
    st = reg.stats
    lum = float(np.median(st["lum"]))
    sat = float(np.median(st["sat"]))
    blur = float(np.median(st["blur"]))
    noise = float(np.median(st["noise"]))
    max_gap = info.get("max_gap_ms", 0.0)
    items, advice = [], []

    def add(name, value, unit, status, note=""):
        items.append(dict(name=name, value=value, unit=unit, status=status, note=note))

    add("Frame rate", round(fps, 1), "fps", _status(fps, 60, 24), "short events need at least 3 frames to be resolved")
    add("Frame-timing jitter", round(100 * jitter, 1), "%", _status(jitter, 0.03, 0.10, False),
        "irregular timestamps blur the time axis")
    add("Face size", round(iod_src, 0), "px between the eyes", _status(iod_src, 80, 50),
        "more pixels per face = finer displacement resolution")
    add("Exposure", round(lum, 0), "mean grey level (0-255)", "good" if 70 <= lum <= 200 else "ok" if 45 <= lum <= 225 else "poor",
        "too dark amplifies noise, too bright clips texture")
    add("Clipped pixels", round(100 * sat, 2), "% of skin", _status(sat, 0.01, 0.05, False))
    add("Sharpness", round(blur, 0), "Laplacian variance (canonical crop)", _status(blur, 400, 120),
        "skin texture is what the motion estimate locks on to")
    add("Pixel noise", round(noise, 2), "grey levels (upper bound)", _status(noise, 2.0, 4.0, False))
    add("Face found", round(100 * face_found_fraction, 0), "% of frames", _status(face_found_fraction, 0.98, 0.9))
    add("Sensitivity (median AU)", round(med, 2), "% of eye distance, found half the time", GRADES[grade_i] in "AB" and "good" or
        ("ok" if GRADES[grade_i] == "C" else "poor"), "smallest movement this clip can show")

    penalties = 0
    if iod_src < 40:
        penalties += 1
        advice.append("Move closer or zoom so that the face is larger: under 40 px between the eyes the movement estimate is coarse.")
    if sat > 0.05:
        penalties += 1
        advice.append("Part of the face is over- or under-exposed; soften harsh light and avoid strong backlight.")
    if blur < 120:
        penalties += 1
        advice.append("The face is soft (focus, motion blur or strong compression); skin texture is needed to measure movement.")
    if face_found_fraction < 0.9:
        penalties += 1
        advice.append("The face was lost in more than 10 % of the frames; keep it centred and unobstructed.")
    if max_gap > 3.0 * 1000.0 / max(fps, 1.0):
        penalties += 1
        advice.append("The recording has dropped or delayed frames; close other applications that use the camera.")
    if fps and fps < 24:
        advice.append(f"At {fps:.0f} fps movements shorter than about {3000.0 / fps:.0f} ms cannot be resolved; use a higher frame rate if the camera allows.")
    if lum < 70:
        advice.append("The picture is dark; add front light - sensor noise rises quickly in dim light.")
    if med > MDA_EDGES[1] and not advice:
        advice.append("More light, a closer face and a steadier head all lower the smallest detectable movement.")
    grade = GRADES[min(len(GRADES) - 1, grade_i + penalties)]
    min_dur = 3000.0 / fps if fps else float("nan")
    med90 = float(np.median(list(mda90.values())))
    summary = (f"Grade {grade}: a movement of about {med:.1f} % of the eye distance ({med * iod_src / 100:.1f} px in this recording) is found about "
               f"half the time, and about {med90:.1f} % ({med90 * iod_src / 100:.1f} px) nine times in ten (rates measured on synthetic faces). "
               f"The shortest event it can resolve is about {min_dur:.0f} ms.")
    return Quality(grade, med, mda50, mda90, float(min_dur), items, advice, summary, float(iod_src))
