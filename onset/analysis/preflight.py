"""Pre-flight check for a webcam recording: a 1-2 second burst of frames, measured with the same machinery as the
analysis, so that the sensitivity it reports is a measurement of *this* camera, light and face - not a guess.

It answers four questions before the person records anything: is there a face and is it large enough, is the
picture usable (light, focus, clipping), is the head steady, and how small a movement could this set-up show?
The sensitivity is previewed on the short burst; the report grades the full clip.
"""
from __future__ import annotations

import numpy as np

from ..vision.measure import Measurer
from ..vision.register import register, track_landmarks
from ..vision.sources import ArraySource
from ..vision.spotting import noise_after_projection
from . import quality as QUAL
from . import templates as TPL

MIN_FRAMES = 12


def _status(v: float, good: float, ok: float, higher_is_better: bool = True) -> str:
    return QUAL._status(v, good, ok, higher_is_better)


def preflight(frames: np.ndarray, times: np.ndarray, mda_scale: float = 1.0) -> dict:
    """``frames``: N x H x W x 3 BGR uint8 (as captured), ``times``: seconds.  Always returns a dict; ``ok`` says whether
    recording would be worthwhile.  ``mda_scale``: the phantom-calibrated correction of the analytic smallest movement."""
    n = len(frames)
    out = dict(ok=False, found=False, items=[], advice=[], face=None, grade=None, mda_pct=None, summary="")
    if n < MIN_FRAMES:
        out["advice"].append("The preview needs a moment longer to measure the picture.")
        return out
    src = ArraySource(frames, np.asarray(times, float))
    info = src.info()
    H, W = frames.shape[1:3]
    try:
        track = track_landmarks(src)
    except ValueError:
        out["advice"].append("No face found. Face the camera, keep your whole face in view, and add light from the front.")
        out["items"].append(dict(name="Face found", value=0, unit="% of frames", status="poor", note=""))
        out["summary"] = "No face in view."
        return out
    found = float(track.valid.mean())
    xy = track.xy[track.valid] if track.valid.any() else track.xy
    box = np.array([xy[..., 0].min(), xy[..., 1].min(), xy[..., 0].max(), xy[..., 1].max()], float)
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    iod = float(track.iod_px)
    items, advice = [], []

    def add(name, value, unit, status, note=""):
        items.append(dict(name=name, value=value, unit=unit, status=status, note=note))

    add("Face found", round(100 * found), "% of frames", _status(found, 0.98, 0.85))
    add("Face size", round(iod), "px between the eyes", _status(iod, 80, 50), "more pixels on the face mean finer measurements")
    off = float(np.hypot((cx - W / 2) / W, (cy - H / 2) / H))
    inside = bool(box[0] > 0.02 * W and box[2] < 0.98 * W and box[1] > 0.0 and box[3] < 0.99 * H)
    add("Framing", round(100 * off), "% off centre", "good" if inside and off < 0.12 else "ok" if inside else "poor",
        "keep the whole face inside the picture")
    if iod < 50:
        advice.append("Move closer to the camera: under 50 px between the eyes the movement estimates are coarse.")
    if not inside:
        advice.append("Part of the face is at the edge of the picture. Centre it.")
    fps = info["fps"]
    add("Frame rate", round(fps, 1), "fps", _status(fps, 60, 24), "short movements need at least three frames")
    add("Timing jitter", round(100 * info["fps_jitter"], 1), "%", _status(info["fps_jitter"], 0.03, 0.12, False))
    if fps and fps < 20:
        advice.append("The camera is delivering fewer than 20 frames a second. Close other programs that use it, or add light so it can expose faster.")

    ok_sens = False
    try:
        reg = register(src, track)
        st = reg.stats
        lum, sat, blur = float(np.median(st["lum"])), float(np.median(st["sat"])), float(np.median(st["blur"]))
        add("Light", round(lum), "mean grey level (0-255)", "good" if 70 <= lum <= 200 else "ok" if 45 <= lum <= 225 else "poor",
            "your screen can be your key light: keep this window bright")
        add("Clipped pixels", round(100 * sat, 2), "% of skin", _status(sat, 0.01, 0.05, False))
        add("Sharpness", round(blur), "Laplacian variance", _status(blur, 400, 120), "skin texture is what movement is measured on")
        if lum < 70:
            advice.append("The picture is dark. Face a window or a lamp, or turn up this screen: sensor noise rises quickly in dim light.")
        if lum > 225 or sat > 0.05:
            advice.append("Part of the face is overexposed. Soften the light or move away from direct light.")
        if blur < 120:
            advice.append("The face is soft. Check focus and clean the lens; strong compression also hides skin texture.")
        head = reg.head
        steady = float(np.sqrt(np.var(head[:, 0]) + np.var(head[:, 1]))) if len(head) > 2 else 0.0   # IOD units
        add("Head steadiness", round(100 * steady, 2), "% of eye distance (spread)", _status(steady, 0.010, 0.03, False),
            "small head movements are removed; large ones are not")
        if steady > 0.03:
            advice.append("Hold your head still while recording. Rest it against a headrest or the back of the chair if that helps.")
        cent = TPL.roi_centroids(reg.rois)
        meas = Measurer(reg, threads=2)
        k = int(max(1, round(0.25 * fps)))
        if 2 * k + 3 <= n:
            ls = meas.lag_series(k, stride=max(1, k // 4))
            sig, _ = noise_after_projection(ls.u, cent, meas.centre)
            mda50, _ = QUAL.mda_from_sigma(sig, cent, meas.centre, reg.canon.iod)
            med = float(np.median(list(mda50.values()))) * float(mda_scale)       # calibrated, like the report's
            grade = QUAL.GRADES[int(np.searchsorted(QUAL.MDA_EDGES, med))]
            add("Smallest movement this set-up can show", round(med, 2), "% of eye distance (preview)",
                "good" if grade in "AB" else "ok" if grade == "C" else "poor", "measured on this burst; the report grades the full clip")
            out.update(grade=grade, mda_pct=med)
            ok_sens = True
            if grade in ("D", "E"):
                advice.append("At this noise level only large movements would be visible. More light and a steadier head help most.")
    except Exception as e:     # a preview must never take the page down
        advice.append("The picture could not be measured in detail yet; try again in a moment.")
        out["debug"] = f"{type(e).__name__}: {e}"
    hard = [it for it in items if it["status"] == "poor" and it["name"] in ("Face found", "Face size", "Framing", "Light")]
    out.update(ok=bool(not hard and found > 0.8), found=True, items=items, advice=advice,
               face=dict(box=[float(box[0] / W), float(box[1] / H), float(box[2] / W), float(box[3] / H)], iod_px=iod,
                         width=int(W), height=int(H)))
    out["summary"] = ("Ready to record." if out["ok"] and not advice else
                      "Usable, with room to improve." if out["ok"] else "Fix the red items first.")
    if not ok_sens and out["ok"] and not advice:
        out["summary"] = "Ready to record."
    return out
