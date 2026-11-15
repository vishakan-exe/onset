"""Action-unit displacement fields for phantom faces.

Each AU is a handful of Gaussian "muscle blobs": an anchor on the face, a size (IOD units), a
direction in image coordinates (x right, y down) and a weight.  The *generator* uses these
anatomically-motivated, spatially distributed fields with random per-event variation (position,
direction, size, left/right asymmetry).  The *analysis engine* deliberately uses a different,
coarser description (ROI-mean direction templates, see analysis/au_engine.py), so recovering the AU
from the video is a genuine inference problem and not a lookup of the generator's own table.

L / R are image sides.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# (anchor, sigma_u, sigma_v, dx, dy, weight, side)  side: L, R or C (central)
Blob = tuple[str, float, float, float, float, float, str]


def _lr(anchor: str, su: float, sv: float, dx: float, dy: float, w: float = 1.0) -> list[Blob]:
    """Mirror-symmetric pair of blobs: dx is the *outward-positive for R* convention."""
    return [(f"{anchor}_L", su, sv, -dx, dy, w, "L"), (f"{anchor}_R", su, sv, dx, dy, w, "R")]


AU_BLOBS: dict[int, list[Blob]] = {
    1: _lr("brow_in", 0.20, 0.16, 0.03, -1.0, 1.0)
       + [("glabella", 0.22, 0.20, 0.0, -0.6, 0.5, "C"), ("forehead", 0.42, 0.30, 0.0, -0.35, 0.4, "C")],
    2: _lr("brow_mid", 0.25, 0.15, 0.10, -1.0, 0.8) + _lr("brow_out", 0.22, 0.15, 0.15, -1.0, 1.0)
       + _lr("temple", 0.30, 0.25, 0.10, -0.55, 0.35),
    4: _lr("brow_in", 0.22, 0.17, -0.45, 0.9, 1.0) + _lr("brow_mid", 0.24, 0.14, -0.15, 0.8, 0.6)
       + [("glabella", 0.20, 0.18, 0.0, 0.5, 0.6, "C")] + _lr("lid", 0.20, 0.10, 0.0, 0.4, 0.3),
    5: _lr("lid", 0.20, 0.09, 0.0, -1.0, 1.0) + _lr("brow_mid", 0.25, 0.14, 0.0, -0.25, 0.25),
    6: _lr("cheek", 0.30, 0.25, -0.35, -0.9, 1.0) + _lr("lidlo", 0.20, 0.08, 0.0, -1.0, 0.7)
       + _lr("canthus", 0.14, 0.14, -0.7, -0.5, 0.8),
    7: _lr("lidlo", 0.20, 0.07, 0.0, -1.0, 1.0) + _lr("lid", 0.20, 0.07, 0.0, 0.35, 0.35),
    9: [("nose_root", 0.14, 0.16, 0.0, -1.0, 0.9, "C"), ("glabella", 0.20, 0.16, 0.0, 0.35, 0.4, "C"),
        ("lip_up", 0.22, 0.09, 0.0, -0.5, 0.4, "C")] + _lr("wing", 0.14, 0.12, 0.25, -0.95, 1.0),
    10: [("lip_up", 0.22, 0.09, 0.0, -1.0, 1.0, "C")] + _lr("nasolab", 0.17, 0.20, 0.15, -0.95, 0.8)
        + _lr("wing", 0.14, 0.12, 0.0, -0.5, 0.4),
    12: _lr("corner", 0.19, 0.16, 0.55, -0.85, 1.0) + _lr("nasolab", 0.20, 0.20, 0.40, -0.8, 0.6)
        + _lr("cheek", 0.35, 0.30, 0.30, -0.9, 0.45),
    14: _lr("corner", 0.17, 0.14, -0.9, -0.15, 1.0) + _lr("nasolab", 0.15, 0.15, -0.3, 0.0, 0.3)
        + [("lip_lo", 0.24, 0.08, 0.0, -0.2, 0.2, "C")],
    15: _lr("corner", 0.17, 0.15, 0.2, 0.97, 1.0) + _lr("chin", 0.16, 0.14, 0.0, 0.5, 0.25),
    17: [("chin", 0.30, 0.18, 0.0, -1.0, 1.0, "C"), ("lip_lo", 0.26, 0.09, 0.0, -0.8, 0.7, "C")],
}
SUPPORTED_AUS = sorted(AU_BLOBS)

# nuisances (not AUs)
BLINK_BLOBS: list[Blob] = _lr("cheek", 0.30, 0.22, 0.0, -0.45, 0.5) + _lr("brow_mid", 0.25, 0.14, 0.0, 0.45, 0.4)
SPEECH_BLOBS: list[Blob] = [("lip_lo", 0.28, 0.12, 0.0, 1.0, 1.0, "C"), ("chin", 0.32, 0.20, 0.0, 1.0, 0.9, "C"),
                            ("lip_up", 0.22, 0.08, 0.0, -0.25, 0.3, "C")] \
    + _lr("corner", 0.18, 0.15, 0.35, 0.2, 0.7) + _lr("cheek", 0.32, 0.28, 0.0, 0.35, 0.3)
# conjugate gaze: both eyes' lid skin moves in the SAME image direction (not mirrored like an AU)
GAZE_BLOBS: list[Blob] = [("eye_L", 0.115, 0.095, 1.0, 0.0, 1.0, "L"), ("eye_R", 0.115, 0.095, 1.0, 0.0, 1.0, "R")]


def _anchor(anchors: dict[str, np.ndarray], name: str) -> np.ndarray:
    return anchors[name]


@dataclass
class Jitter:
    """Per-event variation of the AU geometry (what makes two 'AU12' events differ)."""
    amp_scale: float = 1.0
    rot_deg: float = 0.0
    shift: tuple = (0.0, 0.0)       # IOD units
    size_scale: float = 1.0
    blob_weight_sd: float = 0.0
    side: float = 0.5               # fraction of the effect on image-left (0.5 = symmetric)

    @staticmethod
    def sample(rng: np.random.Generator, strength: float = 1.0, side: float | None = None) -> "Jitter":
        if side is None:
            side = float(np.clip(0.5 + rng.normal(0, 0.07 * strength), 0.2, 0.8))
        return Jitter(amp_scale=float(np.exp(rng.normal(0, 0.12 * strength))),
                      rot_deg=float(rng.normal(0, 9.0 * strength)),
                      shift=(float(rng.normal(0, 0.035 * strength)), float(rng.normal(0, 0.035 * strength))),
                      size_scale=float(np.exp(rng.normal(0, 0.12 * strength))),
                      blob_weight_sd=0.12 * strength, side=side)


def side_weights(side: float) -> tuple[float, float]:
    """(wL, wR): 0.5 -> (1,1);  1.0 -> (1,0) (left only);  0.0 -> (0,1)."""
    return min(1.0, 2.0 * side), min(1.0, 2.0 * (1.0 - side))


def field_from_blobs(blobs: list[Blob], anchors: dict[str, np.ndarray], shape: tuple[int, int], scale: int,
                     iod: float, jitter: Jitter, rng: np.random.Generator, strengths: float | dict = 1.0) -> np.ndarray:
    """Displacement (px per unit amplitude=1 IOD) on a coarse grid: (h, w, 2)."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs, ys = (xs + 0.5) * scale - 0.5, (ys + 0.5) * scale - 0.5
    out = np.zeros((h, w, 2), np.float32)
    wl, wr = side_weights(jitter.side)
    c, s = np.cos(np.deg2rad(jitter.rot_deg)), np.sin(np.deg2rad(jitter.rot_deg))
    for anchor, su, sv, dx, dy, wt, side in blobs:
        sw = wl if side == "L" else wr if side == "R" else 1.0
        if sw <= 0:
            continue
        cx, cy = _anchor(anchors, anchor) + np.array(jitter.shift) * iod
        d = np.array([c * dx - s * dy, s * dx + c * dy], np.float32)
        d = d / (np.hypot(*d) + 1e-9)
        wb = wt * sw * float(np.clip(1 + rng.normal(0, jitter.blob_weight_sd), 0.5, 1.6))
        g = np.exp(-0.5 * (((xs - cx) / (su * iod * jitter.size_scale)) ** 2 +
                           ((ys - cy) / (sv * iod * jitter.size_scale)) ** 2))
        out += (wb * g)[..., None] * d[None, None, :]
    return out * iod


def smooth_bump(t: np.ndarray, t_on: float, t_apex: float, t_off: float) -> np.ndarray:
    """0 -> 1 -> 0 with C1-smooth onset and offset (asymmetric: apex need not be central)."""
    t = np.asarray(t, dtype=np.float64)
    up = np.clip((t - t_on) / max(t_apex - t_on, 1e-6), 0, 1)
    dn = np.clip((t_off - t) / max(t_off - t_apex, 1e-6), 0, 1)
    f = np.where(t < t_apex, up, dn)
    return f * f * (3 - 2 * f)
