"""Face geometry derived from the 468-point mesh: stable landmarks, named anchors and ROIs.

All side names are *image* sides (L = viewer's left).  Regions of interest (ROIs) are the
skin areas whose motion is measured; they are built once per clip from the quiet reference
landmarks (in canonical, eye-line-aligned coordinates) and stay fixed afterwards, so the
measurement never depends on per-frame landmark jitter.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

N_LM = 468

# --- MediaPipe topology (verified visually against real faces) ----------------------------
EYE_L_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_R_RING = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
EYE_L_UPPER = [33, 246, 161, 160, 159, 158, 157, 173, 133]
EYE_L_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
EYE_R_UPPER = [263, 466, 388, 387, 386, 385, 384, 398, 362]
EYE_R_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]
BROW_L_UP, BROW_L_LO = [70, 63, 105, 66, 107], [46, 53, 52, 65, 55]        # outer -> inner
BROW_R_UP, BROW_R_LO = [300, 293, 334, 296, 336], [276, 283, 282, 295, 285]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIP_UP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIP_LO_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
CHEEK_L = [123, 117, 118, 119, 100, 101, 50, 36, 205, 187, 207, 216]
CHEEK_R = [352, 346, 347, 348, 329, 330, 280, 266, 425, 411, 427, 436]

# Landmarks that barely move with expression: nasal dorsum + temples.  Used for registration.
STABLE_REG = [168, 6, 197, 195, 5, 4, 1, 193, 417, 122, 351, 245, 465, 127, 356, 162, 389, 21, 251,
              234, 454, 94, 19, 2]
# The same idea, but as image regions (the flow step fits head motion on these pixels).
STABLE_REGION_PTS = [[168, 6, 197, 195, 5, 4], [127, 162, 21, 54, 103, 109, 108, 69, 68, 71],
                     [356, 389, 251, 284, 332, 338, 337, 299, 298, 301]]

SIDES = ("L", "R")


def eye_centres(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centres of the image-left and image-right eye (mean of the eyelid ring)."""
    return xy[EYE_L_RING].mean(0), xy[EYE_R_RING].mean(0)


def interocular(xy: np.ndarray) -> float:
    a, b = eye_centres(xy)
    return float(np.hypot(*(b - a)))


@dataclass
class Canon:
    """Canonical frame definition (square crop, eyes on a horizontal line)."""
    size: int = 256
    iod: float = 64.0

    @property
    def eye_y(self) -> float:
        return 0.41 * self.size

    def eye_targets(self) -> tuple[np.ndarray, np.ndarray]:
        cx, ey = self.size / 2.0, self.eye_y
        return np.array([cx - self.iod / 2, ey]), np.array([cx + self.iod / 2, ey])


def similarity_from_two_points(src_a, src_b, dst_a, dst_b) -> np.ndarray:
    """2x3 similarity mapping segment (src_a,src_b) onto (dst_a,dst_b)."""
    sa, sb, da, db = (np.asarray(v, dtype=np.float64) for v in (src_a, src_b, dst_a, dst_b))
    sv, dv = sb - sa, db - da
    scale = np.hypot(*dv) / max(np.hypot(*sv), 1e-9)
    ang = np.arctan2(dv[1], dv[0]) - np.arctan2(sv[1], sv[0])
    c, s = np.cos(ang) * scale, np.sin(ang) * scale
    r = np.array([[c, -s], [s, c]])
    t = da - r @ sa
    return np.hstack([r, t[:, None]])


def umeyama(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Least-squares similarity transform (2x3) taking ``src`` onto ``dst`` (both n x 2)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mu_s, mu_d = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = (w[:, None] * d0).T @ s0
    u, sv, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    dm = np.diag([1.0, d])
    r = u @ dm @ vt
    var_s = (w * (s0 ** 2).sum(1)).sum()
    scale = (sv * np.array([1.0, d])).sum() / max(var_s, 1e-12)
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]])


def apply_affine(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ m[:, :2].T + m[:, 2]


def similarity_params(m: np.ndarray) -> tuple[float, float, float, float]:
    """(tx, ty, rotation_rad, scale) of a 2x3 similarity."""
    a, b = m[0, 0], m[1, 0]
    return float(m[0, 2]), float(m[1, 2]), float(np.arctan2(b, a)), float(np.hypot(a, b))


# --------------------------------------------------------------------------------------
# ROIs
# --------------------------------------------------------------------------------------
def _hull_mask(points: np.ndarray, size: int, dilate: float = 0.0) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        cv2.fillConvexPoly(m, np.round(hull * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
    elif len(pts) >= 1:
        for p in pts:
            cv2.circle(m, tuple(int(round(v)) for v in p), 1, 1, -1)
    if dilate > 0:
        k = int(round(dilate))
        if k > 0:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return m.astype(bool)


def _disc(centre: np.ndarray, radius: float, size: int) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (int(round(centre[0])), int(round(centre[1]))), max(int(round(radius)), 1), 1, -1)
    return m.astype(bool)


ROI_NAMES = [
    "fh_L", "fh_R", "brow_out_L", "brow_mid_L", "brow_in_L", "brow_in_R", "brow_mid_R", "brow_out_R",
    "glabella", "lid_up_L", "lid_up_R", "lid_lo_L", "lid_lo_R", "canthus_L", "canthus_R",
    "nose_root", "wing_L", "wing_R", "cheek_L", "cheek_R", "nasolab_L", "nasolab_R",
    "lip_up", "lip_lo", "corner_L", "corner_R", "chin",
]
ROI_GROUPS = {
    "brow": ["fh_L", "fh_R", "brow_out_L", "brow_mid_L", "brow_in_L", "brow_in_R", "brow_mid_R",
             "brow_out_R", "glabella"],
    "eye": ["lid_up_L", "lid_up_R", "lid_lo_L", "lid_lo_R", "canthus_L", "canthus_R"],
    "nose": ["nose_root", "wing_L", "wing_R"],
    "cheek": ["cheek_L", "cheek_R", "nasolab_L", "nasolab_R"],
    "mouth": ["lip_up", "lip_lo", "corner_L", "corner_R", "chin"],
}
GROUP_OF = {r: g for g, rs in ROI_GROUPS.items() for r in rs}
GROUP_ORDER = list(ROI_GROUPS.keys())


def build_rois(lm_c: np.ndarray, canon: Canon) -> dict[str, np.ndarray]:
    """Boolean masks (canon.size^2) in canonical coordinates from reference landmarks ``lm_c``."""
    s, iod = canon.size, canon.iod
    p = lambda idx: lm_c[idx]
    up = np.array([0.0, -1.0])
    masks: dict[str, np.ndarray] = {}

    def hull(idx, dil=0.10, shift=None):
        pts = lm_c[idx]
        if shift is not None:
            pts = np.vstack([pts, pts + np.asarray(shift) * iod])
        return _hull_mask(pts, s, dil * iod)

    for side, up_i, lo_i in (("L", BROW_L_UP, BROW_L_LO), ("R", BROW_R_UP, BROW_R_LO)):
        # outer -> inner ordering for L; for R the list is also outer -> inner
        up_pts, lo_pts = lm_c[up_i], lm_c[lo_i]
        thirds = {"out": slice(0, 2), "mid": slice(1, 4), "in": slice(3, 5)}
        for name, sl in thirds.items():
            pts = np.vstack([up_pts[sl], lo_pts[sl]])
            masks[f"brow_{name}_{side}"] = _hull_mask(pts, s, 0.11 * iod)
        # forehead skin above the brow moves with frontalis
        masks[f"fh_{side}"] = _hull_mask(np.vstack([up_pts, up_pts + up * 0.62 * iod]), s, 0.05 * iod)

    masks["glabella"] = _hull_mask(lm_c[[9, 8, 107, 336, 55, 285]], s, 0.07 * iod)
    masks["nose_root"] = _hull_mask(lm_c[[168, 6, 197, 122, 351, 193, 417]], s, 0.05 * iod)

    for side, up_r, lo_r, brow_lo in (("L", EYE_L_UPPER, EYE_L_LOWER, BROW_L_LO),
                                       ("R", EYE_R_UPPER, EYE_R_LOWER, BROW_R_LO)):
        upper = lm_c[up_r]
        lower = lm_c[lo_r]
        masks[f"lid_up_{side}"] = _hull_mask(np.vstack([upper, upper + up * 0.30 * iod]), s, 0.03 * iod)
        masks[f"lid_lo_{side}"] = _hull_mask(np.vstack([lower, lower - up * 0.30 * iod]), s, 0.03 * iod)
    outer = {"L": 33, "R": 263}
    for side in SIDES:
        sign = -1.0 if side == "L" else 1.0
        centre = lm_c[outer[side]] + np.array([sign * 0.22 * iod, 0.0])
        masks[f"canthus_{side}"] = _disc(centre, 0.24 * iod, s)

    masks["wing_L"] = _hull_mask(lm_c[[49, 64, 98, 129, 48]], s, 0.10 * iod)
    masks["wing_R"] = _hull_mask(lm_c[[279, 294, 327, 358, 278]], s, 0.10 * iod)
    masks["cheek_L"] = _hull_mask(lm_c[CHEEK_L], s, 0.02 * iod)
    masks["cheek_R"] = _hull_mask(lm_c[CHEEK_R], s, 0.02 * iod)
    masks["nasolab_L"] = _hull_mask(lm_c[[129, 203, 206, 216, 57, 186, 98]], s, 0.10 * iod)
    masks["nasolab_R"] = _hull_mask(lm_c[[358, 423, 426, 436, 287, 410, 327]], s, 0.10 * iod)
    masks["lip_up"] = _hull_mask(lm_c[[164, 167, 393, 40, 39, 37, 0, 267, 269, 270, 13]], s, 0.06 * iod)
    masks["lip_lo"] = _hull_mask(lm_c[[14, 87, 317, 84, 17, 314, 181, 405, 18]], s, 0.06 * iod)
    masks["corner_L"] = _disc(lm_c[61], 0.26 * iod, s)
    masks["corner_R"] = _disc(lm_c[291], 0.26 * iod, s)
    masks["chin"] = _hull_mask(lm_c[[200, 199, 175, 152, 148, 377, 176, 400, 421, 201, 18]], s, 0.04 * iod)

    # ROIs never overlap the eyeball opening itself (blinks would swamp them): only lid skin
    eye_open = np.zeros((s, s), bool)
    for ring in (EYE_L_RING, EYE_R_RING):
        eye_open |= _hull_mask(lm_c[ring] * 1.0, s, 0)
    mouth_open = _hull_mask(lm_c[[78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13,
                                  82, 81, 80, 191]], s, 0)
    for name in masks:
        if name.startswith(("lid_", "brow_", "canthus_", "cheek_")):
            masks[name] &= ~eye_open
        if name in ("lip_up", "lip_lo", "corner_L", "corner_R"):
            masks[name] &= ~mouth_open
    return masks


def stable_mask(lm_c: np.ndarray, canon: Canon) -> np.ndarray:
    """Pixels used to estimate (and remove) residual head motion."""
    s, iod = canon.size, canon.iod
    m = np.zeros((s, s), bool)
    m |= _hull_mask(lm_c[[168, 6, 197, 195, 5, 4]], s, 0.10 * iod)
    m |= _hull_mask(lm_c[[127, 162, 21, 54, 103, 109, 108, 69, 68, 71]], s, 0.0)
    m |= _hull_mask(lm_c[[356, 389, 251, 284, 332, 338, 337, 299, 298, 301]], s, 0.0)
    return m


def face_oval_mask(lm_c: np.ndarray, canon: Canon) -> np.ndarray:
    m = np.zeros((canon.size, canon.size), np.uint8)
    cv2.fillPoly(m, [np.round(lm_c[FACE_OVAL]).astype(np.int32)], 1)
    return m.astype(bool)
