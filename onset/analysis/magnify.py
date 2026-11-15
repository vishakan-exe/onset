"""Lagrangian motion magnification with flow shrinkage, and the motion-heat overlay.

The reference frame is *warped* by (1 + alpha) times the measured displacement field, so the texture and the
sensor noise of the reference are kept (nothing is amplified except the measured motion), and displacement
vectors that are not clearly above the noise are shrunk to zero first, so noise-level jitter is not
magnified into fake movement.  Magnification is a *visual aid*; every number in the reports comes from the
measurements, not from the magnified pictures.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..vision.measure import Measurer


def shrink_flow(u: np.ndarray, oval: Optional[np.ndarray] = None, smooth_sigma: float = 2.5, k_noise: float = 3.0,
                floor_px: float = 0.05) -> tuple[np.ndarray, float]:
    """Smooth the flow and shrink vectors toward zero by a James-Stein-style factor  max(0, 1 - (tau / |u|)^2)
    with tau = k_noise x (robust noise level of the smoothed flow over the face).  Returns (flow, tau)."""
    us = np.stack([cv2.GaussianBlur(u[..., 0], (0, 0), smooth_sigma), cv2.GaussianBlur(u[..., 1], (0, 0), smooth_sigma)], -1)
    mag = np.linalg.norm(us, axis=-1)
    region = mag[oval] if oval is not None and oval.any() else mag.ravel()
    noise = 1.4826 * float(np.median(np.abs(region - np.median(region))))
    tau = max(k_noise * noise, floor_px)
    fac = np.maximum(0.0, 1.0 - (tau / np.maximum(mag, 1e-6)) ** 2)
    return us * fac[..., None], float(tau)


def magnify_frame(ref_gray: np.ndarray, u: np.ndarray, alpha: float) -> np.ndarray:
    """I_mag(y) = I_ref(y - (1 + alpha) u(y)): the reference warped by the amplified motion (u: ref -> frame)."""
    h, w = ref_gray.shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = gx - (1.0 + alpha) * u[..., 0]
    my = gy - (1.0 + alpha) * u[..., 1]
    return cv2.remap(ref_gray, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def motion_heat(u: np.ndarray, oval: Optional[np.ndarray], tau: float, vmax: float) -> np.ndarray:
    """Alpha map 0..1 of the (shrunk, smoothed) motion magnitude; zero below the noise threshold."""
    mag = np.linalg.norm(u, axis=-1)
    a = np.clip((mag - 0.5 * tau) / max(vmax - 0.5 * tau, 1e-3), 0.0, 1.0)
    if oval is not None:
        a = a * cv2.GaussianBlur(oval.astype(np.float32), (0, 0), 6)
    return a


def plate_box(canon_size: int, eye_y: float, iod: float) -> tuple[int, int, int, int]:
    """Crop (x0, y0, x1, y1) of the canonical frame shown to the user: the face, a little hair, a little neck."""
    x0 = int(round(canon_size / 2 - 1.5 * iod))
    x1 = int(round(canon_size / 2 + 1.5 * iod))
    y0 = int(round(eye_y - 1.1 * iod))
    y1 = int(round(eye_y + 1.95 * iod))
    return max(0, x0), max(0, y0), min(canon_size, x1), min(canon_size, y1)


def crop_resize(img: np.ndarray, box: tuple, size: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    c = img[y0:y1, x0:x1]
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA if c.shape[0] > size else cv2.INTER_CUBIC)


def magnify_event(meas: Measurer, ref: int, frames: list[int], alpha: float, box: tuple, size: int,
                  apex_frame: int) -> dict:
    """Magnified, plate-cropped frames for an event window plus the motion heat at the apex."""
    reg = meas.reg
    ref_img = reg.gray[ref]
    out = []
    tau_apex, heat = 0.0, None
    us = {}
    for f in frames:
        if f == ref:
            u = np.zeros(ref_img.shape + (2,), np.float32)
            tau = 0.0
        else:
            u_raw, _ = meas.dense(ref, f)
            u, tau = shrink_flow(u_raw, reg.oval)
        us[f] = (u, tau)
        out.append(crop_resize(magnify_frame(ref_img, u, alpha), box, size))
    u_ap, tau_ap = us.get(apex_frame, (None, 0.0))
    if u_ap is not None:
        vmax = float(np.percentile(np.linalg.norm(u_ap, axis=-1)[reg.oval], 99.5)) if reg.oval.any() else 1.0
        heat = crop_resize(motion_heat(u_ap, reg.oval, tau_ap, max(vmax, tau_ap * 1.5 + 0.2)), box, size)
    return dict(frames=np.stack(out), heat=heat, tau=tau_ap)
