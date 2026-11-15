"""Dense optical flow for sub-pixel facial motion.

Two estimators share one interface ``flow(a, b) -> (H, W, 2)`` giving the displacement of image
content from ``a`` to ``b`` in pixels (x, y):

* :class:`LKFlow` - iterative, pyramidal, Gaussian-windowed Lucas-Kanade in float32.  Built for
  *tiny* motions: no 8-bit quantisation, symmetric gradients, no smoothness prior that would
  shrink sub-pixel vectors.  Also returns a per-pixel texture confidence.
* :class:`DISFlow` - OpenCV's Dense Inverse Search, faster but 8-bit and regularised.

``benchmark`` (scripts/benchmark_flow.py) measures both against known synthetic warps so the
choice of estimator is evidence-based and re-checkable.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _to_float_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def _box_gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur approximated by three box passes (O(1) per pixel, ~5x faster than a big kernel)."""
    w = int(round(np.sqrt(4.0 * sigma * sigma + 1.0)))
    w += 1 - (w % 2)
    w = max(w, 3)
    out = cv2.blur(img, (w, w))
    out = cv2.blur(out, (w, w))
    return cv2.blur(out, (w, w))


class LKFlow:
    """Dense inverse-compositional Lucas-Kanade, coarse-to-fine, float32 throughout.

    The structure tensor is computed once per *reference* frame (``prepare``) and reused for every
    pair in which that frame is the reference, so each extra pair costs only a few warps and
    window sums.  Windows are Gaussian (box-approximated); a tiny ridge keeps flat skin finite
    without shrinking real motion.
    """

    def __init__(self, win_sigma: float = 6.0, levels: int = 3, iterations=(2, 2, 2),
                 prefilter_sigma: float = 0.9, ridge: float = 0.3, interp: int = cv2.INTER_CUBIC):
        self.win_sigma = win_sigma
        self.levels = levels
        its = [iterations] * levels if isinstance(iterations, int) else list(iterations)
        # iterations listed fine -> coarse
        self.iterations = (its + [its[-1]] * levels)[:levels]
        self.prefilter_sigma = prefilter_sigma
        self.ridge = ridge
        self.interp = interp
        self._grid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _grid(self, h: int, w: int):
        key = (h, w)
        if key not in self._grid_cache:
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._grid_cache[key] = (xs, ys)
        return self._grid_cache[key]

    def prepare(self, img: np.ndarray, reference: bool = True) -> list[dict]:
        """Pre-filtered pyramid; with ``reference=True`` also the per-level structure tensor."""
        g = _to_float_gray(img)
        if self.prefilter_sigma > 0:
            g = cv2.GaussianBlur(g, (0, 0), self.prefilter_sigma)
        levels = []
        cur = g
        for lvl in range(self.levels):
            lv = {"img": cur}
            if reference:
                gx = cv2.Sobel(cur, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8.0)
                gy = cv2.Sobel(cur, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8.0)
                sig = max(self.win_sigma / (2 ** lvl), 2.0)
                sxx = _box_gauss(gx * gx, sig) + self.ridge
                syy = _box_gauss(gy * gy, sig) + self.ridge
                sxy = _box_gauss(gx * gy, sig)
                det = np.maximum(sxx * syy - sxy * sxy, 1e-6)
                lv.update(gx=gx, gy=gy, sxx=sxx, syy=syy, sxy=sxy, det=det, sig=sig)
            levels.append(lv)
            if lvl < self.levels - 1:
                cur = cv2.pyrDown(cur)
        return levels

    def flow_prepared(self, pa: list[dict], pb: list[dict], return_confidence: bool = False):
        u = None
        for lvl in range(self.levels - 1, -1, -1):
            la, b = pa[lvl], pb[lvl]["img"]
            a = la["img"]
            h, w = a.shape
            if u is None:
                u = np.zeros((h, w, 2), np.float32)
            else:
                u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
            xs, ys = self._grid(h, w)
            for _ in range(self.iterations[lvl]):
                bw = cv2.remap(b, xs + u[..., 0], ys + u[..., 1], self.interp, borderMode=cv2.BORDER_REPLICATE)
                r = a - bw
                bx = _box_gauss(la["gx"] * r, la["sig"])
                by = _box_gauss(la["gy"] * r, la["sig"])
                du = (la["syy"] * bx - la["sxy"] * by) / la["det"]
                dv = (la["sxx"] * by - la["sxy"] * bx) / la["det"]
                np.clip(du, -2.0, 2.0, out=du)
                np.clip(dv, -2.0, 2.0, out=dv)
                u[..., 0] += du
                u[..., 1] += dv
        if return_confidence:
            la = pa[0]
            tr = (la["sxx"] - self.ridge) + (la["syy"] - self.ridge)
            dt = (la["sxx"] - self.ridge) * (la["syy"] - self.ridge) - la["sxy"] ** 2
            conf = 0.5 * (tr - np.sqrt(np.maximum(tr * tr - 4 * dt, 0)))
            return u, conf
        return u

    def __call__(self, a: np.ndarray, b: np.ndarray, return_confidence: bool = False):
        return self.flow_prepared(self.prepare(a, True), self.prepare(b, False), return_confidence)


class DISFlow:
    def __init__(self, preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM, refine: int = 5):
        self.dis = cv2.DISOpticalFlow_create(preset)
        self.dis.setVariationalRefinementIterations(refine)
        self.dis.setUseSpatialPropagation(True)

    def prepare(self, img: np.ndarray):
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return g.astype(np.uint8)

    def flow_prepared(self, a, b, return_confidence: bool = False):
        f = self.dis.calc(a, b, None)
        return (f, None) if return_confidence else f

    def __call__(self, a, b, return_confidence: bool = False):
        return self.flow_prepared(self.prepare(a), self.prepare(b), return_confidence)


class FarnebackFlow:
    def __init__(self):
        self.kw = dict(pyr_scale=0.5, levels=3, winsize=21, iterations=4, poly_n=7, poly_sigma=1.5, flags=0)

    def prepare(self, img):
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return g.astype(np.uint8)

    def flow_prepared(self, a, b, return_confidence: bool = False):
        f = cv2.calcOpticalFlowFarneback(a, b, None, **self.kw)
        return (f, None) if return_confidence else f

    def __call__(self, a, b, return_confidence: bool = False):
        return self.flow_prepared(self.prepare(a), self.prepare(b), return_confidence)


# --------------------------------------------------------------------------------------
# global (rigid) motion removal
# --------------------------------------------------------------------------------------
def fit_similarity_flow(flow: np.ndarray, weights: np.ndarray, centre: tuple[float, float],
                        iters: int = 4) -> np.ndarray:
    """Robust (Tukey IRLS) fit of v(x) = t + [[a,-b],[b,a]] (x - c) to a flow field.

    ``weights`` is a prior weight map (0 outside the fitting region).  Returns the *model field*
    (H, W, 2) which the caller subtracts to leave non-rigid motion.
    """
    h, w = weights.shape
    ys, xs = np.nonzero(weights > 0)
    if len(xs) < 50:
        return np.zeros_like(flow)
    px = xs.astype(np.float64) - centre[0]
    py = ys.astype(np.float64) - centre[1]
    wt = weights[ys, xs].astype(np.float64)
    vx = flow[ys, xs, 0].astype(np.float64)
    vy = flow[ys, xs, 1].astype(np.float64)
    # unknowns (tx, ty, a, b): vx = tx + a*px - b*py ; vy = ty + b*px + a*py
    A = np.zeros((2 * len(xs), 4))
    A[0::2, 0] = 1.0
    A[0::2, 2] = px
    A[0::2, 3] = -py
    A[1::2, 1] = 1.0
    A[1::2, 2] = py
    A[1::2, 3] = px
    y = np.empty(2 * len(xs))
    y[0::2], y[1::2] = vx, vy
    wr = np.repeat(wt, 2)
    robust = np.ones_like(wr)
    p = np.zeros(4)
    for _ in range(iters):
        sw = np.sqrt(wr * robust)
        p, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        res = y - A @ p
        scale = 1.4826 * np.median(np.abs(res)) + 1e-4
        t = np.clip(np.abs(res) / (4.685 * scale), 0, 1)
        robust = (1 - t * t) ** 2
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32) - centre[0], np.arange(h, dtype=np.float32) - centre[1])
    out = np.empty_like(flow)
    out[..., 0] = p[0] + p[2] * gx - p[3] * gy
    out[..., 1] = p[1] + p[3] * gx + p[2] * gy
    return out
