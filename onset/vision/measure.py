"""ROI-level motion measurement on stabilised crops.

For a lag ``k`` the dense flow between frame ``j`` and ``j+k`` is computed, residual head motion is
fitted on rigid facial pixels and subtracted, and the remaining motion is averaged inside each ROI
(weighted by local texture, so flat skin does not drag the mean towards zero).

Spotting uses the *transient*  T_i = x_i - (x_{i-k} + x_{i+k}) / 2  of the skin position x:  a
movement shorter than ~2k leaves a response of nearly its full amplitude at its apex, while slow
drifts (breathing, posture) cancel.  It is obtained from the pair flows as T_i = (F[i-k] - F[i]) / 2.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
import scipy.sparse as sp

from . import geometry as G
from .flow import LKFlow
from .register import Registration


def similarity_params(flow: np.ndarray, weights: np.ndarray, centre, iters: int = 4) -> np.ndarray:
    """Robust (Tukey IRLS) fit of v(x) = t + [[a,-b],[b,a]](x-c) on weighted pixels -> (tx, ty, a, b)."""
    ys, xs = np.nonzero(weights > 0)
    if len(xs) < 50:
        return np.zeros(4)
    px = xs.astype(np.float64) - centre[0]
    py = ys.astype(np.float64) - centre[1]
    wt = weights[ys, xs].astype(np.float64)
    vx = flow[ys, xs, 0].astype(np.float64)
    vy = flow[ys, xs, 1].astype(np.float64)
    m = len(xs)
    A = np.zeros((2 * m, 4))
    A[0::2, 0] = 1.0
    A[0::2, 2] = px
    A[0::2, 3] = -py
    A[1::2, 1] = 1.0
    A[1::2, 2] = py
    A[1::2, 3] = px
    y = np.empty(2 * m)
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
    return p


def similarity_field(p: np.ndarray, shape, centre) -> np.ndarray:
    h, w = shape
    gx = np.arange(w, dtype=np.float32) - centre[0]
    gy = np.arange(h, dtype=np.float32) - centre[1]
    out = np.empty((h, w, 2), np.float32)
    out[..., 0] = p[0] + p[2] * gx[None, :] - p[3] * gy[:, None]
    out[..., 1] = p[1] + p[3] * gx[None, :] + p[2] * gy[:, None]
    return out


@dataclass
class LagSeries:
    k: int
    stride: int
    pos: np.ndarray           # (m,) frame index j of the first frame of each pair
    u: np.ndarray             # (m, R, 2)  ROI-mean flow j -> j+k, px (canonical)
    w: np.ndarray             # (m, R)     texture weight (sum of confidence in the ROI)
    glob: np.ndarray          # (m, 4)     removed global similarity (tx, ty, a, b)


class Measurer:
    def __init__(self, reg: Registration, threads: Optional[int] = None, flow_kwargs: Optional[dict] = None):
        self.reg = reg
        self.names = list(G.ROI_NAMES)
        S = reg.canon.size
        self.S = S
        self.centre = (S / 2.0, reg.canon.eye_y + 0.9 * reg.canon.iod)
        rows, cols = [], []
        for r, name in enumerate(self.names):
            idx = np.flatnonzero(reg.rois[name].ravel())
            rows.append(np.full(len(idx), r))
            cols.append(idx)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        self.A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(self.names), S * S))
        self.threads = threads or max(1, min(4, os.cpu_count() or 1))
        self.flow_kwargs = flow_kwargs or {}
        self._local = {}
        # during a blink the flow around the eyes is large and leaks (through the LK window) into the upper
        # nasal bridge; a lower core, well away from the eyes, keeps the head-motion estimate clean then
        self.fit_w_low = self._lower_core(reg)
        self.blink_frames: Optional[np.ndarray] = None

    @staticmethod
    def _lower_core(reg: Registration) -> np.ndarray:
        S, iod = reg.canon.size, reg.canon.iod
        low = G._hull_mask(reg.ref_c[[195, 5, 4]], S, 0.10 * iod)
        for name in ("wing_L", "wing_R", "lip_up"):
            low &= ~reg.rois[name]
        sub = np.zeros((S, S), np.float32)
        sub[::2, ::2] = 1.0
        return low.astype(np.float32) * sub

    def set_blink_frames(self, flags: np.ndarray, pad: int = 1):
        f = np.asarray(flags, bool)
        if pad > 0:
            g = f.copy()
            for d in range(1, pad + 1):
                g[d:] |= f[:-d]
                g[:-d] |= f[d:]
            f = g
        self.blink_frames = f

    def _fit_weights_for(self, a: int, b: int) -> np.ndarray:
        bf = self.blink_frames
        if bf is not None and (bf[a] or bf[b]):
            return self.fit_w_low
        return self.reg.fit_w

    # -- one pair ------------------------------------------------------------------
    def _flow_engine(self) -> LKFlow:
        import threading
        tid = threading.get_ident()
        if tid not in self._local:
            self._local[tid] = LKFlow(**self.flow_kwargs)
        return self._local[tid]

    def pair(self, a: int, b: int, dense: bool = False):
        eng = self._flow_engine()
        pa = eng.prepare(self.reg.gray[a], True)
        pb = eng.prepare(self.reg.gray[b], False)
        u, conf = eng.flow_prepared(pa, pb, return_confidence=True)
        # inverse-variance weights: the flow variance of a pixel scales with 1 / lambda_min(structure tensor)
        p = similarity_params(u, self._fit_weights_for(a, b) * np.maximum(conf, 0), self.centre)
        u = u - similarity_field(p, u.shape[:2], self.centre)
        cw = np.maximum(conf, 0).astype(np.float32).ravel()
        den = self.A @ cw
        ux = (self.A @ (cw * u[..., 0].ravel())) / np.maximum(den, 1e-6)
        uy = (self.A @ (cw * u[..., 1].ravel())) / np.maximum(den, 1e-6)
        out = (np.stack([ux, uy], 1), den, p)
        if dense:
            return out + (u, conf)
        return out

    def dense(self, a: int, b: int):
        """Residual dense flow a -> b (after head-motion removal) and texture confidence."""
        _, _, _, u, conf = self.pair(a, b, dense=True)
        return u, conf

    # -- many pairs ------------------------------------------------------------------
    def lag_series(self, k: int, stride: int = 1, progress: Optional[Callable[[float], None]] = None) -> LagSeries:
        n = self.reg.gray.shape[0]
        pos = np.arange(0, n - k, stride)
        R = len(self.names)
        U = np.zeros((len(pos), R, 2), np.float32)
        W = np.zeros((len(pos), R), np.float32)
        P = np.zeros((len(pos), 4), np.float32)
        old = cv2.getNumThreads()
        if self.threads > 1:
            cv2.setNumThreads(1)
        try:
            def work(ii):
                j = int(pos[ii])
                u, w, p = self.pair(j, j + k)
                return ii, u, w, p
            done = 0
            with ThreadPoolExecutor(self.threads) as ex:
                for ii, u, w, p in ex.map(work, range(len(pos))):
                    U[ii], W[ii], P[ii] = u, w, p
                    done += 1
                    if progress and done % 25 == 0:
                        progress(done / len(pos))
        finally:
            cv2.setNumThreads(old)
        return LagSeries(k, stride, pos, U, W, P)

    @staticmethod
    def transient(ls: LagSeries):
        """T_i = (F[i-k] - F[i]) / 2 at positions i where both pair flows exist.  Returns (idx, T, W)."""
        k, st = ls.k, ls.stride
        index = {int(j): ii for ii, j in enumerate(ls.pos)}
        idx, ia, ib = [], [], []
        for ii, j in enumerate(ls.pos):
            i = int(j) + k          # candidate centre: pair (i-k -> i) is `ii`, pair (i -> i+k) must exist
            if i in index:
                idx.append(i)
                ia.append(ii)
                ib.append(index[i])
        if not idx:
            return np.zeros(0, int), np.zeros((0, ls.u.shape[1], 2), np.float32), np.zeros((0, ls.u.shape[1]), np.float32)
        ia, ib = np.array(ia), np.array(ib)
        T = 0.5 * (ls.u[ia] - ls.u[ib])
        W = np.minimum(ls.w[ia], ls.w[ib])
        return np.array(idx), T, W

    # -- displacement trace relative to a reference frame ---------------------------------
    def trace(self, ref: int, frames: list[int]):
        R = len(self.names)
        out = np.zeros((len(frames), R, 2), np.float32)
        wts = np.zeros((len(frames), R), np.float32)
        glob = np.zeros((len(frames), 4), np.float32)
        old = cv2.getNumThreads()
        if self.threads > 1:
            cv2.setNumThreads(1)
        try:
            def work(ii):
                f = int(frames[ii])
                if f == ref:
                    return ii, np.zeros((R, 2), np.float32), np.ones(R, np.float32), np.zeros(4, np.float32)
                return (ii,) + self.pair(ref, f)
            with ThreadPoolExecutor(self.threads) as ex:
                for ii, u, w, p in ex.map(work, range(len(frames))):
                    out[ii], wts[ii], glob[ii] = u, w, p
        finally:
            cv2.setNumThreads(old)
        return out, wts, glob
