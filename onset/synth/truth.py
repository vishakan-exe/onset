"""Exact ground truth for the stabilised (canonical) frame of a rendered phantom clip.

The renderer builds frame t as  img(x) = base( p_t(q) )  with  q = H_t^{-1}(x)  (rigid head motion H_t) and
p_t(q) = q - D_t(q - D_t(q))  (two fixed-point steps of the skin displacement D_t).  So the *material*
(base) point that is displayed at head-free position q is  beta = p_t(q).  The flow between frames a and
b that an ideal tracker would report at head-free position q of frame a is therefore

        u(q) = p_b^{-1}( p_a(q) ) - q ,

where p_b^{-1} is found by fixed-point iteration.  This is converted to canonical pixels with the linear
part of the registration transform of the reference frame (the canonical grid is the head-free grid up
to that similarity).
"""
from __future__ import annotations

import cv2
import numpy as np

from .render import COARSE, Renderer, SyntheticClip


class CanonicalTruth:
    def __init__(self, rend: Renderer, clip: SyntheticClip, reg):
        self.rend, self.clip, self.reg = rend, clip, reg
        ph = rend.ph
        self.size = rend.sc.size
        self.S = reg.canon.size
        iod = ph.iod
        c = np.array([ph.cx, ph.cy + 0.5 * iod])
        i = reg.ref_index
        tx, ty, rot, s1 = clip.truth["head"][i]
        t = np.array([tx * iod, ty * iod])
        s = 1.0 + s1
        Rm = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        M = reg.M[i]
        A, b = M[:, :2], M[:, 2]
        gy, gx = np.mgrid[0:self.S, 0:self.S].astype(np.float64)
        pts = np.stack([gx.ravel(), gy.ravel()], 1) - b
        x = pts @ np.linalg.inv(A).T                    # source px at the reference frame
        q = ((x - c - t) / s) @ Rm + c                  # R^-1 (x - c - t) / s + c
        self.qx = q[:, 0].reshape(self.S, self.S).astype(np.float32)
        self.qy = q[:, 1].reshape(self.S, self.S).astype(np.float32)
        self.Lc = A @ (s * Rm)                          # canonical displacement = Lc @ (base displacement)

    # -- fields ------------------------------------------------------------------
    def _coarse(self, t: float) -> np.ndarray:
        return self.rend.coarse_field_at(t, self.rend.activations_at(t))

    @staticmethod
    def _sample(dc: np.ndarray, qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
        return cv2.remap(dc, ((qx + 0.5) / COARSE - 0.5).astype(np.float32), ((qy + 0.5) / COARSE - 0.5).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _p(self, dc, qx, qy):
        d1 = self._sample(dc, qx, qy)
        d2 = self._sample(dc, qx - d1[..., 0], qy - d1[..., 1])
        return qx - d2[..., 0], qy - d2[..., 1]

    def _p_inverse(self, dc, bx, by, iters: int = 12):
        qx, qy = bx.copy(), by.copy()
        for _ in range(iters):
            px, py = self._p(dc, qx, qy)
            qx, qy = qx + (bx - px), qy + (by - py)
        return qx, qy

    # -- public -------------------------------------------------------------------
    def pair_flow(self, ia: int, ib: int) -> np.ndarray:
        """Exact canonical-pixel flow from frame ia to frame ib, defined on frame ia's canonical grid."""
        ta, tb = float(self.clip.times[ia]), float(self.clip.times[ib])
        dca, dcb = self._coarse(ta), self._coarse(tb)
        bx, by = self._p(dca, self.qx, self.qy)
        qx2, qy2 = self._p_inverse(dcb, bx, by)
        u = np.stack([qx2 - self.qx, qy2 - self.qy], -1)
        return (u.reshape(-1, 2) @ self.Lc.T).reshape(self.S, self.S, 2).astype(np.float32)

    def roi_mean(self, flow: np.ndarray, A, conf: np.ndarray) -> np.ndarray:
        cw = np.maximum(conf, 0).ravel().astype(np.float64)
        den = A @ cw
        return np.stack([(A @ (cw * flow[..., 0].ravel())) / np.maximum(den, 1e-9),
                         (A @ (cw * flow[..., 1].ravel())) / np.maximum(den, 1e-9)], 1)


def exact_events(rend, clip, det, events) -> list[dict]:
    """The exact displacement pattern of every scripted event, in the instrument's own units.

    ``events``: scripted movements as dicts with ``aus`` (AU numbers, or {AU: weight}), ``t_on``, ``t_apex``, ``kind`` and
    optionally the generator's own amplitude (``amp`` or ``amp_iod``).

    For each scripted movement the true skin flow from a quiet frame (0.1 s before onset) to the apex is taken from the
    renderer (``CanonicalTruth``), the residual rigid motion is removed with the same robust fit the measurement chain
    uses, and the regions are averaged with the texture weights the chain gets on that very pair.  The result is what a
    perfect measurement would have reported.  It is then decomposed on the true AUs' templates (non-negative least
    squares, after projecting out the head-motion directions the engine also treats as nuisance), which gives the
    amplitude of each AU *in the unit the instrument reports* (the dominant region of the template moves by ``a`` px),
    so that intervals and sensitivity figures can be checked against truth rather than against the generator's own
    parameters.  ``fit_r2`` says how much of the true pattern the hand-written templates can express at all."""
    from scipy.optimize import nnls
    from ..analysis import templates as TPL
    from ..vision.measure import similarity_field, similarity_params

    ct = CanonicalTruth(rend, clip, det.reg)
    meas = det.meas
    times = np.asarray(det.times, float)
    T = TPL.au_templates()
    N = TPL.nuisance_columns(det.cent, meas.centre).reshape(len(TPL.nuisance_columns(det.cent, meas.centre)), -1).T
    Q, _ = np.linalg.qr(N)
    proj = lambda v: v - Q @ (Q.T @ v)                           # noqa: E731
    iod = det.reg.canon.iod
    out = []
    for e in events:
        ia = int(np.argmin(np.abs(times - max(e["t_on"] - 0.10, times[0]))))
        ib = int(np.argmin(np.abs(times - e["t_apex"])))
        pat_hat, _, _ = meas.pair(ia, ib)                        # what the chain reports for exactly this pair
        _, conf = meas.dense(ia, ib)
        u = ct.pair_flow(ia, ib)
        w = meas._fit_weights_for(ia, ib) * np.maximum(conf, 0)
        u = u - similarity_field(similarity_params(u, w, meas.centre), u.shape[:2], meas.centre)
        pat = ct.roi_mean(u, meas.A, conf)                       # (R, 2) canonical px
        aus = sorted(int(a) for a in e["aus"])
        X = np.column_stack([proj(T[TPL.AUS.index(a)].reshape(-1)) for a in aus])
        y = proj(pat.reshape(-1))
        coef, rn = nnls(X, y)
        denom = float((y ** 2).sum())
        gain = float((pat_hat * pat).sum() / max((pat ** 2).sum(), 1e-12))
        out.append(dict(aus=aus, ref_frame=ia, apex_frame=ib, t_ref=float(times[ia]), t_apex=float(times[ib]),
                        pattern=pat.astype(np.float32), pattern_measured=np.asarray(pat_hat, np.float32),
                        a_true_px={a: float(c) for a, c in zip(aus, coef)}, a_true_pct={a: float(c / iod * 100) for a, c in zip(aus, coef)},
                        size_px=float(np.linalg.norm(pat, axis=1).max()), size_pct=float(np.linalg.norm(pat, axis=1).max() / iod * 100),
                        fit_r2=float(1.0 - rn ** 2 / denom) if denom > 0 else float("nan"), pair_gain=gain, kind=e["kind"], amp_generator=float(e.get("amp", e.get("amp_iod", float("nan"))))))
    return out
