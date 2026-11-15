"""Procedural *phantom* faces.

A phantom is to this instrument what a calibration phantom is to a medical scanner: a synthetic
subject whose motion is known exactly.  It is rendered from an analytic height-field (so the
shading changes plausibly), carries dense skin micro-texture (so optical flow has something to
lock onto) and exposes named anchors in pixel coordinates.  No real person is depicted.

Coordinates inside this file are *face units*: u to the right, v downwards, eye line at v = 0 and
the inter-ocular distance (IOD) equal to 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np


@dataclass
class PhantomSpec:
    name: str = "A"
    seed: int = 1
    skin: tuple = (0.86, 0.67, 0.56)
    hair: tuple = (0.10, 0.07, 0.05)
    iris: tuple = (0.30, 0.20, 0.10)
    lip: tuple = (0.66, 0.34, 0.32)
    brow: tuple = (0.12, 0.08, 0.06)
    face_w: float = 1.0
    face_h: float = 1.0
    brow_h: float = 0.0
    nose_len: float = 1.0
    nose_w: float = 1.0
    mouth_w: float = 1.0
    eye_open: float = 1.0
    hair_style: int = 0          # 0 short, 1 side-swept, 2 none
    light: tuple = (-0.50, -0.55)
    contrast: float = 1.0
    texture: float = 1.0         # strength of skin micro-texture

    def variant(self, **kw) -> "PhantomSpec":
        return replace(self, **kw)


# a small family of distinct phantoms (different skin, structure, lighting)
PHANTOMS = {
    "A": PhantomSpec("A", 11, (0.86, 0.67, 0.56), (0.10, 0.07, 0.05), (0.30, 0.20, 0.10)),
    "B": PhantomSpec("B", 23, (0.62, 0.43, 0.32), (0.05, 0.04, 0.04), (0.18, 0.11, 0.06), (0.52, 0.27, 0.25),
                     (0.07, 0.05, 0.04), face_w=1.05, brow_h=0.02, nose_w=1.12, mouth_w=1.05, light=(0.45, -0.5)),
    "C": PhantomSpec("C", 37, (0.93, 0.78, 0.68), (0.45, 0.32, 0.18), (0.25, 0.38, 0.45), (0.74, 0.40, 0.38),
                     (0.35, 0.25, 0.15), face_w=0.95, face_h=1.04, brow_h=-0.02, nose_len=0.95, hair_style=1,
                     light=(-0.3, -0.7)),
    "D": PhantomSpec("D", 41, (0.45, 0.30, 0.22), (0.03, 0.03, 0.03), (0.12, 0.08, 0.05), (0.40, 0.20, 0.19),
                     (0.04, 0.03, 0.03), face_w=1.08, face_h=0.97, nose_w=1.2, mouth_w=1.1, hair_style=2,
                     light=(-0.6, -0.4), contrast=1.1),
    "E": PhantomSpec("E", 53, (0.80, 0.58, 0.45), (0.22, 0.15, 0.10), (0.20, 0.30, 0.18), (0.62, 0.30, 0.28),
                     (0.20, 0.14, 0.09), face_w=0.97, brow_h=0.03, eye_open=1.08, hair_style=1,
                     light=(0.2, -0.65)),
    "F": PhantomSpec("F", 67, (0.90, 0.72, 0.60), (0.46, 0.45, 0.45), (0.34, 0.28, 0.20), (0.70, 0.40, 0.36),
                     (0.42, 0.41, 0.40), face_w=1.02, face_h=1.03, brow_h=-0.01, eye_open=0.9, hair_style=0,
                     light=(-0.55, -0.45), texture=1.2),
}


def _g(u, v, cu, cv_, su, sv):
    return np.exp(-0.5 * (((u - cu) / su) ** 2 + ((v - cv_) / sv) ** 2))


def _smooth(x, a, b):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _soft(d, w):
    return 1.0 / (1.0 + np.exp(-np.clip(d / w, -30, 30)))


def _seg_dist(u, v, p, q):
    p, q = np.asarray(p, np.float32), np.asarray(q, np.float32)
    d = q - p
    t = np.clip(((u - p[0]) * d[0] + (v - p[1]) * d[1]) / float(d @ d), 0, 1)
    return np.hypot(u - (p[0] + t * d[0]), v - (p[1] + t * d[1]))


class Phantom:
    """Renders the static base images and holds the geometry of one phantom."""

    def __init__(self, spec: PhantomSpec | str = "A", size: int = 480, iod: float = 104.0, ss: int = 2):
        if isinstance(spec, str):
            spec = PHANTOMS[spec]
        self.spec, self.size, self.iod, self.ss = spec, size, float(iod), ss
        self.cx, self.cy = size / 2.0, size * 0.43
        self._build()

    # ------------------------------------------------------------------ geometry
    def uv_to_px(self, u, v):
        return (self.cx + np.asarray(u) * self.iod, self.cy + np.asarray(v) * self.iod)

    def anchors(self) -> dict[str, np.ndarray]:
        s = self.spec
        a: dict[str, tuple[float, float]] = {}
        for side, sg in (("L", -1.0), ("R", 1.0)):
            a[f"eye_{side}"] = (sg * 0.5, 0.0)
            a[f"brow_in_{side}"] = (sg * 0.27, -0.40 + s.brow_h)
            a[f"brow_mid_{side}"] = (sg * 0.62, -0.50 + s.brow_h)
            a[f"brow_out_{side}"] = (sg * 0.98, -0.38 + s.brow_h)
            a[f"lid_{side}"] = (sg * 0.5, -0.07 * s.eye_open)
            a[f"lidlo_{side}"] = (sg * 0.5, 0.07 * s.eye_open)
            a[f"canthus_{side}"] = (sg * 0.80, 0.02)
            a[f"cheek_{side}"] = (sg * 0.78, 0.66)
            a[f"wing_{side}"] = (sg * 0.27 * s.nose_w, 0.82 * s.nose_len)
            a[f"nasolab_{side}"] = (sg * 0.42, 1.05)
            a[f"corner_{side}"] = (sg * 0.46 * s.mouth_w, 1.34)
            a[f"temple_{side}"] = (sg * 1.02, -0.30)
        a["chin_L"] = (-0.22, 1.72 * s.face_h)
        a["chin_R"] = (0.22, 1.72 * s.face_h)
        a["glabella"] = (0.0, -0.35 + s.brow_h)
        a["nose_root"] = (0.0, -0.02)
        a["nose_tip"] = (0.0, 0.80 * s.nose_len)
        a["lip_up"] = (0.0, 1.24)
        a["lip_lo"] = (0.0, 1.43)
        a["chin"] = (0.0, 1.78 * s.face_h)
        a["forehead"] = (0.0, -0.75)
        return {k: np.array(self.uv_to_px(*v), dtype=np.float64) for k, v in a.items()}

    # ------------------------------------------------------------------ rendering
    def _build(self):
        s, ss = self.spec, self.ss
        N = self.size * ss
        rng = np.random.default_rng(s.seed)
        iod = self.iod * ss
        ys, xs = np.mgrid[0:N, 0:N].astype(np.float32)
        u = (xs - self.cx * ss) / iod
        v = (ys - self.cy * ss) / iod
        ax, ay, vc = 1.08 * s.face_w, 1.62 * s.face_h, 0.36

        # ---- silhouettes
        vv = (v - vc) / ay
        hw = ax * np.sqrt(np.clip(1 - vv ** 2, 0, 1)) * (1 - 0.17 * _smooth(v, 0.7, 1.95) ** 1.4)
        face = _soft(hw - np.abs(u), 0.018) * (np.abs(vv) < 1.0)
        hairline = -0.80 + 0.26 * (u / ax) ** 2 + 0.03 * np.cos(u * 5.0)
        if s.hair_style == 1:
            hairline = hairline + 0.12 * _smooth(u, -0.2, 0.9) - 0.05
        cran = ((u / (1.36 * s.face_w)) ** 2 + ((v + 0.38) / 1.78) ** 2) < 1.0
        hair_core = cran & ((v < hairline) | ((np.abs(u) > hw * 0.93) & (v < 0.55)))
        edge_noise = cv2.GaussianBlur(rng.standard_normal((N, N)).astype(np.float32), (0, 0), 0.025 * iod)
        edge_noise /= edge_noise.std() + 1e-9
        hair_core = cran & ((v < hairline + 0.025 * edge_noise) | ((np.abs(u) > hw * 0.93) & (v < 0.55)))
        hair = hair_core.astype(np.float32)
        hair = cv2.GaussianBlur(hair, (0, 0), 0.035 * iod)
        if s.hair_style == 2:
            hair = hair * 0.0
        neck = _soft(0.64 - np.abs(u) * (1 + 0.10 * _smooth(v, 1.9, 2.4)), 0.05) * _smooth(v, 1.0, 1.3)
        shirt = _soft(v - (2.02 + 0.32 * u ** 2), 0.06)
        ear = np.zeros_like(u)
        for sg in (-1, 1):
            ear = np.maximum(ear, _g(u, v, sg * ax * 0.985, 0.30, 0.11, 0.27))

        # ---- height field (units: IOD)
        z = 0.9 * np.sqrt(np.clip(1 - (u / (ax * 0.97)) ** 2 - ((v - vc) / ay) ** 2, 0, 1))
        tipv = 0.80 * s.nose_len
        ridge = _g(u, v * 0 + 0, 0.0, 0.0, 0.12 * s.nose_w, 1.0) * _smooth(v, -0.12, 0.45) * (1 - _smooth(v, tipv - 0.05, tipv + 0.1))
        z += 0.15 * ridge + 0.24 * _g(u, v, 0, tipv, 0.17 * s.nose_w, 0.15) + \
            0.11 * (_g(u, v, -0.27 * s.nose_w, tipv - 0.02, 0.10, 0.09) + _g(u, v, 0.27 * s.nose_w, tipv - 0.02, 0.10, 0.09))
        for sg in (-1, 1):
            z += -0.24 * _g(u, v, sg * 0.50, 0.02, 0.36, 0.27)                          # eye sockets
            z += 0.12 * _g(u, v, sg * 0.60, -0.42 + s.brow_h, 0.42, 0.10)                # brow ridge
            z += 0.15 * _g(u, v, sg * 0.80, 0.64, 0.33, 0.30)                            # cheekbones
            z += -0.06 * _g(u, v, sg * 0.70, 1.08, 0.26, 0.32)                           # cheek hollow
            d = _seg_dist(u, v, (sg * 0.30 * s.nose_w, 0.95), (sg * 0.55 * s.mouth_w, 1.30))
            z += -0.032 * np.exp(-0.5 * (d / 0.05) ** 2)                                # nasolabial fold
        z += 0.07 * _g(u, v, 0, 1.24, 0.30 * s.mouth_w, 0.06) + 0.10 * _g(u, v, 0, 1.43, 0.28 * s.mouth_w, 0.075)
        z += -0.03 * _g(u, v, 0, 1.08, 0.05, 0.12) - 0.06 * _g(u, v, 0, 1.335, 0.26 * s.mouth_w, 0.014) * _soft(0.46 * s.mouth_w - np.abs(u), 0.03)
        z += 0.12 * _g(u, v, 0, 1.80 * s.face_h, 0.30, 0.20) - 0.05 * _g(u, v, 0, 1.63, 0.26, 0.05)
        z += 0.06 * _g(u, v, 0, -0.78, 0.60, 0.30)
        pores = cv2.GaussianBlur(rng.standard_normal((N, N)).astype(np.float32), (0, 0), 0.8 * ss)
        pores /= pores.std() + 1e-9
        z_micro = z + 0.0007 * s.texture * pores

        # ---- shading
        du = 1.0 / iod
        zx = cv2.Sobel(z_micro, cv2.CV_32F, 1, 0, ksize=3, scale=1 / (8 * du))
        zy = cv2.Sobel(z_micro, cv2.CV_32F, 0, 1, ksize=3, scale=1 / (8 * du))
        k = 1.05 * s.contrast
        gm = np.sqrt(zx * zx + zy * zy) + 1e-6
        comp = np.tanh(k * gm / 1.0) / (k * gm)       # compress steep slopes (no dark ring at the silhouette)
        nx, ny, nz = -k * zx * comp, -k * zy * comp, np.ones_like(zx)
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nn, ny / nn, nz / nn
        lx, ly = s.light
        lz = 0.78
        ln = np.sqrt(lx * lx + ly * ly + lz * lz)
        lam = np.clip((nx * lx + ny * ly + nz * lz) / ln, 0, 1)
        hx, hy, hz = lx / ln, ly / ln, (lz / ln + 1.0)
        hn = np.sqrt(hx * hx + hy * hy + hz * hz)
        spec = np.clip((nx * hx + ny * hy + nz * hz) / hn, 0, 1) ** 28
        conc = np.clip(z - cv2.GaussianBlur(z, (0, 0), 0.22 * iod), -0.2, 0.0)
        ao = np.clip(1.0 + 3.2 * conc, 0.35, 1.0)
        shade = (0.46 + 0.70 * lam) * ao

        # ---- albedo
        skin = np.array(s.skin, np.float32)
        alb = np.empty((N, N, 3), np.float32)
        alb[:] = skin
        blot = cv2.GaussianBlur(rng.standard_normal((N, N)).astype(np.float32), (0, 0), 26 * ss)
        blot /= blot.std() + 1e-9
        mid = cv2.GaussianBlur(rng.standard_normal((N, N)).astype(np.float32), (0, 0), 3.6 * ss)
        mid /= mid.std() + 1e-9
        tint = np.array([0.09, -0.02, -0.04], np.float32)
        alb += (blot[..., None] * 0.035 * tint * 4 + mid[..., None] * 0.012) * s.texture
        alb *= (1.0 + 0.022 * s.texture * pores)[..., None]
        blush = sum(_g(u, v, sg * 0.78, 0.98, 0.33, 0.27) for sg in (-1, 1))
        alb += blush[..., None] * np.array([0.10, -0.035, -0.020], np.float32)
        alb += (_g(u, v, 0, tipv, 0.2, 0.15)[..., None] * np.array([0.06, -0.02, -0.02], np.float32))
        under = sum(_g(u, v, sg * 0.5, 0.20, 0.30, 0.07) for sg in (-1, 1))
        alb -= under[..., None] * np.array([0.05, 0.05, 0.03], np.float32)
        # freckles / moles: tiny dark discs that act as optical-flow anchors
        spots = np.zeros((N, N), np.float32)
        for _ in range(int(90 * s.texture)):
            uu, vv_ = rng.uniform(-0.95, 0.95), rng.uniform(-0.7, 1.7)
            r = rng.uniform(0.7, 2.0) * ss
            cv2.circle(spots, (int(self.cx * ss + uu * iod), int(self.cy * ss + vv_ * iod)), int(round(r)),
                       float(rng.uniform(0.06, 0.16)), -1)
        for _ in range(int(7 * s.texture)):
            uu, vv_ = rng.uniform(-0.8, 0.8), rng.uniform(-0.2, 1.3)
            cv2.circle(spots, (int(self.cx * ss + uu * iod), int(self.cy * ss + vv_ * iod)),
                       int(round(rng.uniform(1.8, 3.2) * ss)), float(rng.uniform(0.2, 0.32)), -1)
        spots = cv2.GaussianBlur(spots, (0, 0), 0.7 * ss)
        alb *= (1.0 - spots)[..., None]

        # ---- lips
        wm = 0.46 * s.mouth_w
        vm = 1.335 + 0.02 * (u / wm) ** 2
        vt = 1.215 + 0.04 * (np.abs(u) / wm) ** 2 - 0.022 * np.exp(-(u / 0.06) ** 2)
        vb = vm + 0.175 * np.sqrt(np.clip(1 - (u / (0.9 * wm)) ** 2, 0, 1))
        lips = _soft(wm - np.abs(u), 0.01) * _soft(v - vt, 0.008) * _soft(vb - v, 0.012)
        lipc = np.array(s.lip, np.float32)
        alb = alb * (1 - lips[..., None]) + (lipc[None, None, :] * (0.92 + 0.12 * mid[..., None])) * lips[..., None]
        mouth_line = np.exp(-0.5 * ((v - vm) / 0.011) ** 2) * _soft(wm * 1.02 - np.abs(u), 0.02)
        alb *= (1 - 0.55 * mouth_line)[..., None]
        for sg in (-1, 1):  # nostrils
            alb *= (1 - 0.65 * _g(u, v, sg * 0.15 * s.nose_w, tipv + 0.05, 0.045, 0.028))[..., None]

        # ---- brows
        brow_col = np.array(s.brow, np.float32)
        brow_mask = np.zeros((N, N), np.float32)
        t = np.linspace(0, 1, 60, dtype=np.float32)
        for sg in (-1, 1):
            bu = sg * (0.25 + 0.74 * t)
            bv = (-0.40 + s.brow_h) - 0.115 * np.sin(np.pi * np.clip(t * 0.88, 0, 1)) + 0.10 * t ** 2
            th = 0.056 - 0.030 * t
            pts = np.stack([self.cx * ss + bu * iod, self.cy * ss + bv * iod], 1)
            for i in range(len(t) - 1):
                cv2.line(brow_mask, tuple(np.round(pts[i]).astype(int)), tuple(np.round(pts[i + 1]).astype(int)),
                         0.95, max(int(round(2 * th[i] * iod)), 1), lineType=cv2.LINE_AA)
        strands = cv2.GaussianBlur(rng.standard_normal((N, N)).astype(np.float32), (0, 0), 0.9 * ss)
        strands /= strands.std() + 1e-9
        brow_mask = cv2.GaussianBlur(brow_mask, (0, 0), 0.7 * ss) * np.clip(0.8 + 0.25 * strands, 0, 1.2)
        brow_mask = np.clip(brow_mask, 0, 1)
        alb = alb * (1 - brow_mask[..., None]) + brow_col * brow_mask[..., None] * (0.7 + 0.5 * shade[..., None])

        # ---- compose body (skin) ; hair ; ears ; neck ; shirt ; background
        img_open = self._compose(u, v, alb, shade, spec, lam, hair, face, neck, ear, shirt, rng, ss, brow_mask,
                                 eyes="open")
        img_closed = self._compose(u, v, alb, shade, spec, lam, hair, face, neck, ear, shirt, rng, ss, brow_mask,
                                   eyes="closed")
        self.base_open = cv2.resize(img_open, (self.size, self.size), interpolation=cv2.INTER_AREA)
        self.base_closed = cv2.resize(img_closed, (self.size, self.size), interpolation=cv2.INTER_AREA)
        ey = np.mgrid[0:self.size, 0:self.size].astype(np.float32)
        uu = (ey[1] - self.cx) / self.iod
        vv2 = (ey[0] - self.cy) / self.iod
        self.eye_region = np.zeros((self.size, self.size), np.float32)
        for sg in (-1, 1):
            self.eye_region = np.maximum(self.eye_region, _g(uu, vv2, sg * 0.5, 0.0, 0.34, 0.19))

    def _eye_layers(self, u, v, eyes):
        s = self.spec
        out_rgb = np.zeros(u.shape + (3,), np.float32)
        out_a = np.zeros(u.shape, np.float32)
        a = 0.255
        for sg in (-1, 1):
            uu, vv = u - sg * 0.5, v
            e = np.clip(1 - (uu / a) ** 2, 0, 1)
            hu, hl = 0.108 * s.eye_open, 0.068 * s.eye_open
            # almond opening: upper edge a little steeper nasally
            up_edge = -hu * e ** 0.85
            lo_edge = hl * e ** 0.9
            opening = _soft(a - np.abs(uu), 0.006) * _soft(vv - up_edge, 0.006) * _soft(lo_edge - vv, 0.006)
            if eyes == "closed":
                continue
            sclera = np.array([0.90, 0.88, 0.86], np.float32) * (1 - 0.40 * (np.abs(uu) / a) ** 2.2)[..., None]
            sclera[..., 0] += 0.02
            r = np.hypot(uu, vv + 0.012)
            ri, rp = 0.158 * (0.98 + 0.04 * s.eye_open), 0.066
            iris_t = np.clip((ri - r) / 0.012, 0, 1)
            ang = np.arctan2(vv + 0.012, uu)
            rad = 0.78 + 0.22 * np.cos(ang * 17 + 3 * r / ri * np.pi)
            ic = np.array(s.iris, np.float32)
            iris_col = ic[None, None, :] * (0.55 + 0.75 * (r / ri) ** 1.5 * rad)[..., None]
            iris_col *= (1 - 0.55 * _smooth(r, ri * 0.82, ri))[..., None]
            rgb = sclera * (1 - iris_t[..., None]) + iris_col * iris_t[..., None]
            pup = np.clip((rp - r) / 0.01, 0, 1)
            rgb = rgb * (1 - pup[..., None]) + np.array([0.02, 0.02, 0.02], np.float32) * pup[..., None]
            hl_ = _g(uu, vv, -0.045, -0.045, 0.018, 0.018)
            rgb = rgb + hl_[..., None] * 0.9
            lid_shadow = 1 - 0.65 * np.exp(-0.5 * (np.clip(vv - up_edge, 0, None) / 0.035) ** 2)
            rgb *= lid_shadow[..., None]
            inner = _g(uu, vv, -sg * a * 0.92, 0.01, 0.05, 0.04)
            rgb = rgb * (1 - 0.5 * inner[..., None]) + np.array([0.72, 0.42, 0.40], np.float32) * 0.5 * inner[..., None]
            out_rgb = out_rgb * (1 - opening[..., None]) + rgb * opening[..., None]
            out_a = np.maximum(out_a, opening)
        return out_rgb, out_a

    def _compose(self, u, v, alb, shade, spec, lam, hair, face, neck, ear, shirt, rng, ss, brow_mask, eyes):
        s = self.spec
        N = u.shape[0]
        skin_lit = alb * shade[..., None]
        # background
        bg = np.empty((N, N, 3), np.float32)
        r2 = u ** 2 + (v - 0.3) ** 2
        bgv = 0.075 + 0.075 * np.exp(-r2 / 2.4)
        bg[:] = np.stack([bgv * 1.02, bgv * 0.99, bgv * 0.94], -1)
        img = bg
        # shirt
        fabric = cv2.GaussianBlur(rng.standard_normal((N, N)).astype(np.float32), (0, 0), 0.9 * ss)
        fcol = np.array([0.14, 0.145, 0.16], np.float32)[None, None, :] * (1 + 0.10 * fabric[..., None])
        img = img * (1 - shirt[..., None]) + fcol * (0.65 + 0.5 * lam[..., None]) * shirt[..., None]
        # neck (shadowed beneath the jaw)
        neck_shade = (0.62 - 0.30 * np.exp(-((v - 1.95) / 0.28) ** 2))[..., None]
        neck_col = alb * (0.55 + 0.55 * lam[..., None]) * neck_shade
        img = img * (1 - neck[..., None]) + neck_col * neck[..., None]
        # ears
        ear_col = alb * (0.55 + 0.45 * lam[..., None]) * np.array([1.0, 0.86, 0.82], np.float32)
        img = img * (1 - ear[..., None]) + ear_col * ear[..., None]
        # face skin
        eyl_rgb, eyl_a = self._eye_layers(u, v, eyes)
        body = skin_lit
        if eyes == "closed":
            # closed lid: slightly darker skin with a lash line
            lid = sum(_g(u, v, sg * 0.5, 0.0, 0.26, 0.07) for sg in (-1, 1))
            body = body * (1 - 0.10 * lid[..., None])
            for sg in (-1, 1):
                curve = 0.035 + 0.025 * ((u - sg * 0.5) / 0.25) ** 2 * -1
                lash = np.exp(-0.5 * ((v - curve) / 0.011) ** 2) * _soft(0.25 - np.abs(u - sg * 0.5), 0.01)
                body = body * (1 - 0.65 * lash[..., None])
        else:
            body = body * (1 - eyl_a[..., None]) + eyl_rgb * eyl_a[..., None]
            for sg in (-1, 1):  # lashes + crease
                uu, vv = u - sg * 0.5, v
                e = np.clip(1 - (uu / 0.245) ** 2, 0, 1)
                up_edge = -0.098 * s.eye_open * e ** 0.85
                lash = np.exp(-0.5 * ((vv - up_edge + 0.006) / 0.010) ** 2) * _soft(0.255 - np.abs(uu), 0.01)
                body = body * (1 - 0.8 * lash[..., None])
                crease = np.exp(-0.5 * ((vv - (up_edge - 0.075 * e ** 0.6)) / 0.014) ** 2) * _soft(0.30 - np.abs(uu), 0.02)
                body = body * (1 - 0.22 * crease[..., None])
        img = img * (1 - face[..., None]) + body * face[..., None]
        # hair on top
        strand = cv2.GaussianBlur(rng.standard_normal((N, N)).astype(np.float32), (0, 0), 0.5 * ss)
        strand = cv2.GaussianBlur(strand, (0, 0), sigmaX=0.4 * ss, sigmaY=3.0 * ss)
        strand /= strand.std() + 1e-9
        hcol = np.array(s.hair, np.float32)[None, None, :] * (1 + 0.28 * strand[..., None])
        hcol = hcol * (0.6 + 1.1 * lam[..., None])
        # hair also overlaps the face edge softly at the hairline
        img = img * (1 - hair[..., None]) + np.clip(hcol, 0, 1) * hair[..., None]
        # redraw brows over hairline if they overlap
        return np.clip(img, 0, 1) * 255.0

    # ------------------------------------------------------------------ helpers for rendering
    def uint8_open(self) -> np.ndarray:
        return np.clip(self.base_open, 0, 255).astype(np.uint8)
