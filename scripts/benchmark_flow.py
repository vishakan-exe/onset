"""Measure optical-flow accuracy on *known* sub-pixel warps (procedural skin texture).

Usage:  python scripts/benchmark_flow.py [--image face.png]

For every estimator and amplitude it reports
  bias   - ROI-mean estimated / true motion on the moving patch (1.0 is perfect, <1 under-reads)
  rmse   - RMS error of the ROI-mean vector (14 px radius ROI), px
  floor  - RMS ROI-mean estimate where nothing moved: the measurement noise floor, px
  ms     - time per frame pair
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from onset.vision.flow import DISFlow, FarnebackFlow, LKFlow  # noqa: E402


def skin_texture(size: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)

    def noise(sigma, amp):
        n = rng.standard_normal((size, size)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        return amp * n / (n.std() + 1e-9)

    img = 128 + noise(24, 7) + noise(6, 4) + noise(2.0, 3.0) + noise(0.9, 1.8)
    for _ in range(40):  # freckles / pores with a bit more contrast
        x, y = rng.integers(10, size - 10, 2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 3)), float(img[y, x] - rng.uniform(8, 20)), -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    return img


def warp_with_field(img: np.ndarray, d: np.ndarray) -> np.ndarray:
    """b(x) such that content at y moves to y + d(y)  (fixed-point inverse mapping)."""
    h, w = img.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    y = np.stack([xs, ys], -1)
    for _ in range(8):
        dy = cv2.remap(d, y[..., 0], y[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        y = np.stack([xs, ys], -1) - dy
    return cv2.remap(img, y[..., 0], y[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def run(image: np.ndarray | None, amps, noises, seeds=12):
    h = w = 256
    est = {
        "LK (default)": LKFlow(), "LK s4": LKFlow(win_sigma=4.0), "LK s8": LKFlow(win_sigma=8.0),
        "LK 2-level": LKFlow(levels=2, iterations=(2, 3)),
        "DIS medium": DISFlow(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM, 5),
        "DIS fast": DISFlow(cv2.DISOPTICAL_FLOW_PRESET_FAST, 5),
        "Farneback": FarnebackFlow(),
    }
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    shape = np.exp(-((xs - 128) ** 2 + (ys - 110) ** 2) / (2 * 22.0 ** 2)).astype(np.float32)
    patch = shape > 0.5
    still = shape < 0.01
    disc = ((xs - 128) ** 2 + (ys - 110) ** 2) < 14 ** 2          # ROI on the moving patch
    disc_still = ((xs - 60) ** 2 + (ys - 200) ** 2) < 14 ** 2     # ROI where nothing moves
    rows = []
    for name, f in est.items():
        for noise in noises:
            for amp in amps:
                bias, rmse, floor, ms, roi_err, roi_floor = [], [], [], [], [], []
                for s in range(seeds):
                    rng = np.random.default_rng(100 + s)
                    base = skin_texture(h, seed=s) if image is None else image
                    d = np.zeros((h, w, 2), np.float32)
                    d[..., 1] = -amp * shape            # upward bump
                    d[..., 0] = 0.4 * amp * shape
                    b = warp_with_field(base, d)
                    a8 = np.clip(base + rng.normal(0, noise, base.shape), 0, 255).round().astype(np.uint8)
                    b8 = np.clip(b + rng.normal(0, noise, base.shape), 0, 255).round().astype(np.uint8)
                    t = time.perf_counter()
                    fl = f(a8, b8)
                    ms.append((time.perf_counter() - t) * 1000)
                    err = fl - d
                    truth = d[patch]
                    proj = (fl[patch] * truth).sum() / max((truth * truth).sum(), 1e-12)
                    bias.append(proj)
                    rmse.append(float(np.sqrt((err[patch] ** 2).sum(-1).mean())))
                    floor.append(float(np.sqrt((fl[still] ** 2).sum(-1).mean())))
                    tm = d[disc].mean(0)
                    em = fl[disc].mean(0)
                    roi_err.append(np.hypot(*(em - tm)))
                    roi_floor.append(np.hypot(*fl[disc_still].mean(0)))
                    bias[-1] = float((em * tm).sum() / max((tm * tm).sum(), 1e-12))
                rows.append((name, noise, amp, np.mean(bias), np.sqrt(np.mean(np.square(roi_err))),
                             np.sqrt(np.mean(np.square(roi_floor))), np.mean(ms)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    args = ap.parse_args()
    img = None
    if args.image:
        g = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(g, (256, 256)).astype(np.float32)
    amps = [0.15, 0.4, 1.0, 2.0]
    noises = [1.0, 3.0]
    print(f"{'estimator':<12}{'noise':>6}{'amp':>6}{'bias':>8}{'rmse':>8}{'floor':>8}{'ms':>7}")
    for r in run(img, amps, noises):
        print(f"{r[0]:<12}{r[1]:>6.1f}{r[2]:>6.2f}{r[3]:>8.2f}{r[4]:>8.3f}{r[5]:>8.3f}{r[6]:>7.1f}")
