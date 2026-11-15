"""Landmark tracking and stabilisation into a canonical, eye-aligned face frame.

Two passes over the footage keep memory bounded: pass 1 tracks landmarks only; pass 2 warps every
frame into the canonical frame (grey, uint8) and records per-frame quality measurements.  Head
pose is *measured* (and later used to gate events), never silently smoothed away.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np
from scipy.interpolate import interp1d

from . import geometry as G
from .facemesh import FaceTracker
from .sources import FrameSource


@dataclass
class LandmarkTrack:
    xy: np.ndarray            # N x 468 x 2 (source px)
    z: np.ndarray             # N x 468
    score: np.ndarray         # N
    valid: np.ndarray         # N bool (face found by the network at that frame)
    stride: int = 1
    iod_px: float = 0.0       # median inter-ocular distance (source px)

    @property
    def n(self) -> int:
        return len(self.xy)


def track_landmarks(source: FrameSource, stride: int = 1, progress: Optional[Callable[[float], None]] = None
                    ) -> LandmarkTrack:
    """Run the 468-point tracker; with stride > 1 only every stride-th frame is processed, the rest
    interpolated (the head moves smoothly; the flow stage removes what is left)."""
    n = source.n
    xy = np.full((n, G.N_LM, 2), np.nan, np.float32)
    zz = np.full((n, G.N_LM), np.nan, np.float32)
    sc = np.zeros(n, np.float32)
    got = np.zeros(n, bool)          # tracker returned a face at this (processed) frame
    proc = np.zeros(n, bool)         # frame was processed at all
    tr = FaceTracker()
    for i, frame in enumerate(source.frames()):
        if i >= n:
            break
        if i % stride == 0 or i == n - 1:
            proc[i] = True
            res = tr.process(frame)
            if res is not None:
                xy[i], zz[i], sc[i], got[i] = res.xy, res.z, res.score, True
        if progress and i % 20 == 0:
            progress(i / n)
    idx = np.nonzero(got)[0]
    if len(idx) < 3:
        raise ValueError("no face found in the footage")
    t = np.arange(n)
    xy_f = interp1d(idx, xy[idx].reshape(len(idx), -1), axis=0, bounds_error=False,
                    fill_value=(xy[idx[0]].ravel(), xy[idx[-1]].ravel()))(t)
    z_f = interp1d(idx, zz[idx], axis=0, bounds_error=False, fill_value=(zz[idx[0]], zz[idx[-1]]))(t)
    xy = xy_f.reshape(n, G.N_LM, 2).astype(np.float32)
    zz = z_f.astype(np.float32)
    # validity: the nearest *processed* frame found a face
    pidx = np.nonzero(proc)[0]
    nearest = pidx[np.abs(t[:, None] - pidx[None, :]).argmin(axis=1)] if n * len(pidx) < 4e7 else pidx[
        np.clip(np.searchsorted(pidx, t), 0, len(pidx) - 1)]
    valid = got[nearest]
    sc = np.interp(t, idx, sc[idx]).astype(np.float32)
    iods = np.array([G.interocular(x) for x in xy[idx[:: max(1, len(idx) // 50)]]])
    return LandmarkTrack(xy, zz, sc, valid, stride, float(np.median(iods)))


def pick_reference(track: LandmarkTrack, fps: float, window_s: float = 0.30) -> tuple[int, np.ndarray]:
    """Quietest, most neutral stretch -> (centre frame index, median reference landmarks).

    Quiet = the rigid landmarks hardly move inside the window; neutral = the non-rigid landmarks sit
    close to their clip-wide median configuration (the face spends most of a clip near neutral)."""
    n = track.n
    w = int(max(5, min(round(window_s * fps), n // 2)))
    iod = max(track.iod_px, 1.0)
    st_idx = np.asarray(G.STABLE_REG)
    free = np.setdiff1d(np.arange(G.N_LM), st_idx)
    # align every frame on the rigid landmarks to the clip median, then measure how far the rest deviate
    med_st = np.median(track.xy[:, st_idx], axis=0)
    wts = np.ones(len(st_idx))
    aligned = np.empty((n, len(free), 2))
    for i in range(n):
        m = G.umeyama(track.xy[i, st_idx], med_st, wts)
        aligned[i] = G.apply_affine(m, track.xy[i, free])
    shape_med = np.median(aligned, axis=0)
    dev = np.sqrt(((aligned - shape_med) ** 2).sum(-1).mean(-1)) / iod          # per frame, IOD units
    st = track.xy[:, st_idx, :]
    best, best_i = 1e9, w // 2
    step = max(1, w // 4)
    for s in range(0, n - w + 1, step):
        seg = st[s:s + w]
        spread = float(np.mean(np.std(seg, axis=0))) / iod
        score = spread + 2.0 * float(dev[s:s + w].mean())
        if track.valid[s:s + w].mean() < 0.9:
            score += 1e3
        if score < best:
            best, best_i = score, s + w // 2
    s0 = max(0, best_i - w // 2)
    ref = np.median(track.xy[s0:s0 + w], axis=0)
    return best_i, ref.astype(np.float64)


def canon_for(iod_px: float) -> G.Canon:
    iod_c = float(np.clip(iod_px, 48.0, 96.0))
    size = int(round(iod_c * 4.0 / 8.0)) * 8
    return G.Canon(size=size, iod=iod_c)


@dataclass
class Registration:
    canon: G.Canon
    ref_index: int
    ref_xy: np.ndarray             # source px
    ref_c: np.ndarray              # canonical px (468 x 2)
    M: np.ndarray                  # N x 2 x 3 source -> canonical
    gray: np.ndarray               # N x S x S uint8
    pose: np.ndarray               # N x 3 (yaw, pitch, roll) degrees relative to the reference
    head: np.ndarray               # N x 4: face-centre shift x, y (IOD), roll (deg), scale change; all vs the reference
    stats: dict = field(default_factory=dict)
    rois: dict = field(default_factory=dict)
    fit_w: Optional[np.ndarray] = None
    oval: Optional[np.ndarray] = None


def _rot_to_euler(r: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch, yaw, roll = math.atan2(-r[1, 2], r[1, 1]), math.atan2(-r[2, 0], sy), 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _pose_series(track: LandmarkTrack, ref_xy: np.ndarray, ref_z: np.ndarray) -> np.ndarray:
    """Head rotation relative to the reference from the 3-D landmarks (rough, used for gating only)."""
    idx = G.STABLE_REG
    ref3 = np.column_stack([ref_xy[idx], ref_z[idx]]).astype(np.float64)
    ref3 -= ref3.mean(0)
    out = np.zeros((track.n, 3))
    for i in range(0, track.n):
        cur = np.column_stack([track.xy[i, idx], track.z[i, idx]]).astype(np.float64)
        cur -= cur.mean(0)
        u, s, vt = np.linalg.svd(cur.T @ ref3)
        d = np.sign(np.linalg.det(u @ vt))
        r = u @ np.diag([1, 1, d]) @ vt
        out[i] = _rot_to_euler(r)
    return out


def register(source: FrameSource, track: LandmarkTrack, progress: Optional[Callable[[float], None]] = None,
             display_size: int = 0) -> Registration:
    n = source.n
    fps = source.fps or 30.0
    ref_i, ref_xy = pick_reference(track, fps)
    canon = canon_for(track.iod_px)
    ea, eb = G.eye_centres(ref_xy)
    ta, tb = canon.eye_targets()
    M0 = G.similarity_from_two_points(ea, eb, ta, tb)
    ref_c = G.apply_affine(M0, ref_xy)
    ref_z = np.median(track.z[max(0, ref_i - 3): ref_i + 4], axis=0)
    # stabilising transform per frame: stable landmarks -> their canonical reference positions
    M = np.zeros((n, 2, 3))
    weights = np.ones(len(G.STABLE_REG))
    for i in range(n):
        M[i] = G.umeyama(track.xy[i, G.STABLE_REG], ref_c[G.STABLE_REG], weights)
    pose = _pose_series(track, ref_xy, ref_z)
    # pose relative to the reference frame
    pose = pose - np.median(pose[max(0, ref_i - 3): ref_i + 4], axis=0)
    # head kinematics relative to the reference frame: face-centre shift (IOD), roll (deg), scale change
    head = np.zeros((n, 4))
    c0 = track.xy[ref_i, G.STABLE_REG].mean(0)
    _, _, rot0, sc0 = G.similarity_params(M[ref_i])
    for i in range(n):
        _, _, rot, sc = G.similarity_params(M[i])
        shift = (track.xy[i, G.STABLE_REG].mean(0) - c0) / max(track.iod_px, 1.0)
        head[i] = (shift[0], shift[1], math.degrees(rot - rot0), sc / sc0 - 1.0)
    S = canon.size
    gray = np.empty((n, S, S), np.uint8)
    blur = np.zeros(n)
    lum = np.zeros(n)
    sat = np.zeros(n)
    noise = np.zeros(n)
    oval = G.face_oval_mask(ref_c, canon)
    skin = cv2.erode(oval.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    for i, frame in enumerate(source.frames()):
        if i >= n:
            break
        crop = cv2.warpAffine(frame, M[i], (S, S), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray[i] = g
        gf = g.astype(np.float32)
        lap = cv2.Laplacian(gf, cv2.CV_32F, ksize=3)
        blur[i] = float(lap[skin].var())
        vals = g[skin]
        lum[i] = float(vals.mean())
        sat[i] = float(((vals < 6) | (vals > 249)).mean())
        # noise: robust spread of the high-pass residual (skin texture also contributes, so this is an upper bound)
        hp = gf - cv2.GaussianBlur(gf, (0, 0), 1.2)
        noise[i] = 1.4826 * float(np.median(np.abs(hp[skin] - np.median(hp[skin]))))
        if progress and i % 20 == 0:
            progress(i / n)
    # photometric normalisation: undo global gain changes (flicker, auto-exposure) before motion estimation,
    # since optical flow assumes constant brightness.  Multiplicative model, skin-region mean as the reference.
    gain = np.clip(np.median(lum) / np.maximum(lum, 1.0), 0.8, 1.25)
    for i in range(n):
        if abs(gain[i] - 1.0) > 0.003:
            gray[i] = np.clip(gray[i].astype(np.float32) * gain[i] + 0.5, 0, 255).astype(np.uint8)
    rois = G.build_rois(ref_c, canon)
    fit_w = fit_weights(ref_c, canon, rois)
    reg = Registration(canon, ref_i, ref_xy, ref_c, M, gray, pose, head, rois=rois, fit_w=fit_w, oval=oval)
    reg.stats = dict(blur=blur, lum=lum, sat=sat, noise=noise, gain=gain)
    return reg


def fit_weights(ref_c: np.ndarray, canon: G.Canon, rois: dict) -> np.ndarray:
    """Pixels used to estimate (then remove) residual rigid motion: the nasal dorsum only.

    Anything the facial muscles can move (forehead, brows, cheeks, mouth, chin) must stay out of this
    set, otherwise a large expression leaks into the 'head motion' estimate and is subtracted from the
    measurement.  The dorsum is bony; AU9 moves its root, so the nose-root ROI is excluded as well.
    """
    s, iod = canon.size, canon.iod
    w = np.zeros((s, s), np.float32)
    core = G._hull_mask(ref_c[[197, 195, 5, 4]], s, 0.10 * iod)
    root = rois.get("nose_root")
    if root is not None:
        core &= ~cv2.dilate(root.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    for name in ("wing_L", "wing_R", "lip_up"):
        if name in rois:
            core &= ~rois[name]
    w[core] = 1.0
    sub = np.zeros((s, s), np.float32)
    sub[::2, ::2] = 1.0
    return w * sub
