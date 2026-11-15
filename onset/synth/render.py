"""Scenario -> camera-like video with exact ground truth.

A scenario is a continuous-time script (AU events, blinks, speech, gaze shifts, head motion) played
on a phantom; frames are sampled at any fps, so the *same* underlying motion can be filmed by a
200 fps lab camera or a 30 fps webcam.  The camera model adds the nuisances that decide whether
micro-motion is measurable: sensor noise, defocus, lighting drift/flicker, vignetting and JPEG or
H.264 compression.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import cv2
import numpy as np

from .deform import (AU_BLOBS, BLINK_BLOBS, GAZE_BLOBS, SPEECH_BLOBS, Jitter, field_from_blobs, smooth_bump)
from .phantom import PHANTOMS, Phantom, PhantomSpec

COARSE = 4


@dataclass
class Event:
    aus: dict
    t_on: float
    t_apex: float
    t_off: float
    amp: float                    # peak displacement in IOD units (0.015 = 1.5 % of the inter-ocular distance)
    label: str = ""               # category the event was scripted to represent
    kind: str = "micro"           # micro | macro
    jitter: Optional[Jitter] = None
    note: str = ""

    @property
    def duration(self) -> float:
        return self.t_off - self.t_on


@dataclass
class Camera:
    noise: float = 1.5            # sensor noise std in grey levels (at mid-grey)
    blur: float = 0.0             # defocus sigma, px
    jpeg_q: Optional[int] = 92    # per-frame JPEG round trip
    codec_crf: Optional[int] = None   # H.264 round trip (inter-frame compression); None = off
    flicker_amp: float = 0.0
    flicker_hz: float = 0.0       # apparent (aliased) flicker frequency
    gain_drift: float = 0.01
    vignette: float = 0.10
    ts_jitter: float = 0.0        # timestamp jitter as a fraction of the frame interval


@dataclass
class HeadMotion:
    trans: float = 0.008          # IOD, std of translation
    rot_deg: float = 0.30
    scale: float = 0.002
    fmax: float = 1.8             # Hz
    bursts: list = field(default_factory=list)   # (t0, t1, gain) windows with amplified motion


@dataclass
class Scenario:
    phantom: str | PhantomSpec = "A"
    fps: float = 200.0
    duration: float = 2.4
    events: list = field(default_factory=list)
    blinks: list = field(default_factory=list)     # (t_start, duration)
    speech: list = field(default_factory=list)     # (t0, t1)
    gaze: list = field(default_factory=list)       # (t, dx_IOD, dy_IOD)
    head: HeadMotion = field(default_factory=HeadMotion)
    camera: Camera = field(default_factory=Camera)
    seed: int = 0
    size: int = 480
    iod: float = 104.0
    name: str = ""


@dataclass
class SyntheticClip:
    frames: np.ndarray            # N x H x W x 3 uint8 BGR
    times: np.ndarray             # seconds
    fps: float
    scenario: Scenario
    truth: dict
    iod_px: float

    def ground_truth_events(self) -> list[dict]:
        return self.truth["events"]


_PHANTOM_CACHE: dict = {}


def get_phantom(spec, size: int, iod: float) -> Phantom:
    key = (spec if isinstance(spec, str) else repr(spec), size, iod)
    if key not in _PHANTOM_CACHE:
        _PHANTOM_CACHE[key] = Phantom(spec, size=size, iod=iod)
    return _PHANTOM_CACHE[key]


def _lowpass_noise(rng: np.random.Generator, duration: float, fmax: float, n_ch: int, hz: int = 1000):
    n = int(duration * hz) + 2 * hz
    spec = rng.standard_normal((n_ch, n // 2 + 1)) + 1j * rng.standard_normal((n_ch, n // 2 + 1))
    f = np.fft.rfftfreq(n, 1.0 / hz)
    shape = 1.0 / (1.0 + (f / fmax) ** 3) / np.sqrt(np.maximum(f, 0.05))
    shape[0] = 0.0
    x = np.fft.irfft(spec * shape, n=n, axis=1)
    x /= x.std(axis=1, keepdims=True) + 1e-12
    return x[:, hz // 2: hz // 2 + int(duration * hz) + hz // 2], hz


def blink_closure(t: np.ndarray, t0: float, dur: float) -> np.ndarray:
    """0 open -> 1 closed -> 0 open; fast closing (30 %), slower opening."""
    t = np.asarray(t, dtype=np.float64)
    c = np.zeros_like(t)
    a = np.clip((t - t0) / (0.30 * dur), 0, 1)
    b = np.clip((t0 + dur - t) / (0.70 * dur), 0, 1)
    f = np.where(t < t0 + 0.30 * dur, a, b)
    f = np.where((t < t0) | (t > t0 + dur), 0.0, f)
    return f * f * (3 - 2 * f)


def compress_h264(frames: np.ndarray, fps: float, crf: int) -> np.ndarray:
    """Round-trip frames through libx264 (inter-frame compression smooths subtle motion)."""
    n, h, w, _ = frames.shape
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "o.mp4")
        enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                                "-crf", str(crf), "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
        enc.communicate(frames.tobytes())
        dec = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", out, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                             capture_output=True)
        arr = np.frombuffer(dec.stdout, np.uint8)
        arr = arr[: (len(arr) // (h * w * 3)) * h * w * 3].reshape(-1, h, w, 3)
        if len(arr) < n:
            arr = np.concatenate([arr, np.repeat(arr[-1:], n - len(arr), 0)], 0)
        return arr[:n]


class Renderer:
    def __init__(self, sc: Scenario):
        self.sc = sc
        self.ph = get_phantom(sc.phantom, sc.size, sc.iod)
        self.anchors = self.ph.anchors()
        seeds = np.random.SeedSequence(sc.seed).spawn(5)
        self.rng_ev, self.rng_head, self.rng_noise, self.rng_misc, self.rng_cam = (np.random.default_rng(s) for s in seeds)
        n = sc.size // COARSE
        self.coarse_shape = (n, n)
        self._yy, self._xx = np.mgrid[0:sc.size, 0:sc.size].astype(np.float32)
        self._prepare_fields()
        self._prepare_eyes()
        self._prepare_head()

    # ------------------------------------------------------------------ set-up
    def _prepare_fields(self):
        sc, ph = self.sc, self.ph
        self.event_fields = []
        for e in sc.events:
            if e.jitter is None:
                e.jitter = Jitter.sample(self.rng_ev)
            f = np.zeros(self.coarse_shape + (2,), np.float32)
            for au, strength in e.aus.items():
                f += strength * field_from_blobs(AU_BLOBS[au], self.anchors, self.coarse_shape, COARSE, ph.iod,
                                                 e.jitter, self.rng_ev)
            self.event_fields.append(f * e.jitter.amp_scale)
        j0 = Jitter()
        self.f_blink = field_from_blobs(BLINK_BLOBS, self.anchors, self.coarse_shape, COARSE, ph.iod, j0, self.rng_misc)
        self.f_speech = field_from_blobs(SPEECH_BLOBS, self.anchors, self.coarse_shape, COARSE, ph.iod, j0, self.rng_misc)
        self.f_gaze = field_from_blobs(GAZE_BLOBS, self.anchors, self.coarse_shape, COARSE, ph.iod, j0, self.rng_misc)
        # skin mask: keep hair/background still
        diff = np.abs(ph.base_open - ph.base_closed).max(axis=2)
        yy, xx = np.mgrid[0:sc.size, 0:sc.size].astype(np.float32)
        u, v = (xx - ph.cx) / ph.iod, (yy - ph.cy) / ph.iod
        face = (np.abs(u) < 1.0 * (1 - 0.15 * np.clip((v - 0.8) / 1.1, 0, 1))) & (v > -0.95) & (v < 1.95)
        m = cv2.GaussianBlur(face.astype(np.float32), (0, 0), 0.06 * ph.iod)
        self.skin_mask = cv2.resize(m, (self.coarse_shape[1], self.coarse_shape[0]), interpolation=cv2.INTER_AREA)[..., None]

    def _prepare_eyes(self):
        ph, size = self.ph, self.sc.size
        diff = (np.abs(ph.base_open - ph.base_closed).max(axis=2) > 8).astype(np.float32)
        diff = cv2.GaussianBlur(diff, (0, 0), 1.0)
        self.eye_boxes = []
        for sg in (-1, 1):
            cx = ph.cx + sg * 0.5 * ph.iod
            x0, x1 = int(cx - 0.40 * ph.iod), int(cx + 0.40 * ph.iod)
            col = diff[:, x0:x1].max(axis=1)
            rows = np.nonzero(col > 0.3)[0]
            rows = rows[(rows > ph.cy - 0.4 * ph.iod) & (rows < ph.cy + 0.35 * ph.iod)]
            y0, y1 = (int(rows.min()) - 2, int(rows.max()) + 2) if len(rows) else (int(ph.cy - 12), int(ph.cy + 12))
            self.eye_boxes.append((x0, x1, y0, y1, diff[y0:y1, x0:x1].copy()))

    def _prepare_head(self):
        sc = self.sc
        arr, hz = _lowpass_noise(self.rng_head, sc.duration + 1.0, sc.head.fmax, 4)
        self._head_arr, self._head_hz = arr, hz
        self._head_std = np.array([sc.head.trans, sc.head.trans, np.deg2rad(sc.head.rot_deg), sc.head.scale])

    # ------------------------------------------------------------------ time functions
    def head_at(self, t: float) -> np.ndarray:
        idx = np.clip(t * self._head_hz, 0, self._head_arr.shape[1] - 2)
        i = int(idx)
        fr = idx - i
        v = self._head_arr[:, i] * (1 - fr) + self._head_arr[:, i + 1] * fr
        gain = 1.0
        for (t0, t1, g) in self.sc.head.bursts:
            if t0 <= t <= t1:
                gain = max(gain, g)
        return v * self._head_std * gain

    def closure_at(self, t: float) -> float:
        c = 0.0
        for (t0, dur) in self.sc.blinks:
            if t0 <= t <= t0 + dur:
                c = max(c, float(blink_closure(np.array([t]), t0, dur)[0]))
        return c

    def activations_at(self, t: float) -> np.ndarray:
        return np.array([float(smooth_bump(np.array([t]), e.t_on, e.t_apex, e.t_off)[0]) * e.amp
                         for e in self.sc.events], np.float64)

    def speech_level(self, t: float) -> float:
        lv = 0.0
        for (t0, t1) in self.sc.speech:
            if t0 <= t <= t1:
                env = float(smooth_bump(np.array([t]), t0, (t0 + t1) / 2, t1)[0])
                lv = max(lv, env * (0.5 + 0.5 * np.sin(2 * np.pi * 4.3 * (t - t0))) * 0.07)
        return lv

    def gaze_offset(self, t: float) -> float:
        # piece-wise constant offsets with 30 ms saccades
        off = 0.0
        for (t0, dx, _dy) in self.sc.gaze:
            off += dx * float(np.clip((t - t0) / 0.03, 0, 1))
        return off

    # ------------------------------------------------------------------ one frame
    def coarse_field_at(self, t: float, acts: np.ndarray) -> np.ndarray:
        d = np.zeros(self.coarse_shape + (2,), np.float32)
        for a, f in zip(acts, self.event_fields):
            if a != 0.0:
                d += (a / 1.0) * f
        cl = self.closure_at(t)
        if cl > 0:
            d += cl * 0.0035 * self.f_blink
        sp = self.speech_level(t)
        if sp > 0:
            d += sp * self.f_speech
        g = self.gaze_offset(t)
        if g != 0.0:
            d += g * self.f_gaze
        return d * self.skin_mask

    def render_geometry(self, t: float, acts: np.ndarray, base: np.ndarray):
        """Warp ``base`` by skin field then rigid head motion.  Returns (image, maps for truth)."""
        sc, ph = self.sc, self.ph
        size = sc.size
        h = self.head_at(t)
        tx, ty, rot, sc_ = h[0] * ph.iod, h[1] * ph.iod, h[2], 1.0 + h[3]
        cx, cy = ph.cx, ph.cy + 0.5 * ph.iod
        ca, sa = np.cos(rot), np.sin(rot)
        yy, xx = self._yy, self._xx
        # q = R^-1(x)
        xr, yr = (xx - (cx + tx)) / sc_, (yy - (cy + ty)) / sc_
        qx = ca * xr + sa * yr + cx
        qy = -sa * xr + ca * yr + cy
        dc = self.coarse_field_at(t, acts)
        coarse_map_x = ((qx + 0.5) / COARSE - 0.5).astype(np.float32)
        coarse_map_y = ((qy + 0.5) / COARSE - 0.5).astype(np.float32)
        d1 = cv2.remap(dc, coarse_map_x, coarse_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        px, py = qx - d1[..., 0], qy - d1[..., 1]
        d2 = cv2.remap(dc, ((px + 0.5) / COARSE - 0.5).astype(np.float32),
                       ((py + 0.5) / COARSE - 0.5).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        px, py = (qx - d2[..., 0]).astype(np.float32), (qy - d2[..., 1]).astype(np.float32)
        img = cv2.remap(base, px, py, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return img, (tx, ty, rot, sc_)

    def eyes_blend(self, closure: float) -> np.ndarray:
        ph = self.ph
        if closure <= 0:
            return ph.base_open
        out = ph.base_open.copy()
        for (x0, x1, y0, y1, region) in self.eye_boxes:
            hgt = y1 - y0
            ys = np.arange(hgt, dtype=np.float32)[:, None]
            lid_edge = closure * (hgt + 2) * 1.02
            sweep = 1.0 / (1.0 + np.exp(-(lid_edge - ys) / 1.4))
            alpha = (np.clip(region * 1.6, 0, 1) * sweep)[..., None]
            out[y0:y1, x0:x1] = ph.base_open[y0:y1, x0:x1] * (1 - alpha) + ph.base_closed[y0:y1, x0:x1] * alpha
        return out

    def camera_frame(self, img: np.ndarray, t: float, k: int) -> np.ndarray:
        cam, size = self.sc.camera, self.sc.size
        g = 1.0 + cam.gain_drift * np.sin(2 * np.pi * 0.21 * t + 0.7)
        if cam.flicker_amp > 0:
            g += cam.flicker_amp * np.sin(2 * np.pi * cam.flicker_hz * t + 1.3)
        if not hasattr(self, "_vig"):
            r2 = ((self._xx - size / 2) ** 2 + (self._yy - size / 2) ** 2) / (size / 2) ** 2
            self._vig = (1 - cam.vignette * r2)[..., None].astype(np.float32)
        img = img * (self._vig * np.float32(g))
        if cam.blur > 0:
            img = cv2.GaussianBlur(img, (0, 0), cam.blur)
        if cam.noise > 0:
            # luminance-dominated sensor noise: brighter pixels are a little noisier
            n = self.rng_noise.standard_normal(img.shape[:2], dtype=np.float32)
            amp = cam.noise * np.sqrt(0.35 + 0.65 * img[..., 1] / 128.0)
            img = img + (n * amp)[..., None]
        bgr = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if cam.jpeg_q:
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(cam.jpeg_q)])
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return bgr

    # ------------------------------------------------------------------ whole clip
    def render(self, progress: Optional[Callable[[float], None]] = None) -> SyntheticClip:
        sc = self.sc
        n = int(round(sc.duration * sc.fps))
        t_nom = np.arange(n) / sc.fps
        if sc.camera.ts_jitter > 0:
            jit = self.rng_cam.uniform(-0.5, 0.5, n) * sc.camera.ts_jitter / sc.fps
            jit[0] = 0
            times = np.maximum.accumulate(t_nom + jit)
        else:
            times = t_nom
        frames = np.empty((n, sc.size, sc.size, 3), np.uint8)
        acts_all = np.zeros((n, len(sc.events)))
        closure = np.zeros(n)
        head = np.zeros((n, 4))
        speech = np.zeros(n)
        for i, t in enumerate(times):
            acts = self.activations_at(t)
            acts_all[i] = acts
            cl = self.closure_at(t)
            closure[i] = cl
            speech[i] = self.speech_level(t)
            base = self.eyes_blend(cl)
            img, hh = self.render_geometry(t, acts, base)
            head[i] = (hh[0] / self.ph.iod, hh[1] / self.ph.iod, hh[2], hh[3] - 1.0)
            frames[i] = self.camera_frame(img, t, i)
            if progress and i % 25 == 0:
                progress(i / n)
        if sc.camera.codec_crf is not None:
            frames = compress_h264(frames, sc.fps, sc.camera.codec_crf)
        truth = {
            "events": [dict(aus=e.aus, t_on=e.t_on, t_apex=e.t_apex, t_off=e.t_off, amp_iod=e.amp, label=e.label,
                            kind=e.kind, side=e.jitter.side if e.jitter else 0.5, note=e.note) for e in sc.events],
            "blinks": [dict(t_on=b[0], t_off=b[0] + b[1]) for b in sc.blinks],
            "speech": [dict(t_on=a, t_off=b) for a, b in sc.speech],
            "gaze": [dict(t=g[0], dx=g[1]) for g in sc.gaze],
            "activation": acts_all, "closure": closure, "head": head, "speech_level": speech,
            "anchors": {k: v.tolist() for k, v in self.anchors.items()},
        }
        return SyntheticClip(frames, times, sc.fps, sc, truth, self.ph.iod)

    # ------------------------------------------------------------------ exact truth for tests
    def skin_displacement(self, t: float, shape_scale: int = 1) -> np.ndarray:
        """Skin displacement field D(t) in base pixel coordinates (size x size x 2), before head motion."""
        acts = self.activations_at(t)
        dc = self.coarse_field_at(t, acts)
        return cv2.resize(dc, (self.sc.size, self.sc.size), interpolation=cv2.INTER_LINEAR)


def render(sc: Scenario, progress=None) -> SyntheticClip:
    return Renderer(sc).render(progress)
