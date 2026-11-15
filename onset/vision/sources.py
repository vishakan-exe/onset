"""Frame sources: one restartable interface over uploads, webcam captures and synthetic clips.

Every source exposes the *measured* timing of the footage (timestamps per frame), because the whole
instrument is honest about fps: a clip's real frame rate and jitter decide what can be seen.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
from typing import Iterator, Optional

import cv2
import numpy as np


class FrameSource:
    n: int
    times: np.ndarray
    width: int
    height: int
    kind: str = "unknown"
    codec: str = ""

    def frames(self) -> Iterator[np.ndarray]:  # BGR uint8
        raise NotImplementedError

    # -- timing -------------------------------------------------------------------
    @property
    def fps(self) -> float:
        d = np.diff(self.times)
        d = d[d > 0]
        return float(1.0 / np.median(d)) if len(d) else 0.0

    @property
    def fps_jitter(self) -> float:
        """Coefficient of variation of frame intervals (0 = metronomic)."""
        d = np.diff(self.times)
        d = d[d > 0]
        return float(np.std(d) / np.mean(d)) if len(d) > 2 else 0.0

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if self.n > 1 else 0.0

    def info(self) -> dict:
        d = np.diff(self.times)
        return dict(kind=self.kind, n_frames=int(self.n), width=int(self.width), height=int(self.height),
                    fps=round(self.fps, 3), fps_jitter=round(self.fps_jitter, 4), duration_s=round(self.duration, 3),
                    max_gap_ms=round(float(d.max()) * 1000, 2) if len(d) else 0.0, codec=self.codec)


class ArraySource(FrameSource):
    kind = "array"

    def __init__(self, frames: np.ndarray, times: np.ndarray):
        self._f = frames
        self.n = len(frames)
        self.times = np.asarray(times, dtype=np.float64)
        self.height, self.width = frames.shape[1:3]

    def frames(self):
        for f in self._f:
            yield f


class JpegDirSource(FrameSource):
    """Webcam capture: ``000000.jpg`` ... plus ``times.json`` (seconds, from requestVideoFrameCallback)."""
    kind = "capture-frames"
    codec = "jpeg (intra-only)"

    def __init__(self, folder: str):
        self.folder = folder
        self.files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        if not self.files:
            raise ValueError("no frames in capture folder")
        tpath = os.path.join(folder, "times.json")
        if os.path.exists(tpath):
            t = np.asarray(json.load(open(tpath)), dtype=np.float64)
            if len(t) >= len(self.files):
                t = t[: len(self.files)]
            else:  # fewer timestamps than frames: extend at the median interval
                step = np.median(np.diff(t)) if len(t) > 1 else 1 / 30
                t = np.concatenate([t, t[-1] + step * np.arange(1, len(self.files) - len(t) + 1)])
            self.times = t - t[0]
        else:
            self.times = np.arange(len(self.files)) / 30.0
        self.n = len(self.files)
        im = cv2.imread(self.files[0])
        self.height, self.width = im.shape[:2]

    def frames(self):
        for p in self.files:
            yield cv2.imread(p)


def _probe_times(path: str) -> Optional[np.ndarray]:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "frame=best_effort_timestamp_time", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=120).stdout
        vals = []
        for line in out.splitlines():
            line = line.strip().rstrip(",")
            if line and line != "N/A":
                vals.append(float(line))
        return np.asarray(vals, dtype=np.float64) if vals else None
    except Exception:
        return None


def _probe_codec(path: str) -> str:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=codec_name,bit_rate", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out.replace(",", " ")
    except Exception:
        return ""


class VideoSource(FrameSource):
    """Any video file OpenCV/ffmpeg can decode (mp4, mov, webm, avi...)."""
    kind = "video-file"

    def __init__(self, path: str, max_seconds: Optional[float] = None, max_side: Optional[int] = None):
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("cannot open video")
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nominal = cap.get(cv2.CAP_PROP_FPS) or 0.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        self.codec = _probe_codec(path)
        times = _probe_times(path)
        if count <= 0 or times is None or abs(len(times) - count) > max(2, 0.05 * count):
            # webm from MediaRecorder reports no frame count: decode once to count
            c2 = cv2.VideoCapture(path)
            count = 0
            while True:
                ok = c2.grab()
                if not ok:
                    break
                count += 1
            c2.release()
        if times is None or len(times) < count - 1:
            step = 1.0 / nominal if nominal > 1 else 1 / 30.0
            times = np.arange(count) * step
        times = np.sort(times[:count]) if len(times) >= count else times
        self.n = min(count, len(times))
        self.times = times[: self.n] - times[0]
        if max_seconds is not None:
            keep = int(np.searchsorted(self.times, max_seconds))
            if 2 < keep < self.n:
                self.n, self.times = keep, self.times[:keep]
        self.max_side = max_side

    def frames(self):
        cap = cv2.VideoCapture(self.path)
        i = 0
        while i < self.n:
            ok, f = cap.read()
            if not ok:
                break
            if self.max_side and max(f.shape[:2]) > self.max_side:
                s = self.max_side / max(f.shape[:2])
                f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            yield f
            i += 1
        cap.release()


def open_source(path: str, **kw) -> FrameSource:
    if os.path.isdir(path):
        return JpegDirSource(path)
    return VideoSource(path, **kw)
