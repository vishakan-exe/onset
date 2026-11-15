"""Camera presets and scenario factories (nuisance schedules, null clips, event clips)."""
from __future__ import annotations

import numpy as np

from .render import Camera, Event, HeadMotion, Scenario

# name -> (fps, Camera).  Numbers are typical, not measured from a particular device.
CAMERAS: dict[str, tuple[float, Camera]] = {
    "lab200": (200.0, Camera(noise=1.0, blur=0.0, jpeg_q=None, gain_drift=0.005, vignette=0.05)),
    "highspeed100": (100.0, Camera(noise=1.3, blur=0.2, jpeg_q=94, gain_drift=0.006, vignette=0.06)),
    "webcam30": (30.0, Camera(noise=2.5, blur=0.4, jpeg_q=85, gain_drift=0.01, vignette=0.10, flicker_amp=0.008, flicker_hz=7.0)),
    "phone30": (30.0, Camera(noise=2.0, blur=0.5, jpeg_q=None, codec_crf=26, gain_drift=0.01, vignette=0.08)),
    "webcam24_lowlight": (24.0, Camera(noise=4.5, blur=0.7, jpeg_q=78, gain_drift=0.015, vignette=0.14, flicker_amp=0.012, flicker_hz=5.0)),
    "webcam15": (15.0, Camera(noise=3.0, blur=0.5, jpeg_q=85, gain_drift=0.01, vignette=0.10)),
}

NUISANCE_LEVELS = ("still", "normal", "busy")


def nuisance_schedule(rng: np.random.Generator, duration: float, level: str = "normal") -> dict:
    """Blinks, speech, gaze shifts and head motion for a clip of ``duration`` seconds."""
    blinks, speech, gaze = [], [], []
    head = HeadMotion()
    if level == "still":
        head = HeadMotion(trans=0.003, rot_deg=0.12, scale=0.001)
    else:
        rate = 0.30 if level == "normal" else 0.40
        t = rng.uniform(0.3, 1.5)
        while t < duration - 0.5:
            blinks.append((float(t), float(rng.uniform(0.13, 0.30))))
            t += rng.exponential(1.0 / rate) + 0.5
        t = rng.uniform(1.0, 3.0)
        while t < duration - 0.5:
            gaze.append((float(t), float(rng.choice([-1, 1]) * rng.uniform(0.003, 0.010)), 0.0))
            t += rng.uniform(1.5, 4.0)
        if level == "busy":
            head = HeadMotion(trans=0.016, rot_deg=0.7, scale=0.004, fmax=2.4,
                              bursts=[(float(b), float(b) + 0.6, 2.5) for b in rng.uniform(0.5, max(duration - 1.0, 0.6), 2)])
            if duration > 4.0:
                t0 = float(rng.uniform(1.0, duration - 3.0))
                speech.append((t0, t0 + float(rng.uniform(1.2, 2.2))))
    return dict(blinks=blinks, speech=speech, gaze=gaze, head=head)


def _clear_of_events(schedule: list, events: list, margin: float, width=lambda x: x[1]) -> list:
    """Entries ``(t, ...)`` of ``schedule`` whose span [t, t + width] stays more than ``margin`` seconds away from every event."""
    keep = []
    for item in schedule:
        t0 = item[0]
        t1 = t0 + (width(item) if width else 0.0)
        if not any(t0 <= e.t_off + margin and t1 >= e.t_on - margin for e in events):
            keep.append(item)
    return keep


def make_scenario(phantom: str, camera: str, duration: float, events: list[Event] | None = None, seed: int = 0,
                  nuisance: str = "normal", size: int = 480, iod: float = 104.0, name: str = "",
                  camera_overrides: dict | None = None, nuisance_overrides: dict | None = None, keep_clear: float = 0.0) -> Scenario:
    """``keep_clear`` > 0 drops the blinks and gaze shifts of the nuisance schedule that would fall within that many seconds of a
    scripted event (the rest of the schedule is untouched).  The calibration corpus leaves it at 0, because blinks that collide
    with a movement are part of what the instrument has to cope with; the specimen library sets it so that a specimen shows
    what it is about (a smile, a frown) and not an accident of the random blink schedule."""
    fps, cam = CAMERAS[camera]
    if camera_overrides:
        cam = Camera(**{**cam.__dict__, **camera_overrides})
    rng = np.random.default_rng(10_000 + seed)
    nz = nuisance_schedule(rng, duration, nuisance)
    if nuisance_overrides:
        nz.update(nuisance_overrides)
    evs = list(events or [])
    if keep_clear > 0 and evs:
        nz["blinks"] = _clear_of_events(nz["blinks"], evs, keep_clear)
        nz["gaze"] = _clear_of_events(nz["gaze"], evs, keep_clear, width=None)
    return Scenario(phantom=phantom, fps=fps, duration=duration, events=evs, blinks=nz["blinks"],
                    speech=nz["speech"], gaze=nz["gaze"], head=nz["head"], camera=cam, seed=seed, size=size, iod=iod,
                    name=name or f"{phantom}-{camera}-{nuisance}-{seed}")


def make_null(phantom: str, camera: str, duration: float, seed: int, nuisance: str = "normal", **kw) -> Scenario:
    return make_scenario(phantom, camera, duration, [], seed, nuisance, **kw)
