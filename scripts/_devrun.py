"""Development helper: render a scenario, run track -> register -> measure -> spot, return everything."""
from __future__ import annotations
import time
import numpy as np
from onset.synth.render import Renderer
from onset.vision.sources import ArraySource
from onset.vision import register as R
from onset.vision.measure import Measurer
from onset.vision.spotting import spot
from onset.analysis import templates as TPL
from onset.vision.nuisance import cues as nz_cues


def run(sc, threshold=4.6, keep=False, **spot_kw):
    t0 = time.time()
    rend = Renderer(sc)
    clip = rend.render()
    t_render = time.time() - t0
    src = ArraySource(clip.frames, clip.times)
    t0 = time.time()
    track = R.track_landmarks(src)
    reg = R.register(src, track)
    meas = Measurer(reg)
    cues = nz_cues(track, reg, clip.times, src.fps)
    meas.set_blink_frames(cues.blink_mask, pad=max(1, int(round(0.05 * src.fps))))
    cent = TPL.roi_centroids(reg.rois)
    t_reg = time.time() - t0
    t0 = time.time()
    res, ls = spot(meas, clip.times, src.fps, cent, meas.centre, threshold=threshold, return_lagseries=True, blink_windows=cues.blink_windows, **spot_kw)
    t_spot = time.time() - t0
    out = dict(res=res, ls=ls, truth=clip.truth, times=clip.times, fps=src.fps, timing=dict(render=t_render, register=t_reg, spot=t_spot))
    if keep:
        out.update(clip=clip, rend=rend, src=src, track=track, reg=reg, meas=meas, cent=cent, cues=cues)
    return out
