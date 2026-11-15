#!/usr/bin/env python
"""Cache the expensive part of spotting (face tracking, stabilisation, dense flow at every scale) for the phantom
corpus, so that the cheap part - noise model, template statistics, peak picking, side-lobe suppression - can be
iterated on in seconds.  Developer tool: the evaluation itself still goes through ``build_phantom_eval.py``.

    python scripts/build_spot_cache.py --corpus DIR --out DIR2 --workers 2
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def run_one(args):
    path, out_dir = args
    import cv2
    cv2.setNumThreads(1)
    from onset.analysis import templates as TPL
    from onset.synth.render import Event, Renderer
    from onset.synth.scenarios import make_scenario
    from onset.vision.measure import Measurer
    from onset.vision.nuisance import cues as nuisance_cues
    from onset.vision.register import register, track_landmarks
    from onset.vision.sources import ArraySource
    from onset.vision.spotting import compute_lagseries

    with open(path, "rb") as fh:
        d = pickle.load(fh)
    spec = d["spec"]
    dest = os.path.join(out_dir, f"spot_{spec['idx']:04d}.pkl")
    if os.path.exists(dest):
        return dest
    evs = [Event(e["aus"], e["t_on"], e["t_apex"], e["t_off"], amp=e["amp"], kind=e["kind"]) for e in spec["events"]]
    sc = make_scenario(spec["phantom"], spec["camera"], spec["duration"], evs, seed=spec["seed"], nuisance=spec["nuisance"])
    clip = Renderer(sc).render()
    src = ArraySource(clip.frames, clip.times)
    track = track_landmarks(src)
    reg = register(src, track)
    times = np.asarray(src.times, float)
    cues = nuisance_cues(track, reg, times, src.fps)
    meas = Measurer(reg, threads=1)
    meas.set_blink_frames(cues.blink_mask, pad=max(1, int(round(0.05 * src.fps))))
    cent = TPL.roi_centroids(reg.rois)
    specs, lagseries = compute_lagseries(meas, src.fps)
    out = dict(spec=spec, truth=clip.truth, times=times, fps=float(src.fps), n=int(src.n), cent=cent, centre=meas.centre,
               specs=specs, lagseries=lagseries, blink_windows=cues.blink_windows, speech_windows=cues.speech_windows,
               blink=cues.blink, speech=cues.speech, blink_mask=cues.blink_mask, iod_canon=float(reg.canon.iod))
    with open(dest + ".tmp", "wb") as fh:
        pickle.dump(out, fh)
    os.replace(dest + ".tmp", dest)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.corpus, "clip_*.pkl")))
    if a.limit:
        paths = paths[: a.limit]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, p in enumerate(pool.imap_unordered(run_one, [(p, a.out) for p in paths])):
            print(f"[{k + 1}/{len(paths)}] {os.path.basename(p)} elapsed {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
