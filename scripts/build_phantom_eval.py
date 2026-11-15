"""Build the phantom evaluation corpus: many short synthetic clips with scripted events, run through the real
detection + measurement stages, and keep everything the calibration and evaluation scripts need.

Each clip is a pickle in ``--out`` with the exact ground truth (events, blinks, speech, gaze) next to what the
instrument saw (spotting candidates, per-event measurements, capture-quality summary).  The expensive part
(rendering and optical flow) therefore happens once; calibrating the AU engine, choosing thresholds and drawing the
sensitivity envelope are cheap re-computations on these files.

Identities A-C form the *calibration* split and D-F the *test* split: parameters are fitted on faces the
evaluation never sees (the synthetic analogue of subject-independent evaluation).

    python scripts/build_phantom_eval.py --out /path/to/dir --n 140 --workers 2 --seed 1
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PHANTOMS = "ABCDEF"
CAMERAS = [("webcam30", 0.40), ("phone30", 0.20), ("webcam24_lowlight", 0.15), ("webcam15", 0.10), ("highspeed100", 0.15)]
NUISANCE = [("normal", 0.65), ("busy", 0.20), ("still", 0.15)]
SINGLES = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17]
PAIRS = [(1, 2), (6, 12), (4, 7), (9, 10), (4, 5), (1, 4), (5, 7), (12, 14), (15, 17), (4, 9), (7, 12), (2, 5), (1, 5), (10, 12),
         (6, 7), (12, 15), (2, 4)]
TRIPLES = [(1, 2, 5), (4, 5, 7), (6, 7, 12), (4, 7, 9), (1, 2, 4)]


def _choice(rng, items):
    names, w = zip(*items)
    return names[int(rng.choice(len(names), p=np.array(w) / sum(w)))]


def sample_spec(rng: np.random.Generator, idx: int, null_fraction: float = 0.14) -> dict:
    cam = _choice(rng, CAMERAS)
    fast = cam == "highspeed100"
    dur = 3.0 if fast else 6.0
    nuis = _choice(rng, NUISANCE)
    null = rng.random() < null_fraction
    events = []
    slots = [(0.9, dur * 0.40)] if fast else [(1.1, 2.4), (3.6, 4.9)]
    if not null:
        for lo, hi in slots:
            u = rng.random()
            n_au = 1 if u < 0.55 else 2 if u < 0.90 else 3
            if n_au == 1:
                aus = [int(rng.choice(SINGLES))]
            elif n_au == 2:
                aus = [int(a) for a in PAIRS[int(rng.integers(len(PAIRS)))]]
            else:
                aus = [int(a) for a in TRIPLES[int(rng.integers(len(TRIPLES)))]]
            macro = rng.random() < 0.12
            d = float(rng.uniform(0.7, 1.4)) if macro else float(rng.uniform(0.2, 0.5))
            apex = float(rng.uniform(lo, hi))
            amp = float(np.exp(rng.uniform(np.log(0.004), np.log(0.03))))
            events.append(dict(aus={a: 1.0 for a in aus}, t_on=apex - 0.4 * d, t_apex=apex, t_off=apex + 0.6 * d, amp=amp,
                               kind="macro" if macro else "micro"))
    return dict(idx=idx, phantom=PHANTOMS[idx % len(PHANTOMS)], camera=cam, duration=dur, nuisance=nuis,
                seed=int(rng.integers(1, 10_000_000)), events=events, null=null or not events)


def exact_truth(rend, clip, det, spec) -> list[dict]:
    """The scripted events of a corpus clip in the instrument's own units (see ``onset.synth.truth.exact_events``)."""
    from onset.synth.truth import exact_events
    return exact_events(rend, clip, det, spec["events"])


def run_one(args):
    spec, out_dir = args
    import cv2
    cv2.setNumThreads(1)
    from onset.analysis.pipeline import AnalysisConfig, detect, inference_inputs, measure_events
    from onset.synth.render import Event, Renderer
    from onset.synth.scenarios import make_scenario
    from onset.vision.sources import ArraySource

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"clip_{spec['idx']:04d}.pkl")
    if os.path.exists(path):
        return path
    t0 = time.time()
    evs = [Event(e["aus"], e["t_on"], e["t_apex"], e["t_off"], amp=e["amp"], kind=e["kind"]) for e in spec["events"]]
    sc = make_scenario(spec["phantom"], spec["camera"], spec["duration"], evs, seed=spec["seed"], nuisance=spec["nuisance"])
    rend = Renderer(sc)
    clip = rend.render()
    src = ArraySource(clip.frames, clip.times)
    cfg = AnalysisConfig(threads=1, spot_threshold=3.5)       # low threshold: keep sub-threshold candidates, filter later
    det = detect(src, cfg)
    ems = measure_events(det, cfg)
    big = max(det.spot_res.scales, key=lambda r: r.spec.k)
    exact = exact_truth(rend, clip, det, spec) if spec["events"] else []
    rows = []
    for em in ems:
        mask, p_b, p_s = inference_inputs(em)
        rows.append(dict(
            cand=dict(time=em.cand.time, score=em.cand.score, z=em.cand.z_peak, kind=em.cand.kind, au=em.cand.au,
                      tau=em.cand.scale.tau, amp_px=em.cand.amp_px, note=em.cand.note, masked=em.cand.masked),
            ref=em.ref, t_ref=em.ref_t, pattern=em.pattern, sigma=em.sigma, mask=mask, p_blink=p_b, p_speech=p_s,
            blink_cue=em.blink_cue, speech_cue=em.speech_cue, template=em.template, onset=em.onset_t, apex=em.apex_t,
            offset=em.offset_t, sustained=em.sustained, head_excursion=em.head_excursion, notes=list(em.notes),
            t=em.t, D=em.D.astype(np.float32), amp_trace=em.amp_trace, amp_sd=em.amp_sd))
    out = dict(
        spec=spec, truth=clip.truth, exact=exact, fps=src.fps, iod_canon=det.reg.canon.iod, n=src.n, times=det.times,
        frame_score=det.spot_res.frame_score.astype(np.float32), frame_energy=det.spot_res.frame_energy.astype(np.float32),
        candidates=[dict(time=c.time, score=c.score, z=c.z_peak, kind=c.kind, au=c.au, tau=c.scale.tau, amp_px=c.amp_px, note=c.note, masked=c.masked)
                    for c in det.spot_res.candidates],
        events=rows, noise=det.spot_res.noise_summary, quality=det.quality.to_dict(), sigma_pair=big.sigma_pair,
        cent=det.cent, centre=det.meas.centre, blink_windows=det.cues.blink_windows, speech_windows=det.cues.speech_windows,
        seconds=time.time() - t0)
    with open(path + ".tmp", "wb") as fh:
        pickle.dump(out, fh)
    os.replace(path + ".tmp", path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=140)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    specs = [sample_spec(rng, i) for i in range(a.n)]
    t0 = time.time()
    jobs = [(s, a.out) for s in specs]
    if a.workers > 1:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for k, p in enumerate(pool.imap_unordered(run_one, jobs)):
                print(f"[{k + 1}/{a.n}] {os.path.basename(p)}  elapsed {time.time() - t0:.0f}s", flush=True)
    else:
        for k, j in enumerate(jobs):
            p = run_one(j)
            print(f"[{k + 1}/{a.n}] {os.path.basename(p)}  elapsed {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
