"""Render, analyse and package the specimen library into ``data/library``.

For each specimen: the raw colour clip as MP4 + WebM (what a person would see), ``truth.json`` (what was scripted),
``result.json`` and the plate images (what the instrument saw and said) and ``score.json`` (instrument against truth).
Finally ``manifest.json`` indexes them.  Re-running skips specimens that are already complete (``--force`` redoes).

    python scripts/build_library.py [--only id1,id2] [--workers 2] [--force]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from onset import config  # noqa: E402


def build_one(args) -> dict:
    spec_id, out_root, force = args
    import cv2
    cv2.setNumThreads(1)
    from onset.analysis.pipeline import AnalysisConfig, analyse, detect
    from onset.analysis.scoring import score_clip
    from onset.synth import library as LIB
    from onset.synth.truth import exact_events
    from onset.vision.sources import ArraySource

    spec = LIB.get(spec_id)
    out = os.path.join(out_root, spec_id)
    done = os.path.join(out, "DONE")
    if os.path.exists(done) and not force:
        return dict(id=spec_id, skipped=True)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    t0 = time.time()
    rend, clip = LIB.render_with_renderer(spec)
    vids = LIB.write_videos(clip.frames, clip.fps, os.path.join(out, "video"))
    write_thumb(clip, spec, os.path.join(out, "thumb.jpg"))
    src = ArraySource(clip.frames, clip.times)
    cal = config.load_calibration()
    cfg = AnalysisConfig(threads=1, calibration=cal)
    det = detect(src, cfg)
    exact = exact_events(rend, clip, det, clip.truth["events"]) if clip.truth["events"] else []
    truth = LIB.truth_json(clip, spec, exact)
    res = analyse(src, out, cfg, detection=det, label=spec["title"], provenance=dict(
        kind="synthetic phantom", specimen=spec_id, phantom=spec["phantom"], camera=spec["camera"], nuisance=spec["nuisance"],
        note="Procedurally rendered face with scripted movements. Not a person."))
    sc = score_clip(truth["events"], res, truth["blinks"])
    with open(os.path.join(out, "truth.json"), "w") as fh:
        json.dump(truth, fh)
    with open(os.path.join(out, "score.json"), "w") as fh:
        json.dump(sc, fh, default=float)
    open(done, "w").write(time.strftime("%Y-%m-%dT%H:%M:%S"))
    return dict(id=spec_id, seconds=time.time() - t0, score={k: sc[k] for k in ("n_truth", "tp_loose", "tp_strict", "fp", "au_f1", "exact_sets")},
                videos=[os.path.basename(v) for v in vids])


def write_thumb(clip, spec, path: str, size: int = 240):
    """A square still from the colour clip: the first scripted apex, or the middle of the clip when nothing was scripted."""
    import cv2
    import numpy as np
    t = np.asarray(clip.times, float)
    apex = [e["t_apex"] for e in spec.get("events", [])]
    k = int(np.argmin(np.abs(t - apex[0]))) if apex else len(t) // 2
    cv2.imwrite(path, cv2.resize(clip.frames[k], (size, size), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])


def write_static_exports(out_root: str, data_dir: str):
    """The atlas and the evidence as plain files, for the server and for the static showcase."""
    from onset.analysis import atlas as ATLAS
    from onset.analysis import cue_lab as CUE
    canon = None
    for name in ("invisible-frown", "smile-flicker"):
        p = os.path.join(out_root, name, "result.json")
        if os.path.exists(p):
            canon = json.load(open(p))["canon"]
            break
    if canon is None:
        for d in sorted(os.listdir(out_root)):
            p = os.path.join(out_root, d, "result.json")
            if os.path.exists(p):
                canon = json.load(open(p))["canon"]
                break
    if canon is not None:
        with open(os.path.join(data_dir, "atlas.json"), "w") as fh:
            json.dump(ATLAS.build_atlas(canon), fh)
    with open(os.path.join(data_dir, "cue_evidence.json"), "w") as fh:
        json.dump(CUE.evidence_payload(), fh)
    ev = {}
    ev_dir = os.path.join(data_dir, "evidence")
    if os.path.isdir(ev_dir):
        for name in sorted(os.listdir(ev_dir)):
            if name.endswith(".json"):
                ev[name[:-5]] = json.load(open(os.path.join(ev_dir, name)))
    ev["calibration"] = config.load_calibration()
    with open(os.path.join(data_dir, "evidence.json"), "w") as fh:
        json.dump(ev, fh)


def write_manifest(out_root: str):
    from onset.synth import library as LIB
    items = []
    for spec in LIB.SPECIMENS:
        d = os.path.join(out_root, spec["id"])
        if not os.path.exists(os.path.join(d, "DONE")):
            continue
        res = json.load(open(os.path.join(d, "result.json")))
        truth = json.load(open(os.path.join(d, "truth.json")))
        score = json.load(open(os.path.join(d, "score.json")))
        n_events = len([e for e in res["events"] if e["status"] in ("micro", "expression", "unexplained", "sustained")])
        items.append(dict(
            id=spec["id"], group=spec["group"], title=spec["title"], blurb=spec["blurb"], look=spec["look"], confusable=spec["confusable"],
            phantom=spec["phantom"], camera=spec["camera"], nuisance=spec["nuisance"], duration=spec["duration"], fps=res["clip"]["fps"],
            videos=[f for f in ("video.mp4", "video.webm") if os.path.exists(os.path.join(d, f))],
            truth=truth, truth_lines=LIB.describe_truth(truth), grade=res["quality"]["grade"], n_events=n_events,
            score=dict(n_truth=score["n_truth"], tp_loose=score["tp_loose"], tp_strict=score["tp_strict"], fp=score["fp"],
                       au_f1=score["au_f1"], exact_sets=score["exact_sets"], apex_errors=score["apex_errors"]),
            calibrated=res["engine"]["calibrated"]))
    man = dict(version=1, generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
               groups=[dict(id=g, title=t, blurb=b) for g, t, b in LIB.GROUPS],
               note=("Every specimen is a synthetic phantom: a procedurally rendered face with scripted movements and exact "
                     "ground truth. No person appears in any of them. Results on specimens measure the instrument against a "
                     "known answer; they are not evidence of accuracy on people."),
               specimens=items)
    with open(os.path.join(out_root, "manifest.json"), "w") as fh:
        json.dump(man, fh)
    return man


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=config.LIBRARY)
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    from onset.synth import library as LIB
    ids = [s["id"] for s in LIB.SPECIMENS]
    if a.only:
        want = a.only.split(",")
        ids = [i for i in ids if i in want]
    os.makedirs(a.out, exist_ok=True)
    jobs = [(i, a.out, a.force) for i in ids]
    t0 = time.time()
    if a.workers > 1:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(build_one, jobs)):
                print(f"[{k + 1}/{len(jobs)}] {r}  elapsed {time.time() - t0:.0f}s", flush=True)
    else:
        for k, j in enumerate(jobs):
            print(f"[{k + 1}/{len(jobs)}] {build_one(j)}  elapsed {time.time() - t0:.0f}s", flush=True)
    man = write_manifest(a.out)
    write_static_exports(a.out, os.path.dirname(os.path.abspath(a.out)))
    print(f"manifest: {len(man['specimens'])} specimens", flush=True)


if __name__ == "__main__":
    main()
