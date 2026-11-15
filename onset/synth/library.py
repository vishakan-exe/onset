"""The specimen library: scripted synthetic clips ("phantoms") with exact ground truth.

A phantom is a procedurally rendered face - not a photograph and not a person - whose movements are scripted
frame by frame, so every specimen comes with the answer: which action units moved, when, and by how much.  They
exist for three reasons the interface states openly:

* an instrument needs calibration targets; real footage never comes with ground truth at the 0.5 % IOD level,
* the demo has to work without footage, a webcam or anybody's face,
* nothing here involves consent, licences or privacy.

A specimen is *not* evidence about how the system performs on people.  That is what the licensed-dataset
evaluation (``onset.learn``) is for; the interface keeps the two apart.

Every specimen is a small dict (see ``SPECIMENS``); ``scenario_for`` turns it into a renderer scenario, with
optional overrides (camera preset, nuisance level, amplitude scale, camera parameters) that the Stress Lab uses to
re-film the same movement under worse conditions.
"""
from __future__ import annotations

import copy
import os
import subprocess
from typing import Optional

import numpy as np

from .deform import Jitter
from .render import Camera, Event, Renderer, Scenario
from .scenarios import CAMERAS, NUISANCE_LEVELS, make_scenario

GROUPS = [
    ("first", "The first look", "A movement too small to see, and the instrument that finds it."),
    ("categories", "One movement at a time", "Single scripted movements, from a lip corner to a lowered brow."),
    ("ambiguous", "Hard to tell apart", "Pairs that differ by one action unit. Compare them side by side."),
    ("nuisance", "Not expressions", "Blinks, talking, glances and head motion: movement that is not a micro-expression."),
    ("null", "Nothing happened", "Faces that did nothing. The instrument should say so."),
    ("macro", "Too long to be micro", "Real expressions, and two in a row."),
    ("ladder", "Same movement, five cameras", "One scripted smile, filmed with different cameras."),
]


def _ev(aus, on, apex, off, amp_pct, label="", kind="micro", side=None):
    return dict(aus=list(aus), t_on=on, t_apex=apex, t_off=off, amp_pct=amp_pct, label=label, kind=kind, side=side)


SPECIMENS: list[dict] = [
    # ---------------------------------------------------------------------------------------- first look
    dict(id="invisible-frown", group="first", title="The invisible frown", phantom="A", camera="webcam30", nuisance="still",
         duration=8.0, seed=11, events=[_ev([4], 3.90, 4.08, 4.30, 0.8, "negative")],
         blurb="A frown that lasts four tenths of a second and moves the brows by less than a pixel.",
         look="Watch the brows between 3.9 s and 4.3 s.", confusable=["scowl", "worried-brows"]),
    # ---------------------------------------------------------------------------------------- categories
    dict(id="smile-flicker", group="categories", title="Smile flicker", phantom="B", camera="webcam30", nuisance="normal",
         duration=8.0, seed=21, events=[_ev([6, 12], 3.20, 3.38, 3.65, 1.6, "positive")],
         blurb="Cheeks and lip corners lift together for under half a second.", look="Corners of the mouth and cheeks.",
         confusable=["mouth-corner-only", "slow-smile"]),
    dict(id="mouth-corner-only", group="categories", title="Mouth corner, no cheek", phantom="C", camera="webcam30",
         nuisance="normal", duration=8.0, seed=22, events=[_ev([12], 3.00, 3.15, 3.40, 1.2, "positive")],
         blurb="The lip corner pulls up and the cheek stays put.", look="Compare the cheeks with the smile flicker.",
         confusable=["smile-flicker", "one-sided-dimple"]),
    dict(id="brow-flash", group="categories", title="Brow flash", phantom="D", camera="webcam30", nuisance="normal",
         duration=8.0, seed=23, events=[_ev([1, 2], 2.80, 2.95, 3.20, 1.5, "surprise")],
         blurb="Both brows lift; the upper lids do not.", look="Brows only.", confusable=["startle"]),
    dict(id="startle", group="categories", title="Startle", phantom="E", camera="webcam30", nuisance="normal",
         duration=8.0, seed=24, events=[_ev([1, 2, 5], 3.00, 3.15, 3.40, 2.0, "surprise")],
         blurb="Brows and upper lids rise together.", look="Brows and the white above the iris.", confusable=["brow-flash", "four-actions"]),
    dict(id="four-actions", group="categories", title="Four actions at once", phantom="F", camera="webcam30",
         nuisance="normal", duration=8.0, seed=25, events=[_ev([1, 2, 4, 5], 3.10, 3.25, 3.52, 1.6, "negative")],
         blurb="Raised brows pulled together with raised lids. The instrument considers at most three actions at a time, "
               "so it cannot name all four.", look="A known limit of the model.", confusable=["startle", "scowl"]),
    dict(id="scowl", group="categories", title="Scowl", phantom="A", camera="webcam30", nuisance="normal", duration=8.0,
         seed=26, events=[_ev([4, 5, 7], 3.40, 3.58, 3.85, 1.5, "negative")],
         blurb="Brows drawn down, upper lids raised, lower lids tightened.", look="Brow, lids.", confusable=["worried-brows", "invisible-frown"]),
    dict(id="nose-wrinkle", group="categories", title="Nose wrinkle", phantom="B", camera="webcam30", nuisance="normal",
         duration=8.0, seed=27, events=[_ev([9], 3.00, 3.17, 3.42, 1.5, "negative")],
         blurb="The skin beside the nose bridge draws up.", look="The nose wings and the bridge.", confusable=["lip-raise"]),
    dict(id="lip-raise", group="categories", title="Upper lip raise", phantom="C", camera="webcam30", nuisance="normal",
         duration=8.0, seed=28, events=[_ev([10], 3.00, 3.17, 3.42, 1.5, "negative")],
         blurb="The upper lip lifts toward the nose.", look="Above the mouth.", confusable=["nose-wrinkle"]),
    dict(id="sunken-corners", group="categories", title="Sunken corners", phantom="D", camera="webcam30", nuisance="normal",
         duration=8.0, seed=29, events=[_ev([15, 17], 3.30, 3.48, 3.75, 1.5, "negative")],
         blurb="Lip corners pull down while the chin pushes up.", look="Mouth corners and chin.", confusable=["chin-push"]),
    dict(id="chin-push", group="categories", title="Chin push", phantom="A", camera="webcam30", nuisance="normal",
         duration=8.0, seed=30, events=[_ev([17], 3.30, 3.48, 3.75, 1.4, "negative")],
         blurb="The chin boss rises and pushes the lower lip up.", look="Chin.", confusable=["sunken-corners"]),
    dict(id="one-sided-dimple", group="categories", title="One-sided dimple", phantom="E", camera="webcam30",
         nuisance="normal", duration=8.0, seed=31, events=[_ev([14], 3.20, 3.37, 3.62, 1.5, "negative", side=0.85)],
         blurb="One mouth corner tightens inward. The movement is asymmetric.", look="Image-left corner of the mouth.",
         confusable=["mouth-corner-only"]),
    dict(id="worried-brows", group="categories", title="Worried brows", phantom="F", camera="webcam30", nuisance="normal",
         duration=8.0, seed=32, events=[_ev([1, 4], 3.00, 3.18, 3.45, 1.4, "negative")],
         blurb="Inner brows rise and draw together.", look="The inner ends of the brows.", confusable=["scowl", "brow-flash"]),
    # ---------------------------------------------------------------------------------------- ambiguous / at the limit
    dict(id="faint-smile", group="ambiguous", title="At the limit", phantom="B", camera="webcam30", nuisance="normal",
         duration=8.0, seed=33, events=[_ev([12], 3.20, 3.38, 3.65, 0.35, "positive")],
         blurb="A lip-corner movement of a third of a percent of the eye distance: about where this camera stops being "
               "able to tell it from noise.", look="Whether the instrument speaks up, or stays quiet.", confusable=["mouth-corner-only"]),
    dict(id="soft-nose-lip", group="ambiguous", title="Nose or lip?", phantom="D", camera="webcam24_lowlight", nuisance="normal",
         duration=8.0, seed=34, events=[_ev([9, 10], 3.10, 3.27, 3.52, 0.9, "negative")],
         blurb="Both nose and upper-lip movement, filmed in dim light. Their patterns overlap.", look="Which of the two the evidence supports.",
         confusable=["nose-wrinkle", "lip-raise"]),
    dict(id="quiet-brows", group="ambiguous", title="Quiet brows", phantom="C", camera="webcam30", nuisance="normal",
         duration=8.0, seed=35, events=[_ev([2], 3.00, 3.15, 3.40, 0.7, "surprise")],
         blurb="An outer-brow lift that is barely above the noise.", look="Whether it is named, and how sure the readout is.",
         confusable=["brow-flash"]),
    # ---------------------------------------------------------------------------------------- nuisance
    dict(id="blink-storm", group="nuisance", title="Blinking", phantom="C", camera="webcam30", nuisance="normal", duration=8.0,
         seed=41, events=[], blurb="Nothing but blinks. They are the most common movement in any face video.",
         look="Blinks are recognised and set aside.", confusable=[]),
    dict(id="talking", group="nuisance", title="Talking", phantom="D", camera="webcam30", nuisance="busy", duration=8.0,
         seed=42, events=[], blurb="Mouth movement from speech, with a restless head.", look="The mouth is excluded; nothing is called an expression.",
         confusable=[]),
    dict(id="head-swing", group="nuisance", title="Head swing", phantom="E", camera="webcam30", nuisance="busy", duration=8.0,
         seed=43, events=[_ev([12], 4.00, 4.18, 4.45, 1.8, "positive")],
         blurb="A restless head and a smile flicker on top of it.", look="Whether the stabilisation holds.", confusable=["smile-flicker"]),
    dict(id="gaze-darts", group="nuisance", title="Eyes darting", phantom="F", camera="webcam30", nuisance="normal", duration=8.0,
         seed=44, events=[], blurb="Eye movement shifts the skin of the lids without any expression.",
         look="Glances are not facial expressions.", confusable=[]),
    dict(id="codec-smear", group="nuisance", title="Compressed phone video", phantom="A", camera="phone30", nuisance="normal",
         duration=8.0, seed=45, events=[_ev([4, 7], 3.30, 3.48, 3.76, 1.6, "negative")],
         blurb="Video compression smooths small movements and invents others.", look="What survives the codec.",
         confusable=["scowl"]),
    # ---------------------------------------------------------------------------------------- null
    dict(id="still-face", group="null", title="Still face", phantom="B", camera="webcam30", nuisance="still", duration=8.0,
         seed=51, events=[], blurb="Eight seconds of a face that does nothing.", look="The instrument should report no movement.",
         confusable=[]),
    dict(id="sitting-through", group="null", title="Sitting through it", phantom="C", camera="webcam30", nuisance="normal",
         duration=10.0, seed=52, events=[], blurb="Ten seconds with ordinary blinks and glances and no expression.",
         look="How often it stays quiet.", confusable=[]),
    # ---------------------------------------------------------------------------------------- macro
    dict(id="slow-smile", group="macro", title="A slow smile", phantom="D", camera="webcam30", nuisance="normal", duration=8.0,
         seed=61, events=[_ev([6, 12], 2.80, 3.40, 4.00, 3.0, "positive", kind="macro")],
         blurb="A full expression lasting more than a second. Not a micro-expression.", look="The duration is reported.",
         confusable=["smile-flicker"]),
    dict(id="two-in-a-row", group="macro", title="Two in a row", phantom="E", camera="webcam30", nuisance="normal", duration=8.0,
         seed=62, events=[_ev([4], 2.40, 2.57, 2.82, 1.3, "negative"), _ev([12], 4.60, 4.78, 5.05, 1.6, "positive")],
         blurb="A frown, then a smile corner two seconds later.", look="Two separate events.", confusable=[]),
    # ---------------------------------------------------------------------------------------- ladder
    dict(id="ladder-highspeed", group="ladder", title="High-speed camera, 100 frames a second", phantom="A", camera="highspeed100",
         nuisance="normal", duration=4.0, seed=71, events=[_ev([6, 12], 1.60, 1.78, 2.05, 1.4, "positive")],
         blurb="The same scripted smile, filmed at 100 frames a second.", look="Timing resolution.", confusable=["ladder-webcam"]),
    dict(id="ladder-webcam", group="ladder", title="Webcam, 30 frames a second", phantom="A", camera="webcam30", nuisance="normal",
         duration=4.0, seed=71, events=[_ev([6, 12], 1.60, 1.78, 2.05, 1.4, "positive")],
         blurb="The same scripted smile, filmed by an ordinary webcam.", look="", confusable=["ladder-lowlight"]),
    dict(id="ladder-lowlight", group="ladder", title="Webcam in dim light, 24 frames a second", phantom="A",
         camera="webcam24_lowlight", nuisance="normal", duration=4.0, seed=71,
         events=[_ev([6, 12], 1.60, 1.78, 2.05, 1.4, "positive")], blurb="The same smile with more sensor noise.", look="", confusable=["ladder-webcam"]),
    dict(id="ladder-slow", group="ladder", title="Slow camera, 15 frames a second", phantom="A", camera="webcam15", nuisance="normal",
         duration=4.0, seed=71, events=[_ev([6, 12], 1.60, 1.78, 2.05, 1.4, "positive")],
         blurb="The same smile with a frame every 67 ms. The whole movement spans four frames.", look="", confusable=["ladder-webcam"]),
    dict(id="ladder-phone", group="ladder", title="Phone video, compressed", phantom="A", camera="phone30", nuisance="normal",
         duration=4.0, seed=71, events=[_ev([6, 12], 1.60, 1.78, 2.05, 1.4, "positive")],
         blurb="The same smile after H.264 compression.", look="", confusable=["ladder-webcam"]),
]
BY_ID = {s["id"]: s for s in SPECIMENS}


def get(spec_id: str) -> dict:
    return BY_ID[spec_id]


KEEP_CLEAR = 0.45          # seconds: no scripted blink or gaze shift within this distance of a scripted movement (spec["keep_clear"] overrides)


def scenario_for(spec: dict, camera: Optional[str] = None, nuisance: Optional[str] = None, amp_scale: float = 1.0,
                 seed_offset: int = 0, camera_overrides: Optional[dict] = None, duration: Optional[float] = None) -> Scenario:
    """Renderer scenario for a specimen, optionally re-filmed: other camera preset, nuisance level, scaled amplitude."""
    events = []
    for e in spec["events"]:
        jit = None
        if e.get("side") is not None:
            jit = Jitter.sample(np.random.default_rng(spec["seed"] + 5), 1.0, side=e["side"])
        events.append(Event({int(a): 1.0 for a in e["aus"]}, e["t_on"], e["t_apex"], e["t_off"], amp=e["amp_pct"] / 100.0 * amp_scale,
                            label=e.get("label", ""), kind=e.get("kind", "micro"), jitter=jit))
    cam = camera or spec["camera"]
    if cam not in CAMERAS:
        raise ValueError(f"unknown camera preset {cam!r}")
    nz = nuisance or spec["nuisance"]
    if nz not in NUISANCE_LEVELS:
        raise ValueError(f"unknown nuisance level {nz!r}")
    return make_scenario(spec["phantom"], cam, duration or spec["duration"], events, seed=spec["seed"] + seed_offset, nuisance=nz,
                         name=spec["id"], camera_overrides=camera_overrides, keep_clear=float(spec.get("keep_clear", KEEP_CLEAR)))


def render_specimen(spec: dict, **kw):
    return Renderer(scenario_for(spec, **kw)).render()


def render_with_renderer(spec: dict, **kw):
    """(renderer, clip): the renderer is what ``onset.synth.truth`` needs to express the scripted truth in the instrument's units."""
    rend = Renderer(scenario_for(spec, **kw))
    return rend, rend.render()


def truth_json(clip, spec: Optional[dict] = None, exact: Optional[list] = None) -> dict:
    """Ground truth in the units the interface shows (percent of the inter-ocular distance, seconds).

    ``exact`` (from ``onset.synth.truth.exact_events``) expresses every scripted movement in the instrument's own unit: the
    largest displacement of any region (``size_*``, what the sensitivity curves are drawn against) and the amplitude each action
    unit would be read at by a perfect measurement (``au_*``: the dominant region of the AU's template moves by that much).
    ``amp_pct_iod`` / ``amp_px`` then carry the size in those units; the generator's own scripting parameter is kept as
    ``generator_pct_iod`` for the record, because it is not a quantity anyone could read off a face."""
    t = clip.truth
    px_per_pct = clip.iod_px / 100.0
    events = []
    for i, e in enumerate(t["events"]):
        gen = round(e["amp_iod"] * 100, 3)
        row = dict(aus=sorted(int(a) for a in e["aus"]), t_on=e["t_on"], t_apex=e["t_apex"], t_off=e["t_off"],
                   amp_pct_iod=gen, amp_px=round(gen * px_per_pct, 3), generator_pct_iod=gen,
                   label=e["label"], kind=e["kind"], side=round(e["side"], 3), units="generator")
        if exact is not None and i < len(exact):
            x = exact[i]
            row.update(units="instrument", size_pct_iod=round(x["size_pct"], 3), size_px=round(x["size_pct"] * px_per_pct, 3),
                       amp_pct_iod=round(x["size_pct"], 3), amp_px=round(x["size_pct"] * px_per_pct, 3),
                       au_pct_iod={str(a): round(v, 3) for a, v in x["a_true_pct"].items()},
                       au_px={str(a): round(v * px_per_pct, 3) for a, v in x["a_true_pct"].items()},
                       fit_r2=round(x["fit_r2"], 3) if np.isfinite(x["fit_r2"]) else None, pair_gain=round(x["pair_gain"], 3))
        events.append(row)
    return dict(
        events=events,
        blinks=[dict(t_on=round(b["t_on"], 3), t_off=round(b["t_off"], 3)) for b in t["blinks"]],
        speech=[dict(t_on=round(s["t_on"], 3), t_off=round(s["t_off"], 3)) for s in t["speech"]],
        gaze=[dict(t=round(g["t"], 3), dx=g["dx"]) for g in t["gaze"]],
        iod_px=clip.iod_px, fps=clip.fps, duration=float(clip.times[-1] - clip.times[0]), camera=clip.scenario.name)


def write_videos(frames: np.ndarray, fps: float, out_base: str, size: Optional[int] = None) -> list[str]:
    """Encode the raw (colour) frames as MP4 (H.264) and WebM (VP9) so every browser can play them.  Returns the files."""
    n, h, w, _ = frames.shape
    if size and size != w:
        import cv2
        frames = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames])
        n, h, w, _ = frames.shape
    outs = []
    raw = frames.tobytes()
    for ext, args in (("mp4", ["-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]),
                      ("webm", ["-c:v", "libvpx-vp9", "-crf", "26", "-b:v", "0", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4",
                                "-pix_fmt", "yuv420p"])):
        path = f"{out_base}.{ext}"
        try:
            p = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                                "-r", f"{fps:g}", "-i", "-", *args, path], input=raw, capture_output=True, timeout=300)
            if p.returncode == 0 and os.path.exists(path):
                outs.append(path)
        except Exception:
            pass
    return outs


def describe_truth(truth: dict) -> list[str]:
    from ..analysis import templates as TPL
    lines = []
    for e in truth["events"]:
        names = " + ".join(f"AU{a} {TPL.AU_NAMES[a].lower()}" for a in e["aus"])
        if e.get("units") == "instrument":
            size = f"{e['size_pct_iod']:.2f} % of the eye distance at its largest region ({e['size_px']:.2f} px here)"
            if len(e["aus"]) > 1:
                size += "; read per action unit " + ", ".join(f"AU{a} {e['au_pct_iod'][str(a)]:.2f} %" for a in e["aus"])
        else:
            size = f"{e['amp_pct_iod']:.2f} % of the eye distance ({e['amp_px']:.2f} px here)"
        lines.append(f"{e['t_apex']:.2f} s: {names}, {size}, {(e['t_off'] - e['t_on']) * 1000:.0f} ms"
                     + (" (longer than a micro-expression)" if e["kind"] == "macro" else ""))
    if not truth["events"]:
        lines.append("No expression was scripted.")
    if truth["blinks"]:
        lines.append(f"{len(truth['blinks'])} blink(s) at " + ", ".join(f"{b['t_on']:.1f} s" for b in truth["blinks"]))
    if truth["speech"]:
        lines.append("Mouth movement (speech) at " + ", ".join(f"{s['t_on']:.1f}-{s['t_off']:.1f} s" for s in truth["speech"]))
    if truth["gaze"]:
        lines.append(f"{len(truth['gaze'])} gaze shift(s)")
    return lines
