"""A synthetic dataset laid out exactly like CASME II, for checking the learning machinery end to end.

Phantoms (procedural faces with exactly scripted facial movements, see ``onset.synth``) are rendered into frame folders
``<root>/sub01/EP01_01f/img1.jpg ...`` with an annotation sheet that uses CASME II's column names, so the real ``casme2``
adapter, the feature extractor, the protocols, the models, the calibration and the report all run on it unchanged.

What this is for:  proving that the code is wired correctly (subject-disjoint folds, no leakage, calibrated sets, controls that
behave) before licensed data is available.  What it is **not**:  evidence about people.  The 'identities' are variations
of one procedural face model, the emotion labels are assigned from fixed action-unit recipes (like a coding manual would),
and 15 % of clips deliberately carry a label that disagrees with their action units, the way coders disagree.  Any score
obtained here says that the pipeline works, nothing about human micro-expressions.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Optional

import cv2
import numpy as np

from ..synth.phantom import PHANTOMS
from ..synth.render import Event, Renderer
from ..synth.scenarios import make_scenario

FPS = 100.0
# CASME II-like class mix (others 39 %, disgust 25 %, happiness 13 %, repression 11 %, surprise 10 %)
CLASS_P = {"others": 0.39, "disgust": 0.25, "happiness": 0.13, "repression": 0.11, "surprise": 0.12}
RECIPES = {
    "happiness": [(6, 12), (12,), (6, 12)],
    "surprise": [(1, 2, 5), (1, 2), (5,)],
    "disgust": [(9,), (9, 10), (10,), (4, 9)],
    "repression": [(17,), (15, 17), (14,), (14, 17)],
    "others": [(4, 7), (1, 4), (7,), (2,), (4,), (12, 14)],
}
LABEL_NOISE = 0.15
VERSION = 2


def identity(i: int, rng: np.random.Generator):
    """The i-th synthetic person: a phantom with its own colouring, proportions and lighting."""
    base = PHANTOMS["ABCDEF"[i % 6]]
    k = lambda lo, hi: float(rng.uniform(lo, hi))                                   # noqa: E731
    skin = tuple(float(np.clip(c * k(0.92, 1.08), 0.2, 0.98)) for c in base.skin)
    return base.variant(name=f"S{i:02d}", seed=1000 + 17 * i, skin=skin, face_w=base.face_w * k(0.95, 1.06),
                        face_h=base.face_h * k(0.96, 1.05), brow_h=base.brow_h + k(-0.02, 0.02), nose_len=base.nose_len * k(0.94, 1.06),
                        nose_w=base.nose_w * k(0.92, 1.08), mouth_w=base.mouth_w * k(0.94, 1.06), eye_open=base.eye_open * k(0.94, 1.06),
                        light=(base.light[0] + k(-0.15, 0.15), base.light[1] + k(-0.1, 0.1)), texture=base.texture * k(0.9, 1.1))


def _labels(n: int, rng: np.random.Generator) -> list[str]:
    names = list(CLASS_P)
    lab = [names[int(rng.choice(len(names), p=np.array([CLASS_P[x] for x in names])))] for _ in range(n)]
    for j, c in enumerate(names):                                                    # every class present at least twice
        while lab.count(c) < 2 and n >= 2 * len(names):
            lab[int(rng.integers(n))] = c
    return lab


def build(root: str, subjects: int = 12, clips_per_subject: int = 6, seed: int = 0, size: int = 480, quiet: bool = False) -> dict:
    """Render the dataset into ``root`` (resumable: a finished clip folder is kept).  Returns ``dict(sheet, clips, subjects, ...)``."""
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = subjects * clips_per_subject
    labels = _labels(n, rng)
    sheet = os.path.join(root, "CASME2-coding-synthetic.csv")
    meta_path = os.path.join(root, "synthetic.json")
    want = dict(version=VERSION, subjects=subjects, clips=clips_per_subject, seed=seed, size=size)
    if os.path.exists(meta_path) and os.path.exists(sheet):
        have = json.load(open(meta_path))
        if all(have.get(k) == v for k, v in want.items()):
            return dict(sheet=sheet, root=root, clips=have["n_rows"], subjects=subjects, label_counts=have["label_counts"], reused=True)
    rows = []
    k = 0
    for si in range(subjects):
        spec = identity(si, rng)
        for ci in range(clips_per_subject):
            label = labels[k]
            k += 1
            recipe_class = label
            if rng.random() < LABEL_NOISE:                                           # the coder's label disagrees with the movement
                recipe_class = str(rng.choice([c for c in CLASS_P if c != label]))
            aus = RECIPES[recipe_class][int(rng.integers(len(RECIPES[recipe_class])))]
            t_on = float(rng.uniform(0.12, 0.18))
            t_apex = t_on + float(rng.uniform(0.11, 0.17))
            t_off = t_apex + float(rng.uniform(0.12, 0.20))
            dur = t_off + 0.14
            amp = float(np.exp(rng.uniform(np.log(0.012), np.log(0.03))))
            ev = Event({a: 1.0 for a in aus}, t_on, t_apex, t_off, amp=amp, label=label, kind="micro")
            sc = make_scenario(spec, "highspeed100", dur, [ev], seed=int(rng.integers(1, 10_000_000)),
                               nuisance="still" if rng.random() < 0.6 else "normal", size=size, name=f"{spec.name}-{ci}")
            sc.blinks = [b for b in sc.blinks if b[0] + b[1] < t_on - 0.02 or b[0] > t_off + 0.02]     # no blink on top of the movement
            clip = Renderer(sc).render()
            name = f"EP{si + 1:02d}_{ci + 1:02d}f"
            folder = os.path.join(root, f"sub{si + 1:02d}", name)
            os.makedirs(folder, exist_ok=True)
            for j, fr in enumerate(clip.frames):
                cv2.imwrite(os.path.join(folder, f"img{j + 1}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append([si + 1, name, int(round(t_on * FPS)) + 1, int(round(t_apex * FPS)) + 1, int(round(t_off * FPS)) + 1,
                         "+".join(str(a) for a in aus), label])
            if not quiet and (k % 6 == 0 or k == n):
                print(f"  synthetic clips {k}/{n}", flush=True)
    with open(sheet, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Subject", "Filename", "OnsetFrame", "ApexFrame", "OffsetFrame", "Action Units", "Estimated Emotion"])
        w.writerows(rows)
    counts = {c: labels.count(c) for c in CLASS_P}
    with open(os.path.join(root, "README.txt"), "w") as fh:
        fh.write("SYNTHETIC DATA generated by ONSET phantoms. These are procedural faces, not people. The layout copies CASME II so the\n"
                 "same code runs on it; scores obtained here check the pipeline and say nothing about real micro-expressions.\n")
    with open(meta_path, "w") as fh:
        json.dump(dict(**want, n_rows=len(rows), label_counts=counts, label_noise=LABEL_NOISE), fh)
    return dict(sheet=sheet, root=root, clips=len(rows), subjects=subjects, label_counts=counts, reused=False)
