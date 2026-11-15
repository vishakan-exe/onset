"""The action-unit atlas: what each AU is, what the instrument looks for, and which AUs it finds hard to tell apart.

The anatomy column follows the FACS manual's muscle basis (Ekman, Friesen & Hager 2002).  The templates are the same
ROI-level direction patterns the analysis uses (``templates.py``), so the atlas shows what the engine *expects*, not an
illustration.  Confusability is computed, not asserted: the cosine similarity of the noise-whitened, nuisance-projected
templates under a typical noise level.
"""
from __future__ import annotations

import numpy as np

from ..vision import geometry as G
from . import templates as TPL

ANATOMY = {
    1: dict(muscle="Frontalis, medial part", sees="Inner ends of the brows rise; the skin between the brows lifts.",
            looks="Brow-inner and glabella move up; the outer brow barely moves."),
    2: dict(muscle="Frontalis, lateral part", sees="Outer ends of the brows rise; the forehead lifts at the sides.",
            looks="Brow-outer and mid move up and slightly outward."),
    4: dict(muscle="Corrugator supercilii, depressor supercilii", sees="Brows drawn down and together; vertical lines between the brows.",
            looks="Brow-inner moves down and inward; the glabella drops."),
    5: dict(muscle="Levator palpebrae superioris", sees="The upper lid lifts and more of the eye shows.",
            looks="Upper-lid skin moves up; the brow follows a little."),
    6: dict(muscle="Orbicularis oculi, orbital part", sees="Cheeks rise and the skin around the eye gathers (crow's feet).",
            looks="Cheek and lower-lid skin move up; the canthi draw inward."),
    7: dict(muscle="Orbicularis oculi, palpebral part", sees="The lower lid tightens and the eye narrows.",
            looks="Lower-lid skin moves up; the upper lid moves down a little."),
    9: dict(muscle="Levator labii superioris alaeque nasi", sees="The nose wrinkles and the nostril wings rise.",
            looks="Nose wings and the nose bridge move up; the glabella is drawn down."),
    10: dict(muscle="Levator labii superioris", sees="The upper lip lifts.",
             looks="Upper-lip and the skin beside the nose move up."),
    12: dict(muscle="Zygomaticus major", sees="The mouth corners pull up and out (the smile).",
             looks="Corner and nasolabial skin move up and outward; the cheek follows."),
    14: dict(muscle="Buccinator", sees="The mouth corners tighten inward (a dimple).",
             looks="Corner skin moves inward; the effect is often one-sided."),
    15: dict(muscle="Depressor anguli oris", sees="The mouth corners pull down.",
             looks="Corner and nasolabial skin move down."),
    17: dict(muscle="Mentalis", sees="The chin boss rises and pushes the lower lip up.",
             looks="Chin and lower-lip skin move up."),
}

# AU combinations the literature groups under the usual emotion prototypes (expert prior, not data)
PROTOTYPES = [
    dict(label="surprise", aus=[1, 2, 5], note="Brows and upper lids rise."),
    dict(label="happiness", aus=[6, 12], note="Cheeks and lip corners lift."),
    dict(label="anger", aus=[4, 5, 7], note="Brows drawn down, lids tense."),
    dict(label="disgust", aus=[9, 10], note="Nose wrinkles, upper lip lifts."),
    dict(label="sadness", aus=[1, 4, 15], note="Inner brows up, corners down."),
    dict(label="fear", aus=[1, 2, 4, 5], note="Brows raised and drawn together, lids raised."),
    dict(label="contempt", aus=[14], note="One mouth corner tightens (often asymmetric)."),
]


def build_atlas(canon: dict, sigma_px: float = 0.12, top: int = 3) -> dict:
    """``canon``: the ``canon`` block of an analysis result (ROI contours and centroids in plate coordinates).
    ``sigma_px``: per-component ROI noise assumed for the confusability matrix (the webcam-class value)."""
    tmpl = TPL.au_templates()
    R = len(G.ROI_NAMES)
    sigma = np.full((R, 2), sigma_px)
    # the affine nuisance subspace does not depend on the origin or on an isotropic rescaling of the coordinates, so the
    # plate coordinates of the centroids serve
    by_name = {c["roi"]: (c["x"], c["y"]) for c in canon["centroids"]}
    cent = np.array([by_name[n] for n in G.ROI_NAMES], float)
    nuis = TPL.nuisance_columns(cent, cent.mean(0))
    ident = TPL.identifiability(sigma, None, nuis)
    aus = []
    for k, a in enumerate(TPL.AUS):
        order = [j for j in np.argsort(-ident[k]) if j != k][:top]
        vec = []
        for r, name in enumerate(G.ROI_NAMES):
            dx, dy = float(tmpl[k, r, 0]), float(tmpl[k, r, 1])
            if np.hypot(dx, dy) > 0.05:
                vec.append(dict(roi=name, dx=round(dx, 3), dy=round(dy, 3)))
        aus.append(dict(au=a, name=TPL.AU_NAMES[a], **ANATOMY[a], vectors=vec,
                        confusable=[dict(au=TPL.AUS[j], similarity=round(float(ident[k, j]), 3)) for j in order]))
    return dict(aus=aus, prototypes=PROTOTYPES, identifiability=dict(order=list(TPL.AUS), matrix=np.round(ident, 3).tolist(),
                                                                       sigma_px=sigma_px),
                canon=dict(rois=canon["rois"], centroids=canon["centroids"], plate_size=canon.get("plate_size", 176),
                           groups=canon.get("groups")),
                source=("Templates: ROI-level direction patterns written from FACS anatomy (onset/analysis/templates.py). "
                        "Confusability: cosine similarity of the noise-whitened templates."))
