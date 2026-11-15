"""ROI-level action-unit templates and nuisance regressors.

A template is the *expected ROI-mean displacement pattern* of one AU, written by hand from FACS
anatomy (which muscle pulls which region which way).  It is deliberately a coarser object than the
phantom generator's continuous Gaussian-blob fields (synth/deform.py): the generator adds per-event
rotation, shift, size and left/right asymmetry, so recovering an AU from a rendered clip is a genuine
inference problem and not a lookup of the generator's own table.

Conventions
-----------
* canonical-frame pixels, x to the right, y down; L/R are *image* sides.
* ``(out, up)``: ``out`` = displacement away from the face midline (positive for both sides), ``up`` =
  displacement toward the top of the image.  Midline ROIs use ``out`` as a (rare) image-x component.
* The template is scaled so that its largest ROI vector has length 1; an amplitude of ``a`` therefore
  means "the dominant ROI moves by a pixels".  ``amp_to_pct_iod`` converts to the %IOD unit the rest of
  the system reports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vision import geometry as G

AU_NAMES = {
    1: "Inner brow raiser", 2: "Outer brow raiser", 4: "Brow lowerer", 5: "Upper lid raiser",
    6: "Cheek raiser", 7: "Lid tightener", 9: "Nose wrinkler", 10: "Upper lip raiser",
    12: "Lip corner puller", 14: "Dimpler", 15: "Lip corner depressor", 17: "Chin raiser",
}
AUS = sorted(AU_NAMES)

_SIDED = ["fh", "brow_out", "brow_mid", "brow_in", "lid_up", "lid_lo", "canthus", "wing", "cheek", "nasolab", "corner"]
_MIDLINE = ["glabella", "nose_root", "lip_up", "lip_lo", "chin"]

# (out, up) per ROI family, relative units
AU_SPEC: dict[int, dict[str, tuple[float, float]]] = {
    1: dict(brow_in=(0.03, 1.0), brow_mid=(0.0, 0.5), brow_out=(0.0, 0.12), glabella=(0.0, 0.5), fh=(0.0, 0.45)),
    2: dict(brow_out=(0.12, 1.0), brow_mid=(0.08, 0.65), brow_in=(0.0, 0.25), fh=(0.05, 0.6), lid_up=(0.0, 0.15),
            glabella=(0.0, 0.15)),
    4: dict(brow_in=(-0.45, -0.9), brow_mid=(-0.15, -0.6), brow_out=(0.0, -0.2), glabella=(0.0, -0.55),
            nose_root=(0.0, -0.3), lid_up=(0.0, -0.25), fh=(0.0, -0.1)),
    5: dict(lid_up=(0.0, 0.9), brow_mid=(0.0, 0.2), brow_in=(0.0, 0.15)),
    6: dict(cheek=(-0.3, 0.9), lid_lo=(0.0, 0.65), canthus=(-0.6, 0.4), nasolab=(0.0, 0.3), brow_out=(0.0, -0.25)),
    7: dict(lid_lo=(0.0, 0.8), lid_up=(0.0, -0.3), canthus=(-0.3, 0.1), cheek=(0.0, 0.3)),
    9: dict(wing=(-0.1, 1.0), nasolab=(0.0, 0.5), lip_up=(0.0, 0.5), nose_root=(0.0, 0.3), glabella=(0.0, -0.4),
            brow_in=(-0.1, -0.3), cheek=(0.0, 0.2)),
    10: dict(lip_up=(0.0, 1.0), nasolab=(0.1, 0.8), wing=(0.0, 0.45), cheek=(0.0, 0.25), corner=(0.0, 0.15)),
    12: dict(corner=(0.55, 0.85), nasolab=(0.4, 0.7), cheek=(0.25, 0.5), lip_up=(0.0, 0.3), lip_lo=(0.0, 0.15),
             lid_lo=(0.0, 0.2), wing=(0.0, 0.2)),
    14: dict(corner=(-0.8, 0.2), nasolab=(-0.3, 0.1), cheek=(-0.15, 0.0), lip_lo=(0.0, 0.15)),
    15: dict(corner=(0.15, -1.0), nasolab=(0.0, -0.4), chin=(0.0, -0.2), lip_lo=(0.0, -0.3)),
    17: dict(chin=(0.0, 1.0), lip_lo=(0.0, 0.7), corner=(-0.1, 0.1)),
}

# nuisance motion that is *not* an AU (used as unconstrained regressors or for masking)
BLINK_SPEC = dict(lid_up=(0.0, -1.0), lid_lo=(0.0, 0.3), brow_mid=(0.0, -0.3), cheek=(0.0, 0.1))
SPEECH_SPEC = dict(lip_lo=(0.0, -1.0), chin=(0.0, -0.9), lip_up=(0.0, 0.2), corner=(0.3, -0.1), cheek=(0.0, -0.3))


def _expand(spec: dict[str, tuple[float, float]]) -> np.ndarray:
    """-> (R, 2) displacement per ROI in px units (peak length normalised to 1)."""
    t = np.zeros((len(G.ROI_NAMES), 2))
    idx = {n: i for i, n in enumerate(G.ROI_NAMES)}
    for fam, (out, up) in spec.items():
        if fam in _MIDLINE:
            t[idx[fam]] = (out, -up)
        else:
            t[idx[f"{fam}_L"]] = (-out, -up)
            t[idx[f"{fam}_R"]] = (+out, -up)
    peak = np.linalg.norm(t, axis=1).max()
    return t / peak if peak > 0 else t


def au_templates() -> np.ndarray:
    """(K, R, 2) unit-peak ROI displacement templates in the order of ``AUS``."""
    return np.stack([_expand(AU_SPEC[a]) for a in AUS])


def blink_template() -> np.ndarray:
    return _expand(BLINK_SPEC)


def speech_template() -> np.ndarray:
    return _expand(SPEECH_SPEC)


SCAN_KINDS = [("au", a) for a in AUS] + [("blink", 0), ("speech", 0)]


def scan_templates() -> np.ndarray:
    """(K + 2, R, 2): the AU templates followed by the blink and speech templates.  Non-AU movements are scanned
    for too, so their central lobe wins over the side lobes that a transient statistic leaves at +-k."""
    return np.concatenate([au_templates(), blink_template()[None], speech_template()[None]], axis=0)


def scan_label(i: int) -> str:
    kind, a = SCAN_KINDS[i]
    return f"AU{a}" if kind == "au" else kind


def roi_centroids(rois: dict) -> np.ndarray:
    """(R, 2) centroid (x, y) of every ROI mask in canonical px."""
    out = np.zeros((len(G.ROI_NAMES), 2))
    for i, n in enumerate(G.ROI_NAMES):
        ys, xs = np.nonzero(rois[n])
        out[i] = (xs.mean(), ys.mean()) if len(xs) else (np.nan, np.nan)
    return out


def similarity_columns(centroids: np.ndarray, centre) -> np.ndarray:
    """(4, R, 2): translation x/y (1 px each), isotropic scale (per 0.01) and rotation (per 0.01 rad).

    These span the residual rigid motion that the pixel-level stage could not remove; they are fitted
    jointly with the AU templates so that a head-motion error cannot masquerade as an AU (or the reverse).
    """
    px, py = centroids[:, 0] - centre[0], centroids[:, 1] - centre[1]
    cols = np.zeros((4, len(px), 2))
    cols[0, :, 0] = 1.0
    cols[1, :, 1] = 1.0
    cols[2, :, 0], cols[2, :, 1] = 0.01 * px, 0.01 * py
    cols[3, :, 0], cols[3, :, 1] = -0.01 * py, 0.01 * px
    return cols


def affine_extra_columns(centroids: np.ndarray, centre) -> np.ndarray:
    """(2, R, 2): the two remaining directions of a full 2-D affine motion - anisotropic scale (the face squashed
    vertically while stretched horizontally, per 0.01) and shear (per 0.01).

    Together with the four similarity columns they span every weak-perspective motion of a roughly planar face:
    a head nod or turn foreshortens the face along one axis, zoom 'breathing' and codec artefacts do similar
    things.  Without these two columns such a nod shows up as an expression (it did: single-frame 'AU12' events of
    11 sigma on null clips)."""
    px, py = centroids[:, 0] - centre[0], centroids[:, 1] - centre[1]
    cols = np.zeros((2, len(px), 2))
    cols[0, :, 0], cols[0, :, 1] = 0.01 * px, -0.01 * py
    cols[1, :, 0], cols[1, :, 1] = 0.01 * py, 0.01 * px
    return cols


def gaze_column() -> np.ndarray:
    """(1, R, 2): both eyes' lid skin shifts in the same image direction (conjugate gaze)."""
    c = np.zeros((1, len(G.ROI_NAMES), 2))
    for n in ("lid_up_L", "lid_up_R", "lid_lo_L", "lid_lo_R"):
        c[0, G.ROI_NAMES.index(n), 0] = 0.5
    return c


def nuisance_columns(centroids: np.ndarray, centre) -> np.ndarray:
    """(7, R, 2): the 6 affine motion directions (2 translation, scale, rotation, anisotropic scale, shear) + gaze."""
    return np.concatenate([similarity_columns(centroids, centre), affine_extra_columns(centroids, centre), gaze_column()], axis=0)


# --- what each AU *looks like* to the instrument ------------------------------------------------------
def au_roi_weights() -> np.ndarray:
    """(K, R) share of each template's energy in each ROI (for 'which regions' explanations)."""
    t = au_templates()
    e = (t ** 2).sum(-1)
    return e / e.sum(1, keepdims=True)


def identifiability(sigma: np.ndarray | None = None, mask: np.ndarray | None = None, nuisance: np.ndarray | None = None
                    ) -> np.ndarray:
    """(K, K) cosine similarity of noise-whitened, nuisance-projected templates: how easily two AUs are confused
    *with this ROI set and noise*.  1 = indistinguishable, 0 = orthogonal."""
    T = au_templates().reshape(len(AUS), -1)
    w = np.ones(T.shape[1]) if sigma is None else 1.0 / sigma.reshape(-1)
    if mask is not None:
        w = w * np.repeat(mask, 2)
    X = T * w
    if nuisance is not None:
        N = nuisance.reshape(nuisance.shape[0], -1) * w
        Q, _ = np.linalg.qr(N.T)
        X = X - (X @ Q) @ Q.T
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / n
    return Xn @ Xn.T


def amp_to_pct_iod(amp_px: np.ndarray | float, iod_canon: float) -> np.ndarray | float:
    return np.asarray(amp_px) / iod_canon * 100.0
