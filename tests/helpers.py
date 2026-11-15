"""Small builders used by several test modules."""
import numpy as np

from onset.learn import features as F


def make_clipset(n_subjects=12, per_subject=6, n_classes=3, signal=1.0, identity=0.0, seed=0, label_by_subject=False):
    """Random clip features.  ``signal``: how strongly the class shows in the displacement series; ``identity``: how strongly
    a per-subject fingerprint is added; ``label_by_subject``: each subject mostly has one class (so identity predicts the label)."""
    rng = np.random.default_rng(seed)
    T, R = 24, F.R
    N = n_subjects * per_subject
    groups = np.repeat([f"s{i:02d}" for i in range(n_subjects)], per_subject)
    if label_by_subject:
        y = np.repeat(rng.integers(0, n_classes, n_subjects), per_subject)
        flip = rng.random(N) < 0.1
        y = np.where(flip, rng.integers(0, n_classes, N), y)
    else:
        y = rng.integers(0, n_classes, N)
    series = rng.normal(0, 0.3, (N, T, R, 2)).astype(np.float32)
    finger = rng.normal(0, 1.0, (n_subjects, R, 2)).astype(np.float32)
    for i in range(N):
        series[i, 8:16, (y[i] * 5):(y[i] * 5 + 5), :] += signal
        series[i, 8:16] += identity * finger[int(groups[i][1:])]
    feats = {}
    for i in range(N):
        pip = (rng.random(F.K) * 0.3).astype(np.float32)                    # AU evidence carries the label only as strongly as the series does
        pip[y[i] % F.K] += 0.5 * min(signal, 1.0)
        f = F.ClipFeatures(uid=f"u{i}", series=series[i], apex=series[i, 12], apex_frac=0.5, sigma_pct=np.full((R, 2), 0.05, np.float32),
                           pip=pip, amp=pip * 0.5, p_null=0.1, set_aus=np.eye(F.K, dtype=bool)[:3], set_p=np.array([0.5, 0.3, 0.2], np.float32),
                           meta=dict(duration_s=0.5, seconds=1.0))
        feats[f.uid] = f
    X = F.stack([f"u{i}" for i in range(N)], feats)
    return X, y.astype(int), groups


# ---- a plausible canonical layout of the 27 regions (256 px canvas, 64 px between the eyes), for tests that need no face
_FAMILY_XY = dict(fh=(28, -68), brow_out=(36, -43), brow_mid=(22, -42), brow_in=(9, -40), lid_up=(22, -28), lid_lo=(22, -16),
                  canthus=(38, -22), wing=(12, 22), cheek=(38, 12), nasolab=(22, 24), corner=(22, 44))
_MID_XY = dict(glabella=(0, -38), nose_root=(0, -20), lip_up=(0, 37), lip_lo=(0, 54), chin=(0, 77))


def fake_geometry():
    """(centroids (27, 2), centre, iod_canon) with the real region order."""
    from onset.vision import geometry as G
    cx, cy = 128.0, 128.0
    c = np.zeros((len(G.ROI_NAMES), 2))
    for i, n in enumerate(G.ROI_NAMES):
        if n in _MID_XY:
            dx, dy = _MID_XY[n]
            c[i] = (cx + dx, cy + dy)
        else:
            fam, side = n[:-2], n[-1]
            dx, dy = _FAMILY_XY[fam]
            c[i] = (cx + (-dx if side == "L" else dx), cy + dy)
    return c, (cx, cy), 64.0
