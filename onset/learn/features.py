"""From a labelled clip (frame folder + annotated onset/apex/offset) to the arrays the models learn from.

The measurement chain is the instrument's own - landmarks, eye-aligned stabilisation, dense flow, texture-weighted ROI
means - so a model trained here is a model of what the instrument can *measure*, and what it learns from is the same
representation that the report shows.

Per clip (all displacements in **percent of the inter-ocular distance**, so they compare across resolutions):

``series``    (T, 27, 2)  displacement of every region relative to the onset frame, resampled to T steps over [onset, offset];
``apex``      (27, 2)     the same at the annotated apex (the strongest frame when the sheet gives none);
``pip, amp``  (12,)       the AU engine's posterior inclusion probabilities and amplitudes on the apex pattern;
``sets``      top AU-set hypotheses with probabilities, for the expert category map;
``hand``      hand-crafted summary (see ``HAND_NAMES``): apex pattern, regional magnitudes, group magnitudes, timing, AU evidence.

Nothing identifying is stored: arrays of numbers only (no pixels, no landmarks), written to ``cache/<hash>.npz``.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from ..analysis import templates as TPL
from ..analysis.au_engine import AUEngine
from ..vision import geometry as G
from ..vision.measure import Measurer
from ..vision.register import register, track_landmarks
from ..vision.sources import ArraySource
from .datasets import ClipRecord, resolve_frames

R = len(G.ROI_NAMES)
K = len(TPL.AUS)
FEATURE_VERSION = 3
GROUPS = list(G.GROUP_ORDER)
_G_IDX = [[G.ROI_NAMES.index(n) for n in G.ROI_GROUPS[g]] for g in GROUPS]


@dataclass
class FeatureConfig:
    steps: int = 24                 # time steps of ``series``
    max_side: int = 960             # frames larger than this are shrunk (area interpolation) before tracking
    max_frames: int = 400           # clips longer than this are sub-sampled uniformly (flow cost)
    fps_default: float = 100.0      # used when the dataset adapter gives none
    threads: int = 1
    calibration: Optional[dict] = None   # engine parameters (data/calibration.json); None = engine defaults

    def key(self) -> str:
        c = self.calibration or {}
        blob = json.dumps(dict(v=FEATURE_VERSION, steps=self.steps, side=self.max_side, mf=self.max_frames, fps=self.fps_default,
                               eta=c.get("eta"), temper=c.get("temper"), sp=c.get("size_prior")), sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]


@dataclass
class ClipFeatures:
    uid: str
    series: np.ndarray
    apex: np.ndarray
    apex_frac: float
    sigma_pct: np.ndarray
    pip: np.ndarray
    amp: np.ndarray
    p_null: float
    set_aus: np.ndarray             # (S, K) bool
    set_p: np.ndarray               # (S,)
    meta: dict = field(default_factory=dict)

    def hand(self) -> np.ndarray:
        return hand_features(self)


class FeatureError(Exception):
    """A clip that cannot be measured (no face, too few frames, unreadable images)."""


# --------------------------------------------------------------------------------------------------------------
def mirror_index() -> np.ndarray:
    """Index that swaps left and right regions (centre regions map to themselves)."""
    idx = np.arange(R)
    for i, n in enumerate(G.ROI_NAMES):
        if n.endswith("_L"):
            idx[i] = G.ROI_NAMES.index(n[:-2] + "_R")
        elif n.endswith("_R"):
            idx[i] = G.ROI_NAMES.index(n[:-2] + "_L")
    return idx


def mirror(series: np.ndarray) -> np.ndarray:
    """Mirror image of displacement arrays ``(..., R, 2)``: swap sides and flip the horizontal component."""
    out = series[..., mirror_index(), :].copy()
    out[..., 0] *= -1.0
    return out


def hand_features(f: ClipFeatures) -> np.ndarray:
    """Fixed-length summary vector (see ``HAND_NAMES``)."""
    ap = f.apex.reshape(-1)
    mag = np.linalg.norm(f.apex, axis=1)
    gmag = np.array([mag[ix].mean() for ix in _G_IDX])
    path = np.linalg.norm(np.diff(f.series, axis=0), axis=2).sum(0)           # (R,) length of each region's path
    total_mag = np.linalg.norm(f.series, axis=2)                              # (T, R)
    energy = total_mag.mean(1)                                                # (T,)
    pk = int(np.argmax(energy))
    dur = float(f.meta.get("duration_s", 0.0))
    speed = float(np.max(np.abs(np.diff(energy))) * (len(energy) - 1) / dur) if dur > 0 else 0.0
    # path length over twice the peak excursion: 1 for a clean there-and-back, larger for wandering or noisy traces
    tort = float(path.sum() / max(2.0 * total_mag.max(0).sum(), 1e-6))
    timing = np.array([f.apex_frac, pk / max(len(energy) - 1, 1), dur, speed, float(np.log1p(energy.max())),
                       (float(np.clip(tort, 1.0, 10.0)) - 1.0) / 9.0])
    au = np.concatenate([f.pip, np.log1p(f.amp * 4.0), [f.p_null]])
    return np.concatenate([ap, mag, gmag, timing, au]).astype(np.float32)


HAND_NAMES = ([f"apex_{n}_{c}" for n in G.ROI_NAMES for c in "xy"] + [f"mag_{n}" for n in G.ROI_NAMES] + [f"group_{g}" for g in GROUPS]
              + ["apex_frac", "peak_frac", "duration_s", "peak_speed", "log_peak_energy", "tortuosity"]
              + [f"pip_AU{a}" for a in TPL.AUS] + [f"amp_AU{a}" for a in TPL.AUS] + ["p_null"])
HAND_BLOCKS = dict(pattern=slice(0, 2 * R + R + len(GROUPS)), timing=slice(2 * R + R + len(GROUPS), 2 * R + R + len(GROUPS) + 6),
                   au=slice(2 * R + R + len(GROUPS) + 6, None))


# --------------------------------------------------------------------------------------------------------------
def _load_frames(paths: list[str], cfg: FeatureConfig) -> list[np.ndarray]:
    if len(paths) > cfg.max_frames:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, cfg.max_frames).round().astype(int)]
    out = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise FeatureError(f"unreadable image {os.path.basename(p)}")
        s = max(im.shape[:2]) / cfg.max_side
        if s > 1.0:
            im = cv2.resize(im, (int(round(im.shape[1] / s)), int(round(im.shape[0] / s))), interpolation=cv2.INTER_AREA)
        out.append(im)
    if len({im.shape for im in out}) != 1:
        raise FeatureError("frames of one clip differ in size")
    return out


def _noise_from_trace(u: np.ndarray) -> np.ndarray:
    """Per-region, per-component noise (px) of a displacement trace ``(n, R, 2)`` from its second differences, which a smooth
    movement does not feed; the factor sqrt(2) accounts for the reference frame's share, which differencing cancels."""
    if len(u) < 6:
        return np.full((R, 2), 0.05)
    d2 = u[2:] - 2 * u[1:-1] + u[:-2]
    med = np.median(d2, axis=0, keepdims=True)
    mad = np.median(np.abs(d2 - med), axis=0) * 1.4826
    return np.maximum(mad / np.sqrt(6.0) * np.sqrt(2.0), 0.01).astype(np.float64)


def extract(rec: ClipRecord, cfg: FeatureConfig) -> ClipFeatures:
    t0 = time.time()
    paths, a_idx, numbering, problem = resolve_frames(rec.folder, rec.onset, rec.apex, rec.offset)
    if problem:
        raise FeatureError(problem)
    if len(paths) < 6:
        raise FeatureError(f"only {len(paths)} frames between onset and offset")
    frames = _load_frames(paths, cfg)
    n_src = len(paths)
    if len(frames) != n_src and a_idx is not None:
        a_idx = int(round(a_idx * (len(frames) - 1) / max(n_src - 1, 1)))
    fps = (rec.fps or cfg.fps_default) * (len(frames) / max(n_src, 1))
    times = np.arange(len(frames)) / fps
    src = ArraySource(np.stack(frames), times)
    try:
        track = track_landmarks(src)
    except ValueError as e:
        raise FeatureError("no face found") from e
    reg = register(src, track)
    meas = Measurer(reg, threads=cfg.threads)
    cent = TPL.roi_centroids(reg.rois)
    u_px, w, _ = meas.trace(0, list(range(len(frames))))                      # displacement of every frame relative to the onset frame
    iod = float(reg.canon.iod)
    u = u_px.astype(np.float64) / iod * 100.0
    energy = np.linalg.norm(u, axis=2).mean(1)
    if a_idx is None:
        a_idx = int(np.argmax(energy))
    sigma_px = _noise_from_trace(u_px.astype(np.float64))
    # series resampled to a fixed number of steps over the clip
    pos = np.linspace(0, len(frames) - 1, cfg.steps)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(frames) - 1)
    fr = (pos - lo)[:, None, None]
    series = ((1 - fr) * u[lo] + fr * u[hi]).astype(np.float32)
    # AU engine on the apex pattern
    cal = cfg.calibration or {}
    eng = AUEngine(cent, meas.centre, iod, eta=cal.get("eta", 0.15), temper=cal.get("temper", 1.0),
                   **({"size_prior": tuple(cal["size_prior"])} if cal.get("size_prior") else {}))
    post = eng.infer(u_px[a_idx].astype(np.float64), sigma_px, None, 0.05, 0.05, n_samples=200)
    top = sorted(post.set_probs.items(), key=lambda kv: -kv[1])[:16]
    set_aus = np.zeros((len(top), K), bool)
    for i, (s, _) in enumerate(top):
        for a in s:
            set_aus[i, TPL.AUS.index(a)] = True
    dur = (rec.offset - rec.onset) / float(rec.fps or cfg.fps_default)
    return ClipFeatures(
        uid=rec.uid, series=series, apex=u[a_idx].astype(np.float32), apex_frac=float(a_idx / max(len(frames) - 1, 1)),
        sigma_pct=(sigma_px / iod * 100.0).astype(np.float32), pip=post.pip.astype(np.float32),
        amp=(post.amp_given / iod * 100.0).astype(np.float32), p_null=float(post.p_null), set_aus=set_aus,
        set_p=np.array([p for _, p in top], np.float32),
        meta=dict(fps=float(fps), iod_px=float(track.iod_px), n_frames=int(n_src), duration_s=float(dur), numbering=numbering,
                  seconds=time.time() - t0, face_fraction=float(track.valid.mean())))


# --------------------------------------------------------------------------------------------------------------
def _cache_path(cache: str, rec: ClipRecord, cfg: FeatureConfig) -> str:
    h = hashlib.sha1(f"{rec.uid}|{rec.onset}|{rec.offset}|{rec.apex}|{cfg.key()}".encode()).hexdigest()[:20]
    return os.path.join(cache, f"{rec.dataset}_{h}.npz")


def save(path: str, f: ClipFeatures):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, uid=f.uid, series=f.series, apex=f.apex, apex_frac=f.apex_frac, sigma_pct=f.sigma_pct, pip=f.pip, amp=f.amp,
                        p_null=f.p_null, set_aus=f.set_aus, set_p=f.set_p, meta=json.dumps(f.meta))


def load_cached(path: str) -> ClipFeatures:
    z = np.load(path, allow_pickle=False)
    return ClipFeatures(uid=str(z["uid"]), series=z["series"], apex=z["apex"], apex_frac=float(z["apex_frac"]), sigma_pct=z["sigma_pct"],
                        pip=z["pip"], amp=z["amp"], p_null=float(z["p_null"]), set_aus=z["set_aus"], set_p=z["set_p"],
                        meta=json.loads(str(z["meta"])))


def _work(args):
    rec, cfg, path = args
    try:
        f = extract(rec, cfg)
        if path:
            save(path, f)
        return rec.uid, f, None
    except FeatureError as e:
        return rec.uid, None, str(e)
    except Exception as e:                       # a bad clip must not stop a run of hundreds
        return rec.uid, None, f"{type(e).__name__}: {e}"


def extract_many(records: Sequence[ClipRecord], cfg: FeatureConfig, cache: Optional[str] = None, workers: int = 1,
                 progress: Optional[Callable[[int, int], None]] = None) -> tuple[dict, list]:
    """Features for every record, from the cache when present.  Returns ``(features by uid, failures)`` where each
    failure is ``dict(uid, reason)``: clips that could not be measured are reported, never silently dropped."""
    feats: dict = {}
    failures: list = []
    todo = []
    for r in records:
        p = _cache_path(cache, r, cfg) if cache else None
        if p and os.path.exists(p):
            try:
                feats[r.uid] = load_cached(p)
                continue
            except Exception:
                pass
        todo.append((r, cfg, p))
    done = len(records) - len(todo)
    if progress:
        progress(done, len(records))
    if todo:
        if workers > 1:
            with mp.get_context("spawn").Pool(workers) as pool:
                it = pool.imap_unordered(_work, todo)
                for uid, f, err in it:
                    done += 1
                    (feats.__setitem__(uid, f) if f is not None else failures.append(dict(uid=uid, reason=err)))
                    if progress:
                        progress(done, len(records))
        else:
            for t in todo:
                uid, f, err = _work(t)
                done += 1
                (feats.__setitem__(uid, f) if f is not None else failures.append(dict(uid=uid, reason=err)))
                if progress:
                    progress(done, len(records))
    return feats, failures


# --------------------------------------------------------------------------------------------------------------
@dataclass
class ClipSet:
    """Stacked arrays for a list of clips, in the order given."""
    uids: list
    series: np.ndarray               # (N, T, R, 2)
    apex: np.ndarray                 # (N, R, 2)
    hand: np.ndarray                 # (N, F)
    pip: np.ndarray                  # (N, K)
    set_aus: list                    # per clip (S, K) bool
    set_p: list                      # per clip (S,)
    p_null: np.ndarray               # (N,)
    seconds: np.ndarray              # (N,) feature-extraction time per clip

    def __len__(self) -> int:
        return len(self.uids)

    def take(self, idx) -> "ClipSet":
        idx = np.asarray(idx, int)
        return ClipSet([self.uids[i] for i in idx], self.series[idx], self.apex[idx], self.hand[idx], self.pip[idx],
                       [self.set_aus[i] for i in idx], [self.set_p[i] for i in idx], self.p_null[idx], self.seconds[idx])


def stack(uids: Sequence[str], feats: dict) -> ClipSet:
    fs = [feats[u] for u in uids]
    return ClipSet(list(uids), np.stack([f.series for f in fs]), np.stack([f.apex for f in fs]), np.stack([f.hand() for f in fs]),
                   np.stack([f.pip for f in fs]), [f.set_aus for f in fs], [f.set_p for f in fs],
                   np.array([f.p_null for f in fs], np.float32), np.array([f.meta.get("seconds", 0.0) for f in fs], np.float32))
