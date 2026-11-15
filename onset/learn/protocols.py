"""Evaluation protocols (who is trained on, who is tested) and the audit that refuses leaky ones.

All folds are built from **groups** (a subject, qualified by its dataset), never from clips: a micro-expression
classifier that has seen another clip of the same face learns the face.  ``audit`` checks every fold before a model is
fitted and raises ``LeakageError`` rather than reporting an inflated number.

Protocols
  ``loso``       leave one subject out; the standard single-dataset protocol.
  ``composite``  leave one subject out over several datasets pooled (MEGC composite-database style); groups are
                 dataset-qualified so two different people numbered '01' are never merged.
  ``lodo``       leave one dataset out: train on the others, test on the whole held-out dataset (cross-corpus
                 generalisation; the most honest number and usually the lowest).
  ``kfold``      subject-disjoint k folds (quicker, for development; report LOSO for anything you publish).
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


class LeakageError(RuntimeError):
    """A fold lets information about a test subject reach training."""


@dataclass
class Fold:
    name: str
    train: np.ndarray
    test: np.ndarray
    meta: dict = field(default_factory=dict)


def loso(groups: Sequence[str]) -> list[Fold]:
    g = np.asarray(groups)
    return [Fold(f"loso:{s}", np.flatnonzero(g != s), np.flatnonzero(g == s), dict(held_out=str(s))) for s in sorted(set(g.tolist()))]


def composite(groups: Sequence[str]) -> list[Fold]:
    """LOSO over dataset-qualified groups (``ClipRecord.group``)."""
    folds = loso(groups)
    for f in folds:
        f.name = f.name.replace("loso:", "composite:")
    return folds


def lodo(datasets: Sequence[str]) -> list[Fold]:
    d = np.asarray(datasets)
    names = sorted(set(d.tolist()))
    if len(names) < 2:
        raise ValueError("leave-one-dataset-out needs at least two datasets")
    return [Fold(f"lodo:{n}", np.flatnonzero(d != n), np.flatnonzero(d == n), dict(held_out=str(n))) for n in names]


def _assign(groups: np.ndarray, y: Optional[np.ndarray], k: int, seed: int) -> list[list[str]]:
    """Greedy subject-to-fold assignment keeping fold sizes and class mixes close (deterministic given ``seed``).

    Subjects are placed largest first; each goes to the fold where, after adding it, the spread *between* folds is smallest
    (variance across folds of every class count, plus of the fold size).  Every fold receives a subject when there are at
    least ``k`` subjects."""
    rng = np.random.default_rng(seed)
    subjects = sorted(set(groups.tolist()))
    n_cls = (max(y.tolist()) + 1) if y is not None and len(y) else 1
    hist = {s: (np.bincount(y[groups == s], minlength=n_cls) if y is not None else np.array([(groups == s).sum()])).astype(float)
            for s in subjects}
    order = sorted(subjects, key=lambda s: (-(groups == s).sum(), rng.random()))
    folds: list[list[str]] = [[] for _ in range(k)]
    tot = np.zeros((k, len(next(iter(hist.values())))))
    for n_left, s in zip(range(len(order), 0, -1), order):
        empty = [i for i in range(k) if not folds[i]]
        cand = empty if empty and n_left >= len(empty) else list(range(k))     # open every fold before stacking a second subject
        best, best_cost = cand[0], np.inf
        for i in cand:
            t = tot.copy()
            t[i] += hist[s]
            cost = float(t.var(0).sum() + t.sum(1).var()) + 1e-9 * rng.random()
            if cost < best_cost:
                best, best_cost = i, cost
        folds[best].append(s)
        tot[best] += hist[s]
    return folds


def kfold(groups: Sequence[str], y: Optional[Sequence[int]] = None, k: int = 5, seed: int = 0) -> list[Fold]:
    g = np.asarray(groups)
    k = int(min(k, len(set(g.tolist()))))
    parts = _assign(g, None if y is None else np.asarray(y), k, seed)
    out = []
    for i, subs in enumerate(parts):
        te = np.flatnonzero(np.isin(g, subs))
        out.append(Fold(f"kfold:{i + 1}/{k}", np.flatnonzero(~np.isin(g, subs)), te, dict(held_out=sorted(map(str, subs)))))
    return out


def inner_folds(train_idx: np.ndarray, groups: Sequence[str], y: Sequence[int], k: int = 4, seed: int = 0) -> list[Fold]:
    """Subject-disjoint folds *inside* a training set (hyper-parameters, temperature, conformal threshold, the validation split of an ensemble member).
    Indices refer to the full arrays; the outer test subjects are never in ``train_idx`` so they cannot appear here."""
    g = np.asarray(groups)[train_idx]
    yy = np.asarray(y)[train_idx]
    k = int(min(k, len(set(g.tolist()))))
    if k < 2:
        raise ValueError("the training set has a single subject; there is nothing to calibrate on")
    parts = _assign(g, yy, k, seed)
    out = []
    for i, subs in enumerate(parts):
        mask = np.isin(g, subs)
        out.append(Fold(f"inner:{i + 1}/{k}", train_idx[~mask], train_idx[mask], dict(held_out=sorted(map(str, subs)))))
    return out


# --------------------------------------------------------------------------------------------------------------
def _canon_subject(s: str) -> str:
    return re.sub(r"^(sub|s|subject|p)?[\s_\-]*0*", "", str(s).strip().lower()) or "0"


def _first_frame_hash(folder: str) -> Optional[str]:
    """SHA-1 of the first image file of a clip folder (``None`` when there is none): two subjects whose clips begin with the
    very same bytes are almost certainly one clip filed twice."""
    try:
        names = sorted(n for n in os.listdir(folder) if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")))
        if not names:
            return None
        with open(os.path.join(folder, names[0]), "rb") as fh:
            return hashlib.sha1(fh.read()).hexdigest()
    except OSError:
        return None


def audit(records: Sequence, folds: Sequence[Fold], require_full_coverage: bool = True, strict: bool = True) -> dict:
    """Check a protocol before anything is trained.  Returns ``dict(ok, errors, warnings)``; raises ``LeakageError`` when
    ``strict`` and there are errors.

    Errors (a number computed under them would be inflated or meaningless):
      * a group or a clip is on both sides of a fold;
      * the same frame folder is used by two records;
      * a clip is never tested (or tested twice) in a protocol that should test each once;
      * a fold has an empty test or train side.
    Warnings (worth a look): subject labels that normalise to the same person inside one dataset; byte-identical first frames
    in clips of different subjects (a copied clip)."""
    errors, warnings = [], []
    groups = np.array([r.group for r in records])
    uids = [r.uid for r in records]
    if len(set(uids)) != len(uids):
        dup = sorted({u for u in uids if uids.count(u) > 1})[:5]
        errors.append(f"duplicate clip ids: {dup}")
    folders: dict = {}
    for r in records:
        folders.setdefault(os.path.realpath(r.folder), []).append(r.uid)
    shared = {f: u for f, u in folders.items() if len(u) > 1}
    if shared:
        errors.append(f"{len(shared)} frame folders are used by more than one record, e.g. {next(iter(shared.values()))}")
    canon: dict = {}
    for r in records:
        canon.setdefault((r.dataset, _canon_subject(r.subject)), set()).add(r.subject)
    for key, raw in canon.items():
        if len(raw) > 1:
            warnings.append(f"subject labels {sorted(raw)} in {key[0]} look like one person; group them before splitting")
    by_hash: dict = {}
    for r in records:
        h = _first_frame_hash(r.folder)
        if h:
            by_hash.setdefault(h, []).append(r)
    for h, rs in by_hash.items():
        subs = sorted({r.group for r in rs})
        if len(subs) > 1:
            warnings.append(f"clips of different subjects {subs[:4]} start with a byte-identical first frame ({[r.clip_id if hasattr(r, 'clip_id') else r.uid for r in rs][:3]}): copied clip?")
            break
    tested = np.zeros(len(records), int)
    for f in folds:
        if len(f.train) == 0 or len(f.test) == 0:
            errors.append(f"{f.name}: empty train or test side")
            continue
        if set(f.train.tolist()) & set(f.test.tolist()):
            errors.append(f"{f.name}: clips on both sides")
        both = set(groups[f.train].tolist()) & set(groups[f.test].tolist())
        if both:
            errors.append(f"{f.name}: groups on both sides: {sorted(both)[:5]}")
        tested[f.test] += 1
    if require_full_coverage and folds:
        if (tested == 0).any():
            errors.append(f"{int((tested == 0).sum())} clips are never in a test fold")
        if (tested > 1).any():
            errors.append(f"{int((tested > 1).sum())} clips are tested more than once")
    out = dict(ok=not errors, errors=errors, warnings=warnings, n_folds=len(folds), n_clips=len(records),
               n_groups=len(set(groups.tolist())))
    if errors and strict:
        raise LeakageError("; ".join(errors))
    return out


def check_train_classes(y: np.ndarray, fold: Fold, n_classes: int) -> list[str]:
    miss = sorted(set(range(n_classes)) - set(np.asarray(y)[fold.train].tolist()))
    return [f"{fold.name}: classes {miss} absent from the training set"] if miss else []
