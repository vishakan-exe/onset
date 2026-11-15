"""Models for micro-expression category recognition, all behind one small interface.

    model.fit(train_clips, y, groups, n_classes, seed)   ->  model        (groups = subject of every training clip)
    model.logits(clips)                                   ->  (n, C)       uncalibrated scores
    model.cal_logits, model.cal_y                                      predictions for training subjects the model did not
                                                                       learn from (inner subject-disjoint folds / validation
                                                                       subjects): the material for temperature and conformal sets

Baselines first, because a learned model has to beat them to mean anything:

``majority``   always the commonest training class;
``prior``      a class drawn at random with the training frequencies;
``expert``     ONSET's own AU -> category map applied to the engine's AU posterior; **not trained** - it measures how much
               of the task the hand-written FACS prototypes already solve;
``linear``     class-balanced multinomial logistic regression on the hand-crafted summary (apex pattern, regional magnitudes,
               timing, AU evidence); the regularisation is chosen by subject-disjoint inner cross-validation;
``graphnet``   a small spatio-temporal transformer over the 27 regions and the time steps of the clip, trained with class-
               balanced loss, mirror / amplitude / timing augmentation, subject-disjoint early stopping and a seed ensemble.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..analysis import categories as CAT
from ..vision import geometry as G
from . import labels as L
from . import protocols as P
from .features import HAND_BLOCKS, R, ClipSet, mirror

NEG_LOGIT = -20.0


class Model:
    name = "model"

    def __init__(self):
        self.cal_logits: Optional[np.ndarray] = None
        self.cal_y: Optional[np.ndarray] = None
        self.fit_seconds = 0.0

    def fit(self, X: ClipSet, y: np.ndarray, groups: np.ndarray, n_classes: int, seed: int = 0) -> "Model":
        raise NotImplementedError

    def logits(self, X: ClipSet) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: ClipSet, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return self.logits(X).argmax(1)


# --------------------------------------------------------------------------------------------------------------
class MajorityModel(Model):
    name = "majority"

    def fit(self, X, y, groups, n_classes, seed=0):
        self.prior = (np.bincount(y, minlength=n_classes) + 1e-3) / (len(y) + 1e-3 * n_classes)
        return self

    def logits(self, X):
        return np.tile(np.log(self.prior), (len(X), 1))


class PriorGuess(MajorityModel):
    name = "prior"

    def fit(self, X, y, groups, n_classes, seed=0):
        super().fit(X, y, groups, n_classes, seed)
        self.seed = seed
        return self

    def predict(self, X, rng=None):
        rng = rng or np.random.default_rng(self.seed + len(X))
        return rng.choice(len(self.prior), size=len(X), p=self.prior)


class ExpertMapModel(Model):
    """The shipped AU -> category hypotheses (``categories.py``), untrained.  Forced to choose among the scheme's classes; the
    mass the map puts on 'unclear' is returned by ``unclear`` so abstention can be reported separately."""
    name = "expert"

    def __init__(self, scheme: str):
        super().__init__()
        self.scheme_name = scheme
        self.scheme = CAT.SCHEMES[scheme]()
        if list(self.scheme.labels) != list(L.SCHEMES[scheme]):
            raise ValueError("the expert scheme's labels differ from the dataset scheme's")

    def fit(self, X, y, groups, n_classes, seed=0):
        # Nothing is fitted, so the map's predictions for the training clips are as good as held-out predictions: they are the
        # material for its temperature and conformal sets (the map itself never sees a label).
        self.cal_logits, self.cal_y = self.logits(X), np.asarray(y, int)
        return self

    def _probs(self, X: ClipSet):
        out, unc = [], []
        for aus, p, p0 in zip(X.set_aus, X.set_p, X.p_null):
            sp = {}
            for row, pp in zip(aus, p):
                key = frozenset(CAT.TPL.AUS[k] for k in np.flatnonzero(row))
                sp[key] = sp.get(key, 0.0) + float(pp)
            res = CAT.category_posterior(sp, 0.0, self.scheme, float(p0))
            out.append(res.probs)
            unc.append(res.unclear)
        return np.array(out), np.array(unc)

    def logits(self, X):
        return np.log(np.clip(self._probs(X)[0], 1e-6, 1.0))

    def unclear(self, X) -> np.ndarray:
        return self._probs(X)[1]


# --------------------------------------------------------------------------------------------------------------
class Standardizer:
    def fit(self, a: np.ndarray) -> "Standardizer":
        self.mu = a.mean(0)
        self.sd = a.std(0) + 1e-6
        return self

    def __call__(self, a: np.ndarray) -> np.ndarray:
        return (a - self.mu) / self.sd


def _full_logits(clf, X: np.ndarray, n_classes: int) -> np.ndarray:
    """log-probabilities over *all* classes, with a floor for classes the classifier never saw."""
    lp = np.full((len(X), n_classes), NEG_LOGIT)
    lp[:, clf.classes_] = clf.predict_log_proba(X)
    return lp


class LinearModel(Model):
    name = "linear"

    def __init__(self, blocks=("pattern", "timing", "au"), Cs=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0), inner_k: int = 4, name: Optional[str] = None):
        super().__init__()
        self.blocks, self.Cs, self.inner_k = tuple(blocks), tuple(Cs), inner_k
        if name:
            self.name = name

    def _x(self, X: ClipSet) -> np.ndarray:
        return np.concatenate([X.hand[:, HAND_BLOCKS[b]] for b in self.blocks], axis=1)

    def fit(self, X, y, groups, n_classes, seed=0):
        from sklearn.linear_model import LogisticRegression
        t0 = time.time()
        a = self._x(X)
        self.n_classes = n_classes
        self.std = Standardizer().fit(a)
        z = self.std(a)

        def clf(C):
            return LogisticRegression(C=C, class_weight="balanced", max_iter=400, random_state=seed)

        folds = P.inner_folds(np.arange(len(y)), groups, y, k=self.inner_k, seed=seed) if len(set(groups.tolist())) >= 3 else []
        best_C, best_ll, best_oof = self.Cs[len(self.Cs) // 2], np.inf, None
        if folds:
            for C in self.Cs:
                oof = np.zeros((len(y), n_classes))
                for f in folds:
                    m = clf(C).fit(z[f.train], y[f.train])
                    oof[f.test] = _full_logits(m, z[f.test], n_classes)
                pr = np.exp(oof - oof.max(1, keepdims=True))
                pr /= pr.sum(1, keepdims=True)
                ll = float(-np.mean(np.log(np.clip(pr[np.arange(len(y)), y], 1e-9, 1))))
                if ll < best_ll:
                    best_C, best_ll, best_oof = C, ll, oof
        self.C = best_C
        self.clf = clf(best_C).fit(z, y)
        self.cal_logits, self.cal_y = best_oof, (y if best_oof is not None else None)
        self.fit_seconds = time.time() - t0
        return self

    def logits(self, X):
        return _full_logits(self.clf, self.std(self._x(X)), self.n_classes)


# --------------------------------------------------------------------------------------------------------------
def _adjacency() -> np.ndarray:
    """Soft neighbourhood of the 27 regions: same anatomical group, and left/right counterparts."""
    A = np.eye(R)
    grp = {n: g for g, names in G.ROI_GROUPS.items() for n in names}
    for i, a in enumerate(G.ROI_NAMES):
        for j, b in enumerate(G.ROI_NAMES):
            if i != j and grp[a] == grp[b]:
                A[i, j] = 0.5
            if i != j and a[:-2] == b[:-2] and a[-2:] in ("_L", "_R") and b[-2:] in ("_L", "_R"):
                A[i, j] = 1.0
    return A


class GraphNetConfig:
    """``epochs`` is raised, when the training set is small, until the schedule has at least ``min_steps`` optimiser steps: a
    few dozen steps leave the network at chance whatever the epoch count says. ``patience`` > 0 switches on early stopping on
    the validation loss; it is **off** by default because a validation set of a few clips has a noisy loss whose minimum is
    often at the very start of training, and "restore the best weights" then returns a network that has learnt nothing."""

    def __init__(self, d: int = 32, heads: int = 4, layers: int = 2, dropout: float = 0.2, epochs: int = 40, batch: int = 16,
                 lr: float = 3e-3, weight_decay: float = 1e-2, patience: int = 0, members: int = 3, val_fraction: float = 0.2,
                 label_smoothing: float = 0.05, threads: int = 2, readout: str = "region", min_steps: int = 180):
        self.__dict__.update(locals())
        del self.__dict__["self"]


class GraphTemporalNet(Model):
    name = "graphnet"

    def __init__(self, cfg: Optional[GraphNetConfig] = None):
        super().__init__()
        self.cfg = cfg or GraphNetConfig()

    # ----- network -------------------------------------------------------------------------------------------
    @staticmethod
    def _build(cfg: GraphNetConfig, T: int, n_classes: int):
        import torch
        from torch import nn

        class Block(nn.Module):
            def __init__(self, d, heads, drop):
                super().__init__()
                self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
                self.sp = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
                self.tp = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
                self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
                self.drop = nn.Dropout(drop)

            def forward(self, x, bias):
                B, T_, R_, d = x.shape
                h = self.n1(x).reshape(B * T_, R_, d)
                a, _ = self.sp(h, h, h, attn_mask=bias, need_weights=False)          # between regions, within a time step
                x = x + self.drop(a.reshape(B, T_, R_, d))
                h = self.n2(x).permute(0, 2, 1, 3).reshape(B * R_, T_, d)
                a, _ = self.tp(h, h, h, need_weights=False)                          # between time steps, within a region
                x = x + self.drop(a.reshape(B, R_, T_, d).permute(0, 2, 1, 3))
                return x + self.drop(self.ff(self.n3(x)))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                d = cfg.d
                self.inp = nn.Linear(2, d)
                self.roi = nn.Parameter(torch.zeros(R, d))
                self.time = nn.Parameter(torch.zeros(T, d))
                nn.init.normal_(self.roi, std=0.02)
                nn.init.normal_(self.time, std=0.02)
                self.register_buffer("A", torch.tensor(_adjacency(), dtype=torch.float32))
                self.beta = nn.Parameter(torch.tensor(1.0))
                self.blocks = nn.ModuleList([Block(d, cfg.heads, cfg.dropout) for _ in range(cfg.layers)])
                self.norm = nn.LayerNorm(d)
                if cfg.readout == "region":
                    # which region carried a feature matters as much as what the feature was: each region's token is pooled over
                    # time (mean and max, so a brief transient is not averaged away) and squeezed to a few numbers, and the 27
                    # summaries are concatenated before the classifier, which therefore has one weight per region and feature
                    self.proj = nn.Linear(2 * d, 8)
                    self.head = nn.Linear(R * 8, n_classes)
                    self.head_drop = nn.Dropout(cfg.dropout)
                else:                                                               # "mean": a plain average over every token
                    self.head = nn.Linear(d, n_classes)

            def forward(self, s):                                                   # s: (B, T, R, 2)
                x = self.inp(s) + self.roi[None, None] + self.time[None, :, None]
                bias = self.beta * self.A
                for b in self.blocks:
                    x = b(x, bias)
                x = self.norm(x)
                if cfg.readout == "region":
                    z = torch.nn.functional.gelu(self.proj(torch.cat([x.mean(1), x.amax(1)], dim=-1)))      # (B, R, 8)
                    return self.head(self.head_drop(z.flatten(1)))
                return self.head(x.mean((1, 2)))

        return Net()

    # ----- training ------------------------------------------------------------------------------------------
    def _augment(self, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        s = s.copy()
        flip = rng.random(len(s)) < 0.5
        if flip.any():
            s[flip] = mirror(s[flip])
        s *= rng.uniform(0.7, 1.3, (len(s), 1, 1, 1)).astype(np.float32)
        shift = rng.integers(-2, 3, len(s))
        for i, k in enumerate(shift):
            if k:
                s[i] = np.roll(s[i], k, axis=0)
                if k > 0:
                    s[i, :k] = s[i, k:k + 1]
                else:
                    s[i, k:] = s[i, k - 1:k]
        drop = rng.random((len(s), 1, R, 1)) < 0.1
        s = np.where(drop, 0.0, s)
        return (s + rng.normal(0, 0.03, s.shape)).astype(np.float32)

    def _train_one(self, S: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, n_classes: int, seed: int):
        import torch
        from torch import nn
        cfg = self.cfg
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = self._build(cfg, S.shape[1], n_classes)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        counts = np.bincount(y[tr], minlength=n_classes).astype(float)
        w = np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
        w = w / w[counts > 0].mean()
        lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32), label_smoothing=cfg.label_smoothing)
        per_epoch = max(1, int(np.ceil(len(tr) / cfg.batch)))
        epochs = max(cfg.epochs, int(np.ceil(cfg.min_steps / per_epoch)))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=max(epochs * per_epoch, 2))
        best, best_state, bad = np.inf, None, 0
        Sv = torch.tensor(S[va]) if (len(va) and cfg.patience > 0) else None
        for ep in range(epochs):
            net.train()
            order = rng.permutation(tr)
            for i in range(0, len(order), cfg.batch):
                ii = order[i:i + cfg.batch]
                if len(ii) < 2:
                    continue
                xb = torch.tensor(self._augment(S[ii], rng))
                loss = lossf(net(xb), torch.tensor(y[ii]))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
                sched.step()
            if Sv is not None:
                net.eval()
                with torch.no_grad():
                    vl = float(nn.functional.cross_entropy(net(Sv), torch.tensor(y[va]), weight=torch.tensor(w, dtype=torch.float32)))
                if vl < best - 1e-4:
                    best, bad = vl, 0
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def fit(self, X, y, groups, n_classes, seed=0):
        import torch
        t0 = time.time()
        cfg = self.cfg
        torch.set_num_threads(cfg.threads)
        self.n_classes = n_classes
        S = X.series.astype(np.float32)
        self.scale = float(np.sqrt((S ** 2).mean()) + 1e-6)
        S = S / self.scale
        self.nets = []
        self.cal_logits, self.cal_y = None, None
        cal_l, cal_y = [], []
        for m in range(cfg.members):
            # subject-disjoint validation split (different per member): early stopping and calibration material
            folds = P.inner_folds(np.arange(len(y)), groups, y, k=max(2, int(round(1.0 / cfg.val_fraction))), seed=seed * 101 + m) \
                if len(set(groups.tolist())) >= 3 else []
            if folds:
                f = folds[0]
                tr, va = f.train, f.test
            else:
                tr, va = np.arange(len(y)), np.array([], int)
            net = self._train_one(S, y, tr, va, n_classes, seed * 101 + m)
            self.nets.append(net)
            if len(va):
                cal_l.append((va, self._net_logits(net, S[va])))
        if cal_l:
            # ensemble logits for each validation clip, averaged over the members that held it out
            acc = {}
            for va, lg in cal_l:
                for i, l in zip(va, lg):
                    acc.setdefault(int(i), []).append(l)
            idx = sorted(acc)
            self.cal_logits = np.array([np.mean(acc[i], axis=0) for i in idx])
            self.cal_y = y[idx]
        self.fit_seconds = time.time() - t0
        return self

    @staticmethod
    def _net_logits(net, S: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            return net(torch.tensor(S.astype(np.float32))).numpy()

    def logits(self, X):
        S = X.series.astype(np.float32) / self.scale
        return np.mean([self._net_logits(n, S) for n in self.nets], axis=0)

    def member_disagreement(self, X) -> np.ndarray:
        """Epistemic signal: variance of the members' class probabilities, summed over classes."""
        S = X.series.astype(np.float32) / self.scale
        ps = []
        for n in self.nets:
            z = self._net_logits(n, S)
            z = z - z.max(1, keepdims=True)
            e = np.exp(z)
            ps.append(e / e.sum(1, keepdims=True))
        return np.var(np.stack(ps), axis=0).sum(1)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.nets[0].parameters())) if getattr(self, "nets", None) else 0


# --------------------------------------------------------------------------------------------------------------
MODEL_NAMES = ("majority", "prior", "expert", "linear", "linear_pattern", "linear_au", "graphnet")


def make_model(name: str, scheme: str, graph_cfg: Optional[GraphNetConfig] = None) -> Model:
    if name == "majority":
        return MajorityModel()
    if name == "prior":
        return PriorGuess()
    if name == "expert":
        return ExpertMapModel(scheme)
    if name == "linear":
        return LinearModel()
    if name == "linear_pattern":
        return LinearModel(blocks=("pattern",), name="linear_pattern")
    if name == "linear_au":
        return LinearModel(blocks=("au",), name="linear_au")
    if name == "graphnet":
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError("the 'graphnet' model needs PyTorch (pip install torch); the other models do not") from e
        return GraphTemporalNet(graph_cfg)
    raise KeyError(f"unknown model '{name}' (choose from {', '.join(MODEL_NAMES)})")
