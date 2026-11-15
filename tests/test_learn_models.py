import numpy as np
import pytest

from onset.learn import features as F
from onset.learn import models as M
from onset.learn import labels as L

from helpers import make_clipset


def test_hand_features_layout_is_consistent():
    X, y, g = make_clipset(4, 3)
    assert X.hand.shape == (12, len(F.HAND_NAMES))
    sl = F.HAND_BLOCKS
    n_pattern = len(range(*sl["pattern"].indices(X.hand.shape[1])))
    assert n_pattern == 2 * F.R + F.R + len(F.GROUPS)
    total = sum(len(range(*s.indices(X.hand.shape[1]))) for s in sl.values())
    assert total == X.hand.shape[1]
    assert np.isfinite(X.hand).all()


def test_mirror_is_an_involution_and_swaps_sides():
    s = np.random.default_rng(0).normal(size=(5, F.R, 2)).astype(np.float32)
    m = F.mirror(s)
    assert np.allclose(F.mirror(m), s)
    names = F.G.ROI_NAMES
    li = next(i for i, n in enumerate(names) if n.endswith("_L"))
    ri = names.index(names[li][:-2] + "_R")
    assert np.allclose(m[:, li, 0], -s[:, ri, 0]) and np.allclose(m[:, li, 1], s[:, ri, 1])
    centre = next(i for i, n in enumerate(names) if not n.endswith(("_L", "_R")))
    assert np.allclose(m[:, centre, 0], -s[:, centre, 0])


def test_baselines():
    X, y, g = make_clipset(8, 5, 3)
    m = M.make_model("majority", "megc3").fit(X, y, g, 3)
    assert (m.predict(X) == np.bincount(y).argmax()).all()
    p = M.make_model("prior", "megc3").fit(X, y, g, 3, seed=1)
    pred = p.predict(X)
    assert set(np.unique(pred)) <= {0, 1, 2} and len(pred) == len(y)
    assert m.cal_logits is None                                                       # nothing honest to calibrate a constant on


def test_linear_learns_signal_and_provides_subject_disjoint_calibration_material():
    X, y, g = make_clipset(12, 6, 3, signal=1.5)
    tr = np.flatnonzero(g != "s00")
    te = np.flatnonzero(g == "s00")
    m = M.make_model("linear", "megc3").fit(X.take(tr), y[tr], g[tr], 3, seed=0)
    assert (m.predict(X.take(te)) == y[te]).mean() > 0.7
    assert m.cal_logits.shape == (len(tr), 3) and (m.cal_y == y[tr]).all()
    assert m.C in m.Cs


def test_linear_is_no_better_than_chance_on_noise():
    X, y, g = make_clipset(12, 6, 3, signal=0.0)
    accs = []
    for held in ("s00", "s01", "s02"):
        tr, te = np.flatnonzero(g != held), np.flatnonzero(g == held)
        m = M.make_model("linear", "megc3").fit(X.take(tr), y[tr], g[tr], 3, seed=0)
        accs.append((m.predict(X.take(te)) == y[te]).mean())
    assert np.mean(accs) < 0.6


def test_linear_survives_a_class_missing_from_training():
    X, y, g = make_clipset(8, 5, 4)
    tr = np.flatnonzero(y != 3)
    m = M.make_model("linear", "megc3").fit(X.take(tr), y[tr], g[tr], 4, seed=0)
    lg = m.logits(X)
    assert lg.shape == (len(y), 4) and (lg[:, 3] <= M.NEG_LOGIT + 1e-6).all()           # the unseen class gets a floor, not a crash


def test_expert_model_scheme_checks():
    X, y, g = make_clipset(4, 3, 3)
    m = M.make_model("expert", "megc3")
    lg = m.logits(X)
    assert lg.shape == (12, 3) and np.isfinite(lg).all()
    assert ((m.unclear(X) >= 0) & (m.unclear(X) <= 1)).all()
    assert list(m.scheme.labels) == list(L.MEGC3)
    for scheme in L.SCHEMES:
        M.make_model("expert", scheme)


def test_unknown_model_name():
    with pytest.raises(KeyError, match="unknown model"):
        M.make_model("resnet", "megc3")


def test_adjacency_is_symmetric_with_unit_diagonal():
    A = M._adjacency()
    assert A.shape == (F.R, F.R) and np.allclose(A, A.T) and np.allclose(np.diag(A), 1.0)
    assert A.max() == 1.0 and A.min() == 0.0


def test_graphnet_defaults_train_the_whole_schedule():
    cfg = M.GraphNetConfig()
    assert cfg.patience == 0                      # restore-the-best-validation-loss weights is opt-in: on a few clips it returns an untrained net
    assert cfg.min_steps >= 100 and cfg.readout == "region"


@pytest.mark.slow
def test_graphnet_learns_a_regional_pattern_on_unseen_subjects():
    """It must actually learn: 3 classes, each shown by a different block of regions, held-out subjects, chance is 0.33."""
    pytest.importorskip("torch")
    X, y, g = make_clipset(10, 6, 3, signal=2.0)
    tr, te = np.flatnonzero(g != "s00"), np.flatnonzero(g == "s00")
    cfg = M.GraphNetConfig(members=1, d=16, layers=1, heads=2, threads=2)
    m = M.make_model("graphnet", "megc3", cfg).fit(X.take(tr), y[tr], g[tr], 3, seed=0)
    assert (m.predict(X.take(te)) == y[te]).mean() > 0.75
    assert (m.predict(X.take(tr)) == y[tr]).mean() > 0.85                              # and it fits what it was shown


@pytest.mark.slow
def test_graphnet_trains_and_ensembles():
    torch = pytest.importorskip("torch")
    X, y, g = make_clipset(10, 6, 3, signal=2.0)
    tr, te = np.flatnonzero(g != "s00"), np.flatnonzero(g == "s00")
    cfg = M.GraphNetConfig(epochs=12, members=2, d=16, layers=1, heads=2, batch=16, patience=6, threads=2)
    m = M.make_model("graphnet", "megc3", cfg).fit(X.take(tr), y[tr], g[tr], 3, seed=0)
    lg = m.logits(X.take(te))
    assert lg.shape == (len(te), 3) and np.isfinite(lg).all()
    assert m.n_parameters() > 1000
    assert m.cal_logits is not None and len(m.cal_logits) == len(m.cal_y) > 0
    assert (m.member_disagreement(X.take(te)) >= 0).all()
    m2 = M.make_model("graphnet", "megc3", cfg).fit(X.take(tr), y[tr], g[tr], 3, seed=0)
    assert np.allclose(m.logits(X.take(te)), m2.logits(X.take(te)), atol=1e-4)      # same seed, same network
