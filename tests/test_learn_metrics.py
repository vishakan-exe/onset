import numpy as np
import pytest

from onset.learn import metrics as M


def test_confusion_and_per_class():
    y = np.array([0, 0, 0, 1, 1, 2])
    p = np.array([0, 0, 1, 1, 1, 0])
    cm = M.confusion(y, p, 3)
    assert cm.tolist() == [[2, 1, 0], [0, 2, 0], [1, 0, 0]]
    pc = M.per_class(y, p, 3)
    assert pc["recall"].tolist() == pytest.approx([2 / 3, 1.0, 0.0])
    assert pc["precision"].tolist() == pytest.approx([2 / 3, 2 / 3, 0.0])


def test_uf1_uar_known_values():
    y = np.array([0] * 8 + [1] * 2)
    p = np.zeros(10, int)                                                         # always the majority class
    assert M.accuracy(y, p) == 0.8
    assert M.uar(y, p, 2) == pytest.approx(0.5)
    assert M.uf1(y, p, 2) == pytest.approx((2 * 8 / (2 * 8 + 2) + 0.0) / 2)       # class 1 never predicted: F1 = 0
    assert M.uf1(y, y, 2) == 1.0


def test_chance_uf1_matches_simulation(rng):
    y = np.array([0] * 50 + [1] * 30 + [2] * 20)
    sim = np.mean([M.uf1(y, rng.integers(0, 3, 100), 3) for _ in range(1500)])
    assert M.chance_uf1_uniform(y, 3) == pytest.approx(sim, abs=0.01)
    assert M.chance_uf1_uniform(y, 3) != pytest.approx(1 / 3, abs=0.005)           # not simply 1/C on imbalanced data


def test_holm_is_monotone_and_conservative():
    p = [0.01, 0.04, 0.03, 0.5]
    adj = M.holm(p)
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert all(a >= b for a, b in zip(adj, p))
    assert M.holm([0.0001]) == [0.0001]


def test_subject_bootstrap_resamples_subjects(rng):
    subj = np.repeat(np.arange(8), 10)
    y = rng.integers(0, 2, 80)
    p = y.copy()
    est, lo, hi = M.subject_bootstrap(lambda a, b: M.accuracy(a, b), y, p, subj, n_boot=200)
    assert est == 1.0 and lo == 1.0 and hi == 1.0
    p2 = np.where(rng.random(80) < 0.3, 1 - y, y)
    est, lo, hi = M.subject_bootstrap(lambda a, b: M.accuracy(a, b), y, p2, subj, n_boot=300)
    assert lo < est < hi and hi - lo > 0.02


def test_paired_permutation_detects_real_difference_only(rng):
    subj = np.repeat(np.arange(14), 8)
    y = rng.integers(0, 3, len(subj))
    good = np.where(rng.random(len(y)) < 0.9, y, rng.integers(0, 3, len(y)))
    bad = rng.integers(0, 3, len(y))
    fn = lambda a, b: M.uf1(a, b, 3)                                                # noqa: E731
    assert M.paired_permutation(fn, y, good, bad, subj, n_perm=400)["p"] < 0.02
    same = M.paired_permutation(fn, y, good, good.copy(), subj, n_perm=200)
    assert same["p"] == pytest.approx(1.0) and same["difference"] == 0
    other = np.where(rng.random(len(y)) < 0.9, y, rng.integers(0, 3, len(y)))
    assert M.paired_permutation(fn, y, good, other, subj, n_perm=400)["p"] > 0.05


def test_risk_coverage_coverage_decreases(rng):
    probs = rng.dirichlet(np.ones(3), 300)
    y = np.array([rng.choice(3, p=p) for p in probs])
    rc = M.risk_coverage(probs, y)
    cov = [r["coverage"] for r in rc]
    assert cov[0] == 1.0 and all(a >= b - 1e-9 for a, b in zip(cov, cov[1:]))
    assert rc[-1]["accuracy"] >= rc[0]["accuracy"] - 0.05                              # answering only when confident does not hurt


def test_summary_has_everything_a_report_needs():
    y = np.array([0, 1, 2, 0, 1, 2])
    s = M.summary(y, y, 3, ["a", "b", "c"])
    assert s["uf1"] == 1.0 and s["n"] == 6 and [c["label"] for c in s["per_class"]] == ["a", "b", "c"]
    assert s["majority_accuracy"] == pytest.approx(1 / 3)
