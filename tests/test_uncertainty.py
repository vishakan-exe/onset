import numpy as np
import pytest

from onset.analysis import uncertainty as U


def test_reliability_perfect_and_overconfident(rng):
    p = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(float)
    assert U.ece_binary(p, y) < 0.04                                     # honest probabilities
    over = np.clip(p * 1.0, 0, 1)
    over = np.where(over > 0.5, 0.99, 0.01)                              # pushed to the extremes
    assert U.ece_binary(over, y) > 0.15
    bins = U.reliability_curve(p, y, 10)
    assert sum(b["n"] for b in bins) == 4000
    assert all(0 <= b["frac"] <= 1 for b in bins)


def test_losses_basic():
    assert U.brier_binary([1, 0], [1, 0]) == 0
    assert U.log_loss_binary([0.5, 0.5], [1, 0]) == pytest.approx(np.log(2))
    assert np.isfinite(U.log_loss_binary([0.0, 1.0], [1, 0]))            # clipped, never infinite


def test_temperature_recovers_known_scale(rng):
    n, C = 3000, 4
    z = rng.normal(0, 2.0, (n, C))
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(C, p=pi) for pi in p])                      # labels drawn from the true softmax(z)
    T = U.fit_temperature(z * 3.0, y)                                    # model logits 3x too confident
    assert 2.4 < T < 3.7


def test_conformal_coverage_guarantee(rng):
    C, n = 5, 6000
    z = rng.normal(0, 1.5, (n, C))
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(C, p=pi) for pi in p])
    cal, test = slice(0, 3000), slice(3000, None)
    for alpha in (0.05, 0.1, 0.2):
        conf = U.LACConformal(alpha=alpha).fit(p[cal], y[cal])
        cov, size = conf.coverage(p[test], y[test])
        assert cov >= 1 - alpha - 0.02
        assert 1 <= size <= C
    assert U.LACConformal(alpha=0.1).fit(p[cal][:0], y[cal][:0]).qhat == 1.0   # nothing to calibrate on: everything stays in the set


def test_predict_set_never_empty():
    conf = U.LACConformal(alpha=0.1)
    conf.qhat = 0.0
    assert conf.predict_set(np.array([0.3, 0.7]), ["a", "b"]) == ["b"]


def test_wilson_bounds():
    p, lo, hi = U.wilson(0, 10)
    assert p == 0 and lo == 0 and 0 < hi < 0.4
    p, lo, hi = U.wilson(10, 10)
    assert hi == 1 and lo > 0.6
    assert np.isnan(U.wilson(0, 0)[0])
    p, lo, hi = U.wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.21


def test_cluster_bootstrap_wider_than_naive_when_clustered(rng):
    subj = np.repeat(np.arange(10), 20)
    shift = rng.normal(0, 1, 10)[subj]
    x = shift + rng.normal(0, 0.1, 200)                                  # almost all variance is between subjects
    est, lo, hi = U.cluster_bootstrap(x, subj, n_boot=500)
    naive = 1.96 * x.std() / np.sqrt(len(x)) * 2
    assert hi - lo > 2 * naive
    assert lo < est < hi


def test_permutation_pvalue(rng):
    a, b = rng.normal(0, 1, 40), rng.normal(0, 1, 40)
    assert U.permutation_pvalue(a, b, n_perm=500) > 0.05
    assert U.permutation_pvalue(a + 3, b, n_perm=500) < 0.01
