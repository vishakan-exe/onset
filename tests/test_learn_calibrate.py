import numpy as np
import pytest

from onset.learn import calibrate as C


def _logits(rng, n=400, k=3, scale=3.0):
    z = rng.normal(0, 1.5, (n, k))
    p = C.softmax(z)
    y = np.array([rng.choice(k, p=pi) for pi in p])
    return z * scale, y


def test_softmax_rows_sum_to_one(rng):
    p = C.softmax(rng.normal(0, 50, (10, 4)))
    assert np.allclose(p.sum(1), 1) and np.isfinite(p).all()


def test_calibrator_fixes_overconfidence_and_gives_valid_sets(rng):
    z, y = _logits(rng, 600)
    cal = C.Calibrator().fit(z[:300], y[:300], alpha=0.1)
    assert cal.fitted and cal.temperature > 1.5                                   # logits were scaled 3x
    pr = cal.probs(z[300:])
    assert C.reliability(pr, y[300:])["ece"] < C.reliability(C.softmax(z[300:]), y[300:])["ece"]
    st = C.set_stats(cal.sets(pr), y[300:], 3)
    assert st["coverage"] >= 0.85 and 1.0 <= st["mean_size"] <= 3.0


def test_unfitted_calibrator_is_honest_about_not_knowing(rng):
    z, y = _logits(rng, 10)
    cal = C.Calibrator().fit(z, y)                                                # fewer than min_items: stays unfitted
    assert not cal.fitted
    assert all(len(s) == 3 for s in cal.sets(cal.probs(z)))                         # the whole label set


def test_set_stats_edge_cases():
    st = C.set_stats([[0], [0, 1], [1]], np.array([0, 1, 0]), 2)
    assert st["coverage"] == pytest.approx(2 / 3) and st["mean_size"] == pytest.approx(4 / 3)
    assert st["singleton_rate"] == pytest.approx(2 / 3) and st["accuracy_when_singleton"] == pytest.approx(0.5)
    assert C.set_stats([], np.array([], int), 2)["coverage"] is None
