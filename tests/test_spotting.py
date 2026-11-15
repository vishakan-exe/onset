import numpy as np
import pytest

from onset.analysis import templates as TPL
from onset.vision import geometry as G
from onset.vision import spotting as SP
from helpers import fake_geometry

R = len(G.ROI_NAMES)
K = len(TPL.SCAN_KINDS)


def test_scale_specs_follow_the_frame_rate():
    for fps in (15.0, 30.0, 100.0, 200.0):
        specs = SP.scale_specs(fps, 400)
        assert specs, fps
        ks = [s.k for s in specs]
        assert ks == sorted(set(ks))                                                   # distinct, increasing lags
        for s in specs:
            assert s.k % s.stride == 0 and 1 <= s.stride <= s.k                          # both pair flows of a transient lie on the stride grid
    s30 = SP.scale_specs(30.0, 400)
    assert [s.k for s in s30] == [2, 4, 8, 18]                                           # 0.07, 0.13, 0.25, 0.60 s
    assert len(SP.scale_specs(30.0, 7)) < len(s30)                                       # a very short clip drops the long scales


def test_pair_budget_raises_the_stride_for_long_clips():
    short = SP.scale_specs(200.0, 400, pair_budget=10_000)
    long = SP.scale_specs(200.0, 4000, pair_budget=1400)
    assert sum(max(0, (4000 - s.k) // s.stride) for s in long) <= 1400 + max(s.k for s in long)
    assert max(s.stride for s in long) >= max(s.stride for s in short)


def test_robust_sigma_ignores_outliers(rng):
    x = rng.normal(0, 0.5, 5000)
    x[:50] = 100.0
    assert SP.robust_sigma(x) == pytest.approx(0.5, rel=0.1)
    assert SP.robust_sigma(rng.normal(0, 1, (4000, 3))).shape == (3,)


def test_whitened_templates_are_free_of_nuisance_motion():
    c, centre, _ = fake_geometry()
    sig = np.full((R, 2), 0.1)
    Ap, Q, w = SP.whiten_templates(sig, c, centre)
    assert Ap.shape == (K, 2 * R) and Q.shape == (2 * R, 7) and w.shape == (2 * R,)
    assert np.abs(Ap @ Q).max() < 1e-9                                                   # orthogonal to every head-motion direction
    mask = np.ones(R)
    mask[G.ROI_NAMES.index("chin")] = 0.0
    _, _, w2 = SP.whiten_templates(sig, c, centre, mask)
    assert w2[2 * G.ROI_NAMES.index("chin")] == 0 and w2[0] > 0


def test_noise_after_projection_recovers_region_noise_despite_common_motion(rng):
    c, centre, _ = fake_geometry()
    m = 400
    true_sigma = 0.08
    N = TPL.nuisance_columns(c, centre).reshape(7, -1)
    common = (rng.normal(0, 1.0, (m, 7)) * np.array([0.5, 0.5, 1, 1, 0.3, 0.3, 0.2])) @ N          # stabilisation jitter, large and shared
    X = (common + rng.normal(0, true_sigma, (m, 2 * R))).reshape(m, R, 2)
    sig, raw = SP.noise_after_projection(X, c, centre)
    assert raw > 2 * true_sigma                                                                      # the raw spread is dominated by the shared motion
    assert np.median(sig) == pytest.approx(true_sigma, rel=0.25)


def _transients(rng, m, pattern_at=None, amp=0.0, sigma=0.1):
    T = rng.normal(0, sigma, (m, R, 2))
    if pattern_at is not None:
        T[pattern_at] += amp * pattern_templates()[0]
    return T


def pattern_templates(which=("AU12",)):
    return np.stack([TPL.scan_templates()[[TPL.scan_label(i) for i in range(K)].index(w)] for w in which])


def test_greedy_rss_explains_up_to_three_templates_and_leaves_noise_alone(rng):
    c, centre, _ = fake_geometry()
    sig = np.full((R, 2), 0.1)
    Ap, Q, w = SP.whiten_templates(sig, c, centre)
    G_ = Ap @ Ap.T
    nrm2 = (Ap ** 2).sum(1)
    labels = [TPL.scan_label(i) for i in range(K)]
    T = np.zeros((4, R, 2))
    T[0] = 3.0 * TPL.scan_templates()[labels.index("AU12")]
    T[1] = 3.0 * TPL.scan_templates()[labels.index("AU12")] + 2.5 * TPL.scan_templates()[labels.index("AU4")]
    T[2] = rng.normal(0, 3.0 * 0.1, (R, 2)) * 5                                                      # a pattern that is no combination of AUs
    T[3] = rng.normal(0, 0.1, (R, 2))
    Z = T.reshape(4, -1) * w
    Zp = Z - (Z @ Q) @ Q.T
    dots = Zp @ Ap.T
    tot = (Zp ** 2).sum(1)
    rss, used = SP.greedy_rss(tot, dots, G_, nrm2)
    assert used[0] == 1 and rss[0] < 0.15 * tot[0]                                                   # one template explains it
    assert used[1] == 2 and rss[1] < 0.15 * tot[1]                                                   # two templates explain it together
    assert rss[2] > 0.5 * tot[2]                                                                     # a non-AU pattern keeps most of its energy
    assert (rss <= tot + 1e-9).all()


def test_masked_scan_finds_a_movement_the_blink_would_have_hidden(rng):
    c, centre, _ = fake_geometry()
    sig = np.full((R, 2), 0.1)
    Ap, Q, w = SP.whiten_templates(sig, c, centre)
    nrm2 = (Ap ** 2).sum(1)
    labels = [TPL.scan_label(i) for i in range(K)]
    m = 6
    T = rng.normal(0, 0.1, (m, R, 2))
    T[3] += 2.0 * TPL.scan_templates()[labels.index("AU17")]                                         # chin movement ...
    T[3] += 3.0 * TPL.scan_templates()[labels.index("blink")]                                        # ... during a blink
    near = np.zeros(m, bool)
    near[2:5] = True
    ms = SP.masked_scan("blink", SP.EYE_ROIS, near, T, sig, c, centre, nrm2)
    assert ms.zbest[3] > 6 and ms.zbest[0] == 0                                                      # evaluated only where a blink window is near
    assert np.isinf(ms.zeta[0]).all() and np.isfinite(ms.zeta[3, labels.index("AU17")])
    assert np.isneginf(ms.zeta[3, labels.index("AU5")])                                              # a template that lives in the left-out lids is not scored
    empty = SP.masked_scan("blink", SP.EYE_ROIS, np.zeros(m, bool), T, sig, c, centre, nrm2)
    assert empty.zbest.max() == 0


def test_near_windows():
    t = np.linspace(0, 5, 51)
    near = SP._near_windows(t, 0.1, [(2.0, 2.2)], pad=0.1)
    assert near[t == 2.0].all() and not near[t == 4.0].any() and not near[t == 0.0].any()
    assert not SP._near_windows(t, 0.1, None).any()
