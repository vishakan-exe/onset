import numpy as np
import pytest

from onset.analysis import templates as TPL
from onset.analysis.au_engine import AUEngine
from helpers import fake_geometry

SIGMA = 0.10                                     # px per component: a good webcam clip


@pytest.fixture(scope="module")
def eng():
    c, centre, iod = fake_geometry()
    return AUEngine(c, centre, iod, eta=0.15, temper=1.0)


def sigma_arr(v=SIGMA):
    return np.full((27, 2), v)


def pattern(aus_amp, rng=None, noise=SIGMA):
    y = sum(a * TPL.au_templates()[TPL.AUS.index(k)] for k, a in aus_amp.items())
    if rng is not None and noise > 0:
        y = y + rng.normal(0, noise, y.shape)
    return y


@pytest.mark.parametrize("au", TPL.AUS)
def test_each_single_au_is_named_when_clearly_present(eng, au, rng):
    y = pattern({au: 2.0}, rng)
    post = eng.infer(y, sigma_arr(), n_samples=60)
    top = int(np.argmax(post.pip))
    assert post.pip[TPL.AUS.index(au)] > 0.5, (au, dict(zip(TPL.AUS, post.pip.round(2))))
    assert post.p_null < 0.1


def test_noise_alone_names_nothing_and_commits_to_no_hypothesis(eng, rng):
    """The engine is run on spotted candidates, so its prior leaves little mass on 'no AU' (P(null) is not P(noise)).  What
    must hold is the weaker, load-bearing property: noise does not get an AU named, and no hypothesis about it is confident."""
    named, confident = 0, 0
    for _ in range(12):
        post = eng.infer(rng.normal(0, SIGMA, (27, 2)), sigma_arr(), n_samples=40)
        named += int(post.pip.max() > 0.5)
        confident += int(max(post.set_probs.values()) > 0.9)
    assert named <= 2 and confident == 0


def test_pair_of_aus_is_recovered(eng, rng):
    post = eng.infer(pattern({6: 1.6, 12: 2.0}, rng), sigma_arr(), n_samples=60)
    assert post.pip[TPL.AUS.index(6)] > 0.5 and post.pip[TPL.AUS.index(12)] > 0.5
    assert max(post.set_probs, key=post.set_probs.get) == frozenset({6, 12})


def test_amplitude_read_out_is_not_biased_low(eng):
    """Regression: weights that followed the observed magnitude down-weighted the regions that moved most and read amplitudes ~35 % low."""
    ratios = []
    for seed, au in enumerate((12, 4, 17, 9, 6, 15, 2, 7, 10, 1)):
        rng = np.random.default_rng(100 + seed)
        true = 1.8
        post = eng.infer(pattern({au: true}, rng), sigma_arr(), n_samples=200, rng=np.random.default_rng(seed))
        j = TPL.AUS.index(au)
        if post.pip[j] > 0.5:
            ratios.append(post.amp_given[j] / true)
    assert len(ratios) >= 9
    assert 0.8 < np.median(ratios) < 1.25, np.round(ratios, 2)


def test_amplitude_interval_brackets_the_truth_most_of_the_time(eng):
    hit = 0
    n = 20
    for seed in range(n):
        rng = np.random.default_rng(500 + seed)
        post = eng.infer(pattern({12: 1.5}, rng), sigma_arr(), n_samples=300, rng=np.random.default_rng(seed))
        j = TPL.AUS.index(12)
        hit += int(post.amp_given_lo[j] <= 1.5 <= post.amp_given_hi[j])
    assert hit >= 0.6 * n                                                   # intervals are not tiny and not centred on something else


def test_rigid_head_motion_is_not_an_expression(eng, rng):
    c, centre, _ = fake_geometry()
    nuis = TPL.nuisance_columns(c, centre)
    named = 0
    for k in (0, 1, 2, 3):                                                  # translation x/y, scale, rotation
        for _ in range(3):
            y = 2.5 * nuis[k] / np.abs(nuis[k]).max() + rng.normal(0, SIGMA, (27, 2))
            post = eng.infer(y, sigma_arr(), n_samples=30)
            named += int(post.pip.max() > 0.5)
        want = 2.5 / np.abs(nuis[k]).max()
        assert post.rigid[k] == pytest.approx(want, rel=0.25), (k, post.rigid.round(3))       # the motion went into the nuisance columns ...
    assert named <= 2                                                       # ... so what is left is noise, which names an AU rarely


def test_blink_is_attributed_to_blink_when_the_prior_allows_it(eng, rng):
    y = 2.0 * TPL.blink_template() + rng.normal(0, SIGMA, (27, 2))
    post = eng.infer(y, sigma_arr(), p_blink=0.6, n_samples=30)
    assert post.p_blink > 0.5 and post.pip.max() < 0.5


def test_masking_a_region_removes_its_evidence(eng, rng):
    from onset.vision import geometry as G
    y = pattern({17: 2.0}, rng)
    mask = np.ones(27)
    for n in G.ROI_GROUPS["mouth"]:
        mask[G.ROI_NAMES.index(n)] = 0.0
    full = eng.infer(y, sigma_arr(), n_samples=20).pip[TPL.AUS.index(17)]
    masked = eng.infer(y, sigma_arr(), mask=mask, n_samples=20).pip[TPL.AUS.index(17)]
    assert full > 0.5 > masked


def test_posterior_bookkeeping(eng, rng):
    post = eng.infer(pattern({4: 1.5, 7: 1.0}, rng), sigma_arr(), n_samples=100)
    assert sum(post.set_probs.values()) == pytest.approx(1.0, abs=1e-6)
    assert ((post.pip >= 0) & (post.pip <= 1 + 1e-9)).all()
    assert post.pip.sum() <= 3 + 1e-6                                       # at most three AUs per hypothesis
    pres = post.amp_given > 0
    assert (post.amp_given_lo[pres] <= post.amp_given[pres] + 1e-9).all() and (post.amp_given[pres] <= post.amp_given_hi[pres] + 1e-9).all()
    assert post.hypotheses[0]["p"] >= post.hypotheses[-1]["p"]
    assert set(post.fit) == {"data", "aus", "nuisance", "residual"}
    assert post.fit["data"].shape == (27, 2) and post.rigid.shape == (len(TPL.nuisance_columns(*fake_geometry()[:2])),)


def test_inference_is_deterministic_for_a_seed(eng, rng):
    y = pattern({12: 1.5}, rng)
    a = eng.infer(y, sigma_arr(), n_samples=80, rng=np.random.default_rng(3))
    b = eng.infer(y, sigma_arr(), n_samples=80, rng=np.random.default_rng(3))
    assert np.allclose(a.pip, b.pip) and np.allclose(a.amp_given, b.amp_given)


def test_fit_hypothesis_contract(eng, rng):
    y = pattern({12: 2.0}, rng)
    f = eng.fit_hypothesis([12], y, sigma_arr())
    assert set(f) >= {"fit", "au", "events", "nuis", "resid", "coef", "sd", "chi2", "weights"}
    assert f["fit"].shape == f["au"].shape == (27, 2)
    j = TPL.AUS.index(12)
    assert 1.5 < f["coef"][j] < 2.5
    assert np.allclose(f["fit"] + f["resid"], y)
    f2 = eng.fit_hypothesis([12], 2 * y, sigma_arr())
    assert f2["coef"][j] > f["coef"][j] * 1.5                               # monotone in the data
    f3 = eng.fit_hypothesis([12], -y, sigma_arr())
    assert f3["coef"][j] == 0.0                                             # an AU only pulls one way
    sig = np.full((27, 2), 0.1)
    explicit = eng.fit_hypothesis([12], y, sig, signal=f["au"])
    assert explicit["coef"][j] == pytest.approx(f["coef"][j], rel=0.05)


def test_eta_zero_uses_plain_noise_weights(rng):
    c, centre, iod = fake_geometry()
    e0 = AUEngine(c, centre, iod, eta=0.0)
    y = pattern({12: 3.0}, rng)
    f = e0.fit_hypothesis([12], y, sigma_arr())
    assert np.allclose(f["weights"], 1.0 / SIGMA)


def test_more_noise_means_more_doubt(eng, rng):
    y_clean = pattern({12: 0.6}, np.random.default_rng(1), noise=0.0)
    quiet = eng.infer(y_clean + np.random.default_rng(2).normal(0, 0.05, (27, 2)), sigma_arr(0.05), n_samples=20)
    loud = eng.infer(y_clean + np.random.default_rng(2).normal(0, 0.5, (27, 2)), sigma_arr(0.5), n_samples=20)
    j = TPL.AUS.index(12)
    assert quiet.pip[j] > loud.pip[j]
    assert loud.p_null > quiet.p_null


def test_size_prior_shifts_the_number_of_aus():
    c, centre, iod = fake_geometry()
    few = AUEngine(c, centre, iod, size_prior=(0.05, 0.9, 0.04, 0.01))
    many = AUEngine(c, centre, iod, size_prior=(0.05, 0.05, 0.3, 0.6))
    y = pattern({6: 0.9, 12: 0.9}, np.random.default_rng(0))
    s_few, s_many = few.infer(y, sigma_arr(), n_samples=10), many.infer(y, sigma_arr(), n_samples=10)
    n_few = sum(len(k) * v for k, v in s_few.set_probs.items())
    n_many = sum(len(k) * v for k, v in s_many.set_probs.items())
    assert n_many >= n_few
