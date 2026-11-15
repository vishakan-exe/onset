import numpy as np
import pytest

from onset.analysis import categories as CAT
from onset.analysis import uncertainty as UNC
from onset.learn import labels as L


def post(*aus_sets_p, scheme="basic7", artefact=0.0, p_null=0.0, **kw):
    sp = {frozenset(s): p for s, p in aus_sets_p}
    return CAT.category_posterior(sp, artefact, CAT.SCHEMES[scheme](), p_null, **kw)


@pytest.mark.parametrize("aus,want", [((6, 12), "happiness"), ((12,), "happiness"), ((9, 10), "disgust"), ((1, 2, 5), "surprise"),
                                      ((4, 7), "anger"), ((14,), "contempt"), ((15, 17), "sadness")])
def test_prototypes_win_in_basic7(aus, want):
    r = post((aus, 1.0))
    assert r.top == want and not r.abstain, (aus, r.top, r.probs.round(2), r.unclear)


def test_empty_set_abstains():
    r = post(((), 1.0))
    assert r.abstain and r.unclear > r.top_p


def test_probabilities_are_a_distribution_with_unclear():
    r = post(((12,), 0.5), ((9,), 0.3), ((), 0.2))
    assert r.probs.sum() + r.unclear == pytest.approx(1.0)
    assert (r.probs >= 0).all() and r.margin >= 0


def test_uncertainty_over_aus_propagates_to_categories():
    sure = post(((12,), 1.0))
    mixed = post(((12,), 0.5), ((9,), 0.5))
    assert mixed.top_p < sure.top_p
    assert mixed.probs[CAT.BASIC7.index("disgust")] > 0.2


def test_artefact_is_carved_out_first():
    r = post(((12,), 1.0), artefact=0.8)
    assert r.abstain and r.artefact == 0.8
    assert r.probs.sum() + r.unclear == pytest.approx(0.2)


def test_megc3_groups_the_negative_emotions():
    assert post(((9, 10), 1.0), scheme="megc3").top == "negative"
    assert post(((6, 12), 1.0), scheme="megc3").top == "positive"
    assert post(((1, 2, 5), 1.0), scheme="megc3").top == "surprise"
    assert post(((15, 17), 1.0), scheme="megc3").top == "negative"
    assert CAT.megc3().labels == list(L.MEGC3)


def test_casme2_scheme_has_others_and_matches_the_learning_label_order():
    s = CAT.casme2_5()
    assert s.labels == list(L.CASME2_5) and s.W.shape == (5, 12)
    assert post(((17,), 1.0), scheme="casme2_5").top in ("repression", "others")
    for name, fn in CAT.SCHEMES.items():
        assert list(fn().labels) == list(L.SCHEMES[name])


def test_conformal_set_is_attached_when_a_calibrator_is_given():
    wide, tight = UNC.LACConformal(alpha=0.1), UNC.LACConformal(alpha=0.1)
    wide.qhat, tight.qhat = 0.8, 0.6                                   # a category is kept when its probability is at least 1 - qhat
    both = post(((12,), 0.6), ((9,), 0.4), calibrator=wide)
    assert {"happiness", "disgust"} <= set(both.set)
    assert post(((12,), 0.6), ((9,), 0.4), calibrator=tight).set == ["happiness"]    # a stricter calibrator commits to one
    assert post(((12,), 1.0)).set == []                                # no calibrator, no set


def test_fit_weights_learns_a_simple_rule_and_generalises():
    rng = np.random.default_rng(0)
    sets, labels = [], []
    for _ in range(120):
        if rng.random() < 0.5:
            s = [12] + ([6] if rng.random() < 0.5 else [])
            labels.append("happy")
        else:
            s = [9] + ([10] if rng.random() < 0.5 else [])
            labels.append("disgust")
        sets.append(s)
    sch = CAT.fit_weights(sets[:80], labels[:80], ["happy", "disgust"], iters=600)
    pred = [sch.labels[int(np.argmax(sch.probs_given_set(s)[:-1]))] for s in sets[80:]]
    assert np.mean(np.array(pred) == np.array(labels[80:])) > 0.95
    assert sch.probs_given_set([12])[0] > 0.8
