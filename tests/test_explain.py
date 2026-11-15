"""The explanations: direction words, the evidence ledger, the deletion test, the contrast and the narrative."""
from types import SimpleNamespace

import numpy as np
import pytest

from onset.analysis import explain as EX
from onset.analysis import templates as TPL
from onset.analysis.au_engine import AUEngine
from onset.vision import geometry as G
from helpers import fake_geometry

SIGMA = 0.10


@pytest.fixture(scope="module")
def eng():
    c, centre, iod = fake_geometry()
    return AUEngine(c, centre, iod, eta=0.15, temper=1.0)


@pytest.fixture(scope="module")
def iod():
    return fake_geometry()[2]


def _sigma():
    return np.full((27, 2), SIGMA)


def _au12(seed=3, amp=2.0):
    rng = np.random.default_rng(seed)
    return amp * TPL.au_templates()[TPL.AUS.index(12)] + rng.normal(0, SIGMA, (27, 2))


@pytest.fixture(scope="module")
def case(eng, iod):
    y = _au12()
    post = eng.infer(y, _sigma(), n_samples=60)
    return y, post, EX.evidence_ledger(eng, post, y, _sigma(), None, iod)


# ------------------------------------------------------------------------------------------------ directions
@pytest.mark.parametrize("side,dx,dy,word", [
    ("R", 1, 0, "outward"), ("L", -1, 0, "outward"),                  # outward is away from the midline, whichever side it is
    ("R", -1, 0, "inward"), ("L", 1, 0, "inward"),
    ("R", 0, -1, "up"), ("L", 0, -1, "up"), ("R", 0, 1, "down"),       # image y runs downward
    ("R", 1, -1, "up and outward"), ("L", -1, -1, "up and outward"),
    ("R", -1, -1, "up and inward"), ("R", 1, 1, "down and outward"), ("R", -1, 1, "down and inward"),
    ("R", 0, 0, "still"), ("C", 0, 0, "still")])
def test_direction_words(side, dx, dy, word):
    assert EX.direction_word(dx, dy, side) == word


def test_every_direction_has_a_word_and_mirroring_preserves_it():
    for ang in np.linspace(-179, 180, 73):
        dx, dy = np.cos(np.radians(ang)), -np.sin(np.radians(ang))
        w = EX.direction_word(dx, dy, "R")
        assert w and EX.direction_word(-dx, dy, "L") == w


def test_side_of_a_region_name():
    assert EX._side("corner_L") == "L" and EX._side("corner_R") == "R" and EX._side("chin") == "C"


# ------------------------------------------------------------------------------------------------ the ledger
def test_ledger_has_one_row_per_region_sorted_by_evidence(case):
    y, post, led = case
    assert [d["roi"] for d in sorted(led, key=lambda d: G.ROI_NAMES.index(d["roi"]))] == list(G.ROI_NAMES)
    ev = [d["evidence"] for d in led]
    assert ev == sorted(ev, reverse=True)


def test_ledger_points_at_the_regions_the_named_unit_moves(case):
    y, post, led = case
    top = {d["roi"].rsplit("_", 1)[0] for d in led[:6]}
    assert top <= {"corner", "nasolab", "cheek"}                       # the lip-corner puller's regions, and nothing else
    assert all(12 in d["aus"] for d in led[:6]) and all(d["direction"].endswith("outward") for d in led[:4])


def test_regions_the_unit_does_not_touch_carry_no_evidence(case):
    y, post, led = case
    by = {d["roi"]: d for d in led}
    for quiet in ("chin", "nose_root", "brow_out_R"):
        assert abs(by[quiet]["evidence"]) < 0.5, quiet
    assert max(d["evidence"] for d in led) > 20


def test_masked_regions_are_marked_unused_and_carry_zero(eng, case, iod):
    y, post, _ = case
    mask = np.ones(27)
    mask[G.ROI_NAMES.index("brow_in_R")] = 0
    led = EX.evidence_ledger(eng, post, y, _sigma(), mask, iod)
    row = next(d for d in led if d["roi"] == "brow_in_R")
    assert row == dict(roi="brow_in_R", group="brow", used=False, evidence=0.0)
    assert sum(1 for d in led if d["used"]) == 26


def test_ledger_magnitudes_are_in_percent_of_eye_distance(case, iod):
    y, post, led = case
    for d in led:
        if d["used"]:
            assert d["mag_pct_iod"] == pytest.approx(np.hypot(d["dx_px"], d["dy_px"]) / iod * 100)


# ------------------------------------------------------------------------------------------------ the deletion test
def test_deletion_test_on_a_clear_movement_is_faithful_or_robust(eng, case):
    y, post, led = case
    dt = EX.deletion_test(eng, y, _sigma(), None, led, [12], post.p_blink, post.p_speech, n_max=6, n_random=6)
    assert dt["verdict"] in ("faithful", "robust") and dt["baseline"] > 5
    assert dt["n"] == list(range(0, len(dt["ranked"])))
    assert len(dt["ranked"]) == len(dt["random_mean"]) == len(dt["random_sd"]) == len(dt["ranked_p"])
    assert dt["ranked"][0] == pytest.approx(dt["baseline"]) and dt["removed_order"][0] == led[0]["roi"]
    assert min(dt["ranked"]) <= dt["baseline"]                          # removing its best evidence can only cost support
    assert 0 < dt["p_empirical"] <= 1 and set(dt["flips"]) == {"ranked", "random_median", "random_never", "n_random"}


def test_deletion_test_has_nothing_to_test_when_nothing_was_concluded(eng, iod):
    y = np.random.default_rng(5).normal(0, SIGMA, (27, 2))
    post = eng.infer(y, _sigma(), n_samples=40)
    led = EX.evidence_ledger(eng, post, y, _sigma(), None, iod)
    dt = EX.deletion_test(eng, y, _sigma(), None, led, [], post.p_blink, post.p_speech, n_max=4, n_random=4)
    assert dt["verdict"] == "na" and dt["faithful"] is None and dt["baseline"] <= 0
    assert dt["ranked"] == [dt["baseline"]] and dt["flips"]["ranked"] is None


def test_deletion_test_is_reproducible(eng, case):
    y, post, led = case
    a = EX.deletion_test(eng, y, _sigma(), None, led, [12], post.p_blink, post.p_speech, n_max=3, n_random=4, seed=7)
    b = EX.deletion_test(eng, y, _sigma(), None, led, [12], post.p_blink, post.p_speech, n_max=3, n_random=4, seed=7)
    assert a["ranked"] == b["ranked"] and a["random_mean"] == b["random_mean"]


def test_support_is_the_weakest_named_unit_and_logit_is_finite():
    post = SimpleNamespace(pip=np.array([0.9 if a == 12 else 0.2 if a == 6 else 0.01 for a in TPL.AUS]), p_null=0.3)
    assert EX._support(post, [12]) == pytest.approx(np.log(0.9 / 0.1))
    assert EX._support(post, [12, 6]) == pytest.approx(np.log(0.2 / 0.8))        # one lost unit is a changed conclusion
    assert EX._support(post, []) == pytest.approx(np.log(0.7 / 0.3))
    assert np.isfinite(EX._logit(0.0)) and np.isfinite(EX._logit(1.0)) and EX._logit(0.5) == pytest.approx(0.0)


def test_first_flip_counts_deletions_until_support_goes_negative():
    assert EX._first_flip([3, 2, 1, -0.5, -2]) == 3
    assert EX._first_flip([3, 2, 1]) is None
    assert EX._first_flip([-1, 2]) is None                              # the starting point does not count as a flip


# ------------------------------------------------------------------------------------------------ contrast and narrative
def test_contrast_names_the_regions_that_separate_two_hypotheses(eng, case):
    y, post, _ = case
    rows = EX.contrast(eng, y, _sigma(), None, dict(aus=[12]), dict(aus=[6]), top=5)
    assert len(rows) == 5 and [abs(r["favours_a"]) for r in rows] == sorted((abs(r["favours_a"]) for r in rows), reverse=True)
    assert sum(r["favours_a"] > 0 for r in rows) >= 4                   # a smile pattern is better explained as a smile than as a cheek raiser
    assert all(len(r["a_vec"]) == 2 and len(r["measured"]) == 2 for r in rows)


def test_narrative_names_the_unit_the_regions_and_the_duration(case, iod):
    y, post, led = case
    text = EX.narrative(post, led, iod, 0.3)
    assert "AU12" in text and "300 ms" in text and "corner" in text and "% IOD" in text


def test_narrative_without_a_nameable_pattern_says_so_and_gives_the_probability(eng, iod):
    y = np.random.default_rng(5).normal(0, SIGMA, (27, 2))
    post = eng.infer(y, _sigma(), n_samples=40)
    text = EX.narrative(post, EX.evidence_ledger(eng, post, y, _sigma(), None, iod), iod, None)
    assert "does not match any single AU pattern" in text and "%" in text


def test_narrative_defers_to_blinks_and_speech(iod):
    base = dict(aus=list(TPL.AUS), pip=np.zeros(len(TPL.AUS)), p_null=0.5)
    assert "blink" in EX.narrative(SimpleNamespace(**base, p_blink=0.8, p_speech=0.1), [], iod, None)
    assert "talking" in EX.narrative(SimpleNamespace(**base, p_blink=0.1, p_speech=0.8), [], iod, None)
