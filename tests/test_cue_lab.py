import numpy as np
import pytest
from scipy import stats

from onset.analysis import cue_lab as C
from onset.vision import geometry as G


# ---------------------------------------------------------------------------------------------------- no verdicts, ever
@pytest.mark.parametrize("bad", [{"liar": 0.2}, {"a": [{"b": {"is_lying": True}}]}, {"x": {"deception_score": 1}},
                                 {"Verdict": "guilty"}, [{"truthful": 1}]])
def test_verdict_shaped_keys_are_refused_wherever_they_are(bad):
    with pytest.raises(ValueError, match="verdict-shaped key"):
        C.assert_no_verdict(bad)


def test_ordinary_keys_that_merely_contain_forbidden_letters_pass():
    C.assert_no_verdict({"allies": 1, "supplies": [{"belies": 2}], "rate_per_min": 3.0})


def test_everything_the_module_publishes_is_free_of_verdicts():
    C.assert_no_verdict(C.evidence_payload())
    C.assert_no_verdict(C.cue_lab_demo_null())
    C.assert_no_verdict(C.predictive_values(0.9, 0.9, 0.05))
    C.assert_no_verdict(C.audit_claims(_result(), None))


def test_the_evidence_table_says_which_sources_were_actually_opened():
    opened = sorted(k for k, v in C.REFS.items() if v.get("opened"))
    assert opened == ["bond2006", "burgoon2018"]               # docs/ETHICS.md states this; change both together
    for row in C.evidence_payload()["evidence"]:
        assert row["status"] in C.STATUS_TEXT
        assert all(r in C.REFS for r in row["refs"])
        # a row that cites nothing must say so (or be ONSET's own observation about its calibration data)
        assert row["refs"] or row["status"] == "confound" or "no source" in " ".join(row["known"]).lower(), row["id"]


def test_the_lie_claim_is_always_not_supported():
    cards = {c["id"]: c for c in C.audit_claims(_result(), None)}
    assert cards["lied"]["status"] == "not_supported" and cards["felt"]["status"] == "not_supported"


# ---------------------------------------------------------------------------------------------------- base rates
def test_predictive_values_match_a_hand_calculation():
    r = C.predictive_values(0.9, 0.9, 0.05)
    assert r["ppv"] == pytest.approx(0.045 / 0.14)
    assert r["npv"] == pytest.approx(0.855 / 0.86)
    assert r["per_1000"] == pytest.approx(dict(true_positive=45, false_negative=5, false_positive=95, true_negative=855))


def test_predictive_values_reject_impossible_inputs():
    with pytest.raises(ValueError):
        C.predictive_values(1.2, 0.9, 0.05)


def test_accuracy_needs_a_large_standardised_difference():
    assert C.d_needed_for_balanced_accuracy(0.5) == pytest.approx(0.0, abs=1e-9)
    d90 = C.d_needed_for_balanced_accuracy(0.9)
    assert d90 == pytest.approx(2 * stats.norm.ppf(0.9)) and d90 > 2.5
    assert C.d_from_auc(C.auc_from_d(1.3)) == pytest.approx(1.3, abs=1e-6)


# ---------------------------------------------------------------------------------------------------- small counts
def test_garwood_interval_with_no_events():
    r = C.poisson_rate_ci(0, 60.0)
    assert r["lo"] == 0.0 and r["hi"] == pytest.approx(3.689, abs=1e-3)           # "rule of three", exactly


def test_garwood_interval_for_ten_events():
    r = C.poisson_rate_ci(10, 120.0)
    assert r["rate_per_min"] == pytest.approx(5.0)
    assert r["lo"] == pytest.approx(4.795 / 2, rel=1e-3) and r["hi"] == pytest.approx(18.390 / 2, rel=1e-3)


def test_rate_test_equal_counts_equal_time_is_a_tie():
    r = C.rate_ratio_test(5, 60.0, 5, 60.0)
    assert r["rr"] == pytest.approx(1.0) and r["p"] == pytest.approx(1.0) and r["lo"] < 1.0 < r["hi"]


def test_rate_test_all_events_in_one_stretch():
    r = C.rate_ratio_test(10, 60.0, 0, 60.0)
    assert r["p"] == pytest.approx(2 * 0.5 ** 10) and r["rr"] == float("inf")


def test_rate_test_accounts_for_unequal_observation_time():
    r = C.rate_ratio_test(3, 30.0, 9, 90.0)                                           # the same rate in a stretch three times longer
    assert r["rr"] == pytest.approx(1.0) and r["p"] > 0.9


def test_rate_test_without_observable_time_makes_no_claim():
    r = C.rate_ratio_test(3, 0.0, 2, 60.0)
    assert r["p"] == 1.0 and np.isnan(r["rr"]) and r["note"]


def test_a_handful_of_events_can_never_be_significant():
    assert C.min_events_for_significance(60.0, 60.0) == 6                                # 2 * 0.5**6 = 0.031 < 0.05 but 2 * 0.5**5 = 0.0625
    assert C.min_events_for_significance(60.0, 60.0, alpha=0.01) == 8                    # 2 * 0.5**8 = 0.0078
    lopsided = C.min_events_for_significance(10.0, 110.0)
    assert lopsided == 2                                                                 # (1/12)**2 = 0.007: two events both landing in the short stretch is a real coincidence
    assert lopsided < C.min_events_for_significance(60.0, 60.0)                          # equal stretches are the hardest case: the bound is a worst case, not a typical one


def test_holm_by_hand():
    assert C.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    adj = C.holm_adjust(list(np.random.default_rng(0).random(20)))
    assert all(0.0 <= a <= 1.0 for a in adj)


def test_holm_never_lowers_a_p_value():
    p = list(np.random.default_rng(1).random(15) * 0.2)
    assert all(a >= b - 1e-12 for a, b in zip(C.holm_adjust(p), p))


def test_forking_paths_formula():
    assert C.forking_paths(1) == pytest.approx(0.05)
    assert C.forking_paths(20) == pytest.approx(1 - 0.95 ** 20)


def test_null_simulation_shows_the_forking_paths_and_holm_removes_them():
    s = C.simulate_null_comparisons(rate_per_min=6.0, seconds_a=60.0, seconds_b=60.0, n_tests=9, n_sims=500, seed=3)
    assert s["p_any_uncorrected"] > 0.12                                               # discrete small counts make the exact test conservative, but still well above 5 %
    assert s["p_any_holm"] <= 0.06 and s["p_any_holm"] < s["p_any_uncorrected"]
    assert s["analytic_independent_continuous"] == pytest.approx(1 - 0.95 ** 9)
    assert C.simulate_null_comparisons(6.0, 60.0, 60.0, 9, n_sims=60, seed=3) == C.simulate_null_comparisons(6.0, 60.0, 60.0, 9, n_sims=60, seed=3)


# ---------------------------------------------------------------------------------------------------- the shift test
def test_circular_shift_test_finds_a_planted_difference_and_not_a_null():
    rng = np.random.default_rng(0)
    n = 600
    ma, mb = np.zeros(n, bool), np.zeros(n, bool)
    ma[100:250], mb[350:500] = True, True
    x = np.convolve(rng.normal(0, 1, n), np.ones(5) / 5, mode="same")                    # autocorrelated noise
    null = C.circular_shift_test(x, ma, mb, n_perm=800, seed=1)
    assert null["p"] > 0.05 and np.isfinite(null["effect"])
    x2 = x.copy()
    x2[ma] += 2.0
    hit = C.circular_shift_test(x2, ma, mb, n_perm=800, seed=1)
    # a shift that happens to line the planted block up with a segment again is not rare (a few dozen of 600), so the
    # test cannot get far below a couple of percent on one block: that is the price of keeping the autocorrelation
    assert hit["p"] < 0.05 and hit["p"] < null["p"] / 5 and hit["diff"] > 1.5 and hit["effect"] > 3.0


def test_circular_shift_test_refuses_to_test_thin_data():
    ma, mb = np.zeros(100, bool), np.zeros(100, bool)
    ma[:2], mb[50:60] = True, True
    r = C.circular_shift_test(np.random.default_rng(0).normal(size=100), ma, mb)
    assert r["p"] == 1.0 and r["note"]


# ---------------------------------------------------------------------------------------------------- segment study
def _result(T=40.0, fps=25.0, events=None, blink_windows=((10.0, 10.3),), speech_windows=(), brow_boost=None, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T, 1.0 / fps)
    n = len(t)
    groups = {g: np.convolve(rng.normal(1.0, 0.2, n), np.ones(5) / 5, mode="same") for g in G.GROUP_ORDER}
    if brow_boost:
        t0, t1, amount = brow_boost
        groups["brow"][(t >= t0) & (t <= t1)] += amount
    if events is None:
        events = [dict(id="e1", status="micro", t_apex=5.0, pattern=[dict(roi="brow_in_L", dx=0.0, dy=-1.0, used=True)]),
                  dict(id="e2", status="micro", t_apex=7.0, pattern=[dict(roi="lip_up", dx=0.0, dy=-1.0, used=True)]),
                  dict(id="e3", status="micro", t_apex=30.0, pattern=[dict(roi="brow_mid_R", dx=0.0, dy=-1.0, used=True)])]
    return dict(timeline=dict(t=t.tolist(), groups={g: v.tolist() for g, v in groups.items()}, blink=np.zeros(n).tolist(),
                              speech=np.zeros(n).tolist(), face_found=np.ones(n).tolist()),
                nuisance=dict(blink_windows=[list(w) for w in blink_windows], speech_windows=[list(w) for w in speech_windows]),
                events=events, quality=dict(grade="B"), engine=dict(calibrated=True))


def test_segment_cues_count_by_region_and_correct_the_observable_time():
    a, b = C.segment_cues(_result(), [dict(label="a", t0=0.0, t1=20.0), dict(label="b", t0=20.0, t1=40.0)])
    assert a["n_events"]["all"] == 2 and b["n_events"]["all"] == 1
    assert a["n_events"]["brow"] == 1 and a["n_events"]["mouth"] == 1 and b["n_events"]["brow"] == 1
    assert a["observable_seconds"]["eye"] == pytest.approx(20.0 - 0.3, abs=0.05)         # the eye cannot be seen while blinking
    assert a["observable_seconds"]["brow"] == pytest.approx(20.0, abs=0.05) and a["n_blinks"] == 1 and b["n_blinks"] == 0


def test_segment_cues_remove_speech_time_from_the_mouth_only():
    a, = C.segment_cues(_result(speech_windows=((2.0, 6.0),)), [dict(label="a", t0=0.0, t1=20.0)])
    assert a["observable_seconds"]["mouth"] == pytest.approx(16.0, abs=0.05) and a["observable_seconds"]["brow"] == pytest.approx(20.0, abs=0.05)
    assert a["speech_fraction"] == pytest.approx(0.2, abs=0.01)


def test_a_backwards_segment_is_an_error():
    with pytest.raises(ValueError, match="ends before it starts"):
        C.segment_cues(_result(), [dict(label="x", t0=5.0, t1=5.0)])


def test_comparison_of_identical_behaviour_finds_nothing_and_says_why():
    r = C.compare_segments(_result(), dict(label="a", t0=0.0, t1=20.0), dict(label="b", t0=20.0, t1=40.0), n_perm=600)
    assert r["family_size"] == len(r["tests"]) == 1 + len(G.GROUP_ORDER) + 1 + len(G.GROUP_ORDER)
    assert not any(t["survives"] for t in r["tests"])
    assert r["total_events"] < r["min_events_needed"] and "No difference can be shown" in r["summary"]
    assert all(t["p_holm"] >= t["p"] - 1e-12 for t in r["tests"])
    C.assert_no_verdict(r)
    assert "truthful" in r["statement"] and "nothing about truthfulness" in r["statement"]


def test_comparison_picks_up_a_planted_difference_in_one_region_after_correction():
    r = C.compare_segments(_result(brow_boost=(20.0, 40.0, 1.5)), dict(label="a", t0=0.0, t1=20.0), dict(label="b", t0=20.0, t1=40.0), n_perm=800)
    brow = next(t for t in r["tests"] if t["name"] == "Movement activity, brow region")
    assert brow["survives"] and brow["diff"] < 0
    assert sum(t["survives"] for t in r["tests"]) == 1                                       # only the planted one
    assert "remain below the threshold after correcting" in r["summary"]
    differs = next(c for c in C.audit_claims(_result(), r) if c["id"] == "differs")
    assert differs["status"] == "conditional"


def test_a_comparison_in_a_speech_heavy_stretch_is_flagged_as_confounded():
    res = _result(speech_windows=((1.0, 15.0),))
    r = C.compare_segments(res, dict(label="a", t0=0.0, t1=20.0), dict(label="b", t0=20.0, t1=40.0), n_perm=300)
    differs = next(c for c in C.audit_claims(res, r) if c["id"] == "differs")
    assert any("confounded" in w for w in differs["why"])
