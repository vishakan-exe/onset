"""The calibration script's statistical helpers: the detection-probability fit, the conformal widening of amplitude intervals and the
choice of the spotting threshold.  (The script itself needs the phantom corpus; these are the parts that carry the claims.)"""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_spec = importlib.util.spec_from_file_location("calibrate_engine", os.path.join(ROOT, "scripts", "calibrate_engine.py"))
CAL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CAL)


def _sample(a, b, n, c=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(np.log(0.2), np.log(6.0), n)
    p = c / (1 + np.exp(-(a + b * x)))
    return x, (rng.random(n) < p).astype(float)


# ------------------------------------------------------------------------------------------------ detection against size
def test_logistic_fit_recovers_the_curve_it_was_drawn_from():
    x, y = _sample(a=-0.5, b=2.5, n=4000)
    a, b = CAL.fit_logistic(x, y)
    assert a == pytest.approx(-0.5, abs=0.25) and b == pytest.approx(2.5, abs=0.3)
    assert CAL.at_prob(a, b, 0.5) == pytest.approx(np.exp(0.5 / 2.5), rel=0.15)            # the 50 % point is where a + b log(x) = 0


def test_a_perfectly_separable_sample_gets_a_capped_slope_not_infinity():
    x = np.log(np.array([0.2, 0.3, 0.4, 0.5, 1.5, 2.0, 3.0, 4.0]))
    y = (x > np.log(1.0)).astype(float)
    a, b = CAL.fit_logistic(x, y)
    assert 0.3 <= b <= 40.0 and np.isfinite(a)
    assert 0.8 < CAL.at_prob(a, b, 0.5) < 1.25                                                  # and the threshold sits between the two groups


def test_the_ceiling_variant_does_not_flatten_the_slope_to_make_room_for_misses():
    x, y = _sample(a=-0.5, b=4.0, n=6000, c=0.8)
    a, b = CAL.fit_logistic(x, y)
    a2, b2, c = CAL.fit_logistic(x, y, ceiling=True)
    assert c == pytest.approx(0.8, abs=0.06) and b2 == pytest.approx(4.0, abs=0.7)
    assert b < b2                                                                               # forcing the curve to 1 flattened it
    assert 0.3 <= c <= 1.0


def test_at_prob_is_relative_to_the_ceiling():
    a, b, c = -0.5, 3.0, 0.8
    s50 = CAL.at_prob(a, b, 0.5, c)
    pts = {round(p["x"], 6): p["p"] for p in CAL.curve_pts(a, b, 0.2, 6.0, c=c)}
    assert c / (1 + np.exp(-(a + b * np.log(s50)))) == pytest.approx(0.5, abs=1e-6)           # "50 %" means a probability of 0.5, reachable below the ceiling
    assert max(pts.values()) <= c + 1e-9 and min(pts.values()) >= 0


# ------------------------------------------------------------------------------------------------ amplitude intervals
def _rows(n, spread, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        est = rng.uniform(0.5, 2.0)
        half = 0.1 * est
        true = est + rng.normal(0, spread * half)
        rows.append(dict(est=est, lo=est - half, hi=est + half, true=true, split="cal"))
    return rows


def test_interval_widening_is_one_when_there_is_too_little_to_calibrate_on():
    assert CAL.interval_k(_rows(10, 2.0)) == 1.0


def test_the_widening_factor_restores_nominal_coverage_on_the_rows_it_was_fitted_on():
    rows = _rows(400, spread=1.5)                                                               # errors about 1.5 half-widths: intervals far too narrow
    cov1 = CAL.coverage(rows)
    k = CAL.interval_k(rows, alpha=0.10)
    assert cov1 < 0.6 and k > 1.5
    assert CAL.coverage(rows, k) >= 0.90                                                        # the split-conformal guarantee, on its own calibration set
    assert CAL.coverage(rows, k) < 0.97                                                         # without being wastefully wide


def test_intervals_that_are_already_honest_are_not_widened_much():
    rows = _rows(600, spread=0.6)
    assert CAL.interval_k(rows) < 1.15 and CAL.coverage(rows) > 0.88


def test_apply_k_scales_each_side_of_the_estimate():
    r = dict(est=1.0, lo=0.9, hi=1.2, true=1.0)
    lo, hi = CAL.apply_k(r, 2.0)
    assert lo == pytest.approx(0.8) and hi == pytest.approx(1.4)
    assert CAL.coverage([]) is None


# ------------------------------------------------------------------------------------------------ the spotting threshold
def _table(rows):
    return [dict(T=T, cal=dict(fa_per_10s=fa, f1=f1)) for T, fa, f1 in rows]


def test_threshold_is_the_best_calibration_f1_within_the_false_alarm_budget():
    table = _table([(4.5, 1.2, 0.80), (5.0, 0.7, 0.78), (5.25, 0.45, 0.74), (5.5, 0.30, 0.70), (6.0, 0.10, 0.60)])
    T, met = CAL.choose_threshold(table, 0.5)
    assert (T, met) == (5.25, True)


def test_threshold_ties_go_to_the_more_conservative_value():
    table = _table([(5.0, 0.4, 0.70), (5.25, 0.3, 0.70), (5.5, 0.2, 0.60)])
    assert CAL.choose_threshold(table, 0.5) == (5.25, True)


def test_threshold_never_goes_below_the_floor_however_good_it_looks():
    table = _table([(3.5, 0.1, 0.99), (4.0, 0.1, 0.95), (4.5, 0.4, 0.70), (5.0, 0.3, 0.65)])
    T, met = CAL.choose_threshold(table, 0.5)
    assert T == 4.5 and met is True and CAL.T_FLOOR == 4.5


def test_when_no_threshold_meets_the_budget_the_caller_is_told_and_the_safest_is_returned():
    table = _table([(4.5, 3.0, 0.80), (6.0, 1.2, 0.60), (9.0, 0.9, 0.2)])
    assert CAL.choose_threshold(table, 0.5) == (9.0, False)


# ------------------------------------------------------------------------------------------------ the sensitivity factor and its check
def _rec(snr, detected, n_au=1):
    return dict(snr=snr, detected=detected, n_au=n_au)


def test_the_curve_fitted_on_one_group_is_scored_on_the_other_groups_single_unit_movements():
    a, b = 0.0, 2.0                                                                             # 50 % at 1x the estimate, 90 % at exp(ln 9 / 2) = 3x
    recs = ([_rec(0.5, False)] * 6 + [_rec(1.0, True)] * 3 + [_rec(1.0, False)] * 3 + [_rec(4.0, True)] * 9 + [_rec(4.0, False)]
            + [_rec(2.0, True, n_au=2)])                                                        # a two-unit movement is not part of the single-unit curve
    t = CAL.mda_transfer([a, b], recs)
    assert t["n"] == 22
    assert t["predicted"] == pytest.approx((6 * 0.2 + 6 * 0.5 + 10 / (1 + 4.0 ** -2.0)) / 22, abs=1e-6)
    assert t["observed"] == pytest.approx(12 / 22)
    assert t["at_90_point"] == pytest.approx(3.0, abs=1e-6)
    assert (t["n_at_or_above"], t["found_at_or_above"]) == (10, 9)
    assert 0.5 < t["found_ci_lo"] < 0.9 < t["found_ci_hi"] <= 1.0


def test_the_transfer_check_says_nothing_when_there_is_no_curve_or_too_little_to_score():
    recs = [_rec(1.0, True)] * 5
    assert CAL.mda_transfer(None, recs) == {} and CAL.mda_transfer([0.0, 2.0], recs) == {}


def test_a_check_with_no_movement_at_the_nine_in_ten_size_reports_that_instead_of_inventing_a_rate():
    t = CAL.mda_transfer([0.0, 2.0], [_rec(0.5, False)] * 6 + [_rec(1.0, True)] * 6)
    assert t["n_at_or_above"] == 0 and t["found_ci_lo"] is None and t["found_ci_hi"] is None


def test_the_slim_summary_keeps_only_what_the_evidence_page_draws():
    slim = CAL.slim_mda(dict(n_single=10, n_all=20, single_au_50pct=2.0, single_au_90pct=7.0, a50_ci=[1, 3], a90_ci=[5, 9], bins=[], curve=[1, 2], fit=[0, 1]))
    assert set(slim) == {"n_single", "n_all", "single_au_50pct", "single_au_90pct", "a50_ci", "a90_ci", "bins"}
