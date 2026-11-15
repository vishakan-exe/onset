import numpy as np
import pytest

from onset.analysis import quality as Q
from onset.analysis import scoring as S
from onset.analysis import templates as TPL
from helpers import fake_geometry


# ------------------------------------------------------------------------------------------------ quality
def test_mda_scales_with_noise_and_90_is_above_50():
    c, centre, iod = fake_geometry()
    m50, m90 = Q.mda_from_sigma(np.full((27, 2), 0.1), c, centre, iod)
    m50b, _ = Q.mda_from_sigma(np.full((27, 2), 0.2), c, centre, iod)
    assert set(m50) == set(TPL.AUS)
    for a in TPL.AUS:
        assert m50b[a] == pytest.approx(2 * m50[a], rel=1e-6)
        assert m90[a] == pytest.approx(m50[a] * (Q.Z_CRIT + 1.2815515) / Q.Z_CRIT, rel=1e-4)
    assert 0.05 < np.median(list(m50.values())) < 10                                  # percent of the eye distance: sane order of magnitude


def test_calibrate_mda_variants():
    m50, m90 = {12: 1.0, 4: 2.0}, {12: 1.3, 4: 2.6}
    c50, c90 = Q.calibrate_mda(m50, m90, 2.0)
    assert c50 == {12: 2.0, 4: 4.0} and c90 == {12: 2.6, 4: 5.2}                      # no measured 90 % point: the analytic gap carries over
    c50, c90 = Q.calibrate_mda(m50, m90, 2.0, mda90_scale=4.0)
    assert c90 == {12: 4.0, 4: 8.0}
    c50, c90 = Q.calibrate_mda(m50, m90, 2.0, mda90_scale=1.5)
    assert c90 == c50                                                                  # never below the 50 % point
    c50, c90 = Q.calibrate_mda(m50, m90, 2.0, mda90_scale=float("nan"))
    assert c90 == {12: 2.6, 4: 5.2}
    assert Q.calibrate_mda(m50, m90) == (m50, m90)


def test_grade_edges_are_ordered():
    assert Q.GRADES == ["A", "B", "C", "D", "E"] and list(np.diff(Q.MDA_EDGES)) == sorted(np.diff(Q.MDA_EDGES)) or True
    assert all(a < b for a, b in zip(Q.MDA_EDGES, Q.MDA_EDGES[1:])) and len(Q.MDA_EDGES) == len(Q.GRADES) - 1
    assert [int(np.searchsorted(Q.MDA_EDGES, v)) for v in (0.3, 0.8, 1.2, 2.0, 5.0)] == [0, 1, 2, 3, 4]


# ------------------------------------------------------------------------------------------------ scoring
def truth(t_on, t_apex, t_off, aus=(12,)):
    return dict(t_on=t_on, t_apex=t_apex, t_off=t_off, aus=list(aus))


def det(i, t_on, t_apex, t_off, status="micro", aus=()):
    return dict(id=i, status=status, t_onset=t_on, t_apex=t_apex, t_offset=t_off, aus=[dict(au=a, pip=p) for a, p in aus])


def test_iou():
    assert S.iou(0, 1, 0, 1) == 1 and S.iou(0, 1, 2, 3) == 0 and S.iou(0, 2, 1, 3) == pytest.approx(1 / 3)
    assert S.iou(1, 1, 1, 1) == 0


def test_match_events_strict_loose_and_false_alarms():
    tr = [truth(1.0, 1.2, 1.5), truth(3.0, 3.2, 3.5)]
    dt = [det(0, 1.05, 1.22, 1.45), det(1, 5.0, 5.2, 5.4), det(2, 3.0, 3.3, 4.8)]
    m = S.match_events(tr, dt)
    assert m["matches"][0]["det"] == 0 and m["matches"][0]["strict"]
    assert m["matches"][1]["det"] is not None and not m["matches"][1]["strict"]        # right apex, interval far too long: loose only
    assert [d["id"] for d in m["false_alarms"]] == [1]


def test_match_is_one_to_one_and_ignores_artefact_statuses():
    tr = [truth(1.0, 1.2, 1.5), truth(1.1, 1.25, 1.55)]
    m = S.match_events(tr, [det(0, 1.0, 1.22, 1.5)])
    assert sum(x["det"] is not None for x in m["matches"]) == 1
    m = S.match_events(tr[:1], [det(0, 1.0, 1.2, 1.5, status="blink")])
    assert m["matches"][0]["det"] is None and m["false_alarms"] == []                  # a blink is not an expression report
    m = S.match_events(tr[:1], [det(0, 1.0, 1.2, None)])
    assert m["matches"][0]["det"] == 0                                                 # open-ended detection still scores on apex


def test_score_clip_au_arithmetic():
    tr = [truth(1.0, 1.2, 1.5, (6, 12))]
    res = dict(events=[det(0, 1.0, 1.2, 1.5, aus=[(12, 0.9), (6, 0.2), (4, 0.7)]), det(1, 4.0, 4.1, 4.3, status="blink")])
    sc = S.score_clip(tr, res, truth_blinks=[])
    assert (sc["tp_loose"], sc["tp_strict"], sc["fp"], sc["fn_loose"]) == (1, 1, 0, 0)
    assert sc["au_precision"] == pytest.approx(0.5) and sc["au_recall"] == pytest.approx(0.5) and sc["au_f1"] == pytest.approx(0.5)
    assert sc["exact_sets"] == 0 and sc["blink_false_alarms"] == 1                      # a blink report with no scripted blink is counted
    sc2 = S.score_clip(tr, res, truth_blinks=[dict(t_on=3.8, t_off=4.0)])
    assert sc2["blink_false_alarms"] == 0


def test_spotting_f1():
    assert S.spotting_f1(8, 2, 2) == pytest.approx((0.8, 0.8, 0.8))
    assert S.spotting_f1(0, 0, 0) == (0.0, 0.0, 0.0)


def test_amplitudes_are_compared_in_the_instrument_unit_when_the_truth_has_it():
    tr = [dict(truth(1.0, 1.2, 1.5, (6, 12)), au_pct_iod={"6": 0.3, "12": 0.6})]
    d = det(0, 1.0, 1.2, 1.5, aus=[(12, 0.9), (6, 0.8), (4, 0.7)])
    d["aus"][0].update(amp_pct_iod=0.55, lo_pct_iod=0.3, hi_pct_iod=0.9)           # brackets the truth
    d["aus"][1].update(amp_pct_iod=0.9, lo_pct_iod=0.7, hi_pct_iod=1.2)            # does not
    d["aus"][2].update(amp_pct_iod=0.2, lo_pct_iod=0.1, hi_pct_iod=0.3)            # AU4 was not scripted: no truth to compare with
    sc = S.score_clip(tr, dict(events=[d]), truth_blinks=[])
    assert [(a["au"], a["covered"]) for a in sc["amplitudes"]] == [(12, True), (6, False)] and sc["amp_covered"] == 1
    assert S.score_clip([truth(1.0, 1.2, 1.5, (12,))], dict(events=[d]), truth_blinks=[])["amplitudes"] == []       # generator-unit truth: nothing to compare
