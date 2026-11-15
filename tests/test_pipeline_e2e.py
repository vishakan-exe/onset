"""The whole chain on small phantom clips: a smile, a blink, nothing.  Each analysis takes about ten seconds, so these are marked slow.

They check what the person will rely on: the movement is found where it was scripted and named, a blink is not called an
expression, a face that did nothing produces no event, the numbers are honest JSON, and only grayscale plates leave the analysis."""
import json
import os

import cv2
import numpy as np
import pytest

from onset.analysis import pipeline as PL
from onset.analysis.pipeline import AnalysisConfig, analyse
from onset.synth.render import Renderer
from onset.synth.scenarios import make_null, make_scenario
from onset.vision.sources import ArraySource

pytestmark = pytest.mark.slow


def _strict_json(path):
    def refuse(c):
        raise AssertionError(f"result.json contains {c}")
    with open(path) as fh:
        return json.load(fh, parse_constant=refuse)


def _run(scenario, root, name):
    clip = Renderer(scenario).render()
    out = str(root / name)
    os.makedirs(out)
    return analyse(ArraySource(clip.frames, clip.times), out, AnalysisConfig(threads=1)), out, clip


@pytest.fixture(scope="module")
def blink_run(tmp_path_factory):
    sc = make_scenario("B", "webcam30", 4.0, [], seed=4, nuisance="still", size=360, nuisance_overrides=dict(blinks=[(1.6, 0.22)]))
    return _run(sc, tmp_path_factory.mktemp("blink"), "run")


@pytest.fixture(scope="module")
def null_run(tmp_path_factory):
    return _run(make_null("B", "webcam30", 4.0, 1, nuisance="still", size=360), tmp_path_factory.mktemp("null"), "run")


def test_the_scripted_movement_is_found_where_it_was_scripted_and_named(analysed):
    res, out = analysed
    micro = [e for e in res["events"] if e["status"] == "micro"]
    assert len(micro) == 1
    e = micro[0]
    assert abs(e["t_apex"] - 1.70) < 0.12
    named = {a["au"]: a for a in e["aus"]}
    assert 12 in named and named[12]["pip"] > 0.9
    assert 1.0 < named[12]["amp_pct_iod"] < 2.4                     # scripted 2.2 % of the eye distance; the read-out is a little low, never high
    assert named[12]["lo_pct_iod"] < named[12]["amp_pct_iod"] < named[12]["hi_pct_iod"]


def test_a_blink_is_called_a_blink_not_an_expression(blink_run):
    res, out, clip = blink_run
    assert [e["status"] for e in res["events"]] == ["blink"]
    assert abs(res["events"][0]["t_apex"] - 1.7) < 0.2
    assert not any(e["status"] == "micro" for e in res["events"])


def test_a_face_that_did_nothing_yields_no_event(null_run):
    res, out, clip = null_run
    assert res["events"] == [] and clip.truth["events"] == []


def test_the_report_grades_the_recording_and_states_what_it_could_see(analysed):
    res, out = analysed
    q = res["quality"]
    assert q["grade"] in "ABC" and q["mda_median_pct"] > 0 and q["min_duration_ms"] > 0
    assert q["items"] and all({"name", "value", "status"} <= set(i) for i in q["items"])
    assert set(q["mda90_pct"]) == set(q["mda50_pct"]) and all(q["mda90_pct"][k] > q["mda50_pct"][k] for k in q["mda50_pct"])
    assert "synthetic faces" in q["summary"]                          # the sensitivity claim says what it was measured on


def test_stage_timings_are_recorded_and_total_is_their_sum(analysed):
    res, out = analysed
    t = res["timing"]
    for k in ("track_register_s", "spot_s", "measure_s", "infer_s", "explain_s", "render_s", "total_s"):
        assert t[k] >= 0
    assert t["total_s"] == pytest.approx(sum(v for k, v in t.items() if k != "total_s"))


def test_the_result_is_strict_json_without_nan_or_infinity(analysed, blink_run, null_run):
    for res, out in (analysed, blink_run[:2], null_run[:2]):
        data = _strict_json(os.path.join(out, "result.json"))
        assert data["version"] == PL.VERSION and data["clip"]["n_frames"] > 50


def test_only_grayscale_plates_and_numbers_are_kept(analysed):
    res, out = analysed
    names = sorted(os.listdir(out))
    assert "result.json" in names
    assert all(n.endswith((".jpg", ".png", ".json")) for n in names)              # no video, no raw frames
    for n in names:
        im = cv2.imread(os.path.join(out, n), cv2.IMREAD_UNCHANGED) if n.endswith((".jpg", ".png")) else None
        if im is None:
            continue
        if n.endswith(".jpg"):                                                      # plates and magnified strips: the face, in grey
            assert im.ndim == 2 or (im[..., 0] == im[..., 1]).all() and (im[..., 1] == im[..., 2]).all(), n
        else:                                                                       # motion-heat overlays: one flat colour, the movement map is the alpha channel
            assert im.shape[2] == 4 and all(len(np.unique(im[..., c])) == 1 for c in range(3)), n


def test_limits_are_part_of_every_result(analysed):
    res, out = analysed
    assert res["limits"] and all(isinstance(x, str) and x for x in (res["limits"] if isinstance(res["limits"], list) else res["limits"].values()))


def test_the_analysis_is_deterministic(phantom_clip, analysed, tmp_path):
    res, out = analysed
    again = analyse(ArraySource(phantom_clip.frames, phantom_clip.times), str(tmp_path), AnalysisConfig(threads=1))
    a = [(e["status"], round(e["t_apex"], 3), [(x["au"], round(x["amp_pct_iod"], 4)) for x in e["aus"]]) for e in res["events"]]
    b = [(e["status"], round(e["t_apex"], 3), [(x["au"], round(x["amp_pct_iod"], 4)) for x in e["aus"]]) for e in again["events"]]
    assert a == b


def test_a_clip_that_is_too_short_or_has_no_face_raises_an_error_that_says_what_to_do(tmp_path):
    frames = np.random.default_rng(0).integers(0, 255, (40, 96, 96, 3), dtype=np.uint8)
    with pytest.raises(PL.AnalysisError, match="No face"):
        analyse(ArraySource(frames, np.arange(40) / 30.0), str(tmp_path), AnalysisConfig(threads=1))
    with pytest.raises(PL.AnalysisError, match="too short|shorter"):
        analyse(ArraySource(frames[:8], np.arange(8) / 30.0), str(tmp_path), AnalysisConfig(threads=1))


def test_exact_truth_in_the_instruments_units_matches_the_scripted_size(phantom_clip):
    """The scripted 2.2 % of the eye distance is a generator parameter; in the instrument's unit the largest region moves about that much."""
    from onset.analysis.pipeline import detect
    from onset.synth.truth import exact_events
    from onset.synth.render import Event, Renderer as R
    ev = Event({12: 1.0}, 1.45, 1.70, 2.05, amp=0.022, label="smile", kind="micro")
    rend = R(make_scenario("A", "webcam30", 3.2, [ev], seed=5, nuisance="still", size=360, name="pytest-au12"))
    clip = rend.render()
    det = detect(ArraySource(clip.frames, clip.times), AnalysisConfig(threads=1))
    ex = exact_events(rend, clip, det, clip.truth["events"])
    assert len(ex) == 1 and ex[0]["aus"] == [12]
    assert 1.5 < ex[0]["size_pct"] < 3.0 and 1.2 < ex[0]["a_true_pct"][12] < 2.8
    assert ex[0]["fit_r2"] > 0.7 and 0.8 < ex[0]["pair_gain"] < 1.3
