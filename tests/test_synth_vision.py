"""Synthetic specimens (their table, schedules and truth) and the vision building blocks (geometry, flow, sources)."""
import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from onset.analysis import templates as TPL
from onset.synth import library as LIB
from onset.synth import scenarios as SC
from onset.synth.render import Event
from onset.vision import flow as FL
from onset.vision import geometry as G
from onset.vision import sources as SRC


# ------------------------------------------------------------------------------------------------ the specimen table
def test_specimen_table_is_consistent():
    ids = [s["id"] for s in LIB.SPECIMENS]
    assert len(ids) == len(set(ids)) >= 30
    groups = {g[0] for g in LIB.GROUPS}
    assert {s["group"] for s in LIB.SPECIMENS} <= groups
    for s in LIB.SPECIMENS:
        assert s["camera"] in SC.CAMERAS and s["nuisance"] in SC.NUISANCE_LEVELS and s["phantom"] in "ABCDEF"
        assert s["title"] and s["blurb"] and s["duration"] >= 4
        for c in s.get("confusable", []):
            assert c in ids, (s["id"], c)
        for e in s["events"]:
            assert 0 < e["t_on"] < e["t_apex"] < e["t_off"] <= s["duration"] - 0.5, (s["id"], e)
            assert set(e["aus"]) <= set(TPL.AUS) and 0.1 <= e["amp_pct"] <= 12, (s["id"], e)
            assert e["kind"] in ("micro", "macro")
            if e["kind"] == "micro":
                assert e["t_off"] - e["t_on"] <= 0.5 + 1e-9, (s["id"], "a micro-expression is at most half a second")


def test_every_group_has_specimens_and_the_null_group_scripts_nothing():
    by_group = {}
    for s in LIB.SPECIMENS:
        by_group.setdefault(s["group"], []).append(s)
    assert set(by_group) == {g[0] for g in LIB.GROUPS}
    assert all(not s["events"] for s in by_group["null"])
    assert all(s["events"] for g, v in by_group.items() if g not in ("null", "nuisance") for s in v)


def test_confusable_pairs_are_mutual_where_they_are_declared_between_two_specimens():
    by = {s["id"]: s for s in LIB.SPECIMENS}
    for s in LIB.SPECIMENS:
        for c in s.get("confusable", []):
            assert by[c]["group"] in ("categories", "ambiguous", "macro", "ladder", "first", "nuisance") and c != s["id"]


def test_scenario_for_scales_amplitude_and_validates_names():
    spec = LIB.get("smile-flicker")
    base = LIB.scenario_for(spec).events[0].amp
    assert LIB.scenario_for(spec, amp_scale=2.0).events[0].amp == pytest.approx(2 * base) and base == pytest.approx(0.016)
    with pytest.raises(ValueError, match="camera"):
        LIB.scenario_for(spec, camera="nokia3310")
    with pytest.raises(ValueError, match="nuisance"):
        LIB.scenario_for(spec, nuisance="earthquake")
    assert LIB.scenario_for(spec, camera="phone30").fps == 30.0 and LIB.scenario_for(spec, camera="lab200").fps == 200.0


def test_the_same_seed_gives_the_same_schedule_and_a_different_one_differs():
    a = SC.nuisance_schedule(np.random.default_rng(3), 8.0, "normal")
    b = SC.nuisance_schedule(np.random.default_rng(3), 8.0, "normal")
    c = SC.nuisance_schedule(np.random.default_rng(4), 8.0, "normal")
    assert a["blinks"] == b["blinks"] and a["blinks"] != c["blinks"]
    still = SC.nuisance_schedule(np.random.default_rng(3), 8.0, "still")
    assert still["blinks"] == [] and still["gaze"] == [] and still["speech"] == []
    assert SC.nuisance_schedule(np.random.default_rng(3), 8.0, "busy")["speech"]            # a busy clip has someone talking in it


def test_keep_clear_removes_blinks_next_to_a_scripted_movement_and_only_those():
    ev = Event({12: 1.0}, 3.0, 3.15, 3.4, amp=0.02)
    inside = [(2.9, 0.2), (3.5, 0.2), (3.6, 0.2)]                  # touching / within 0.45 s of the movement
    outside = [(0.5, 0.2), (6.0, 0.2)]
    kept = SC._clear_of_events(inside + outside, [ev], 0.45)
    assert kept == outside
    sc = SC.make_scenario("A", "webcam30", 8.0, [ev], seed=2, nuisance="normal", nuisance_overrides=dict(blinks=inside + outside), keep_clear=0.0)
    assert len(sc.blinks) == 5                                       # without keep_clear nothing is dropped (the calibration corpus relies on this)


def test_null_scenarios_script_no_movement():
    sc = SC.make_null("C", "webcam30", 6.0, 7)
    assert sc.events == [] and sc.duration == 6.0


# ------------------------------------------------------------------------------------------------ truth in the instrument's units
def _stub_clip():
    truth = dict(events=[dict(aus=[6, 12], t_on=1.0, t_apex=1.2, t_off=1.5, amp_iod=0.02, label="positive", kind="micro", side=0.0)],
                 blinks=[dict(t_on=2.0, t_off=2.2)], speech=[], gaze=[dict(t=3.0, dx=0.004)])
    return SimpleNamespace(truth=truth, iod_px=100.0, fps=30.0, times=np.arange(0, 4, 1 / 30), scenario=SimpleNamespace(name="stub"))


def test_truth_json_in_generator_units_says_so():
    t = LIB.truth_json(_stub_clip())
    e = t["events"][0]
    assert e["units"] == "generator" and e["amp_pct_iod"] == 2.0 and e["amp_px"] == 2.0 and e["generator_pct_iod"] == 2.0
    assert t["blinks"] == [dict(t_on=2.0, t_off=2.2)] and t["iod_px"] == 100.0


def test_truth_json_with_exact_events_uses_the_instruments_unit_and_keeps_the_generator_value():
    exact = [dict(size_pct=1.7, a_true_pct={6: 1.1, 12: 1.9}, fit_r2=0.87, pair_gain=1.02)]
    t = LIB.truth_json(_stub_clip(), exact=exact)
    e = t["events"][0]
    assert e["units"] == "instrument" and e["amp_pct_iod"] == 1.7 and e["size_pct_iod"] == 1.7 and e["generator_pct_iod"] == 2.0
    assert e["au_pct_iod"] == {"6": 1.1, "12": 1.9} and e["au_px"] == {"6": 1.1, "12": 1.9} and e["fit_r2"] == 0.87
    json.dumps(t)                                                    # it is what gets written next to every specimen


def test_describe_truth_reads_in_plain_language():
    exact = [dict(size_pct=1.7, a_true_pct={6: 1.1, 12: 1.9}, fit_r2=0.87, pair_gain=1.02)]
    lines = LIB.describe_truth(LIB.truth_json(_stub_clip(), exact=exact))
    assert "AU6 cheek raiser + AU12 lip corner puller" in lines[0] and "at its largest region" in lines[0] and "500 ms" in lines[0]
    assert "AU12 1.90 %" in lines[0] and any("blink" in l for l in lines) and any("gaze" in l for l in lines)
    empty = SimpleNamespace(truth=dict(events=[], blinks=[], speech=[], gaze=[]), iod_px=100, fps=30, times=np.arange(0, 4, 1 / 30),
                            scenario=SimpleNamespace(name="x"))
    assert LIB.describe_truth(LIB.truth_json(empty)) == ["No expression was scripted."]


# ------------------------------------------------------------------------------------------------ geometry
def test_two_point_similarity_maps_its_two_points_exactly():
    m = G.similarity_from_two_points((10, 20), (50, 20), (0, 0), (80, 0))
    out = G.apply_affine(m, np.array([[10, 20], [50, 20]], float))
    assert np.allclose(out, [[0, 0], [80, 0]]) and G.similarity_params(m)[3] == pytest.approx(2.0)


def test_umeyama_recovers_a_similarity_from_noisy_points_and_never_reflects():
    rng = np.random.default_rng(0)
    src = rng.normal(0, 50, (60, 2))
    th, s, t = np.radians(17), 1.3, np.array([12.0, -7.0])
    R = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = src @ R.T + t + rng.normal(0, 0.05, src.shape)
    m = G.umeyama(src, dst)
    tx, ty, rot, scale = G.similarity_params(m)
    assert rot == pytest.approx(th, abs=0.003) and scale == pytest.approx(s, abs=0.003) and np.allclose([tx, ty], t, atol=0.1)
    assert np.linalg.det(m[:, :2]) > 0
    mirrored = src * np.array([-1, 1])
    assert np.linalg.det(G.umeyama(src, mirrored)[:, :2]) > 0                    # a mirror image is fitted as the best rotation, not a reflection


def test_weighted_umeyama_follows_the_weights():
    src = np.array([[0, 0], [100, 0], [0, 100], [100, 100], [50, 50]], float)
    dst = src + np.array([5.0, 0.0])
    dst[4] += [40, 40]                                                              # one wild point
    m_even = G.umeyama(src, dst)
    m_w = G.umeyama(src, dst, weights=[1, 1, 1, 1, 1e-6])
    assert np.allclose(G.similarity_params(m_w)[:2], [5, 0], atol=0.01)
    assert np.hypot(*(G.similarity_params(m_even)[:2] - np.array([5, 0]))) > 3


def test_canonical_frame_puts_the_eyes_level_and_symmetric():
    c = G.Canon(size=384, iod=96)
    a, b = c.eye_targets()
    assert a[1] == b[1] == c.eye_y == pytest.approx(0.41 * 384) and b[0] - a[0] == 96 and (a[0] + b[0]) / 2 == 192


def test_interocular_distance_from_the_eye_rings():
    xy = np.zeros((G.N_LM, 2))
    xy[G.EYE_L_RING] = [100, 100]
    xy[G.EYE_R_RING] = [220, 100]
    assert G.interocular(xy) == pytest.approx(120.0)
    la, ra = G.eye_centres(xy)
    assert np.allclose(la, [100, 100]) and np.allclose(ra, [220, 100])


def test_the_roi_table_is_complete_and_symmetric():
    assert len(G.ROI_NAMES) == 27 and len(set(G.ROI_NAMES)) == 27
    for n in G.ROI_NAMES:
        if n.endswith("_L"):
            assert n[:-2] + "_R" in G.ROI_NAMES
    assert set(G.GROUP_OF) == set(G.ROI_NAMES)


# ------------------------------------------------------------------------------------------------ optical flow
def _texture(n=160, seed=2):
    return cv2.GaussianBlur((np.random.default_rng(seed).random((n, n)) * 255).astype(np.uint8), (0, 0), 2.5)


def _shift(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, img.shape[::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


@pytest.mark.parametrize("dx,dy", [(0.4, -0.3), (1.2, 0.8), (-2.5, 1.5)])
def test_lucas_kanade_measures_a_known_translation(dx, dy):
    a = _texture()
    u = FL.LKFlow()(a, _shift(a, dx, dy))
    core = u[40:-40, 40:-40]
    assert core[..., 0].mean() == pytest.approx(dx, abs=0.08) and core[..., 1].mean() == pytest.approx(dy, abs=0.08)
    assert core.std(axis=(0, 1)).max() < 0.15                                      # uniform motion gives a uniform field


def test_identical_images_have_no_flow():
    a = _texture()
    assert np.abs(FL.LKFlow()(a, a.copy())).max() < 0.01


def test_confidence_is_higher_on_texture_than_on_flat_skin():
    img = np.full((160, 160), 128, np.uint8)
    img[:, 80:] = _texture()[:, :80]                                               # left half flat, right half textured
    _, conf = FL.LKFlow()(img, _shift(img, 0.5, 0), return_confidence=True)
    assert conf[:, 100:150].mean() > 20 * max(conf[:, 10:60].mean(), 1e-6)


def test_the_other_flow_backends_see_a_translation_too():
    a = _texture()
    b = _shift(a, 1.5, 0.0)
    for f in (FL.DISFlow(), FL.FarnebackFlow()):
        u = f(a, b)
        assert u[40:-40, 40:-40, 0].mean() == pytest.approx(1.5, abs=0.3)


def test_similarity_fit_removes_head_motion_and_leaves_local_motion():
    h = w = 120
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32) - 60, np.arange(h, dtype=np.float32) - 60)
    flow = np.zeros((h, w, 2), np.float32)
    a, b = 0.01, 0.02                                                              # scale and rotation
    flow[..., 0] = 1.5 + a * gx - b * gy
    flow[..., 1] = -0.7 + b * gx + a * gy
    weights = np.zeros((h, w), np.float32)
    weights[:30] = 1.0                                                              # the stable region: the forehead
    local = flow.copy()
    local[80:100, 50:70] += [0.0, 3.0]                                              # a mouth-area movement, outside the weighted region
    model = FL.fit_similarity_flow(local, weights, (60, 60))
    rest = local - model
    assert np.abs(rest[:20]).max() < 0.05                                           # rigid part gone where it was fitted
    assert rest[85:95, 55:65, 1].mean() == pytest.approx(3.0, abs=0.1)              # the local movement survives intact
    assert np.all(FL.fit_similarity_flow(flow, np.zeros((h, w), np.float32), (60, 60)) == 0)


# ------------------------------------------------------------------------------------------------ frame sources
def test_array_source_reports_the_measured_rate_and_jitter():
    frames = np.zeros((31, 8, 8, 3), np.uint8)
    s = SRC.ArraySource(frames, np.arange(31) / 30.0)
    assert s.fps == pytest.approx(30.0) and s.fps_jitter == pytest.approx(0.0, abs=1e-9) and s.duration == pytest.approx(1.0)
    t = np.arange(31) / 30.0
    t[10:] += 0.02
    j = SRC.ArraySource(frames, t)
    assert j.fps_jitter > 0.05 and j.info()["max_gap_ms"] == pytest.approx(53.33, abs=0.1)
    assert len(list(s.frames())) == 31


def _write_jpegs(folder, n, times=None):
    for i in range(n):
        im = np.full((32, 48, 3), i, np.uint8)
        cv2.imwrite(os.path.join(folder, f"{i:06d}.jpg"), im)
    if times is not None:
        json.dump(list(times), open(os.path.join(folder, "times.json"), "w"))


def test_jpeg_folder_uses_the_recorded_timestamps_and_zero_bases_them(tmp_path):
    _write_jpegs(str(tmp_path), 20, times=[5.0 + i * 0.04 for i in range(20)])
    s = SRC.JpegDirSource(str(tmp_path))
    assert s.n == 20 and s.times[0] == 0.0 and s.fps == pytest.approx(25.0) and (s.width, s.height) == (48, 32)


def test_jpeg_folder_copes_with_missing_timestamps(tmp_path):
    _write_jpegs(str(tmp_path), 20, times=[i * 0.04 for i in range(12)])           # 8 frames lack a time
    s = SRC.JpegDirSource(str(tmp_path))
    assert s.n == 20 and len(s.times) == 20 and np.all(np.diff(s.times) > 0) and s.fps == pytest.approx(25.0)
    other = tmp_path / "b"
    other.mkdir()
    _write_jpegs(str(other), 10)
    assert SRC.JpegDirSource(str(other)).fps == pytest.approx(30.0)               # no time file at all: the nominal 30 fps is assumed
    with pytest.raises(ValueError, match="no frames"):
        SRC.JpegDirSource(str(tmp_path / "empty_nope"))


def test_video_source_reads_frames_and_trims_to_the_maximum_length(tmp_path):
    path = str(tmp_path / "v.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    for i in range(90):
        w.write(np.full((48, 64, 3), i * 2, np.uint8))
    w.release()
    s = SRC.open_source(path)
    assert s.n == 90 and s.fps == pytest.approx(30.0, abs=0.5) and (s.width, s.height) == (64, 48)
    assert sum(1 for _ in s.frames()) == 90
    t = SRC.VideoSource(path, max_seconds=1.0)
    assert 28 <= t.n <= 31 and sum(1 for _ in t.frames()) == t.n
    big = SRC.VideoSource(path, max_side=32)
    assert next(iter(big.frames())).shape[:2] == (24, 32)
    with pytest.raises(ValueError, match="cannot open"):
        SRC.VideoSource(str(tmp_path / "missing.avi"))


def test_open_source_dispatches_on_folders(tmp_path):
    _write_jpegs(str(tmp_path), 12, times=[i / 30 for i in range(12)])
    assert isinstance(SRC.open_source(str(tmp_path)), SRC.JpegDirSource)
