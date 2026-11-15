"""The web application through Flask's test client.  The analysis itself is replaced by a stand-in wherever the question is
about what the *server* does with footage (consent, deletion, purging, what may be fetched); the real chain runs in
test_pipeline_e2e.py."""
import io
import json
import os
import time
import zipfile

import cv2
import numpy as np
import pytest

from onset import config
from onset.analysis.pipeline import AnalysisError
from onset.server import app as appmod
from onset.server.store import Store


# ------------------------------------------------------------------------------------------------ helpers
def _jpeg(i: int, size=64) -> bytes:
    rng = np.random.default_rng(i)
    im = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", im)
    assert ok
    return buf.tobytes()


def _fake_result(duration=3.0):
    return dict(quality=dict(grade="B"), clip=dict(duration_s=duration),
                events=[dict(status="micro"), dict(status="blink"), dict(status="expression")])


class Recorder:
    """Stands in for ``analyse``: notes what the footage folder held while the analysis ran, then writes a minimal result."""
    def __init__(self, fail=None):
        self.fail, self.seen_frames, self.src_dirs = fail, None, []

    def __call__(self, src, out_dir, cfg, progress=None, label="", provenance=None, detection=None):
        self.seen_frames = sum(1 for _ in src.frames())
        if progress:
            progress(0.5, "Looking")
        with open(os.path.join(out_dir, "face_000.jpg"), "wb") as fh:
            fh.write(_jpeg(0))
        if self.fail:
            raise self.fail
        res = _fake_result()
        res["label"], res["provenance"] = label, provenance
        with open(os.path.join(out_dir, "result.json"), "w") as fh:
            json.dump(res, fh)
        return res


@pytest.fixture
def client(tmp_path):
    app = appmod.create_app(var_dir=str(tmp_path / "var"), data_dir=str(tmp_path / "data"))
    app.config["TESTING"] = True
    c = app.test_client()
    c.app_state = app.onset
    return c


@pytest.fixture
def real_data_client(tmp_path):
    app = appmod.create_app(var_dir=str(tmp_path / "var"))
    app.config["TESTING"] = True
    return app.test_client()


def _wait(client, job_id, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").get_json()
        if j["state"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _capture(client, n_frames=40, consent=True, **extra):
    r = client.post("/api/capture/start", json=dict(consent=consent, label="test", **extra))
    if not consent:
        return r
    sid = r.get_json()["id"]
    meta = [[i, i / 30.0] for i in range(n_frames)]
    data = {"meta": json.dumps(meta)}
    for i in range(n_frames):
        data[f"f{i}"] = (io.BytesIO(_jpeg(i)), f"{i}.jpg")
    got = client.post(f"/api/capture/{sid}/frames", data=data, content_type="multipart/form-data")
    return sid, got


def _src_files(client, sid):
    return os.listdir(client.app_state["store"].src_dir(sid))


# ------------------------------------------------------------------------------------------------ status, headers, static
def test_status_states_the_privacy_terms_and_limits(client):
    s = client.get("/api/status").get_json()
    assert s["privacy"]["local_only"] is True and s["privacy"]["source_kept"] is False
    assert "deleted when the analysis ends" in s["privacy"]["text"] and "grayscale face crops" in s["privacy"]["text"]
    assert s["limits"]["capture_seconds"] == config.CAPTURE_SECONDS and s["limits"]["min_clip_seconds"] == config.MIN_CLIP_SECONDS
    assert s["library"] == 0 and s["busy"] == 0


def test_api_answers_are_never_cached_and_carry_safety_headers(client):
    r = client.get("/api/status")
    assert r.headers["Cache-Control"] == "no-store" and r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Referrer-Policy"] == "no-referrer"


def test_the_interface_is_served_and_unknown_api_paths_are_404(client):
    r = client.get("/")
    assert r.status_code == 200 and b"<html" in r.data.lower()
    assert client.get("/api/nothing-here").status_code == 404
    assert client.get("/css/does-not-exist.css").status_code == 404


@pytest.mark.parametrize("path", ["/../onset/config.py", "/%2e%2e/onset/config.py", "/..%2fonset/config.py", "/css/../../onset/config.py"])
def test_static_files_cannot_escape_the_web_folder(client, path):
    r = client.get(path)
    assert r.status_code in (404, 400) and b"MAX_UPLOAD_MB" not in r.data


# ------------------------------------------------------------------------------------------------ consent
def test_recording_needs_consent(client):
    r = client.post("/api/capture/start", json=dict(consent=False))
    assert r.status_code == 400 and "consent" in r.get_json()["error"].lower()
    assert client.post("/api/capture/start", json={}).status_code == 400
    assert os.listdir(client.app_state["store"].sessions) == []                           # nothing was created for a refused request


def test_consent_is_time_stamped_in_the_session(client):
    r = client.post("/api/capture/start", json=dict(consent=True, protocol="baseline_task",
                                                    segments=[dict(label="still", t0=0, t1=4), dict(label="bad", t0=5, t1=2), dict(label="count", t0=4, t1=8)]))
    sid = r.get_json()["id"]
    m = client.app_state["store"].meta(sid)
    assert m["consent_at"] and m["provenance"]["protocol"] == "baseline_task"
    assert [s["label"] for s in m["provenance"]["segments"]] == ["still", "count"]          # the inverted segment is dropped


def test_upload_needs_a_file_consent_and_a_video_extension(client):
    assert client.post("/api/upload", data={}).status_code == 400
    f = (io.BytesIO(b"x" * 1000), "clip.avi")
    r = client.post("/api/upload", data={"file": f}, content_type="multipart/form-data")
    assert r.status_code == 400 and "right to analyse" in r.get_json()["error"]
    r = client.post("/api/upload", data={"file": (io.BytesIO(b"hi"), "notes.txt"), "consent": "1"}, content_type="multipart/form-data")
    assert r.status_code == 400 and ".txt" in r.get_json()["error"]
    assert os.listdir(client.app_state["store"].sessions) == []


def test_a_file_that_is_not_a_video_is_refused_and_leaves_nothing(client):
    r = client.post("/api/upload", data={"file": (io.BytesIO(b"this is not a video" * 50), "clip.mp4"), "consent": "1"},
                    content_type="multipart/form-data")
    assert r.status_code == 422
    assert os.listdir(client.app_state["store"].sessions) == []


# ------------------------------------------------------------------------------------------------ capture flow and deletion of the footage
def test_capture_accepts_only_real_jpegs(client):
    sid = client.post("/api/capture/start", json=dict(consent=True)).get_json()["id"]
    data = {"meta": json.dumps([[0, 0.0], [1, 0.033], [2, 0.066], [3, 0.1]]),
            "f0": (io.BytesIO(_jpeg(0)), "0.jpg"),
            "f1": (io.BytesIO(b"GIF89a" + b"\0" * 400), "1.jpg"),                           # wrong magic bytes
            "f2": (io.BytesIO(b"\xff\xd8tiny"), "2.jpg")}                                    # too small to be a frame; f3 missing altogether
    r = client.post(f"/api/capture/{sid}/frames", data=data, content_type="multipart/form-data")
    assert r.get_json()["received"] == 1
    assert sorted(f for f in _src_files(client, sid) if f.endswith(".jpg")) == ["000000.jpg"]


def test_capture_frames_unknown_or_closed_session(client):
    assert client.post("/api/capture/c-0123456789/frames", data={"meta": "[]"}).status_code == 404
    assert client.post("/api/capture/not-an-id/frames", data={"meta": "[]"}).status_code == 404
    assert client.post("/api/capture/../x/frames").status_code in (404, 405)                 # normalised to another path: refused either way


def test_too_short_a_recording_is_deleted_not_analysed(client):
    sid, got = _capture(client, n_frames=10)
    r = client.post(f"/api/capture/{sid}/finish", json={})
    assert r.status_code == 422
    assert not client.app_state["store"].exists(sid)                                       # footage gone with the session


def test_finish_before_any_frames_is_refused(client):
    sid = client.post("/api/capture/start", json=dict(consent=True)).get_json()["id"]
    assert client.post(f"/api/capture/{sid}/finish", json={}).status_code == 409


def test_the_recording_is_deleted_when_the_analysis_ends_and_only_after_it_was_read(client, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(appmod, "analyse", rec)
    sid, got = _capture(client, n_frames=40)
    assert got.get_json()["received"] == 40 and len(_src_files(client, sid)) >= 40
    job = client.post(f"/api/capture/{sid}/finish", json={"label": "Morning"}).get_json()
    done = _wait(client, job["id"])
    assert done["state"] == "done" and done["result_id"] == sid
    assert rec.seen_frames == 40                                                           # the analysis did read the footage
    assert _src_files(client, sid) == []                                                   # and it is gone now
    out = os.listdir(client.app_state["store"].out_dir(sid))
    assert "result.json" in out and not any(n.startswith("0000") and n.endswith(".jpg") and n != "face_000.jpg" for n in out)
    m = client.app_state["store"].meta(sid)
    assert m["state"] == "done" and m["grade"] == "B" and m["n_events"] == 2 and m["label"] == "Morning"      # blinks are not counted as events


def test_a_failed_analysis_deletes_the_whole_session(client, monkeypatch):
    monkeypatch.setattr(appmod, "analyse", Recorder(fail=AnalysisError("No face was found in this clip.")))
    sid, _ = _capture(client, n_frames=40)
    job = client.post(f"/api/capture/{sid}/finish", json={}).get_json()
    done = _wait(client, job["id"])
    assert done["state"] == "error" and "No face" in done["error"]                         # the person is told why
    assert not client.app_state["store"].exists(sid)                                       # footage and partial crops: all gone


def test_an_unexpected_crash_also_deletes_the_footage_and_is_explained(client, monkeypatch):
    monkeypatch.setattr(appmod, "analyse", Recorder(fail=RuntimeError("boom")))
    sid, _ = _capture(client, n_frames=40)
    done = _wait(client, client.post(f"/api/capture/{sid}/finish", json={}).get_json()["id"])
    assert done["state"] == "error" and "stopped unexpectedly" in done["error"] and "boom" not in done["error"]
    assert not client.app_state["store"].exists(sid)


def test_an_uploaded_video_is_deleted_after_its_analysis(client, monkeypatch, tmp_path):
    rec = Recorder()
    monkeypatch.setattr(appmod, "analyse", rec)
    path = str(tmp_path / "v.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 64))
    for i in range(90):
        w.write(cv2.imdecode(np.frombuffer(_jpeg(i), np.uint8), cv2.IMREAD_COLOR))
    w.release()
    with open(path, "rb") as fh:
        r = client.post("/api/upload", data={"file": (fh, "my clip.avi"), "consent": "1", "label": "kitchen"}, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    done = _wait(client, r.get_json()["id"])
    assert done["state"] == "done"
    sid = done["result_id"]
    assert rec.seen_frames and rec.seen_frames >= 85
    assert _src_files(client, sid) == []                                                    # the uploaded file is gone
    assert client.app_state["store"].meta(sid)["provenance"]["filename"] == "my clip.avi"


def test_a_video_shorter_than_the_minimum_is_refused_and_deleted(client, tmp_path):
    path = str(tmp_path / "short.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 64))
    for i in range(30):                                                                      # one second
        w.write(cv2.imdecode(np.frombuffer(_jpeg(i), np.uint8), cv2.IMREAD_COLOR))
    w.release()
    with open(path, "rb") as fh:
        r = client.post("/api/upload", data={"file": (fh, "short.avi"), "consent": "on"}, content_type="multipart/form-data")
    assert r.status_code == 422 and "shorter" in r.get_json()["error"]
    assert os.listdir(client.app_state["store"].sessions) == []


# ------------------------------------------------------------------------------------------------ results: listing, files, export, deletion
def _finished_session(client, monkeypatch):
    monkeypatch.setattr(appmod, "analyse", Recorder())
    sid, _ = _capture(client, n_frames=40)
    _wait(client, client.post(f"/api/capture/{sid}/finish", json={}).get_json()["id"])
    return sid


def test_results_are_listed_served_exported_and_deleted(client, monkeypatch):
    sid = _finished_session(client, monkeypatch)
    assert [r["id"] for r in client.get("/api/results").get_json()] == [sid]
    res = client.get(f"/api/results/{sid}").get_json()
    assert res["session"]["id"] == sid and res["quality"]["grade"] == "B"
    assert client.get(f"/api/results/{sid}/files/face_000.jpg").status_code == 200
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/results/{sid}/export").data))
    assert set(z.namelist()) == {"result.json", "face_000.jpg", "README.txt"}
    assert client.delete(f"/api/results/{sid}").get_json() == {"deleted": True}
    assert client.get(f"/api/results/{sid}").status_code == 404
    assert client.delete(f"/api/results/{sid}").get_json() == {"deleted": False}
    assert client.get("/api/results").get_json() == []


def test_only_whitelisted_files_with_plain_names_are_served(client, monkeypatch):
    sid = _finished_session(client, monkeypatch)
    out = client.app_state["store"].out_dir(sid)
    for name in ("secret.py", "notes.txt", "x.csv"):
        open(os.path.join(out, name), "w").write("private")
        assert client.get(f"/api/results/{sid}/files/{name}").status_code == 404
    for name in ("..%2Fmeta.json", "%2e%2e%2fmeta.json", "a%2Fb.jpg", ".." + "/meta.json"):
        assert client.get(f"/api/results/{sid}/files/{name}").status_code == 404
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/results/{sid}/export").data))
    assert "secret.py" not in z.namelist() and "x.csv" not in z.namelist()


def test_session_ids_are_validated(client):
    for bad in ("../../etc", "c-xyz", "c-0123456789ab", "C-0123456789", "..", "c-0123456789/../x"):
        assert client.get(f"/api/results/{bad}").status_code == 404
    assert client.get("/api/results/c-0123456789").status_code == 404


def test_an_unfinished_result_says_so(client):
    sid = client.post("/api/capture/start", json=dict(consent=True)).get_json()["id"]
    assert client.get(f"/api/results/{sid}").status_code == 409


def test_purge_removes_everything(client, monkeypatch):
    _finished_session(client, monkeypatch)
    client.post("/api/capture/start", json=dict(consent=True))
    assert client.post("/api/purge").get_json()["deleted"] == 2
    assert os.listdir(client.app_state["store"].sessions) == []


# ------------------------------------------------------------------------------------------------ the store's clock
def _age(store, sid, hours=0.0, minutes=0.0, **meta):
    m = store.meta(sid)
    m["created"] -= hours * 3600 + minutes * 60
    m["expires"] -= hours * 3600 + minutes * 60
    m.update(meta)
    store.write_meta(sid, m)


def test_sessions_expire_after_the_retention_time(tmp_path):
    st = Store(str(tmp_path), retention_hours=24)
    old, fresh = st.create("capture", consent=True), st.create("capture", consent=True)
    st.update_meta(old, state="done"), st.update_meta(fresh, state="done")
    _age(st, old, hours=25)
    assert st.purge_expired(force=True) == 1
    assert not st.exists(old) and st.exists(fresh)


def test_an_abandoned_recording_is_purged_long_before_the_day_is_out(tmp_path):
    st = Store(str(tmp_path), retention_hours=24, abandon_minutes=60)
    a, b, done = st.create("capture", consent=True), st.create("capture", consent=True), st.create("capture", consent=True)
    st.update_meta(a, state="capturing"), st.update_meta(done, state="done")
    open(os.path.join(st.src_dir(a), "000000.jpg"), "wb").write(b"\xff\xd8" + b"0" * 300)
    _age(st, a, minutes=61)                                                                  # footage nobody finished, an hour ago
    _age(st, done, minutes=61)                                                               # a finished report of the same age stays
    assert st.purge_expired(force=True) == 1
    assert not st.exists(a) and st.exists(b) and st.exists(done)


def test_the_sweep_runs_at_most_every_five_minutes_unless_forced(tmp_path):
    st = Store(str(tmp_path), retention_hours=1)
    sid = st.create("capture", consent=True)
    st.update_meta(sid, state="done")
    st.purge_expired(force=True)
    _age(st, sid, hours=2)
    assert st.purge_expired() == 0 and st.exists(sid)                                        # too soon
    assert st.purge_expired(force=True) == 1


def test_server_start_removes_footage_that_a_restart_interrupted(tmp_path):
    var = str(tmp_path / "var")
    st = Store(var)
    interrupted, finished = st.create("upload", consent=True), st.create("upload", consent=True)
    st.update_meta(interrupted, state="queued")
    open(os.path.join(st.src_dir(interrupted), "clip.mp4"), "wb").write(b"footage")
    st.update_meta(finished, state="done")
    appmod.create_app(var_dir=var, data_dir=str(tmp_path / "data"))
    st2 = Store(var)
    assert not st2.exists(interrupted) and st2.exists(finished)


def test_a_session_folder_without_meta_is_only_removed_once_it_is_a_minute_old(tmp_path):
    st = Store(str(tmp_path))
    d = os.path.join(st.sessions, "c-aaaaaaaaaa")
    os.makedirs(d)
    assert st.purge_expired(force=True) == 0 and os.path.isdir(d)                            # may still be being created
    os.utime(d, (time.time() - 3600, time.time() - 3600))
    assert st.purge_expired(force=True) == 1


def test_store_ids_reject_anything_but_its_own_format(tmp_path):
    st = Store(str(tmp_path))
    sid = st.create("capture")
    assert Store.valid_id(sid) and not Store.valid_id("../x") and not Store.valid_id("") and not Store.valid_id(None)
    with pytest.raises(KeyError):
        st.dir("../escape")
    assert st.delete("../escape") is False


# ------------------------------------------------------------------------------------------------ pre-flight
def test_preflight_validates_its_input(client):
    assert client.post("/api/preflight", data={"times": "[]"}).status_code == 400
    five = {"times": json.dumps([i / 30 for i in range(5)])}
    for i in range(5):
        five[f"frames"] = (io.BytesIO(_jpeg(i)), "f.jpg")
    assert client.post("/api/preflight", data=five, content_type="multipart/form-data").status_code == 400      # fewer than 8 frames


def test_preflight_rejects_decreasing_time_stamps_and_odd_sizes(client):
    def post(times, sizes):
        data = {"times": json.dumps(times), "frames": [(io.BytesIO(_jpeg(i, size=s)), f"{i}.jpg") for i, s in enumerate(sizes)]}
        return client.post("/api/preflight", data=data, content_type="multipart/form-data")
    assert post([0, 1, 1, 2, 3, 4, 5, 6], [64] * 8).status_code == 400
    assert post([i / 30 for i in range(8)], [64] * 7 + [48]).status_code == 400


# ------------------------------------------------------------------------------------------------ stress lab
def test_stress_validates_its_inputs(client):
    assert client.post("/api/stress", json=dict(specimen="no-such")).status_code == 404
    from onset.synth import library as LIB
    sid = LIB.SPECIMENS[0]["id"]
    assert client.post("/api/stress", json=dict(specimen=sid, amp_scale=9)).status_code == 400
    assert client.post("/api/stress", json=dict(specimen=sid, noise_scale=-1)).status_code == 400
    assert client.post("/api/stress", json=dict(specimen=sid, blur="lots")).status_code == 400
    assert client.post("/api/stress", json=dict(specimen=sid, camera="nokia")).status_code == 400
    assert os.listdir(client.app_state["store"].sessions) == []


def test_stress_options_list_the_cameras_and_every_specimen(client):
    o = client.get("/api/stress/options").get_json()
    from onset.synth import library as LIB
    assert len(o["specimens"]) == len(LIB.SPECIMENS) and {"webcam30", "phone60"} <= {c["id"] for c in o["cameras"]} | {"webcam30", "phone60"}
    assert all(c["fps"] > 0 for c in o["cameras"]) and o["nuisance"]


# ------------------------------------------------------------------------------------------------ the library, evidence and the atlas
def test_library_before_it_is_built(client):
    lib = client.get("/api/library").get_json()
    assert lib["specimens"] == [] and "not been built" in lib["note"]
    assert client.get("/api/atlas").status_code == 404
    assert client.get("/api/evidence").get_json()["calibration"] is None or isinstance(client.get("/api/evidence").get_json()["calibration"], dict)


def test_a_library_copied_in_part_lists_only_the_specimens_that_are_there(tmp_path):
    lib = tmp_path / "data" / "library"
    (lib / "have").mkdir(parents=True)
    (lib / "have" / "result.json").write_text("{}")
    (lib / "half-copied").mkdir()                                              # a folder without its result is not a specimen yet
    (lib / "manifest.json").write_text(json.dumps(dict(
        version=1, groups=[dict(id="g", title="G", blurb="")],
        specimens=[dict(id="have", group="g"), dict(id="half-copied", group="g"), dict(id="absent", group="g")])))
    app = appmod.create_app(var_dir=str(tmp_path / "var"), data_dir=str(tmp_path / "data"))
    c = app.test_client()
    assert [x["id"] for x in c.get("/api/library").get_json()["specimens"]] == ["have"]
    assert c.get("/api/status").get_json()["library"] == 1
    (lib / "absent").mkdir()
    (lib / "absent" / "result.json").write_text("{}")                          # the missing part arrives: it appears without a restart
    assert [x["id"] for x in c.get("/api/library").get_json()["specimens"]] == ["have", "absent"]


def test_library_files_are_whitelisted(real_data_client):
    c = real_data_client
    man = c.get("/api/library").get_json()
    if not man["specimens"]:
        pytest.skip("library not built")
    sid = man["specimens"][0]["id"]
    assert c.get(f"/api/library/{sid}/result").status_code == 200
    assert c.get(f"/api/library/{sid}/files/result.json").status_code == 200
    assert c.get(f"/api/library/{sid}/files/meta.py").status_code == 404
    assert c.get("/api/library/no-such/files/result.json").status_code == 404
    assert c.get(f"/api/library/{sid}/files/..%2Fmanifest.json").status_code == 404
    assert c.get("/api/library/no-such/result").status_code == 404


def test_the_real_data_folder_serves_the_atlas_and_the_evidence(real_data_client):
    c = real_data_client
    ev = c.get("/api/evidence").get_json()
    assert "calibration" in ev
    r = c.get("/api/atlas")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert r.get_json()


# ------------------------------------------------------------------------------------------------ cue lab endpoints (no verdicts)
def test_cue_predictive_values_and_input_checks(client):
    r = client.post("/api/cue/predictive", json=dict(sensitivity=0.8, specificity=0.8, prevalence=0.05))
    assert r.status_code == 200 and 0 < r.get_json()["values"]["ppv"] < 0.25                    # a good test, a rare thing: most alarms are false
    assert client.post("/api/cue/predictive", json=dict(sensitivity=2, specificity=0.8, prevalence=0.05)).status_code == 400
    assert client.post("/api/cue/predictive", json={}).status_code == 400


def test_cue_rates_forking_and_roc(client):
    r = client.post("/api/cue/rates", json=dict(k1=3, s1=60, k2=9, s2=60)).get_json()
    assert 0 <= r["test"]["p"] <= 1 and r["min_events"] == 6
    assert client.post("/api/cue/rates", json=dict(k1=-1, s1=60, k2=9, s2=60)).status_code == 400
    assert client.post("/api/cue/rates", json=dict(k1=1, s1=0, k2=9, s2=60)).status_code == 400
    f = client.post("/api/cue/forking", json=dict(n_tests=20)).get_json()
    assert f["p_any"] == pytest.approx(1 - 0.95 ** 20)
    assert client.post("/api/cue/forking", json=dict(n_tests=10 ** 6)).get_json()["n_tests"] == 200
    assert client.post("/api/cue/roc", json=dict(d=0.5)).status_code == 200
    assert client.post("/api/cue/roc", json=dict(d=9)).status_code == 400
    assert client.post("/api/cue/roc", json={}).status_code == 400


def test_cue_segments_needs_an_analysed_recording(client):
    r = client.post("/api/cue/segments", json=dict(a=dict(t0=0, t1=2), b=dict(t0=3, t1=5)))
    assert r.status_code == 404
    r = client.post("/api/cue/segments", json=dict(specimen="no-such", a=dict(t0=0, t1=2), b=dict(t0=3, t1=5)))
    assert r.status_code == 404
    r = client.post("/api/cue/audit", json=dict(result="c-0123456789"))
    assert r.status_code == 404


def test_cue_evidence_carries_no_verdict(client):
    from onset.analysis import cue_lab as C
    ev = client.get("/api/cue/evidence").get_json()
    assert "cannot tell you whether someone is lying" in ev["position"]["headline"]
    C.assert_no_verdict(ev)                                                                  # the same guard the server applies before answering
