"""The job queue, and the choice of the quiet reference frame for measuring a spotted movement."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from onset.analysis import events as EV
from onset.analysis.pipeline import AnalysisError
from onset.server.jobs import JobManager


def _wait_for(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


# ------------------------------------------------------------------------------------------------ jobs
def test_a_job_runs_reports_progress_and_hands_back_its_result_id():
    jm = JobManager()
    seen, ready = [], threading.Event()

    def fn(progress):
        ready.wait(5)                                                         # the test knows the job's id only after submit() returns
        progress(0.4, "Looking at the face")
        seen.append(jm.get(job.id).stage)
        progress(7.0)                                                         # clamped
        seen.append(jm.get(job.id).progress)
        return "c-0123456789"
    job = jm.submit("analysis", fn)
    ready.set()
    assert _wait_for(lambda: jm.get(job.id).state == "done")
    j = jm.get(job.id)
    assert j.result_id == "c-0123456789" and j.progress == 1.0 and j.stage == "Done" and j.error is None
    assert seen == ["Looking at the face", 1.0]
    d = j.to_dict()
    assert d["state"] == "done" and d["elapsed_s"] >= 0 and d["kind"] == "analysis"


def test_an_analysis_error_reaches_the_person_unchanged():
    jm = JobManager()

    def fn(progress):
        raise AnalysisError("No face was found in this clip. Face the camera and add light from the front.")
    job = jm.submit("analysis", fn)
    assert _wait_for(lambda: jm.get(job.id).state == "error")
    assert jm.get(job.id).error.startswith("No face was found") and jm.get(job.id).result_id is None


def test_an_unexpected_error_is_explained_without_leaking_its_text():
    jm = JobManager()

    def fn(progress):
        raise ValueError("/home/someone/private/path.mp4 exploded")
    job = jm.submit("analysis", fn)
    assert _wait_for(lambda: jm.get(job.id).state == "error")
    msg = jm.get(job.id).error
    assert "stopped unexpectedly" in msg and "ValueError" in msg and "private" not in msg


def test_one_worker_runs_jobs_one_at_a_time_in_order_and_says_who_is_waiting():
    jm = JobManager(workers=1)
    gate, order = threading.Event(), []

    def slow(progress):
        gate.wait(5)
        order.append("first")
        return "a"

    def quick(progress):
        order.append("second")
        return "b"
    j1 = jm.submit("analysis", slow)
    assert _wait_for(lambda: jm.get(j1.id).state == "running")
    j2 = jm.submit("analysis", quick)
    j3 = jm.submit("analysis", quick)
    assert jm.get(j2.id).state == "queued" and jm.get(j2.id).queue_position == 1 and jm.get(j3.id).queue_position == 2
    assert jm.busy() == 3
    gate.set()
    assert _wait_for(lambda: jm.get(j3.id).state == "done")
    assert order == ["first", "second", "second"] and jm.busy() == 0


def test_the_queue_has_a_limit_and_refuses_politely_beyond_it():
    jm = JobManager(workers=1, max_queue=2)
    gate = threading.Event()
    jm.submit("analysis", lambda p: gate.wait(5) and "x")
    jm.submit("analysis", lambda p: "y")
    with pytest.raises(RuntimeError, match="busy"):
        jm.submit("analysis", lambda p: "z")
    gate.set()
    assert _wait_for(lambda: jm.busy() == 0)
    jm.submit("analysis", lambda p: "again")                                  # room again once the queue has drained


def test_the_job_table_is_trimmed_of_old_finished_jobs():
    jm = JobManager(workers=1, max_queue=100)
    ids = []
    for _ in range(70):
        j = jm.submit("analysis", lambda p: "r")
        ids.append(j.id)
        assert _wait_for(lambda j=j: jm.get(j.id).state == "done")
    assert len(jm.jobs) <= 62 and jm.get(ids[-1]) is not None and jm.get(ids[0]) is None


# ------------------------------------------------------------------------------------------------ the quiet reference frame
FPS = 30.0


def _spot(n, loud=(), level=0.1, loud_level=5.0):
    e = np.full(n, level)
    for a, b in loud:
        e[a:b] = loud_level
    return SimpleNamespace(frame_energy=e)


def _cues(n, blink=(), speech=()):
    b, s = np.zeros(n), np.zeros(n)
    for a, z in blink:
        b[a:z] = 1.0
    for a, z in speech:
        s[a:z] = 1.0
    return SimpleNamespace(blink=b, speech=s)


def test_the_reference_is_the_quiet_frame_closest_to_the_event_before_it():
    n = 300
    ref, side = EV.choose_reference(_spot(n, loud=[(148, 153)]), 150, FPS, n)
    assert side == "before" and 150 - 36 <= ref <= 150 - 6                       # 0.2-1.2 s before the peak
    assert ref >= 143                                                            # and as close as quiet allows


def test_the_reference_steps_back_past_a_blink_and_past_mouth_movement():
    n = 300
    quiet = EV.choose_reference(_spot(n), 150, FPS, n, _cues(n), np.arange(n) / FPS)[0]
    blinked = EV.choose_reference(_spot(n), 150, FPS, n, _cues(n, blink=[(136, 147)]), np.arange(n) / FPS)[0]
    talking = EV.choose_reference(_spot(n), 150, FPS, n, _cues(n, speech=[(120, 147)]), np.arange(n) / FPS)[0]
    assert blinked < 136 - 2 < quiet and talking < 120 - 2                       # clear of the nuisance, widened by 0.1 s on each side


def test_the_reference_skips_earlier_movement_that_is_not_quiet():
    n = 300
    ref, side = EV.choose_reference(_spot(n, loud=[(100, 140), (148, 153)]), 150, FPS, n)
    assert side == "before" and not (100 <= ref < 140)


def test_near_the_start_of_the_clip_the_baseline_is_taken_after_the_event():
    n = 300
    ref, side = EV.choose_reference(_spot(n), 5, FPS, n)
    assert side == "after" and 5 + 13 <= ref <= 5 + 42


def test_with_no_room_at_all_the_function_says_so():
    ref, side = EV.choose_reference(_spot(8), 3, FPS, 8)
    assert (ref, side) == (0, "none")


def test_when_nowhere_is_quiet_the_least_noisy_frame_is_used():
    n = 300
    e = np.random.default_rng(0).uniform(4, 6, n)
    e[120] = 3.0
    ref, side = EV.choose_reference(SimpleNamespace(frame_energy=e), 150, FPS, n)
    assert side == "before" and 114 <= ref <= 144 and e[ref] <= np.percentile(e[114:145], 35) + 1.0


def test_micro_expressions_are_defined_by_half_a_second():
    assert EV.MICRO_MAX_S == 0.5
