"""Shared fixtures.  Slow fixtures render a short phantom clip once per session and are used only by tests marked ``slow``."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def phantom_clip():
    """A 3.2 s webcam-like phantom clip with one scripted AU12 movement (apex 1.7 s, 2.2 % of the eye distance) and its truth."""
    from onset.synth.render import Event, Renderer
    from onset.synth.scenarios import make_scenario
    ev = Event({12: 1.0}, 1.45, 1.70, 2.05, amp=0.022, label="smile", kind="micro")
    sc = make_scenario("A", "webcam30", 3.2, [ev], seed=5, nuisance="still", size=360, name="pytest-au12")
    return Renderer(sc).render()


@pytest.fixture(scope="session")
def analysed(phantom_clip, tmp_path_factory):
    """The full pipeline run on ``phantom_clip``: (result dict, output folder)."""
    from onset.analysis.pipeline import AnalysisConfig, analyse
    from onset.vision.sources import ArraySource
    out = str(tmp_path_factory.mktemp("analysis"))
    res = analyse(ArraySource(phantom_clip.frames, phantom_clip.times), out, AnalysisConfig(threads=1))
    return res, out
