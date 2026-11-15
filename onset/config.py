"""Paths and limits shared by the server, the scripts and the tests.

Everything the instrument writes at run time lives under ``ONSET_VAR`` (default ``./var``); the shipped, read-only
material (calibration, specimen library, evidence) lives under ``data/``.
"""
from __future__ import annotations

import json
import os
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.environ.get("ONSET_DATA", os.path.join(ROOT, "data"))
LIBRARY = os.path.join(DATA, "library")
EVIDENCE = os.path.join(DATA, "evidence")
CALIBRATION = os.path.join(DATA, "calibration.json")
WEB = os.path.join(ROOT, "web")
VAR = os.environ.get("ONSET_VAR", os.path.join(ROOT, "var"))

MAX_UPLOAD_MB = 256
MAX_CLIP_SECONDS = 30.0
MIN_CLIP_SECONDS = 2.0
CAPTURE_SECONDS = 12.0
RETENTION_HOURS = 24.0


def load_calibration(path: Optional[str] = None) -> Optional[dict]:
    """Engine / spotting / conformal parameters fitted on the phantom corpus, or ``None`` when not shipped."""
    p = path or CALIBRATION
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)
