"""Scoring of detected events against scripted ground truth (phantom clips, library specimens, real annotations).

Spotting follows the MEGC convention: a detection is a true positive when the intersection-over-union of the
detected and the annotated [onset, offset] intervals is at least 0.5.  We also report a looser apex-timing match
(the apex within +-0.15 s) because the offset of a fading movement is inherently fuzzy.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

EXPRESSION_STATUSES = {"micro", "expression", "unexplained", "sustained"}


def iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def match_events(truth: list[dict], detected: list[dict], iou_thr: float = 0.5, apex_tol: float = 0.15) -> dict:
    """Greedy one-to-one matching by apex distance.  Returns matches and the unmatched sets."""
    det = [d for d in detected if d["status"] in EXPRESSION_STATUSES]
    used: set = set()
    matches = []
    for ti, t in enumerate(truth):
        best, bd = None, 1e9
        for di, d in enumerate(det):
            if di in used:
                continue
            dist = abs(d["t_apex"] - t["t_apex"])
            if dist < bd:
                best, bd = di, dist
        if best is None or bd > max(apex_tol, 0.3 * (t["t_off"] - t["t_on"])):
            matches.append(dict(truth=ti, det=None))
            continue
        d = det[best]
        t_off = d["t_offset"] if d["t_offset"] is not None else d["t_apex"] + 0.3
        i = iou(t["t_on"], t["t_off"], d["t_onset"], t_off)
        used.add(best)
        matches.append(dict(truth=ti, det=best, iou=i, apex_err=float(d["t_apex"] - t["t_apex"]), strict=bool(i >= iou_thr),
                            event_id=d["id"]))
    unmatched = [d for di, d in enumerate(det) if di not in used]
    return dict(matches=matches, false_alarms=unmatched, detections=det)


def au_sets(truth_event: dict) -> set:
    return {int(a) for a in truth_event["aus"]}


def score_clip(truth_events: list[dict], result: dict, truth_blinks: Optional[list] = None, pip_thr: float = 0.5,
               iou_thr: float = 0.5) -> dict:
    m = match_events(truth_events, result["events"], iou_thr)
    tp_loose = sum(1 for x in m["matches"] if x["det"] is not None)
    tp_strict = sum(1 for x in m["matches"] if x.get("strict"))
    au_tp = au_fp = au_fn = 0
    exact = 0
    rows = []
    amps = []
    for x in m["matches"]:
        if x["det"] is None:
            continue
        d = m["detections"][x["det"]]
        pred = {a["au"] for a in d["aus"] if a["pip"] >= pip_thr}
        tru = au_sets(truth_events[x["truth"]])
        au_tp += len(pred & tru)
        au_fp += len(pred - tru)
        au_fn += len(tru - pred)
        exact += int(pred == tru)
        rows.append(dict(truth=sorted(tru), pred=sorted(pred), event=d["id"], status=d["status"]))
        expected = truth_events[x["truth"]].get("au_pct_iod")          # present when the truth is in the instrument's own unit
        if expected:
            for a in d["aus"]:
                k = str(a["au"])
                if a["pip"] >= pip_thr and k in expected and all(q in a for q in ("amp_pct_iod", "lo_pct_iod", "hi_pct_iod")):
                    amps.append(dict(au=a["au"], event=d["id"], expected=float(expected[k]), est=float(a["amp_pct_iod"]),
                                     lo=float(a["lo_pct_iod"]), hi=float(a["hi_pct_iod"]),
                                     covered=bool(a["lo_pct_iod"] <= expected[k] <= a["hi_pct_iod"])))
    prec = au_tp / max(au_tp + au_fp, 1)
    rec = au_tp / max(au_tp + au_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    blink_fa = 0
    if truth_blinks is not None:
        for d in result["events"]:
            if d["status"] == "blink" and not any(b["t_on"] - 0.3 <= d["t_apex"] <= b["t_off"] + 0.4 for b in truth_blinks):
                blink_fa += 1
    return dict(n_truth=len(truth_events), tp_loose=tp_loose, tp_strict=tp_strict, fp=len(m["false_alarms"]),
                fn_loose=len(truth_events) - tp_loose, au_precision=prec, au_recall=rec, au_f1=f1, exact_sets=exact,
                n_matched=len(rows), rows=rows, amplitudes=amps, amp_covered=sum(a["covered"] for a in amps), blink_false_alarms=blink_fa,
                false_alarm_details=[dict(t=d["t_apex"], status=d["status"], aus=[a["au"] for a in d["aus"] if a["pip"] >= 0.5]) for d in m["false_alarms"]],
                apex_errors=[x["apex_err"] for x in m["matches"] if x["det"] is not None])


def spotting_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, (2 * p * r / max(p + r, 1e-9))
