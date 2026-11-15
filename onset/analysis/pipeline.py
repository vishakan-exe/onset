"""End-to-end analysis of one clip: tracking -> stabilisation -> spotting -> measurement -> inference ->
explanation -> rendering.  The result is a JSON-serialisable dict plus image files in ``outdir``.

Nothing in here is specific to the source of the footage: uploads, webcam recordings and synthetic specimens
all arrive as a ``FrameSource``.  Provenance (who/what made the footage) is carried through untouched so the
interface can always say what it is showing.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from ..jsonsafe import clean as _clean_json
from ..vision import geometry as G
from ..vision.measure import Measurer
from ..vision.nuisance import NuisanceCues, cues as nuisance_cues
from ..vision.register import register, track_landmarks
from ..vision.sources import FrameSource
from ..vision.spotting import SPOT_THRESHOLD, Candidate, SpottingResult, spot, whiten_templates
from . import categories as CAT
from . import explain as EXP
from . import quality as QUAL
from . import templates as TPL
from . import uncertainty as UNC
from .au_engine import AUEngine, AUPosterior
from .events import MICRO_MAX_S, EventMeasurement, measure_event
from .magnify import crop_resize, magnify_event, plate_box

VERSION = "1.0"
EYE_GROUP = list(G.ROI_GROUPS["eye"])
MOUTH_GROUP = list(G.ROI_GROUPS["mouth"]) + list(G.ROI_GROUPS["cheek"])      # the lower face moves with talking and chewing

LIMITS = [
    "Amplitudes are the mean displacement of the dominant ROI of an AU template, in canonical pixels and in "
    "percent of the inter-ocular distance (%IOD); they are not muscle forces.",
    "AU probabilities come from a model with hand-written ROI templates (FACS anatomy). On synthetic phantoms its "
    "calibration is measured; on real people it has to be evaluated on licensed datasets (see the evaluation protocol).",
    "Category labels are hypotheses derived from AU evidence with an expert prior. They are not a trained recogniser and "
    "datasets disagree on how AUs map to emotion labels.",
    "A movement of the face does not tell what a person feels, and micro-movements are not evidence of lying.",
]


@dataclass
class AnalysisConfig:
    max_seconds: float = 30.0
    spot_threshold: float = SPOT_THRESHOLD
    max_events: int = 10
    explain_events: int = 4
    n_explain_random: int = 8
    sprite_size: int = 176
    sprite_cols: int = 12
    sprite_rows: int = 10
    sprite_frames_max: int = 480
    magnify_alpha: float = 6.0
    magnify_events: int = 5
    magnify_size: int = 176
    engine_eta: float = 0.15
    engine_temper: float = 1.0
    conformal_alpha: float = 0.1
    schemes: tuple = ("megc3", "casme2_5", "basic7")
    threads: Optional[int] = None
    calibration: Optional[dict] = None          # loaded from data/calibration.json when present


class AnalysisError(Exception):
    """A problem the user can act on (no face, clip too short, ...)."""


class Progress:
    def __init__(self, cb: Optional[Callable[[float, str], None]] = None):
        self.cb, self.stage = cb, ""

    def __call__(self, frac: float, stage: Optional[str] = None):
        if stage:
            self.stage = stage
        if self.cb:
            self.cb(float(min(max(frac, 0.0), 1.0)), self.stage)

    def sub(self, lo: float, hi: float, stage: str):
        self(lo, stage)
        return lambda p: self(lo + (hi - lo) * p, stage)


def _np(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(type(o))


def _round(a, nd=3):
    return np.round(np.asarray(a, float), nd).tolist()


# ------------------------------------------------------------------------------------------------------
def _group_activity(spot: SpottingResult, n: int, centroids: np.ndarray, centre) -> dict:
    """Per-group movement strength on every frame: the RMS over the group's ROIs of the whitened transient, in noise
    units, with the residual head motion (the affine nuisance directions) projected out first - without that the
    stabilisation jitter, which moves all regions together, swamps the lanes.  Maximum over time scales.  Under
    noise alone the value sits around 1.3-1.8."""
    frames = np.arange(n)
    out = {g: np.zeros(n) for g in G.GROUP_ORDER}
    for r in spot.scales:
        if len(r.centre) == 0:
            continue
        _, Q, w = whiten_templates(r.sigma, centroids, centre)
        m = len(r.T)
        Z = r.T.reshape(m, -1) * w
        Zp = Z - (Z @ Q) @ Q.T
        z2 = (Zp.reshape(m, -1, 2) ** 2).sum(-1)                  # (m, R)
        for g in G.GROUP_ORDER:
            idx = [G.ROI_NAMES.index(n_) for n_ in G.ROI_GROUPS[g]]
            act = np.sqrt(z2[:, idx].mean(1))
            out[g] = np.maximum(out[g], np.interp(frames, r.centre, act, left=0, right=0))
    return out


def _downsample(a: np.ndarray, n_out: int) -> np.ndarray:
    if len(a) <= n_out:
        return a
    idx = np.linspace(0, len(a) - 1, n_out).round().astype(int)
    return a[idx]


def _contours(rois: dict, box: tuple, size: int) -> dict:
    x0, y0, x1, y1 = box
    sx, sy = size / (x1 - x0), size / (y1 - y0)
    out = {}
    for name, m in rois.items():
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in cs:
            if cv2.contourArea(c) < 4:
                continue
            c = cv2.approxPolyDP(c, 1.0, True)[:, 0, :].astype(float)
            polys.append(np.round(np.column_stack([(c[:, 0] - x0) * sx, (c[:, 1] - y0) * sy]), 1).tolist())
        out[name] = polys
    return out


def _save_sheet(imgs: list[np.ndarray], path: str, cols: int, quality: int = 82):
    n = len(imgs)
    h, w = imgs[0].shape[:2]
    rows = int(np.ceil(n / cols))
    sheet = np.zeros((rows * h, cols * w), np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return rows


def _select_sprite_frames(n: int, events: list, times: np.ndarray, cap: int) -> np.ndarray:
    need = set()
    for ev in events:
        t0 = ev["t_onset"] - 0.30
        t1 = (ev["t_offset"] if ev["t_offset"] is not None else ev["t_apex"] + 0.5) + 0.30
        need.update(np.flatnonzero((times >= t0) & (times <= t1)).tolist())
    if n <= cap:
        return np.arange(n)
    room = max(cap - len(need), cap // 4)
    stride = int(np.ceil(n / room))
    base = set(range(0, n, stride))
    sel = sorted(need | base)
    if len(sel) > cap:                                   # events alone exceed the cap: thin them uniformly
        sel = sorted(np.array(sel)[np.linspace(0, len(sel) - 1, cap).round().astype(int)].tolist())
    return np.array(sel)


# ------------------------------------------------------------------------------------------------------
def inference_inputs(em: EventMeasurement) -> tuple[np.ndarray, float, float]:
    """ROI mask and blink / speech priors handed to the AU engine for one measured movement.  Eyelid regions are
    dropped when a blink overlaps the event, the mouth region when talking does; the cues set the prior odds that a
    blink or mouth-movement component is present (never zero, never one)."""
    c = em.cand
    mask = np.ones(len(G.ROI_NAMES))
    if em.blink_cue > 0.3 or c.kind == "blink":
        for nme in EYE_GROUP:
            mask[G.ROI_NAMES.index(nme)] = 0.0
    if em.speech_cue > 0.5:
        for nme in MOUTH_GROUP:
            mask[G.ROI_NAMES.index(nme)] = 0.0
    # A candidate whose best template is the blink one is very likely a blink (landmarks confirm most of them).  A
    # mouth-movement-like candidate is only nudged: a chin raise or a pressed lip looks the same, so without any sign of
    # talking in the landmarks the model decides between "speech" and the AUs on the data.
    p_b = float(np.clip(0.03 + 0.92 * em.blink_cue + (0.5 if c.kind == "blink" else 0.0), 0.02, 0.97))
    p_s = float(np.clip(0.03 + 0.92 * em.speech_cue + (0.2 if c.kind == "speech" else 0.0), 0.02, 0.97))
    return mask, p_b, p_s


def _event_status(cand: Candidate, post: AUPosterior, em: EventMeasurement) -> tuple[str, str, list]:
    flags = []
    exc_iod, exc_deg = em.head_excursion
    if exc_iod > 0.05 or exc_deg > 2.5:
        flags.append(f"head moved during the event ({exc_iod * 100:.1f}% IOD, {exc_deg:.1f} deg)")
    if em.blink_cue > 0.3:
        flags.append("overlaps a blink: eyelid regions excluded")
    if em.speech_cue > 0.5:
        flags.append("overlaps mouth movement: lower face unreliable")
    if em.sustained:
        flags.append("displacement persists to the end of the measured window")
    # a movement found with the blink's (or the talking's) regions left out is evidenced by the other regions: the
    # prior that a blink overlaps it must not relabel it
    if cand.kind == "blink" or (post.p_blink > 0.5 and getattr(cand, "masked", "") != "blink"):
        return "blink", "Blink", flags
    if post.p_speech > 0.5 and getattr(cand, "masked", "") != "speech":
        return "speech", "Mouth movement (talking / chewing)", flags
    if em.sustained:
        return "sustained", "Sustained displacement (not brief)", flags
    if post.p_null > 0.5:
        return "unexplained", "Movement without a clear AU pattern", flags
    dur = em.duration
    if dur is not None and dur > MICRO_MAX_S:
        return "expression", f"Expression-like movement, {dur * 1000:.0f} ms (longer than a micro-expression)", flags
    if dur is None:
        return "expression", "Expression-like movement (end not observed)", flags
    return "micro", f"Brief facial movement, {dur * 1000:.0f} ms", flags


def _au_rows(post: AUPosterior, iod_canon: float, thr: float = 0.05, k_widen: float = 1.0, iod_src: Optional[float] = None) -> list[dict]:
    """One row per AU that has a non-negligible inclusion probability.  The amplitude is the size *if the AU is
    present* with a 90 % interval; the interval is the posterior's, widened by the factor ``k_widen`` that made 90 %
    of the true amplitudes fall inside it on held-out phantom identities (1.0 when the installation is uncalibrated).
    ``amp_px`` is in the recording's own pixels (``iod_src`` = distance between the eyes there), not the stabilised canvas."""
    rows = []
    to_px = (iod_src / iod_canon) if iod_src else 1.0
    for j, a in enumerate(post.aus):
        if post.pip[j] < thr:
            continue
        est, lo_g, hi_g = float(post.amp_given[j]), float(post.amp_given_lo[j]), float(post.amp_given_hi[j])
        lo, hi = max(0.0, est - k_widen * (est - lo_g)), est + k_widen * (hi_g - est)
        rows.append(dict(au=a, name=TPL.AU_NAMES[a], pip=float(post.pip[j]),
                         amp_px=est * to_px, amp_pct_iod=est / iod_canon * 100,
                         lo_pct_iod=lo / iod_canon * 100, hi_pct_iod=hi / iod_canon * 100))
    rows.sort(key=lambda r: -r["pip"])
    return rows


def _event_traces(engine: AUEngine, em: EventMeasurement, aus: list[int], mask: np.ndarray) -> dict:
    """Per-AU amplitude along the event window (posterior-mean fit of the leading AU set at every frame)."""
    if not aus:
        return {}
    out = {a: [] for a in aus}
    for i in range(len(em.frames)):
        f = engine.fit_hypothesis(aus, em.D[i], em.sigma, mask)
        for a in aus:
            out[a].append(float(f["coef"][TPL.AUS.index(a)]))
    return out


@dataclass
class Detection:
    """Everything up to and including spotting: the part of the analysis that needs the pixels."""
    info: dict
    track: object
    reg: object
    times: np.ndarray
    cues: NuisanceCues
    meas: Measurer
    cent: np.ndarray
    spot_res: SpottingResult
    face_fraction: float
    quality: object
    timing: dict


def detect(source: FrameSource, cfg: AnalysisConfig, P: Optional[Progress] = None) -> Detection:
    """Track -> stabilise -> nuisance cues -> multi-scale spotting -> capture quality."""
    P = P or Progress(None)
    cal = cfg.calibration or {}
    timing: dict = {}
    info = source.info()
    fps = source.fps
    n = source.n
    if n < 12 or fps <= 0:
        raise AnalysisError("The clip is too short to analyse (it needs at least a second of video).")
    if info["duration_s"] < 0.8:
        raise AnalysisError("The clip is shorter than one second.")
    t0 = time.time()
    try:
        track = track_landmarks(source, progress=P.sub(0.00, 0.10, "Finding the face"))
    except ValueError as e:
        raise AnalysisError("No face could be found in the footage. Face the camera, keep the face fully in view and "
                            "well lit.") from e
    face_fraction = float(track.valid.mean())
    reg = register(source, track, progress=P.sub(0.10, 0.20, "Stabilising the face"))
    times = np.asarray(source.times, float)
    cues = nuisance_cues(track, reg, times, fps)
    timing["track_register_s"] = time.time() - t0
    meas = Measurer(reg, threads=cfg.threads)
    meas.set_blink_frames(cues.blink_mask, pad=max(1, int(round(0.05 * fps))))
    cent = TPL.roi_centroids(reg.rois)
    t0 = time.time()
    spot_res = spot(meas, times, fps, cent, meas.centre, threshold=cal.get("spot_threshold", cfg.spot_threshold),
                    progress=P.sub(0.20, 0.62, "Scanning for movement"), blink_windows=cues.blink_windows,
                    speech_windows=cues.speech_windows)
    timing["spot_s"] = time.time() - t0
    big = max(spot_res.scales, key=lambda r: r.spec.k)
    quality = QUAL.assess(info, track, reg, big.sigma_pair, cent, meas.centre, face_fraction,
                          mda_scale=float(cal.get("mda_scale", 1.0)), mda90_scale=cal.get("mda90_scale"))
    return Detection(info, track, reg, times, cues, meas, cent, spot_res, face_fraction, quality, timing)


def measure_events(det: Detection, cfg: AnalysisConfig, P: Optional[Progress] = None) -> list[EventMeasurement]:
    """Measure the strongest candidates properly (quiet reference, traces, apex pattern, timing); one per movement."""
    P = P or Progress(None)
    fps = det.info["fps"]
    cands = sorted(det.spot_res.candidates, key=lambda c: -c.score)[: cfg.max_events * 2]
    ems: list[EventMeasurement] = []
    P(0.62, "Measuring movements")
    for i, c in enumerate(cands):
        em = measure_event(det.meas, det.spot_res, c, det.cues, det.times, fps, det.cent, det.meas.centre)
        dup = next((o for o in ems if abs(o.apex_t - em.apex_t) < 0.15 and (o.template == em.template or abs(o.apex_t - em.apex_t) < 0.06)), None)
        if dup is not None:
            if c.score > dup.cand.score:
                ems[ems.index(dup)] = em
            continue
        ems.append(em)
        P(0.62 + 0.10 * (i + 1) / max(len(cands), 1), "Measuring movements")
        if len(ems) >= cfg.max_events:
            break
    ems.sort(key=lambda e: e.apex_t)
    return ems


def analyse(source: FrameSource, outdir: str, config: Optional[AnalysisConfig] = None,
            progress: Optional[Callable[[float, str], None]] = None, label: str = "", provenance: Optional[dict] = None,
            write_images: bool = True, detection: Optional["Detection"] = None) -> dict:
    """``detection``: the result of ``detect(source, cfg)`` when the caller already ran it (the library builder needs the
    registration to express scripted truth in the instrument's units); it is used instead of detecting again."""
    cfg = config or AnalysisConfig()
    cal = cfg.calibration or {}
    P = Progress(progress)
    os.makedirs(outdir, exist_ok=True)
    det = detection if detection is not None else detect(source, cfg, P)
    info, fps, n = det.info, source.fps, source.n
    track, reg, times, cues, meas, cent, spot_res = det.track, det.reg, det.times, det.cues, det.meas, det.cent, det.spot_res
    quality = det.quality
    timing = dict(det.timing)
    iod_c = reg.canon.iod
    engine = AUEngine(cent, meas.centre, iod_c, eta=cal.get("eta", cfg.engine_eta), temper=cal.get("temper", cfg.engine_temper),
                      **({"size_prior": tuple(cal["size_prior"])} if cal.get("size_prior") else {}))
    amp_k = float(cal.get("amp_interval_k", 1.0))

    # ---------------------------------------------------------------- measure + infer events
    t0 = time.time()
    ems = measure_events(det, cfg, P)
    timing["measure_s"] = time.time() - t0

    t0 = time.time()
    events = []
    posts: dict[int, AUPosterior] = {}
    masks: dict[int, np.ndarray] = {}
    P(0.72, "Reasoning about each movement")
    schemes = {name: CAT.SCHEMES[name]() for name in cfg.schemes}
    conf = (cal.get("conformal") or {})
    calibrators = {name: UNC.LACConformal(alpha=float(c["alpha"]), qhat=float(c["qhat"]), n_cal=int(c.get("n_cal", 0)))
                   for name, c in conf.items() if name in schemes}
    for i, em in enumerate(ems):
        c = em.cand
        mask, p_b, p_s = inference_inputs(em)
        post = engine.infer(em.pattern, em.sigma, mask, p_b, p_s)
        status, status_text, flags = _event_status(c, post, em)
        art = 1.0 - (1.0 - post.p_blink) * (1.0 - post.p_speech)
        cats = {}
        for name, sch in schemes.items():
            cal_c = calibrators.get(name)
            res = CAT.category_posterior(post.set_probs, art, sch, post.p_null, calibrator=cal_c)
            cats[name] = dict(labels=res.labels, probs=_round(res.probs, 4), unclear=round(res.unclear, 4), artefact=round(res.artefact, 4),
                              top=res.top, top_p=round(res.top_p, 4), margin=round(res.margin, 4), abstain=res.abstain,
                              set=res.set, set_alpha=(calibrators[name].alpha if name in calibrators else None), note=res.note)
        rows = _au_rows(post, iod_c, k_widen=amp_k, iod_src=float(det.track.iod_px))
        named = [r["au"] for r in rows if r["pip"] >= 0.5]
        withheld = []
        if status in ("blink", "speech"):
            # the lids or the mouth moved for their own reasons: AU read-outs from the rest of the pattern would be
            # leftovers of a mis-measured large movement, so they are kept out of the findings (and shown as withheld)
            withheld, rows, named = rows[:4], [], []
        events.append(dict(
            id=f"e{i + 1}", status=status, status_text=status_text, flags=flags + em.notes,
            t_onset=float(em.onset_t), t_apex=float(em.apex_t), t_offset=None if em.offset_t is None else float(em.offset_t),
            duration_ms=None if em.duration is None else float(em.duration * 1000), micro=bool(status == "micro"),
            spot=dict(score=float(c.score), tau_s=float(c.scale.tau), kind=c.kind, best_template=em.template),
            t_ref=float(em.ref_t), head_excursion=dict(iod=float(em.head_excursion[0]), roll_deg=float(em.head_excursion[1])),
            blink_cue=float(em.blink_cue), speech_cue=float(em.speech_cue),
            p_null=float(post.p_null), p_blink=float(post.p_blink), p_speech=float(post.p_speech),
            aus=rows, named_aus=named, aus_withheld=withheld,
            hypotheses=[dict(aus=h["aus"], blink=h["blink"], speech=h["speech"], p=round(h["p"], 4),
                             amp_pct_iod={str(k): round(v / iod_c * 100, 3) for k, v in h["amp"].items()}) for h in post.hypotheses],
            categories=cats, chi2_per_dof=float(post.chi2), log_bf_vs_null=float(post.log_bf_null),
        ))
        posts[i], masks[i] = post, mask
    timing["infer_s"] = time.time() - t0

    # ---------------------------------------------------------------- explanations
    t0 = time.time()
    P(0.80, "Building explanations")
    order = [i for i, e in enumerate(events) if e["status"] in ("micro", "expression")]
    order.sort(key=lambda i: -events[i]["spot"]["score"])
    for rank, i in enumerate(order):
        em, post, mask = ems[i], posts[i], masks[i]
        ledger = EXP.evidence_ledger(engine, post, em.pattern, em.sigma, mask, iod_c)
        events[i]["ledger"] = ledger
        events[i]["narrative"] = EXP.narrative(post, ledger, iod_c, em.duration)
        events[i]["traces"] = dict(t=_round(em.t, 4), amp_pct_iod=_round(em.amp_trace / iod_c * 100, 3), template=em.template,
                                   noise_pct_iod=float(em.amp_sd / iod_c * 100),
                                   per_au={str(a): _round(np.array(v) / iod_c * 100, 3) for a, v in
                                           _event_traces(engine, em, events[i]["named_aus"], mask).items()})
        if rank < cfg.explain_events:
            events[i]["faithfulness"] = EXP.deletion_test(engine, em.pattern, em.sigma, mask, ledger, events[i]["named_aus"],
                                                          post_p(post, "blink", em), post_p(post, "speech", em),
                                                          n_random=cfg.n_explain_random, seed=7 + i)
            if len(post.hypotheses) > 1:
                events[i]["contrast"] = EXP.contrast(engine, em.pattern, em.sigma, mask, post.hypotheses[0], post.hypotheses[1])
    for i, e in enumerate(events):
        if e["status"] not in ("micro", "expression"):
            e.setdefault("ledger", [])
            e.setdefault("narrative", EXP.narrative(posts[i], [], iod_c, ems[i].duration))
            e.setdefault("traces", dict(t=_round(ems[i].t, 4), amp_pct_iod=_round(ems[i].amp_trace / iod_c * 100, 3),
                                        template=ems[i].template, noise_pct_iod=float(ems[i].amp_sd / iod_c * 100), per_au={}))
        e["pattern"] = [dict(roi=nm, dx=float(ems[i].pattern[r, 0]), dy=float(ems[i].pattern[r, 1]),
                             sx=float(ems[i].sigma[r, 0]), sy=float(ems[i].sigma[r, 1]), used=bool(masks[i][r] > 0))
                        for r, nm in enumerate(G.ROI_NAMES)]
    timing["explain_s"] = time.time() - t0

    # ---------------------------------------------------------------- timeline + images
    t0 = time.time()
    P(0.86, "Rendering plates")
    S = reg.canon.size
    box = plate_box(S, reg.canon.eye_y, iod_c)
    size = cfg.sprite_size
    groups = _group_activity(spot_res, n, cent, meas.centre)
    n_out = 1500
    tl_idx = np.linspace(0, n - 1, min(n, n_out)).round().astype(int)
    timeline = dict(
        t=_round(times[tl_idx], 4), score=_round(spot_res.frame_score[tl_idx], 2),
        groups={g: _round(groups[g][tl_idx], 2) for g in G.GROUP_ORDER},
        group_noise={g: float(np.median(groups[g][groups[g] > 0])) if (groups[g] > 0).any() else 1.0 for g in G.GROUP_ORDER},
        blink=_round(cues.blink[tl_idx], 3), speech=_round(cues.speech[tl_idx], 3),
        head_x=_round(cues.head_pos[tl_idx, 0] * 100, 3), head_y=_round(cues.head_pos[tl_idx, 1] * 100, 3),
        head_roll=_round(cues.head_roll[tl_idx], 3), face_found=_round(track.valid[tl_idx].astype(float), 0),
        luminance=_round(reg.stats["lum"][tl_idx], 1), threshold=float(spot_res.threshold))
    sprites = None
    if write_images:
        sel = _select_sprite_frames(n, events, times, cfg.sprite_frames_max)
        imgs = [crop_resize(reg.gray[i], box, size) for i in sel]
        per = cfg.sprite_cols * cfg.sprite_rows
        sheets = []
        for k in range(0, len(imgs), per):
            name = f"plates_{k // per}.jpg"
            rows = _save_sheet(imgs[k:k + per], os.path.join(outdir, name), cfg.sprite_cols)
            sheets.append(dict(file=name, count=len(imgs[k:k + per]), rows=rows))
        sprites = dict(size=size, cols=cfg.sprite_cols, per_sheet=per, sheets=sheets, frame_index=sel.tolist(),
                       times=_round(times[sel], 4))
        # magnified clips for the strongest expression events
        for rank, i in enumerate(order[: cfg.magnify_events]):
            em = ems[i]
            lo_t = em.onset_t - 0.12
            hi_t = (em.offset_t if em.offset_t is not None else em.apex_t + 0.4) + 0.12
            fr = np.flatnonzero((times >= lo_t) & (times <= hi_t))
            if len(fr) > 40:
                fr = fr[np.linspace(0, len(fr) - 1, 40).round().astype(int)]
            if len(fr) < 3:
                continue
            apex_f = int(em.frames[em.apex_i])
            if apex_f not in fr:
                fr = np.sort(np.append(fr, apex_f))
            m = magnify_event(meas, em.ref, [int(f) for f in fr], cfg.magnify_alpha, box, cfg.magnify_size, apex_f)
            name = f"mag_{events[i]['id']}.jpg"
            rows = _save_sheet(list(m["frames"]), os.path.join(outdir, name), 8)
            heat_name = None
            if m["heat"] is not None:
                a = np.clip(m["heat"] * 255, 0, 255).astype(np.uint8)
                rgba = np.zeros(a.shape + (4,), np.uint8)
                rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3] = 40, 170, 255, (a * 0.85).astype(np.uint8)   # BGRA amber
                heat_name = f"heat_{events[i]['id']}.png"
                cv2.imwrite(os.path.join(outdir, heat_name), rgba)
            events[i]["magnified"] = dict(sheet=name, rows=rows, cols=8, size=cfg.magnify_size, alpha=cfg.magnify_alpha,
                                          times=_round(times[fr], 4), heat=heat_name, shrink_tau_px=float(m["tau"]))
    timing["render_s"] = time.time() - t0
    # the sum of the stages, so that it counts detection (tracking, stabilising, spotting) whether ``analyse`` ran it or the caller did
    timing["total_s"] = float(sum(v for k, v in timing.items() if k != "total_s"))
    noise = {k: v for k, v in spot_res.noise_summary.items()}

    result = dict(
        version=VERSION, label=label, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        provenance=provenance or dict(kind="unknown"),
        clip=info, quality=quality.to_dict(),
        canon=dict(size=int(S), iod=float(iod_c), eye_y=float(reg.canon.eye_y), plate_box=list(box), plate_size=size,
                   centroids=[dict(roi=nm, x=float((cent[r, 0] - box[0]) * size / (box[2] - box[0])),
                                   y=float((cent[r, 1] - box[1]) * size / (box[3] - box[1]))) for r, nm in enumerate(G.ROI_NAMES)],
                   rois=_contours(reg.rois, box, size), groups=G.ROI_GROUPS),
        spotting=dict(threshold=float(spot_res.threshold), scales=[dict(tau=r.spec.tau, k=r.spec.k, stride=r.spec.stride) for r in spot_res.scales],
                      noise=noise, n_candidates=len(spot_res.candidates)),
        nuisance=dict(blink_windows=cues.blink_windows, speech_windows=cues.speech_windows),
        timeline=timeline, events=events, sprites=sprites, timing=timing,
        engine=dict(eta=engine.eta, temper=engine.temper, tau_a_pct=[float(x / iod_c * 100) for x in engine.tau_a],
                    calibrated=bool(cal), amp_interval_k=amp_k, spot_threshold=float(spot_res.threshold),
                    mda_scale=float(cal.get("mda_scale", 1.0)), conformal_alpha=(float(next(iter(conf.values()))["alpha"]) if conf else None)),
        limits=LIMITS,
    )
    P(0.98, "Writing the report")
    with open(os.path.join(outdir, "result.json"), "w") as fh:
        json.dump(_clean_json(result), fh, default=_np, separators=(",", ":"), allow_nan=False)
    P(1.0, "Done")
    return result


def post_p(post: AUPosterior, which: str, em: EventMeasurement) -> float:
    return float(np.clip(0.03 + 0.92 * (em.blink_cue if which == "blink" else em.speech_cue), 0.02, 0.97))
