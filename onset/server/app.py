"""The ONSET web application (Flask).

Everything runs on this machine.  Routes:

    /                         the interface (static files from ``web/``)
    /api/status               versions, capabilities, limits, privacy terms
    /api/preflight            measure a short burst of webcam frames (light, face size, steadiness, sensitivity)
    /api/capture/...          record a clip in the browser, upload it frame by frame, analyse it
    /api/upload               analyse a video file
    /api/stress               re-film a specimen with a worse camera and analyse it
    /api/jobs/<id>            progress of an analysis
    /api/results/...          finished analyses (JSON, plate images), deletion, export
    /api/library/...          the specimen library (baked results, videos, ground truth)
    /api/evidence, /api/atlas calibration and evaluation evidence; the action-unit atlas
    /api/cue/...              Cue Lab (behavioural cues - no deception output; every response is checked)
"""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
import time
import zipfile
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory
from flask.json.provider import DefaultJSONProvider

from .. import config
from ..jsonsafe import clean
from ..analysis import cue_lab as CUE
from ..analysis import atlas as ATLAS
from ..analysis.pipeline import VERSION, AnalysisConfig, AnalysisError, analyse, detect
from ..analysis.preflight import preflight
from ..synth import library as LIB
from ..vision.sources import ArraySource, JpegDirSource, VideoSource
from .jobs import JobManager
from .store import Store

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv", ".mpg", ".mpeg"}
FILE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SERVE_EXT = {".jpg", ".jpeg", ".png", ".mp4", ".webm", ".json"}


class _SafeJSON(DefaultJSONProvider):
    sort_keys = False

    def dumps(self, obj, **kw):
        kw.setdefault("allow_nan", False)
        return super().dumps(clean(obj), **kw)


def _err(msg: str, code: int = 400, **extra):
    r = jsonify(dict(error=msg, **extra))
    r.status_code = code
    return r


def create_app(var_dir: Optional[str] = None, data_dir: Optional[str] = None, retention_hours: Optional[float] = None,
               analysis_threads: Optional[int] = None) -> Flask:
    web_dir = config.WEB
    app = Flask(__name__, static_folder=None)
    app.json = _SafeJSON(app)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    store = Store(var_dir or config.VAR, retention_hours or config.RETENTION_HOURS)
    store.purge_unfinished()                         # footage of a recording or analysis that a restart interrupted: gone at once
    store.purge_expired(force=True)
    jobs = JobManager(workers=1)
    data = data_dir or config.DATA
    lib_dir = os.path.join(data, "library")
    ev_dir = os.path.join(data, "evidence")
    calibration = config.load_calibration(os.path.join(data, "calibration.json"))

    def make_cfg() -> AnalysisConfig:
        return AnalysisConfig(calibration=calibration, threads=analysis_threads, max_seconds=config.MAX_CLIP_SECONDS)

    @app.before_request
    def _sweep():
        store.purge_expired()

    @app.after_request
    def _headers(resp: Response):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # ======================================================================================== status
    @app.get("/api/status")
    def status():
        try:
            import torch  # noqa: F401
            has_torch = True
        except Exception:
            has_torch = False
        man = _library_manifest()
        return jsonify(dict(
            version=VERSION, calibrated=bool(calibration), calibration=(calibration or {}).get("summary"), torch=has_torch,
            library=len(man["specimens"]) if man else 0, busy=jobs.busy(),
            limits=dict(max_upload_mb=config.MAX_UPLOAD_MB, max_clip_seconds=config.MAX_CLIP_SECONDS,
                        min_clip_seconds=config.MIN_CLIP_SECONDS, capture_seconds=config.CAPTURE_SECONDS),
            privacy=dict(
                local_only=True, retention_hours=store.retention,
                source_kept=False,
                text=("Everything runs on this computer. The recording is deleted when the analysis ends (or after "
                      f"{store.abandon:g} minutes if it is never finished); only grayscale face crops for display and the numbers stay, "
                      f"and they are deleted automatically after {store.retention:g} hours or whenever you press Delete."))))

    # ======================================================================================== pre-flight
    @app.post("/api/preflight")
    def api_preflight():
        files = request.files.getlist("frames")
        try:
            times = json.loads(request.form.get("times", "[]"))
        except ValueError:
            return _err("times must be a JSON list")
        if len(files) < 8 or len(files) != len(times) or len(files) > 120:
            return _err("send between 8 and 120 frames with one time stamp each")
        frames = []
        for f in files:
            im = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                return _err("a frame could not be decoded")
            frames.append(im)
        h, w = frames[0].shape[:2]
        if any(x.shape[:2] != (h, w) for x in frames) or max(h, w) > 2200:
            return _err("frames must share one size (at most 2200 px)")
        t = np.asarray(times, float)
        t = t - t[0]
        if np.any(np.diff(t) <= 0):
            return _err("time stamps must increase")
        return jsonify(preflight(np.stack(frames), t, mda_scale=float((calibration or {}).get("mda_scale", 1.0))))

    # ======================================================================================== capture
    @app.post("/api/capture/start")
    def capture_start():
        body = request.get_json(silent=True) or {}
        if not body.get("consent"):
            return _err("Recording needs your consent. Tick the box to continue.")
        segments = []
        for sg in (body.get("segments") or [])[:4]:
            try:
                t0, t1 = float(sg["t0"]), float(sg["t1"])
                if 0 <= t0 < t1 <= config.CAPTURE_SECONDS + 1:
                    segments.append(dict(label=str(sg.get("label", "segment"))[:40], t0=t0, t1=t1))
            except (KeyError, TypeError, ValueError):
                continue
        protocol = str(body.get("protocol", "free"))
        protocol = protocol if protocol in ("free", "baseline_task") else "free"
        sid = store.create("capture", str(body.get("label", "")),
                           dict(kind="webcam recording", source=str(body.get("device", ""))[:80], protocol=protocol, segments=segments),
                           consent=True)
        return jsonify(dict(id=sid, max_seconds=config.CAPTURE_SECONDS, expires_in_hours=store.retention))

    @app.post("/api/capture/<sid>/frames")
    def capture_frames(sid):
        if not store.exists(sid):
            return _err("unknown capture", 404)
        m = store.meta(sid)
        if m["kind"] != "capture" or m["state"] not in ("created", "capturing"):
            return _err("this capture is closed", 409)
        try:
            meta = json.loads(request.form.get("meta", "[]"))
        except ValueError:
            return _err("meta must be JSON")
        d = store.src_dir(sid)
        tpath = os.path.join(d, "times.csv")
        have = sum(1 for _ in open(tpath)) if os.path.exists(tpath) else 0
        if have + len(meta) > int(config.CAPTURE_SECONDS * 130) + 60:
            return _err("too many frames", 413)
        added = 0
        with open(tpath, "a") as fh:
            for idx, t in meta:
                f = request.files.get(f"f{int(idx)}")
                if f is None:
                    continue
                raw = f.read()
                if len(raw) < 200 or raw[:2] != b"\xff\xd8" or len(raw) > 3_000_000:
                    continue
                with open(os.path.join(d, f"{int(idx):06d}.jpg"), "wb") as out:
                    out.write(raw)
                fh.write(f"{int(idx)},{float(t):.6f}\n")
                added += 1
        if m["state"] == "created":
            store.update_meta(sid, state="capturing")
        return jsonify(dict(received=added))

    @app.post("/api/capture/<sid>/finish")
    def capture_finish(sid):
        if not store.exists(sid):
            return _err("unknown capture", 404)
        m = store.meta(sid)
        if m["kind"] != "capture" or m["state"] != "capturing":
            return _err("nothing to analyse in this capture", 409)
        body = request.get_json(silent=True) or {}
        d = store.src_dir(sid)
        stamps = {}
        for line in open(os.path.join(d, "times.csv")):
            if line.strip():
                i_s, t_s = line.strip().split(",")
                stamps[int(i_s)] = float(t_s)
        present = [(i, stamps[i]) for i in sorted(stamps) if os.path.exists(os.path.join(d, f"{i:06d}.jpg"))]
        if len(present) < 24:
            store.delete(sid)
            return _err("The recording is too short or too few frames arrived. Try again.", 422)
        times = np.array([t for _, t in present])
        if np.any(np.diff(times) <= 0):
            return _err("Frame time stamps are out of order. Try recording again.", 422)
        json.dump([float(t) for t in times], open(os.path.join(d, "times.json"), "w"))
        # frames that did not arrive are not in times.json either, so file order and time order agree
        for f in os.listdir(d):
            if f.endswith(".jpg") and int(f[:-4]) not in {i for i, _ in present}:
                os.remove(os.path.join(d, f))
        label = str(body.get("label") or m.get("label") or "Webcam recording")[:80]
        store.update_meta(sid, state="queued", label=label)
        job = _submit_analysis(sid, lambda: JpegDirSource(d), label, m["provenance"])
        return jsonify(job.to_dict())

    # ======================================================================================== upload
    @app.post("/api/upload")
    def upload():
        f = request.files.get("file")
        if f is None or not f.filename:
            return _err("Choose a video file to analyse.")
        if request.form.get("consent") not in ("1", "true", "on"):
            return _err("Confirm that you have the right to analyse this video (and the consent of the person in it).")
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in VIDEO_EXT:
            return _err(f"{ext or 'This'} is not a video format we can read. Use mp4, mov, webm, avi or mkv.")
        label = (request.form.get("label") or os.path.splitext(os.path.basename(f.filename))[0])[:80]
        sid = store.create("upload", label, dict(kind="uploaded video", filename=os.path.basename(f.filename)[:120]), consent=True)
        path = os.path.join(store.src_dir(sid), "clip" + ext)
        f.save(path)
        try:
            probe = VideoSource(path, max_seconds=config.MAX_CLIP_SECONDS)
            if probe.duration < config.MIN_CLIP_SECONDS:
                store.delete(sid)
                return _err(f"The video is shorter than {config.MIN_CLIP_SECONDS:g} seconds. Use a longer clip.", 422)
        except ValueError:
            store.delete(sid)
            return _err("That file could not be opened as a video. Try re-exporting it as mp4.", 422)
        store.update_meta(sid, state="queued")
        job = _submit_analysis(sid, lambda: VideoSource(path, max_seconds=config.MAX_CLIP_SECONDS), label, store.meta(sid)["provenance"])
        return jsonify(job.to_dict())

    def _submit_analysis(sid: str, make_source, label: str, provenance: dict, extra=None):
        def fn(progress):
            finished = False
            try:
                src = make_source()
                analyse(src, store.out_dir(sid), make_cfg(), progress=progress, label=label, provenance=provenance)
                if extra:
                    extra(sid, src)
                finished = True
            finally:
                if finished:
                    store.drop_source(sid)           # the report needs the grayscale plates, not the footage
                else:
                    store.delete(sid)                # nothing was delivered, so nothing is kept: not the footage, not partial crops
            res = json.load(open(os.path.join(store.out_dir(sid), "result.json")))
            store.update_meta(sid, state="done", grade=res["quality"]["grade"], duration_s=res["clip"]["duration_s"],
                              n_events=len([e for e in res["events"] if e["status"] in ("micro", "expression", "unexplained", "sustained")]))
            return sid
        try:
            return jobs.submit("analysis", fn)
        except RuntimeError:
            store.delete(sid)
            abort(_err("The analyser is busy with other clips. Wait a moment and try again.", 429))

    # ======================================================================================== stress lab
    @app.post("/api/stress")
    def stress():
        body = request.get_json(silent=True) or {}
        spec_id = str(body.get("specimen", ""))
        if spec_id not in LIB.BY_ID:
            return _err("unknown specimen", 404)
        spec = LIB.get(spec_id)
        camera = body.get("camera") or spec["camera"]
        nuisance = body.get("nuisance") or spec["nuisance"]
        try:
            amp_scale = float(body.get("amp_scale", 1.0))
            noise_scale = float(body.get("noise_scale", 1.0))
            blur_add = float(body.get("blur", 0.0))
            crf = body.get("codec_crf")
            crf = None if crf in (None, "", 0) else int(crf)
        except (TypeError, ValueError):
            return _err("numbers expected")
        if not (0.2 <= amp_scale <= 4.0 and 0.0 <= noise_scale <= 6.0 and 0.0 <= blur_add <= 3.0):
            return _err("amplitude 0.2-4x, noise 0-6x, blur 0-3 px")
        from ..synth.scenarios import CAMERAS
        if camera not in CAMERAS:
            return _err("unknown camera preset")
        base = CAMERAS[camera][1]
        over = dict(noise=base.noise * noise_scale, blur=base.blur + blur_add)
        if crf is not None:
            over["codec_crf"] = crf
        label = f"{spec['title']}, re-filmed"
        sid = store.create("stress", label, dict(kind="synthetic phantom (stress test)", specimen=spec_id, camera=camera, nuisance=nuisance,
                                                  amp_scale=amp_scale, noise_scale=noise_scale, blur_add=blur_add, codec_crf=crf,
                                                  note="Procedurally rendered face with scripted movements. Not a person."), consent=True)
        store.update_meta(sid, state="queued")

        def work(progress):
            from ..analysis.scoring import score_clip
            progress(0.01, "Filming the phantom")
            from ..synth.truth import exact_events
            rend, clip = LIB.render_with_renderer(spec, camera=camera, nuisance=nuisance, amp_scale=amp_scale, camera_overrides=over)
            src = ArraySource(clip.frames, clip.times)
            cfg = make_cfg()

            def sub(f, s):
                progress(0.05 + 0.95 * f, s)
            det = detect(src, cfg)                       # the registration is needed to express the scripted truth in the instrument's unit
            exact = exact_events(rend, clip, det, clip.truth["events"]) if clip.truth["events"] else []
            res = analyse(src, store.out_dir(sid), cfg, progress=sub, label=label, provenance=store.meta(sid)["provenance"], detection=det)
            truth = LIB.truth_json(clip, spec, exact)
            sc = score_clip(truth["events"], res, truth["blinks"])
            json.dump(truth, open(os.path.join(store.out_dir(sid), "truth.json"), "w"))
            json.dump(sc, open(os.path.join(store.out_dir(sid), "score.json"), "w"), default=float)
            store.drop_source(sid)
            store.update_meta(sid, state="done", grade=res["quality"]["grade"], duration_s=res["clip"]["duration_s"],
                              n_events=len([e for e in res["events"] if e["status"] in ("micro", "expression", "unexplained", "sustained")]))
            return sid

        def fn(progress):
            try:
                return work(progress)
            except BaseException:
                store.delete(sid)                        # a failed re-filming leaves nothing behind
                raise
        try:
            return jsonify(jobs.submit("stress", fn).to_dict())
        except RuntimeError:
            store.delete(sid)
            return _err("The analyser is busy with other clips. Wait a moment and try again.", 429)

    @app.get("/api/stress/options")
    def stress_options():
        from ..synth.scenarios import CAMERAS, NUISANCE_LEVELS
        return jsonify(dict(cameras=[dict(id=k, fps=v[0], noise=v[1].noise, blur=v[1].blur, jpeg_q=v[1].jpeg_q, codec_crf=v[1].codec_crf)
                                     for k, v in CAMERAS.items()], nuisance=list(NUISANCE_LEVELS),
                            specimens=[dict(id=s["id"], title=s["title"], group=s["group"]) for s in LIB.SPECIMENS]))

    # ======================================================================================== jobs, results
    @app.get("/api/jobs/<jid>")
    def job(jid):
        j = jobs.get(jid)
        if j is None:
            return _err("unknown job", 404)
        return jsonify(j.to_dict())

    @app.get("/api/results")
    def results_list():
        return jsonify([dict(id=m["id"], kind=m["kind"], label=m.get("label", ""), created=m["created"], expires=m["expires"],
                             grade=m.get("grade"), duration_s=m.get("duration_s"), n_events=m.get("n_events")) for m in store.list()])

    @app.get("/api/results/<sid>")
    def result(sid):
        if not store.exists(sid):
            return _err("This analysis was deleted or has expired.", 404)
        p = os.path.join(store.out_dir(sid), "result.json")
        if not os.path.exists(p):
            return _err("This analysis is not finished.", 409)
        res = json.load(open(p))
        res["session"] = store.meta(sid)
        for name in ("truth", "score"):
            q = os.path.join(store.out_dir(sid), f"{name}.json")
            if os.path.exists(q):
                res[name] = json.load(open(q))
        return jsonify(res)

    @app.get("/api/results/<sid>/files/<name>")
    def result_file(sid, name):
        if not store.exists(sid) or not FILE_RE.match(name) or os.path.splitext(name)[1].lower() not in SERVE_EXT:
            abort(404)
        return send_from_directory(store.out_dir(sid), name, max_age=0)

    @app.delete("/api/results/<sid>")
    def result_delete(sid):
        return jsonify(dict(deleted=store.delete(sid)))

    @app.post("/api/purge")
    def purge():
        return jsonify(dict(deleted=store.purge_all()))

    @app.get("/api/results/<sid>/export")
    def result_export(sid):
        if not store.exists(sid):
            abort(404)
        buf = io.BytesIO()
        out = store.out_dir(sid)
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name in sorted(os.listdir(out)):
                if os.path.splitext(name)[1].lower() in SERVE_EXT:
                    z.write(os.path.join(out, name), name)
            z.writestr("README.txt", "ONSET analysis export. result.json holds every number shown in the report; the images are "
                                     "grayscale face plates and magnified clips. Provenance and limits are inside result.json.\n")
        buf.seek(0)
        return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"onset-{sid}.zip")

    # ======================================================================================== library
    _lib_cache: dict = {}

    def _library_manifest():
        """The manifest, listing only the specimens whose files are actually here: the library may have been copied in parts or be
        part-way through a rebuild, and a specimen that is not there should not appear as a broken entry."""
        p = os.path.join(lib_dir, "manifest.json")
        if not os.path.exists(p):
            return None
        mt = os.path.getmtime(p)
        if _lib_cache.get("mt") != mt:
            _lib_cache.update(mt=mt, man=json.load(open(p)))
        man = _lib_cache["man"]
        here = [s for s in man.get("specimens", []) if os.path.exists(os.path.join(lib_dir, str(s.get("id", "")), "result.json"))]
        return dict(man, specimens=here)

    @app.get("/api/library")
    def library():
        man = _library_manifest()
        if man is None:
            return jsonify(dict(version=1, groups=[], specimens=[], note="The specimen library has not been built yet."))
        return jsonify(man)

    @app.get("/api/library/<spec_id>/result")
    def library_result(spec_id):
        if spec_id not in LIB.BY_ID:
            abort(404)
        p = os.path.join(lib_dir, spec_id, "result.json")
        if not os.path.exists(p):
            abort(404)
        res = json.load(open(p))
        res["truth"] = json.load(open(os.path.join(lib_dir, spec_id, "truth.json")))
        res["score"] = json.load(open(os.path.join(lib_dir, spec_id, "score.json")))
        return jsonify(res)

    @app.get("/api/library/<spec_id>/files/<name>")
    def library_file(spec_id, name):
        if spec_id not in LIB.BY_ID or not FILE_RE.match(name) or os.path.splitext(name)[1].lower() not in SERVE_EXT:
            abort(404)
        return send_from_directory(os.path.join(lib_dir, spec_id), name, max_age=3600)

    # ======================================================================================== evidence, atlas
    @app.get("/api/evidence")
    def evidence():
        out = {}
        if os.path.isdir(ev_dir):
            for name in sorted(os.listdir(ev_dir)):
                if name.endswith(".json"):
                    out[name[:-5]] = json.load(open(os.path.join(ev_dir, name)))
        out["calibration"] = calibration
        return jsonify(out)

    @app.get("/api/atlas")
    def atlas():
        p = os.path.join(data, "atlas.json")
        if os.path.exists(p):
            return jsonify(json.load(open(p)))
        return _err("The atlas has not been built yet.", 404)

    # ======================================================================================== cue lab
    def _cue(payload):
        CUE.assert_no_verdict(payload)
        return jsonify(payload)

    def _result_for(body) -> dict:
        """Result JSON from a session id or a library specimen id in the request."""
        if body.get("specimen"):
            sid = str(body["specimen"])
            p = os.path.join(lib_dir, sid, "result.json") if sid in LIB.BY_ID else None
        elif body.get("result"):
            sid = str(body["result"])
            p = os.path.join(store.out_dir(sid), "result.json") if store.exists(sid) else None
        else:
            p = None
        if not p or not os.path.exists(p):
            abort(_err("Choose an analysed recording first.", 404))
        return json.load(open(p))

    @app.get("/api/cue/evidence")
    def cue_evidence():
        return _cue(CUE.evidence_payload())

    @app.post("/api/cue/predictive")
    def cue_predictive():
        b = request.get_json(silent=True) or {}
        try:
            return _cue(dict(values=CUE.predictive_values(b["sensitivity"], b["specificity"], b["prevalence"]),
                             curve=CUE.base_rate_curve(b["sensitivity"], b["specificity"])))
        except (KeyError, TypeError, ValueError) as e:
            return _err(f"sensitivity, specificity and prevalence between 0 and 1 are needed ({e})")

    @app.post("/api/cue/roc")
    def cue_roc():
        b = request.get_json(silent=True) or {}
        try:
            d = float(b["d"])
            prev = float(b.get("prevalence", 0.5))
        except (KeyError, TypeError, ValueError):
            return _err("d (effect size) is needed")
        if not 0 <= d <= 6:
            return _err("d must lie between 0 and 6")
        return _cue(dict(auc=CUE.auc_from_d(d), roc=CUE.binormal_roc(d), best=CUE.best_operating_point(d, prev),
                         d_for_accuracy={str(a): CUE.d_needed_for_balanced_accuracy(a) for a in (0.6, 0.7, 0.8, 0.9)}))

    @app.post("/api/cue/rates")
    def cue_rates():
        b = request.get_json(silent=True) or {}
        try:
            k1, s1, k2, s2 = int(b["k1"]), float(b["s1"]), int(b["k2"]), float(b["s2"])
        except (KeyError, TypeError, ValueError):
            return _err("k1, s1, k2, s2 are needed")
        if min(k1, k2) < 0 or min(s1, s2) <= 0:
            return _err("counts must be >= 0 and durations > 0")
        return _cue(dict(a=CUE.poisson_rate_ci(k1, s1), b=CUE.poisson_rate_ci(k2, s2), test=CUE.rate_ratio_test(k1, s1, k2, s2),
                         min_events=CUE.min_events_for_significance(s1, s2)))

    @app.post("/api/cue/forking")
    def cue_forking():
        b = request.get_json(silent=True) or {}
        try:
            n = int(b.get("n_tests", 9))
        except (TypeError, ValueError):
            return _err("n_tests must be an integer")
        n = min(max(n, 1), 200)
        return _cue(dict(n_tests=n, p_any=CUE.forking_paths(n), demo=CUE.cue_lab_demo_null(int(b.get("seed", 0))) if b.get("demo") else None))

    @app.post("/api/cue/segments")
    def cue_segments():
        b = request.get_json(silent=True) or {}
        res = _result_for(b)
        try:
            a = dict(label=str(b["a"].get("label", "Baseline")), t0=float(b["a"]["t0"]), t1=float(b["a"]["t1"]))
            c = dict(label=str(b["b"].get("label", "Target")), t0=float(b["b"]["t0"]), t1=float(b["b"]["t1"]))
            comp = CUE.compare_segments(res, a, c, n_perm=2000)
        except (KeyError, TypeError, ValueError) as e:
            return _err(f"Two segments with start and end times are needed ({e})")
        return _cue(dict(comparison=comp, audit=CUE.audit_claims(res, comp)))

    @app.post("/api/cue/audit")
    def cue_audit():
        b = request.get_json(silent=True) or {}
        return _cue(dict(audit=CUE.audit_claims(_result_for(b))))

    # ======================================================================================== web
    @app.get("/")
    def index():
        return send_from_directory(web_dir, "index.html", max_age=0)

    @app.get("/<path:path>")
    def static_files(path):
        if path.startswith("api/"):
            abort(404)
        full = os.path.join(web_dir, path)
        if not os.path.isfile(full):
            abort(404)
        return send_from_directory(web_dir, path, max_age=0)

    app.onset = dict(store=store, jobs=jobs, calibration=calibration)
    return app


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run the ONSET interface on this computer.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    app = create_app()
    print(f"ONSET {VERSION}  -  http://{a.host}:{a.port}   (data stays on this computer)")
    app.run(host=a.host, port=a.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
