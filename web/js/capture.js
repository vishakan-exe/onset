// Capture: record a short clip with the webcam (or choose a video file), have it analysed on this computer, and
// land on the bench.  Nothing is analysed live: the clip is recorded first, so that the analysis can look at the
// whole recording, at every scale, with the quiet stretches as its reference.
//
//   set-up check (measured by the same machinery as the analysis)  ->  consent  ->  3-2-1  ->  record
//   ->  review (contact strip, frame timing)  ->  analyse (the developing tray)  ->  bench
import { h, svg, clear, fmt, icon, clamp, toast, sleep } from "./util.js";
import { api, pollJob, STATIC } from "./api.js";
import { stageHead, note, errorBox, busy, store, reduceMotion } from "./ui.js";

const STAGES = ["Finding the face", "Stabilising the face", "Scanning for movement", "Measuring movements", "Reasoning about each movement", "Building explanations", "Rendering plates", "Writing the report"];
const HAS_RVFC = typeof HTMLVideoElement !== "undefined" && "requestVideoFrameCallback" in HTMLVideoElement.prototype;
const CONSENT_TEXT = "I am the person on camera, or they have agreed. I understand that the clip is analysed on this computer and deleted when the analysis ends; grayscale face crops for the report stay for 24 hours or until I delete them.";

// ------------------------------------------------------------------------------------------------ camera
class Camera {
  constructor(video) { this.video = video; this.stream = null; this.track = null; }
  async start(deviceId) {
    this.stop();
    const base = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, min: 15 } };
    const tries = [
      { audio: false, video: deviceId ? { ...base, deviceId: { exact: deviceId } } : { ...base, facingMode: "user" } },
      { audio: false, video: deviceId ? { deviceId: { exact: deviceId } } : true },
    ];
    let err;
    for (const c of tries) {
      try { this.stream = await navigator.mediaDevices.getUserMedia(c); err = null; break; } catch (e) { err = e; if (e.name !== "OverconstrainedError") break; }
    }
    if (err) throw cameraError(err);
    this.track = this.stream.getVideoTracks()[0];
    this.video.srcObject = this.stream;
    this.video.muted = true;
    await this.video.play();
    if (!this.video.videoWidth) await new Promise((r) => this.video.addEventListener("loadeddata", r, { once: true }));
    return this.settings();
  }
  settings() {
    const s = this.track ? this.track.getSettings() : {};
    return { width: s.width || this.video.videoWidth, height: s.height || this.video.videoHeight, fps: s.frameRate || 0, deviceId: s.deviceId, label: this.track ? this.track.label : "" };
  }
  stop() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null; this.track = null; this.video.srcObject = null;
  }
}
function cameraError(e) {
  const msg = {
    NotAllowedError: "The browser did not allow camera access. Click the camera icon in the address bar, allow it for this page, and try again.",
    NotFoundError: "No camera was found on this computer.",
    NotReadableError: "The camera is in use by another program. Close it there, then try again.",
    SecurityError: "Browsers only allow the camera on https pages or on localhost. Open this page as http://localhost:8000.",
    AbortError: "The camera could not be started. Unplug it and plug it in again, then retry.",
  }[e.name] || `The camera could not be started (${e.name || e.message}).`;
  const err = new Error(msg);
  err.name = e.name;
  return err;
}

// ------------------------------------------------------------------------------------------------ frame source: exact timestamps
/** Calls back once per new camera frame with the frame drawn on a canvas and its capture time (seconds). */
class FrameTap {
  constructor(video, maxSide = 960) {
    this.video = video;
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d", { alpha: false });
    this.maxSide = maxSide;
    this.running = false;
    this.key = null;
    this.exact = HAS_RVFC;
  }
  size() {
    const w = this.video.videoWidth, hh = this.video.videoHeight, k = Math.min(1, this.maxSide / Math.max(w, hh));
    return [Math.round((w * k) / 2) * 2, Math.round((hh * k) / 2) * 2];
  }
  start(onFrame) {
    const [w, hh] = this.size();
    this.canvas.width = w; this.canvas.height = hh;
    this.running = true;
    const v = this.video;
    if (HAS_RVFC) {
      const cb = (now, md) => {
        if (!this.running) return;
        if (this.key == null) this.key = md.captureTime != null ? "captureTime" : md.receiveTime != null ? "receiveTime" : "now";
        const ms = this.key === "now" ? now : md[this.key] != null ? md[this.key] : now;
        this.ctx.drawImage(v, 0, 0, w, hh);
        onFrame(this.canvas, ms / 1000, md);
        v.requestVideoFrameCallback(cb);
      };
      v.requestVideoFrameCallback(cb);
    } else {
      let last = -1;
      const loop = (now) => {
        if (!this.running) return;
        if (v.currentTime !== last) { last = v.currentTime; this.ctx.drawImage(v, 0, 0, w, hh); onFrame(this.canvas, now / 1000, {}); }
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    }
  }
  stop() { this.running = false; }
}
const toBlob = (canvas, q) => new Promise((res) => canvas.toBlob(res, "image/jpeg", q));

// ------------------------------------------------------------------------------------------------ pre-flight loop
class Preflight {
  constructor(tap, onResult) { this.tap = tap; this.onResult = onResult; this.on = false; this.busyNow = false; this.timer = null; }
  start() { if (this.on) return; this.on = true; this.loop(); }
  stop() { this.on = false; clearTimeout(this.timer); }
  async loop() {
    while (this.on) {
      if (document.visibilityState === "visible") {
        try { const r = await this.once(); if (this.on && r) this.onResult(r); } catch (e) { if (this.on) this.onResult({ error: e.message }); }
      }
      await sleep(1800);
    }
  }
  async once() {
    // the same pixels as the recording will have (same size cap), so the face-size and sensitivity numbers are honest
    const v = this.tap.video;
    const [cw, ch] = this.tap.size();
    const work = document.createElement("canvas");
    work.width = cw; work.height = ch;
    const g = work.getContext("2d", { alpha: false });
    const shots = [];
    const t0 = performance.now();
    await new Promise((resolve) => {
      const grab = (t) => {
        g.drawImage(v, 0, 0, cw, ch);
        const copy = document.createElement("canvas");
        copy.width = cw; copy.height = ch;
        copy.getContext("2d", { alpha: false }).drawImage(work, 0, 0);
        shots.push(toBlob(copy, 0.8).then((b) => [b, t]));
      };
      if (HAS_RVFC) {
        let key = null;
        const cb = (now, md) => {
          if (!this.on) return resolve();
          if (key == null) key = md.captureTime != null ? "captureTime" : "now";
          grab((key === "now" ? now : md.captureTime) / 1000);
          if (shots.length >= 20 || performance.now() - t0 > 3000) return resolve();
          v.requestVideoFrameCallback(cb);
        };
        v.requestVideoFrameCallback(cb);
      } else {
        const iv = setInterval(() => { grab(performance.now() / 1000); if (shots.length >= 18 || !this.on) { clearInterval(iv); resolve(); } }, 80);
      }
    });
    const got = (await Promise.all(shots)).filter((x) => x[0]).sort((a, b) => a[1] - b[1]);
    if (got.length < 8) return null;
    return api.preflight(got.map((x) => x[0]), got.map((x) => x[1]));
  }
}

// ------------------------------------------------------------------------------------------------ uploader (frames in small batches while recording)
class Uploader {
  constructor(id) { this.id = id; this.queue = []; this.chain = Promise.resolve(); this.sent = 0; this.failed = null; }
  add(item) { this.queue.push(item); if (this.queue.length >= 12) this.flush(); }
  flush() {
    const batch = this.queue.splice(0, this.queue.length);
    if (!batch.length) return this.chain;
    this.chain = this.chain.then(async () => {
      for (let attempt = 0; attempt < 3; attempt++) {
        try { await api.captureFrames(this.id, batch); this.sent += batch.length; return; } catch (e) { if (attempt === 2) { this.failed = e; throw e; } await sleep(400 * (attempt + 1)); }
      }
    });
    return this.chain;
  }
  async done() { this.flush(); await this.chain; if (this.failed) throw this.failed; }
}

// ------------------------------------------------------------------------------------------------ small visuals
function statusIcon(s) { return icon(s === "good" ? "check" : s === "ok" ? "info" : "warn"); }

/** Dipstick: where this set-up's smallest detectable movement falls between "very sensitive" and "only large movements". */
function dipstick(mda) {
  const edges = [0.15, 0.6, 1.0, 1.6, 2.5, 4];             // grade bands A..E (%IOD, log scale)
  const lo = Math.log(0.15), hi = Math.log(4);
  const pos = (v) => `${(clamp((Math.log(Math.max(v, 0.15)) - lo) / (hi - lo), 0, 1) * 100).toFixed(1)}%`;
  const bands = ["A", "B", "C", "D", "E"].map((g, i) => h("i", { class: `band b${g}`, style: { left: pos(edges[i]), width: `${(clamp((Math.log(edges[i + 1]) - Math.log(edges[i])) / (hi - lo), 0, 1) * 100).toFixed(1)}%` } }, h("b", {}, g)));
  const needle = mda == null ? null : h("i", { class: "needle", style: { left: pos(mda) } });
  return h("div", { class: "dip", role: "img", "aria-label": mda == null ? "Not measured yet" : `The smallest movement this set-up can show is about ${mda.toFixed(1)} percent of the eye distance` }, bands, needle);
}

function trayCanvas() {
  const c = h("canvas", { class: "tray-canvas", "aria-hidden": "true" });
  return c;
}

// ------------------------------------------------------------------------------------------------ main
export async function mount(host, { seg }, ctx) {
  const mode = seg[0] === "upload" ? "upload" : "record";
  const cleanup = [];
  host.append(stageHead({
    kicker: "Station 01 · Capture",
    title: mode === "upload" ? "Analyse a video file" : "Record a clip",
    lede: mode === "upload"
      ? "Choose a recording of a face. It is analysed here, on this computer, and the file is deleted as soon as the analysis ends. Grayscale face crops for the report stay for 24 hours or until you delete them."
      : "Record up to a dozen seconds with your webcam. The recording is analysed afterwards, on this computer, as a whole — and deleted as soon as the analysis ends. Grayscale face crops for the report stay for 24 hours or until you delete them.",
    actions: [h("div", { class: "seg", role: "group", "aria-label": "Source" },
      h("button", { type: "button", "aria-pressed": String(mode === "record"), onclick: () => ctx.navigate("#/capture") }, "Webcam"),
      h("button", { type: "button", "aria-pressed": String(mode === "upload"), onclick: () => ctx.navigate("#/capture/upload") }, "Video file"))],
  }));
  if (STATIC) {
    host.append(note("caution", h("b", {}, "This showcase has no server. "), "Recording and analysing need the local ONSET server (python -m onset). Everything in the specimen cabinet can be explored here."),
      h("p", {}, h("a", { class: "btn primary", href: "#/cabinet" }, icon("cabinet"), "Open the specimen cabinet")));
    return;
  }
  const limits = (ctx.status && ctx.status.limits) || {};
  const priv = ctx.status && ctx.status.privacy;
  const privacy = h("p", { class: "legend-text" }, priv ? priv.text : "Everything runs on this computer.");
  const box = h("div", { class: "capture" });
  host.append(box);
  if (mode === "upload") { uploadMode(box, ctx, limits, privacy); return; }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    box.append(note("poor", h("b", {}, "This browser cannot use the camera here. "), "Browsers only offer the camera on https pages or on localhost. Open this page as http://localhost:8000, or use a recorded video file instead."),
      h("p", {}, h("a", { class: "btn", href: "#/capture/upload" }, icon("upload"), "Analyse a video file")));
    return;
  }
  const rec = new RecordStation(box, ctx, limits, privacy);
  return { dispose: () => rec.destroy() };
}

// ================================================================================================ record
class RecordStation {
  constructor(box, ctx, limits, privacy) {
    this.box = box; this.ctx = ctx; this.limits = limits; this.privacy = privacy;
    this.state = "idle";
    this.maxSeconds = Math.min(limits.capture_seconds || 12, 12);
    this.pf = null;
    this.build();
  }

  build() {
    const box = this.box;
    // viewport ------------------------------------------------------------------------------
    this.video = h("video", { class: "cam", playsinline: "", muted: "", autoplay: "", "aria-label": "Live camera preview, mirrored" });
    this.guide = svg("svg", { class: "guide", viewBox: "0 0 100 100", preserveAspectRatio: "none", "aria-hidden": "true" });
    this.guide.innerHTML = '<ellipse cx="50" cy="48" rx="21" ry="30" fill="none" stroke="rgba(255,255,255,.55)" stroke-width=".5" stroke-dasharray="2.2 1.6"/><path d="M33 40.5h34" stroke="rgba(255,255,255,.35)" stroke-width=".35" stroke-dasharray="1 1.4"/>';
    this.faceBox = h("div", { class: "facebox", hidden: true });
    this.stampL = h("span", { class: "stamp tl" }, "Camera off");
    this.stampR = h("span", { class: "stamp tr num" }, "");
    this.big = h("div", { class: "big-count", "aria-hidden": "true" });
    this.rec = h("div", { class: "rec", hidden: true }, h("i", {}), h("span", { class: "num" }, "0.0 s"));
    this.cueCard = h("div", { class: "cue-card", hidden: true });
    this.live = h("p", { class: "visually-hidden", "aria-live": "assertive" });
    this.off = h("div", { class: "cam-off" }, h("p", {}, "The camera is off."), h("p", { class: "small" }, "Press “Start the camera” to begin. Nothing is recorded until you press Record."));
    this.plate = h("div", { class: "plate cam-plate" }, this.video, this.guide, this.faceBox, this.off, this.big, this.rec, this.cueCard, this.stampL, this.stampR);
    this.lightBox = h("div", { class: "viewport" }, h("div", { class: "crop" }, this.plate));
    this.timing = h("canvas", { class: "timing", height: 38, "aria-hidden": "true", hidden: true });
    this.camBar = h("div", { class: "cam-bar" });
    // (the left column is assembled after the controls exist, below)

    // right: check -----------------------------------------------------------------------------
    this.checkHost = h("div", { class: "check-host" }, h("p", { class: "muted" }, "Start the camera to check the light, the framing and how small a movement this set-up could show."));
    this.dipHost = h("div", { class: "dip-host" });
    const right = h("div", { class: "cap-right" },
      h("section", { class: "sec first" },
        h("header", {}, h("h2", { class: "label ink" }, "Set-up check"), h("p", { class: "legend-text" }, "Measured every couple of seconds with the same machinery that will analyse the clip.")),
        this.dipHost, this.checkHost));

    // controls: consent, protocol, duration ---------------------------------------------
    this.consent = h("input", { type: "checkbox", id: "consent", onchange: () => this.sync() });
    this.protocol = h("select", { id: "protocol", "aria-label": "Protocol", onchange: () => store.set("protocol", this.protocol.value) },
      h("option", { value: "free" }, "Just record"),
      h("option", { value: "baseline_task" }, "A still baseline, then a silent counting task"));
    this.protocol.value = store.get("protocol", "free");
    this.duration = h("select", { id: "dur", "aria-label": "Length of the clip", onchange: () => store.set("dur", this.duration.value) },
      [6, 8, 10, 12].filter((d) => d <= this.maxSeconds + 0.01).map((d) => h("option", { value: d }, `${d} seconds`)));
    this.duration.value = String(store.get("dur", 8));
    if (!this.duration.value) this.duration.value = "8";
    this.label = h("input", { type: "text", id: "label", maxlength: 60, placeholder: "e.g. Morning, window light", "aria-label": "Name for this recording" });
    this.keylight = h("input", { type: "checkbox", id: "kl", onchange: () => this.lightBox.classList.toggle("lit", this.keylight.checked) });
    this.btnStart = h("button", { class: "btn primary", type: "button", onclick: () => this.startCamera() }, icon("capture"), "Start the camera");
    this.btnRec = h("button", { class: "btn pen", type: "button", disabled: true, onclick: () => this.countdown() }, icon("record"), "Record");
    this.btnStop = h("button", { class: "btn", type: "button", hidden: true, onclick: () => this.stopRecording() }, icon("stop"), "Stop now");
    this.hint = h("p", { class: "small muted hint" });
    // the buttons sit directly under the preview so the face stays in view while recording
    this.goBlock = h("div", { class: "cap-go" },
      h("label", { class: "check consent" }, this.consent, h("span", {}, CONSENT_TEXT)),
      h("div", { class: "actions" }, this.btnStart, this.btnRec, this.btnStop), this.hint);
    box.append(h("div", { class: "cap-left" }, this.lightBox, this.timing, this.camBar, this.live, this.goBlock), right);
    const form = h("section", { class: "cap-form" },
      h("div", { class: "form-grid" },
        h("div", { class: "field" }, h("label", { for: "label" }, "Name (optional)"), this.label),
        h("div", { class: "field" }, h("label", { for: "dur" }, "Length"), this.duration),
        h("div", { class: "field wide" }, h("label", { for: "protocol" }, "What to do while recording"), this.protocol,
          h("p", { class: "small muted" }, "The second option marks two stretches of the clip (still, then counting). The Cue Lab can compare them — with the statistics that keep such comparisons honest."))),
      h("label", { class: "check" }, this.keylight, h("span", {}, "Use my screen as a light. ", h("span", { class: "muted" }, "Turns the area around the preview white and bright; face the screen, not a window behind you."))),
      this.privacy);
    this.formHost = form;
    this.formWrap = h("div", { class: "cap-bottom" }, form);
    box.append(this.formWrap);
    // closing the tab with a recording that was never sent for analysis: ask the server to delete it now (it purges it within an hour regardless)
    this.onHide = () => {
      if (this.capId && this.state !== "analysing") try { fetch(`api/results/${this.capId}`, { method: "DELETE", keepalive: true }); } catch { /* ignore */ }
    };
    addEventListener("pagehide", this.onHide);
    this.sync();
  }

  sync() {
    const live = this.state === "live";
    this.btnRec.disabled = !(live && this.consent.checked);
    this.btnStart.hidden = this.state !== "idle" && this.state !== "error";
    this.hint.textContent = live && !this.consent.checked ? "Tick the consent box to enable recording." : "";
  }

  // ---------------------------------------------------------------------------------------------- camera + pre-flight
  async startCamera() {
    this.btnStart.disabled = true;
    this.cam = this.cam || new Camera(this.video);
    try {
      const s = await this.cam.start(this.deviceId);
      this.state = "live";
      this.off.hidden = true;
      this.plate.classList.add("on");
      this.stampL.textContent = `${s.width} × ${s.height}${s.fps ? ` · ${Math.round(s.fps)} fps` : ""}`;
      if (s.width && s.height) this.plate.style.aspectRatio = `${s.width} / ${s.height}`;
      this.tap = new FrameTap(this.video);
      this.pf = new Preflight(this.tap, (r) => this.showCheck(r));
      this.pf.start();
      await this.listDevices();
      this.sync();
      this.live.textContent = "Camera on.";
    } catch (e) {
      this.state = "error";
      toast(e.message, "error");
      this.off.hidden = false;
      this.off.replaceChildren(h("p", {}, e.message));
      this.sync();
    } finally { this.btnStart.disabled = false; }
  }
  async listDevices() {
    try {
      const ds = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
      clear(this.camBar);
      if (ds.length > 1) {
        const sel = h("select", { "aria-label": "Camera", onchange: async () => { this.deviceId = sel.value; this.pf && this.pf.stop(); await this.startCamera(); } },
          ds.map((d, i) => h("option", { value: d.deviceId }, d.label || `Camera ${i + 1}`)));
        const cur = this.cam.settings().deviceId;
        if (cur) sel.value = cur;
        this.camBar.append(h("label", { class: "small" }, "Camera ", sel));
      }
      this.camBar.append(h("p", { class: "small muted" }, this.cam.settings().label || "", HAS_RVFC ? "" : " · This browser cannot give exact frame times; timing will be approximate."));
    } catch { /* optional */ }
  }
  showCheck(r) {
    if (this.state !== "live") return;
    clear(this.checkHost);
    if (r.error) { this.checkHost.append(errorBox(new Error(r.error))); return; }
    this.dipHost.replaceChildren(dipstick(r.mda_pct));
    if (r.mda_pct != null) {
      this.dipHost.append(h("p", { class: "dip-cap" }, h("b", { class: "num" }, `${r.mda_pct.toFixed(2)} %`), " of the eye distance is the smallest movement this set-up could show",
        r.face ? h("span", { class: "muted num" }, ` (about ${(r.mda_pct * r.face.iod_px / 100).toFixed(1)} px)`) : null, "."));
    }
    this.checkHost.append(h("p", { class: `summary ${r.ok ? "ok" : "bad"}` }, h("b", {}, r.summary)));
    if (r.items && r.items.length) {
      this.checkHost.append(h("ul", { class: "qitems" }, r.items.map((it) => h("li", { class: `qi ${it.status}` }, statusIcon(it.status),
        h("span", { class: "qi-name" }, it.name), h("span", { class: "qi-v num" }, `${it.value} ${it.unit}`), it.note && it.status !== "good" ? h("span", { class: "qi-note small muted" }, it.note) : null))));
    }
    if (r.advice && r.advice.length) this.checkHost.append(h("ul", { class: "advice" }, r.advice.map((a) => h("li", {}, icon("info"), a))));
    if (r.face) {
      const [x0, y0, x1, y1] = r.face.box;
      Object.assign(this.faceBox.style, { left: `${(1 - x1) * 100}%`, width: `${(x1 - x0) * 100}%`, top: `${y0 * 100}%`, height: `${(y1 - y0) * 100}%` });
      this.faceBox.hidden = false; this.faceBox.classList.toggle("bad", !r.ok);
    } else this.faceBox.hidden = true;
  }

  // ---------------------------------------------------------------------------------------------- recording
  async countdown() {
    if (this.state !== "live") return;
    this.state = "counting";
    this.sync();
    this.btnRec.hidden = true;
    try { this.lightBox.style.scrollMarginTop = "12px"; this.lightBox.scrollIntoView({ block: "start", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); } catch { /* optional */ }
    this.pf && this.pf.stop();
    this.faceBox.hidden = true;
    for (let i = 3; i >= 1; i--) {
      this.big.textContent = String(i);
      this.big.classList.remove("pop"); void this.big.offsetWidth; this.big.classList.add("pop");
      this.live.textContent = String(i);
      await sleep(900);
      if (this.state !== "counting") return;
    }
    this.big.textContent = "";
    this.record();
  }

  async record() {
    const dur = parseFloat(this.duration.value) || 8;
    const protocol = this.protocol.value;
    const segs = protocol === "baseline_task" ? [{ label: "Still baseline", t0: 0, t1: dur >= 8 ? 4 : 3 }, { label: "Silent counting task", t0: dur >= 8 ? 4 : 3, t1: dur }] : [];
    let start;
    try {
      start = await api.captureStart({ consent: this.consent.checked, label: this.label.value, device: this.cam.settings().label || "", protocol, segments: segs });
    } catch (e) { toast(e.message, "error"); this.state = "live"; this.btnRec.hidden = false; this.sync(); this.pf && this.pf.start(); return; }
    this.capId = start.id;
    this.up = new Uploader(start.id);
    this.state = "recording";
    this.btnStop.hidden = false;
    this.rec.hidden = false;
    this.timing.hidden = false;
    this.cueCard.hidden = protocol !== "baseline_task";
    this.dts = [];
    this.items = 0;
    this.idx = 0;
    this.t0 = null;
    this.lastT = null;
    this.stopAt = null;
    this.thumbs = [];
    this.pending = 0;
    const tap = this.tap;
    const totalFrames = [];
    this.recTimes = totalFrames;
    this.sizeCanvas();
    const resolveEnd = new Promise((res) => { this.endRec = res; });
    tap.start((canvas, t) => {
      if (this.state !== "recording") return;
      if (this.t0 == null) this.t0 = t;
      const rel = t - this.t0;
      if (this.lastT != null && rel <= this.lastT + 1e-6) return;     // duplicate delivery
      if (this.lastT != null) this.dts.push(rel - this.lastT);
      this.lastT = rel;
      const idx = this.idx++;
      totalFrames.push(rel);
      this.pending += 1;
      // copy the frame: the tap's canvas is overwritten by the next one
      const copy = document.createElement("canvas"); copy.width = canvas.width; copy.height = canvas.height;
      copy.getContext("2d", { alpha: false }).drawImage(canvas, 0, 0);
      toBlob(copy, 0.9).then((blob) => { this.pending -= 1; if (blob) { this.up.add({ idx, t: rel, blob }); if (idx % Math.max(1, Math.floor(totalFrames.length / 10)) === 0 && this.thumbs.length < 60) this.thumbs.push({ idx, rel, blob }); } });
      this.tick(rel, dur, segs);
      if (rel >= dur) this.endRec();
    });
    await resolveEnd;
    tap.stop();
    await this.finishRecording();
  }
  sizeCanvas() { const c = this.timing; c.width = Math.max(200, Math.floor(c.parentElement.clientWidth || 600)); }
  tick(rel, dur, segs) {
    this.rec.querySelector("span").textContent = `${rel.toFixed(1)} s / ${dur} s`;
    this.stampR.textContent = `${rel.toFixed(1)} s`;
    if (segs.length) {
      const seg = rel < segs[0].t1 ? 0 : 1;
      if (this.cueCard.dataset.seg !== String(seg)) {
        this.cueCard.dataset.seg = String(seg);
        this.cueCard.replaceChildren(seg === 0 ? h("p", {}, h("b", {}, "Stay still."), " Look at the camera and relax your face.") : h("p", {}, h("b", {}, "Count backwards from 100 in sevens,"), " silently, in your head."));
      }
    }
    // frame-timing ribbon: every tick is one frame interval; tall = late frame
    const c = this.timing, g = c.getContext("2d");
    const w = c.width, hh = c.height;
    g.clearRect(0, 0, w, hh);
    g.fillStyle = "#fbeeea"; g.fillRect(0, 0, w, hh);
    const dts = this.dts, med = dts.length ? [...dts].sort((a, b) => a - b)[dts.length >> 1] : 0.033;
    const n = dts.length, per = w / Math.max(60, Math.min(n, (dur / Math.max(med, 0.01)) * 1.02));
    for (let i = 0; i < n; i++) {
      const late = dts[i] > 1.6 * med;
      const bh = clamp((dts[i] / (med * 2.4)) * (hh - 8), 2, hh - 6);
      g.fillStyle = late ? "#c0301a" : "#13223a";
      g.fillRect(Math.floor(i * per), hh - 3 - bh, Math.max(1, per - 1), bh);
    }
  }
  stopRecording() { if (this.state === "recording" && this.endRec) this.endRec(); }

  async finishRecording() {
    this.state = "uploading";
    this.btnStop.hidden = true; this.rec.hidden = true; this.cueCard.hidden = true;
    this.live.textContent = "Recording finished.";
    this.stampR.textContent = "";
    // wait for the last JPEGs, then send the rest
    for (let i = 0; i < 200 && this.pending > 0; i++) await sleep(25);
    try { await this.up.done(); } catch (e) { return this.fail(`The recording could not be sent to the analyser (${e.message}).`); }
    const n = this.recTimes.length;
    if (n < 24) { await api.remove(this.capId).catch(() => {}); return this.fail("The recording is too short or too few frames arrived. Try again, and stop other programs that use the camera."); }
    this.review(n);
  }
  fail(msg) {
    this.state = "live"; toast(msg, "error");
    this.btnRec.hidden = false; this.sync(); this.pf && this.pf.start();
    this.tap && this.tap.stop();
  }

  // ---------------------------------------------------------------------------------------------- review
  async review(n) {
    this.state = "review";
    const t = this.recTimes, dur = t[t.length - 1];
    const dts = this.dts, med = dts.length ? [...dts].sort((a, b) => a - b)[dts.length >> 1] : 0;
    const late = dts.filter((d) => d > 1.6 * med).length;
    const fps = med ? 1 / med : 0;
    clear(this.formWrap);
    this.goBlock.hidden = true;
    const strip = h("div", { class: "contact", role: "list" });
    const shots = this.thumbs.slice(0, 10);
    for (const s of shots) {
      const cv = h("canvas", { width: 160, height: Math.round(160 * this.video.videoHeight / this.video.videoWidth) || 120 });
      strip.append(h("figure", { role: "listitem" }, cv, h("figcaption", { class: "num" }, `${s.rel.toFixed(1)} s`)));
      createImageBitmap(s.blob, { resizeWidth: 160 }).then((bm) => { cv.getContext("2d").drawImage(bm, 0, 0, cv.width, cv.height); bm.close && bm.close(); }).catch(() => {});
    }
    const warn = [];
    if (late > 0.1 * n) warn.push(`${late} frame intervals were more than 1.6 times the usual length: the camera or computer was struggling. Close other programs and try again if you can.`);
    if (!this.tap.exact) warn.push("This browser could not give exact frame times; timing is approximate.");
    this.formWrap.append(h("section", { class: "cap-form review" },
      h("h2", { class: "label ink" }, "Recorded"),
      h("p", { class: "prose" }, `${n} frames over ${dur.toFixed(1)} seconds — about ${fps.toFixed(1)} frames a second${late ? `, ${late} late` : ", none late"}. Check that the face stayed in view and the light was even.`),
      strip,
      warn.length ? h("ul", { class: "advice" }, warn.map((w) => h("li", {}, icon("warn"), w))) : null,
      h("div", { class: "actions" },
        h("button", { class: "btn pen", type: "button", onclick: () => this.analyse() }, icon("check"), "Analyse this clip"),
        h("button", { class: "btn", type: "button", onclick: () => this.discard() }, icon("trash"), "Discard and record again"))));
  }
  async discard() {
    try { await api.remove(this.capId); } catch { /* ignore */ }
    this.capId = null;
    clear(this.formWrap); this.timing.hidden = true;
    this.build2();
  }
  build2() {
    // re-create the form and return to the live state
    this.formWrap.append(this.formHost);
    this.goBlock.hidden = false;
    this.btnRec.hidden = false; this.state = "live"; this.sync();
    this.pf && this.pf.start();
  }

  // ---------------------------------------------------------------------------------------------- analysis (the developing tray)
  async analyse() {
    this.state = "analysing";
    this.pf && this.pf.stop();
    if (this.cam) this.cam.stop();
    const tray = new Tray(this.box.parentElement);
    this.box.hidden = true;
    try {
      const job = await api.captureFinish(this.capId, this.label.value);
      const done = await pollJob(job.id, (j) => tray.update(j));
      tray.finish();
      await sleep(reduceMotion() ? 0 : 500);
      this.ctx.navigate(`#/bench/session/${done.result_id}`);
    } catch (e) {
      tray.fail(e.message, () => { tray.dispose(); this.box.hidden = false; this.discard(); this.startCamera(); });
    }
  }

  destroy() {
    this.pf && this.pf.stop();
    this.tap && this.tap.stop();
    this.cam && this.cam.stop();
    // footage that was recorded but never sent for analysis is deleted when the person leaves; the server also purges it after an hour
    if (this.capId && this.state !== "analysing") api.remove(this.capId).catch(() => {});
    removeEventListener("pagehide", this.onHide);
    this.capId = null;
    this.state = "gone";
  }
}

// ================================================================================================ the developing tray
class Tray {
  constructor(parent) {
    this.parent = parent;
    this.t0 = performance.now();
    this.trace = [];
    this.canvas = h("canvas", { class: "tray-canvas", "aria-hidden": "true" });
    this.list = h("ol", { class: "procedure" }, STAGES.map((s) => h("li", { dataset: { stage: s } }, h("span", { class: "mark" }), s)));
    this.line = h("p", { class: "prose", role: "status", "aria-live": "polite" }, "Waiting for the analyser");
    this.root = h("section", { class: "tray" },
      h("div", {}, h("p", { class: "kicker" }, "Developing"), h("h2", { class: "display tray-title" }, "Looking at every frame"),
        this.line, this.list, h("p", { class: "legend-text" }, "A clip of this length takes about half a minute. Nothing leaves this computer.")),
      h("div", { class: "tray-paper" }, this.canvas));
    parent.append(this.root);
    this.raf = requestAnimationFrame(() => this.paint());
  }
  update(job) {
    const p = job.progress || 0;
    this.p = p;
    this.trace.push([(performance.now() - this.t0) / 1000, p]);
    this.line.textContent = job.state === "queued" ? `Waiting for the analyser (${job.queue_position || 1} ahead)` : `${job.stage} · ${Math.round(p * 100)} %`;
    const idx = STAGES.indexOf(job.stage);
    [...this.list.children].forEach((li, i) => { li.className = idx < 0 ? "" : i < idx ? "done" : i === idx ? "now" : ""; });
  }
  finish() { this.p = 1; this.trace.push([(performance.now() - this.t0) / 1000, 1]); [...this.list.children].forEach((li) => { li.className = "done"; }); this.line.textContent = "Done."; }
  fail(msg, retry) {
    cancelAnimationFrame(this.raf);
    clear(this.root);
    this.root.append(errorBox(new Error(msg), retry));
  }
  dispose() { cancelAnimationFrame(this.raf); this.root.remove(); }
  paint() {
    const c = this.canvas, dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(240, c.clientWidth), hh = Math.max(160, c.clientHeight);
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(hh * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#fbeeea"; g.fillRect(0, 0, w, hh);
    const now = (performance.now() - this.t0) / 1000, span = Math.max(30, now * 1.15);
    g.lineWidth = 1;
    for (let s = 0; s <= span; s += 1) { const x = (s / span) * (w - 24) + 12; g.strokeStyle = s % 5 === 0 ? "rgba(176,72,58,.40)" : "rgba(196,98,84,.18)"; g.beginPath(); g.moveTo(Math.round(x) + 0.5, 0); g.lineTo(Math.round(x) + 0.5, hh); g.stroke(); }
    for (let k = 0; k <= 10; k++) { const y = hh - 14 - (k / 10) * (hh - 28); g.strokeStyle = k % 5 === 0 ? "rgba(176,72,58,.40)" : "rgba(196,98,84,.18)"; g.beginPath(); g.moveTo(0, Math.round(y) + 0.5); g.lineTo(w, Math.round(y) + 0.5); g.stroke(); }
    g.strokeStyle = "#c0301a"; g.lineWidth = 2; g.lineJoin = "round"; g.beginPath();
    const pts = [[0, 0], ...this.trace, [now, this.p || 0]];
    pts.forEach(([t, p], i) => { const x = (t / span) * (w - 24) + 12, y = hh - 14 - p * (hh - 28); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
    g.stroke();
    g.fillStyle = "#4f5a66"; g.font = "500 11.5px Archivo, sans-serif"; g.textAlign = "right";
    g.fillText(`${now.toFixed(0)} s`, w - 10, hh - 2);
    g.textAlign = "left"; g.fillText("100 %", 14, 14); g.fillText("progress, against seconds elapsed", 14, hh - 2);
    this.raf = requestAnimationFrame(() => this.paint());
  }
}

// ================================================================================================ upload
function uploadMode(box, ctx, limits, privacy) {
  let file = null;
  const input = h("input", { type: "file", accept: "video/*,.mp4,.mov,.m4v,.webm,.avi,.mkv", class: "visually-hidden", id: "file", onchange: () => pick(input.files[0]) });
  const drop = h("label", { class: "drop", for: "file", tabindex: "0" }, icon("upload"), h("b", {}, "Choose a video, or drop it here"), h("span", { class: "small muted" }, `mp4, mov, webm, avi or mkv · up to ${limits.max_upload_mb || 256} MB · ${limits.min_clip_seconds || 2}–${limits.max_clip_seconds || 30} seconds`));
  const preview = h("video", { class: "up-video", controls: "", muted: "", playsinline: "", hidden: true });
  const info = h("p", { class: "small muted num" });
  const label = h("input", { type: "text", id: "ulabel", maxlength: 60, placeholder: "Name (optional)" });
  const consent = h("input", { type: "checkbox", id: "uconsent", onchange: () => sync() });
  const go = h("button", { class: "btn pen", type: "button", disabled: true, onclick: () => send() }, icon("check"), "Analyse");
  const status = h("div", { class: "up-status" });
  const sync = () => { go.disabled = !(file && consent.checked); };
  const pick = (f) => {
    if (!f) return;
    file = f;
    drop.querySelector("b").textContent = f.name;
    preview.hidden = false; preview.src = URL.createObjectURL(f);
    preview.addEventListener("loadedmetadata", () => { info.textContent = `${preview.videoWidth} × ${preview.videoHeight} · ${preview.duration.toFixed(1)} s · ${(f.size / 1048576).toFixed(1)} MB`; }, { once: true });
    if (!label.value) label.value = f.name.replace(/\.[^.]+$/, "").slice(0, 60);
    sync();
  };
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); pick(e.dataTransfer.files[0]); });
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  async function send() {
    go.disabled = true;
    clear(status);
    const tray = new Tray(box.parentElement);
    box.hidden = true;
    try {
      const job = await api.upload(file, label.value, consent.checked);
      const done = await pollJob(job.id, (j) => tray.update(j));
      tray.finish();
      await sleep(reduceMotion() ? 0 : 500);
      ctx.navigate(`#/bench/session/${done.result_id}`);
    } catch (e) {
      tray.fail(e.message, () => { tray.dispose(); box.hidden = false; go.disabled = false; sync(); });
    }
  }
  box.append(
    h("div", { class: "cap-left" }, input, drop, h("div", { class: "plate up-plate", hidden: false }, preview), info),
    h("div", { class: "cap-right" },
      h("section", { class: "sec first" },
        h("header", {}, h("h2", { class: "label ink" }, "What makes a good file")),
        h("ul", { class: "advice" },
          ["A single face, facing the camera, filling a good part of the picture (at least 80 pixels between the eyes).",
            "Thirty frames a second or more; faster is better for short movements. Heavily compressed or re-encoded video hides small movements.",
            "Even light, no strong backlight; the head reasonably still.",
            "The report grades the recording and says how small a movement it could show, so a poor file is reported as such."].map((t) => h("li", {}, icon("info"), t))))),
    h("div", { class: "cap-bottom" }, h("section", { class: "cap-form" },
      h("div", { class: "form-grid" }, h("div", { class: "field" }, h("label", { for: "ulabel" }, "Name"), label)),
      h("label", { class: "check consent" }, consent, h("span", {}, "I have the right to analyse this video, and the person in it has agreed. I understand it is analysed on this computer and the file is deleted when the analysis ends; grayscale face crops for the report stay for 24 hours or until I delete them.")),
      privacy, h("div", { class: "actions" }, go), status)));
}
