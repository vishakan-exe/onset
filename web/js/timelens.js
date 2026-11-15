// The Time Lens: a multichannel strip-chart on chart paper.
// The paper's grid carries meaning: a vertical line every 100 ms (a micro-expression is at most five of the heavy
// squares wide, 500 ms), and in the detection lane the heavy horizontal line IS the spotting threshold.
import { h, clamp, fmt, icon, tip, withTable, GROUPS, GROUP_LABEL, STATUS_WORD } from "./util.js";

const INK = "#13223a", GRAPHITE = "#4f5a66", PEN = "#c0301a", PAPER = "#fbeeea";
const MINOR = "rgba(196,98,84,0.20)", MAJOR = "rgba(176,72,58,0.40)";

export function eventLabel(e) {
  if (e.named_aus && e.named_aus.length) return e.named_aus.map((a) => `AU${a}`).join(" + ");
  return STATUS_WORD[e.status] || e.status;
}
const isExpr = (e) => e.status === "micro" || e.status === "expression" || e.status === "unexplained" || e.status === "sustained";

export class TimeLens {
  constructor(host, { result, bus, compact = false }) {
    this.r = result;
    this.bus = bus;
    this.tl = result.timeline;
    this.events = result.events;
    this.t = this.tl.t;
    this.n = this.t.length;
    this.T0 = this.t[0];
    this.T1 = this.t[this.n - 1];
    this.view = [this.T0, this.T1];
    this.cur = this.T0;
    this.sel = null;
    this.hover = null;
    this.drag = false;
    this.compact = compact;
    this.fps = result.clip.fps || 30;
    this.marks = [];                  // extra annotation bands: {t0, t1, label, color}
    this.build();
    this.bus.on("time", ({ t, from }) => { if (from !== this) { this.cur = t; this.followPlayhead(); this.draw(); } });
    this.bus.on("seek", ({ t, from }) => { if (from !== this) { this.cur = t; this.draw(); } });
    this.bus.on("select", ({ id }) => { this.sel = this.events.find((e) => e.id === id) || null; this.announce(); this.draw(); });
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(this.paper);
  }

  // ------------------------------------------------------------------------------------------------ DOM
  build() {
    this.canvas = h("canvas", { tabindex: "0", role: "application", "aria-label": "Timeline of facial movement by region. Left and right arrows move the playhead by one frame, shift with the arrows jumps between detected movements, plus and minus zoom.", style: { width: "100%", display: "block", cursor: "crosshair", touchAction: "none" } });
    this.paper = h("div", { class: "lens-paper" }, this.canvas);
    this.live = h("p", { class: "visually-hidden", "aria-live": "polite" });
    const btn = (label, fn, ic) => h("button", { class: "btn quiet small", type: "button", onclick: fn, "aria-label": label }, ic ? icon(ic) : null, ic ? null : label);
    this.controls = h("div", { class: "lens-controls" },
      btn("Whole clip", () => this.zoomTo(this.T0, this.T1)),
      btn("Selected movement", () => this.zoomEvent()),
      btn("Zoom in", () => this.zoomBy(0.6)), btn("Zoom out", () => this.zoomBy(1 / 0.6)));
    // table twin
    const cols = ["Movement", "Kind", "Onset (s)", "Apex (s)", "Offset (s)", "Duration (ms)", "Detection score"];
    const rows = this.events.map((e) => [`${e.id}: ${eventLabel(e)}`, STATUS_WORD[e.status] || e.status, e.t_onset.toFixed(2), e.t_apex.toFixed(2), e.t_offset == null ? "not observed" : e.t_offset.toFixed(2), e.duration_ms == null ? "n/a" : Math.round(e.duration_ms), e.spot.score.toFixed(1)]);
    const tw = withTable(this.paper, cols, rows, "Show movements as a table");
    this.tableBtn = tw.button;
    this.host = h("div", { class: "lens" }, this.controls, this.paper, this.live, tw.table, h("div", { class: "lens-foot" }, this.legend(), this.tableBtn));
    this.canvas.addEventListener("pointerdown", (e) => this.down(e));
    this.canvas.addEventListener("pointermove", (e) => this.move(e));
    this.canvas.addEventListener("pointerup", () => { this.drag = false; });
    this.canvas.addEventListener("pointercancel", () => { this.drag = false; });
    this.canvas.addEventListener("pointerleave", () => { this.hover = null; tip.hide(); this.draw(); });
    this.canvas.addEventListener("keydown", (e) => this.key(e));
    this.canvas.addEventListener("wheel", (e) => { if (e.ctrlKey || e.metaKey) { e.preventDefault(); this.zoomBy(e.deltaY > 0 ? 1.25 : 0.8, this.xToT(this.localX(e))); } }, { passive: false });
  }
  legend() {
    const key = (col, txt, dashed) => h("span", { class: "lens-key" }, h("i", { style: { background: dashed ? "transparent" : col, borderTop: dashed ? `2px dashed ${col}` : "none" } }), txt);
    return h("div", { class: "lens-legend small muted" },
      key(INK, "Movement measured"), key(PEN, "Detection threshold"), key(PEN, "Detected movement", false),
      h("span", {}, "Vertical scale: units of typical noise (grey band). Grid: 100 ms across, heavy lines every 500 ms."));
  }
  mount(parent) { parent.append(this.host); this.resize(); return this; }

  // ------------------------------------------------------------------------------------------------ geometry
  layout() {
    const cmp = this.compact || this.W < 560;
    const H = { strip: 26, score: cmp ? 52 : 70, group: cmp ? 34 : 42, ribbon: 11, head: cmp ? 30 : 38, axis: 28, gap: 8 };
    this.strip = H.strip;
    let y = H.strip;
    const lanes = [];
    lanes.push({ key: "score", label: "Detection", top: y, h: H.score }); y += H.score + H.gap;
    for (const g of GROUPS) { lanes.push({ key: g, label: GROUP_LABEL[g], top: y, h: H.group }); y += H.group + 3; }
    y += H.gap - 3;
    lanes.push({ key: "blink", label: "Blinks", top: y, h: H.ribbon }); y += H.ribbon + 4;
    lanes.push({ key: "speech", label: "Speech", top: y, h: H.ribbon }); y += H.ribbon + H.gap;
    lanes.push({ key: "head", label: "Head shift", top: y, h: H.head }); y += H.head + 4;
    this.axisTop = y;
    this.H = y + H.axis;
    this.lanes = lanes;
    this.gl = cmp ? 70 : 104;
    this.gr = 14;
  }
  resize() {
    const w = Math.max(280, Math.floor(this.paper.clientWidth || 600));
    this.W = w;
    this.layout();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(this.H * dpr);
    this.canvas.style.height = `${this.H}px`;
    this.dpr = dpr;
    this.prepare();
    this.draw();
  }
  prepare() {
    const p = (a, q) => { const s = Float64Array.from(a).sort(); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };
    this.yScore = clamp(p(this.tl.score, 0.995) * 1.1, 8, 16);
    let gmax = 0;
    for (const g of GROUPS) gmax = Math.max(gmax, p(this.tl.groups[g], 0.995));
    this.yGroup = clamp(gmax * 1.1, 5, 14);
    const hx = this.tl.head_x, hy = this.tl.head_y;
    const mx = hx.reduce((a, b) => a + b, 0) / hx.length, my = hy.reduce((a, b) => a + b, 0) / hy.length;
    this.headMag = hx.map((v, i) => Math.hypot(v - mx, hy[i] - my));
    this.yHead = clamp(p(this.headMag, 0.995) * 1.15, 0.5, 8);
  }
  tToX(t) { return this.gl + ((t - this.view[0]) / (this.view[1] - this.view[0])) * (this.W - this.gl - this.gr); }
  xToT(x) { return this.view[0] + ((x - this.gl) / (this.W - this.gl - this.gr)) * (this.view[1] - this.view[0]); }
  localX(e) { return e.clientX - this.canvas.getBoundingClientRect().left; }
  localY(e) { return e.clientY - this.canvas.getBoundingClientRect().top; }

  // ------------------------------------------------------------------------------------------------ interaction
  down(e) {
    this.canvas.focus({ preventScroll: true });
    const x = this.localX(e), y = this.localY(e);
    if (x < this.gl) return;
    const hit = this.eventAt(x, y);
    if (hit) { this.bus.emit("select", { id: hit.id }); this.bus.emit("seek", { t: hit.t_apex, from: this }); this.cur = hit.t_apex; this.draw(); return; }
    this.drag = true;
    this.canvas.setPointerCapture(e.pointerId);
    this.scrub(x);
  }
  move(e) {
    const x = this.localX(e), y = this.localY(e);
    if (this.drag) { this.scrub(x); return; }
    if (x < this.gl || x > this.W - this.gr) { this.hover = null; tip.hide(); this.draw(); return; }
    this.hover = this.xToT(x);
    this.draw();
    this.showTip(e, x, y);
  }
  scrub(x) {
    const t = clamp(this.xToT(x), this.T0, this.T1);
    this.cur = t;
    this.bus.emit("seek", { t, from: this });
    this.draw();
  }
  eventAt(x, y) {
    const top = this.lanes[0].top;
    if (y > top + 18) {
      // bands below: only the score lane's bracket row selects
      return null;
    }
    const t = this.xToT(x);
    let best = null, bd = 1e9;
    for (const ev of this.events) {
      const a = ev.t_onset - 0.04, b = (ev.t_offset != null ? ev.t_offset : ev.t_apex + 0.3) + 0.04;
      if (t >= a && t <= b) { const d = Math.abs(t - ev.t_apex); if (d < bd) { best = ev; bd = d; } }
    }
    return best;
  }
  key(e) {
    const step = 1 / this.fps;
    const jump = (dir) => {
      const ts = this.events.filter(isExpr).map((x) => x.t_apex).sort((a, b) => a - b);
      const next = dir > 0 ? ts.find((v) => v > this.cur + 0.02) : [...ts].reverse().find((v) => v < this.cur - 0.02);
      if (next != null) { this.cur = next; const ev = this.events.find((x) => x.t_apex === next); if (ev) this.bus.emit("select", { id: ev.id }); this.bus.emit("seek", { t: next, from: this }); this.followPlayhead(); this.draw(); }
    };
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      const d = e.key === "ArrowRight" ? 1 : -1;
      if (e.shiftKey) jump(d);
      else { this.cur = clamp(this.cur + d * step, this.T0, this.T1); this.bus.emit("seek", { t: this.cur, from: this }); this.followPlayhead(); this.draw(); }
    } else if (e.key === "+" || e.key === "=") { e.preventDefault(); this.zoomBy(0.6); }
    else if (e.key === "-") { e.preventDefault(); this.zoomBy(1 / 0.6); }
    else if (e.key === "0") { this.zoomTo(this.T0, this.T1); }
    else if (e.key === "Home") { this.cur = this.T0; this.bus.emit("seek", { t: this.cur, from: this }); this.draw(); }
    else if (e.key === "End") { this.cur = this.T1; this.bus.emit("seek", { t: this.cur, from: this }); this.draw(); }
  }
  announce() {
    const e = this.sel;
    this.live.textContent = e ? `Selected ${e.id}: ${eventLabel(e)}, apex at ${e.t_apex.toFixed(2)} seconds.` : "";
  }
  zoomTo(a, b) {
    const min = 0.8;
    if (b - a < min) { const c = (a + b) / 2; a = c - min / 2; b = c + min / 2; }
    a = Math.max(this.T0, a); b = Math.min(this.T1, b);
    this.view = [a, b];
    this.draw();
  }
  zoomBy(f, around) {
    const c = around != null ? around : (this.view[0] + this.view[1]) / 2;
    const a = c - (c - this.view[0]) * f, b = c + (this.view[1] - c) * f;
    this.zoomTo(a, b);
  }
  zoomEvent() {
    const e = this.sel;
    if (!e) return;
    this.zoomTo(e.t_onset - 0.6, (e.t_offset != null ? e.t_offset : e.t_apex + 0.4) + 0.6);
  }
  followPlayhead() {
    const [a, b] = this.view;
    if (this.cur < a || this.cur > b) { const w = b - a; this.view = [clamp(this.cur - 0.2 * w, this.T0, this.T1 - w), clamp(this.cur - 0.2 * w, this.T0, this.T1 - w) + w]; }
  }
  showTip(e, x, y) {
    const t = this.hover;
    const i = this.nearest(t);
    const rows = [{ value: this.tl.score[i].toFixed(1), label: "strongest match to any movement (spotting threshold " + this.tl.threshold + ")" }];
    for (const g of GROUPS) rows.push({ value: this.tl.groups[g][i].toFixed(1), label: `${GROUP_LABEL[g]} activity (noise level about ${(this.tl.group_noise ? this.tl.group_noise[g] : 1.5).toFixed(1)})` });
    if (this.tl.blink[i] > 0.3) rows.push("Eyes closing or closed");
    if (this.tl.speech[i] > 0.3) rows.push("Mouth moving (speech)");
    rows.push({ value: this.headMag[i].toFixed(2) + " %", label: "head shift, of the eye distance" });
    tip.show({ x: e.clientX, y: e.clientY }, fmt.s(this.t[i]), rows);
  }
  nearest(t) {
    let lo = 0, hi = this.n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (this.t[m] <= t) lo = m; else hi = m; }
    return Math.abs(this.t[lo] - t) <= Math.abs(this.t[hi] - t) ? lo : hi;
  }

  // ------------------------------------------------------------------------------------------------ drawing
  draw() {
    if (!this.W) return;
    const c = this.canvas.getContext("2d");
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, this.W, this.H);
    c.fillStyle = PAPER;
    c.fillRect(0, 0, this.W, this.H);
    try { c.fontStretch = "semi-condensed"; } catch { /* older browsers */ }
    const [v0, v1] = this.view;
    const x0 = this.gl, x1 = this.W - this.gr;
    // vertical grid
    const span = v1 - v0, pxPer100 = ((x1 - x0) * 0.1) / span;
    c.lineWidth = 1;
    const start = Math.ceil(v0 * 10 - 1e-9);
    for (let k = start; k / 10 <= v1 + 1e-9; k++) {
      const major = k % 5 === 0;
      if (!major && pxPer100 < 5) continue;
      const x = Math.round(this.tToX(k / 10)) + 0.5;
      c.strokeStyle = major ? MAJOR : MINOR;
      c.beginPath(); c.moveTo(x, 4); c.lineTo(x, this.axisTop); c.stroke();
    }
    // lanes and events are clipped to the plot, so a trace never runs into the label column
    c.save();
    c.beginPath(); c.rect(x0, 0, x1 - x0, this.axisTop + 1); c.clip();
    for (const L of this.lanes) this.lane(c, L);
    this.eventBands(c);
    c.restore();
    for (const L of this.lanes) this.laneTitle(c, L);
    // axis (the playhead's own readout sits on it, so the tick labels it would half-cover are left out)
    this.axis(c, this.cur >= v0 && this.cur <= v1 ? this.tToX(this.cur) : null);
    // hover + playhead
    if (this.hover != null) { const x = Math.round(this.tToX(this.hover)) + 0.5; c.strokeStyle = GRAPHITE; c.lineWidth = 1; c.beginPath(); c.moveTo(x, 4); c.lineTo(x, this.axisTop); c.stroke(); }
    if (this.cur >= v0 && this.cur <= v1) {
      const x = Math.round(this.tToX(this.cur)) + 0.5;
      c.strokeStyle = INK; c.lineWidth = 1.5; c.beginPath(); c.moveTo(x, 2); c.lineTo(x, this.axisTop + 5); c.stroke();
      // the playhead reads out its own time on the axis, over whichever tick label it covers
      const txt = this.cur.toFixed(2);
      c.font = "600 11.5px Archivo, sans-serif";
      const w = c.measureText(txt).width + 12, bx = clamp(x - w / 2, x0 - 2, x1 + 2 - w);
      c.fillStyle = INK; c.beginPath(); c.roundRect ? c.roundRect(bx, this.axisTop + 6, w, 18, 3) : c.rect(bx, this.axisTop + 6, w, 18); c.fill();
      c.fillStyle = PAPER; c.textAlign = "center"; c.textBaseline = "middle"; c.fillText(txt, bx + w / 2, this.axisTop + 15.5);
    }
  }
  laneTitle(c, L) {
    const sub = L.key === "score" ? "strongest match" : L.key === "head" ? "% of eye distance" : null;
    c.fillStyle = INK; c.font = "560 12.5px Archivo, sans-serif"; c.textAlign = "left"; c.textBaseline = "middle";
    c.fillText(L.label, 10, L.top + Math.min(11, L.h / 2));
    if (sub) { c.fillStyle = GRAPHITE; c.font = "500 11px Archivo, sans-serif"; c.fillText(sub, 10, L.top + Math.min(11, L.h / 2) + 14); }
  }
  series(c, L, values, ymax, opts = {}) {
    const [v0, v1] = this.view;
    const base = L.top + L.h;
    c.beginPath();
    let started = false, lastX = 0;
    for (let i = 0; i < this.n; i++) {
      const t = this.t[i];
      if (t < v0 - 0.2 || t > v1 + 0.2) continue;
      const x = this.tToX(t), y = base - clamp(values[i] / ymax, 0, 1.04) * L.h;
      if (!started) { c.moveTo(x, y); started = true; } else c.lineTo(x, y);
      lastX = x;
    }
    if (!started) return;
    c.lineWidth = opts.width || 1.5; c.lineJoin = "round"; c.lineCap = "round"; c.strokeStyle = opts.color || INK; c.stroke();
    if (opts.wash) {
      c.lineTo(lastX, base); c.lineTo(this.tToX(Math.max(v0 - 0.2, this.T0)), base); c.closePath();
      c.fillStyle = opts.wash; c.fill();
    }
  }
  lane(c, L) {
    const x0 = this.gl, x1 = this.W - this.gr;
    if (L.key === "score") {
      this.hlines(c, L, this.yScore, [0, 5, 10, 15].filter((v) => v < this.yScore), this.tl.threshold);
      this.series(c, L, this.tl.score, this.yScore, { wash: "rgba(19,34,58,0.08)", width: 1.7 });
      const ty = L.top + L.h - (this.tl.threshold / this.yScore) * L.h;
      c.strokeStyle = PEN; c.lineWidth = 1.5; c.beginPath(); c.moveTo(x0, ty); c.lineTo(x1, ty); c.stroke();
      c.font = "600 11px Archivo, sans-serif"; c.textAlign = "right"; c.textBaseline = "bottom";
      const tl = `threshold ${this.tl.threshold}`;
      c.lineWidth = 3; c.strokeStyle = PAPER; c.lineJoin = "round"; c.strokeText(tl, x1 - 4, ty - 2);
      c.fillStyle = PEN; c.fillText(tl, x1 - 4, ty - 2);
    } else if (GROUPS.includes(L.key)) {
      const noise = (this.tl.group_noise && this.tl.group_noise[L.key]) || 1.5;
      c.fillStyle = "rgba(19,34,58,0.045)";
      c.fillRect(x0, L.top + L.h - clamp((noise * 1.5) / this.yGroup, 0, 1) * L.h, x1 - x0, clamp((noise * 1.5) / this.yGroup, 0, 1) * L.h);
      this.hlines(c, L, this.yGroup, [0, this.yGroup / 2]);
      this.series(c, L, this.tl.groups[L.key], this.yGroup, { width: 1.4 });
    } else if (L.key === "blink" || L.key === "speech") {
      const wins = this.r.nuisance[L.key === "blink" ? "blink_windows" : "speech_windows"] || [];
      for (const [a, b] of wins) {
        const xa = this.tToX(a), xb = Math.max(this.tToX(b), xa + 2);
        c.fillStyle = "rgba(19,34,58,0.55)";
        c.fillRect(xa, L.top, xb - xa, L.h);
      }
      c.strokeStyle = "rgba(19,34,58,0.25)"; c.lineWidth = 1; c.strokeRect(x0 + 0.5, L.top + 0.5, x1 - x0 - 1, L.h - 1);
    } else if (L.key === "head") {
      this.hlines(c, L, this.yHead, [0]);
      this.series(c, L, this.headMag, this.yHead, { color: GRAPHITE, width: 1.3 });
    }
  }
  hlines(c, L, ymax, ticks, heavy) {
    const x0 = this.gl, x1 = this.W - this.gr;
    for (const v of ticks) {
      const y = Math.round(L.top + L.h - (v / ymax) * L.h) + 0.5;
      c.strokeStyle = v === heavy ? MAJOR : (v === 0 ? "rgba(19,34,58,0.45)" : MINOR);
      c.lineWidth = 1; c.beginPath(); c.moveTo(x0, y); c.lineTo(x1, y); c.stroke();
    }
    if (ticks.length > 1) {
      c.font = "500 10.5px Archivo, sans-serif"; c.textAlign = "left"; c.textBaseline = "bottom";
      const top = ticks[ticks.length - 1];
      if (top > 0) {
        const lab = String(Math.round(top * 10) / 10), ty = L.top + L.h - (top / ymax) * L.h - 2;
        c.lineWidth = 3; c.strokeStyle = PAPER; c.lineJoin = "round"; c.strokeText(lab, x0 + 4, ty);
        c.fillStyle = GRAPHITE; c.fillText(lab, x0 + 4, ty);
      }
    }
  }
  eventBands(c) {
    const [v0, v1] = this.view;
    const top = 4, bottom = this.axisTop;
    for (const e of this.events) {
      const off = e.t_offset != null ? e.t_offset : e.t_apex + 0.3;
      if (off < v0 || e.t_onset > v1) continue;
      const xa = this.tToX(e.t_onset), xb = Math.max(this.tToX(off), xa + 3);
      const sel = this.sel && this.sel.id === e.id;
      const expr = isExpr(e);
      c.fillStyle = expr ? `rgba(192,48,26,${sel ? 0.17 : 0.07})` : `rgba(19,34,58,${sel ? 0.12 : 0.045})`;
      c.fillRect(xa, top, xb - xa, bottom - top);
      // bracket in the detection lane
      const y = this.strip - 8;
      c.strokeStyle = expr ? PEN : GRAPHITE; c.lineWidth = sel ? 2.2 : 1.6;
      c.beginPath(); c.moveTo(xa, y + 5); c.lineTo(xa, y); c.lineTo(xb, y); c.lineTo(xb, y + 5);
      if (e.t_offset == null) { c.setLineDash([3, 3]); }
      c.stroke(); c.setLineDash([]);
      // apex tick
      const xp = this.tToX(e.t_apex);
      c.beginPath(); c.moveTo(xp, y - 3); c.lineTo(xp, y + 5); c.stroke();
      if (xb - xa > 18 || sel) {
        c.fillStyle = expr ? PEN : GRAPHITE; c.font = `${sel ? 650 : 580} 11.5px Archivo, sans-serif`; c.textAlign = "left"; c.textBaseline = "bottom";
        const lab = eventLabel(e);
        const w = c.measureText(lab).width;
        const lx = Math.max(this.gl + 2, Math.min(xa + 2, this.W - this.gr - w));
        c.fillText(lab, lx, y - 3);
      }
    }
    for (const m of this.marks) {
      const xa = this.tToX(m.t0), xb = this.tToX(m.t1);
      c.fillStyle = m.color || "rgba(29,95,168,0.10)";
      c.fillRect(xa, top, xb - xa, bottom - top);
      if (m.label) { c.fillStyle = INK; c.font = "600 11.5px Archivo, sans-serif"; c.textAlign = "left"; c.textBaseline = "bottom"; c.fillText(m.label, xa + 4, bottom - 2); }
    }
  }
  axis(c, skipX = null) {
    const [v0, v1] = this.view;
    const span = v1 - v0;
    const pxPerS = (this.W - this.gl - this.gr) / Math.max(span, 1e-6);
    const step = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20].find((st) => st * pxPerS >= 46) || 20;       // labels never closer than ~46 px
    const dec = step >= 1 ? 0 : Math.abs(step * 10 - Math.round(step * 10)) < 1e-9 ? 1 : 2;
    c.fillStyle = GRAPHITE; c.font = "500 11.5px Archivo, sans-serif"; c.textAlign = "center"; c.textBaseline = "top";
    c.strokeStyle = "rgba(19,34,58,0.5)"; c.lineWidth = 1;
    c.beginPath(); c.moveTo(this.gl, this.axisTop + 0.5); c.lineTo(this.W - this.gr, this.axisTop + 0.5); c.stroke();
    for (let t = Math.ceil(v0 / step) * step; t <= v1 + 1e-9; t += step) {
      const x = this.tToX(t);
      c.beginPath(); c.moveTo(Math.round(x) + 0.5, this.axisTop); c.lineTo(Math.round(x) + 0.5, this.axisTop + 5); c.stroke();
      if (skipX == null || Math.abs(x - skipX) >= 34) c.fillText(t.toFixed(dec), x, this.axisTop + 8);
    }
    c.textAlign = "right"; c.fillText("seconds", this.gl - 8, this.axisTop + 8);
  }
  destroy() { this.ro.disconnect(); tip.hide(); }
}
