// The plate player: stabilised grayscale face crops from the analysis' sprite sheets, with an SVG overlay for
// regions and displacement arrows.  Nothing here recomputes measurements; it only draws what the analysis stored.
import { h, svg, clamp, fmt, icon, roiLabel, tip } from "./util.js";
import { fileUrl } from "./api.js";

function loadImage(url) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error(`image failed: ${url}`));
    im.src = url;
  });
}

export class Sprites {
  constructor(result, ref) {
    this.meta = result.sprites;
    this.ref = ref;
    this.sheets = [];
    this.mags = new Map();
    this.heats = new Map();
    this.times = Float64Array.from(this.meta.times);
  }
  async load() {
    this.sheets = await Promise.all(this.meta.sheets.map((s) => loadImage(fileUrl(this.ref, s.file))));
  }
  async loadEvent(ev) {
    if (!ev.magnified || this.mags.has(ev.id)) return;
    const m = ev.magnified;
    try {
      const [im, heat] = await Promise.all([loadImage(fileUrl(this.ref, m.sheet)), m.heat ? loadImage(fileUrl(this.ref, m.heat)) : null]);
      this.mags.set(ev.id, im);
      if (heat) this.heats.set(ev.id, heat);
    } catch { this.mags.set(ev.id, null); }
  }
  index(t) {
    const a = this.times;
    let lo = 0, hi = a.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (a[m] <= t) lo = m; else hi = m; }
    return Math.abs(a[lo] - t) <= Math.abs(a[hi] - t) ? lo : hi;
  }
  cell(k) {
    const per = this.meta.per_sheet, cols = this.meta.cols, size = this.meta.size;
    const sheet = this.sheets[Math.floor(k / per)];
    const pos = k % per;
    return { img: sheet, sx: (pos % cols) * size, sy: Math.floor(pos / cols) * size, size };
  }
  magCell(ev, t) {
    const m = ev.magnified, img = this.mags.get(ev.id);
    if (!m || !img) return null;
    const ts = m.times;
    if (t < ts[0] - 0.05 || t > ts[ts.length - 1] + 0.05) return null;
    let best = 0;
    for (let i = 1; i < ts.length; i++) if (Math.abs(ts[i] - t) < Math.abs(ts[best] - t)) best = i;
    return { img, sx: (best % m.cols) * m.size, sy: Math.floor(best / m.cols) * m.size, size: m.size };
  }
}

const ARROW = {
  measured: { color: "#f3f6fa", label: "Measured" },
  explained: { color: "#7fb6f4", label: "Explained by the named AUs" },
  residual: { color: "#ff8f7a", label: "Left over" },
};

export class PlatePlayer {
  /**
   * opts: {result, ref, bus, selected(): event|null, mode, onReady}
   * The bus carries: seek {t}, time {t}, select {id}, play {playing}.
   */
  constructor(host, opts) {
    this.host = host;
    this.r = opts.result;
    this.ref = opts.ref;
    this.bus = opts.bus;
    this.sprites = new Sprites(this.r, this.ref);
    this.t = 0;
    this.playing = false;
    this.speed = 1;
    this.mode = "plate";
    this.show = { regions: false, measured: true, explained: false, residual: false };
    this.loop = true;
    this.sel = null;
    this.highlightRoi = null;
    this.events = this.r.events;
    this.canon = this.r.canon;
    this.lastTick = 0;
    this.build();
    this.bus.on("seek", ({ t, from }) => { if (from !== this) { this.t = t; this.draw(); } });
    this.bus.on("select", ({ id }) => this.select(id));
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(this.plate);
    this.sprites.load().then(() => { this.ready = true; this.t = this.sprites.times[0] || 0; this.resize(); this.bus.emit("ready", {}); });
  }

  // ---------------------------------------------------------------------------------------------- DOM
  build() {
    const c = this.canon;
    this.canvas = h("canvas", { "aria-hidden": "true" });
    this.overlay = svg("svg", { class: "overlay", viewBox: `0 0 ${c.plate_size} ${c.plate_size}`, "aria-hidden": "true" });
    this.regionsG = svg("g", { class: "regions", fill: "none" });
    this.arrowsG = svg("g", { class: "arrows" });
    this.overlay.append(this.regionsG, this.arrowsG);
    this.stampMode = h("span", { class: "stamp tl" }, "Stabilised plate");
    this.stampTime = h("span", { class: "stamp tr num" }, "0.00 s");
    const ibox = c.plate_box;
    this.iodPlate = (c.iod * c.plate_size) / (ibox[2] - ibox[0]);
    this.scale = h("div", { class: "scale-bar" }, h("span", { class: "num" }, "10 % of the eye distance"), h("i", {}));
    this.plate = h("div", { class: "plate" }, this.canvas, this.overlay, this.stampMode, this.stampTime, this.scale);
    this.wrap = h("div", { class: "plate-wrap" }, h("div", { class: "crop" }, this.plate));
    this.wrap.tabIndex = 0;
    this.wrap.setAttribute("role", "group");
    this.wrap.setAttribute("aria-label", "Face plate. Space plays or pauses, arrow keys step one frame.");
    this.wrap.addEventListener("keydown", (e) => this.key(e));

    // transport
    this.btnPlay = h("button", { class: "btn icon primary", type: "button", "aria-label": "Play", onclick: () => this.toggle() }, icon("play"));
    const step = (d) => h("button", { class: "btn icon", type: "button", "aria-label": d < 0 ? "Previous frame" : "Next frame", onclick: () => this.step(d) }, icon(d < 0 ? "prev" : "next"));
    this.speedSel = h("select", { "aria-label": "Playback speed", style: { width: "auto", minHeight: "38px" }, onchange: (e) => { this.speed = parseFloat(e.target.value); } },
      [[1, "1×"], [0.5, "½×"], [0.25, "¼×"], [0.1, "¹⁄₁₀×"]].map(([v, l]) => h("option", { value: v }, l)));
    this.modeSeg = h("div", { class: "seg", role: "group", "aria-label": "View" }, [
      ["plate", "Plate"], ["magnified", "Magnified"], ["motion", "Motion"]].map(([k, l]) =>
      h("button", { type: "button", "aria-pressed": String(k === "plate"), dataset: { mode: k }, onclick: () => this.setMode(k) }, l)));
    this.overlaySeg = h("div", { class: "seg", role: "group", "aria-label": "Overlays" }, [
      ["regions", "Regions"], ["measured", "Arrows"], ["explained", "Explained"], ["residual", "Left over"]].map(([k, l]) =>
      h("button", { type: "button", "aria-pressed": String(!!this.show[k]), dataset: { ov: k }, onclick: () => this.toggleOverlay(k) }, l)));
    this.transport = h("div", { class: "transport" }, step(-1), this.btnPlay, step(1), this.speedSel, this.modeSeg);
    this.transport2 = h("div", { class: "transport", style: { marginTop: "10px" } }, this.overlaySeg);
    this.caption = h("p", { class: "legend-text plate-caption" }, "");
    this.host.append(this.wrap, this.transport, this.transport2, this.caption);
    this.drawRegions();
  }

  /** Hide or show the instrument's own controls (view modes, overlays, caption): the first look begins without them. */
  setInstrument(on) { this.modeSeg.hidden = !on; this.transport2.hidden = !on; this.caption.hidden = !on; }

  resize() {
    const r = this.plate.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const px = Math.max(64, Math.round(r.width * dpr));
    if (this.canvas.width !== px) { this.canvas.width = px; this.canvas.height = px; }
    this.draw();
  }

  // ---------------------------------------------------------------------------------------------- state
  select(id) {
    this.sel = this.events.find((e) => e.id === id) || null;
    if (this.sel) this.sprites.loadEvent(this.sel).then(() => this.draw());
    this.updateModes();
    this.drawArrows();
    this.draw();
  }
  updateModes() {
    const has = !!(this.sel && this.sel.magnified);
    for (const b of this.modeSeg.querySelectorAll("button")) {
      if (b.dataset.mode !== "plate") { b.disabled = !has; b.title = has ? "" : "Select a brief movement to see it magnified"; }
    }
    if (!has && this.mode !== "plate") this.setMode("plate");
  }
  setMode(m) {
    this.mode = m;
    for (const b of this.modeSeg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.mode === m));
    this.draw();
  }
  toggleOverlay(k) {
    this.show[k] = !this.show[k];
    for (const b of this.overlaySeg.querySelectorAll("button")) b.setAttribute("aria-pressed", String(!!this.show[b.dataset.ov]));
    this.drawRegions();
    this.drawArrows();
  }
  toggle() { this.playing ? this.pause() : this.play(); }
  play() {
    if (!this.ready) return;
    const [a, b] = this.range();
    if (this.t >= b - 1e-3) this.t = a;
    this.playing = true;
    this.lastTick = performance.now();
    this.btnPlay.replaceChildren(icon("pause"));
    this.btnPlay.setAttribute("aria-label", "Pause");
    this.bus.emit("play", { playing: true });
    requestAnimationFrame((n) => this.tick(n));
  }
  pause() {
    this.playing = false;
    this.btnPlay.replaceChildren(icon("play"));
    this.btnPlay.setAttribute("aria-label", "Play");
    this.bus.emit("play", { playing: false });
  }
  range() {
    const tt = this.sprites.times;
    if (this.sel && this.loop && this.playing !== "all") {
      const e = this.sel;
      const off = e.t_offset != null ? e.t_offset : e.t_apex + 0.4;
      return [Math.max(tt[0], e.t_onset - 0.35), Math.min(tt[tt.length - 1], off + 0.35)];
    }
    return [tt[0], tt[tt.length - 1]];
  }
  tick(now) {
    if (!this.playing) return;
    const dt = ((now - this.lastTick) / 1000) * this.speed;
    this.lastTick = now;
    const [a, b] = this.range();
    this.t += dt;
    if (this.t > b) { if (this.sel && this.loop) this.t = a; else { this.t = b; this.pause(); } }
    if (this.t < a - 0.4 && this.sel && this.loop) this.t = a;
    this.bus.emit("time", { t: this.t, from: this });
    this.draw();
    requestAnimationFrame((n) => this.tick(n));
  }
  step(d) {
    this.pause();
    const k = clamp(this.sprites.index(this.t) + d, 0, this.sprites.times.length - 1);
    this.t = this.sprites.times[k];
    this.bus.emit("time", { t: this.t, from: this });
    this.draw();
  }
  key(e) {
    if (e.key === " ") { e.preventDefault(); this.toggle(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); this.step(-1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); this.step(1); }
  }

  // ---------------------------------------------------------------------------------------------- drawing
  draw() {
    if (!this.ready) return;
    const ctx = this.canvas.getContext("2d");
    const W = this.canvas.width;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    const k = this.sprites.index(this.t);
    let cell = this.sprites.cell(k);
    let label = "Stabilised plate";
    if (this.mode !== "plate" && this.sel) {
      const m = this.sprites.magCell(this.sel, this.t);
      if (m && this.mode === "magnified") { cell = m; label = `Magnified ×${this.sel.magnified.alpha} (a visual aid; the numbers come from the measurements)`; }
      else if (this.mode === "motion") label = "Motion at the apex";
      else if (this.mode === "magnified") label = "Magnified: outside the event window";
    }
    ctx.drawImage(cell.img, cell.sx, cell.sy, cell.size, cell.size, 0, 0, W, W);
    if (this.mode === "motion" && this.sel) {
      const heat = this.sprites.heats.get(this.sel.id);
      if (heat) { const near = Math.abs(this.t - this.sel.t_apex) < 0.45 ? 1 : 0.35; ctx.globalAlpha = near; ctx.drawImage(heat, 0, 0, W, W); ctx.globalAlpha = 1; }
    }
    this.stampMode.textContent = label.length > 44 ? "Magnified" + ` ×${this.sel ? this.sel.magnified.alpha : ""}` : label;
    this.stampMode.title = label;
    this.stampTime.textContent = fmt.s(this.sprites.times[k]);
    this.updateScale();
    this.updateCaption();
    if (this.show.measured || this.show.explained || this.show.residual) this.drawArrows(true);
  }
  updateScale() {
    const css = this.plate.clientWidth || 1;
    const px10 = 0.10 * this.iodPlate * (css / this.canon.plate_size);
    this.scale.querySelector("i").style.width = `${px10.toFixed(1)}px`;
  }
  updateCaption() {
    const e = this.sel;
    if (!e) { this.caption.textContent = "Pick a movement on the timeline to see where and how the face moved."; return; }
    const inside = this.t >= e.t_onset - 0.02 && this.t <= (e.t_offset != null ? e.t_offset : e.t_apex + 0.4);
    this.caption.textContent = `${e.status_text}. ${inside ? "The movement is in progress." : "Outside the selected movement."}`;
  }
  drawRegions() {
    while (this.regionsG.firstChild) this.regionsG.removeChild(this.regionsG.firstChild);
    if (!this.show.regions) return;
    for (const [name, polys] of Object.entries(this.canon.rois)) {
      for (const poly of polys) {
        const d = poly.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join("") + "Z";
        const path = svg("path", { d, stroke: "#c9d6e4", "stroke-width": 0.5, "stroke-opacity": 0.7, fill: "rgba(255,255,255,0.001)", "data-roi": name, style: "pointer-events:all" });
        path.addEventListener("pointermove", (ev) => this.roiTip(ev, name));
        path.addEventListener("pointerleave", () => { tip.hide(); this.emphasise(null); });
        this.regionsG.append(path);
      }
    }
  }
  roiTip(ev, name) {
    this.emphasise(name);
    const row = this.sel && this.sel.ledger ? this.sel.ledger.find((l) => l.roi === name) : null;
    const rows = row && row.used ? [{ value: `${row.mag_pct_iod.toFixed(2)} %`, label: `of the eye distance, ${row.direction}` }, { value: row.z.toFixed(1), label: "times the noise level" }] : ["No measurement for this region in the selected movement"];
    tip.show({ x: ev.clientX, y: ev.clientY }, roiLabel(name), rows);
  }
  emphasise(name) {
    this.highlightRoi = name;
    for (const p of this.regionsG.querySelectorAll("path")) {
      const on = name && p.dataset.roi === name;
      p.setAttribute("stroke", on ? "#ffd27a" : "#c9d6e4");
      p.setAttribute("stroke-width", on ? 1.2 : 0.5);
      p.setAttribute("stroke-opacity", on ? 1 : 0.7);
    }
  }
  /** Arrow length follows the event's amplitude trace in time so the arrows breathe with the movement. */
  envelope(e) {
    const tr = e.traces;
    if (!tr || !tr.t || !tr.t.length) return 1;
    const a = tr.amp_pct_iod, t = tr.t;
    let peak = 0;
    for (const v of a) peak = Math.max(peak, v);
    if (peak <= 0) return 1;
    if (this.t <= t[0]) return 0;
    if (this.t >= t[t.length - 1]) return clamp(a[a.length - 1] / peak, 0, 1.2);
    let i = 1;
    while (i < t.length && t[i] < this.t) i++;
    const f = (this.t - t[i - 1]) / Math.max(t[i] - t[i - 1], 1e-6);
    return clamp((a[i - 1] + f * (a[i] - a[i - 1])) / peak, 0, 1.2);
  }
  drawArrows(fast = false) {
    const g = this.arrowsG;
    while (g.firstChild) g.removeChild(g.firstChild);
    const e = this.sel;
    if (!e || !e.ledger || !e.ledger.length) return;
    const box = this.canon.plate_box, k = this.canon.plate_size / (box[2] - box[0]);
    const cen = Object.fromEntries(this.canon.centroids.map((c) => [c.roi, c]));
    const rows = e.ledger.filter((l) => l.used && cen[l.roi]);
    let maxLen = 0;
    for (const l of rows) maxLen = Math.max(maxLen, Math.hypot(l.dx_px, l.dy_px) * k);
    const gain = clamp(15 / Math.max(maxLen, 1e-3), 3, 90) * this.envelope(e);
    this.gain = clamp(15 / Math.max(maxLen, 1e-3), 3, 90);
    const kinds = [["residual", (l) => [l.resid_dx, l.resid_dy]], ["explained", (l) => [l.explained_dx, l.explained_dy]], ["measured", (l) => [l.dx_px, l.dy_px]]];
    for (const [kind, vec] of kinds) {
      if (!this.show[kind]) continue;
      const col = ARROW[kind].color;
      for (const l of rows) {
        const [vx, vy] = vec(l);
        const c = cen[l.roi];
        const dx = vx * k * gain, dy = vy * k * gain, len = Math.hypot(dx, dy);
        if (len < 0.35) continue;
        const ux = dx / len, uy = dy / len, hx = c.x + dx, hy = c.y + dy, s = Math.min(2.6, 0.35 * len + 0.8);
        const hl = this.highlightRoi === l.roi;
        g.append(svg("g", { stroke: col, fill: col, "stroke-width": hl ? 1.2 : 0.8, "stroke-linecap": "round", opacity: kind === "measured" ? 0.95 : 0.9 },
          svg("circle", { cx: c.x, cy: c.y, r: 0.9, stroke: "none" }),
          svg("line", { x1: c.x, y1: c.y, x2: hx - ux * s * 0.6, y2: hy - uy * s * 0.6 }),
          svg("path", { d: `M${hx},${hy}L${hx - ux * s - uy * s * 0.55},${hy - uy * s + ux * s * 0.55}L${hx - ux * s + uy * s * 0.55},${hy - uy * s - ux * s * 0.55}Z`, stroke: "none" })));
      }
    }
  }
  setHighlight(roi) { this.emphasise(roi); this.drawArrows(); }
  destroy() { this.pause(); this.ro.disconnect(); }
}
