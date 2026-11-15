// Chronophotograph strip: the selected movement laid out in time, like a contact sheet.
//   row 1  the stabilised plates as recorded
//   row 2  the same instants magnified (a visual aid; flagged as such)
//   row 3  where the face moved: the apex pattern scaled by the measured time-course of the movement
import { h, svg, clamp, fmt } from "./util.js";

const N = 8;

export class Chrono {
  constructor(host, { result, sprites, bus }) {
    this.r = result;
    this.sprites = sprites;
    this.bus = bus;
    this.ev = null;
    this.cells = [];
    this.root = h("div", { class: "chrono" });
    this.head = h("div", { class: "chrono-head" });
    this.rows = h("div", { class: "chrono-rows" });
    this.note = h("p", { class: "legend-text" });
    this.root.append(this.head, this.rows, this.note);
    host.append(this.root);
    this.bus.on("select", ({ id }) => this.setEvent(this.r.events.find((e) => e.id === id) || null));
    this.bus.on("time", ({ t }) => this.mark(t));
    this.bus.on("seek", ({ t }) => this.mark(t));
    this.setEvent(null);
  }

  times(ev) {
    const t0 = ev.t_onset - 0.1;
    const t1 = (ev.t_offset != null ? ev.t_offset : ev.t_apex + 0.3) + 0.1;
    const ts = Array.from({ length: N }, (_, i) => t0 + ((t1 - t0) * i) / (N - 1));
    let k = 0;
    for (let i = 1; i < N; i++) if (Math.abs(ts[i] - ev.t_apex) < Math.abs(ts[k] - ev.t_apex)) k = i;
    ts[k] = ev.t_apex;
    const lo = this.sprites.times[0], hi = this.sprites.times[this.sprites.times.length - 1];
    return { ts: ts.map((t) => clamp(t, lo, hi)), apex: k };
  }

  envelope(ev, t) {
    const tr = ev.traces;
    if (!tr || !tr.t || !tr.t.length) return 0;
    const a = tr.amp_pct_iod, ts = tr.t;
    let peak = 0;
    for (const v of a) peak = Math.max(peak, v);
    if (peak <= 0) return 0;
    if (t <= ts[0]) return 0;
    if (t >= ts[ts.length - 1]) return clamp(a[a.length - 1] / peak, 0, 1.2);
    let i = 1;
    while (i < ts.length && ts[i] < t) i++;
    const f = (t - ts[i - 1]) / Math.max(ts[i] - ts[i - 1], 1e-6);
    return clamp((a[i - 1] + f * (a[i] - a[i - 1])) / peak, 0, 1.2);
  }

  setEvent(ev) {
    this.ev = ev;
    this.cells = [];
    this.rows.replaceChildren();
    this.head.replaceChildren();
    if (!ev || !this.sprites.sheets.length) {
      this.head.append(h("p", { class: "muted" }, "Select a movement to lay it out in time."));
      this.note.textContent = "";
      return;
    }
    const { ts, apex } = this.times(ev);
    this.sampleTimes = ts;
    this.head.append(h("h3", { class: "label ink" }, "The movement in time"), h("span", { class: "small muted num" }, `${ts.length} instants from ${fmt.s(ts[0])} to ${fmt.s(ts[ts.length - 1])}`));
    const mk = (title, sub) => {
      const row = h("div", { class: "chrono-row" });
      row.append(h("div", { class: "chrono-title" }, h("span", { class: "label ink" }, title), sub ? h("span", { class: "small muted" }, sub) : null));
      const strip = h("div", { class: "film", role: "list" });
      row.append(strip);
      this.rows.append(row);
      return strip;
    };
    const r1 = mk("As recorded", "stabilised plates");
    const r2 = mk(ev.magnified ? `Magnified ×${ev.magnified.alpha}` : "Magnified", ev.magnified ? "a visual aid" : "not made for this movement");
    const r3 = mk("Where it moved", "apex pattern × measured time-course");
    const hasMag = !!ev.magnified;
    ts.forEach((t, i) => {
      const k = this.sprites.index(t);
      const label = i === apex ? "apex" : `${t - ev.t_apex >= 0 ? "+" : "−"}${Math.round(Math.abs(t - ev.t_apex) * 1000)} ms`;
      const cell1 = this.cell(r1, i, apex, label, t);
      const c1 = h("canvas", { width: 176, height: 176, "aria-hidden": "true" });
      cell1.btn.prepend(c1);
      const cell2 = this.cell(r2, i, apex, label, t, true);
      const c2 = h("canvas", { width: 176, height: 176, "aria-hidden": "true" });
      cell2.btn.prepend(c2);
      const cell3 = this.cell(r3, i, apex, label, t, true);
      const sv = svg("svg", { viewBox: `0 0 ${this.r.canon.plate_size} ${this.r.canon.plate_size}`, "aria-hidden": "true", class: "mini-arrows" });
      cell3.btn.prepend(sv);
      this.cells.push({ t, k, c1, c2, sv, els: [cell1.el, cell2.el, cell3.el], hasMag });
    });
    this.paint();
    this.sprites.loadEvent(ev).then(() => this.ev === ev && this.paint());
    this.note.textContent = ev.magnified
      ? `Magnification multiplies the measured displacement by ${1 + ev.magnified.alpha}, with the measurement noise suppressed; it helps the eye and is not a measurement. The third row draws the pattern measured at the apex, scaled frame by frame by how far the movement had developed.`
      : "The third row draws the pattern measured at the apex, scaled frame by frame by how far the movement had developed.";
  }

  cell(strip, i, apex, label, t, dim) {
    const btn = h("button", { type: "button", class: "frame-btn", "aria-label": `Show ${label}`, onclick: () => { this.bus.emit("seek", { t, from: this }); this.mark(t); } });
    const cap = h("span", { class: `frame-cap num${i === apex ? " apex" : ""}` }, label);
    const el = h("div", { class: `frame${i === apex ? " is-apex" : ""}`, role: "listitem" }, btn, cap);
    strip.append(el);
    return { el, btn };
  }

  paint() {
    const ev = this.ev;
    if (!ev) return;
    const box = this.r.canon.plate_box, kk = this.r.canon.plate_size / (box[2] - box[0]);
    const cen = Object.fromEntries(this.r.canon.centroids.map((c) => [c.roi, c]));
    const rows = (ev.ledger || []).filter((l) => l.used && cen[l.roi]);
    let maxLen = 0;
    for (const l of rows) maxLen = Math.max(maxLen, Math.hypot(l.dx_px, l.dy_px) * kk);
    const gain = clamp(15 / Math.max(maxLen, 1e-3), 3, 90);
    for (const c of this.cells) {
      const g1 = c.c1.getContext("2d");
      const cell = this.sprites.cell(c.k);
      g1.drawImage(cell.img, cell.sx, cell.sy, cell.size, cell.size, 0, 0, 176, 176);
      const g2 = c.c2.getContext("2d");
      const m = this.sprites.magCell(ev, c.t);
      if (m) g2.drawImage(m.img, m.sx, m.sy, m.size, m.size, 0, 0, 176, 176);
      else { g2.fillStyle = "#0e1013"; g2.fillRect(0, 0, 176, 176); g2.fillStyle = "#8d98a5"; g2.font = "500 13px Archivo, sans-serif"; g2.textAlign = "center"; g2.fillText(ev.magnified ? "outside window" : "n/a", 88, 90); }
      while (c.sv.firstChild) c.sv.removeChild(c.sv.firstChild);
      c.sv.append(svg("rect", { x: 0, y: 0, width: this.r.canon.plate_size, height: this.r.canon.plate_size, fill: "#0e1013" }));
      const env = this.envelope(ev, c.t);
      for (const l of rows) {
        const cc = cen[l.roi];
        const dx = l.dx_px * kk * gain * env, dy = l.dy_px * kk * gain * env, len = Math.hypot(dx, dy);
        c.sv.append(svg("circle", { cx: cc.x, cy: cc.y, r: 0.8, fill: "#6f7c8a" }));
        if (len < 0.35) continue;
        const ux = dx / len, uy = dy / len, hx = cc.x + dx, hy = cc.y + dy, s = Math.min(2.6, 0.35 * len + 0.8);
        c.sv.append(svg("g", { stroke: "#f3f6fa", fill: "#f3f6fa", "stroke-width": 0.8, "stroke-linecap": "round" },
          svg("line", { x1: cc.x, y1: cc.y, x2: hx - ux * s * 0.6, y2: hy - uy * s * 0.6 }),
          svg("path", { d: `M${hx},${hy}L${hx - ux * s - uy * s * 0.55},${hy - uy * s + ux * s * 0.55}L${hx - ux * s + uy * s * 0.55},${hy - uy * s - ux * s * 0.55}Z`, stroke: "none" })));
      }
    }
  }

  mark(t) {
    if (!this.cells.length) return;
    let best = 0;
    this.cells.forEach((c, i) => { if (Math.abs(c.t - t) < Math.abs(this.cells[best].t - t)) best = i; });
    const near = Math.abs(this.cells[best].t - t) < (this.cells[1].t - this.cells[0].t) * 0.6;
    this.cells.forEach((c, i) => c.els.forEach((el) => el.classList.toggle("now", near && i === best)));
  }
}
