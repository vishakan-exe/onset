// Chart primitives shared by the Cue Lab, the Evidence page and the Atlas: hairline grid, 2 px lines, 8 px+ markers
// with a surface ring, a legend whenever there is more than one series, a crosshair tooltip that follows the keyboard
// as well as the pointer, and a table twin for every chart.  Text is always drawn in text colours; the series colour
// lives on the mark.
import { h, svg, clear, clamp, tip, withTable } from "./util.js";

export const SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"];
const SURFACE = "var(--mat-2)";

/** An SVG that is rebuilt at its real pixel width whenever its container resizes, so text keeps its real size. */
export function fit(host, height, draw) {
  const el = svg("svg", { role: "img", class: "chart-svg" });
  host.append(el);
  const run = () => {
    const w = Math.max(260, Math.floor(host.clientWidth || 320));
    el.setAttribute("viewBox", `0 0 ${w} ${height}`);
    el.setAttribute("width", w);
    el.setAttribute("height", height);
    clear(el);
    draw(el, w, height);
  };
  const ro = new ResizeObserver(run);
  ro.observe(host);
  run();
  return { el, run, destroy: () => ro.disconnect() };
}

export function linear(d0, d1, r0, r1) {
  const f = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  f.invert = (p) => d0 + ((p - r0) / (r1 - r0 || 1)) * (d1 - d0);
  f.domain = [d0, d1];
  return f;
}
export function logScale(d0, d1, r0, r1) {
  const a = Math.log(d0), b = Math.log(d1);
  const f = (v) => r0 + ((Math.log(Math.max(v, d0 * 1e-6)) - a) / (b - a)) * (r1 - r0);
  f.invert = (p) => Math.exp(a + ((p - r0) / (r1 - r0)) * (b - a));
  f.domain = [d0, d1];
  f.log = true;
  return f;
}
export function niceTicks(lo, hi, n = 5) {
  const span = hi - lo, raw = span / Math.max(1, n);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.round(v / step) * step);
  return out;
}
export function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }
  return out;
}
export const path = (pts) => pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
export const fmtNum = (v) => (Math.abs(v) >= 1000 ? `${v / 1000}K` : Math.abs(v) >= 10 || Number.isInteger(v) ? String(Math.round(v * 100) / 100) : String(Math.round(v * 100) / 100));

function textEl(x, y, s, cls, anchor = "start", baseline = "auto") {
  const t = svg("text", { x, y, class: cls, "text-anchor": anchor, "dominant-baseline": baseline });
  t.textContent = s;
  return t;
}

/** Gridlines, ticks and axis titles for a plot area.  m = {l, r, t, b} margins. */
export function axes(el, { w, hh, m, xs, ys, xTicks, yTicks, xFmt = fmtNum, yFmt = fmtNum, xTitle, yTitle, grid = true }) {
  const x0 = m.l, x1 = w - m.r, y0 = m.t, y1 = hh - m.b;
  // an explicit tick outside the plotted domain would be drawn outside the plot: keep only what the scale can place inside it
  const inX = (v) => { const q = xs(v); return Number.isFinite(q) && q >= x0 - 0.5 && q <= x1 + 0.5; };
  const inY = (v) => { const q = ys(v); return Number.isFinite(q) && q >= Math.min(y0, y1) - 0.5 && q <= Math.max(y0, y1) + 0.5; };
  xTicks = xTicks.filter(inX); yTicks = yTicks.filter(inY);
  const g = svg("g", { class: "axes" });
  if (grid) for (const v of yTicks) g.append(svg("line", { x1: x0, x2: x1, y1: ys(v), y2: ys(v), class: "grid" }));
  g.append(svg("line", { x1: x0, x2: x1, y1: y1, y2: y1, class: "axis" }), svg("line", { x1: x0, x2: x0, y1: y0, y2: y1, class: "axis" }));
  for (const v of xTicks) { g.append(svg("line", { x1: xs(v), x2: xs(v), y1: y1, y2: y1 + 4, class: "axis" }), textEl(xs(v), y1 + 17, xFmt(v), "tick", "middle")); }
  for (const v of yTicks) g.append(textEl(x0 - 7, ys(v), yFmt(v), "tick", "end", "middle"));
  if (xTitle) g.append(textEl((x0 + x1) / 2, hh - 4, xTitle, "axis-title", "middle"));
  if (yTitle) { const t = textEl(0, 0, yTitle, "axis-title", "middle"); t.setAttribute("transform", `translate(12 ${(y0 + y1) / 2}) rotate(-90)`); g.append(t); }
  el.append(g);
  return g;
}

function legend(series) {
  if (series.length < 2) return null;
  return h("ul", { class: "legend", "aria-label": "Legend" }, series.map((s, i) => h("li", {}, h("i", { class: s.dash ? "dash" : "", style: { background: s.dash ? "transparent" : s.color || SERIES[i], borderTopColor: s.color || SERIES[i] } }), s.name)));
}

/**
 * A line chart.  opts: {height, margin, x:{domain, ticks, format, title, log}, y:{domain, ticks, format, title},
 * series:[{name, color, points:[[x,y]], width, dash, band:[[x,lo,hi]], end:label}], dots:[{x,y,label,color,series}],
 * rules:[{y|x, label}], caption, table:{columns, rows}, readout(xv) -> [{series,name,value,color}] | null}
 */
export function lineChart(host, o) {
  const height = o.height || 260;
  const m = Object.assign({ l: 50, r: 22, t: 14, b: 38 }, o.margin || {});
  const wrap = h("figure", { class: "chartfig" });
  if (o.title) wrap.append(h("figcaption", { class: "label ink" }, o.title));
  const lg = legend(o.series);
  if (lg) wrap.append(lg);
  const box = h("div", { class: "chart" });
  wrap.append(box);
  if (o.caption) wrap.append(h("p", { class: "legend-text" }, o.caption));
  host.append(wrap);
  let table = null;
  if (o.table) {
    const tw = withTable(box, o.table.columns, o.table.rows);
    table = tw;
    wrap.append(h("div", { class: "chart-foot" }, tw.button), tw.table);
  }
  const xdom = o.x.domain, ydom = o.y.domain;
  const chart = fit(box, height, (el, w, hh) => {
    const x0 = m.l, x1 = w - m.r, y0 = m.t, y1 = hh - m.b;
    const xs = (o.x.log ? logScale : linear)(xdom[0], xdom[1], x0, x1);
    const ys = linear(ydom[0], ydom[1], y1, y0);
    const xTicks = o.x.ticks || (o.x.log ? logTicks(xdom[0], xdom[1]) : niceTicks(xdom[0], xdom[1], Math.max(3, Math.floor((x1 - x0) / 80))));
    const yTicks = o.y.ticks || niceTicks(ydom[0], ydom[1], 4);
    axes(el, { w, hh, m, xs, ys, xTicks, yTicks, xFmt: o.x.format, yFmt: o.y.format, xTitle: o.x.title, yTitle: o.y.title });
    el.setAttribute("aria-label", o.aria || o.title || "Chart");
    for (const r of o.rules || []) {
      if (r.y != null) { el.append(svg("line", { x1: x0, x2: x1, y1: ys(r.y), y2: ys(r.y), class: "rule" })); if (r.label) el.append(textEl(x1 - 4, ys(r.y) - 5, r.label, "lab-ink", "end")); }
      if (r.x != null) { el.append(svg("line", { x1: xs(r.x), x2: xs(r.x), y1: y0, y2: y1, class: "rule" })); if (r.label) el.append(textEl(xs(r.x) + 5, y0 + 11, r.label, "lab-ink")); }
    }
    o.series.forEach((s, i) => {
      const col = s.color || SERIES[i];
      if (s.band && s.band.length) {
        const up = s.band.map(([x, lo, hi]) => [xs(x), ys(hi)]), dn = s.band.map(([x, lo]) => [xs(x), ys(lo)]).reverse();
        el.append(svg("path", { d: path(up) + path(dn).replace(/^M/, "L") + "Z", fill: col, opacity: 0.12 }));
      }
      const pts = s.points.filter((p) => p[1] != null && isFinite(p[1])).map(([x, y]) => [xs(x), ys(clamp(y, ydom[0], ydom[1]))]);
      el.append(svg("path", { d: path(pts), fill: "none", stroke: col, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash || null }));
      if (s.end && pts.length) { const [ex, ey] = pts[pts.length - 1]; el.append(textEl(Math.min(ex + 6, w - 2), ey, s.end, "lab-ink", ex + 6 > w - 40 ? "end" : "start", "middle")); }
    });
    for (const d of o.dots || []) {
      const cx = xs(d.x), cy = ys(clamp(d.y, ydom[0], ydom[1]));
      el.append(svg("circle", { cx, cy, r: 5, fill: d.color || "var(--ink)", stroke: SURFACE, "stroke-width": 2 }));
      if (d.label) el.append(textEl(clamp(cx + 9, x0 + 2, x1 - 2), cy + (d.below ? 20 : -9), d.label, "lab-ink", cx > x1 - 110 ? "end" : "start"));   // below: for a rising curve, the free side
    }
    // hover layer: crosshair snaps to the nearest x of the first series; the readout lists every series there
    if (o.readout || o.series[0]) {
      const cross = svg("line", { y1: y0, y2: y1, class: "cross", visibility: "hidden" });
      const pin = svg("g", { visibility: "hidden" });
      const hit = svg("rect", { x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: "transparent", tabindex: 0, role: "application", "aria-label": `${o.title || "Chart"}. Use left and right arrow keys to read values.` });
      el.append(cross, pin, hit);
      const base = o.series[0].points;
      let idx = 0;
      const show = (i, at) => {
        idx = clamp(i, 0, base.length - 1);
        const xv = base[idx][0];
        cross.setAttribute("x1", xs(xv)); cross.setAttribute("x2", xs(xv)); cross.setAttribute("visibility", "visible");
        clear(pin);
        const rows = o.readout ? o.readout(xv, idx) : o.series.map((s, k) => { const p = s.points[Math.min(idx, s.points.length - 1)]; return p && p[1] != null ? { key: s.color || SERIES[k], value: (o.y.format || fmtNum)(p[1]), label: s.name } : null; }).filter(Boolean);
        o.series.forEach((s, k) => { const p = s.points.find((q) => q[0] === xv) || s.points[Math.min(idx, s.points.length - 1)]; if (p && p[1] != null && isFinite(p[1])) pin.append(svg("circle", { cx: xs(p[0]), cy: ys(clamp(p[1], ydom[0], ydom[1])), r: 4.5, fill: s.color || SERIES[k], stroke: SURFACE, "stroke-width": 2 })); });
        pin.setAttribute("visibility", "visible");
        const r = el.getBoundingClientRect();
        tip.show(at || { x: r.left + xs(xv), y: r.top + y0 + 20 }, `${o.x.title ? o.x.title + " " : ""}${(o.x.format || fmtNum)(xv)}`, rows);
      };
      const nearest = (clientX) => { const r = el.getBoundingClientRect(), v = xs.invert(clientX - r.left); let best = 0; for (let i = 1; i < base.length; i++) if (Math.abs(base[i][0] - v) < Math.abs(base[best][0] - v)) best = i; return best; };
      hit.addEventListener("pointermove", (e) => show(nearest(e.clientX), { x: e.clientX, y: e.clientY }));
      hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); pin.setAttribute("visibility", "hidden"); tip.hide(); });
      hit.addEventListener("keydown", (e) => { if (e.key === "ArrowRight") { e.preventDefault(); show(idx + Math.max(1, Math.round(base.length / 40))); } else if (e.key === "ArrowLeft") { e.preventDefault(); show(idx - Math.max(1, Math.round(base.length / 40))); } });
      hit.addEventListener("blur", () => { cross.setAttribute("visibility", "hidden"); pin.setAttribute("visibility", "hidden"); tip.hide(); });
    }
  });
  return { figure: wrap, chart, destroy: () => chart.destroy() };
}

/**
 * Horizontal interval rows ("forest" plot): each row has a label, a point estimate with an interval, and a trailing
 * value.  rows: [{label, est, lo, hi, text, note, key}], x: {domain, log, ticks, format, title, ref}
 */
export function forest(host, o) {
  const rowH = o.rowHeight || 30;
  const m = Object.assign({ l: o.labelWidth || 190, r: 70, t: 8, b: 36 }, o.margin || {});
  const height = m.t + m.b + rowH * o.rows.length;
  const wrap = h("figure", { class: "chartfig" });
  if (o.title) wrap.append(h("figcaption", { class: "label ink" }, o.title));
  const box = h("div", { class: "chart" });
  wrap.append(box);
  if (o.caption) wrap.append(h("p", { class: "legend-text" }, o.caption));
  host.append(wrap);
  if (o.table) { const tw = withTable(box, o.table.columns, o.table.rows); wrap.append(h("div", { class: "chart-foot" }, tw.button), tw.table); }
  const chart = fit(box, height, (el, w, hh) => {
    const narrow = w < 520;                                            // phone: rows that offer a short label use it, in a narrower label column, so the plot keeps room
    const useShort = narrow && o.rows.every((r) => r.short);
    const x0 = useShort ? Math.min(m.l, Math.max(92, Math.round(w * 0.3))) : m.l, x1 = w - m.r;
    const xs = (o.x.log ? logScale : linear)(o.x.domain[0], o.x.domain[1], x0, x1);
    const lo = Math.min(...o.x.domain), hi = Math.max(...o.x.domain);
    const ticks = (o.x.ticks || (o.x.log ? logTicks(o.x.domain[0], o.x.domain[1]) : niceTicks(o.x.domain[0], o.x.domain[1], 5))).filter((v) => v >= lo - 1e-9 && v <= hi + 1e-9);
    el.setAttribute("aria-label", o.aria || o.title || "Intervals");
    for (const v of ticks) { el.append(svg("line", { x1: xs(v), x2: xs(v), y1: m.t, y2: hh - m.b, class: "grid" }), textEl(xs(v), hh - m.b + 16, (o.x.format || fmtNum)(v), "tick", "middle")); }
    el.append(svg("line", { x1: x0, x2: x1, y1: hh - m.b, y2: hh - m.b, class: "axis" }));
    if (o.x.title) el.append(textEl((x0 + x1) / 2, hh - 4, narrow && o.x.titleShort ? o.x.titleShort : o.x.title, "axis-title", "middle"));
    if (o.x.ref != null) el.append(svg("line", { x1: xs(o.x.ref), x2: xs(o.x.ref), y1: m.t, y2: hh - m.b, class: "rule" }));
    o.rows.forEach((r, i) => {
      const y = m.t + rowH * i + rowH / 2;
      const g = svg("g", { class: "frow", tabindex: 0, role: "img", "aria-label": `${r.label}: ${r.text || ""}` });
      const lab = textEl(10, y, useShort ? r.short : r.label, r.dim ? "lab-dim" : "lab-ink", "start", "middle");
      g.append(svg("rect", { x: 0, y: y - rowH / 2, width: w, height: rowH, fill: "transparent" }), lab);
      if (r.est != null && isFinite(r.est)) {
        const lo = r.lo != null && isFinite(r.lo) ? r.lo : o.x.domain[0], hi = r.hi != null && isFinite(r.hi) ? r.hi : o.x.domain[1];
        const a = xs(clamp(lo, o.x.domain[0], o.x.domain[1])), b = xs(clamp(hi, o.x.domain[0], o.x.domain[1]));
        g.append(svg("line", { x1: a, x2: b, y1: y, y2: y, stroke: r.color || "var(--ink)", "stroke-width": 2, "stroke-linecap": "round", opacity: r.dim ? 0.5 : 1 }));
        if (!isFinite(r.hi)) g.append(svg("path", { d: `M${b - 5},${y - 4}L${b},${y}L${b - 5},${y + 4}`, fill: "none", stroke: r.color || "var(--ink)", "stroke-width": 2 }));
        g.append(svg("circle", { cx: xs(clamp(r.est, o.x.domain[0], o.x.domain[1])), cy: y, r: 4.5, fill: r.color || "var(--ink)", stroke: SURFACE, "stroke-width": 2 }));
      }
      if (r.text) g.append(textEl(w - 6, y, r.text, r.strong ? "lab-strong" : "lab-ink", "end", "middle"));
      const showTip = (ev) => { const bb = el.getBoundingClientRect(); tip.show(ev && ev.clientX != null ? { x: ev.clientX, y: ev.clientY } : { x: bb.left + x0, y: bb.top + y }, r.label, r.rows || (r.note ? [r.note] : [])); };
      g.addEventListener("pointermove", showTip); g.addEventListener("focus", () => showTip(null));
      g.addEventListener("pointerleave", () => tip.hide()); g.addEventListener("blur", () => tip.hide());
      el.append(g);
    });
  });
  return { figure: wrap, chart, destroy: () => chart.destroy() };
}

/** Horizontal bars (one series, one colour): rows [{label, value, text, note}], x domain [0, max]. */
export function bars(host, o) {
  const rowH = o.rowHeight || 28, bh = Math.min(16, rowH - 10);
  const m = Object.assign({ l: o.labelWidth || 170, r: 64, t: 4, b: o.axis === false ? 4 : 30 }, o.margin || {});
  const height = m.t + m.b + rowH * o.rows.length;
  const wrap = h("figure", { class: "chartfig" });
  if (o.title) wrap.append(h("figcaption", { class: "label ink" }, o.title));
  const box = h("div", { class: "chart" });
  wrap.append(box);
  if (o.caption) wrap.append(h("p", { class: "legend-text" }, o.caption));
  host.append(wrap);
  if (o.table) { const tw = withTable(box, o.table.columns, o.table.rows); wrap.append(h("div", { class: "chart-foot" }, tw.button), tw.table); }
  const chart = fit(box, height, (el, w, hh) => {
    const x0 = m.l, x1 = w - m.r;
    const xs = linear(o.x.domain[0], o.x.domain[1], x0, x1);
    el.setAttribute("aria-label", o.aria || o.title || "Bars");
    if (o.axis !== false) {
      const ticks = (o.x.ticks || niceTicks(o.x.domain[0], o.x.domain[1], 5)).filter((v) => v >= o.x.domain[0] - 1e-9 && v <= o.x.domain[1] + 1e-9);
      for (const v of ticks) { el.append(svg("line", { x1: xs(v), x2: xs(v), y1: m.t, y2: hh - m.b, class: "grid" }), textEl(xs(v), hh - m.b + 16, (o.x.format || fmtNum)(v), "tick", "middle")); }
      el.append(svg("line", { x1: x0, x2: x0, y1: m.t, y2: hh - m.b, class: "axis" }));
      if (o.x.title) el.append(textEl((x0 + x1) / 2, hh - 4, o.x.title, "axis-title", "middle"));
    }
    o.rows.forEach((r, i) => {
      const y = m.t + rowH * i + rowH / 2;
      const g = svg("g", { class: "brow", tabindex: 0, role: "img", "aria-label": `${r.label}: ${r.text || r.value}` });
      const len = Math.max(0, xs(clamp(r.value, o.x.domain[0], o.x.domain[1])) - x0);
      g.append(svg("rect", { x: 0, y: y - rowH / 2, width: w, height: rowH, fill: "transparent" }), textEl(10, y, r.label, "lab-ink", "start", "middle"));
      if (len > 0.5) g.append(svg("path", { d: `M${x0},${y - bh / 2}H${x0 + Math.max(0, len - 4)}a4,4 0 0 1 4,4V${y + bh / 2 - 4}a4,4 0 0 1 -4,4H${x0}Z`, fill: r.color || o.color || "var(--s1)" }));
      g.append(textEl(x0 + len + 8, y, r.text != null ? r.text : fmtNum(r.value), "lab-strong", "start", "middle"));
      const showTip = (ev) => { const bb = el.getBoundingClientRect(); tip.show(ev && ev.clientX != null ? { x: ev.clientX, y: ev.clientY } : { x: bb.left + x0 + len, y: bb.top + y }, r.label, r.rows || [{ value: r.text != null ? r.text : fmtNum(r.value), label: r.note || "" }]); };
      g.addEventListener("pointermove", showTip); g.addEventListener("focus", () => showTip(null));
      g.addEventListener("pointerleave", () => tip.hide()); g.addEventListener("blur", () => tip.hide());
      el.append(g);
    });
  });
  return { figure: wrap, chart, destroy: () => chart.destroy() };
}

/** A square heatmap with one sequential hue (cell value 0..1 -> light to dark) and labelled axes. */
export function heatmap(host, o) {
  const n = o.labels.length;
  const wrap = h("figure", { class: "chartfig" });
  if (o.title) wrap.append(h("figcaption", { class: "label ink" }, o.title));
  const box = h("div", { class: "chart" });
  wrap.append(box);
  if (o.caption) wrap.append(h("p", { class: "legend-text" }, o.caption));
  host.append(wrap);
  if (o.table) { const tw = withTable(box, o.table.columns, o.table.rows); wrap.append(h("div", { class: "chart-foot" }, tw.button), tw.table); }
  const chart = fit(box, 360, (el, w) => {
    const lab = 44, side = Math.min(w - lab - 8, 340), cell = side / n;
    el.setAttribute("aria-label", o.aria || o.title || "Heat map");
    el.setAttribute("height", lab + side);
    el.setAttribute("viewBox", `0 0 ${w} ${lab + side}`);
    o.labels.forEach((l, i) => {
      el.append(textEl(lab + cell * i + cell / 2, lab - 8, l, "tick", "middle"), textEl(lab - 8, lab + cell * i + cell / 2, l, "tick", "end", "middle"));
    });
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = o.values[i][j], self = i === j;
      const g = svg("g", { class: "hcell", tabindex: self ? null : 0, role: "img", "aria-label": `${o.labels[i]} and ${o.labels[j]}: ${v.toFixed(2)}` });
      const a = self ? 0 : clamp(v, 0, 1);
      g.append(svg("rect", { x: lab + cell * j + 1, y: lab + cell * i + 1, width: cell - 2, height: cell - 2, rx: 2, fill: self ? "var(--key-soft)" : `rgba(29,95,168,${0.06 + 0.9 * a})` }));
      if (!self && cell >= 24 && a > 0.45) g.append(textEl(lab + cell * j + cell / 2, lab + cell * i + cell / 2, v.toFixed(2).replace(/^0/, ""), "hval", "middle", "middle"));
      const showTip = (ev) => { const bb = el.getBoundingClientRect(); tip.show(ev && ev.clientX != null ? { x: ev.clientX, y: ev.clientY } : { x: bb.left + lab + cell * j, y: bb.top + lab + cell * i }, `${o.labels[i]} and ${o.labels[j]}`, [{ value: v.toFixed(2), label: o.valueLabel || "" }]); };
      if (!self) { g.addEventListener("pointermove", showTip); g.addEventListener("focus", () => showTip(null)); g.addEventListener("pointerleave", () => tip.hide()); g.addEventListener("blur", () => tip.hide()); }
      el.append(g);
    }
  });
  return { figure: wrap, chart, destroy: () => chart.destroy() };
}
