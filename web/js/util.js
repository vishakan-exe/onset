// Small DOM helpers, formatters and the shared tooltip.  No framework: the interface is a handful of stations.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const SVGNS = "http://www.w3.org/2000/svg";

function apply(el, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k === "style" && typeof v === "object") {
      for (const [sk, sv] of Object.entries(v)) { if (sk.startsWith("--")) el.style.setProperty(sk, sv); else el.style[sk] = sv; }
    }
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
}
function append(el, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false) continue;
    el.append(k.nodeType ? k : document.createTextNode(String(k)));
  }
}
/** h("div", {class: "x"}, "text", child) - text is always inserted as text, never as markup. */
export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  apply(el, attrs);
  append(el, kids);
  return el;
}
export function svg(tag, attrs, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  apply(el, attrs);
  append(el, kids);
  return el;
}
export const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export function debounce(fn, ms = 120) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
export function raf(fn) { let q = false; return (...a) => { if (q) return; q = true; requestAnimationFrame(() => { q = false; fn(...a); }); }; }

export const fmt = {
  pct: (v, d = 1) => (v == null || !isFinite(v) ? "n/a" : `${v.toFixed(d)} %`),
  s: (t, d = 2) => (t == null || !isFinite(t) ? "n/a" : `${t.toFixed(d)} s`),
  ms: (s) => (s == null || !isFinite(s) ? "n/a" : `${Math.round(s * 1000)} ms`),
  n: (v, d = 1) => (v == null || !isFinite(v) ? "n/a" : v.toFixed(d)),
  prob: (p) => (p == null ? "n/a" : p >= 0.995 ? "> 99 %" : p <= 0.005 ? "< 1 %" : `${Math.round(p * 100)} %`),
  clock: (t) => { const m = Math.floor(t / 60); return `${m}:${(t - 60 * m).toFixed(2).padStart(5, "0")}`; },
  date: (ts) => new Date(ts * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }),
};

export const AU_NAMES = {
  1: "Inner brow raiser", 2: "Outer brow raiser", 4: "Brow lowerer", 5: "Upper lid raiser", 6: "Cheek raiser", 7: "Lid tightener",
  9: "Nose wrinkler", 10: "Upper lip raiser", 12: "Lip corner puller", 14: "Dimpler", 15: "Lip corner depressor", 17: "Chin raiser",
};
export const GROUPS = ["brow", "eye", "nose", "cheek", "mouth"];
export const GROUP_LABEL = { brow: "Brow", eye: "Eye", nose: "Nose", cheek: "Cheek", mouth: "Mouth" };
const FAMILY = {
  fh: "Forehead", brow_out: "Outer brow", brow_mid: "Mid brow", brow_in: "Inner brow", lid_up: "Upper lid", lid_lo: "Lower lid",
  canthus: "Eye corner", wing: "Nose wing", cheek: "Cheek", nasolab: "Nasolabial fold", corner: "Mouth corner", glabella: "Between the brows",
  nose_root: "Nose bridge", lip_up: "Upper lip", lip_lo: "Lower lip", chin: "Chin",
};
export function roiLabel(name) {
  const m = name.match(/^(.*)_(L|R)$/);
  const fam = FAMILY[m ? m[1] : name] || name;
  return m ? `${fam}, ${m[2] === "L" ? "left of image" : "right of image"}` : fam;
}
export const roiShort = (name) => {
  const m = name.match(/^(.*)_(L|R)$/);
  return (FAMILY[m ? m[1] : name] || name) + (m ? ` ${m[2]}` : "");
};
export const STATUS_WORD = {
  micro: "Brief movement", expression: "Longer expression", blink: "Blink", speech: "Mouth movement", unexplained: "Unexplained movement",
  sustained: "Sustained displacement",
};

// ------------------------------------------------------------------------------------------------ tooltip
let tipEl = null;
function ensureTip() {
  if (!tipEl) { tipEl = h("div", { class: "tip", role: "tooltip" }); document.body.append(tipEl); }
  return tipEl;
}
export const tip = {
  /** rows: array of {key?: cssColor, value, label} or strings; at: {x, y} client coordinates */
  show(at, title, rows = []) {
    const el = ensureTip();
    clear(el);
    if (title) el.append(h("div", { class: "sub" }, title));
    for (const r of rows) {
      if (typeof r === "string") { el.append(h("div", {}, r)); continue; }
      const row = h("div", { class: "row" });
      if (r.key) row.append(h("span", { class: "key", style: { background: r.key } }));
      row.append(h("b", {}, r.value), h("span", { class: "sub" }, r.label || ""));
      el.append(row);
    }
    el.classList.add("on");
    const pad = 14, w = el.offsetWidth, hh = el.offsetHeight;
    let x = at.x + pad, y = at.y + pad;
    if (x + w > innerWidth - 8) x = at.x - w - pad;
    if (y + hh > innerHeight - 8) y = at.y - hh - pad;
    el.style.left = `${Math.max(6, x)}px`;
    el.style.top = `${Math.max(6, y)}px`;
  },
  hide() { if (tipEl) tipEl.classList.remove("on"); },
};

export function toast(msg, kind = "") {
  let host = $("#toast");
  if (!host) { host = h("div", { id: "toast", role: "status", "aria-live": "polite" }); document.body.append(host); }
  const t = h("div", { class: `toast ${kind}` }, msg);
  host.append(t);
  setTimeout(() => t.remove(), kind === "error" ? 7000 : 3600);
}

/** A "table view" twin for a chart: a button toggles between the chart and a table of its values. */
export function withTable(chartEl, columns, rows, label = "Show as table") {
  const table = h("table", { class: "table-view", hidden: true },
    h("thead", {}, h("tr", {}, columns.map((c, i) => h("th", { class: i ? "r" : "" }, c)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c, i) => h("td", { class: i ? "r" : "" }, c == null ? "" : String(c)))))));
  const btn = h("button", { class: "linkish small", type: "button", "aria-pressed": "false" }, label);
  btn.addEventListener("click", () => {
    const on = table.hidden;
    table.hidden = !on;
    chartEl.hidden = on;
    btn.setAttribute("aria-pressed", String(on));
    btn.textContent = on ? "Show as chart" : label;
  });
  return { button: btn, table };
}

export function download(blobOrText, name, type = "application/json") {
  const blob = blobOrText instanceof Blob ? blobOrText : new Blob([blobOrText], { type });
  const a = h("a", { href: URL.createObjectURL(blob), download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

/** Minimal pub-sub used between the plate, the timeline and the panels. */
export class Bus {
  constructor() { this.h = new Map(); }
  on(ev, fn) { (this.h.get(ev) || this.h.set(ev, new Set()).get(ev)).add(fn); return () => this.h.get(ev).delete(fn); }
  emit(ev, detail) { for (const fn of this.h.get(ev) || []) fn(detail); }
}

export function icon(name) {
  const d = ICONS[name] || "";
  const el = document.createElementNS(SVGNS, "svg");
  el.setAttribute("viewBox", "0 0 24 24");
  el.setAttribute("aria-hidden", "true");
  el.innerHTML = d;   // static strings defined in this file only
  return el;
}
const ICONS = {
  capture: '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="3.2"/><path d="M12 3.5l3 5.2M20.5 12l-5.2 3M12 20.5l-3-5.2M3.5 12l5.2-3"/>',
  bench: '<rect x="4" y="3.5" width="16" height="10" rx="1"/><path d="M3 17.5h3l1.6-2.4 2.3 4.6 2.2-6.2 1.7 4H21"/>',
  cabinet: '<rect x="4" y="3.5" width="16" height="17" rx="1"/><path d="M4 9.2h16M4 14.8h16M10.5 6.4h3M10.5 12h3M10.5 17.7h3"/>',
  cue: '<path d="M12 4v16M5 20h14M12 6.5L4.5 8.5M12 6.5l7.5 2"/><path d="M4.5 8.5L2.5 14a3 3 0 0 0 4 0zM19.5 8.5L17.5 14a3 3 0 0 0 4 0z"/>',
  evidence: '<path d="M4 20V4M4 20h16"/><path d="M7 16.5l4-4 3 2 5-7"/><circle cx="7" cy="16.5" r="1"/><circle cx="11" cy="12.5" r="1"/><circle cx="14" cy="14.5" r="1"/><circle cx="19" cy="7.5" r="1"/>',
  atlas: '<path d="M12 3.5c-3.6 0-5.8 2.6-5.8 6 0 4.2 2.2 8.5 5.8 8.5s5.8-4.3 5.8-8.5c0-3.4-2.2-6-5.8-6z"/><path d="M6.2 10.5H4M17.8 10.5H20M12 3.5V2M9 10h1M14 10h1M9.5 14.5c1.5 1 3.5 1 5 0"/>',
  play: '<path d="M7 4.5v15l12-7.5z"/>',
  pause: '<path d="M7 4.5v15M17 4.5v15"/>',
  prev: '<path d="M16 5l-9 7 9 7M5 5v14"/>',
  next: '<path d="M8 5l9 7-9 7M19 5v14"/>',
  record: '<circle cx="12" cy="12" r="6.5" fill="currentColor"/>',
  stop: '<rect x="6.5" y="6.5" width="11" height="11" fill="currentColor"/>',
  upload: '<path d="M12 16V4M7 9l5-5 5 5M4.5 20h15"/>',
  trash: '<path d="M5 7h14M9.5 7V4.5h5V7M7 7l1 13h8l1-13"/>',
  print: '<path d="M7 9V3.5h10V9M7 17H4.5V9h15v8H17M7 14h10v6.5H7z"/>',
  download: '<path d="M12 4v12M7 11l5 5 5-5M4.5 20h15"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  warn: '<path d="M12 3.5l9.5 16.5h-19z"/><path d="M12 10v5M12 17.4v.1"/>',
  info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5.5M12 7.8v.1"/>',
  loop: '<path d="M4 11V9.5A3.5 3.5 0 0 1 7.5 6H18M15 3l3 3-3 3M20 13v1.5A3.5 3.5 0 0 1 16.5 18H6M9 21l-3-3 3-3"/>',
};
