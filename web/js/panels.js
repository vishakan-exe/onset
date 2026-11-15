// Findings log and the explanation panels.  Every panel states what is measured, what is modelled, and how sure.
import { h, svg, clear, clamp, fmt, icon, roiLabel, roiShort, AU_NAMES, STATUS_WORD, GROUP_LABEL, withTable, tip } from "./util.js";
import { eventLabel } from "./timelens.js";

const isExpr = (e) => e.status === "micro" || e.status === "expression" || e.status === "unexplained" || e.status === "sustained";
const pct = (p) => (p == null ? "n/a" : p >= 0.995 ? "> 99 %" : p <= 0.005 ? "< 1 %" : `${Math.round(p * 100)} %`);

export function eventTitle(e) {
  if (e.status === "blink") return "Blink";
  if (e.status === "speech") return "Mouth movement (speech or chewing)";
  if (e.named_aus && e.named_aus.length) return e.named_aus.map((a) => `${AU_NAMES[a] || "AU" + a}`).join(" + ");
  return "Movement without a clear pattern";
}
const hypLabel = (hy) => (hy.aus.length ? hy.aus.map((a) => `AU${a}`).join(" + ") : "no AU") + (hy.blink ? " + blink" : "") + (hy.speech ? " + speech" : "");

function section(title, sub, ...kids) {
  return h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, title), sub ? h("p", { class: "legend-text" }, sub) : null), ...kids);
}

/** A redrawn-on-resize SVG so text stays at its real size. */
function fit(host, height, draw) {
  const el = svg("svg", { role: "img" });
  host.append(el);
  const run = () => {
    const w = Math.max(240, Math.floor(host.clientWidth || 320));
    el.setAttribute("viewBox", `0 0 ${w} ${height}`); el.setAttribute("width", w); el.setAttribute("height", height);
    clear(el); draw(el, w, height);
  };
  const ro = new ResizeObserver(run); ro.observe(host); run();
  return { el, run, destroy: () => ro.disconnect() };
}

// ------------------------------------------------------------------------------------------------ the log of movements
function spark(ev, w = 78, hgt = 26) {
  const tr = ev.traces;
  const el = svg("svg", { class: "spark", viewBox: `0 0 ${w} ${hgt}`, width: w, height: hgt, "aria-hidden": "true" });
  if (!tr || !tr.amp_pct_iod || tr.amp_pct_iod.length < 3) return el;
  const a = tr.amp_pct_iod, peak = Math.max(0.05, ...a.map((v) => Math.abs(v)));
  const x = (i) => 1 + (i / (a.length - 1)) * (w - 2), y = (v) => hgt - 3 - ((v + 0.15 * peak) / (1.15 * peak)) * (hgt - 6);
  el.append(svg("line", { x1: 0, x2: w, y1: y(0), y2: y(0), class: "spark-zero" }));
  el.append(svg("path", { d: a.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(clamp(v, -0.15 * peak, 1.05 * peak)).toFixed(1)}`).join(""), class: "spark-line", fill: "none" }));
  return el;
}

export class Findings {
  constructor(host, { result, bus }) {
    this.r = result; this.bus = bus; this.rows = new Map();
    const evs = result.events;
    const n = (k) => evs.filter((e) => k(e)).length;
    const parts = [];
    if (n(isExpr)) parts.push(`${n(isExpr)} movement${n(isExpr) === 1 ? "" : "s"} of the face`);
    if (n((e) => e.status === "blink")) parts.push(`${n((e) => e.status === "blink")} blink${n((e) => e.status === "blink") === 1 ? "" : "s"}`);
    if (n((e) => e.status === "speech")) parts.push(`${n((e) => e.status === "speech")} mouth movement${n((e) => e.status === "speech") === 1 ? "" : "s"}`);
    this.head = h("p", { class: "small muted" }, evs.length ? parts.join(", ") : "Nothing above the detection threshold");
    this.list = h("ol", { class: "log", "aria-label": "Detected movements" });
    for (const e of evs) {
      const expr = isExpr(e);
      const btn = h("button", { type: "button", class: `log-row${expr ? "" : " artefact"}`, onclick: () => this.bus.emit("select", { id: e.id, from: this }), onkeydown: (ev) => this.key(ev, e) },
        h("span", { class: "log-t num" }, e.t_apex.toFixed(2), h("small", {}, " s")),
        h("span", { class: "log-main" },
          h("span", { class: "log-title" }, eventLabel(e), e.status === "micro" ? null : h("span", { class: "chip" }, STATUS_WORD[e.status] || e.status)),
          h("span", { class: "log-sub num small muted" }, e.duration_ms != null ? `${Math.round(e.duration_ms)} ms` : "end not seen", " · score ", e.spot.score.toFixed(1))),
        spark(e));
      const li = h("li", {}, btn);
      this.rows.set(e.id, btn);
      this.list.append(li);
    }
    if (!evs.length) this.list.append(h("li", { class: "muted" }, "No movement rose above the noise. That is a measurement, not an error: with this recording nothing larger than the smallest detectable movement happened."));
    host.append(this.head, this.list);
    this.bus.on("select", ({ id }) => this.mark(id));
  }
  mark(id) {
    for (const [k, b] of this.rows) { b.setAttribute("aria-current", String(k === id)); }
    const b = this.rows.get(id);
    if (b && b.scrollIntoView) b.scrollIntoView({ block: "nearest" });
  }
  key(ev, e) {
    if (ev.key !== "ArrowDown" && ev.key !== "ArrowUp") return;
    ev.preventDefault();
    const ids = this.r.events.map((x) => x.id), i = ids.indexOf(e.id) + (ev.key === "ArrowDown" ? 1 : -1);
    const nb = this.rows.get(ids[clamp(i, 0, ids.length - 1)]);
    if (nb) { nb.focus(); this.bus.emit("select", { id: ids[clamp(i, 0, ids.length - 1)], from: this }); }
  }
}

// ------------------------------------------------------------------------------------------------ the explanation
export class Explainer {
  constructor(host, { result, bus, plate }) {
    this.r = result; this.bus = bus; this.plate = plate; this.scheme = "megc3";
    this.host = host; this.cleanup = [];
    this.bus.on("select", ({ id }) => this.show(this.r.events.find((e) => e.id === id) || null));
    this.show(null);
  }
  dispose() { this.cleanup.forEach((f) => f()); this.cleanup = []; }

  show(ev) {
    this.dispose();
    clear(this.host);
    if (!ev) { this.host.append(h("p", { class: "muted" }, "Select a movement in the log or on the timeline. Each finding opens with what the instrument saw, then how sure it is, and then why.")); return; }
    this.ev = ev;
    this.host.append(this.finding(ev));
    if (!isExpr(ev)) { this.host.append(this.artefact(ev)); return; }
    this.host.append(this.whatMoved(ev), this.categories(ev), this.evidence(ev), this.deletion(ev), this.contrast(ev));
  }

  // 1 -- the finding itself
  finding(ev) {
    const when = `${ev.t_onset.toFixed(2)} to ${ev.t_offset != null ? ev.t_offset.toFixed(2) : "?"} s`;
    const dur = ev.duration_ms != null ? `${Math.round(ev.duration_ms)} ms` : "end not seen";
    const flags = (ev.flags || []).map((f) => h("li", {}, icon("info"), f));
    const kind = ev.micro ? "Brief movement" : (STATUS_WORD[ev.status] || ev.status);
    // the size of a movement is how far its largest region moved (the quantity the specimen truth and the quality grade use), not the
    // fitted amplitude of the best-matching template, which is a different thing and is listed with the action units below
    const used = (ev.ledger || []).filter((l) => l.used && Number.isFinite(l.mag_pct_iod));
    const top = used.length ? used.reduce((a, b) => (b.mag_pct_iod > a.mag_pct_iod ? b : a)) : null;
    return h("section", { class: "sec finding" },
      h("p", { class: "label" }, `${ev.id} · ${kind} · ${when} · ${dur}`),
      h("h2", { class: "display finding-title" }, eventTitle(ev)),
      h("p", { class: "prose" }, ev.narrative),
      h("dl", { class: "facts" },
        top ? h("div", {}, h("dt", {}, "Size at its largest region"), h("dd", { class: "num" }, `${top.mag_pct_iod.toFixed(2)} % of the eye distance`, h("small", { class: "muted" }, ` · ${roiLabel(top.roi)}`))) : null,
        h("div", {}, h("dt", {}, "Detection score"), h("dd", { class: "num" }, `${ev.spot.score.toFixed(1)}`, h("small", { class: "muted" }, ` (threshold ${this.r.timeline.threshold})`))),
        h("div", {}, h("dt", {}, "Matched pattern"), h("dd", {}, ev.spot.best_template === "blink" ? "blink" : ev.spot.best_template || "n/a")),
        h("div", {}, h("dt", {}, "Head shift during it"), h("dd", { class: "num" }, ev.head_excursion ? `${(ev.head_excursion.iod * 100).toFixed(2)} % of the eye distance` : "n/a"))),
      flags.length ? h("ul", { class: "flags" }, flags) : null);
  }

  artefact(ev) {
    const w = ev.aus_withheld || [];
    return section("Why no expression is claimed", null,
      h("p", { class: "prose" }, ev.status === "blink"
        ? `The eyelids sweep across the eye in a blink and move the skin around it. Regions around the eyes are excluded and no action unit is reported for this event. The probability that this is a blink is ${pct(ev.p_blink)}.`
        : `Moving the mouth to talk or chew shifts the lower face far more than any expression would. No action unit is reported for this event. The probability that this is mouth movement is ${pct(ev.p_speech)}.`),
      w.length ? h("details", { class: "fold" }, h("summary", {}, `What the model would have said without the artefact filter (${w.length} action unit${w.length === 1 ? "" : "s"})`),
        h("p", { class: "legend-text" }, "Shown only so the filter can be checked. These are not findings: an eyelid or mouth movement can imitate them."),
        h("ul", { class: "plain" }, w.map((a) => h("li", { class: "num" }, `AU${a.au} ${a.name}: ${pct(a.pip)}`)))) : null);
  }

  // 2 -- which action units, how sure, how large
  whatMoved(ev) {
    const aus = (ev.aus || []).filter((a) => a.pip >= 0.05).slice(0, 7);
    const mda = this.r.quality.mda50_pct || {};
    const axMax = Math.max(0.5, ...aus.map((a) => a.hi_pct_iod || a.amp_pct_iod));
    const nice = axMax <= 1 ? Math.ceil(axMax * 4) / 4 : Math.ceil(axMax * 2) / 2;
    const body = h("div", { class: "au-table", role: "table", "aria-label": "Action units for the selected movement" },
      h("div", { class: "au-row au-head", role: "row" }, h("span", { role: "columnheader" }, "Action unit"), h("span", { role: "columnheader" }, "Chance it is involved"), h("span", { role: "columnheader" }, "How large (% of the eye distance)")),
      aus.map((a) => {
        const floor = mda[a.au];
        const left = (v) => `${clamp((v / nice) * 100, 0, 100)}%`;
        return h("div", { class: `au-row${a.pip >= 0.5 ? " named" : ""}`, role: "row" },
          h("span", { class: "au-name", role: "cell" }, h("b", { class: "num" }, `AU${a.au}`), " ", a.name),
          h("span", { class: "au-pip", role: "cell" }, h("i", { class: "bar", style: { "--v": `${Math.round(a.pip * 100)}%` } }), h("span", { class: "num" }, pct(a.pip))),
          h("span", { class: "au-amp", role: "cell" },
            h("span", { class: "amp", style: { "--lo": left(a.lo_pct_iod), "--est": left(a.amp_pct_iod), "--hi": left(a.hi_pct_iod), "--mda": floor != null ? left(floor) : "-10%" } }, h("i", { class: "range" }), h("i", { class: "dot" }), floor != null ? h("i", { class: "floor", title: `Smallest movement this recording can show for AU${a.au}: ${floor.toFixed(2)} %` }) : null),
            h("span", { class: "num small" }, `${a.amp_pct_iod.toFixed(2)}`, h("span", { class: "muted" }, ` (${a.lo_pct_iod.toFixed(2)} to ${a.hi_pct_iod.toFixed(2)})`))));
      }));
    const axis = h("div", { class: "au-axis small muted num" }, h("span", {}, "0"), h("span", {}, `${nice / 2} %`), h("span", {}, `${nice} %`));
    const note = h("p", { class: "legend-text" }, "Bars: probability that the action unit is part of the movement. Dot and line: estimated size with its 90 % interval. The tick on each scale is the smallest movement this recording could show; a dot near it is barely distinguishable from noise.");
    const hyp = (ev.hypotheses || []).slice(0, 4);
    const alt = hyp.length > 1 ? h("details", { class: "fold" }, h("summary", {}, "Other combinations the model considered"),
      h("ul", { class: "plain" }, hyp.map((x, i) => h("li", { class: "num" }, `${hypLabel(x)}: ${pct(x.p)}${i === 0 ? " (leading)" : ""}`)))) : null;
    return section("What moved", "Action units (AUs) are the standard FACS description of facial movements.", aus.length ? body : h("p", { class: "muted" }, "No single action unit is probable enough to list."), aus.length ? axis : null, aus.length ? note : null, alt);
  }

  // 3 -- category hypotheses, with abstention
  categories(ev) {
    const cats = ev.categories || {};
    const schemes = Object.keys(cats);
    const wrap = h("div", { class: "cats" });
    const names = { megc3: "3 classes (MEGC)", casme2_5: "5 classes (CASME II)", basic7: "7 emotions" };
    const render = () => {
      clear(wrap);
      const c = cats[this.scheme];
      if (!c) return;
      let why = null;
      if (c.abstain) {
        if (c.artefact > 0.5) why = "Most likely a blink or mouth movement, not an expression.";
        else if (c.unclear > c.top_p) why = "The pattern does not match any pattern this label set describes.";
        else why = `No label reaches 50 % (the highest is ${c.top} at ${pct(c.top_p)}).`;
      }
      wrap.append(h("p", { class: `verdictline${c.abstain ? " abstain" : ""}` }, icon(c.abstain ? "info" : "check"),
        c.abstain ? h("span", {}, h("b", {}, "No label offered. "), why) : h("span", {}, h("b", {}, `${c.top}`), ` leads with ${pct(c.top_p)}, ${Math.round(c.margin * 100)} points ahead of the next label.`)));
      const rows = c.labels.map((l, i) => ({ l, p: c.probs[i], kind: "label" })).sort((a, b) => b.p - a.p);
      rows.push({ l: "None of these", p: c.unclear, kind: "unclear" }, { l: "Not an expression", p: c.artefact, kind: "artefact" });
      wrap.append(h("div", { class: "bars" }, rows.map((r) => h("div", { class: `bar-row ${r.kind}${r.l === c.top && !c.abstain ? " top" : ""}` },
        h("span", { class: "bar-l" }, r.l), h("span", { class: "bar-t" }, h("i", { style: { "--v": `${Math.round(r.p * 100)}%` } })), h("span", { class: "bar-v num" }, pct(r.p))))));
      if (Array.isArray(c.set) && c.set.length) {
        const level = Math.round((1 - (c.set_alpha != null ? c.set_alpha : 0.1)) * 100);
        wrap.append(h("p", { class: "setline" }, h("b", { class: "num" }, `${level} % set: `), c.set.join(", "),
          h("span", { class: "muted" }, c.set.length === 1 ? " (the one label that cannot be ruled out)" : ` (${c.set.length} labels that cannot be ruled out)`)));
      }
      wrap.append(h("p", { class: "legend-text" }, c.note, c.set && c.set.length ? ` The set is calibrated on synthetic faces so that it holds the scripted label at least ${Math.round((1 - (c.set_alpha != null ? c.set_alpha : 0.1)) * 10)} times in 10; on real faces that is not guaranteed.` : ""));
    };
    const seg = h("div", { class: "seg", role: "group", "aria-label": "Label set" }, schemes.map((k) =>
      h("button", { type: "button", "aria-pressed": String(k === this.scheme), onclick: (e) => { this.scheme = k; for (const b of seg.children) b.setAttribute("aria-pressed", String(b === e.currentTarget)); render(); } }, names[k] || k)));
    if (!schemes.includes(this.scheme)) this.scheme = schemes[0];
    for (const b of seg.children) b.setAttribute("aria-pressed", String(b.textContent === (names[this.scheme] || this.scheme)));
    render();
    return section("What it could mean", "Labels are hypotheses derived from the action units with an expert prior. They are not a trained recogniser, and a movement of the face does not show what a person feels.", seg, wrap);
  }

  // 4 -- where the evidence sits
  evidence(ev) {
    const led = (ev.ledger || []).filter((l) => l.used);
    if (!led.length) return section("Where the evidence sits", null, h("p", { class: "muted" }, "No region-level ledger for this movement."));
    const total = led.reduce((s, l) => s + Math.max(0, l.evidence), 0) || 1;
    const share = new Map(led.map((l) => [l.roi, Math.max(0, l.evidence) / total]));
    const maxShare = Math.max(...share.values());
    const mapHost = h("div", { class: "emap" });
    const polys = [];
    const canon = this.r.canon;
    const s = canon.plate_size;
    const sv = svg("svg", { viewBox: `0 0 ${s} ${s}`, role: "img", "aria-label": `Map of the face regions shaded by how much each contributed to the finding. Strongest: ${roiLabel([...share.entries()].sort((a, b) => b[1] - a[1])[0][0])}.` });
    sv.append(svg("rect", { width: s, height: s, fill: "#0e1013" }));
    for (const [name, ps] of Object.entries(canon.rois)) {
      const sh = share.get(name) || 0;
      for (const poly of ps) {
        const d = poly.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join("") + "Z";
        const path = svg("path", { d, class: "emap-roi", "data-roi": name, fill: `rgba(127,182,244,${(0.08 + 0.85 * (sh / maxShare)).toFixed(3)})`, stroke: "#8794a3", "stroke-width": 0.4 });
        path.addEventListener("pointermove", (e) => this.hoverRoi(e, name, ev)); path.addEventListener("pointerleave", () => this.hoverRoi(null));
        sv.append(path); polys.push(path);
      }
    }
    mapHost.append(h("div", { class: "plate-wrap small-plate" }, h("div", { class: "crop" }, h("div", { class: "plate" }, sv, h("span", { class: "stamp tl" }, "Evidence by region")))));
    const sorted = [...led].sort((a, b) => b.evidence - a.evidence);
    const tbl = h("table", { class: "ledger" },
      h("thead", {}, h("tr", {}, ["Region", "Moved", "How far", "× noise", "Share of evidence"].map((c, i) => h("th", { class: i > 1 ? "r" : "" }, c)))),
      h("tbody", {}, sorted.map((l, i) => {
        const meas = Math.hypot(l.dx_px, l.dy_px), res = Math.hypot(l.resid_dx, l.resid_dy);
        const explained = meas > 1e-6 ? clamp(1 - (res * res) / (meas * meas), 0, 1) : 0;
        const tr = h("tr", { class: i >= 8 ? "more" : "", hidden: i >= 8, tabindex: 0, dataset: { roi: l.roi }, title: `${Math.round(explained * 100)} % of this region's movement is explained by the named action units` },
          h("td", {}, roiLabel(l.roi)), h("td", {}, l.direction), h("td", { class: "r num" }, `${l.mag_pct_iod.toFixed(2)} %`), h("td", { class: "r num" }, l.z.toFixed(1)),
          h("td", { class: "r" }, h("span", { class: "share" }, h("i", { style: { width: `${Math.round(100 * share.get(l.roi) / maxShare)}%` } }), h("span", { class: "num" }, `${Math.round(100 * share.get(l.roi))} %`))));
        tr.addEventListener("pointerenter", () => this.hoverRoi(null, l.roi)); tr.addEventListener("focus", () => this.hoverRoi(null, l.roi));
        tr.addEventListener("pointerleave", () => this.hoverRoi(null)); tr.addEventListener("blur", () => this.hoverRoi(null));
        return tr;
      })));
    const more = sorted.length > 8 ? h("button", { class: "linkish small", type: "button", onclick: (e) => { const open = e.currentTarget.dataset.open !== "1"; e.currentTarget.dataset.open = open ? "1" : "0"; tbl.querySelectorAll("tr.more").forEach((r) => { r.hidden = !open; }); e.currentTarget.textContent = open ? "Show fewer regions" : `Show all ${sorted.length} regions`; } }, `Show all ${sorted.length} regions`) : null;
    this._roiPaths = polys;
    return section("Where the evidence sits", "Each region's share is how much better the named action units explain its movement than no movement at all, after head motion is removed. × noise is the displacement divided by the measurement noise for that region.",
      h("div", { class: "split" }, mapHost, h("div", {}, tbl, more)));
  }
  hoverRoi(e, name, ev) {
    const on = e ? name : (arguments.length > 1 ? name : null);
    for (const p of this._roiPaths || []) { const hit = on && p.dataset.roi === on; p.setAttribute("stroke", hit ? "#ffd27a" : "#8794a3"); p.setAttribute("stroke-width", hit ? 1.4 : 0.4); }
    if (this.plate && this.plate.setHighlight) this.plate.setHighlight(on);
    if (e && ev) {
      const l = (ev.ledger || []).find((x) => x.roi === name);
      tip.show({ x: e.clientX, y: e.clientY }, roiLabel(name), l && l.used ? [{ value: `${l.mag_pct_iod.toFixed(2)} %`, label: `of the eye distance, ${l.direction}` }, { value: l.z.toFixed(1), label: "times the noise level" }] : ["No measurement"]);
    } else if (!e) tip.hide();
  }

  // 5 -- does the conclusion survive without those regions?
  deletion(ev) {
    const f = ev.faithfulness;
    if (!f) return h("span");
    const verdicts = {
      faithful: `Faithful. Taking away the regions the explanation points to ${f.flips.ranked != null ? `overturns the conclusion after ${f.flips.ranked} region${f.flips.ranked === 1 ? "" : "s"}` : "weakens the conclusion sharply"}; taking away random regions ${f.flips.random_median != null ? `needs about ${f.flips.random_median} before it does` : "mostly does not"}. The highlighted regions are doing the work.`,
      robust: "Robust. The conclusion stays above 95 % even with the strongest regions removed. The test cannot single out regions here, but the finding does not hinge on a few of them.",
      diffuse: "Diffuse. The conclusion fades about as fast when random regions are removed. The evidence is spread over the face; treat the highlighted regions as indicative, not decisive.",
      inverted: "Warning. Random regions matter more than the highlighted ones, so the ranking of regions should not be trusted for this movement.",
      na: "Nothing to test: the conclusion was not above even odds to begin with.",
    };
    const host = h("div", { class: "chart del" });
    const sig = (v) => 1 / (1 + Math.exp(-v));
    const draw = (el, w, H) => {
      const m = { l: 40, r: 126, t: 12, b: 38 };
      const nmax = f.n[f.n.length - 1] || 1, X = (n) => m.l + (n / nmax) * (w - m.l - m.r), Y = (p) => m.t + (1 - p) * (H - m.t - m.b);
      el.setAttribute("aria-label", `Chart: chance that the conclusion holds as regions are removed. ${verdicts[f.verdict]}`);
      for (const p of [0, 0.5, 1]) { el.append(svg("line", { x1: m.l, x2: w - m.r, y1: Y(p), y2: Y(p), class: p === 0.5 ? "grid-major" : "grid" })); el.append(svg("text", { x: m.l - 7, y: Y(p) + 4, class: "tick", "text-anchor": "end" }, `${Math.round(p * 100)}%`)); }
      for (let n = 0; n <= nmax; n++) el.append(svg("text", { x: X(n), y: H - 20, class: "tick", "text-anchor": "middle" }, String(n)));
      el.append(svg("text", { x: (m.l + w - m.r) / 2, y: H - 3, class: "axis-title", "text-anchor": "middle" }, "regions removed"));
      if (f.ranked.length > 1) {
        const lo = f.random_mean.map((v, i) => sig(v - f.random_sd[i])), hi = f.random_mean.map((v, i) => sig(v + f.random_sd[i]));
        const band = hi.map((v, i) => `${i ? "L" : "M"}${X(f.n[i])},${Y(v)}`).join("") + lo.map((v, i) => `L${X(f.n[f.n.length - 1 - i])},${Y(lo[lo.length - 1 - i])}`).join("") + "Z";
        el.append(svg("path", { d: band, class: "del-band" }));
        el.append(svg("path", { d: f.random_mean_p.map((v, i) => `${i ? "L" : "M"}${X(f.n[i])},${Y(v)}`).join(""), class: "del-random", fill: "none" }));
        el.append(svg("path", { d: f.ranked_p.map((v, i) => `${i ? "L" : "M"}${X(f.n[i])},${Y(v)}`).join(""), class: "del-ranked", fill: "none" }));
        f.ranked_p.forEach((v, i) => el.append(svg("circle", { cx: X(f.n[i]), cy: Y(v), r: 3, class: "del-dot" })));
        const e = f.n.length - 1;
        el.append(svg("text", { x: X(f.n[e]) + 9, y: Y(f.ranked_p[e]) + (f.ranked_p[e] < f.random_mean_p[e] ? 4 : 4), class: "lab-pen" }, "Strongest first"));
        el.append(svg("text", { x: X(f.n[e]) + 9, y: Y(f.random_mean_p[e]) + 4 + (Math.abs(Y(f.ranked_p[e]) - Y(f.random_mean_p[e])) < 14 ? -12 : 0), class: "lab-ink" }, "Random regions"));
      } else el.append(svg("circle", { cx: X(0), cy: Y(f.ranked_p[0]), r: 3.5, class: "del-dot" }));
    };
    fit(host, 190, draw);
    const rows = f.n.map((n, i) => [n, f.removed_order[i - 1] ? roiShort(f.removed_order[i - 1]) : "none", pct(f.ranked_p[i] ?? null), pct(f.random_mean_p[i] ?? null)]);
    const tw = withTable(host, ["Regions removed", "Last removed (strongest first)", "Chance, strongest first", "Chance, random regions (average)"], rows, "Show as a table");
    return section("Does it survive without those regions?", `The model is re-run with the highest-evidence regions deleted, and again with the same number of random regions deleted. The line is the chance that the weakest of the named action units still holds.`,
      host, tw.table, h("p", { class: `verdictline ${f.verdict}` }, icon(f.verdict === "inverted" ? "warn" : "info"), h("span", {}, verdicts[f.verdict])), tw.button);
  }

  // 6 -- why this and not the runner-up
  contrast(ev) {
    const c = ev.contrast, hy = ev.hypotheses || [];
    if (!c || !c.length || hy.length < 2) return h("span");
    const a = hypLabel(hy[0]), b = hypLabel(hy[1]);
    const mx = Math.max(...c.map((r) => Math.abs(r.favours_a)), 1);
    const rows = c.map((r) => h("div", { class: "div-row" },
      h("span", { class: "div-l" }, roiLabel(r.roi)),
      h("span", { class: "div-track", role: "img", "aria-label": `${roiLabel(r.roi)} favours ${r.favours_a >= 0 ? a : b}` },
        h("i", { class: r.favours_a >= 0 ? "a" : "b", style: { width: `${Math.round(50 * Math.abs(r.favours_a) / mx)}%`, [r.favours_a >= 0 ? "left" : "right"]: "50%" } })),
      h("span", { class: "div-v num small muted" }, Math.abs(r.favours_a).toFixed(1))));
    return section(`Why ${a} rather than ${b}?`, `The leading and the runner-up combinations are fitted to the same measurements. These regions separate them most; bars to the right favour ${a}, to the left ${b}. The unit is a chi-square difference, larger = clearer.`,
      h("div", { class: "div" }, h("div", { class: "div-axis small" }, h("span", { class: "muted" }, `← favours ${b}`), h("span", { class: "pen" }, `favours ${a} →`)), rows),
      hy[0].p - hy[1].p < 0.15 ? h("p", { class: "verdictline abstain" }, icon("info"), h("span", {}, `The two combinations are close (${pct(hy[0].p)} against ${pct(hy[1].p)}): this recording cannot separate them reliably.`)) : null);
  }
}

// ------------------------------------------------------------------------------------------------ recording quality
export function qualityPanel(result) {
  const q = result.quality;
  const grades = ["A", "B", "C", "D", "E"], gi = grades.indexOf(q.grade);
  const rail = h("ol", { class: "grade-rail", "aria-hidden": "true" }, grades.map((g, i) => h("li", { class: i === gi ? "on" : "" }, g)));
  const items = (q.items || []).map((it) => h("li", { class: `qi ${it.status}` },
    icon(it.status === "good" ? "check" : it.status === "ok" ? "info" : "warn"),
    h("span", { class: "qi-name" }, it.name), h("span", { class: "qi-v num" }, `${it.value} ${it.unit}`), it.note ? h("span", { class: "qi-note small muted" }, it.note) : null));
  const mda = q.mda50_pct || {}, mda90 = q.mda90_pct || {};
  const mx = Math.max(0.5, ...Object.values(mda90));
  const sens = Object.keys(mda).map(Number).sort((a, b) => a - b).map((au) => h("div", { class: "sens-row" },
    h("span", { class: "sens-l" }, h("b", { class: "num" }, `AU${au}`), " ", AU_NAMES[au] || ""),
    h("span", { class: "sens-t", role: "img", "aria-label": `AU${au}: shown half the time at ${mda[au].toFixed(2)} percent, nine times in ten at ${(mda90[au] || mda[au]).toFixed(2)} percent` },
      h("i", { class: "r", style: { left: `${(mda[au] / mx) * 100}%`, width: `${(((mda90[au] || mda[au]) - mda[au]) / mx) * 100}%` } }), h("i", { class: "m", style: { left: `${(mda[au] / mx) * 100}%` } })),
    h("span", { class: "num small muted" }, `${mda[au].toFixed(2)} – ${(mda90[au] || mda[au]).toFixed(2)} %`)));
  return h("div", { class: "quality" },
    h("div", { class: "grade" }, h("span", { class: "big-num grade-letter" }, q.grade), rail),
    h("p", { class: "prose" }, q.summary),
    q.advice && q.advice.length ? h("ul", { class: "advice" }, q.advice.map((a) => h("li", {}, icon("info"), a))) : null,
    h("ul", { class: "qitems" }, items),
    section("Smallest movement this recording can show", "For each action unit: the size (% of the eye distance) that is picked up half the time, and nine times in ten. Anything smaller than the left end is below what this recording can resolve.", h("div", { class: "sens" }, sens)));
}
