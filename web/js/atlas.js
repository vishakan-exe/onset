// The Atlas: the twelve movements the instrument looks for, where each shows on the face, and which pairs it finds hard
// to tell apart.  Everything here is read from the templates the analysis itself matches against (onset/analysis/atlas.py),
// so the page cannot drift from the instrument.  A table twin sits next to every drawing.
import { h, svg, clear, clamp, roiLabel, roiShort, tip, GROUP_LABEL } from "./util.js";
import { api } from "./api.js";
import { stageHead, note, guarded } from "./ui.js";
import { heatmap } from "./charts.js";

const PRIMARY = "#7fb6f4";     // the pattern being shown (cyanotype: what the model explains with)
const COMPARE = "#f3f6fa";     // a second pattern laid over it
const ARROW_PX = 17;           // longest arrow on the 176 px plate; sizes are relative, never millimetres

const len = (v) => Math.hypot(v.dx, v.dy);
const uniq = (xs) => [...new Set(xs)];
const family = (roi) => roiShort(roi).replace(/ [LR]$/, "");

/** "up and outward", "down", "inward" ... from a template vector (image coordinates: y grows downward). */
function directionWords(roi, dx, dy) {
  const n = Math.hypot(dx, dy) || 1, out = [];
  if (Math.abs(dy) / n > 0.38) out.push(dy < 0 ? "up" : "down");
  if (Math.abs(dx) / n > 0.38) {
    const m = roi.match(/_(L|R)$/);
    if (m) out.push((m[1] === "L" ? dx < 0 : dx > 0) ? "outward" : "inward");
    else out.push(dx < 0 ? "toward image left" : "toward image right");
  }
  return out.join(" and ") || "slightly";
}
const overlapWords = (s) => (s >= 0.8 ? "nearly the same pattern" : s >= 0.6 ? "strong overlap" : s >= 0.4 ? "partial overlap" : "little overlap");

export async function mount(host, { seg }, ctx) {
  host.append(stageHead({
    kicker: "Atlas",
    title: "What the instrument looks for",
    lede: "Twelve facial movements, each as the pattern of regions the analysis expects to see move. The patterns are written from the anatomy in the FACS manual, not learned from faces, and this page shows exactly the templates the instrument matches against.",
  }));
  const body = h("div", { class: "atlas" });
  host.append(body);
  const data = await guarded(body, "Loading the atlas", () => api.atlas());
  if (!data) return {};
  clear(body);

  const byAu = new Map(data.aus.map((a) => [a.au, a]));
  const groupOf = {};
  for (const [g, names] of Object.entries(data.canon.groups || {})) for (const n of names) groupOf[n] = g;
  const mainGroup = (a) => {
    const w = {};
    for (const v of a.vectors) { const g = groupOf[v.roi] || "face"; w[g] = (w[g] || 0) + len(v); }
    const best = Object.entries(w).sort((p, q) => q[1] - p[1])[0];
    return best ? GROUP_LABEL[best[0]] || best[0] : "";
  };

  let sel = byAu.has(+seg[0]) ? +seg[0] : 12;
  let cmp = null;
  const charts = [];

  // ---------------------------------------------------------------------------------------------- the ruler
  const ruler = h("div", { class: "ruler", role: "tablist", "aria-label": "Facial movements (action units)" });
  const tabs = new Map();
  for (const a of data.aus) {
    const b = h("button", { class: "ruler-btn", type: "button", role: "tab", id: `au-tab-${a.au}`, "aria-controls": "au-panel", title: a.name,
      onclick: () => select(a.au), onkeydown: (e) => rulerKey(e, a.au) },
    h("span", { class: "ruler-n num" }, String(a.au)), h("span", { class: "ruler-g" }, mainGroup(a)));
    tabs.set(a.au, b);
    ruler.append(b);
  }
  function rulerKey(e, au) {
    const order = data.aus.map((a) => a.au), i = order.indexOf(au);
    let j = null;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") j = (i + 1) % order.length;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = (i - 1 + order.length) % order.length;
    else if (e.key === "Home") j = 0;
    else if (e.key === "End") j = order.length - 1;
    if (j == null) return;
    e.preventDefault();
    select(order[j]);
    tabs.get(order[j]).focus();
  }
  const markTabs = () => { for (const [au, b] of tabs) { b.setAttribute("aria-selected", String(au === sel)); b.tabIndex = au === sel ? 0 : -1; } };

  const panel = h("div", { class: "atlas-panel", role: "tabpanel", id: "au-panel", tabindex: "-1" });
  body.append(ruler, panel);

  function select(au, scroll = false) {
    sel = au; cmp = null;
    markTabs(); renderPanel();
    try { history.replaceState(null, "", `#/atlas/${au}`); } catch { /* sandboxed frames */ }
    ctx.setTitle(`AU ${au} · Atlas`);
    if (scroll) ruler.scrollIntoView({ block: "start", behavior: "smooth" });
  }

  // ---------------------------------------------------------------------------------------------- the plate
  function arrowsFor(vectors, color, gain, cls) {
    const cen = Object.fromEntries(data.canon.centroids.map((c) => [c.roi, c]));
    const g = svg("g", { class: `atlas-arrows ${cls}`, stroke: color, fill: color, "stroke-linecap": "round", "stroke-width": 1 });
    for (const v of vectors) {
      const c = cen[v.roi];
      if (!c) continue;
      const dx = v.dx * gain, dy = v.dy * gain, l = Math.hypot(dx, dy);
      if (l < 0.6) continue;
      const ux = dx / l, uy = dy / l, hx = c.x + dx, hy = c.y + dy, s = Math.min(3.2, 0.34 * l + 0.9);
      g.append(svg("g", { style: { transformOrigin: `${c.x}px ${c.y}px` } },
        svg("circle", { cx: c.x, cy: c.y, r: 1.1, stroke: "none" }),
        svg("line", { x1: c.x, y1: c.y, x2: hx - ux * s * 0.6, y2: hy - uy * s * 0.6 }),
        svg("path", { d: `M${hx},${hy}L${hx - ux * s - uy * s * 0.55},${hy - uy * s + ux * s * 0.55}L${hx - ux * s + uy * s * 0.55},${hy - uy * s - ux * s * 0.55}Z`, stroke: "none" })));
    }
    return g;
  }

  function plateFigure(a, b) {
    const S = data.canon.plate_size;
    const vecOf = new Map(a.vectors.map((v) => [v.roi, v]));
    const bSet = new Set(b ? b.vectors.map((v) => v.roi) : []);
    const sv = svg("svg", { class: "overlay", viewBox: `0 0 ${S} ${S}`, role: "img",
      "aria-label": `Expected movement for AU ${a.au}, ${a.name}${b ? `, compared with AU ${b.au}` : ""}. The table beside the plate lists the same information.` });
    const regions = svg("g", { fill: "none" });
    for (const [name, polys] of Object.entries(data.canon.rois)) {
      for (const poly of polys) {
        const d = poly.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join("") + "Z";
        const cls = `atlas-roi${vecOf.has(name) ? " on" : ""}${bSet.has(name) ? " cmp" : ""}`;
        const p = svg("path", { d, class: cls, "data-roi": name });
        p.addEventListener("pointermove", (ev) => {
          const v = vecOf.get(name);
          tip.show({ x: ev.clientX, y: ev.clientY }, roiLabel(name), [v ? `Expected to move ${directionWords(name, v.dx, v.dy)}` : `Not part of the AU ${a.au} pattern`]);
        });
        p.addEventListener("pointerleave", () => tip.hide());
        regions.append(p);
      }
    }
    const maxLen = Math.max(...a.vectors.map(len), ...(b ? b.vectors.map(len) : [0]), 1e-6);
    const gain = ARROW_PX / maxLen;
    sv.append(regions);
    if (b) sv.append(arrowsFor(b.vectors, COMPARE, gain, "cmp"));
    sv.append(arrowsFor(a.vectors, PRIMARY, gain, "main"));
    const plate = h("div", { class: "plate" }, sv,
      h("span", { class: "stamp tl" }, `AU ${a.au} · expected pattern`),
      h("span", { class: "stamp tr num" }, `${Object.keys(data.canon.rois).length} regions`));
    const legend = h("ul", { class: "plate-key" },
      h("li", {}, h("i", { style: { "--c": PRIMARY } }), `AU ${a.au}`),
      b ? h("li", {}, h("i", { style: { "--c": COMPARE } }), `AU ${b.au}`) : null);
    return h("figure", { class: "atlas-plate" },
      h("div", { class: "plate-wrap" }, h("div", { class: "crop" }, plate)),
      legend,
      h("figcaption", { class: "legend-text" }, "Arrow lengths show direction and relative size only: the instrument measures real displacement in each recording, and these are the shapes it expects."));
  }

  // ---------------------------------------------------------------------------------------------- the dossier
  function contrast(a, b) {
    const A = new Map(a.vectors.map((v) => [v.roi, v])), B = new Map(b.vectors.map((v) => [v.roi, v]));
    const both = [...A.keys()].filter((k) => B.has(k)), onlyA = [...A.keys()].filter((k) => !B.has(k)), onlyB = [...B.keys()].filter((k) => !A.has(k));
    const opposite = both.filter((k) => A.get(k).dx * B.get(k).dx + A.get(k).dy * B.get(k).dy < 0);
    const line = (label, names) => (names.length ? h("p", {}, h("b", {}, label), " ", uniq(names.map(family)).join(", ")) : null);
    return h("div", { class: "contrast", "aria-live": "polite" },
      line("Both move:", both.filter((k) => !opposite.includes(k))),
      line("Move in opposite directions:", opposite),
      line(`Only AU ${a.au}:`, onlyA),
      line(`Only AU ${b.au}:`, onlyB),
      h("p", { class: "legend-text" }, "Where the two patterns share regions, the instrument has to rely on the regions that differ, which is why a recording with a poor view of those regions leaves the pair unresolved."));
  }

  function dossier(a) {
    const rows = [...a.vectors].sort((p, q) => len(q) - len(p));
    const top = len(rows[0]) || 1;
    const prot = data.prototypes.filter((p) => p.aus.includes(a.au));
    const regionTable = h("table", { class: "table-view region-table" },
      h("caption", { class: "visually-hidden" }, `Regions expected to move in AU ${a.au}`),
      h("thead", {}, h("tr", {}, h("th", {}, "Region"), h("th", {}, "Expected to move"), h("th", { class: "r" }, "Relative size"))),
      h("tbody", {}, rows.map((v) => {
        const tr = h("tr", { tabindex: 0 },
          h("td", {}, roiLabel(v.roi)), h("td", {}, directionWords(v.roi, v.dx, v.dy)),
          h("td", { class: "r" }, h("span", { class: "relbar", role: "img", "aria-label": `${Math.round((100 * len(v)) / top)} % of the largest` }, h("i", { style: { width: `${Math.round((100 * len(v)) / top)}%` } }))));
        const on = (flag) => () => { for (const p of panel.querySelectorAll(`.atlas-roi[data-roi="${v.roi}"]`)) p.classList.toggle("pick", flag); };
        tr.addEventListener("pointerenter", on(true)); tr.addEventListener("pointerleave", on(false));
        tr.addEventListener("focus", on(true)); tr.addEventListener("blur", on(false));
        return tr;
      })));

    const mistaken = h("ul", { class: "mist" }, a.confusable.map((c) => {
      const o = byAu.get(c.au);
      const pressed = cmp === c.au;
      return h("li", { class: "mist-row" },
        h("span", { class: "au-chip num" }, `AU ${c.au}`),
        h("span", { class: "mist-name" }, h("b", {}, o.name), h("span", { class: "muted" }, ` · ${overlapWords(c.similarity)}`)),
        h("span", { class: "mist-bar", style: { "--v": `${Math.round(clamp(c.similarity, 0, 1) * 100)}%` }, "aria-hidden": "true" }),
        h("b", { class: "num mist-v" }, c.similarity.toFixed(2)),
        h("button", { class: "btn small", type: "button", "aria-pressed": String(pressed),
          onclick: () => { cmp = pressed ? null : c.au; renderPanel(); } }, pressed ? "Hide on plate" : "Show on plate"));
    }));
    const weak = a.confusable[0] && a.confusable[0].similarity < 0.3;

    return h("div", { class: "atlas-dossier" },
      h("header", { class: "dossier-head" },
        h("p", { class: "au-big num", "aria-hidden": "true" }, `AU ${a.au}`),
        h("div", {}, h("h2", { class: "display dossier-title" }, a.name), h("p", { class: "muscle" }, a.muscle))),
      h("section", { class: "dsec" }, h("h3", { class: "label ink" }, "What you would see"), h("p", { class: "prose" }, a.sees)),
      h("section", { class: "dsec" }, h("h3", { class: "label ink" }, "What the instrument measures"), h("p", { class: "prose" }, a.looks),
        h("details", { class: "fold", open: true }, h("summary", {}, "Regions and directions (table view of the plate)"), regionTable)),
      h("section", { class: "dsec" },
        h("h3", { class: "label ink" }, "Easily mistaken for"),
        weak ? h("p", { class: "legend-text" }, "No other pattern overlaps strongly with this one at the assumed noise level.") : null,
        mistaken, cmp ? contrast(a, byAu.get(cmp)) : null),
      prot.length ? h("section", { class: "dsec" }, h("h3", { class: "label ink" }, "Appears in these conventional combinations"),
        h("ul", { class: "inline-list" }, prot.map((p) => h("li", {}, h("b", {}, p.label), ` (${p.aus.map((n) => `AU ${n}`).join(" + ")})`)))) : null);
  }

  function renderPanel() {
    const a = byAu.get(sel);
    tip.hide();
    clear(panel);
    panel.setAttribute("aria-labelledby", `au-tab-${sel}`);
    panel.append(h("div", { class: "atlas-main" }, plateFigure(a, cmp ? byAu.get(cmp) : null), dossier(a)));
  }

  // ---------------------------------------------------------------------------------------------- below the fold
  const combos = h("section", { class: "atlas-sec" },
    h("header", {}, h("h2", { class: "display sec-title" }, "Conventional combinations"),
      h("p", { class: "prose" }, "The FACS literature groups certain movements under familiar labels. These are expert conventions the analysis uses as weak priors when it offers a category hypothesis. They are not measurements of feeling: the same movement can occur with many feelings, or with none.")),
    h("ol", { class: "combos" }, data.prototypes.map((p) => h("li", { class: "combo" },
      h("span", { class: "combo-label" }, p.label),
      h("span", { class: "combo-aus" }, p.aus.map((n, i) => [i ? h("span", { class: "plus", "aria-hidden": "true" }, "+") : null, h("button", { class: "au-link num", type: "button", onclick: () => select(n, true), "aria-label": `Show AU ${n}` }, String(n))])),
      h("span", { class: "legend-text" }, p.note)))));

  const idn = data.identifiability, labels = idn.order.map(String);
  const pairs = [];
  for (let i = 0; i < idn.order.length; i++) for (let j = i + 1; j < idn.order.length; j++) pairs.push([idn.order[i], idn.order[j], idn.matrix[i][j]]);
  pairs.sort((p, q) => q[2] - p[2]);
  const hard = h("section", { class: "atlas-sec" },
    h("header", {}, h("h2", { class: "display sec-title" }, "What it cannot easily tell apart"),
      h("p", { class: "prose" }, `Two movements are hard to separate when they move the same regions in the same directions. The numbers are the similarity of the two expected patterns after the head-motion nuisance is projected out, at a noise level of ${idn.sigma_px} px per region (webcam class). 1 means the patterns are identical; 0 means unrelated or opposite.`)));
  const split = h("div", { class: "hard-split" });
  hard.append(split);
  const hm = heatmap(split, {
    title: "Similarity of expected patterns", labels, values: idn.matrix, valueLabel: "similarity", aria: "Similarity between every pair of action-unit patterns",
    table: { columns: ["AU", ...labels], rows: idn.matrix.map((r, i) => [labels[i], ...r.map((v) => v.toFixed(2))]) },
  });
  charts.push(hm);
  split.append(h("div", { class: "pairs" },
    h("h3", { class: "label ink" }, "The five closest pairs"),
    h("ol", {}, pairs.slice(0, 5).map(([x, y, s]) => h("li", {},
      h("span", { class: "num" }, h("button", { class: "au-link num", type: "button", onclick: () => { select(x, true); } }, String(x)), " and ", h("button", { class: "au-link num", type: "button", onclick: () => { select(y, true); } }, String(y))),
      h("span", { class: "num mist-v" }, s.toFixed(2)), h("span", { class: "muted" }, `${byAu.get(x).name} and ${byAu.get(y).name}: ${overlapWords(s)}`)))),
    h("p", { class: "legend-text" }, "When a recording leaves a close pair unresolved, the analysis reports both and the uncertainty between them instead of picking one.")));

  const limits = note("caution", h("p", {}, h("b", {}, "What the atlas does not claim. "), "The patterns describe where skin moves, in a stabilised face. They do not say what a person feels, and a real face rarely matches a template exactly: anatomy, asymmetry and the camera all differ. A match in a recording is an approximate fit with a stated uncertainty, never proof of a muscle or of an emotion."));
  body.append(combos, hard, limits);

  markTabs();
  renderPanel();
  ctx.setTitle(`AU ${sel} · Atlas`);
  return { dispose() { tip.hide(); for (const c of charts) c.destroy(); } };
}
