// Cue Lab.  A critical look at reading lies from a face, and the research instrument that belongs in its place.
// Nothing on this page produces a verdict about a person: the server refuses to return any field that looks like
// one, and this page checks again before showing a response.
import { h, svg, clear, fmt, icon, toast, debounce, clamp } from "./util.js";
import { api, ref as mkRef, STATIC } from "./api.js";
import { lineChart, forest, bars, SERIES } from "./charts.js";
import { stageHead, note, busy, errorBox } from "./ui.js";

const FORBIDDEN = new Set(["lie", "lies", "lying", "liar", "deceptive", "deception_score", "deception_prob", "truthful", "truth_score", "truth_probability", "lie_probability", "verdict", "is_lying"]);
function noVerdict(o, path = "") {
  if (Array.isArray(o)) o.forEach((v, i) => noVerdict(v, `${path}[${i}]`));
  else if (o && typeof o === "object") for (const [k, v] of Object.entries(o)) { if (FORBIDDEN.has(k.toLowerCase())) throw new Error(`A verdict-shaped field (${k}) was returned and discarded.`); noVerdict(v, `${path}/${k}`); }
  return o;
}
const call = async (p) => noVerdict(await p);
const pct = (v, d = 0) => (v == null || !isFinite(v) ? "n/a" : `${(100 * v).toFixed(d)} %`);
/** A share with enough decimals that a small value does not read as zero. */
const pcf = (v) => pct(v, v > 0 && v < 0.01 ? 2 : v < 0.1 ? 1 : 0);

function slider({ id, label, min, max, step, value, format, help, onChange }) {
  const out = h("output", { class: "num", for: id }, format(value));
  const inp = h("input", { type: "range", id, min, max, step, value, "aria-valuetext": format(value) });
  inp.addEventListener("input", () => { const v = parseFloat(inp.value); out.textContent = format(v); inp.setAttribute("aria-valuetext", format(v)); onChange && onChange(v); });
  const el = h("div", { class: "field slider" }, h("label", { for: id }, label, " ", out), inp, help ? h("p", { class: "small muted" }, help) : null);
  return { el, get: () => parseFloat(inp.value), set: (v) => { inp.value = v; out.textContent = format(v); } };
}
function numberField({ id, label, min, value, step = 1, onChange, help }) {
  const inp = h("input", { type: "number", id, min, step, value, inputmode: "decimal" });
  inp.addEventListener("input", () => onChange && onChange());
  return { el: h("div", { class: "field" }, h("label", { for: id }, label), inp, help ? h("p", { class: "small muted" }, help) : null), get: () => parseFloat(inp.value), set: (v) => { inp.value = v; } };
}
function section(id, kicker, title, lede) {
  return h("section", { class: "cue-sec", id: `cue-${id}`, "aria-labelledby": `cue-${id}-h` },
    h("header", {}, h("p", { class: "kicker" }, kicker), h("h2", { class: "display", id: `cue-${id}-h` }, title), lede ? h("p", { class: "prose lede" }, lede) : null));
}

// ================================================================================================ the page
export async function mount(host, { q }, ctx) {
  host.append(stageHead({
    kicker: "Station 04 · Cue Lab",
    title: "Can a face reveal a lie?",
    lede: "ONSET measures facial movement. It cannot tell you whether someone is lying, and nothing on this page will produce a verdict about a person. What it can do is show why, and give you the statistics a serious study of behavioural cues would need.",
  }));
  const layout = h("div", { class: "cue" });
  const rail = h("nav", { class: "cue-rail", "aria-label": "On this page" });
  const body = h("div", { class: "cue-body" });
  layout.append(rail, body);
  host.append(layout);
  const parts = [];
  body.append(busy("Loading the evidence"));
  let ev = null;
  try { ev = await call(api.cue.evidence()); } catch (e) { clear(body); body.append(errorBox(e)); return; }
  clear(body);

  const SECTIONS = [
    ["position", "The position"], ["evidence", "What the evidence says"], ["base", "Base rates"], ["effect", "Effect size"],
    ["counts", "Small counts"], ["forking", "Forking paths"], ["study", "A segment study"], ["law", "Law and consent"], ["protocol", "What a study needs"],
  ];
  for (const [id, label] of SECTIONS) rail.append(h("a", { href: `#cue-${id}`, onclick: (e) => { e.preventDefault(); document.getElementById(`cue-${id}`).scrollIntoView({ behavior: "smooth", block: "start" }); } }, label));

  position(body, ev);
  evidence(body, ev);
  parts.push(baseRates(body));
  parts.push(effectSize(body));
  parts.push(counts(body));
  parts.push(forking(body));
  parts.push(await segmentStudy(body, q, ctx));
  law(body, ev);
  protocol(body, ev);
  references(body, ev);

  const io = new IntersectionObserver((es) => { for (const e of es) if (e.isIntersecting) { for (const a of rail.children) a.classList.toggle("on", a.getAttribute("href") === `#${e.target.id}`); } }, { rootMargin: "-20% 0px -70% 0px" });
  body.querySelectorAll(".cue-sec").forEach((s) => io.observe(s));
  return { dispose: () => { io.disconnect(); parts.forEach((p) => p && p.destroy && p.destroy()); } };
}

// ------------------------------------------------------------------------------------------------ 1. position
function position(body, ev) {
  const P = ev.position;
  const s = section("position", "The position", "What ONSET will and will not say");
  s.append(h("p", { class: "pull" }, P.headline),
    h("div", { class: "two-col" },
      h("div", {}, h("h3", { class: "label ink" }, "What it can do"), h("ul", { class: "ticks" }, P.can.map((t) => h("li", {}, icon("check"), t)))),
      h("div", {}, h("h3", { class: "label ink" }, "What it cannot do"), h("ul", { class: "ticks no" }, P.cannot.map((t) => h("li", {}, icon("warn"), t))))),
    h("div", { class: "two-col" },
      h("div", {}, h("h3", { class: "label ink" }, "Why"), h("ul", { class: "plain-list" }, P.why.map((t) => h("li", {}, t)))),
      h("div", {}, h("h3", { class: "label ink" }, "What it offers instead"), h("ul", { class: "plain-list" }, P.instead.map((t) => h("li", {}, t))))),
    note("pen", h("b", {}, "Bottom line. "), ev.bottom_line));
  body.append(s);
}

// ------------------------------------------------------------------------------------------------ 2. the evidence
const STAMP = { validated_limit: "Validated limit", mixed: "Mixed", hypothesis: "Hypothesis", unsupported: "Not supported", confound: "Confound" };
function evidence(body, ev) {
  const s = section("evidence", "The evidence", "What the sources say about each cue", "Every row is a claim people make about faces and lies, what ONSET can measure about it, and how far the sources opened during this build support it. A reference marked ‘not opened’ is a pointer, not a verified fact.");
  const list = h("div", { class: "dossier-list" });
  for (const e of ev.evidence) {
    const refs = e.ref_details.map((r) => h("li", { class: "small" }, h("span", { class: `opened ${r.opened ? "yes" : "no"}` }, r.opened ? "opened" : "not opened"), " ", r.cite, r.doi ? h("span", { class: "muted" }, ` doi:${r.doi}`) : null, r.via ? h("span", { class: "muted" }, ` — quoted via ${r.via}`) : null));
    list.append(h("article", { class: "dossier" },
      h("div", { class: "dossier-stamp" }, h("span", { class: `stamp-status ${e.status}`, title: e.status_text }, STAMP[e.status] || e.status)),
      h("div", { class: "dossier-main" },
        h("h3", {}, e.cue),
        h("p", { class: "claim" }, h("span", { class: "label" }, "The claim "), e.claim),
        h("p", { class: "measure small" }, h("span", { class: "label" }, "What ONSET measures "), e.onset_measure),
        h("details", { class: "fold" }, h("summary", {}, "What is known, and how sure"),
          h("ul", { class: "plain-list" }, e.known.map((k) => h("li", {}, k))),
          h("p", { class: "assess" }, h("b", {}, "Assessment. "), e.assessment),
          refs.length ? h("ul", { class: "refs" }, refs) : null))));
  }
  s.append(list);
  body.append(s);
}

// ------------------------------------------------------------------------------------------------ 3. base rates
function baseRates(body) {
  const s = section("base", "Base rates", "Even a good detector mostly accuses the innocent", "Pick how well a hypothetical detector does on each kind of person, and how often the thing it looks for occurs. The grid is 1,000 people.");
  const sens = slider({ id: "br-sens", label: "Sensitivity", min: 0.5, max: 0.99, step: 0.01, value: 0.9, format: (v) => pct(v), help: "How often it flags someone who is lying.", onChange: () => run() });
  const spec = slider({ id: "br-spec", label: "Specificity", min: 0.5, max: 0.99, step: 0.01, value: 0.9, format: (v) => pct(v), help: "How often it correctly leaves a truthful person alone.", onChange: () => run() });
  const prev = slider({ id: "br-prev", label: "How often is the condition present?", min: 0.005, max: 0.5, step: 0.005, value: 0.05, format: (v) => pct(v, 1), help: "In many real settings lying is the exception, not the rule.", onChange: () => run() });
  const grid = svg("svg", { class: "dots", viewBox: "0 0 600 230", role: "img", "aria-label": "1000 people" });
  const dots = [];
  const cols = 50, size = 10.4, gap = 1.6;
  for (let i = 0; i < 1000; i++) {
    const c = svg("circle", { cx: (i % cols) * (size + gap) + size / 2 + 1, cy: Math.floor(i / cols) * (size + gap) + size / 2 + 1, r: size / 2 - 0.5, class: "dot-tn" });
    dots.push(c); grid.append(c);
  }
  const keyLine = h("ul", { class: "legend dotkey" });
  const verdict = h("p", { class: "prose big-sentence", "aria-live": "polite" });
  const chartHost = h("div", { class: "chart-host" });
  const out = h("div", { class: "lab-out" }, grid, keyLine, verdict, chartHost);
  s.append(h("div", { class: "lab" }, h("div", { class: "lab-controls" }, sens.el, spec.el, prev.el), out));
  body.append(s);
  let latest = 0, chart = null;
  async function run() {
    const my = ++latest;
    let r;
    try { r = await call(api.cue.predictive({ sensitivity: sens.get(), specificity: spec.get(), prevalence: prev.get() })); } catch (e) { clear(out); out.append(errorBox(e)); return; }
    if (my !== latest) return;
    const v = r.values, k = v.per_1000;
    const tp = Math.round(k.true_positive), fp = Math.round(k.false_positive), fn = Math.round(k.false_negative), tn = 1000 - tp - fp - fn;
    const order = [["dot-tp", tp], ["dot-fp", fp], ["dot-fn", fn], ["dot-tn", tn]];
    let i = 0;
    for (const [cls, n] of order) for (let j = 0; j < n && i < 1000; j++) dots[i++].setAttribute("class", cls);
    grid.setAttribute("aria-label", `1000 people: ${tp} flagged and affected, ${fp} flagged and not affected, ${fn} affected but not flagged, ${tn} neither`);
    clear(keyLine);
    keyLine.append(
      h("li", {}, h("i", { class: "k-tp" }), `${tp} flagged, and the condition is present`),
      h("li", {}, h("i", { class: "k-fp" }), `${fp} flagged, and it is not`),
      h("li", {}, h("i", { class: "k-fn" }), `${fn} missed`),
      h("li", {}, h("i", { class: "k-tn" }), `${tn} correctly left alone`));
    const flagged = tp + fp;
    verdict.replaceChildren(`Of the ${flagged} people flagged, `, h("b", {}, `${tp}`), ` actually have the condition: a positive call is right `, h("b", {}, pct(v.ppv)), " of the time. ",
      v.ppv < 0.5 ? "Most flagged people are wrongly flagged." : "");
    const c = r.curve, pts = (arr) => c.prevalence.map((p, j) => [p, arr[j]]).filter((p) => p[1] != null);
    chartHost.replaceChildren();
    if (chart) chart.destroy();
    chart = lineChart(chartHost, {
      title: "How a positive call’s reliability depends on how common the condition is", height: 250, margin: { l: 56, r: 30, t: 14, b: 40 },
      x: { domain: [0, 1], title: "How often the condition is present", format: (x) => pct(x), ticks: [0, 0.25, 0.5, 0.75, 1] },
      y: { domain: [0, 1], title: "Chance a call is right", format: (x) => pct(x), ticks: [0, 0.25, 0.5, 0.75, 1] },
      series: [{ name: "A flagged person really has it (PPV)", points: pts(c.ppv), color: SERIES[1] }, { name: "An unflagged person really does not (NPV)", points: pts(c.npv), color: SERIES[0] }],
      dots: [{ x: v.prevalence, y: v.ppv, color: SERIES[1], label: `PPV ${pct(v.ppv)}` }],
      rules: [{ x: v.prevalence }],
      table: { columns: ["Prevalence", "PPV", "NPV"], rows: c.prevalence.filter((_, j) => j % 10 === 0).map((p, j) => [pct(p), pct(c.ppv[j * 10]), pct(c.npv[j * 10])]) },
    });
  }
  run();
  return { destroy: () => chart && chart.destroy() };
}

// ------------------------------------------------------------------------------------------------ 4. effect size
function effectSize(body) {
  const s = section("effect", "Effect size", "What a cue’s separation buys you", "Suppose some measurable cue differs between two groups of people by d standard deviations. How well could it tell them apart? The curve is a model, not data: two bell curves, d apart.");
  const d = slider({ id: "es-d", label: "Standardised difference (d)", min: 0, max: 3, step: 0.05, value: 0.4, format: (v) => v.toFixed(2), help: "0.2 is conventionally ‘small’, 0.5 ‘medium’, 0.8 ‘large’.", onChange: () => run() });
  const prev = slider({ id: "es-prev", label: "How often the condition is present", min: 0.01, max: 0.5, step: 0.01, value: 0.1, format: (v) => pct(v), onChange: () => run() });
  const chartHost = h("div", { class: "chart-host" });
  const text = h("div", { class: "lab-text", "aria-live": "polite" });
  s.append(h("div", { class: "lab" }, h("div", { class: "lab-controls" }, d.el, prev.el, text), h("div", { class: "lab-out" }, chartHost)));
  body.append(s);
  let latest = 0, chart = null;
  async function run() {
    const my = ++latest;
    let r;
    try { r = await call(api.cue.roc({ d: d.get(), prevalence: prev.get() })); } catch (e) { clear(chartHost); chartHost.append(errorBox(e)); return; }
    if (my !== latest) return;
    chartHost.replaceChildren();
    if (chart) chart.destroy();
    const pts = r.roc.fpr.map((x, i) => [x, r.roc.tpr[i]]);
    const b = r.best;
    chart = lineChart(chartHost, {
      title: "ROC curve for the modelled cue", height: 300, margin: { l: 56, r: 24, t: 14, b: 42 },
      x: { domain: [0, 1], title: "People wrongly flagged", format: (x) => pct(x), ticks: [0, 0.25, 0.5, 0.75, 1] },
      y: { domain: [0, 1], title: "People correctly flagged", format: (x) => pct(x), ticks: [0, 0.25, 0.5, 0.75, 1] },
      series: [{ name: "The cue", points: pts, color: SERIES[1] }, { name: "Guessing", points: [[0, 0], [1, 1]], color: "var(--faint)", dash: "4 4", width: 1.5 }],
      dots: [{ x: 1 - b.specificity, y: b.sensitivity, color: SERIES[1], label: "most accurate cut-off" }],
      table: { columns: ["False-positive rate", "True-positive rate"], rows: pts.filter((_, i) => i % 12 === 0).map(([x, y]) => [pct(x), pct(y)]) },
    });
    text.replaceChildren(
      h("p", { class: "prose" }, "Area under the curve: ", h("b", { class: "num" }, r.auc.toFixed(2)), " (0.50 is guessing)."),
      h("p", { class: "prose" }, `At the cut-off with the highest overall accuracy, with the condition present in ${pct(prev.get())} of people: it flags ${pcf(b.sensitivity)} of them and wrongly flags ${pcf(1 - b.specificity)} of the rest. A positive call is right ${pct(b.ppv)} of the time.`),
      b.sensitivity < 0.05 ? h("p", { class: "prose" }, `Look at what that cut-off does: it flags almost no one, and is right ${pct(b.accuracy)} of the time only because the condition is rare. Overall accuracy rewards staying silent when something is uncommon, so a weak cue can look accurate while finding nobody.`) : null,
      h("p", { class: "small muted" }, "For one cue to reach 60 / 70 / 80 / 90 % accuracy on an even split it would need d = ", Object.values(r.d_for_accuracy).map((x) => x.toFixed(2)).join(" / "), "."));
  }
  run();
  return { destroy: () => chart && chart.destroy() };
}

// ------------------------------------------------------------------------------------------------ 5. small counts
function counts(body) {
  const s = section("counts", "Small counts", "Three events against five is not a difference", "Counting events in two stretches of a recording is the basic measurement of a cue study. With the few events that fit in a minute of video, the uncertainty is large; the intervals below are exact, not approximations.");
  const k1 = numberField({ id: "k1", label: "Events in segment A", min: 0, value: 3, onChange: () => run() });
  const s1 = numberField({ id: "s1", label: "Seconds observed in A", min: 1, value: 60, onChange: () => run() });
  const k2 = numberField({ id: "k2", label: "Events in segment B", min: 0, value: 5, onChange: () => run() });
  const s2 = numberField({ id: "s2", label: "Seconds observed in B", min: 1, value: 60, onChange: () => run() });
  const chartHost = h("div", { class: "chart-host" });
  const text = h("div", { class: "lab-text", "aria-live": "polite" });
  s.append(h("div", { class: "lab" }, h("div", { class: "lab-controls two" }, k1.el, s1.el, k2.el, s2.el), h("div", { class: "lab-out" }, chartHost, text)));
  body.append(s);
  let latest = 0, charts = [];
  const run = debounce(async () => {
    const my = ++latest;
    const a = { k1: Math.max(0, Math.round(k1.get() || 0)), s1: Math.max(1, s1.get() || 60), k2: Math.max(0, Math.round(k2.get() || 0)), s2: Math.max(1, s2.get() || 60) };
    let r;
    try { r = await call(api.cue.rates(a)); } catch (e) { clear(chartHost); chartHost.append(errorBox(e)); return; }
    if (my !== latest) return;
    charts.forEach((c) => c.destroy()); charts = [];
    chartHost.replaceChildren();
    const ra = r.a, rb = r.b, t = r.test;
    const hiMax = Math.max(ra.hi, rb.hi, 1) * 1.2;
    charts.push(forest(chartHost, {
      title: "Event rate per minute, with exact 95 % intervals", rowHeight: 34, labelWidth: 120,
      x: { domain: [Math.max(0.05, Math.min(0.5, ra.rate_per_min / 4 || 0.5)), Math.max(hiMax, 10)], log: true, title: "events per minute", format: (v) => String(v), ref: null },
      rows: [
        { label: "Segment A", est: Math.max(ra.rate_per_min, 0.06), lo: Math.max(ra.lo, 0.06), hi: ra.hi, text: `${ra.rate_per_min.toFixed(1)} /min`, strong: true, color: SERIES[0], rows: [{ value: `${ra.k} events`, label: `in ${(ra.minutes * 60).toFixed(0)} s` }, { value: `${ra.lo.toFixed(1)} to ${ra.hi.toFixed(1)}`, label: "per minute, 95 % interval" }] },
        { label: "Segment B", est: Math.max(rb.rate_per_min, 0.06), lo: Math.max(rb.lo, 0.06), hi: rb.hi, text: `${rb.rate_per_min.toFixed(1)} /min`, strong: true, color: SERIES[1], rows: [{ value: `${rb.k} events`, label: `in ${(rb.minutes * 60).toFixed(0)} s` }, { value: `${rb.lo.toFixed(1)} to ${rb.hi.toFixed(1)}`, label: "per minute, 95 % interval" }] },
      ],
      table: { columns: ["Segment", "Events", "Minutes", "Rate per minute", "95 % low", "95 % high"], rows: [["A", ra.k, ra.minutes.toFixed(2), ra.rate_per_min.toFixed(2), ra.lo.toFixed(2), ra.hi.toFixed(2)], ["B", rb.k, rb.minutes.toFixed(2), rb.rate_per_min.toFixed(2), rb.lo.toFixed(2), rb.hi.toFixed(2)]] },
    }));
    const rr = t.rr;
    text.replaceChildren(
      t.note ? h("p", { class: "prose" }, t.note) : h("p", { class: "prose" }, "Rate ratio A : B = ", h("b", { class: "num" }, isFinite(rr) ? rr.toFixed(2) : "undefined"), ` (95 % interval ${t.lo.toFixed(2)} to ${isFinite(t.hi) ? t.hi.toFixed(2) : "no upper limit"}); exact two-sided p = `, h("b", { class: "num" }, t.p < 0.001 ? "< 0.001" : t.p.toFixed(3)), "."),
      h("p", { class: "prose" }, t.p < 0.05 ? "That is below the usual 0.05 line — but see the next section: one test is rarely the only one." : "That is nowhere near convincing. The interval for the ratio is wide enough to include no difference at all."),
      h("p", { class: "small muted" }, `For segments of these lengths, at least ${r.min_events} events in total would be needed for any difference to reach p < 0.05, even if every event fell in one segment.`));
  }, 180);
  run();
  return { destroy: () => charts.forEach((c) => c.destroy()) };
}

// ------------------------------------------------------------------------------------------------ 6. forking paths
function forking(body) {
  const s = section("forking", "Forking paths", "Test enough cues and something will look significant", "A person who behaves identically in two segments still shows a ‘significant’ difference in some cue now and then. The more cues are tested, the more often.");
  const n = slider({ id: "fp-n", label: "Number of cues tested", min: 1, max: 40, step: 1, value: 9, format: (v) => String(v), onChange: () => run() });
  const demoBtn = h("button", { class: "btn", type: "button", onclick: () => demo() }, icon("record"), "Simulate a person who behaves the same in both minutes");
  const chartHost = h("div", { class: "chart-host" });
  const demoHost = h("div", { class: "chart-host demo" });
  const text = h("p", { class: "prose big-sentence", "aria-live": "polite" });
  s.append(h("div", { class: "lab" }, h("div", { class: "lab-controls" }, n.el, demoBtn), h("div", { class: "lab-out" }, text, chartHost, demoHost)));
  body.append(s);
  let latest = 0, chart = null, demoChart = null;
  const run = debounce(async () => {
    const my = ++latest;
    let r;
    try { r = await call(api.cue.forking({ n_tests: n.get() })); } catch (e) { clear(chartHost); chartHost.append(errorBox(e)); return; }
    if (my !== latest) return;
    text.replaceChildren(`Testing ${r.n_tests} independent cue${r.n_tests === 1 ? "" : "s"} at the 5 % level, the chance that at least one looks significant when nothing is going on is `, h("b", {}, pct(r.p_any, 0)), ".");
    chartHost.replaceChildren(); chart && chart.destroy();
    const pts = Array.from({ length: 40 }, (_, i) => [i + 1, 1 - 0.95 ** (i + 1)]);
    chart = lineChart(chartHost, {
      title: "Chance of at least one false alarm", height: 230, margin: { l: 56, r: 26, t: 12, b: 40 },
      x: { domain: [1, 40], title: "cues tested", ticks: [1, 10, 20, 30, 40], format: (v) => String(v) },
      y: { domain: [0, 1], title: "chance", ticks: [0, 0.25, 0.5, 0.75, 1], format: (v) => pct(v) },
      series: [{ name: "Chance of at least one ‘significant’ result", points: pts, color: SERIES[1] }],
      dots: [{ x: r.n_tests, y: r.p_any, color: SERIES[1], label: pct(r.p_any) }],
      table: { columns: ["Cues tested", "Chance"], rows: pts.filter((_, i) => [0, 4, 8, 13, 19, 29, 39].includes(i)).map(([x, y]) => [x, pct(y)]) },
    });
  }, 120);
  async function demo() {
    demoBtn.disabled = true;
    demoHost.replaceChildren(busy("Simulating 1,500 such people"));
    try {
      const r = await call(api.cue.forking({ n_tests: 9, demo: true }));
      const d = r.demo;
      demoHost.replaceChildren(); demoChart && demoChart.destroy();
      demoChart = bars(demoHost, {
        title: "Simulated people with identical behaviour in both minutes, nine cues each", labelWidth: 230, x: { domain: [0, 0.5], format: (v) => pct(v), ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5], title: "share of people with at least one ‘significant’ cue" },
        rows: [{ label: "Uncorrected tests", value: d.sim.p_any_uncorrected, text: pct(d.sim.p_any_uncorrected), color: SERIES[1] }, { label: "Corrected for nine tests (Holm)", value: d.sim.p_any_holm, text: pct(d.sim.p_any_holm), color: SERIES[0] }],
        caption: d.caption,
        table: { columns: ["Method", "Share"], rows: [["Uncorrected", pct(d.sim.p_any_uncorrected, 1)], ["Holm-corrected", pct(d.sim.p_any_holm, 1)]] },
      });
    } catch (e) { demoHost.replaceChildren(errorBox(e)); }
    demoBtn.disabled = false;
  }
  run();
  return { destroy: () => { chart && chart.destroy(); demoChart && demoChart.destroy(); } };
}

// ------------------------------------------------------------------------------------------------ 7. segment study
async function segmentStudy(body, q, ctx) {
  const s = section("study", "A segment study", "The research alternative: compare two stretches of one recording", "Counts, rates corrected for the time each facial region could actually be seen, and movement activity — within one person, with exact small-count statistics, and every comparison corrected for the number of comparisons. The result says whether measured behaviour differed. It cannot say why.");
  body.append(s);
  if (STATIC) { s.append(note("caution", "This needs the local ONSET server and an analysed recording.")); return null; }
  let sessions = [], lib = [];
  try { sessions = await api.results(); } catch { /* none */ }
  try { lib = (await api.library()).specimens || []; } catch { /* none */ }
  const select = h("select", { id: "st-rec", "aria-label": "Recording" });
  const addGroup = (label, items) => { if (items.length) select.append(h("optgroup", { label }, items.map((o) => h("option", { value: o.value }, o.label)))); };
  addGroup("Your analysed recordings", sessions.map((r) => ({ value: `result:${r.id}`, label: `${r.label || "Recording"} · ${fmt.date(r.created)}` })));
  addGroup("Specimens (synthetic)", lib.filter((x) => x.duration >= 6).map((x) => ({ value: `specimen:${x.id}`, label: x.title })));
  if (!select.options.length) { s.append(note("info", "Analyse a recording first (Capture), or open a specimen. The segment study works on analysed recordings. For a recording that has two natural stretches, choose “A still baseline, then a silent counting task” when you record.")); return null; }
  if (q.result) select.value = `result:${q.result}`; else if (q.specimen) select.value = `specimen:${q.specimen}`;
  else if (sessions.length) select.value = `result:${sessions[0].id}`;
  const a0 = numberField({ id: "sa0", label: "Segment A starts (s)", min: 0, step: 0.1, value: 0, onChange: () => paintStrip() });
  const a1 = numberField({ id: "sa1", label: "Segment A ends (s)", min: 0, step: 0.1, value: 4, onChange: () => paintStrip() });
  const b0 = numberField({ id: "sb0", label: "Segment B starts (s)", min: 0, step: 0.1, value: 4, onChange: () => paintStrip() });
  const b1 = numberField({ id: "sb1", label: "Segment B ends (s)", min: 0, step: 0.1, value: 8, onChange: () => paintStrip() });
  const la = h("input", { type: "text", id: "sla", value: "Baseline", maxlength: 30, "aria-label": "Name of segment A" });
  const lb = h("input", { type: "text", id: "slb", value: "Target", maxlength: 30, "aria-label": "Name of segment B" });
  const strip = h("div", { class: "seg-strip", role: "img", "aria-label": "The two segments along the recording" });
  const go = h("button", { class: "btn primary", type: "button", onclick: () => compare() }, icon("check"), "Compare the segments");
  const out = h("div", { class: "study-out", "aria-live": "polite" });
  let current = null, charts = [];
  s.append(h("div", { class: "lab study" },
    h("div", { class: "lab-controls two" },
      h("div", { class: "field wide" }, h("label", { for: "st-rec" }, "Recording"), select),
      h("div", { class: "field" }, h("label", { for: "sla" }, "Name of A"), la), h("div", { class: "field" }, h("label", { for: "slb" }, "Name of B"), lb),
      a0.el, a1.el, b0.el, b1.el),
    strip, h("div", { class: "actions" }, go), out));

  const pick = () => { const [kind, id] = select.value.split(":"); return kind === "result" ? { result: id } : { specimen: id }; };
  async function loadCurrent() {
    const [kind, id] = select.value.split(":");
    try {
      current = await api.result(mkRef(kind === "result" ? "session" : "library", id));
    } catch (e) { current = null; strip.replaceChildren(errorBox(e)); return; }
    const dur = current.clip.duration_s;
    const segs = (current.provenance && current.provenance.segments) || [];
    if (segs.length >= 2) { a0.set(segs[0].t0); a1.set(segs[0].t1); b0.set(segs[1].t0); b1.set(segs[1].t1); la.value = segs[0].label; lb.value = segs[1].label; }
    else { a0.set(0); a1.set(+(dur / 2).toFixed(1)); b0.set(+(dur / 2).toFixed(1)); b1.set(+dur.toFixed(1)); }
    paintStrip();
  }
  function paintStrip() {
    if (!current) return;
    const dur = current.clip.duration_s, W = 100;
    const pos = (t) => `${clamp(t / dur, 0, 1) * W}%`;
    clear(strip);
    const seg = (cls, t0, t1, label) => h("i", { class: `sg ${cls}`, style: { left: pos(t0), width: `calc(${pos(t1)} - ${pos(t0)})` } }, h("b", {}, label));
    strip.append(seg("a", a0.get(), a1.get(), "A"), seg("b", b0.get(), b1.get(), "B"));
    for (const e of current.events) strip.append(h("u", { class: `ev ${e.status}`, style: { left: pos(e.t_apex) }, title: `${e.status} at ${e.t_apex.toFixed(2)} s` }));
    strip.append(h("span", { class: "ax l num small" }, "0 s"), h("span", { class: "ax r num small" }, `${dur.toFixed(1)} s`));
  }
  select.addEventListener("change", loadCurrent);
  await loadCurrent();

  async function compare() {
    go.disabled = true;
    charts.forEach((c) => c.destroy()); charts = [];
    out.replaceChildren(busy("Comparing the segments"));
    try {
      const r = await call(api.cue.segments({ ...pick(), a: { label: la.value || "A", t0: a0.get(), t1: a1.get() }, b: { label: lb.value || "B", t0: b0.get(), t1: b1.get() } }));
      renderStudy(out, r, charts);
    } catch (e) { out.replaceChildren(errorBox(e)); }
    go.disabled = false;
  }
  return { destroy: () => charts.forEach((c) => c.destroy()) };
}

function renderStudy(out, r, charts) {
  const c = r.comparison, A = c.a, B = c.b;
  clear(out);
  out.append(h("p", { class: "prose big-sentence" }, c.summary));
  // what was observed
  const seg = (S, color) => h("div", { class: "seg-card" }, h("h4", {}, h("i", { style: { background: color } }), S.label), h("dl", { class: "facts" },
    [["Time", `${S.seconds.toFixed(1)} s`], ["Face in view", pct(S.face_found_fraction)], ["Movement events", String(S.n_events.all)], ["Blinks", String(S.n_blinks)], ["Speech", pct(S.speech_fraction)], ["Eye region observable", `${S.observable_seconds.eye.toFixed(1)} s`], ["Mouth region observable", `${S.observable_seconds.mouth.toFixed(1)} s`]]
      .map(([k, v]) => h("div", {}, h("dt", {}, k), h("dd", { class: "num" }, v)))));
  out.append(h("div", { class: "two-col" }, seg(A, SERIES[0]), seg(B, SERIES[1])));
  // rate tests as a forest plot of rate ratios
  const rate = c.tests.filter((t) => t.kind === "rate");
  const rows = rate.map((t) => ({
    label: t.name.replace("Movement events", "Events"), est: isFinite(t.rate_ratio) && t.rate_ratio > 0 ? t.rate_ratio : null, lo: t.rr_lo > 0 ? t.rr_lo : 0.02, hi: isFinite(t.rr_hi) ? t.rr_hi : 100,
    text: `p ${t.p_holm < 0.001 ? "< 0.001" : t.p_holm.toFixed(2)}${t.survives ? " ●" : ""}`, strong: t.survives, dim: t.a.k + t.b.k === 0,
    color: t.survives ? SERIES[1] : "var(--ink)",
    rows: [{ value: `${t.a.k} v ${t.b.k}`, label: "events in A v B" }, { value: isFinite(t.rate_ratio) ? t.rate_ratio.toFixed(2) : "undefined", label: "rate ratio A : B" }, { value: t.p.toFixed(3), label: "p, uncorrected" }, { value: t.p_holm.toFixed(3), label: `p, corrected for ${c.family_size} comparisons` }],
  }));
  charts.push(forest(out, {
    title: "Rate of events in A relative to B (1 = no difference)", rowHeight: 28, labelWidth: 200,
    x: { domain: [0.05, 20], log: true, ref: 1, title: "rate ratio A : B", format: (v) => String(v), ticks: [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20] },
    rows, caption: "Each line is an exact 95 % interval; arrows mean it runs off the scale. The p on the right is corrected for the whole family of comparisons; ● marks one that stays below 0.05.",
    table: { columns: ["Comparison", "A", "B", "Rate ratio", "Interval", "p", "Corrected p"], rows: rate.map((t) => [t.name, t.a.k, t.b.k, isFinite(t.rate_ratio) ? t.rate_ratio.toFixed(2) : "n/a", `${t.rr_lo.toFixed(2)} to ${isFinite(t.rr_hi) ? t.rr_hi.toFixed(2) : "∞"}`, t.p.toFixed(3), t.p_holm.toFixed(3)]) },
  }));
  // activity tests
  const act = c.tests.filter((t) => t.kind === "activity");
  out.append(h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, "Movement activity per region"), h("p", { class: "legend-text" }, "Average measured movement in each region (blinks and speech excluded), compared with a test that keeps the recording’s own rhythm.")),
    h("table", { class: "table-view tv-plain" }, h("thead", {}, h("tr", {}, ["Region", "A", "B", "Difference", "Effect", "p", "Corrected p"].map((x, i) => h("th", { class: i ? "r" : "" }, x)))),
      h("tbody", {}, act.map((t) => h("tr", { class: t.survives ? "sig" : "" }, h("td", {}, t.name.replace("Movement activity, ", "")), h("td", { class: "r num" }, fmtv(t.a.mean)), h("td", { class: "r num" }, fmtv(t.b.mean)), h("td", { class: "r num" }, fmtv(t.diff)), h("td", { class: "r num" }, fmtv(t.effect, 2)), h("td", { class: "r num" }, t.p.toFixed(3)), h("td", { class: "r num" }, t.p_holm.toFixed(3))))))));
  out.append(h("p", { class: "small muted" }, c.statement, ` With ${c.total_events} events in total, at least ${c.min_events_needed} would be needed for segments of these lengths to show any difference.`));
  // the audit
  out.append(h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, "Which sentences could be defended?"), h("p", { class: "legend-text" }, "For this data, the claims a reader might want to make, and whether the analysis supports them.")),
    h("div", { class: "audit" }, r.audit.map((a) => h("article", { class: `audit-card ${a.status}` },
      h("h4", {}, a.claim), h("span", { class: `stamp-status ${a.status === "supported" ? "validated_limit" : a.status === "conditional" ? "mixed" : "unsupported"}` }, a.status === "supported" ? "Supported" : a.status === "conditional" ? "Only with care" : "Not supported"),
      h("ul", { class: "plain-list small" }, a.why.map((w) => h("li", {}, w))),
      a.would_need && a.would_need.length ? h("details", { class: "fold" }, h("summary", {}, "What it would take"), h("ul", { class: "plain-list small" }, a.would_need.map((w) => h("li", {}, w)))) : null)))));
}
const fmtv = (v, d = 3) => (v == null || !isFinite(v) ? "n/a" : v.toFixed(d));

// ------------------------------------------------------------------------------------------------ 8. law, 9. protocol, references
function law(body, ev) {
  const s = section("law", "Law and consent", "Rules that apply before anything is recorded", "A face video is personal data, and several jurisdictions restrict emotion inference. This is a pointer, not legal advice; the texts below were not opened during this build.");
  s.append(h("div", { class: "reg-list" }, ev.regulation.map((r) => h("article", { class: "reg" }, h("h3", {}, r.topic), h("p", { class: "prose" }, r.text)))));
  body.append(s);
}
function protocol(body, ev) {
  const s = section("protocol", "What a study needs", "Before any of this could carry weight", "A checklist for a pre-registered behavioural-cue study. Tick them off for your own protocol; nothing is stored.");
  s.append(h("ol", { class: "protocol" }, ev.protocol.map((p, i) => h("li", {}, h("label", { class: "check" }, h("input", { type: "checkbox" }), h("span", {}, h("b", { class: "num" }, `${i + 1}. `), p.text))))));
  body.append(s);
}
function references(body, ev) {
  const all = Object.entries(ev.refs);
  const s = h("section", { class: "cue-sec refs-sec" }, h("header", {}, h("h2", { class: "display" }, "References")),
    h("ul", { class: "refs" }, all.map(([id, r]) => h("li", { class: "small" }, h("span", { class: `opened ${r.opened ? "yes" : "no"}` }, r.opened ? "opened" : "not opened"), " ", r.cite, r.doi ? h("span", { class: "muted" }, ` doi:${r.doi}`) : null, h("br"), h("span", { class: "muted" }, r.finding)))));
  body.append(s);
}
