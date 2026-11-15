// Evidence: what has been measured about the instrument, on what, and what has not been measured.
// Everything here comes from data/evidence/*.json, produced by the calibration scripts; nothing is typed in by hand.
// The numbers describe synthetic phantoms.  They calibrate the instrument; they say nothing about people.
import { h, svg, clear, fmt, icon } from "./util.js";
import { api } from "./api.js";
import { lineChart, forest, bars, SERIES } from "./charts.js";
import { stageHead, note, busy, errorBox } from "./ui.js";

const pc = (v, d = 0) => (v == null || !isFinite(v) ? "n/a" : `${(100 * v).toFixed(d)}\u00a0%`);
const n2 = (v, d = 2) => (v == null || !isFinite(v) ? "n/a" : v.toFixed(d));
const CAM = { webcam30: "Webcam 30 fps", phone30: "Phone 30 fps (compressed)", webcam24_lowlight: "Webcam, dim light, 24 fps", webcam15: "Slow webcam 15 fps", highspeed100: "High-speed 100 fps", all: "All cameras" };
const NUIS = { still: "Held still", normal: "Ordinary (blinks, small nods)", busy: "Restless head, more blinking" };

function sec(id, kicker, title, lede) {
  return h("section", { class: "ev-sec", id: `ev-${id}`, "aria-labelledby": `ev-${id}-h` },
    h("header", {}, h("p", { class: "kicker" }, kicker), h("h2", { class: "display", id: `ev-${id}-h` }, title), lede ? h("p", { class: "prose lede" }, lede) : null));
}
/** A ledger line: a measured number, then what it means and what it does not. */
function line(value, unit, title, text, label) {
  return h("div", { class: "ledger-line" },
    h("div", { class: "ledger-num" }, h("b", { class: "num" }, value), h("span", { class: "unit" }, unit)),
    h("div", {}, h("h3", {}, title), h("p", { class: "prose" }, text), label ? h("p", { class: "small muted" }, label) : null));
}

export async function mount(host, params, ctx) {
  host.append(stageHead({
    kicker: "Station 05 · Evidence",
    title: "What has been measured, and on what",
    lede: "These are measurements of the instrument against scripted ground truth on synthetic faces: calibration, not accuracy on people. They show what the instrument finds, what it misses, how often it speaks up with nothing to say, and how far its probabilities can be trusted — and where they cannot.",
  }));
  const body = h("div", { class: "evidence" }, busy("Reading the evidence files"));
  host.append(body);
  let ev;
  try { ev = await api.evidence(); } catch (e) { clear(body); body.append(errorBox(e)); return; }
  clear(body);
  const E = ev.phantom_eval;
  const parts = [];
  if (!E) {
    body.append(note("caution", h("b", {}, "No evidence has been generated yet. "), "Run ", h("code", {}, "python scripts/build_phantom_eval.py"), " and ", h("code", {}, "python scripts/calibrate_engine.py"), " to produce it."));
    realData(body, ev);
    return;
  }
  body.append(note("pen", h("b", {}, "Read this first. "), E.caveat));
  ledger(body, E);
  parts.push(...spotting(body, E));
  parts.push(...sensitivity(body, E));
  parts.push(...reliability(body, E));
  parts.push(...uncertainty(body, E));
  controls(body, E);
  grid(body, E);
  learnRuns(body, ev, parts);
  realData(body, ev);
  return { dispose: () => parts.forEach((p) => p && p.destroy && p.destroy()) };
}

// ------------------------------------------------------------------------------------------------ the ledger
function biasPhrase(r) {
  if (r == null || !isFinite(r)) return "";
  if (r < 0.95) return `so it reads about ${Math.round((1 - r) * 100)} % low`;
  if (r > 1.05) return `so it reads about ${Math.round((r - 1) * 100)} % high`;
  return "so it reads close to the truth";
}
function ledger(body, E) {
  const S = E.summary;
  const s = sec("ledger", "Instrument readings", "Five readings on faces the calibration never saw, and one correction", S.corpus);
  const L = h("div", { class: "ledger-lines" });
  L.append(...[
    line(pc(S.recall_test), "found", "Scripted movements found", `At the standard threshold, ${pc(S.recall_test)} of the scripted movements were found with their timing overlapping the true interval by at least half (${pc(S.recall_loose_test)} when only the apex time must agree). The rest were too small or too brief for the camera or were hidden by blinks, speech or head motion.`, "Test faces only. Scripted amplitudes were drawn from a range that includes movements below the noise floor on purpose."),
    line(n2(S.fa_per_10s_test), "per 10 s", "False alarms", `On the test faces the instrument reported ${n2(S.fa_per_10s_test)} movements per ten seconds that were not scripted: ${pc(1 - S.precision_test)} of what it reported was not a scripted movement. See below for what produced them.`, "A movement-free clip is not the same as a motionless face: blinks, speech and head motion are there on purpose."),
    line(n2(S.au_f1_test), "F1", "Action units named", `For the movements it found, the action units it named (probability above one half) matched the scripted ones with F1 ${n2(S.au_f1_test)}; ${pc(S.au_exact_set_test)} of the time the whole set was exactly right.`, "The phantom generator and the instrument’s templates were written separately, but both come from the same author and the same anatomy: expect real faces to be harder."),
    line(n2(S.au_ece_test, 3), "ECE", "Are the probabilities honest?", `When it says an action unit is present with probability p, how often is it? The calibration error (ECE) is ${n2(S.au_ece_test, 3)} after calibration, against ${n2(S.au_ece_default_test, 3)} with default settings; 0 would be perfect.`, "Measured on the same synthetic faces; probabilities on real faces will be less reliable until recalibrated on real annotations."),
    S.amp_coverage_test == null ? null : line(pc(S.amp_coverage_test), "covered", "Size read-outs and their intervals", `For the action units it named correctly, the size it reported (how far the region that moves most travelled, as a share of the eye distance) was a median ${n2(S.amp_ratio_median)}× the true size, ${biasPhrase(S.amp_ratio_median)}. The 90 % intervals, widened by a factor fitted on the calibration faces, contained the true size ${pc(S.amp_coverage_test)} of the time on the test faces.`, "True size is measured in the instrument’s own units from the generator’s exact skin motion, so the comparison is about the measurement, not about unit conventions."),
    line(`${n2(S.mda_scale, 1)}×`, "scale", "How the sensitivity estimate was corrected", `The smallest-detectable-movement estimate computed from noise alone was optimistic by about ${n2(S.mda_scale, 1)}×, because it assumes ideal matched filters and a clean template; the correction is applied to every capture grade shown on the bench.`, `Fitted on all six faces, because it only rescales a number the report prints and no held-out score on this page depends on it. ${S.mda_scale_cal != null && S.mda_scale_test != null ? `Fitted on faces A–C alone it is ${n2(S.mda_scale_cal, 2)}×, on D–F alone ${n2(S.mda_scale_test, 2)}×: see Sensitivity.` : ""}`),
  ].filter(Boolean));
  s.append(L);
  body.append(s);
}

// ------------------------------------------------------------------------------------------------ spotting
function spotting(body, E) {
  const S = E.spotting;
  const s = sec("spotting", "Spotting", "Which movements are found, and what it costs in false alarms", "Lowering the detection threshold finds more of the small movements and also more of the noise. The standard threshold sits where false alarms are rare.");
  const rows = S.table.map((r) => ({ T: r.T, fa: r.test.fa_per_10s, strict: r.test.recall_strict, loose: r.test.recall_loose, p: r.test.precision }))
    .filter((r) => isFinite(r.fa)).sort((a, b) => a.fa - b.fa);
  const host = h("div", { class: "chart-host" });
  s.append(host);
  const pts = (k) => rows.map((r) => [Math.max(r.fa, 0.04), r[k]]);
  const op = S.table.find((r) => r.T === S.chosen);
  const c1 = lineChart(host, {
    title: "Share of scripted movements found, against false alarms (test faces)", height: 290, margin: { l: 56, r: 20, t: 14, b: 44 },
    x: { domain: [0.04, 6], log: true, title: "false alarms per 10 seconds of footage", ticks: [0.05, 0.1, 0.2, 0.5, 1, 2, 5], format: (v) => String(v) },
    y: { domain: [0, 1], title: "movements found", ticks: [0, 0.25, 0.5, 0.75, 1], format: (v) => pc(v) },
    series: [{ name: "Timing overlaps the true interval by half", points: pts("strict"), color: SERIES[0] }, { name: "Apex time within 0.15 s", points: pts("loose"), color: SERIES[1] }],
    dots: op ? [{ x: Math.max(op.test.fa_per_10s, 0.04), y: op.test.recall_strict, color: SERIES[0], label: `threshold ${S.chosen}`, below: true }] : [],
    readout: (x, i) => { const r = rows[i]; return [{ value: String(r.T), label: "threshold" }, { value: pc(r.strict), key: SERIES[0], label: "found, strict timing" }, { value: pc(r.loose), key: SERIES[1], label: "found, apex only" }, { value: n2(r.fa), label: "false alarms per 10 s" }]; },
    table: { columns: ["Threshold", "False alarms / 10 s", "Found (strict)", "Found (apex)", "Share of reports that were scripted"], rows: rows.map((r) => [r.T, n2(r.fa), pc(r.strict), pc(r.loose), pc(r.p)]) },
  });
  // by camera and by nuisance
  const camRows = Object.entries(S.by_camera).map(([k, v]) => ({ label: CAM[k] || k, value: v.recall_loose, text: `${pc(v.recall_loose)} of ${v.n_truth}`, note: `${v.fa_per_10s.toFixed(2)} false alarms per 10 s` }));
  const nuRows = Object.entries(S.by_nuisance).map(([k, v]) => ({ label: NUIS[k] || k, value: v.recall_loose, text: `${pc(v.recall_loose)} of ${v.n_truth}`, note: `${v.fa_per_10s.toFixed(2)} false alarms per 10 s` }));
  s.append(h("p", { class: "legend-text" }, "The chart above uses the test faces. The two below, and the movement-free count, pool all six faces (every clip of the corpus): each camera or level of restlessness has too few clips to split between calibration and test, and nothing was fitted from them."));
  const two = h("div", { class: "two-col" });
  const hA = h("div", { class: "chart-host" }), hB = h("div", { class: "chart-host" });
  two.append(hA, hB);
  s.append(two);
  const c2 = bars(hA, { title: "Found, by camera", rows: camRows, labelWidth: 190, margin: { r: 110 }, x: { domain: [0, 1], format: (v) => pc(v), ticks: [0, 0.5, 1] }, color: SERIES[0], table: { columns: ["Camera", "Found", "Scripted"], rows: Object.entries(S.by_camera).map(([k, v]) => [CAM[k] || k, pc(v.recall_loose), v.n_truth]) } });
  const c3 = bars(hB, { title: "Found, by how restless the person was", rows: nuRows, labelWidth: 200, margin: { r: 110 }, x: { domain: [0, 1], format: (v) => pc(v), ticks: [0, 0.5, 1] }, color: SERIES[0], table: { columns: ["Movement while filming", "Found", "Scripted"], rows: Object.entries(S.by_nuisance).map(([k, v]) => [NUIS[k] || k, pc(v.recall_loose), v.n_truth]) } });
  // false alarms: what produced them
  const m = S.movement_free;
  const src = m.fa_sources || {};
  const lab = { blink: "a blink", speech: "mouth movement", head: "head movement", noise: "noise alone", near_event: "next to a real movement" };
  s.append(h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, "Movement-free footage"), h("p", { class: "legend-text" }, `${m.seconds.toFixed(0)} seconds of scripted-movement-free phantom clips (blinks, glances and head motion included): ${m.fp} reports, ${n2(m.fa_per_10s)} per ten seconds. What each was triggered by:`)),
    h("ul", { class: "plain-list" }, Object.entries(src).map(([k, v]) => h("li", {}, h("b", { class: "num" }, String(v)), " ", lab[k] || k)))));
  body.append(s);
  return [c1, c2, c3];
}

// ------------------------------------------------------------------------------------------------ sensitivity
const quant = (v, d = 2) => (v == null ? "n/a" : n2(v, d));
function sensitivity(body, E) {
  const S = E.sensitivity;
  const out = [];
  const s = sec("sens", "Sensitivity", "How small a movement can each camera show?", "For each camera class: the chance a scripted movement was found, against its true size. Size is how far the region that moves most travelled, as a percent of the distance between the eyes (a 1 % movement is about a pixel on a 100-pixel eye distance). Each curve climbs to its own ceiling, because even large movements are not found every time: a blink, speech or a head movement can hide them. The smallest movement a class of camera shows is read where its curve reaches half of that ceiling.");
  const host = h("div", { class: "chart-host" });
  s.append(host);
  const cams = Object.keys(S.envelope).filter((k) => k !== "all" && S.envelope[k].curve && S.envelope[k].curve.length);
  if (cams.length) {
    const xmin = Math.min(...cams.map((k) => S.envelope[k].curve[0].x)), xmax = Math.max(...cams.map((k) => S.envelope[k].curve[S.envelope[k].curve.length - 1].x));
    const series = cams.map((k, i) => ({ name: CAM[k] || k, points: S.envelope[k].curve.map((c) => [c.x, c.p]), color: SERIES[i % SERIES.length] }));
    const hi = (ci) => (ci && ci[1] != null && ci[1] < 1000 ? n2(ci[1]) : "unbounded");
    out.push(lineChart(host, {
      title: "Chance a scripted movement is found, by its true size", height: 300, margin: { l: 56, r: 20, t: 14, b: 44 },
      x: { domain: [Math.max(0.05, xmin * 0.95), Math.min(30, xmax * 1.05)], log: true, title: "true size of the movement, % of the eye distance", ticks: [0.1, 0.2, 0.5, 1, 2, 5, 10, 20], format: (v) => String(v) },
      y: { domain: [0, 1], title: "found", ticks: [0, 0.25, 0.5, 0.75, 1], format: (v) => pc(v) },
      series, rules: [{ y: 0.5 }],
      table: { columns: ["Camera", "Scripted", "Found", "Ceiling (large movements found)", "Half the ceiling at (% of eye distance)", "95 % interval", "Nine tenths of it at"], rows: cams.map((k) => { const e = S.envelope[k]; return [CAM[k] || k, e.n, e.detected, e.ceiling != null ? pc(e.ceiling) : "n/a", n2(e.a50), e.a50_ci ? `${n2(e.a50_ci[0])} to ${hi(e.a50_ci)}` : "n/a", e.a90 != null && e.a90 < 1000 ? n2(e.a90) : "not reached"]; }) },
    }));
  } else host.append(note("caution", "Too few scripted movements per camera to draw a detection curve."));
  s.append(h("p", { class: "legend-text" }, "Curves are logistic fits that top out at their own ceiling, to all scripted movements of each camera class (all six faces; descriptive, nothing was fitted from them), with the clip as the unit of resampling. The dim-light and high-speed classes rest on a few dozen movements each, so their curves are indicative only and how they rank against the others is not established; the compressed phone video is the one that is clearly worse at small sizes. The ceiling is a property of this phantom corpus, where blinks, speech and head bursts land on movements at random; the bars above show how steeply the found share falls as the person gets more restless."));

  // the clip-level "smallest detectable movement" against what was actually found
  const M = S.mda_check;
  if (M && M.curve && M.curve.length) {
    const hostM = h("div", { class: "chart-host" });
    const bins = (M.bins || []).filter((b) => b.n >= 3);
    const mid = (b) => Math.sqrt(b.lo * b.hi);
    const r90 = M.single_au_90pct != null && M.single_au_90pct < 100 ? `${n2(M.single_au_90pct, 1)}×` : "a size beyond the range tested";
    s.append(h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, "Is the recording’s own “smallest detectable movement” honest?"),
      h("p", { class: "legend-text" }, `Every recording is given an estimate of the smallest movement it can show, computed from its noise. If that estimate were exact, a movement of exactly that size (1×) would be found half the time. Measured on the single-action-unit movements of all six faces the half-way point is ${n2(M.single_au_50pct, 2)}× the estimate and the 90 % point is ${r90}; the grades on the bench use these corrected values.`)), hostM));
    out.push(lineChart(hostM, {
      title: "Chance a single-action-unit movement is found, by size relative to the clip’s own estimate", height: 270, margin: { l: 56, r: 20, t: 14, b: 44 },
      x: { domain: [M.curve[0].x, M.curve[M.curve.length - 1].x], log: true, title: "true size ÷ the recording’s predicted smallest detectable movement", ticks: [0.2, 0.5, 1, 2, 5], format: (v) => `${v}×` },
      y: { domain: [0, 1], title: "found", ticks: [0, 0.25, 0.5, 0.75, 1], format: (v) => pc(v) },
      series: [{ name: "Fitted detection curve", points: M.curve.map((c) => [c.x, c.p]), color: SERIES[0] }],
      dots: bins.map((b) => ({ x: mid(b), y: b.rate, color: SERIES[1] })),
      rules: [{ x: 1, label: "estimate exact" }, { y: 0.5 }],
      table: { columns: ["Size ÷ estimate", "Movements", "Found", "Share", "95 % interval"], rows: (M.bins || []).map((b) => [`${n2(b.lo)} to ${n2(b.hi)}×`, b.n, b.detected, b.n ? pc(b.rate) : "n/a", b.n ? `${pc(b.ci_lo)} to ${pc(b.ci_hi)}` : "n/a"]) },
    }));
    s.append(mdaSplit(S, M));
  }

  // capture grade vs detection
  const gr = S.grades.filter((g) => g.clips > 0);
  const rows = S.grades.map((g) => ({ label: `Grade ${g.grade}${g.clips ? "" : " (no clips)"}`, est: g.rate, lo: g.ci_lo, hi: g.ci_hi, text: g.clips ? `${g.detected} of ${g.events_ge_1pct}` : "—", dim: !g.clips, color: SERIES[0], rows: [{ value: String(g.clips), label: "clips" }, { value: `${g.detected} of ${g.events_ge_1pct}`, label: "movements ≥ 1 % of the eye distance found" }] }));
  const host2 = h("div", { class: "chart-host" });
  s.append(h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, "Does the capture grade mean anything?"), h("p", { class: "legend-text" }, "The grade shown on the bench is meant to predict how many movements of a given size a recording can show. Among the phantom clips of each grade (all six faces), the share of movements at least 1 % of the eye distance that were found:")), host2));
  out.push(forest(host2, {
    rowHeight: 30, labelWidth: 150, x: { domain: [0, 1], title: "movements found (95 % interval)", format: (v) => pc(v), ticks: [0, 0.25, 0.5, 0.75, 1] }, rows,
    caption: gr.length < 2 ? "Only one grade occurs among the phantom clips, so the grade cannot be validated against detection here; real footage of varied quality is needed." : "",
    table: { columns: ["Grade", "Clips", "Movements ≥ 1 %", "Found", "Share"], rows: S.grades.map((g) => [g.grade, g.clips, g.events_ge_1pct, g.detected, g.rate == null ? "n/a" : pc(g.rate)]) },
  }));

  // how faithfully the chain reproduces the true movement
  const F = S.fidelity;
  if (F && Object.keys(F).length) {
    const names = { single_au: "One action unit", multi_au: "Two or three action units", found: "Movements that were found", missed: "Movements that were missed", all: "Everything" };
    const order = [...Object.keys(F).filter((k) => CAM[k] && k !== "all"), "all", "single_au", "multi_au", "found", "missed"].filter((k) => F[k]);
    const trs = order.map((k) => { const v = F[k]; return h("tr", {}, h("td", {}, names[k] || CAM[k] || k), h("td", { class: "r num" }, String(v.n)), h("td", { class: "r num" }, `${quant(v.pair_gain.q50)}  (${quant(v.pair_gain.q25)} to ${quant(v.pair_gain.q75)})`), h("td", { class: "r num" }, quant(v.fit_r2.q50)), h("td", { class: "r num" }, quant(v.size.q50))); });
    s.append(h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, "How faithfully the chain reproduces the true movement"),
      h("p", { class: "legend-text" }, "For each scripted movement (all six faces) the generator knows exactly how the skin moved. “Measured ÷ true” is the part of that motion the instrument reports on the action-unit templates (1 would be exact; head-motion removal and texture-poor regions pull it below or above). “Share the templates can express” is how much of the true pattern a combination of the twelve hand-written templates can describe at all: what it cannot express stays unexplained, and the posteriors above already carry that as model error.")),
      h("div", { class: "scroll-x" }, h("table", { class: "table-view tv-plain" }, h("thead", {}, h("tr", {}, ["Group", "Movements", "Measured ÷ true, median (middle half)", "Share templates express", "Median size (% of eye distance)"].map((x, i) => h("th", { class: i ? "r" : "" }, x)))), h("tbody", {}, trs)))));
  }
  body.append(s);
  return out;
}


// the correction factor is the one fitted quantity that uses all six faces; this block shows how far it moves between the two groups
function mdaSplit(S, M) {
  const P = S.mda_split;
  const stack = (top, sub) => [top, sub ? h("span", { class: "small muted", style: { display: "block" } }, sub) : null];
  const at = (v, c) => (v == null ? "n/a" : stack(`${n2(v, 2)}×`, c && c.length === 2 ? `${n2(c[0], 2)} to ${n2(c[1], 2)}` : null));
  const sec2 = h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, "Does the correction carry over to other faces?"),
    h("p", { class: "legend-text" }, "The two correction factors are the one thing on this page fitted on all six faces: they only rescale a number the report prints, and no held-out score here depends on them. To see how far they move from one set of faces to another they are refitted on each group alone; “Moves” counts the single-action-unit movements each fit used.")));
  if (!P || !P.calibration || !P.test) return sec2;
  const rows = [
    [stack("All six faces", "used by the bench"), M.n_single, at(M.single_au_50pct, M.a50_ci), at(M.single_au_90pct, M.a90_ci)],
    [stack("Faces A–C"), P.calibration.n_single, at(P.calibration.single_au_50pct, P.calibration.a50_ci), at(P.calibration.single_au_90pct, P.calibration.a90_ci)],
    [stack("Faces D–F"), P.test.n_single, at(P.test.single_au_50pct, P.test.a50_ci), at(P.test.single_au_90pct, P.test.a90_ci)],
  ];
  sec2.append(h("div", { class: "scroll-x" }, h("table", { class: "table-view tv-plain" },
    h("thead", {}, h("tr", {}, ["Fitted on", "Moves", "Half found at", "Nine in ten at"].map((x, i) => h("th", { class: i ? "r" : "" }, x)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c, i) => h("td", { class: i ? "r num" : "" }, c))))))));
  const ratio = P.calibration.single_au_50pct && P.test.single_au_50pct ? P.test.single_au_50pct / P.calibration.single_au_50pct : null;
  const T = P.transfer;
  const bits = ["Intervals, under the values, are 95\u00a0% from resampling whole clips."];
  const ca = P.calibration.a50_ci, ta = P.test.a50_ci;
  const overlap = ca && ta ? (ca[1] >= ta[0] && ta[1] >= ca[0] ? "the intervals overlap, so three faces a side cannot separate them" : "the intervals do not overlap") : null;
  if (ratio) bits.push(`Faces D–F needed movements ${n2(ratio, 1)} times as large as faces A–C to be found half the time${overlap ? `; ${overlap}` : ""}.`);
  if (T && T.n) {
    bits.push(`Applied to the D–F movements, the curve fitted on A–C predicts that ${pc(T.predicted)} would be found; ${pc(T.observed)} were.`);
    if (T.n_at_or_above) bits.push(`Of the ${T.n_at_or_above} that were at least as large as its “nine in ten” size (${n2(T.at_90_point, 2)}×), ${T.found_at_or_above} were found${T.found_ci_lo != null ? ` (95\u00a0% interval ${pc(T.found_ci_lo)} to ${pc(T.found_ci_hi)})` : ""}.`);
  }
  const spread = M.single_au_50pct ? Math.max(...[P.calibration.single_au_50pct, P.test.single_au_50pct].filter((v) => v != null).map((v) => Math.abs(v / M.single_au_50pct - 1))) : null;
  bits.push(`So the correction carries over to new faces roughly, not precisely${spread != null ? `: the two groups sit within about ${Math.round(spread * 100)}\u00a0% of the pooled value either side` : ""}. Read the report’s smallest-movement figure with that spread in mind, and remember it is measured on phantoms.`);
  sec2.append(h("p", { class: "legend-text" }, bits.join(" ")));
  return sec2;
}

// ------------------------------------------------------------------------------------------------ reliability
function reliability(body, E) {
  const G = E.engine;
  const s = sec("rel", "Action-unit probabilities", "When it says 80 %, is it right 80 % of the time?", "Every named action unit comes with a probability. A reliability diagram groups the claims by stated probability and checks how often each group was true. On the diagonal is honest.");
  const host = h("div", { class: "chart-host" });
  s.append(host);
  const rel = G.test.reliability.filter((b) => b.n > 0), relD = (G.default_test.reliability || []).filter((b) => b.n > 0);
  const series = [
    { name: "Calibrated settings", points: rel.map((b) => [b.mean_p, b.frac]), color: SERIES[0] },
    { name: "Default settings", points: relD.map((b) => [b.mean_p, b.frac]), color: SERIES[3] },
    { name: "Perfectly honest", points: [[0, 0], [1, 1]], color: "var(--faint)", width: 1.5 },
  ];
  const c1 = lineChart(host, {
    title: "Stated probability against how often it was true (test faces)", height: 330, margin: { l: 56, r: 20, t: 14, b: 44 },
    x: { domain: [0, 1], title: "probability stated by the instrument", ticks: [0, 0.25, 0.5, 0.75, 1], format: (v) => pc(v) },
    y: { domain: [0, 1], title: "how often it was true", ticks: [0, 0.25, 0.5, 0.75, 1], format: (v) => pc(v) },
    series,
    dots: [...rel.map((b) => ({ x: b.mean_p, y: b.frac, color: SERIES[0] })), ...relD.map((b) => ({ x: b.mean_p, y: b.frac, color: SERIES[3] }))],
    readout: (x, i) => { const b = rel[Math.min(i, rel.length - 1)]; return b ? [{ value: pc(b.mean_p), label: "stated" }, { value: pc(b.frac), key: SERIES[0], label: `true (calibrated), ${b.n} claims` }] : []; },
    table: { columns: ["Stated", "True (calibrated)", "Claims", "Stated (default)", "True (default)", "Claims (default)"], rows: Array.from({ length: Math.max(rel.length, relD.length) }, (_, i) => [rel[i] ? pc(rel[i].mean_p) : "", rel[i] ? pc(rel[i].frac) : "", rel[i] ? rel[i].n : "", relD[i] ? pc(relD[i].mean_p) : "", relD[i] ? pc(relD[i].frac) : "", relD[i] ? relD[i].n : ""]) },
  });
  // per AU
  const per = Object.entries(G.test.per_au).filter(([, v]) => v.n_true > 0 || v.fp > 0).sort((a, b) => +a[0] - +b[0]);
  const rows = per.map(([au, v]) => ({ label: `AU${au}`, est: v.f1, lo: null, hi: null, text: `F1 ${n2(v.f1)}  (${v.tp}/${v.n_true})`, rows: [{ value: pc(v.precision), label: "precision" }, { value: pc(v.recall), label: "recall" }, { value: `${v.tp} of ${v.n_true}`, label: "found among scripted" }, { value: String(v.fp), label: "named when not scripted" }] }));
  const host2 = h("div", { class: "chart-host" });
  s.append(h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, "By action unit"), h("p", { class: "legend-text" }, "F1 for each action unit on the test faces, among the movements the instrument found. Some action units look alike to this instrument; the Atlas shows which.")), host2));
  const c2 = bars(host2, { rows: per.map(([au, v]) => ({ label: `AU${au}`, value: v.f1, text: `F1 ${n2(v.f1)} · ${v.tp} of ${v.n_true}`, rows: [{ value: pc(v.precision), label: "precision" }, { value: pc(v.recall), label: "recall" }, { value: `${v.tp} of ${v.n_true}`, label: "found among scripted" }, { value: String(v.fp), label: "named when not scripted" }] })), labelWidth: 70, margin: { r: 150 }, x: { domain: [0, 1], ticks: [0, 0.5, 1] }, color: SERIES[0], table: { columns: ["Action unit", "Scripted", "Found", "Wrongly named", "Precision", "Recall", "F1"], rows: per.map(([au, v]) => [`AU${au}`, v.n_true, v.tp, v.fp, pc(v.precision), pc(v.recall), n2(v.f1)]) } });
  // by size relative to the clip's own smallest detectable movement
  const snr = G.by_snr || {};
  if (Object.keys(snr).length) {
    const trs = Object.entries(snr).map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", { class: "r num" }, String(v.n)), h("td", { class: "r num" }, n2(v.f1)), h("td", { class: "r num" }, pc(v.exact_set)), h("td", { class: "r num" }, pc(v.precision)), h("td", { class: "r num" }, pc(v.recall)), h("td", { class: "r num" }, n2(v.ece, 3))));
    s.append(h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, "Reliability depends on size"), h("p", { class: "legend-text" }, "The same measures on all six faces, split by how large the movement was compared with the smallest movement the recording could show. Small movements are named less reliably; the probabilities are meant to say so.")),
      h("div", { class: "scroll-x" }, h("table", { class: "table-view tv-plain" }, h("thead", {}, h("tr", {}, ["Movement size", "Movements", "F1", "Exact set", "Precision", "Recall", "ECE"].map((x, i) => h("th", { class: i ? "r" : "" }, x)))), h("tbody", {}, trs)))));
  }
  body.append(s);
  return [c1, c2];
}

// ------------------------------------------------------------------------------------------------ uncertainty
function ratioBins(points, nb = 6) {
  const pts = points.filter((p) => p.true > 0 && p.est > 0).sort((a, b) => a.true - b.true);
  if (pts.length < 24) return [];
  const q = (v, f) => { const a = [...v].sort((x, y) => x - y); const i = f * (a.length - 1), lo = Math.floor(i), hi = Math.ceil(i); return a[lo] + (a[hi] - a[lo]) * (i - lo); };
  const out = [];
  for (let i = 0; i < nb; i++) {
    const sel = pts.slice(Math.floor((i * pts.length) / nb), Math.floor(((i + 1) * pts.length) / nb));
    if (sel.length < 4) continue;
    const r = sel.map((p) => p.est / p.true);
    out.push({ x: q(sel.map((p) => p.true), 0.5), lo_x: sel[0].true, hi_x: sel[sel.length - 1].true, n: sel.length, med: q(r, 0.5), q25: q(r, 0.25), q75: q(r, 0.75), cover: sel.filter((p) => p.lo <= p.true && p.true <= p.hi).length / sel.length });
  }
  return out;
}
function uncertainty(body, E) {
  const C = E.categories || {}, A = E.engine.amplitude_interval;
  const out = [];
  const s = sec("unc", "Uncertainty", "Do the intervals and sets cover what they promise?", "A 90 % interval should contain the truth 90 % of the time; a 90 % prediction set should contain the true class 90 % of the time. Both are checked on faces the calibration never saw.");
  const rows = [];
  if (A && A.n) {
    const nT = A.n_test != null ? A.n_test : A.n;
    if (A.coverage_90_test != null) rows.push({ label: "Size, 90 % interval as computed", est: A.coverage_90_test, text: `${pc(A.coverage_90_test)} of ${nT}`, color: SERIES[3], note: "before the conformal widening" });
    const cw = A.coverage_90_widened_test != null ? A.coverage_90_widened_test : A.coverage_90;
    if (cw != null) rows.push({ label: A.k && A.k !== 1 ? `Size, 90 % interval widened ×${n2(A.k)}` : "Size, 90 % interval", est: cw, text: `${pc(cw)} of ${nT}`, color: SERIES[1], note: "share of correctly named action units whose true size lay inside the reported interval; the widening factor was fitted on the calibration faces only" });
  }
  for (const [k, v] of Object.entries(C)) rows.push({ label: `Class set, ${k}`, est: v.coverage_test, text: `${pc(v.coverage_test)} of ${v.n_test}`, color: SERIES[0], rows: [{ value: pc(v.coverage_test), label: "coverage on test faces" }, { value: n2(v.mean_set_size, 2), label: "average set size" }, { value: pc(v.accuracy_when_committing), label: "accuracy when it commits to one class" }, { value: pc(v.commit_rate), label: "how often it commits" }, { value: pc(v.majority_baseline), label: "always saying the commonest class" }] });
  const host = h("div", { class: "chart-host" });
  s.append(host);
  out.push(forest(host, {
    title: "Coverage against the 90 % promised", rowHeight: 32, labelWidth: 240, x: { domain: [0, 1], ref: 0.9, title: "share covered (the line is the promised 90 %)", format: (v) => pc(v), ticks: [0, 0.25, 0.5, 0.75, 0.9, 1] }, rows,
    table: { columns: ["Quantity", "Coverage", "Test items"], rows: rows.map((r) => [r.label, r.text.split(" of ")[0], r.text.split(" of ")[1]]) },
    caption: "The class sets come from the instrument’s probabilities for the expert action-unit-to-category map, so they measure how uncertainty about the action units carries through to the labels; they do not test whether the labels are right about people. Where coverage falls below 90 % the instrument is overconfident there.",
  }));
  // size read-out against truth
  const rb = A && A.points ? ratioBins(A.points) : [];
  if (rb.length) {
    const hostR = h("div", { class: "chart-host" });
    s.append(h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, "Size read-out against the true size"),
      h("p", { class: "legend-text" }, `For correctly named action units: reported size ÷ true size, in groups of equal numbers of movements. A median of 1 would mean no bias. ${A.ratio_median != null ? `Overall the median is ${n2(A.ratio_median)}.` : ""} Part of any shortfall is the measurement chain (see the fidelity table above) and part is the shrinkage the model applies when the evidence is weak.`)), hostR));
    out.push(lineChart(hostR, {
      title: "Reported size ÷ true size", height: 250, margin: { l: 56, r: 20, t: 14, b: 44 },
      x: { domain: [Math.max(0.05, rb[0].lo_x * 0.9), rb[rb.length - 1].hi_x * 1.1], log: true, title: "true size of the action unit, % of the eye distance", ticks: [0.1, 0.2, 0.5, 1, 2, 5], format: (v) => String(v) },
      y: { domain: [0, 2], title: "reported ÷ true", ticks: [0, 0.5, 1, 1.5, 2], format: (v) => `${v}×` },
      series: [{ name: "Median (band: middle half)", points: rb.map((b) => [b.x, b.med]), band: rb.map((b) => [b.x, b.q25, b.q75]), color: SERIES[1] }],
      dots: rb.map((b) => ({ x: b.x, y: b.med, color: SERIES[1] })), rules: [{ y: 1, label: "no bias" }],
      table: { columns: ["True size (median, %)", "Movements", "Median reported ÷ true", "Middle half", "Intervals covering the truth"], rows: rb.map((b) => [n2(b.x), b.n, n2(b.med), `${n2(b.q25)} to ${n2(b.q75)}`, pc(b.cover)]) },
    }));
  }
  if (Object.keys(C).length) {
    const cats = h("table", { class: "table-view tv-plain" }, h("thead", {}, h("tr", {}, ["Label scheme", "Commits", "Right when it commits", "Always the commonest class", "Average set size"].map((x, i) => h("th", { class: i ? "r" : "" }, x)))),
      h("tbody", {}, Object.entries(C).map(([k, v]) => h("tr", {}, h("td", {}, k), h("td", { class: "r num" }, pc(v.commit_rate)), h("td", { class: "r num" }, pc(v.accuracy_when_committing)), h("td", { class: "r num" }, pc(v.majority_baseline)), h("td", { class: "r num" }, n2(v.mean_set_size))))));
    s.append(h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, "Abstention")), h("p", { class: "legend-text" }, "The instrument declines to name a class when the evidence is weak. Right when it commits is judged only on the items it committed to; compare with simply always giving the commonest class."), cats));
  }
  body.append(s);
  return out;
}

// ------------------------------------------------------------------------------------------------ controls and grid
function controls(body, E) {
  const C = E.controls;
  const s = sec("controls", "Controls", "Does the machinery find structure where there is none?", "Three checks that the results above are not artefacts of the way they were produced.");
  const sh = C.shuffled_truth_f1, fa = C.false_alarm_named_aus;
  s.append(h("ul", { class: "ctl" },
    h("li", {}, h("b", { class: "num" }, n2(sh.mean)), h("div", {}, h("h3", {}, "Shuffled truth"), h("p", { class: "prose" }, `With the true action-unit sets randomly reassigned to other movements, F1 falls to ${n2(sh.mean)} on average (95th percentile ${n2(sh.q95)}); the real value is ${n2(sh.observed)}. The agreement is not an accident of the label frequencies.`))),
    h("li", {}, h("b", { class: "num" }, n2(C.prior_only_f1)), h("div", {}, h("h3", { class: "label-less" }, "The prior alone"), h("p", { class: "prose" }, `Always naming the most common action units (AU${C.prior_only_aus.join(", AU")}) with no look at the video gives F1 ${n2(C.prior_only_f1)}. The instrument has to beat this to say anything.`))),
    h("li", {}, h("b", { class: "num" }, pc(fa.mean_named)), h("div", {}, h("h3", {}, "Action units named for false alarms"), h("p", { class: "prose" }, `For the ${fa.n_false_alarms} reports that were not scripted movements, the instrument named an action unit ${pc(fa.share_with_any)} of the time — so a named action unit on a doubtful event is itself weak evidence.`)))));
  body.append(s);
}
function grid(body, E) {
  const g = E.engine.grid;
  if (!g || !g.length) return;
  const ch = E.engine.chosen;
  const s = sec("grid", "How the settings were chosen", "The calibration search", "Three settings of the action-unit model were chosen by their log-loss on the calibration faces: how wide it assumes the model error to be (noise width), how much it tempers its own confidence, and how many action units it expects a micro-movement to involve (the prior over 0, 1, 2 or 3). The table shows the best prior for each pair of the first two.");
  const etas = [...new Set(g.map((r) => r.eta))].sort((a, b) => a - b), temps = [...new Set(g.map((r) => r.temper))].sort((a, b) => a - b);
  const best = (e, t) => g.filter((r) => r.eta === e && r.temper === t && r.cal).sort((a, b) => a.cal.log_loss - b.cal.log_loss)[0];
  const vals = g.filter((r) => r.cal).map((r) => r.cal.log_loss), lo = Math.min(...vals), hi = Math.max(...vals);
  const tbl = h("table", { class: "gridtab", "aria-label": "Log-loss on the calibration faces by setting" },
    h("thead", {}, h("tr", {}, h("th", {}, "tempering ↓   noise width →"), etas.map((e) => h("th", { class: "num" }, String(e))))),
    h("tbody", {}, temps.map((t) => h("tr", {}, h("th", { class: "num" }, String(t)), etas.map((e) => {
      const r = best(e, t);
      if (!r) return h("td", {}, "");
      const a = 1 - (r.cal.log_loss - lo) / Math.max(hi - lo, 1e-9);
      const chosen = e === ch.eta && t === ch.temper;
      return h("td", { class: `num${chosen ? " chosen" : ""}`, style: { background: `rgba(29,95,168,${0.06 + 0.5 * a})` }, title: `log-loss ${r.cal.log_loss.toFixed(3)}, prior ${r.size_prior.join(" / ")}` }, r.cal.log_loss.toFixed(2), chosen ? h("b", { class: "visually-hidden" }, " (chosen)") : null);
    })))));
  const edge = etas.indexOf(ch.eta) === etas.length - 1 || temps.indexOf(ch.temper) === 0 || etas.indexOf(ch.eta) === 0 || temps.indexOf(ch.temper) === temps.length - 1;
  s.append(h("div", { class: "scroll-x" }, tbl), h("p", { class: "legend-text" }, `Darker is better (lower log-loss on the calibration faces, the faces the choice was made on). The chosen setting is outlined: noise width ${ch.eta}, tempering ${ch.temper}, prior over the number of action units ${ch.size_prior ? ch.size_prior.map((v) => pc(v)).join(" / ") : "n/a"}.`,
    edge ? " The optimum lies on the edge of the grid searched, so a wider search might do better; the result should be read with that in mind." : " The optimum lies inside the region searched."));
  body.append(s);
}

// ------------------------------------------------------------------------------------------------ learning-harness runs (when present)
const REFERENCE_ROWS = new Set(["majority", "prior", "expert"]);       // what a model has to beat: drawn in grey, not as contenders
const MODEL_SHORT = { majority: "Commonest class", prior: "Random guess", expert: "Prototypes", linear: "Logistic", linear_pattern: "Logistic (pattern)", linear_au: "Logistic (AU)", graphnet: "Graph net" };
const MODEL_LABEL = { majority: "Always the commonest class", prior: "Random guess, training frequencies", expert: "ONSET prototypes (untrained)", linear: "Logistic regression", linear_pattern: "Logistic regression, pattern only", linear_au: "Logistic regression, AU evidence only", graphnet: "Graph-temporal transformer" };
function learnRuns(body, ev, parts) {
  const keys = Object.keys(ev).filter((k) => k.startsWith("learn_") && ev[k] && ev[k].models);
  if (!keys.length) return;
  const s = sec("learn", "Learning harness", "Training and evaluation runs", "Subject-independent evaluation of the learned models, written by onset.learn. Nobody appears in both training and test; the headline numbers weigh every class equally; intervals resample people, not clips. A run on synthetic data checks the machinery; it is labelled as such and is not a benchmark.");
  for (const k of keys) {
    const r = ev[k];
    const d = r.data || {};
    const art = h("article", { class: "run" });
    art.append(h("h3", {}, r.title || k, r.synthetic ? h("span", { class: "tag-syn" }, "synthetic data: machinery check") : null),
      h("p", { class: "small muted" }, r.note || ""),
      h("p", { class: "legend-text" }, `Scheme ${r.scheme} (${(r.classes || []).join(", ")}), protocol ${r.protocol}, ${r.n_folds} folds, ${(r.seeds || []).length} seed${(r.seeds || []).length === 1 ? "" : "s"}. ${d.n_clips} clips from ${d.n_subjects} subjects; class counts ${(d.classes || []).map((c, i) => `${c} ${d.class_counts[i]}`).join(", ")}. Leakage audit ${r.audit && r.audit.ok ? "passed" : "FAILED"}${r.audit && r.audit.warnings && r.audit.warnings.length ? ` with ${r.audit.warnings.length} warning${r.audit.warnings.length === 1 ? "" : "s"}` : ""}.`));
    // headline forest: UF1 with interval and the chance line
    const names = Object.keys(r.models);
    const any = r.models[names[0]];
    const rows = names.map((n) => { const m = r.models[n], u = m.headline.uf1; return { label: MODEL_LABEL[n] || n, short: MODEL_SHORT[n], est: u.value, lo: u.lo, hi: u.hi, text: `UF1 ${n2(u.value)}`, color: n === r.best ? SERIES[0] : REFERENCE_ROWS.has(n) ? "var(--faint)" : SERIES[2], dim: n === "majority" || n === "prior", rows: [{ value: n2(u.value), label: "UF1" }, { value: n2(m.headline.uar.value), label: "unweighted recall" }, { value: pc(m.headline.accuracy.value), label: "accuracy" }, { value: n2(m.calibration.ece, 3), label: "calibration error" }] }; });
    const hostF = h("div", { class: "chart-host" });
    art.append(hostF);
    parts.push(forest(hostF, {
      title: "UF1 with a 95 % interval over subjects", rowHeight: 30, labelWidth: 250, x: { domain: [0, 1], ref: any.chance.uf1, title: "unweighted F1 (the line is the expected score of uniform guessing)", titleShort: "unweighted F1 · line = chance", format: (v) => n2(v, 1), ticks: [0, 0.2, 0.4, 0.6, 0.8, 1] }, rows,
      table: { columns: r.columns, rows: r.rows },
      caption: "A model that does not clear the line, or the untrained prototype row, has not learned anything beyond what is already known.",
    }));
    if (r.rows) art.append(h("div", { class: "scroll-x" }, h("table", { class: "table-view tv-plain" }, h("thead", {}, h("tr", {}, r.columns.map((c, i) => h("th", { class: i > 1 ? "r" : "" }, c)))), h("tbody", {}, r.rows.map((row) => h("tr", {}, row.map((c, i) => h("td", { class: i > 1 ? "r num" : "" }, String(c)))))))));
    if (r.pairwise && r.pairwise.length) art.append(h("div", { class: "scroll-x" }, h("table", { class: "table-view tv-plain" }, h("caption", { class: "small muted" }, "Paired permutation tests on UF1 (people are swapped between the two models), Holm-adjusted"),
      h("thead", {}, h("tr", {}, ["A", "B", "A − B", "p", "p, adjusted"].map((x, i) => h("th", { class: i > 1 ? "r" : "" }, x)))),
      h("tbody", {}, r.pairwise.map((t) => h("tr", {}, h("td", {}, MODEL_LABEL[t.a] || t.a), h("td", {}, MODEL_LABEL[t.b] || t.b), h("td", { class: "r num" }, `${t.difference >= 0 ? "+" : "−"}${Math.abs(t.difference).toFixed(3)}`), h("td", { class: "r num" }, t.p.toFixed(4)), h("td", { class: "r num" }, t.p_holm.toFixed(4))))))));
    const ctl = Object.entries(r.controls || {});
    if (ctl.length) art.append(h("ul", { class: "ctl" }, ctl.map(([ck, cv]) => {
      const big = ck === "shuffled_labels" ? n2(cv.uf1_mean) : ck === "identity_probe" ? (cv.accuracy != null ? pc(cv.accuracy) : "n/a") : ck === "leakage_gap" ? `${cv.inflation >= 0 ? "+" : "−"}${Math.abs(cv.inflation).toFixed(2)}` : "";
      const nm = { shuffled_labels: "Shuffled labels", identity_probe: "Can identity be read from the features?", leakage_gap: "What a leaky split would have added" }[ck] || ck;
      return h("li", {}, h("b", { class: "num" }, big), h("div", {}, h("h3", {}, nm), h("p", { class: "prose" }, cv.reading || cv.reason || "")));
    })));
    const lat = r.latency && r.latency.feature_extraction_s_per_clip;
    if (lat) art.append(h("p", { class: "small muted" }, `Feature extraction ${lat.mean.toFixed(2)} s per clip on average (90th percentile ${lat.p90.toFixed(2)} s); model inference per clip is in the table.`));
    s.append(art);
  }
  body.append(s);
}

// ------------------------------------------------------------------------------------------------ what has not been measured
function realData(body, ev) {
  const s = sec("real", "What has not been measured", "Real faces", "No analysis in this installation has been run on real micro-expression footage. Public datasets require a signed agreement from their owners and cannot be redistributed, so none is bundled; running the same evaluation on data you are licensed to use is a single command.");
  const D = [
    ["CASME II", "Yan, Li, Wang, Zhao, Liu, Chen & Fu (2014), PLOS ONE 9(1): e86041", "http://casme.psych.ac.cn/casme/e2"],
    ["SAMM", "Davison, Lansley, Costen, Tan & Yap (2016), IEEE Trans. Affective Computing 9(1)", "https://personalpages.manchester.ac.uk/staff/adrian.davison/SAMM.html"],
    ["MMEW", "Ben, Ren, Zhang et al. (2021), IEEE TPAMI 44(9), a survey that introduces the dataset", "http://www.dpailab.com/database.html"],
    ["4DME", "Li, Cheng, Li, Behzad, Shen, Zafeiriou, Pantic & Zhao (2022), IEEE Trans. Affective Computing", "https://www.oulu.fi/en/university/faculties-and-units/faculty-information-technology-and-electrical-engineering/center-machine-vision-and-signal-analysis"],
    ["CAS(ME)³", "Li, Dong, Lu, Wang, Yan, Ma, Liu, Huang & Fu (2022), IEEE TPAMI", "http://casme.psych.ac.cn/casme/e4"],
  ];
  s.append(h("div", { class: "scroll-x" }, h("table", { class: "table-view tv-plain" }, h("thead", {}, h("tr", {}, ["Dataset", "Reference", "Where to request it"].map((x) => h("th", {}, x)))),
    h("tbody", {}, D.map(([name, cite, url]) => h("tr", {}, h("td", {}, h("b", {}, name)), h("td", {}, cite), h("td", {}, h("a", { href: url, rel: "noopener noreferrer", target: "_blank" }, "request page"))))))),
    h("p", { class: "small muted" }, "Each dataset contains identifiable faces and needs a signed licence agreement from its owners. Details — number of participants and clips, frame rate, label scheme — are in the papers; check them before you cite any figure. docs/DATASETS.md lists the file layouts the learning harness reads."),
    h("div", { class: "cmd" }, h("p", { class: "label ink" }, "Evaluating on licensed data"), h("pre", { class: "code" }, "python -m onset.learn.run --data casme2=/path/to/CASME2 --scheme casme2_5 \\\n    --protocol loso --out runs/casme2\npython -m onset.learn.report runs/casme2\n\n# no data yet? check that the machinery works on phantom faces\npython -m onset.learn.run --synthetic --out runs/synthetic_check\npython -m onset.learn.report runs/synthetic_check --synthetic")),
    h("p", { class: "prose" }, "Results are reported with leave-one-subject-out evaluation (no person appears in both training and test), class-balanced metrics next to accuracy, intervals from resampling subjects, and the calibration and abstention measures shown above."));
  body.append(s);
}
