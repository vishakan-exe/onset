// The bench: one analysed recording, laid out so that every claim sits next to the evidence for it.
//
//   plate (what the face did)  |  time lens + log (when, how much)
//   chronophotograph strip     (the selected movement in time)
//   explanation (what moved, what it could mean, where the evidence sits)  |  recording quality, limits, scripted truth
import { h, clear, fmt, icon, toast, Bus, STATUS_WORD } from "./util.js";
import { api, ref as mkRef, STATIC } from "./api.js";
import { PlatePlayer } from "./plate.js";
import { TimeLens, eventLabel } from "./timelens.js";
import { Chrono } from "./chrono.js";
import { Findings, Explainer, qualityPanel } from "./panels.js";
import { stageHead, note, busy, errorBox, GRADE_WORDS } from "./ui.js";

const isExpr = (e) => e.status === "micro" || e.status === "expression" || e.status === "unexplained" || e.status === "sustained";

const PROVENANCE = {
  "synthetic phantom": ["Synthetic specimen · not a person", ""],
  "synthetic phantom (stress test)": ["Re-filmed synthetic specimen · not a person", ""],
  "webcam recording": ["Your webcam recording", "real"],
  "uploaded video": ["Your video file", "real"],
};

export function provenanceBadge(result) {
  const k = (result.provenance && result.provenance.kind) || "";
  const [text, cls] = PROVENANCE[k] || [k || "Recording", "real"];
  return h("span", { class: `provenance ${cls}`, title: (result.provenance && result.provenance.note) || "" }, text);
}

function metaLine(result, session) {
  const c = result.clip;
  const bits = [
    ["Length", `${c.duration_s.toFixed(1)} s`],
    ["Frame rate", `${c.fps.toFixed(0)} fps`],
    ["Frame", `${c.width} × ${c.height}`],
  ];
  if (session && session.expires) {
    const hrs = Math.max(0, (session.expires * 1000 - Date.now()) / 3.6e6);
    bits.push(["Stored for", hrs >= 1 ? `${Math.round(hrs)} h more` : "under an hour"]);
  }
  return h("p", { class: "meta-line num" }, bits.map(([k, v]) => h("span", {}, k + " ", h("b", {}, v))));
}

export class Bench {
  /**
   * opts: {result, ref, ctx, reveal (bool|undefined), onDelete, head: bool}
   * With `result.truth` (a phantom) the scripted truth can be revealed on the chart; it is hidden until asked for.
   */
  constructor(host, opts) {
    this.host = host;
    this.r = opts.result;
    this.ref = opts.ref;
    this.ctx = opts.ctx;
    this.opts = opts;
    this.bus = new Bus();
    this.parts = [];
    this.build();
  }

  build() {
    const r = this.r, host = this.host;
    const gradeWord = GRADE_WORDS[r.quality.grade] || "";
    if (this.opts.head !== false) {
      const actions = [];
      if (this.ref.kind === "session" && !STATIC) {
        actions.push(h("a", { class: "btn", href: `api/results/${this.ref.id}/export`, download: "" }, icon("download"), "Export"));
        actions.push(h("button", { class: "btn danger", type: "button", onclick: () => this.remove() }, icon("trash"), "Delete"));
      }
      actions.push(h("button", { class: "btn", type: "button", onclick: () => window.print() }, icon("print"), "Print"));
      host.append(h("header", { class: "bench-head" },
        h("div", {},
          h("p", { class: "kicker" }, "Bench · analysed recording"),
          h("h1", { class: "display", tabindex: "-1" }, r.label || "Recording"),
          h("div", { class: "meta-line" }, provenanceBadge(r), h("span", {}, "Recording quality ", h("b", {}, `Grade ${r.quality.grade}`), gradeWord ? ` · ${gradeWord.toLowerCase()}` : "")),
          metaLine(r, r.session)),
        h("div", { class: "actions" }, actions)));
    }
    if (r.engine && !r.engine.calibrated) {
      host.append(note("caution", h("b", {}, "Default parameters. "), "The detection threshold and the action-unit model have not been calibrated on the phantom corpus in this installation; probabilities are less trustworthy than the Evidence page describes."));
    }

    const grid = h("div", { class: "bench" });
    host.append(grid);
    const plateHost = h("section", { class: "b-plate", "aria-label": "Face plate" });
    const lensHost = h("section", { class: "b-lens", "aria-label": "Timeline" });
    const chronoHost = h("section", { class: "b-chrono", "aria-label": "The selected movement laid out in time" });
    const explainHost = h("section", { class: "b-explain", "aria-label": "Explanation" });
    const sideHost = h("aside", { class: "b-side", "aria-label": "Recording quality and limits" });
    grid.append(plateHost, lensHost, chronoHost, explainHost, sideHost);

    plateHost.append(h("h2", { class: "label ink sec-label" }, "The face, stabilised"));
    this.plate = new PlatePlayer(plateHost, { result: r, ref: this.ref, bus: this.bus });
    this.parts.push(this.plate);

    lensHost.append(h("h2", { class: "label ink sec-label" }, "Time lens"),
      h("p", { class: "legend-text" }, "Each lane is one part of the face; the paper’s vertical lines are 100 ms apart. The red line is the detection threshold."));
    this.lens = new TimeLens(lensHost, { result: r, bus: this.bus });
    this.lens.mount(lensHost);
    this.parts.push(this.lens);
    lensHost.append(h("h2", { class: "label ink sec-label", style: { marginTop: "26px" } }, "Movements found"));
    this.findings = new Findings(lensHost, { result: r, bus: this.bus });

    this.chrono = new Chrono(chronoHost, { result: r, sprites: this.plate.sprites, bus: this.bus });
    this.explainer = new Explainer(explainHost, { result: r, bus: this.bus, plate: this.plate });
    this.parts.push({ destroy: () => this.explainer.dispose() });

    sideHost.append(
      h("section", { class: "sec first" }, h("header", {}, h("h3", { class: "label ink" }, "Recording quality"), h("p", { class: "legend-text" }, "How small a movement this particular recording could show. It is measured from the noise in this clip.")), qualityPanel(r)));
    if (r.truth) sideHost.append(this.truthPanel());
    sideHost.append(this.limitsPanel());
    this.sideHost = sideHost;

    this.bus.on("ready", () => this.start());
    if (this.plate.ready) this.start();
    if (this.opts.reveal) this.setReveal(true);
  }

  start() {
    if (this.started) return;
    this.started = true;
    const evs = this.r.events;
    const want = this.opts.select && evs.find((e) => e.id === this.opts.select);
    const first = want || evs.find(isExpr) || evs[0];
    if (first) this.select(first.id);
  }
  select(id) { this.bus.emit("select", { id, from: this }); const e = this.r.events.find((x) => x.id === id); if (e) { this.bus.emit("seek", { t: e.t_apex, from: this }); this.plate.t = e.t_apex; this.plate.draw(); this.lens.cur = e.t_apex; this.lens.zoomEvent && this.lens.zoomEvent(); } }

  // ------------------------------------------------------------------------------------------------ scripted truth
  truthPanel() {
    const r = this.r, truth = r.truth, score = r.score;
    this.truthBody = h("div", { class: "truth-body", hidden: true });
    this.truthBtn = h("button", { class: "btn", type: "button", "aria-pressed": "false", onclick: () => this.setReveal(!this.revealed) }, icon("check"), "Reveal what was scripted");
    const wrap = h("section", { class: "sec truth" },
      h("header", {}, h("h3", { class: "label ink" }, "Scripted truth"),
        h("p", { class: "legend-text" }, "This clip is a synthetic phantom, so the true answer exists. It stays hidden until you ask, so that you can judge the instrument first.")),
      this.truthBtn, this.truthBody);
    const rows = [];
    for (const e of truth.events) {
      const inst = e.units === "instrument";
      const per = inst && e.aus.length > 1 ? ` · read per action unit: ${e.aus.map((a) => `AU${a} ${e.au_pct_iod[String(a)].toFixed(2)} %`).join(", ")}` : "";
      rows.push(h("li", {}, h("b", { class: "num" }, `${e.t_apex.toFixed(2)} s`), " ", e.aus.map((a) => `AU${a}`).join(" + "),
        h("span", { class: "muted" }, ` · ${(e.t_off - e.t_on) * 1000 | 0} ms · ${e.amp_pct_iod.toFixed(2)} % of the eye distance${inst ? " at its largest region" : ""}${per}` + (e.kind === "macro" ? " · longer than a micro-expression" : ""))));
    }
    if (!truth.events.length) rows.push(h("li", { class: "muted" }, "No expression was scripted."));
    this.truthBody.append(h("ul", { class: "truth-list" }, rows));
    if (truth.blinks && truth.blinks.length) this.truthBody.append(h("p", { class: "small muted" }, `${truth.blinks.length} scripted blink${truth.blinks.length === 1 ? "" : "s"}.`));
    if (truth.speech && truth.speech.length) this.truthBody.append(h("p", { class: "small muted" }, "Scripted mouth movement (speech) at " + truth.speech.map((s) => `${s.t_on.toFixed(1)}–${s.t_off.toFixed(1)} s`).join(", ") + "."));
    if (score && score.amplitudes && score.amplitudes.length) {
      const inside = score.amplitudes.filter((a) => a.covered).length;
      this.truthBody.append(h("div", { class: "ampcheck" },
        h("h4", { class: "label" }, "Size against the expected reading"),
        h("p", { class: "small muted" }, "The expected reading is what a perfect measurement would give for the scripted movement, in the unit the report uses (how far the action unit's main region moves, as % of the distance between the eyes). The generator's own scripting parameter is not comparable and is not shown."),
        h("ul", { class: "truth-list" }, score.amplitudes.map((a) => h("li", {}, h("b", { class: "num" }, `AU${a.au}`), " ",
          h("span", { class: "num" }, `read ${a.est.toFixed(2)} % (90 % interval ${a.lo.toFixed(2)} to ${a.hi.toFixed(2)})`), h("span", { class: "muted" }, ` · expected ${a.expected.toFixed(2)} % · `), h("span", { class: a.covered ? "ok" : "miss" }, a.covered ? "inside the interval" : "outside the interval")))),
        score.amplitudes.length > 1 ? h("p", { class: "small muted" }, `${inside} of ${score.amplitudes.length} intervals contain the expected reading. Over the whole phantom calibration about nine in ten do; a handful of clips cannot confirm that, only show that it is not wildly off.`) : null));
    }
    if (score) {
      const found = score.tp_loose, n = score.n_truth;
      this.truthBody.append(h("div", { class: "verdictline" }, icon("info"), h("div", {},
        n ? h("span", {}, h("b", {}, `${found} of ${n}`), ` scripted movement${n === 1 ? "" : "s"} found (timing within a tenth of a second or so); ${score.tp_strict} also overlap the scripted interval by at least half. `)
          : h("span", {}, "Nothing was scripted. "),
        h("b", {}, `${score.fp}`), ` movement${score.fp === 1 ? "" : "s"} reported that were not scripted.`,
        score.rows && score.rows.length ? h("span", {}, ` Action units named for the found movements: ${score.rows.map((x) => `scripted ${x.truth.map((a) => "AU" + a).join("+") || "none"}, named ${x.pred.map((a) => "AU" + a).join("+") || "none"}`).join("; ")}.`) : null)));
    }
    return wrap;
  }
  setReveal(on) {
    if (!this.r.truth) return;
    this.revealed = on;
    this.truthBody.hidden = !on;
    this.truthBtn.setAttribute("aria-pressed", String(on));
    this.truthBtn.lastChild.textContent = on ? "Hide what was scripted" : "Reveal what was scripted";
    this.lens.marks = on ? this.r.truth.events.map((e) => ({ t0: e.t_on, t1: e.t_off, label: `scripted ${e.aus.map((a) => "AU" + a).join("+")}`, color: "rgba(29,95,168,0.16)" })) : [];
    this.lens.draw();
    this.bus.emit("reveal", { on });
  }

  // ------------------------------------------------------------------------------------------------ limits
  limitsPanel() {
    return h("section", { class: "sec" },
      h("header", {}, h("h3", { class: "label ink" }, "What this analysis cannot tell you")),
      h("ul", { class: "limits" }, (this.r.limits || []).map((l) => h("li", { class: "small" }, l))));
  }

  async remove() {
    if (!confirm("Delete this analysis and its images now?")) return;
    try { await api.remove(this.ref.id); toast("Deleted."); this.ctx.navigate("#/bench"); } catch (e) { toast(e.message, "error"); }
  }
  destroy() { for (const p of this.parts) { try { p.destroy && p.destroy(); } catch { /* ignore */ } } }
}

// ------------------------------------------------------------------------------------------------ the list of recent analyses
async function mountList(host, ctx) {
  host.append(stageHead({ kicker: "Station 02 · Bench", title: "Analysed recordings", lede: "Each analysis is kept on this computer for a day, then deleted. The recording itself is removed as soon as it has been analysed; only grayscale face plates and numbers stay." }));
  const body = h("div", { class: "bench-list" }, busy("Reading the stored analyses"));
  host.append(body);
  let rows = [];
  try { rows = await api.results(); } catch (e) { clear(body); body.append(errorBox(e)); return; }
  clear(body);
  if (!rows.length) {
    body.append(h("div", { class: "empty" }, h("p", { class: "prose" }, "Nothing has been analysed in this session yet."),
      h("div", { class: "actions" },
        h("a", { class: "btn primary", href: "#/capture" }, icon("capture"), "Record a clip"),
        h("a", { class: "btn", href: "#/capture/upload" }, icon("upload"), "Analyse a video file"),
        h("a", { class: "btn", href: "#/cabinet" }, icon("cabinet"), "Open the specimen cabinet"))));
    return;
  }
  const list = h("ol", { class: "log" });
  for (const r of rows) {
    const hrs = Math.max(0, (r.expires * 1000 - Date.now()) / 3.6e6);
    list.append(h("li", {}, h("a", { class: "log-row link-row", href: `#/bench/session/${r.id}` },
      h("span", { class: "log-t num" }, r.grade || "–"),
      h("span", { class: "log-main" }, h("span", { class: "log-title" }, r.label || "Recording"),
        h("span", { class: "log-sub small muted num" }, `${fmt.date(r.created)} · ${r.duration_s ? r.duration_s.toFixed(1) + " s" : ""} · ${r.n_events == null ? "" : r.n_events + " movement" + (r.n_events === 1 ? "" : "s")}`)),
      h("span", { class: "small muted" }, hrs >= 1 ? `deleted in ${Math.round(hrs)} h` : "deleted within the hour"))));
  }
  body.append(list);
}

// ------------------------------------------------------------------------------------------------ route
export async function mount(host, { seg, q }, ctx) {
  if (!seg.length) { await mountList(host, ctx); return; }
  const [kind, id] = seg;
  if ((kind !== "session" && kind !== "library") || !id) { ctx.navigate("#/bench"); return; }
  const ref = mkRef(kind, id);
  const hold = h("div", {}, busy("Opening the analysis"));
  host.append(hold);
  let result;
  try { result = await api.result(ref); } catch (e) {
    clear(host);
    host.append(stageHead({ kicker: "Bench", title: "This analysis is not available" }), errorBox(e), h("p", {}, h("a", { class: "btn", href: "#/bench" }, "Back to the bench")));
    return;
  }
  clear(host);
  ctx.setTitle(result.label || "Bench");
  const bench = new Bench(host, { result, ref, ctx, select: q.e, reveal: q.reveal === "1" });
  return { dispose: () => bench.destroy() };
}
