// The specimen cabinet: synthetic faces with scripted movements and exact ground truth.  They exist so that the
// instrument can be tried without footage, a webcam or anybody's consent, and so that every claim it makes can be
// checked against a known answer.  Four things can be done here:
//   look       watch a specimen with your own eyes, mark what you see, then have the truth revealed
//   bench      open the instrument's reading of it
//   compare    set two look-alike specimens side by side and see where the evidence differs
//   stress     re-film a specimen with a worse camera, more noise, blur or compression and watch it degrade
import { h, svg, clear, fmt, icon, clamp, toast, roiLabel, AU_NAMES, tip } from "./util.js";
import { api, ref as mkRef, fileUrl, pollJob, STATIC } from "./api.js";
import { Bench } from "./bench.js";
import { stageHead, note, busy, errorBox, store, reduceMotion } from "./ui.js";

let MAN = null;
async function manifest() {
  if (MAN) return MAN;
  MAN = await api.library();
  return MAN;
}
const isExpr = (e) => e.status === "micro" || e.status === "expression" || e.status === "unexplained" || e.status === "sustained";
const CAMERA_WORDS = {
  webcam30: "Webcam, 30 frames a second", phone30: "Phone, 30 fps, compressed", webcam24_lowlight: "Webcam in dim light, 24 fps",
  webcam15: "Slow webcam, 15 fps", highspeed100: "High-speed camera, 100 fps",
};
const NUISANCE_WORDS = { still: "Held still", normal: "Ordinary blinking and a little head movement", busy: "Restless head, more blinking" };

export async function mount(host, { seg, q }, ctx) {
  const [a, b, c] = seg;
  let man;
  try { man = await manifest(); } catch (e) { host.append(stageHead({ kicker: "Station 03 · Cabinet", title: "Specimen cabinet" }), errorBox(e)); return; }
  if (!a) return overview(host, man, ctx);
  if (a === "stress") return stressLab(host, man, ctx, q);
  if (a === "compare") return compare(host, man, ctx, b, c);
  return viewer(host, man, ctx, a, q);
}

// ================================================================================================ overview
function slide(s, showGrade = false) {
  const thumb = STATIC ? `data/library/${s.id}/thumb.jpg` : `api/library/${s.id}/files/thumb.jpg`;
  return h("a", { class: "slide", href: `#/cabinet/${s.id}`, "aria-label": s.title },
    h("span", { class: "slide-film" }, h("img", { src: thumb, alt: "", loading: "lazy", width: 220, height: 220, onerror: (e) => { e.target.style.visibility = "hidden"; } })),
    h("span", { class: "slide-cap" }, h("b", {}, s.title), h("span", { class: "small muted num" }, `${s.duration.toFixed(0)} s · ${(CAMERA_WORDS[s.camera] || s.camera).split(",")[0]}`)));
}

function overview(host, man, ctx) {
  host.append(stageHead({
    kicker: "Station 03 · Cabinet",
    title: "Specimen cabinet",
    lede: "Faces drawn by a program, with movements scripted to the frame. No person appears in any of them, so there is nobody to consent and nothing to protect — and every claim the instrument makes about them can be checked against the truth.",
    actions: [h("a", { class: "btn", href: "#/cabinet/stress" }, icon("record"), "Stress lab"), STATIC ? null : h("a", { class: "btn", href: "#/capture/upload" }, icon("upload"), "Test an unseen video")],
  }));
  if (!man.specimens.length) {
    host.append(note("caution", h("b", {}, "The cabinet is empty. "), "Build it with ", h("code", {}, "python scripts/build_library.py"), "."));
    return;
  }
  host.append(h("p", { class: "how-to" }, h("b", {}, "How to use it. "), "Open a slide, watch it with your own eyes, mark what you see, then reveal what was scripted. The bench opens underneath, with the instrument’s reading of the same clip. These are authored illustrations, not a test set: how many of them the instrument finds is not its accuracy."));
  const byGroup = new Map(man.groups.map((g) => [g.id, { g, items: [] }]));
  for (const s of man.specimens) (byGroup.get(s.group) || byGroup.set(s.group, { g: { id: s.group, title: s.group, blurb: "" }, items: [] }).get(s.group)).items.push(s);
  const cab = h("div", { class: "cabinet" });
  for (const { g, items } of byGroup.values()) {
    if (!items.length) continue;
    cab.append(h("section", { class: "drawer", "aria-labelledby": `d-${g.id}` },
      h("header", { class: "drawer-label" }, h("h2", { id: `d-${g.id}`, class: "display" }, g.title), h("p", { class: "small muted" }, g.blurb), h("span", { class: "num small" }, `${items.length} slide${items.length === 1 ? "" : "s"}`)),
      h("div", { class: "rail" }, items.map((s) => slide(s)))));
  }
  host.append(cab);
  host.append(h("details", { class: "fold permissions" }, h("summary", {}, "About these specimens: permissions, privacy, and what they cannot show"),
    h("div", { class: "prose" },
      h("p", {}, "Everything in this cabinet is synthetic. The faces are generated procedurally, the movements are scripted, and the cameras are simulated. There is no footage of any person, so no consent, licence or privacy question arises."),
      h("p", {}, "That is also the limit. Results on specimens say how the instrument behaves against a known answer — whether it finds a scripted movement, how it fares with noise, blinks or head motion. They are not evidence of accuracy on real people. Public micro-expression datasets (CASME II, SAMM, MMEW and others) can only be used under a signed licence from their owners and may not be redistributed, so none ships here; the Evidence page explains how to run the same analysis on data you are licensed to use."),
      h("p", {}, "The specimens are authored, not sampled. Each was written to make one point, and its scripted movements are kept clear of blinks and speech so that a miss means something. How many of them the instrument finds is therefore not a detection rate. The Evidence page measures that on a larger corpus of 240 clips in which blinks, speech and head motion fall wherever they fall."))));
}

// ================================================================================================ the viewer: look, mark, reveal
async function viewer(host, man, ctx, id, q) {
  const spec = man.specimens.find((s) => s.id === id);
  if (!spec) { host.append(stageHead({ kicker: "Cabinet", title: "No such specimen" }), h("p", {}, h("a", { class: "btn", href: "#/cabinet" }, "Back to the cabinet"))); return; }
  ctx.setTitle(spec.title);
  const group = man.groups.find((g) => g.id === spec.group);
  const lede = h("p", { class: "prose lede" }, "Watch the whole clip first. Press ", h("kbd", {}, "M"), " (or the button) at any moment you think the face moved. Then reveal what was scripted.");
  host.append(stageHead({ kicker: `Cabinet · ${group ? group.title : ""}`, title: spec.title, lede, actions: [h("a", { class: "btn quiet", href: "#/cabinet" }, "All slides")] }));

  const marks = [];
  let revealed = false, bench = null;
  const ref = mkRef("library", id);
  const video = h("video", { class: "spec-video", muted: "", playsinline: "", loop: "", preload: "auto", "aria-label": "The specimen as a camera saw it. Press M to mark a moment." },
    spec.videos.map((f) => h("source", { src: fileUrl(ref, f), type: f.endsWith("webm") ? "video/webm" : "video/mp4" })));
  video.muted = true;
  const stampT = h("span", { class: "stamp tr num" }, "0.00 s");
  const plate = h("div", { class: "plate spec-plate" }, video, h("span", { class: "stamp tl" }, "As a camera saw it"), stampT);
  const track = h("div", { class: "mark-track", role: "img", "aria-label": "Your marks along the clip" }, h("i", { class: "head" }));
  const btnPlay = h("button", { class: "btn icon primary", type: "button", "aria-label": "Play", onclick: () => (video.paused ? video.play() : video.pause()) }, icon("play"));
  const step = (d) => h("button", { class: "btn icon", type: "button", "aria-label": d < 0 ? "Previous frame" : "Next frame", onclick: () => { video.pause(); video.currentTime = clamp(video.currentTime + d / (spec.fps || 30), 0, video.duration || spec.duration); } }, icon(d < 0 ? "prev" : "next"));
  const speed = h("select", { "aria-label": "Playback speed", onchange: () => { video.playbackRate = parseFloat(speed.value); }, style: { width: "auto" } }, [[1, "1×"], [0.5, "½×"], [0.25, "¼×"]].map(([v, l]) => h("option", { value: v }, l)));
  const btnMark = h("button", { class: "btn pen", type: "button", onclick: () => mark() }, "Mark this moment");
  const btnNone = h("button", { class: "btn", type: "button", onclick: () => { marks.length = 0; paintMarks(); toast("Noted: you saw nothing."); revealBtn.focus(); } }, "I see nothing");
  const marksLine = h("p", { class: "small muted num marks-line", "aria-live": "polite" }, "No marks yet.");
  const revealBtn = h("button", { class: "btn primary", type: "button", onclick: () => reveal() }, icon("check"), "Reveal what was scripted");
  const left = h("div", { class: "view-left" },
    h("div", { class: "plate-wrap" }, h("div", { class: "crop" }, plate)),
    track,
    h("div", { class: "transport" }, step(-1), btnPlay, step(1), speed, btnMark, btnNone),
    marksLine);
  const result = h("div", { class: "view-right" });
  result.append(h("section", { class: "sec first" },
    h("header", {}, h("h2", { class: "label ink" }, "Your reading")),
    h("p", { class: "prose" }, spec.group === "null" ? "Does anything on this face look like an expression to you?" : "Is there a movement that is not a blink and not the head turning?"),
    h("p", { class: "small muted" }, "Reaction time is about a third of a second, so marks within half a second of a scripted movement count as hits. Nothing here is a score you have to beat: micro-movements are meant to be hard to see."),
    h("div", { class: "actions" }, revealBtn)));
  host.append(h("div", { class: "viewer" }, left, result));
  const benchHost = h("div", { class: "viewer-bench" });
  host.append(benchHost);

  const updateHead = () => {
    const d = video.duration || spec.duration;
    track.querySelector(".head").style.left = `${(video.currentTime / d) * 100}%`;
    stampT.textContent = fmt.s(video.currentTime);
  };
  video.addEventListener("timeupdate", updateHead);
  video.addEventListener("play", () => { btnPlay.replaceChildren(icon("pause")); btnPlay.setAttribute("aria-label", "Pause"); });
  video.addEventListener("pause", () => { btnPlay.replaceChildren(icon("play")); btnPlay.setAttribute("aria-label", "Play"); });
  const raf = () => { if (!video.paused) updateHead(); id_raf = requestAnimationFrame(raf); };
  let id_raf = requestAnimationFrame(raf);

  function paintMarks() {
    track.querySelectorAll(".m").forEach((n) => n.remove());
    const d = video.duration || spec.duration;
    for (const m of marks) track.append(h("i", { class: "m", style: { left: `${(m / d) * 100}%` }, title: fmt.s(m) }));
    if (revealed) for (const e of spec.truth.events) track.append(h("i", { class: "truth", style: { left: `${(e.t_on / d) * 100}%`, width: `${((e.t_off - e.t_on) / d) * 100}%` }, title: `scripted ${e.aus.map((a) => "AU" + a).join("+")}` }));
    marksLine.textContent = marks.length ? `Your marks: ${marks.map((m) => fmt.s(m)).join(", ")}` : "No marks yet.";
  }
  function mark() { if (revealed) return; marks.push(video.currentTime); marks.sort((a, b) => a - b); paintMarks(); }
  const onKey = (e) => { if ((e.key === "m" || e.key === "M") && !/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) { e.preventDefault(); mark(); } };
  document.addEventListener("keydown", onKey);

  async function reveal() {
    if (revealed) return;
    revealed = true;
    revealBtn.disabled = true; btnMark.disabled = true; btnNone.disabled = true;
    paintMarks();
    const tol = 0.5, evs = spec.truth.events;
    const hits = marks.filter((m) => evs.some((e) => m >= e.t_on - tol && m <= e.t_off + tol)).length;
    const hit = evs.filter((e) => marks.some((m) => m >= e.t_on - tol && m <= e.t_off + tol)).length;
    const summary = evs.length
      ? `You marked ${marks.length} moment${marks.length === 1 ? "" : "s"}; ${hit} of the ${evs.length} scripted movement${evs.length === 1 ? "" : "s"} ${evs.length === 1 ? "was" : "were"} within half a second of a mark${marks.length - hits > 0 ? `, and ${marks.length - hits} mark${marks.length - hits === 1 ? "" : "s"} fell elsewhere` : ""}.`
      : `You marked ${marks.length} moment${marks.length === 1 ? "" : "s"}. Nothing was scripted in this clip${marks.length ? "; what you saw was most likely a blink, a glance or the head settling" : ""}.`;
    result.replaceChildren(h("section", { class: "sec first" },
      h("header", {}, h("h2", { class: "label ink" }, "What was scripted")),
      h("p", { class: "prose" }, spec.blurb),
      h("ul", { class: "truth-list" }, spec.truth_lines.map((l) => h("li", {}, l))),
      h("p", { class: "prose" }, summary),
      spec.look ? h("p", { class: "legend-text" }, spec.look) : null,
      spec.confusable && spec.confusable.length ? h("div", { class: "lookalikes" }, h("span", { class: "label" }, "Look-alikes"), spec.confusable.map((o) => { const t = man.specimens.find((s) => s.id === o); return t ? h("a", { class: "btn small", href: `#/cabinet/compare/${spec.id}/${o}` }, t.title) : null; })) : null));
    benchHost.append(busy("Opening the instrument’s reading"));
    try {
      const r = await api.result(ref);
      clear(benchHost);
      benchHost.append(h("h2", { class: "display bench-title" }, "What the instrument saw"));
      bench = new Bench(benchHost, { result: r, ref, ctx, head: false, reveal: true });
      benchHost.scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "start" });
    } catch (e) { clear(benchHost); benchHost.append(errorBox(e)); }
  }
  if (q.reveal === "1") reveal();
  return { dispose: () => { cancelAnimationFrame(id_raf); document.removeEventListener("keydown", onKey); if (bench) bench.destroy(); video.pause(); } };
}

// ================================================================================================ compare
function mainEvent(r) { return r.events.filter(isExpr).sort((a, b) => b.spot.score - a.spot.score)[0] || null; }

function applyPlate(canvas, r, sprites) { /* filled by drawApex */ }

async function loadSprite(ref, r) {
  const imgs = await Promise.all(r.sprites.sheets.map((s) => new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = fileUrl(ref, s.file); })));
  return imgs;
}

function apexPlate(r, ev, imgs, gainFor) {
  const S = r.canon.plate_size;
  const cv = h("canvas", { width: 352, height: 352, "aria-hidden": "true" });
  const ov = svg("svg", { viewBox: `0 0 ${S} ${S}`, "aria-hidden": "true" });
  const wrap = h("div", { class: "plate cmp-plate" }, cv, ov);
  const sp = r.sprites;
  // nearest frame to the apex
  let k = 0;
  for (let i = 1; i < sp.times.length; i++) if (Math.abs(sp.times[i] - ev.t_apex) < Math.abs(sp.times[k] - ev.t_apex)) k = i;
  const sheet = imgs[Math.floor(k / sp.per_sheet)], pos = k % sp.per_sheet;
  cv.getContext("2d").drawImage(sheet, (pos % sp.cols) * sp.size, Math.floor(pos / sp.cols) * sp.size, sp.size, sp.size, 0, 0, 352, 352);
  const box = r.canon.plate_box, kk = S / (box[2] - box[0]);
  const cen = Object.fromEntries(r.canon.centroids.map((c) => [c.roi, c]));
  const gain = gainFor;
  for (const l of ev.ledger.filter((x) => x.used && cen[x.roi])) {
    const c = cen[l.roi];
    const dx = l.dx_px * kk * gain, dy = l.dy_px * kk * gain, len = Math.hypot(dx, dy);
    ov.append(svg("circle", { cx: c.x, cy: c.y, r: 0.9, fill: "#f3f6fa" }));
    if (len < 0.35) continue;
    const ux = dx / len, uy = dy / len, hx = c.x + dx, hy = c.y + dy, s = Math.min(2.6, 0.35 * len + 0.8);
    ov.append(svg("g", { stroke: "#f3f6fa", fill: "#f3f6fa", "stroke-width": 0.8, "stroke-linecap": "round" },
      svg("line", { x1: c.x, y1: c.y, x2: hx - ux * s * 0.6, y2: hy - uy * s * 0.6 }),
      svg("path", { d: `M${hx},${hy}L${hx - ux * s - uy * s * 0.55},${hy - uy * s + ux * s * 0.55}L${hx - ux * s + uy * s * 0.55},${hy - uy * s - ux * s * 0.55}Z`, stroke: "none" })));
  }
  return wrap;
}

function auRows(evA, evB) {
  const m = new Map();
  for (const a of evA.aus) m.set(a.au, { au: a.au, A: a.pip, B: 0 });
  for (const a of evB.aus) { const r = m.get(a.au) || { au: a.au, A: 0, B: 0 }; r.B = a.pip; m.set(a.au, r); }
  return [...m.values()].filter((r) => Math.max(r.A, r.B) >= 0.08).sort((x, y) => Math.max(y.A, y.B) - Math.max(x.A, x.B)).slice(0, 8);
}

function diffs(rA, evA, rB, evB) {
  const pa = Object.fromEntries(evA.pattern.map((p) => [p.roi, p])), pb = Object.fromEntries(evB.pattern.map((p) => [p.roi, p]));
  const la = Object.fromEntries(evA.ledger.map((l) => [l.roi, l])), lb = Object.fromEntries(evB.ledger.map((l) => [l.roi, l]));
  const iod = rA.canon.iod;
  const out = [];
  for (const roi of Object.keys(pa)) {
    const a = pa[roi], b = pb[roi];
    if (!b || !a.used || !b.used) continue;
    const dx = a.dx - b.dx, dy = a.dy - b.dy, se = Math.sqrt(a.sx ** 2 + a.sy ** 2 + b.sx ** 2 + b.sy ** 2) / Math.SQRT2;
    out.push({ roi, d: Math.hypot(dx, dy), pct: (100 * Math.hypot(dx, dy)) / iod, z: Math.hypot(dx, dy) / Math.max(se, 1e-6), a: la[roi], b: lb[roi], dx, dy });
  }
  return out.sort((x, y) => y.z - x.z);
}

async function compare(host, man, ctx, idA, idB) {
  const A = man.specimens.find((s) => s.id === idA), B = man.specimens.find((s) => s.id === idB);
  host.append(stageHead({ kicker: "Cabinet · compare", title: A && B ? `${A.title} and ${B.title}` : "Compare two specimens", lede: "Two movements that are easy to mistake for each other, measured the same way. Where the evidence differs is what the instrument has to go on.", actions: [h("a", { class: "btn quiet", href: "#/cabinet" }, "All slides")] }));
  if (!A || !B) { host.append(note("caution", "Choose two specimens: open one in the cabinet and follow its look-alike links.")); return; }
  const body = h("div", { class: "compare" }, busy("Reading both specimens"));
  host.append(body);
  let rA, rB, imA, imB;
  try {
    [rA, rB] = await Promise.all([api.result(mkRef("library", A.id)), api.result(mkRef("library", B.id))]);
    [imA, imB] = await Promise.all([loadSprite(mkRef("library", A.id), rA), loadSprite(mkRef("library", B.id), rB)]);
  } catch (e) { clear(body); body.append(errorBox(e)); return; }
  const eA = mainEvent(rA), eB = mainEvent(rB);
  clear(body);
  if (!eA || !eB) { body.append(note("caution", "The instrument reported no movement in ", !eA ? A.title : B.title, ", so there is nothing to put side by side. That is itself the difference between the two.")); return; }

  // common arrow gain so that the two plates are comparable
  const kk = rA.canon.plate_size / (rA.canon.plate_box[2] - rA.canon.plate_box[0]);
  const maxLen = Math.max(...[...eA.ledger, ...eB.ledger].filter((l) => l.used).map((l) => Math.hypot(l.dx_px, l.dy_px) * kk), 1e-3);
  const gain = clamp(15 / maxLen, 3, 90);
  const title = (s, e) => h("div", { class: "cmp-title" }, h("h2", { class: "display" }, s.title), h("p", { class: "small muted num" }, `${fmt.s(e.t_apex)} · ${eventTitleShort(e)}`));
  const cols = h("div", { class: "cmp-cols" },
    h("div", {}, title(A, eA), h("div", { class: "plate-wrap" }, h("div", { class: "crop" }, apexPlate(rA, eA, imA, gain)))),
    h("div", {}, title(B, eB), h("div", { class: "plate-wrap" }, h("div", { class: "crop" }, apexPlate(rB, eB, imB, gain)))));
  body.append(cols, h("p", { class: "legend-text" }, "Arrows show where the skin moved at the apex of each movement, drawn at the same scale on both plates."));

  // AU evidence
  const rows = auRows(eA, eB);
  const au = h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, "Action units: how likely each is"), h("p", { class: "legend-text" }, "The instrument’s probability that each action unit took part in the movement.")),
    h("div", { class: "pair-bars" }, rows.map((r) => h("div", { class: "pair-row" },
      h("span", { class: "pair-l" }, h("b", { class: "num" }, `AU${r.au}`), " ", AU_NAMES[r.au] || ""),
      h("span", { class: "pair-t" }, h("i", { class: "a", style: { width: `${r.A * 100}%` } }), h("i", { class: "b", style: { width: `${r.B * 100}%` } })),
      h("span", { class: "num small" }, h("span", { class: "pen" }, `${Math.round(r.A * 100)} %`), " / ", h("span", {}, `${Math.round(r.B * 100)} %`))))),
    h("p", { class: "small muted" }, h("span", { class: "key a" }), A.title, "  ", h("span", { class: "key b" }), B.title));
  body.append(au);

  // categories
  const scheme = "megc3";
  const cA = eA.categories[scheme], cB = eB.categories[scheme];
  const lab = cA.labels;
  const catBars = (c) => h("div", { class: "bars" }, [...lab.map((l, i) => [l, c.probs[i]]), ["none of these", c.unclear]].map(([l, p]) => h("div", { class: `bar-row${l === c.top && !c.abstain ? " top" : ""}${l === "none of these" ? " unclear" : ""}` }, h("span", { class: "bar-l" }, l), h("span", { class: "bar-t" }, h("i", { style: { "--v": `${p * 100}%` } })), h("span", { class: "bar-v num" }, `${Math.round(p * 100)} %`))));
  body.append(h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, "What each could mean (hypotheses, not verdicts)"), h("p", { class: "legend-text" }, "Composite classes (positive, negative, surprise) derived from the action units with an expert prior. “None of these” is the instrument declining to choose.")),
    h("div", { class: "cmp-cols" }, h("div", {}, h("h4", { class: "label ink" }, A.title), catBars(cA), cA.abstain ? h("p", { class: "small muted" }, "The instrument abstains here.") : null),
      h("div", {}, h("h4", { class: "label ink" }, B.title), catBars(cB), cB.abstain ? h("p", { class: "small muted" }, "The instrument abstains here.") : null))));

  // where they differ
  const df = diffs(rA, eA, rB, eB).slice(0, 8);
  const mx = Math.max(...df.map((d) => d.z), 3);
  body.append(h("section", { class: "sec" }, h("header", {}, h("h3", { class: "label ink" }, "Where the two differ most"), h("p", { class: "legend-text" }, "For each region of the face: how far apart the two measured movements are, in units of their combined measurement uncertainty. Bars shorter than about 2 are not distinguishable.")),
    h("p", { class: "div-ref-note" }, "The dashed line marks 2, the smallest difference that can be told from measurement noise."),
    h("div", { class: "div" }, df.map((d) => h("div", { class: "div-row" },
      h("span", { class: "div-l" }, roiLabel(d.roi)),
      h("span", { class: "div-track one" }, h("i", { class: "a", style: { left: "0", width: `${(d.z / mx) * 100}%` } }), h("i", { class: "ref", style: { left: `${(2 / mx) * 100}%` }, title: "2: the limit of what can be told apart" })),
      h("span", { class: "div-v num small" }, d.z.toFixed(1)))))));
}
function eventTitleShort(e) { return e.named_aus && e.named_aus.length ? e.named_aus.map((a) => "AU" + a).join(" + ") : "no clear pattern"; }

// ================================================================================================ stress lab
async function stressLab(host, man, ctx, q) {
  host.append(stageHead({ kicker: "Cabinet · stress lab", title: "Stress lab", lede: "Take a specimen with a known answer and film it again with a worse camera, more noise, blur, compression or a smaller movement. The instrument has to cope — or say that it cannot." }));
  if (STATIC) { host.append(note("caution", "The stress lab renders and analyses new clips and needs the local ONSET server.")); return; }
  let opt;
  try { opt = await api.stressOptions(); } catch (e) { host.append(errorBox(e)); return; }
  const spec = h("select", { id: "st-spec" }, opt.specimens.filter((s) => s.group !== "ladder").map((s) => h("option", { value: s.id }, s.title)));
  spec.value = q.specimen && opt.specimens.some((s) => s.id === q.specimen) ? q.specimen : "smile-flicker";
  const cam = h("select", { id: "st-cam" }, [h("option", { value: "" }, "As in the specimen"), ...opt.cameras.map((c) => h("option", { value: c.id }, `${CAMERA_WORDS[c.id] || c.id}`))]);
  const nz = h("select", { id: "st-nz" }, [h("option", { value: "" }, "As in the specimen"), ...opt.nuisance.map((n) => h("option", { value: n }, NUISANCE_WORDS[n] || n))]);
  const sl = (id, min, max, step, val, fmtv) => {
    const out = h("output", { class: "num" }, fmtv(val));
    const inp = h("input", { type: "range", id, min, max, step, value: val, oninput: () => { out.textContent = fmtv(parseFloat(inp.value)); } });
    return { inp, out };
  };
  const amp = sl("st-amp", 0.2, 3, 0.05, 1, (v) => `${v.toFixed(2)} ×`);
  const noise = sl("st-noise", 0, 5, 0.1, 1, (v) => `${v.toFixed(1)} ×`);
  const blur = sl("st-blur", 0, 3, 0.1, 0, (v) => `${v.toFixed(1)} px`);
  const crf = h("select", { id: "st-crf" }, [["", "None"], ["23", "Light (CRF 23)"], ["30", "Moderate (CRF 30)"], ["38", "Heavy (CRF 38)"]].map(([v, l]) => h("option", { value: v }, l)));
  const go = h("button", { class: "btn pen", type: "button", onclick: run }, icon("record"), "Re-film and analyse");
  const status = h("div", {});
  const field = (label, ctl, out, help) => h("div", { class: "field" }, h("label", { for: ctl.id }, label, out ? h("span", { class: "muted" }, " ", out) : null), ctl, help ? h("p", { class: "small muted" }, help) : null);
  host.append(h("div", { class: "stress" },
    h("section", { class: "cap-form" },
      h("div", { class: "form-grid" },
        field("Specimen", spec), field("Camera", cam), field("Movement while filming", nz),
        field("Size of the movement", amp.inp, amp.out, "Scales the scripted amplitude. 0.3× is below what most cameras can show."),
        field("Sensor noise", noise.inp, noise.out, "Multiplies the camera’s noise."),
        field("Extra blur", blur.inp, blur.out, "Out-of-focus or motion blur, in pixels."),
        field("Video compression", crf)),
      h("div", { class: "actions" }, go), status),
    h("section", { class: "sec first" }, h("header", {}, h("h2", { class: "label ink" }, "Or open the ladder"), h("p", { class: "legend-text" }, "One scripted smile, already filmed five ways.")),
      h("div", { class: "lookalikes" }, man.specimens.filter((s) => s.group === "ladder").map((s) => h("a", { class: "btn small", href: `#/cabinet/${s.id}` }, s.title))))));
  async function run() {
    go.disabled = true; clear(status);
    try {
      const body = { specimen: spec.value, camera: cam.value || null, nuisance: nz.value || null, amp_scale: parseFloat(amp.inp.value), noise_scale: parseFloat(noise.inp.value), blur: parseFloat(blur.inp.value), codec_crf: crf.value ? parseInt(crf.value, 10) : null };
      const job = await api.stress(body);
      status.append(busy("Filming and analysing"));
      const p = h("p", { class: "small muted num", role: "status" });
      status.append(p);
      const done = await pollJob(job.id, (j) => { p.textContent = `${j.stage} · ${Math.round((j.progress || 0) * 100)} %`; });
      ctx.navigate(`#/bench/session/${done.result_id}`);
    } catch (e) { clear(status); status.append(errorBox(e)); go.disabled = false; }
  }
}
