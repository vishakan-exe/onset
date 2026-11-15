// The first look.  A frown that lasts four tenths of a second and moves the brows by less than a pixel is played
// back as a person would see it; then the instrument "develops" it, the way a latent image comes up in a tray:
// the chart paper emerges, the movement is found, magnified, and laid out frame by frame.
// The page ends where the instrument's honesty begins: what it can measure, and what it cannot say.
import { h, clear, fmt, icon, Bus, STATUS_WORD } from "./util.js";
import { api, ref as mkRef } from "./api.js";
import { PlatePlayer } from "./plate.js";
import { TimeLens } from "./timelens.js";
import { Chrono } from "./chrono.js";
import { busy, note, store, reduceMotion } from "./ui.js";

const SPECIMEN = "invisible-frown";
const isExpr = (e) => e.status === "micro" || e.status === "expression" || e.status === "unexplained" || e.status === "sustained";

function step(n, title, text, state) {
  return h("li", { class: `step ${state}`, "aria-current": state === "now" ? "step" : null },
    h("span", { class: "step-n num" }, String(n)), h("div", {}, h("b", {}, title), text ? h("p", { class: "small muted" }, text) : null));
}

export async function mount(host, params, ctx) {
  const seen = store.get("seen", false);
  const parts = [];
  let alive = true;

  const hero = h("section", { class: "hero" });
  const left = h("div", { class: "hero-text" });
  const right = h("div", { class: "hero-stage" });
  hero.append(left, right);
  host.append(hero);

  left.append(
    h("p", { class: "kicker" }, "ONSET · a light table for facial micro-movement"),
    h("h1", { class: "display hero-title", tabindex: "-1" }, "Four tenths of a second."),
    h("p", { class: "prose lede" }, "A frown crossed this face and moved the brows by less than a pixel. You are going to look at it twice: once the way a person would, and once with the instrument."));
  const steps = h("ol", { class: "steps", "aria-label": "The three parts of this first look" });
  const narr = h("div", { class: "narr", "aria-live": "polite" });
  left.append(steps, narr);

  right.append(busy("Preparing the specimen"));

  let man = null, result = null;
  try {
    man = await api.library();
    if (!man.specimens || !man.specimens.some((s) => s.id === SPECIMEN)) throw new Error("The specimen cabinet has not been built yet.");
    result = await api.result(mkRef("library", SPECIMEN));
  } catch (e) {
    clear(right);
    right.append(note("caution", h("b", {}, "The specimen is not available. "), e.message || "", " The rest of the instrument still works."));
    drawActions(narr, ctx, null);
    await recents(host, ctx);
    return;
  }
  if (!alive) return;
  clear(right);
  const bus = new Bus();
  const stageBox = h("div", { class: "hero-plate" });
  const lensBox = h("div", { class: "hero-lens", hidden: true });
  const chronoBox = h("div", { class: "hero-chrono", hidden: true });
  right.append(stageBox, lensBox);
  const plate = new PlatePlayer(stageBox, { result, ref: mkRef("library", SPECIMEN), bus });
  plate.setInstrument(false);          // the first look is unaided: no overlays, no magnifier until the instrument is brought in
  parts.push(plate);
  host.append(h("section", { class: "hero-after" }, chronoBox));

  const ev = result.events.find(isExpr) || null;
  let state = 1;

  const paintSteps = () => {
    clear(steps);
    const st = (n) => (state > n ? "done" : state === n ? "now" : "todo");
    steps.append(
      step(1, "Look", "Watch the brows, the way a person would.", st(1)),
      step(2, "Say what you saw", "Be honest. Nothing is scored.", st(2)),
      step(3, "Develop", "Let the instrument find it.", st(3)));
  };

  const play = () => { plate.loop = false; plate.play(); };
  bus.on("play", ({ playing }) => {
    if (!playing && state <= 2 && plate.t >= plate.range()[1] - 0.05) setTimeout(() => { if (alive && state <= 2) play(); }, 700);
  });
  bus.on("ready", () => { if (state === 1) play(); });

  const stage1 = () => {
    state = 1; paintSteps(); clear(narr);
    narr.append(
      h("p", { class: "prose" }, "Eight seconds of a stabilised face, so the head does not move. Somewhere in it the brows draw down."),
      h("p", { class: "small muted" }, "This is a synthetic face: drawn by a program, with a movement scripted to the frame. It is a calibration target, not a person."),
      h("div", { class: "actions" }, h("button", { class: "btn primary", type: "button", onclick: stage2 }, "I have watched it")));
  };
  const stage2 = () => {
    state = 2; paintSteps(); clear(narr);
    const reply = h("p", { class: "prose reply", hidden: true });
    const develop = h("button", { class: "btn pen", type: "button", hidden: true, onclick: stage3 }, icon("record"), "Develop it");
    const ans = (txt, said) => h("button", { class: "btn", type: "button", onclick: () => {
      reply.hidden = false;
      reply.textContent = said === "saw"
        ? "Possibly you did, and possibly you guessed. The next step settles it: the instrument will say when, where and how large."
        : "That is the usual result. The movement is smaller than the width of one pixel on this video and is over in less than half a second.";
      develop.hidden = false; develop.focus();
    } }, txt);
    narr.append(h("p", { class: "prose" }, "Did anything move?"), h("div", { class: "actions" }, ans("I saw a movement", "saw"), ans("I saw nothing", "none")), reply, develop);
  };
  const stage3 = async () => {
    state = 3; paintSteps(); clear(narr);
    plate.pause();
    plate.setInstrument(true);
    lensBox.hidden = false;
    const lens = new TimeLens(lensBox, { result, bus, compact: true });
    lens.mount(lensBox);
    parts.push(lens);
    if (!reduceMotion()) lens.host.animate([{ clipPath: "inset(0 100% 0 0)" }, { clipPath: "inset(0 0 0 0)" }], { duration: 2600, easing: "cubic-bezier(.45,.05,.2,1)" });
    chronoBox.hidden = false;
    const chrono = new Chrono(chronoBox, { result, sprites: plate.sprites, bus });
    if (ev) {
      await new Promise((r) => setTimeout(r, reduceMotion() ? 0 : 1500));
      if (!alive) return;
      bus.emit("select", { id: ev.id, from: lens });
      bus.emit("seek", { t: ev.t_apex, from: lens });
      plate.t = ev.t_apex;
      plate.setMode(ev.magnified ? "magnified" : "plate");
      if (!plate.show.measured) plate.toggleOverlay("measured");
      plate.loop = true;
      plate.play();
      lens.zoomEvent && lens.zoomEvent();
    }
    narrate(ev);
    const final = h("div", { class: "actions" },
      h("a", { class: "btn pen", href: "#/capture" }, icon("capture"), "Record a clip of your own"),
      h("a", { class: "btn", href: `#/bench/library/${SPECIMEN}` }, icon("bench"), "Open this on the bench"),
      h("a", { class: "btn", href: "#/cabinet" }, icon("cabinet"), "Browse the specimen cabinet"));
    narr.append(final);
    store.set("seen", true);
  };

  /** How the movement compares with the smallest the camera could show: adaptive, never flattering. */
  function sizeSentence(au, q) {
    const mda = q && q.mda_median_pct;
    if (!au || !mda) return "The instrument could not estimate the smallest movement this camera can show, so the size above has no yardstick.";
    const r = au.amp_pct_iod / mda;
    const lead = `The smallest movement this camera could show half the time is about ${mda.toFixed(1)} %. `;
    if (r >= 2) return lead + `This movement is about ${r.toFixed(1)} times larger, so the instrument can separate it from noise; it is small enough that you may well have missed it by eye.`;
    if (r >= 1) return lead + `This movement is only ${r.toFixed(1)} times that: close to the limit of what the instrument can tell from noise, so treat it with caution.`;
    return lead + "This movement is below that: the instrument reports it, but it cannot rule out that it is noise.";
  }

  function narrate(e) {
    const q = result.quality;
    if (e) {
      const au = e.aus && e.aus[0];
      const names = e.named_aus && e.named_aus.length ? e.named_aus.map((a) => `AU${a}`).join(" + ") : null;
      narr.append(
        h("p", { class: "prose" },
          `At ${fmt.s(e.t_apex)} the instrument found a movement`, names ? ` that fits ${names}` : "", ", lasting about ",
          h("b", {}, e.duration_ms != null ? `${Math.round(e.duration_ms)} ms` : "less than half a second"),
          au ? h("span", {}, ". Its size: ", h("b", { class: "num" }, `${au.amp_pct_iod.toFixed(1)} % of the distance between the eyes`), ` (between ${au.lo_pct_iod.toFixed(1)} and ${au.hi_pct_iod.toFixed(1)} % with 90 % confidence), about `, h("b", { class: "num" }, `${au.amp_px.toFixed(1)} px`), " on this video") : "", "."),
        h("p", { class: "prose" }, sizeSentence(au, q)));
    } else {
      narr.append(h("p", { class: "prose" }, "At this size and with this camera the instrument stays quiet. That is also a result: a movement too small to tell from noise is not reported."));
    }
    narr.append(
      note("pen", h("b", {}, "What it cannot do. "), "It measures how skin moved. It cannot tell what the person felt, whether the movement was intended, or whether anything was concealed. Small movements like this also occur in faces that are expressing nothing, so finding one is not evidence that anything was felt."));
  }

  paintSteps();
  stage1();
  if (seen) await recents(host, ctx);
  return { dispose: () => { alive = false; parts.forEach((p) => p.destroy && p.destroy()); } };
}

function drawActions(narr, ctx, result) {
  narr.append(h("div", { class: "actions" },
    h("a", { class: "btn pen", href: "#/capture" }, icon("capture"), "Record a clip"),
    h("a", { class: "btn", href: "#/capture/upload" }, icon("upload"), "Analyse a video file"),
    h("a", { class: "btn", href: "#/cabinet" }, icon("cabinet"), "Specimen cabinet")));
}

async function recents(host, ctx) {
  let rows = [];
  try { rows = await api.results(); } catch { return; }
  if (!rows.length) return;
  const sec = h("section", { class: "recent" }, h("h2", { class: "label ink" }, "Pick up where you left off"));
  const list = h("ol", { class: "log" });
  for (const r of rows.slice(0, 4)) {
    list.append(h("li", {}, h("a", { class: "log-row link-row", href: `#/bench/session/${r.id}` },
      h("span", { class: "log-t num" }, r.grade || "–"),
      h("span", { class: "log-main" }, h("span", { class: "log-title" }, r.label || "Recording"), h("span", { class: "log-sub small muted num" }, fmt.date(r.created))),
      h("span", { class: "small muted" }, r.n_events == null ? "" : `${r.n_events} movement${r.n_events === 1 ? "" : "s"}`))));
  }
  sec.append(list);
  host.append(sec);
}
