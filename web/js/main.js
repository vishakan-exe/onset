// The shell: spine navigation, a hash router, server status, and one clear place for failures.
//
//   #/                              the development (first look) and recent analyses
//   #/capture   #/capture/upload    record a clip with the webcam / analyse a video file
//   #/bench/<session|library>/<id>  one analysed recording
//   #/cabinet   #/cabinet/<id>      specimen library; #/cabinet/stress, #/cabinet/compare/<a>/<b>
//   #/cue   #/evidence   #/atlas
import { $, h, clear, icon, toast } from "./util.js";
import { api, STATIC } from "./api.js";
import { busy, errorBox } from "./ui.js";

const STATIONS = [
  { id: "capture", label: "Capture", icon: "capture", hash: "#/capture" },
  { id: "bench", label: "Bench", icon: "bench", hash: "#/bench" },
  { id: "cabinet", label: "Cabinet", icon: "cabinet", hash: "#/cabinet" },
  { id: "cue", label: "Cue Lab", icon: "cue", hash: "#/cue" },
  { id: "evidence", label: "Evidence", icon: "evidence", hash: "#/evidence" },
  { id: "atlas", label: "Atlas", icon: "atlas", hash: "#/atlas" },
];
const MODULES = {
  home: () => import("./firstrun.js"),
  capture: () => import("./capture.js"),
  bench: () => import("./bench.js"),
  cabinet: () => import("./cabinet.js"),
  cue: () => import("./cuelab.js"),
  evidence: () => import("./evidence.js"),
  atlas: () => import("./atlas.js"),
};
const TITLES = { home: "Light table", capture: "Capture", bench: "Bench", cabinet: "Specimen cabinet", cue: "Cue Lab", evidence: "Evidence", atlas: "Atlas" };

const stage = $("#stage");
const ctx = {
  status: null,
  navigate: (hash) => { if (location.hash === hash) route(); else location.hash = hash; },
  setTitle: (t) => { document.title = t ? `${t} · ONSET` : "ONSET · a light table for facial micro-movement"; },
  refreshStatus: () => loadStatus(),
};
let current = null;          // {name, dispose}
let token = 0;

// ------------------------------------------------------------------------------------------------ spine
function buildSpine() {
  const tabs = $("#tabs");
  for (const s of STATIONS) {
    tabs.append(h("button", { class: "tab", type: "button", dataset: { station: s.id }, onclick: () => ctx.navigate(s.hash), "aria-label": s.label }, icon(s.icon), h("span", {}, s.label)));
  }
}
function markTab(name) {
  for (const b of document.querySelectorAll(".tab")) {
    if (b.dataset.station === name) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
  }
}
function paintFoot() {
  const foot = $("#spine-foot");
  clear(foot);
  const s = ctx.status;
  if (!s) {
    foot.append(h("div", {}, h("span", { class: "dot off" }), h("b", {}, "Server not reachable")), h("div", {}, "Start it with ", h("code", {}, "python -m onset"), " and reload."));
    return;
  }
  foot.append(
    h("div", {}, h("span", { class: "dot" }), h("b", {}, STATIC ? "Showcase" : "Local"), STATIC ? " · static files" : ` · v${s.version}`),
    h("div", {}, STATIC ? "Nothing is computed here." : "Everything stays on this computer. Recordings are deleted after the analysis."),
    s.calibrated ? h("div", {}, "Calibrated on phantoms") : h("div", {}, "Not calibrated yet"));
  if (!STATIC) {
    foot.append(h("button", { class: "linkish small foot-btn", type: "button", onclick: purge }, "Delete all my analyses now"));
  }
}
async function purge() {
  if (!confirm("Delete every analysis stored by this instrument? Specimens in the cabinet are not affected.")) return;
  try {
    const r = await api.purge();
    toast(`Deleted ${r.deleted} stored ${r.deleted === 1 ? "analysis" : "analyses"}.`);
    if (location.hash.startsWith("#/bench/session")) ctx.navigate("#/");
  } catch (e) { toast(e.message, "error"); }
}
async function loadStatus() {
  try { ctx.status = await api.status(); } catch { ctx.status = null; }
  paintFoot();
  return ctx.status;
}

// ------------------------------------------------------------------------------------------------ routing
function parse() {
  const raw = (location.hash || "#/").replace(/^#\/?/, "");
  const [path, query] = raw.split("?");
  const seg = path.split("/").filter(Boolean).map(decodeURIComponent);
  const q = Object.fromEntries(new URLSearchParams(query || ""));
  const name = !seg.length ? "home" : MODULES[seg[0]] ? seg[0] : "home";
  return { name, seg: seg.slice(1), q };
}

let statusReady = null;
async function route() {
  const my = ++token;
  const { name, seg, q } = parse();
  if (current && current.dispose) { try { current.dispose(); } catch (e) { console.error(e); } }
  current = null;
  markTab(name);
  ctx.setTitle(name === "home" ? "" : TITLES[name]);
  clear(stage);
  stage.append(busy("Opening " + TITLES[name].toLowerCase()));
  let mod;
  try {
    await statusReady;
    mod = await MODULES[name]();
  } catch (e) {
    console.error(e);
    if (my !== token) return;
    clear(stage);
    stage.append(errorBox(new Error(`This page could not be loaded (${e.message}).`), route));
    return;
  }
  if (my !== token) return;
  const host = h("div", { class: `station station-${name}` });
  clear(stage);
  stage.append(host);
  try {
    const out = await mod.mount(host, { seg, q }, ctx);
    if (my !== token) { out && out.dispose && out.dispose(); return; }
    current = { name, dispose: out && out.dispose };
  } catch (e) {
    console.error(e);
    if (my !== token) return;
    clear(host);
    host.append(errorBox(e, route));
  }
  window.scrollTo(0, 0);
  const hd = host.querySelector("h1");
  if (hd) hd.focus({ preventScroll: true }); else stage.focus({ preventScroll: true });
}

window.addEventListener("hashchange", route);
window.addEventListener("unhandledrejection", (ev) => { console.error(ev.reason); });
buildSpine();
paintFoot();
statusReady = loadStatus().then(() => { if (!STATIC) setInterval(loadStatus, 30000); });
route();
