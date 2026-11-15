// Shared composite pieces used by every station: the station head, honest-limits callouts, loading and error states.
import { h, icon, clear } from "./util.js";

/** The head of a station: a registration-mark kicker, a wide display title and a serif lede. */
export function stageHead({ kicker, title, lede, actions, id }) {
  return h("header", { class: "stage-head" },
    h("div", { class: "stage-title" },
      kicker ? h("p", { class: "kicker" }, kicker) : null,
      h("h1", { class: "display", tabindex: "-1", id }, title),
      lede ? h("p", { class: "prose lede" }, lede) : null),
    actions && actions.length ? h("div", { class: "actions" }, actions) : null);
}

/** A margin note: kind is info | caution | poor | pen. The text is always plain; nothing here is a verdict. */
export function note(kind, ...kids) {
  const ic = kind === "caution" || kind === "poor" ? "warn" : "info";
  return h("aside", { class: `note ${kind}` }, icon(ic), h("div", {}, ...kids));
}

export function busy(label = "Loading") {
  return h("div", { class: "busy", role: "status" }, h("span", { class: "busy-bar" }), h("span", {}, label));
}

export function errorBox(err, retry) {
  const msg = err && err.message ? err.message : String(err || "Something went wrong.");
  return h("div", { class: "note poor errbox", role: "alert" }, icon("warn"),
    h("div", {}, h("p", {}, msg), retry ? h("button", { class: "btn small", type: "button", onclick: retry }, "Try again") : null));
}

/** Replace a host's children with a loading line, run `fn`, and show an error box if it throws. */
export async function guarded(host, label, fn) {
  clear(host);
  host.append(busy(label));
  try {
    const out = await fn();
    return out;
  } catch (e) {
    clear(host);
    host.append(errorBox(e, () => guarded(host, label, fn)));
    return null;
  }
}

/** key/value definition list */
export function facts(pairs) {
  return h("dl", { class: "facts" }, pairs.filter(Boolean).map(([k, v, t]) => h("div", { title: t || null }, h("dt", {}, k), h("dd", { class: "num" }, v))));
}

export function pill(text, kind = "") { return h("span", { class: `pill ${kind}` }, text); }

/** Plain-language names for the capture-quality grades. */
export const GRADE_WORDS = {
  A: "Very sensitive", B: "Sensitive", C: "Moderate", D: "Coarse", E: "Only large movements",
};

export function tag(el, fn) { fn && fn(el); return el; }

/** localStorage that never throws (private windows, blocked storage). */
export const store = {
  get(k, d = null) { try { const v = localStorage.getItem(`onset.${k}`); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(`onset.${k}`, JSON.stringify(v)); } catch { /* ignore */ } },
};

export const reduceMotion = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
