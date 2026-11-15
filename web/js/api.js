// All network access.  The same interface can run against the local server or, in the published showcase, against
// static files (window.ONSET_STATIC): then results come from data/library/<id>/ and nothing is computed.
export const STATIC = !!window.ONSET_STATIC;

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

async function j(url, opts = {}) {
  let r;
  try {
    r = await fetch(url, opts);
  } catch (e) {
    throw new ApiError(STATIC ? "This view needs the local ONSET server." : "The ONSET server is not reachable. Is it still running?", 0);
  }
  let body = null;
  const ct = r.headers.get("content-type") || "";
  if (ct.includes("json")) body = await r.json().catch(() => null);
  if (!r.ok) throw new ApiError((body && body.error) || `Request failed (${r.status}).`, r.status);
  return body;
}
const post = (url, data) => j(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data || {}) });

/** A reference to an analysed recording: {kind: "library"|"session", id}. */
export const ref = (kind, id) => ({ kind, id });
export function fileUrl(r, name) {
  if (STATIC) return `data/library/${r.id}/${name}`;
  return r.kind === "library" ? `api/library/${r.id}/files/${name}` : `api/results/${r.id}/files/${name}`;
}

export const api = {
  status: () => (STATIC ? Promise.resolve({ version: "static", calibrated: true, library: 0, static: true, privacy: { local_only: true, retention_hours: 24 }, limits: {} }) : j("api/status")),
  library: () => j(STATIC ? "data/library/manifest.json" : "api/library"),
  async result(r) {
    if (r.kind === "library") {
      if (STATIC) {
        const [res, truth, score] = await Promise.all([j(`data/library/${r.id}/result.json`), j(`data/library/${r.id}/truth.json`), j(`data/library/${r.id}/score.json`)]);
        return { ...res, truth, score };
      }
      return j(`api/library/${r.id}/result`);
    }
    return j(`api/results/${r.id}`);
  },
  results: () => (STATIC ? Promise.resolve([]) : j("api/results")),
  remove: (id) => j(`api/results/${id}`, { method: "DELETE" }),
  purge: () => post("api/purge"),
  preflight(blobs, times) {
    const fd = new FormData();
    blobs.forEach((b, i) => fd.append("frames", b, `f${i}.jpg`));
    fd.append("times", JSON.stringify(times));
    return j("api/preflight", { method: "POST", body: fd });
  },
  captureStart: (data) => post("api/capture/start", data),
  captureFrames(id, items) {            // items: [{idx, t, blob}]
    const fd = new FormData();
    fd.append("meta", JSON.stringify(items.map((x) => [x.idx, x.t])));
    items.forEach((x) => fd.append(`f${x.idx}`, x.blob, `${x.idx}.jpg`));
    return j(`api/capture/${id}/frames`, { method: "POST", body: fd });
  },
  captureFinish: (id, label) => post(`api/capture/${id}/finish`, { label }),
  upload(file, label, consent) {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("label", label || "");
    fd.append("consent", consent ? "1" : "0");
    return j("api/upload", { method: "POST", body: fd });
  },
  stressOptions: () => j("api/stress/options"),
  stress: (data) => post("api/stress", data),
  job: (id) => j(`api/jobs/${id}`),
  evidence: () => j(STATIC ? "data/evidence.json" : "api/evidence"),
  atlas: () => j(STATIC ? "data/atlas.json" : "api/atlas"),
  cue: {
    evidence: () => j(STATIC ? "data/cue_evidence.json" : "api/cue/evidence"),
    predictive: (d) => post("api/cue/predictive", d),
    roc: (d) => post("api/cue/roc", d),
    rates: (d) => post("api/cue/rates", d),
    forking: (d) => post("api/cue/forking", d),
    segments: (d) => post("api/cue/segments", d),
    audit: (d) => post("api/cue/audit", d),
  },
};

/** Poll a job until it ends; ``onProgress({progress, stage, state, queue_position})`` is called on every tick. */
export async function pollJob(id, onProgress, signal) {
  for (;;) {
    if (signal && signal.aborted) throw new ApiError("Cancelled", 0);
    const job = await api.job(id);
    onProgress && onProgress(job);
    if (job.state === "done") return job;
    if (job.state === "error") throw new ApiError(job.error || "The analysis failed.", 500);
    await new Promise((r) => setTimeout(r, job.state === "queued" ? 1200 : 700));
  }
}
