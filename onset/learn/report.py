"""Turn a finished run folder into a readable report and into the evidence file the interface shows.

    python -m onset.learn.report runs/casme2_loso                     # writes runs/casme2_loso/report.md and data/evidence/learn_<name>.json
    python -m onset.learn.report runs/synthetic_check --synthetic     # marks the run as a machinery check, not a benchmark
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

BASELINES = ("majority", "prior")
LABELS = dict(majority="always the commonest class", prior="random guess with the training frequencies",
              expert="ONSET's hand-written FACS prototypes (untrained)", linear="logistic regression: pattern + timing + AU evidence",
              linear_pattern="logistic regression: displacement pattern only", linear_au="logistic regression: AU evidence only",
              graphnet="graph-temporal transformer over the 27 regions")


def _f(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


def _ci(h, d=3):
    if h is None:
        return "-"
    lo, hi = h.get("lo"), h.get("hi")
    return f"{_f(h['value'], d)} [{_f(lo, d)}, {_f(hi, d)}]"


def _best(res: dict) -> Optional[str]:
    cands = [n for n in res["models"] if n not in BASELINES]
    return max(cands, key=lambda n: res["models"][n]["headline"]["uf1"]["value"]) if cands else None


def table(res: dict) -> tuple[list, list]:
    cols = ["model", "what it is", "UF1 (95 % CI)", "UAR (95 % CI)", "accuracy (95 % CI)", "ECE", "set coverage", "mean set size", "ms / clip"]
    rows = []
    for n, m in res["models"].items():
        h = m["headline"]
        rows.append([n, LABELS.get(n, ""), _ci(h["uf1"]), _ci(h["uar"]), _ci(h["accuracy"]), _f(m["calibration"]["ece"]),
                     _f(m["sets"]["coverage"]), _f(m["sets"]["mean_size"], 2), _f(m["latency"]["inference_ms_per_clip"], 2)])
    any_m = next(iter(res["models"].values()))
    rows.append(["chance", "uniform guessing (expected)", _f(any_m["chance"]["uf1"]), _f(any_m["chance"]["uar"]), _f(any_m["chance"]["majority_accuracy"]) + " (majority share)", "", "", "", ""])
    return cols, rows


def markdown(res: dict, cfg: Optional[dict] = None, synthetic: bool = False) -> str:
    d = res["data"]
    out = [f"# Learning run: {res['name']}", ""]
    if synthetic:
        out += ["> **Synthetic data.** Procedural faces with scripted movements. This run checks that the code is wired correctly; it says nothing about human micro-expressions.", ""]
    out += [f"* scheme `{res['scheme']}` ({', '.join(res['classes'])}), protocol `{res['protocol']}`, {res['n_folds']} folds, seeds {res['seeds']}",
            f"* {d['n_clips']} clips from {d['n_subjects']} subjects; class counts {dict(zip(d['classes'], d['class_counts']))}",
            f"* {d['excluded_by_scheme']} clips outside the scheme (not relabelled), {d['measurement_failures']} clips could not be measured, {d['annotation_problems']} annotation rows unusable",
            f"* leakage audit: {'passed' if res['audit']['ok'] else 'FAILED'}" + ("; warnings: " + "; ".join(res["audit"]["warnings"]) if res["audit"]["warnings"] else ""), ""]
    cols, rows = table(res)
    out += ["## Headline (subject-independent, pooled over folds)", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    out += ["", "UF1 = unweighted F1 over classes, UAR = unweighted average recall. Intervals resample subjects (the unit that is independent), "
            "not clips. Chance is the expected UF1 of uniform guessing on this class mix.", ""]
    b = _best(res)
    if b:
        m = res["models"][b]
        out += [f"## Per class: {b}", "", "| class | support | predicted | precision | recall | F1 |", "|---|---|---|---|---|---|"]
        out += [f"| {c['label']} | {c['support']} | {c['predicted']} | {_f(c['precision'])} | {_f(c['recall'])} | {_f(c['f1'])} |" for c in m["per_class"]]
        out += ["", "Confusion (rows = truth, columns = prediction):", "", "| |" + "|".join(res["classes"]) + "|", "|---|" + "---|" * len(res["classes"])]
        out += [f"| **{res['classes'][i]}** |" + "|".join(str(v) for v in row) + "|" for i, row in enumerate(m["confusion"])]
        s = m["sets"]
        out += ["", f"Conformal prediction sets at 1-alpha = {1 - s['alpha']:.2f}: coverage {_f(s['coverage'])}, mean size {_f(s['mean_size'], 2)}, "
                    f"singleton rate {_f(s['singleton_rate'])}, accuracy when singleton {_f(s['accuracy_when_singleton'])}.", ""]
    if res.get("pairwise"):
        out += ["## Paired subject-level permutation tests (UF1)", "", "| A | B | A - B | p | p (Holm) |", "|---|---|---|---|---|"]
        out += [f"| {t['a']} | {t['b']} | {t['difference']:+.3f} | {t['p']:.4f} | {t['p_holm']:.4f} |" for t in res["pairwise"]]
        out.append("")
    if res.get("controls"):
        out += ["## Controls", ""]
        for k, v in res["controls"].items():
            out.append(f"* **{k.replace('_', ' ')}**: {v.get('reading', v.get('reason', ''))}")
        out.append("")
    lat = res["latency"]["feature_extraction_s_per_clip"]
    out += ["## Latency", "", f"Feature extraction (landmarks, stabilisation, dense flow, AU engine) {lat['mean']:.2f} s per clip on average (median {lat['median']:.2f}, 90th percentile {lat['p90']:.2f}); "
            "model inference is in the table above.", "", "## Reproducibility", ""]
    if cfg:
        v = cfg.get("versions", {})
        out += [f"* package versions: " + ", ".join(f"{k} {x}" for k, x in v.items() if x and k != "platform"), f"* annotation sha256: {cfg.get('annotation_sha256')}",
                f"* feature configuration key: {cfg.get('feature_config', {}).get('key')}", f"* engine calibration used for features: {bool(cfg.get('feature_config', {}).get('calibration'))}"]
    out += ["", f"_{res['caveat']}_", ""]
    return "\n".join(out)


def evidence_payload(res: dict, cfg: Optional[dict] = None, synthetic: bool = False) -> dict:
    cols, rows = table(res)
    b = _best(res)
    models = {}
    for n, m in res["models"].items():
        models[n] = dict(headline={k: dict(value=v["value"], lo=v["lo"], hi=v["hi"], seed_sd=v.get("seed_sd")) for k, v in m["headline"].items()},
                         per_class=m["per_class"], confusion=m["confusion"], calibration=m["calibration"], sets=m["sets"],
                         risk_coverage=m["risk_coverage"], latency=m["latency"], chance=m["chance"], n_parameters=m.get("n_parameters"),
                         label=LABELS.get(n, ""))
    note = ("Synthetic data: procedural faces with scripted movements and invented labels, laid out like CASME II. This run checks that the "
            "folds, models, calibration and controls are wired correctly. It is not a benchmark and says nothing about human micro-expressions."
            if synthetic else
            "Subject-independent results on the stated dataset and label scheme, pooled over folds. Intervals resample subjects. They describe "
            "this annotated data, not people in general and not deception or emotion.")
    return dict(kind="learning_run", name=res["name"], title=f"Learning run: {res['name']}", synthetic=bool(synthetic), note=note,
                created=res["created"], scheme=res["scheme"], protocol=res["protocol"], classes=res["classes"], n_folds=res["n_folds"],
                seeds=res["seeds"], data=res["data"], audit=res["audit"], columns=cols, rows=rows, best=b, models=models,
                pairwise=res.get("pairwise", []), controls=res.get("controls", {}), latency=res["latency"], caveat=res["caveat"],
                graph_config=res.get("graph_config"), versions=(cfg or {}).get("versions"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m onset.learn.report", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run folder written by onset.learn.run")
    ap.add_argument("--synthetic", action="store_true", help="mark the run as a machinery check on phantom data")
    ap.add_argument("--evidence-dir", default=None, help="where to write learn_<name>.json (default <project>/data/evidence; 'none' to skip)")
    a = ap.parse_args(argv)
    res = json.load(open(os.path.join(a.run, "results.json")))
    cfg_path = os.path.join(a.run, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None
    synthetic = a.synthetic or bool(cfg and cfg.get("args", {}).get("synthetic"))
    md = markdown(res, cfg, synthetic)
    with open(os.path.join(a.run, "report.md"), "w") as fh:
        fh.write(md)
    print(f"wrote {os.path.join(a.run, 'report.md')}")
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ev_dir = a.evidence_dir or os.path.join(here, "data", "evidence")
    if ev_dir.lower() != "none":
        os.makedirs(ev_dir, exist_ok=True)
        p = os.path.join(ev_dir, f"learn_{res['name']}.json")
        with open(p, "w") as fh:
            json.dump(evidence_payload(res, cfg, synthetic), fh, indent=1)
        print(f"wrote {p}")
    return md


if __name__ == "__main__":
    main()
