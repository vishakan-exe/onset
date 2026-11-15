"""Command line: run a subject-independent evaluation on licensed micro-expression data (or on the synthetic check set).

    python -m onset.learn.run --data casme2=/data/CASME2 --scheme casme2_5 --protocol loso --out runs/casme2_loso
    python -m onset.learn.run --data casme2=/data/CASME2 --data samm=/data/SAMM --scheme megc3 --protocol composite --out runs/megc3
    python -m onset.learn.run --data casme2=/data/CASME2 --data samm=/data/SAMM --scheme megc3 --protocol lodo --out runs/cross_corpus
    python -m onset.learn.run --synthetic --out runs/synthetic_check          # checks the machinery; NOT a benchmark

Writes into ``--out``: ``results.json`` (everything), ``predictions.csv`` (one row per clip x model x seed), ``config.json``
(arguments, package versions, seeds, annotation hashes, feature configuration), ``failures.csv`` (clips that could not be
measured), ``problems.csv`` (annotation rows that could not be used).  Then ``python -m onset.learn.report <out>`` turns the
run into the tables the interface shows.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import os
import platform
import sys
import time
from typing import Optional, Sequence

import numpy as np

from . import controls as CTL
from . import datasets as D
from . import evaluate as EV
from . import features as FE
from . import labels as L
from . import models as MD
from . import protocols as P

DETERMINISTIC = ("majority", "expert")       # the same predictions for every seed: run once
BASELINES = ("majority", "prior")
DEFAULT_MODELS = "majority,prior,expert,linear,graphnet"
SLOW = ("graphnet",)                         # models that report progress fold by fold (a leave-one-subject-out run takes hours on a CPU)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    raise TypeError(f"not serialisable: {type(o).__name__}")


def _clean(o):
    """Replace NaN/inf (not valid JSON) by None, recursively."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def versions() -> dict:
    out = dict(python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__)
    for mod in ("scipy", "sklearn", "pandas", "cv2", "torch", "openpyxl"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = None
    return out


# --------------------------------------------------------------------------------------------------------------
def _kv(items: Sequence[str], what: str) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"{what} expects NAME=PATH, got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_manifests(data: dict, annotations: dict, limit: Optional[int], template: str, fps: float) -> list:
    mans = []
    for name, root in data.items():
        if name == "generic":
            if "generic" not in annotations:
                raise SystemExit("--data generic=ROOT needs --annotations generic=SHEET")
            mans.append(D.generic(root, annotations["generic"], template, "custom", fps, limit))
        else:
            mans.append(D.load(name, root, annotations.get(name), limit))
    return mans


def select(records: list, scheme: str, feats: dict):
    """Clips that belong to the scheme *and* were measured, in a fixed order."""
    keep, y, excluded = D.with_scheme(records, scheme)
    ok = [(r, c) for r, c in zip(keep, y) if r.uid in feats]
    return [r for r, _ in ok], np.array([c for _, c in ok], int), excluded


def build_folds(protocol: str, recs: list, y: np.ndarray, k: int, seed: int) -> list:
    groups = [r.group for r in recs]
    if protocol == "loso":
        if len({r.dataset for r in recs}) > 1:
            print("note: several datasets with --protocol loso: groups are dataset-qualified, i.e. composite LOSO", file=sys.stderr)
        return P.loso(groups)
    if protocol == "composite":
        return P.composite(groups)
    if protocol == "lodo":
        return P.lodo([r.dataset for r in recs])
    if protocol == "kfold":
        return P.kfold(groups, y, k=k, seed=seed)
    raise SystemExit(f"unknown protocol '{protocol}'")


def data_summary(recs, y, classes, excluded, mans, failures) -> dict:
    n_c = np.bincount(y, minlength=len(classes))
    per_ds = {}
    for r, c in zip(recs, y):
        d = per_ds.setdefault(r.dataset, dict(clips=0, subjects=set(), classes=[0] * len(classes)))
        d["clips"] += 1
        d["subjects"].add(r.subject)
        d["classes"][c] += 1
    for d in per_ds.values():
        d["subjects"] = len(d["subjects"])
    return dict(classes=list(classes), class_counts=n_c.tolist(), n_clips=len(recs), n_subjects=len({r.group for r in recs}),
                per_dataset=per_ds, excluded_by_scheme=len(excluded),
                excluded_labels={k: sum(1 for r in excluded if (r.emotion or "(none)") == k) for k in {(r.emotion or "(none)") for r in excluded}},
                measurement_failures=len(failures), annotation_problems=sum(len(m.problems) for m in mans),
                annotation_sha256={m.dataset: m.annotation_sha256 for m in mans})


# --------------------------------------------------------------------------------------------------------------
def run_experiment(manifests: list, scheme: str, protocol: str, models: Sequence[str], seeds: Sequence[int], out: str, cache: Optional[str],
                   workers: int = 1, alpha: float = 0.1, n_boot: int = 1000, n_perm: int = 2000, k: int = 5, controls: bool = True,
                   calibration: Optional[dict] = None, graph_cfg: Optional[MD.GraphNetConfig] = None, feature_cfg: Optional[FE.FeatureConfig] = None,
                   args: Optional[dict] = None, quiet: bool = False) -> dict:
    t_start = time.time()
    os.makedirs(out, exist_ok=True)
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    classes = list(L.SCHEMES[scheme])
    records = [r for m in manifests for r in m.records]
    log(f"{len(records)} annotated clips from {len(manifests)} dataset(s); {sum(len(m.problems) for m in manifests)} annotation rows could not be used")
    fcfg = feature_cfg or FE.FeatureConfig(calibration=calibration)
    keep0, _, _ = D.with_scheme(records, scheme)
    log(f"{len(keep0)} clips carry a label of scheme '{scheme}' ({', '.join(classes)})")
    t0 = time.time()
    feats, failures = FE.extract_many(keep0, fcfg, cache, workers=workers,
                                      progress=lambda i, n: (i % 10 == 0 or i == n) and log(f"  measured {i}/{n} clips ({time.time() - t0:.0f} s)"))
    if failures:
        log(f"{len(failures)} clips could not be measured (see failures.csv)")
    recs, y, excluded = select(records, scheme, feats)
    if len(recs) < 10:
        raise SystemExit(f"only {len(recs)} usable clips: nothing meaningful can be evaluated")
    groups = np.array([r.group for r in recs])
    X = FE.stack([r.uid for r in recs], feats)
    folds = build_folds(protocol, recs, y, k, seeds[0] if seeds else 0)
    audit = P.audit(recs, folds)
    warns = list(audit["warnings"])
    absent = sorted({w for f in folds for w in P.check_train_classes(y, f, len(classes))})
    n_by_c = np.bincount(y, minlength=len(classes))
    if (n_by_c == 0).any():
        warns.append(f"classes with no clip at all: {[classes[i] for i in np.flatnonzero(n_by_c == 0)]}")
    summary = data_summary(recs, y, classes, excluded, manifests, failures)
    log(f"{len(recs)} clips / {summary['n_subjects']} subjects / {len(folds)} folds; class counts {dict(zip(classes, n_by_c.tolist()))}; audit ok")
    for w in warns + absent[:3]:
        log("  warning:", w)

    # ----- models x seeds ---------------------------------------------------------------------------------------------
    cps: dict = {}
    for name in models:
        if name == "graphnet":
            try:
                import torch  # noqa: F401
            except ImportError:
                log("  skipping graphnet: PyTorch is not installed (pip install torch)")
                continue
        factory = (lambda n=name: MD.make_model(n, scheme, graph_cfg))
        runs = []
        t1 = time.time()
        for s in (seeds[:1] if name in DETERMINISTIC else seeds):
            tick = ((lambda i, n, nm=name, sd=s, t=time.time(): log(f"    {nm} seed {sd}: fold {i}/{n} ({time.time() - t:.0f} s)"))
                    if name in SLOW else None)
            runs.append(EV.cross_predict(factory, X, y, groups, folds, len(classes), seed=s, alpha=alpha, progress=tick))
        if name in DETERMINISTIC:
            runs = [runs[0]] * len(seeds)
        cps[name] = runs
        u = np.mean([np.mean(r.pred == y) for r in runs])
        log(f"  {name:15s} accuracy {u:.3f}   ({time.time() - t1:.0f} s)")
    model_names = list(cps)
    reports = {n: EV.model_report(cps[n], y, groups, classes, n_boot=n_boot) for n in model_names}
    learned = [n for n in model_names if n not in BASELINES]
    pairs = [(a, b) for a, b in itertools.combinations([n for n in learned], 2)] + [(n, "majority") for n in learned if "majority" in cps]
    tests = EV.pairwise(cps, y, groups, classes, pairs, n_perm=n_perm) if pairs else []

    # ----- negative controls ------------------------------------------------------------------------------------------
    ctl = {}
    if controls:
        lin = (lambda: MD.make_model("linear", scheme))
        log("  controls: shuffled labels, identity probe, leakage gap")
        real = reports["linear"]["headline"]["uf1"]["value"] if "linear" in reports else None
        ctl["shuffled_labels"] = CTL.shuffled_labels(lin, X, y, groups, folds, len(classes), seeds=tuple(seeds[:3]) or (0,), alpha=alpha, real_uf1=real)
        ctl["identity_probe"] = CTL.identity_probe(X, groups, seed=seeds[0] if seeds else 0)
        if len(set(groups.tolist())) >= 6:
            ctl["leakage_gap"] = CTL.leakage_gap(lin, X, y, groups, len(classes), k=k, seed=seeds[0] if seeds else 0, alpha=alpha)
        for v in ctl.values():
            log("   -", v.get("reading", v.get("reason", "")))

    # ----- latency ---------------------------------------------------------------------------------------------------
    fx = np.asarray(X.seconds, float)
    latency = dict(feature_extraction_s_per_clip=dict(mean=float(fx.mean()), median=float(np.median(fx)), p90=float(np.quantile(fx, 0.9))),
                   note="feature extraction = landmarks + stabilisation + dense flow + AU engine on the clip; it dominates model inference")

    results = dict(
        name=os.path.basename(os.path.normpath(out)), created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        scheme=scheme, classes=classes, protocol=protocol, n_folds=len(folds), alpha=alpha, seeds=list(seeds),
        data=summary, audit=dict(ok=audit["ok"], warnings=warns + absent), models=reports, pairwise=tests, controls=ctl, latency=latency,
        graph_config=(graph_cfg.__dict__ if graph_cfg is not None and "graphnet" in cps else None),
        wall_seconds=float(time.time() - t_start),
        caveat=("Numbers are subject-independent and pooled over folds; intervals resample subjects. They describe this dataset, this label "
                "scheme and these annotations - not people in general, and not deception or emotion detection."))
    config = dict(args=args or {}, versions=versions(), feature_config=dict(key=fcfg.key(), steps=fcfg.steps, max_side=fcfg.max_side,
                  max_frames=fcfg.max_frames, calibration=fcfg.calibration), seeds=list(seeds), annotation_sha256=summary["annotation_sha256"],
                  scheme_map={k: L.map_label(k, scheme) for k in sorted({r.emotion for r in records if r.emotion})})
    _write(out, results, config, recs, y, groups, cps, folds, classes, failures, manifests)
    log(f"wrote {out}/results.json  ({results['wall_seconds']:.0f} s)")
    return results


def _write(out, results, config, recs, y, groups, cps, folds, classes, failures, manifests):
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(_clean(results), fh, indent=1, default=_jsonable)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(_clean(config), fh, indent=1, default=_jsonable)
    with open(os.path.join(out, "predictions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "dataset", "subject", "fold", "truth", "model", "seed", "pred", "p_pred", "set"] + [f"p_{c}" for c in classes])
        for name, runs in cps.items():
            for si, cp in enumerate(runs):
                for i, r in enumerate(recs):
                    w.writerow([r.uid, r.dataset, r.subject, folds[cp.fold[i]].name, classes[y[i]], name, cp.seed, classes[cp.pred[i]],
                                f"{cp.probs[i, cp.pred[i]]:.4f}", "|".join(classes[j] for j in cp.sets[i])] + [f"{p:.4f}" for p in cp.probs[i]])
    with open(os.path.join(out, "failures.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "reason"])
        for f in failures:
            w.writerow([f["uid"], f["reason"]])
    with open(os.path.join(out, "problems.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "row", "clip", "problem"])
        for m in manifests:
            for p in m.problems:
                w.writerow([m.dataset, p["row"], p["clip"], p["problem"]])


# --------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m onset.learn.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", default=[], metavar="NAME=ROOT",
                    help="dataset adapter and folder; repeat for several (casme2, samm, mmew, fourd, casme3a, generic)")
    ap.add_argument("--annotations", action="append", default=[], metavar="NAME=SHEET", help="annotation sheet when it cannot be found under ROOT")
    ap.add_argument("--template", default="{root}/{subject}/{clip}", help="generic adapter: folder template ({root},{subject},{clip},{emotion})")
    ap.add_argument("--fps", type=float, default=0.0, help="generic adapter: frame rate of the release")
    ap.add_argument("--synthetic", action="store_true", help="run on a phantom dataset laid out like CASME II (checks the machinery; not a benchmark)")
    ap.add_argument("--synthetic-subjects", type=int, default=12)
    ap.add_argument("--synthetic-clips", type=int, default=6, help="clips per synthetic subject")
    ap.add_argument("--scheme", choices=list(L.SCHEMES), default="megc3")
    ap.add_argument("--protocol", choices=["loso", "composite", "lodo", "kfold"], default="loso")
    ap.add_argument("--folds", type=int, default=5, help="k for --protocol kfold")
    ap.add_argument("--models", default=DEFAULT_MODELS, help=f"comma list of {', '.join(MD.MODEL_NAMES)}")
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (0..N-1)")
    ap.add_argument("--alpha", type=float, default=0.1, help="conformal miscoverage level")
    ap.add_argument("--boot", type=int, default=1000, help="subject-bootstrap replicates for intervals")
    ap.add_argument("--perm", type=int, default=2000, help="permutations for paired tests")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--cache", default=None, help="feature cache folder (default <out>/features)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N usable annotation rows per dataset (quick tests)")
    ap.add_argument("--calibration", default=None, help="engine calibration json (default data/calibration.json when present)")
    ap.add_argument("--max-side", type=int, default=960)
    ap.add_argument("--graph-epochs", type=int, default=40)
    ap.add_argument("--graph-members", type=int, default=3)
    ap.add_argument("--graph-dim", type=int, default=32)
    ap.add_argument("--graph-threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> dict:
    a = build_parser().parse_args(argv)
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cal_path = a.calibration or os.path.join(here, "data", "calibration.json")
    calibration = json.load(open(cal_path)) if os.path.exists(cal_path) else None
    out = a.out
    os.makedirs(out, exist_ok=True)
    data, ann = _kv(a.data, "--data"), _kv(a.annotations, "--annotations")
    if a.synthetic:
        from . import synthetic as SY
        root = os.path.join(out, "synthetic_data")
        info = SY.build(root, subjects=a.synthetic_subjects, clips_per_subject=a.synthetic_clips, seed=0)
        print(f"synthetic dataset: {info['clips']} clips, {info['subjects']} identities in {root}", flush=True)
        data, ann = {"casme2": root}, {"casme2": info["sheet"]}
        if a.scheme == "basic7":
            raise SystemExit("the synthetic set only carries the megc3 and casme2_5 labels")
    if not data:
        raise SystemExit("nothing to do: pass --data NAME=ROOT (e.g. --data casme2=/path/to/CASME2) or --synthetic")
    mans = load_manifests(data, ann, a.limit, a.template, a.fps)
    for m in mans:
        s = m.summary()
        print(f"{m.dataset}: {s['clips']} usable of {s['rows']} rows, {s['subjects']} subjects, {s['problems']} problems, emotions {s['emotions']}", flush=True)
    names = [n.strip() for n in a.models.split(",") if n.strip()]
    bad = [n for n in names if n not in MD.MODEL_NAMES]
    if bad:
        raise SystemExit(f"unknown model(s) {bad}; choose from {', '.join(MD.MODEL_NAMES)}")
    gcfg = MD.GraphNetConfig(epochs=a.graph_epochs, members=a.graph_members, d=a.graph_dim, threads=a.graph_threads)
    fcfg = FE.FeatureConfig(max_side=a.max_side, calibration=calibration)
    return run_experiment(mans, a.scheme, a.protocol, names, list(range(a.seeds)), out, a.cache or os.path.join(out, "features"), a.workers,
                          a.alpha, a.boot, a.perm, a.folds, not a.no_controls, calibration, gcfg, fcfg, vars(a))


if __name__ == "__main__":
    main()
