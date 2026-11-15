import json

import numpy as np
import pytest

from onset.learn import controls as CTL
from onset.learn import evaluate as EV
from onset.learn import metrics as MET
from onset.learn import models as MD
from onset.learn import protocols as P
from onset.learn import report as REP

from helpers import make_clipset

lin = lambda: MD.make_model("linear", "megc3")                                       # noqa: E731


def test_cross_predict_scores_every_clip_once_with_subject_disjoint_folds():
    X, y, g = make_clipset(10, 5, 3, signal=2.0)
    folds = P.loso(g)
    cp = EV.cross_predict(lin, X, y, g, folds, 3, seed=0)
    assert (cp.pred >= 0).all() and (cp.fold >= 0).all()
    assert all(s is not None and len(s) >= 1 for s in cp.sets)
    assert np.allclose(cp.probs.sum(1), 1)
    for i, f in enumerate(folds):
        assert set(np.flatnonzero(cp.fold == i)) == set(f.test)
    assert MET.uf1(y, cp.pred, 3) > 0.7
    assert cp.infer_ms >= 0 and len(cp.fit_seconds) == len(folds)


def test_cross_predict_with_shuffled_training_labels_falls_to_chance():
    X, y, g = make_clipset(12, 6, 3, signal=2.0)
    folds = P.loso(g)
    real = EV.cross_predict(lin, X, y, g, folds, 3, seed=0)
    shuf = EV.cross_predict(lin, X, y, g, folds, 3, seed=0, shuffle_train=True)
    chance = MET.chance_uf1_uniform(y, 3)
    assert MET.uf1(y, real.pred, 3) > chance + 0.3
    assert MET.uf1(y, shuf.pred, 3) < chance + 0.12


def test_expert_gets_calibration_material_from_its_own_training_clips():
    X, y, g = make_clipset(8, 6, 3)
    cp = EV.cross_predict(lambda: MD.make_model("expert", "megc3"), X, y, g, P.kfold(g, y, k=4), 3, seed=0)
    assert any(t is not None for t in cp.temperatures)                                  # untrained model: its training-set predictions are held-out predictions
    assert "unclear" in cp.extras


def test_baselines_get_the_whole_label_set_as_prediction_set():
    X, y, g = make_clipset(8, 6, 3)
    cp = EV.cross_predict(lambda: MD.make_model("majority", "megc3"), X, y, g, P.loso(g), 3)
    assert all(len(s) == 3 for s in cp.sets)


def test_model_report_structure_and_interval():
    X, y, g = make_clipset(10, 6, 3, signal=2.0)
    folds = P.loso(g)
    cps = [EV.cross_predict(lin, X, y, g, folds, 3, seed=s) for s in (0, 1)]
    rep = EV.model_report(cps, y, g, ["a", "b", "c"], n_boot=150)
    h = rep["headline"]["uf1"]
    assert h["lo"] <= h["value"] <= h["hi"] and rep["n_seeds"] == 2 and rep["n_subjects"] == 10
    assert h["seed_sd"] is not None and set(rep) >= {"per_class", "confusion", "calibration", "sets", "risk_coverage", "latency", "chance"}
    assert [c["label"] for c in rep["per_class"]] == ["a", "b", "c"]
    assert rep["chance"]["uf1"] == pytest.approx(MET.chance_uf1_uniform(y, 3))
    json.dumps(rep, default=float)                                                       # serialisable


def test_pairwise_holm_and_direction():
    X, y, g = make_clipset(12, 6, 3, signal=2.0)
    folds = P.loso(g)
    preds = {"lin": [EV.cross_predict(lin, X, y, g, folds, 3)], "maj": [EV.cross_predict(lambda: MD.make_model("majority", "megc3"), X, y, g, folds, 3)]}
    rows = EV.pairwise(preds, y, g, ["a", "b", "c"], [("lin", "maj")], n_perm=300)
    assert rows[0]["difference"] > 0.2 and rows[0]["p"] < 0.05 and rows[0]["p_holm"] >= rows[0]["p"]


def test_shuffled_label_control_passes_on_clean_pipeline_and_fails_on_a_leak():
    X, y, g = make_clipset(12, 6, 3, signal=2.0)
    ok = CTL.shuffled_labels(lin, X, y, g, P.loso(g), 3, seeds=(0, 1), real_uf1=0.9)
    assert ok["passed"] and ok["uf1_mean"] < ok["chance_uf1"] + CTL.SUSPICIOUS_MARGIN and "as expected" in ok["reading"]
    assert ok["margin_over_shuffled"] > 0.4

    class Leaky:                                                                        # a model that peeks at the test labels it is asked about
        name = "leaky"
        cal_logits = cal_y = None
        fit_seconds = 0.0
        def fit(self, X, y, groups, n_classes, seed=0):
            return self
        def logits(self, X):
            return np.eye(3)[self.truth[[int(u[1:]) for u in X.uids]]] * 5.0
        def predict(self, X, rng=None):
            return self.logits(X).argmax(1)
    leak = Leaky()
    leak.truth = y
    out = CTL.shuffled_labels(lambda: leak, X, y, g, P.loso(g), 3, seeds=(0,))
    assert not out["passed"] and "TOO HIGH" in out["reading"]


def test_identity_probe_sees_fingerprints():
    X, y, g = make_clipset(10, 8, 3, signal=0.5, identity=3.0)
    strong = CTL.identity_probe(X, g)
    assert strong["accuracy"] > 0.6 and "carry identity" in strong["reading"]
    Xn, yn, gn = make_clipset(10, 8, 3, signal=0.5, identity=0.0, seed=2)
    weak = CTL.identity_probe(Xn, gn)
    assert weak["accuracy"] < strong["accuracy"]
    tiny = CTL.identity_probe(*[make_clipset(2, 3)[i] for i in (0, 2)])
    assert tiny.get("skipped")


def test_leakage_gap_shows_inflation_when_identity_predicts_the_label():
    X, y, g = make_clipset(14, 8, 3, signal=0.0, identity=2.0, label_by_subject=True, seed=3)
    out = CTL.leakage_gap(lin, X, y, g, 3, k=5)
    assert out["uf1_clip_level_INVALID"] > out["uf1_subject_disjoint"] + 0.25
    assert out["inflation"] > 0.25 and "leaky split" in out["reading"]


def test_report_outputs(tmp_path):
    from onset.learn import run as RUN
    X, y, g = make_clipset(8, 6, 3, signal=2.0)
    folds = P.loso(g)
    cps = {n: [EV.cross_predict(lambda n=n: MD.make_model(n, "megc3"), X, y, g, folds, 3)] for n in ("majority", "linear")}
    classes = ["positive", "negative", "surprise"]
    reps = {n: EV.model_report(c, y, g, classes, n_boot=60) for n, c in cps.items()}
    res = dict(name="t", created="2026-01-01T00:00:00+00:00", scheme="megc3", classes=classes, protocol="loso", n_folds=len(folds), alpha=0.1, seeds=[0],
               data=dict(classes=classes, class_counts=np.bincount(y).tolist(), n_clips=len(y), n_subjects=8, per_dataset={}, excluded_by_scheme=0,
                         measurement_failures=0, annotation_problems=0, annotation_sha256={}),
               audit=dict(ok=True, warnings=[]), models=reps, pairwise=EV.pairwise(cps, y, g, classes, [("linear", "majority")], n_perm=100),
               controls={}, latency=dict(feature_extraction_s_per_clip=dict(mean=1.0, median=1.0, p90=1.2), note=""), graph_config=None, wall_seconds=1.0,
               caveat="caveat")
    md = REP.markdown(res, None, synthetic=True)
    assert "Synthetic data" in md and "| linear |" in md and "Per class: linear" in md and "Paired subject-level" in md
    ev = REP.evidence_payload(res, None, synthetic=True)
    assert ev["synthetic"] and ev["best"] == "linear" and len(ev["rows"]) == 3 and ev["columns"][0] == "model"
    json.dumps(RUN._clean(ev), default=RUN._jsonable, allow_nan=False)
