"""The documents quote the numbers the data files hold, and each figure claims only the scope it was computed on.

These are the checks a careful reader would do by hand with the documents next to the JSON.  They exist because the first pass
found slips of exactly this kind: a recall of 0.7045 written as 0.71, a training time of 48 s written as 40 s, and a fitted factor
described as never having seen a test face when it had.  (They read the shipped files in ``data/``; they do not re-run anything.)"""
import json
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _json(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def _doc(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _missing(text, wanted):
    return [w for w in wanted if w not in text]


@pytest.fixture(scope="module")
def ev():
    return _json("data", "evidence", "phantom_eval.json")


@pytest.fixture(scope="module")
def cal():
    return _json("data", "calibration.json")


@pytest.fixture(scope="module")
def learn():
    return _json("data", "evidence", "learn_synthetic_check.json")


@pytest.fixture(scope="module")
def evaluation():
    return _doc("docs", "EVALUATION.md")


# ------------------------------------------------------------------------------------------------ one number, one place
def test_the_sensitivity_factor_is_the_same_number_in_every_file_and_its_check_adds_up(cal, ev):
    S = ev["sensitivity"]
    assert cal["mda_scale"] == pytest.approx(S["mda_check"]["single_au_50pct"]) == pytest.approx(ev["summary"]["mda_scale"])
    assert cal["mda90_scale"] == pytest.approx(S["mda_check"]["single_au_90pct"])
    sp = S["mda_split"]
    assert sp["calibration"]["n_single"] + sp["test"]["n_single"] == S["mda_check"]["n_single"]
    assert sp["transfer"]["n"] == sp["test"]["n_single"]
    assert ev["summary"]["mda_scale_cal"] == pytest.approx(sp["calibration"]["single_au_50pct"])
    assert ev["summary"]["mda_scale_test"] == pytest.approx(sp["test"]["single_au_50pct"])


def test_the_pooled_factor_lies_between_the_two_groups_it_was_pooled_from(ev):
    sp = ev["sensitivity"]["mda_split"]
    lo, hi = sorted((sp["calibration"]["single_au_50pct"], sp["test"]["single_au_50pct"]))
    assert lo <= ev["sensitivity"]["mda_check"]["single_au_50pct"] <= hi


def test_the_scope_statement_names_the_smallest_movement_factors_as_the_one_exception(ev):
    sc = ev["scope"]
    assert "smallest-detectable-movement" in sc["fitted_on_all_faces"] and "sensitivity.mda_split" in sc["fitted_on_all_faces"]
    assert "smallest" not in sc["fitted_on_calibration_faces"]
    assert {"fitted_on_calibration_faces", "fitted_on_all_faces", "held_out_test_faces", "all_faces_descriptive"} <= set(sc)


# ------------------------------------------------------------------------------------------------ the phantom corpus
def test_the_headline_numbers_in_the_documents_are_the_ones_in_the_data(ev, cal, evaluation):
    s = ev["summary"]
    two = ["recall_test", "recall_loose_test", "fa_per_10s_test", "precision_test", "au_f1_test", "au_exact_set_test", "amp_coverage_test", "amp_ratio_median"]
    wanted = [f"{s[k]:.2f}" for k in two] + [f"{s['au_ece_test']:.3f}", f"{s['au_ece_default_test']:.3f}", f"{s['mda_scale']:.2f}",
                                              f"{cal['mda90_scale']:.1f}", f"{s['spot_threshold']:.2f}"]
    assert not _missing(evaluation, wanted)
    card = _doc("docs", "MODEL_CARD.md")
    assert not _missing(card, [f"{s[k]:.2f}" for k in two[:6] + ["amp_coverage_test", "amp_ratio_median"]] + [f"{s['au_ece_test']:.3f}"])


def test_the_category_set_figures_in_the_documents_are_the_ones_in_the_data(ev, evaluation):
    cats = ev["categories"]
    assert not _missing(evaluation, [f"{cats[k]['coverage_test']:.2f}" for k in cats] + [f"{cats[k]['mean_set_size']:.1f}" for k in cats])


def test_recall_by_condition_in_the_evaluation_is_the_recall_in_the_data(ev, evaluation):
    sp = ev["spotting"]
    n, c = sp["by_nuisance"], sp["by_camera"]
    phrases = [f"still {n['still']['recall_loose']:.2f}", f"normal {n['normal']['recall_loose']:.2f}", f"busy {n['busy']['recall_loose']:.2f}",
               f"webcam 30 fps {c['webcam30']['recall_loose']:.2f}", f"phone 30 fps {c['phone30']['recall_loose']:.2f}",
               f"low-light 24 fps {c['webcam24_lowlight']['recall_loose']:.2f}", f"15 fps {c['webcam15']['recall_loose']:.2f}",
               f"100 fps {c['highspeed100']['recall_loose']:.2f}"]
    flat = " ".join(evaluation.split())
    assert not _missing(flat, phrases)
    card = " ".join(_doc("docs", "MODEL_CARD.md").split())
    assert not _missing(card, [f"{n['still']['recall_loose']:.2f} (still head)", f"{n['normal']['recall_loose']:.2f} (normal)", f"{n['busy']['recall_loose']:.2f} (busy)"])


def test_the_detection_ceilings_quoted_are_the_range_of_the_fitted_ceilings(ev, evaluation):
    cs = [v["ceiling"] for k, v in ev["sensitivity"]["envelope"].items() if k != "all"]
    assert "about 65–90 %" in evaluation and round(min(cs) * 20) / 20 == pytest.approx(0.65) and round(max(cs) * 10) / 10 == pytest.approx(0.9)


def test_the_controls_quoted_in_the_evaluation_are_the_controls_in_the_data(ev, evaluation):
    c = ev["controls"]
    flat = " ".join(evaluation.split())
    assert not _missing(flat, [f"{c['shuffled_truth_f1']['mean']:.2f}", f"{c['shuffled_truth_f1']['q95']:.2f}", f"{c['prior_only_f1']:.2f}",
                               f"{round(100 * c['false_alarm_named_aus']['share_with_any'])} %",
                               f"{c['false_alarm_named_aus']['mean_named']:.2f}", str(c["false_alarm_named_aus"]["n_false_alarms"])])
    mf = ev["spotting"]["movement_free"]
    assert not _missing(flat, [f"{mf['fp']} false alarms in {mf['seconds']:.0f} s", f"{mf['fa_per_10s']:.2f} per 10 s"])


def test_the_sensitivity_split_quoted_in_the_evaluation_is_the_split_in_the_data(ev, evaluation):
    sp = ev["sensitivity"]["mda_split"]
    flat = " ".join(evaluation.split())
    wanted = [f"{sp['calibration']['single_au_50pct']:.2f}×", f"{sp['test']['single_au_50pct']:.2f}×", f"{sp['calibration']['single_au_90pct']:.2f}×",
              f"{sp['test']['single_au_90pct']:.2f}×",
              f"| {ev['sensitivity']['mda_check']['n_single']} |", f"| {sp['calibration']['n_single']} |", f"| {sp['test']['n_single']} |"]
    wanted += [f"{lo:.2f}–{hi:.2f}" for lo, hi in (sp["calibration"]["a50_ci"], sp["test"]["a50_ci"])]
    t = sp["transfer"]
    wanted += [f"predicts that {round(100 * t['predicted'])} % would be found; {round(100 * t['observed'])} %",
               f"Of the {t['n_at_or_above']} D–F movements", f"{t['found_at_or_above']} were found"]
    assert not _missing(flat, wanted)
    assert f"{sp['test']['single_au_50pct'] / sp['calibration']['single_au_50pct']:.1f} times as large" in flat


# ------------------------------------------------------------------------------------------------ the learning harness reference run
def test_the_reference_run_figures_in_the_evaluation_are_the_ones_in_the_data(learn, evaluation):
    flat = " ".join(evaluation.split())
    rows = {r[0]: r for r in learn["rows"]}
    uf1 = lambda name: rows[name][2].split()[0]                                   # "0.756 [0.70, 0.80]" -> "0.756"
    wanted = [f"{float(uf1(m)):.2f}" for m in ("majority", "prior", "expert", "linear", "graphnet")]
    lat = learn["models"]["graphnet"]["latency"]
    wanted += [f"about {round(lat['fit_seconds_per_fold'])} s to train", f"{lat['inference_ms_per_clip']:.1f} ms per"]
    wanted += [f"{learn['latency']['feature_extraction_s_per_clip']['mean']:.1f} s per clip"]
    ece = [learn["models"][m]["calibration"]["ece"] for m in ("expert", "linear", "graphnet")]
    cov = [learn["models"][m]["sets"]["coverage"] for m in ("expert", "linear", "graphnet")]
    size = [learn["models"][m]["sets"]["mean_size"] for m in ("expert", "linear", "graphnet")]
    wanted += [f"ECE is {min(ece):.2f} to {max(ece):.2f}", f"({min(cov):.2f} to {max(cov):.2f} against", f"{min(size):.1f} to {max(size):.1f} labels"]
    ctl = learn["controls"]
    wanted += [f"{ctl['shuffled_labels']['uf1_mean']:.2f} against {ctl['shuffled_labels']['chance_uf1']:.2f}",
               f"{round(100 * ctl['identity_probe']['accuracy'])} % against {round(100 * ctl['identity_probe']['chance'])} % by chance"]
    assert not _missing(flat, wanted)


def test_the_pairwise_tests_quoted_in_the_evaluation_are_the_ones_in_the_data(learn, evaluation):
    flat = " ".join(evaluation.split())
    by = {(p["a"], p["b"]): p for p in learn["pairwise"]}
    lin_graph = by.get(("linear", "graphnet")) or by.get(("graphnet", "linear"))
    assert lin_graph is not None
    assert f"{abs(lin_graph['difference']):.3f}" in flat and f"{lin_graph['p_holm']:.2f}" in flat
    against_majority = [p for p in learn["pairwise"] if "majority" in (p["a"], p["b"]) and {p["a"], p["b"]} & {"linear", "graphnet", "expert"}]
    assert against_majority and all(f"{p['p_holm']:.3f}" in flat for p in against_majority[:1])
