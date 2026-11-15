"""Cue Lab: a research instrument for behavioural cues - and a critical look at reading lies from a face.

ONSET measures facial movement.  It does **not** output a deception score, a lie probability or a truthful /
deceptive label, and nothing in this module can be used to produce one.  The module exists because the idea
deserves a serious answer, not a refusal and not a gimmick:

* ``POSITION``, ``CUE_EVIDENCE``, ``REFS``, ``REGULATION`` - what the evidence supports, with every claim marked
  as *validated limit*, *hypothesis*, *mixed*, *unsupported* or *confound*, and every reference marked as
  opened or not opened during this build (so nobody mistakes a pointer for a verified fact);
* base-rate and effect-size calculators - why even an accurate-sounding detector would mostly accuse the
  innocent when lies are rare;
* ``segment_cues`` / ``compare_segments`` - the research-oriented alternative: descriptive, *within-person*,
  exposure-corrected behavioural measurements for stretches of one recording (baseline vs. target), tested
  with exact small-count statistics and corrected for the number of comparisons;
* ``audit_claims`` - for the data at hand, which sentences could responsibly be written and which could not;
* ``PROTOCOL`` - what a study would need before any of this could carry weight;
* ``assert_no_verdict`` - a guard used by the server and the tests: no result may carry a verdict-shaped key.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
from scipy import stats
from scipy.stats import norm

from ..vision import geometry as G

# ======================================================================================================
# 1. The position, the evidence and its provenance
# ======================================================================================================
POSITION = dict(
    headline="ONSET measures facial movement. It cannot tell you whether someone is lying.",
    can=[
        "Detect and time brief facial movements and say which action-unit pattern they most resemble, with uncertainty.",
        "Report how often such movements occurred in a stretch of video - rates corrected for the time each facial "
        "region was actually observable, with exact confidence intervals.",
        "Compare two stretches of the same recording (for example a baseline question and a target question) with "
        "tests that suit small counts and that are corrected for the number of comparisons made.",
        "Show what a hypothetical detector with a given effect size would achieve at a given base rate.",
    ],
    cannot=[
        "Establish that a person is lying or telling the truth.",
        "Turn a movement count or an AU pattern into a probability of deception.",
        "Say what the person felt or intended.",
        "Support any decision about a person: employment, legal, immigration, security, education.",
    ],
    why=[
        "No facial cue with validated individual-level accuracy was found among the sources opened for this build. "
        "The best-known candidate, the micro-expression, is rare and also occurs in people telling the truth.",
        "Base rates: when lies are uncommon, even a detector that is right 90% of the time on each class produces "
        "mostly false accusations (calculator below).",
        "Confounds: talking, chewing, thinking, blinking, being filmed, culture and personality all move the face. "
        "A within-person baseline is the minimum for any comparison, and even then the difference has many causes.",
        "Ground truth: a study needs to know who lied. Instructed, low-stakes lies are easy to collect and differ from "
        "high-stakes deception; verified high-stakes ground truth is scarce.",
    ],
    instead=[
        "A behavioural-cue instrument: measure observable movement per segment, test pre-registered hypotheses, "
        "report effect sizes with intervals, and say 'differs' - never 'lied'.",
        "A teaching instrument: let the user see how base rates, small counts and many comparisons produce confident "
        "but empty conclusions.",
    ],
)

STATUS_TEXT = {
    "validated_limit": "Validated limit - a well-supported finding about what cannot be concluded",
    "mixed": "Mixed or contested",
    "hypothesis": "Research hypothesis - proposed, not validated for individuals",
    "unsupported": "Not supported by the evidence found",
    "confound": "Confound - real movement with non-deceptive causes",
}

# opened=True: the page was actually read during this build.  via: the claims quoted here come from a secondary source.
REFS: dict[str, dict] = {
    "bond2006": dict(
        cite="Bond, C. F., Jr., & DePaulo, B. M. (2006). Accuracy of deception judgments. Personality and Social "
             "Psychology Review, 10(3), 214-234.",
        doi="10.1207/s15327957pspr1003_2", opened=True,
        finding="Meta-analysis of 206 documents and 24,483 judges: 54% of lie/truth judgments were correct "
                "(47% of lies and 61% of truths identified)."),
    "burgoon2018": dict(
        cite="Burgoon, J. K. (2018). Microexpressions are not the best way to catch a liar. Frontiers in Psychology, 9, 1672.",
        doi="10.3389/fpsyg.2018.01672", opened=True,
        finding="Opinion article arguing that micro-expression theory's propositions mostly fail; proposes the 'rigidity "
                "effect' (reduced movement under high stakes) as an alternative to be tested."),
    "porter2008": dict(
        cite="Porter, S., & ten Brinke, L. (2008). Reading between the lies: Identifying concealed and falsified "
             "emotions in universal facial expressions. Psychological Science.",
        doi="10.1111/j.1467-9280.2008.02116.x", opened=False, via="burgoon2018",
        finding="Figures used here (2% of about 700 high-stakes expressions were micro-expressions) are as reported by "
                "Burgoon (2018); the original article was not opened."),
    "depaulo2003": dict(
        cite="DePaulo, B. M., Lindsay, J. J., Malone, B. E., Muhlenbruck, L., Charlton, K., & Cooper, H. (2003). Cues to "
             "deception. Psychological Bulletin.",
        opened=False, finding="Not opened in this build. A meta-analysis of cues to deception; read it for effect sizes per cue."),
    "hartwig2011": dict(
        cite="Hartwig, M., & Bond, C. F., Jr. (2011). Why do lie-catchers fail? A lens model meta-analysis of human "
             "lie judgments. Psychological Bulletin.",
        opened=False, finding="Not opened in this build. Lens-model analysis of why judges perform near chance."),
    "vrij2019": dict(
        cite="Vrij, A., Hartwig, M., & Granhag, P. A. (2019). Reading lies: Nonverbal communication and deception. "
             "Annual Review of Psychology.",
        opened=False, finding="Not opened in this build. Review of nonverbal cues to deception."),
    "barrett2019": dict(
        cite="Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019). Emotional expressions "
             "reconsidered: Challenges to inferring emotion from human facial movements. Psychological Science in the "
             "Public Interest.",
        opened=False, finding="Not opened in this build (its record was refused by the fetch proxy). Review of what "
                              "facial movements do and do not reveal about emotion."),
    "yan2013": dict(
        cite="Yan, W.-J., Wu, Q., Liang, J., Chen, Y.-H., & Fu, X. (2013). How fast are the leaked facial expressions: "
             "The duration of micro-expressions. Journal of Nonverbal Behavior.",
        opened=False, finding="Not opened in this build. Basis for the usual upper duration bound of about 0.5 s."),
    "nrc2003": dict(
        cite="National Research Council (2003). The Polygraph and Lie Detection. Washington, DC: National Academies Press.",
        opened=False, finding="Not opened in this build. Committee review of polygraph validity."),
    "euai": dict(
        cite="Regulation (EU) 2024/1689 (Artificial Intelligence Act): Article 5(1)(f) and Annex III.",
        opened=False, finding="Not opened in this build; check the current consolidated text and any amendments."),
}

CUE_EVIDENCE: list[dict] = [
    dict(
        id="micro_rate", cue="Micro-expressions: brief, low-amplitude facial movements",
        onset_measure="Event count and rate per minute, duration, action-unit pattern (Bench, Evidence).",
        claim="Leaked micro-expressions reveal concealed emotion, and so reveal lying.",
        status="unsupported",
        known=[
            "Rare in the one high-stakes data set coded in detail: 2% of about 700 expressions were micro-expressions "
            "(Porter & ten Brinke 2008, as reported by Burgoon 2018).",
            "A later analysis counted six among people who were deceiving and eight among genuine pleaders, "
            "so they occurred in both groups (as reported by Burgoon 2018).",
            "Untrained observers could not separate truth from deception with them (Porter et al. 2012, as reported by "
            "Burgoon 2018).",
        ],
        assessment="Occurrence is real but rare and not specific to lying. Finding one proves nothing; finding none proves nothing.",
        refs=["burgoon2018", "porter2008"]),
    dict(
        id="human_accuracy", cue="Human lie detection from behaviour in general",
        onset_measure="Not measured; context for every other row.",
        claim="Trained or attentive people can tell lies from truth by watching behaviour.",
        status="validated_limit",
        known=["Across 206 documents and 24,483 judges, 54% of lie/truth judgments were correct: 47% of lies and 61% of "
               "truths were identified (Bond & DePaulo 2006)."],
        assessment="A statement about human judges, not a proof that no algorithm could do better - but nothing opened for this "
                "build shows that one can, with real stakes and unseen people.",
        refs=["bond2006"]),
    dict(
        id="rigidity", cue="Reduced overall movement ('rigidity') under high stakes",
        onset_measure="Movement activity per facial region and segment, excluding blinks and speech (Cue Lab, segment study).",
        claim="Liars move the face less, as they over-control their behaviour.",
        status="hypothesis",
        known=["Proposed by Burgoon (2018) as an alternative to micro-expressions. In one experiment guilty participants "
               "showed rigidity on 8 of 10 facial movements measured (Pentland et al. 2017, as reported by Burgoon 2018).",
               "The same article says the causal mechanism is unresolved and needs experiments."],
        assessment="Interesting and testable, at an early stage. Reduced movement also follows from nervousness, concentration, "
                "instructions to stay still, or simply from the person's habits.",
        refs=["burgoon2018"]),
    dict(
        id="smile", cue="Smiles and other positive expressions while answering",
        onset_measure="AU6 / AU12 events and their rate (Bench).",
        claim="A smile at the wrong moment signals deceit (or its absence signals truth).",
        status="unsupported",
        known=["Burgoon (2018) notes that felt emotion and expression do not map one-to-one and that smiles and other "
               "expressions occur in both liars and truth-tellers."],
        assessment="Smiles are frequent, social and context-driven; they are not a diagnostic of anything but a smile.",
        refs=["burgoon2018"]),
    dict(
        id="emotion_mapping", cue="A facial configuration shows the emotion behind it",
        onset_measure="Category hypotheses derived from AU evidence (Bench). They are labelled as hypotheses.",
        claim="Reading the AU pattern reveals what the person feels.",
        status="mixed",
        known=["Burgoon (2018) summarises a study in which most participants who reported a given emotion did not show the "
               "matching action units, and none of the participants who reported surprise showed the prototypical display.",
               "Barrett et al. (2019) review this at length (not opened in this build)."],
        assessment="ONSET's category labels are conveniences for comparing with annotated data sets, not statements about feelings.",
        refs=["burgoon2018", "barrett2019"]),
    dict(
        id="blink", cue="Blink rate and pattern",
        onset_measure="Blink events and rate per minute, exposure-corrected (Cue Lab).",
        claim="Lying changes how often a person blinks.",
        status="hypothesis",
        known=["No source on blinking and deception was opened for this build."],
        assessment="Left as an open hypothesis to be read up and, if pursued, pre-registered. ONSET measures blinks mainly "
                "because they are the largest confound for the eye region.",
        refs=[]),
    dict(
        id="gaze", cue="Gaze aversion",
        onset_measure="Not measured (only a conjugate-gaze shift of the lids is modelled as a nuisance).",
        claim="Liars look away.",
        status="unsupported",
        known=["Widely held as a folk belief and reviewed in the deception literature (see Vrij et al. 2019 and DePaulo et al. "
               "2003 - neither was opened in this build, so no figure is quoted here)."],
        assessment="Not offered. Treat any tool that scores gaze aversion as evidence of lying with suspicion until you have read the reviews.",
        refs=["depaulo2003", "vrij2019"]),
    dict(
        id="talk_motion", cue="Speech, chewing, laughter, nodding, posture shifts",
        onset_measure="Flagged and excluded: mouth-movement windows, head excursion, sustained displacement.",
        claim="(None - these are the reasons naive movement counts differ between two stretches of video.)",
        status="confound",
        known=["On the synthetic phantoms used to calibrate ONSET, ordinary blinks, speech and head motion produce far "
               "larger movement signals than any scripted micro-movement. The instrument labels them rather than hiding them."],
        assessment="Any comparison between segments must be balanced for them or restricted to times when they are absent.",
        refs=[]),
    dict(
        id="training", cue="Micro-expression recognition training",
        onset_measure="Not applicable.",
        claim="Training people (or models) to recognise micro-expressions improves lie detection.",
        status="unsupported",
        known=["Recognising posed expressions in images is a different skill from telling lies from truths in real "
               "behaviour; no source showing a lie-detection benefit was opened for this build."],
        assessment="Do not infer lie-detection value from improved expression-recognition scores, human or machine.",
        refs=[]),
]

BOTTOM_LINE = ("Among the sources opened during this build, none established a facial cue that identifies deception for an "
               "individual. Population-level differences, where reported, are small, context-dependent, or still hypotheses. "
               "The responsible output of this module is therefore a measurement and a test, never a verdict.")

REGULATION: list[dict] = [
    dict(topic="EU AI Act - prohibited practice",
         text="Article 5(1)(f) prohibits AI systems that infer the emotions of a person in the workplace or in education "
              "institutions, except for medical or safety reasons. The prohibitions have applied since February 2025.",
         opened=False, refs=["euai"]),
    dict(topic="EU AI Act - high-risk uses",
         text="Annex III lists emotion-recognition systems and polygraph-like tools (law enforcement; migration, asylum and "
              "border control) among high-risk uses, with heavy obligations. Check the exact points and any amendments.",
         opened=False, refs=["euai"]),
    dict(topic="Research use",
         text="Systems developed and used solely for scientific research are treated differently from deployed products, "
              "but a demo that looks like a product invites product use. ONSET therefore refuses to output verdicts.",
         opened=False, refs=["euai"]),
    dict(topic="Consent and personal data",
         text="A face video is personal data. Collect it only with informed, revocable consent for a stated purpose, keep it "
              "only as long as needed, and get your institution's ethics approval before recording other people. "
              "(Check your national law, for example India's Digital Personal Data Protection Act, 2023, and your institution's rules.)",
         opened=False, refs=[]),
]

PROTOCOL: list[dict] = [
    dict(id="question", text="State one hypothesis per cue before recording (for example: 'blink rate differs between the "
                             "baseline and the target segment'), and the smallest difference you would care about."),
    dict(id="groundtruth", text="Design a procedure that creates known ground truth (for example a staged-scenario paradigm "
                                "with verified statements), and say how closely its stakes resemble the situation you care about."),
    dict(id="baseline", text="Record a within-person baseline in the same room, lighting and posture, with the same task demands."),
    dict(id="counterbalance", text="Counterbalance the order of segments and questions across participants; randomise who "
                                   "is asked what, so segment effects are not time or fatigue effects."),
    dict(id="power", text="Plan the number of participants and events beforehand. A handful of events per segment cannot "
                         "reach significance (the audit shows the exact minimum)."),
    dict(id="confounds", text="Annotate or detect speech, blinks and head motion, and analyse them as covariates or exclude "
                             "them symmetrically in both segments."),
    dict(id="preregister", text="Pre-register the cues, the test, the exclusion rules and the correction for multiple "
                              "comparisons. Everything else is exploratory and must be labelled so."),
    dict(id="independence", text="Evaluate any predictive model on participants it has never seen (subject-independent), "
                               "never on other clips of the same person."),
    dict(id="baserate", text="Report positive and negative predictive values at the base rates of the intended use, not only "
                           "accuracy on a balanced sample."),
    dict(id="intervals", text="Report effect sizes with confidence intervals, including for results that did not reach "
                            "significance."),
    dict(id="ethics", text="Obtain ethics approval and informed consent, including consent to the specific analysis; give "
                         "participants the right to withdraw their video."),
    dict(id="use", text="Decide in advance what the result may and may not be used for. Individual decisions about people "
                      "are outside what this evidence can support."),
]

FORBIDDEN_KEYS = ("lie", "lies", "lying", "liar", "deceptive", "deception_score", "deception_prob", "truthful",
                  "truth_score", "truth_probability", "lie_probability", "verdict", "is_lying")


def assert_no_verdict(obj, path: str = "") -> None:
    """Raise ValueError when any dict key anywhere in ``obj`` looks like a verdict.  Used on every API result."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_KEYS:
                raise ValueError(f"verdict-shaped key '{k}' at {path or '/'}: ONSET does not output deception labels")
            assert_no_verdict(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_verdict(v, f"{path}[{i}]")


def evidence_payload() -> dict:
    """Everything the interface shows on the 'Can a face reveal a lie?' tab."""
    rows = []
    for e in CUE_EVIDENCE:
        row = dict(e)
        row["status_text"] = STATUS_TEXT[e["status"]]
        row["ref_details"] = [dict(id=r, **REFS[r]) for r in e["refs"]]
        rows.append(row)
    return dict(position=POSITION, evidence=rows, bottom_line=BOTTOM_LINE, status_text=STATUS_TEXT,
                regulation=[dict(r, ref_details=[dict(id=i, **REFS[i]) for i in r["refs"]]) for r in REGULATION],
                protocol=PROTOCOL, refs=REFS)


# ======================================================================================================
# 2. Base rates and effect sizes
# ======================================================================================================
def predictive_values(sensitivity: float, specificity: float, prevalence: float) -> dict:
    """What a positive / negative call means, given how often the target condition occurs."""
    sens, spec, p = float(sensitivity), float(specificity), float(prevalence)
    for v in (sens, spec, p):
        if not 0.0 <= v <= 1.0:
            raise ValueError("sensitivity, specificity and prevalence must lie in [0, 1]")
    tp, fn, fp, tn = sens * p, (1 - sens) * p, (1 - spec) * (1 - p), spec * (1 - p)
    ppv = tp / (tp + fp) if tp + fp > 0 else float("nan")
    npv = tn / (tn + fn) if tn + fn > 0 else float("nan")
    return dict(sensitivity=sens, specificity=spec, prevalence=p, ppv=ppv, npv=npv,
                false_discovery_rate=1 - ppv if ppv == ppv else float("nan"),
                lr_positive=sens / (1 - spec) if spec < 1 else float("inf"),
                lr_negative=(1 - sens) / spec if spec > 0 else float("inf"),
                per_1000=dict(true_positive=1000 * tp, false_negative=1000 * fn, false_positive=1000 * fp,
                              true_negative=1000 * tn),
                accuracy=sens * p + spec * (1 - p))


def base_rate_curve(sensitivity: float, specificity: float, n: int = 101) -> dict:
    prev = np.linspace(0.0, 1.0, n)
    ppv = np.full(n, np.nan)
    npv = np.full(n, np.nan)
    for i, p in enumerate(prev):
        r = predictive_values(sensitivity, specificity, float(p))
        ppv[i], npv[i] = r["ppv"], r["npv"]
    return dict(prevalence=prev.tolist(), ppv=[None if x != x else float(x) for x in ppv],
                npv=[None if x != x else float(x) for x in npv])


def auc_from_d(d: float) -> float:
    """Equal-variance binormal model: AUC = Phi(d / sqrt 2) for a standardised mean difference d."""
    return float(norm.cdf(d / math.sqrt(2.0)))


def d_from_auc(auc: float) -> float:
    return float(norm.ppf(auc) * math.sqrt(2.0))


def binormal_roc(d: float, n: int = 201) -> dict:
    """ROC of a cue whose value is N(0, 1) for the truthful and N(d, 1) for the deceptive (a model, not data)."""
    c = np.linspace(-4.0, d + 4.0, n)
    fpr = 1 - norm.cdf(c)
    tpr = 1 - norm.cdf(c - d)
    order = np.argsort(fpr)
    return dict(fpr=fpr[order].tolist(), tpr=tpr[order].tolist(), auc=auc_from_d(d), d=float(d))


def best_operating_point(d: float, prevalence: float = 0.5) -> dict:
    """Threshold that maximises overall accuracy at the stated prevalence, and what it implies."""
    c = np.linspace(-4.0, d + 4.0, 2001)
    sens, fpr = 1 - norm.cdf(c - d), 1 - norm.cdf(c)
    acc = prevalence * sens + (1 - prevalence) * (1 - fpr)
    i = int(np.argmax(acc))
    out = predictive_values(float(sens[i]), float(1 - fpr[i]), prevalence)
    out.update(threshold=float(c[i]), d=float(d))
    return out


def d_needed_for_balanced_accuracy(accuracy: float) -> float:
    """Standardised mean difference at which a single cue reaches ``accuracy`` on a 50/50 sample (equal-variance model)."""
    return float(2.0 * norm.ppf(accuracy))


# ======================================================================================================
# 3. Small-count statistics
# ======================================================================================================
def poisson_rate_ci(k: int, seconds: float, alpha: float = 0.05) -> dict:
    """Exact (Garwood) interval for an event rate, per minute."""
    if seconds <= 0:
        return dict(k=int(k), minutes=0.0, rate_per_min=float("nan"), lo=float("nan"), hi=float("nan"))
    lo = 0.0 if k == 0 else stats.chi2.ppf(alpha / 2, 2 * k) / 2.0
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2.0
    m = seconds / 60.0
    return dict(k=int(k), minutes=float(m), rate_per_min=float(k / m), lo=float(lo / m), hi=float(hi / m))


def rate_ratio_test(k1: int, s1: float, k2: int, s2: float, alpha: float = 0.05) -> dict:
    """Exact conditional test of equal Poisson rates in two stretches of observation time ``s1`` and ``s2`` seconds:
    given the total, ``k1`` is Binomial(k1 + k2, s1 / (s1 + s2)) under the null.  Returns the rate ratio
    (segment 1 over segment 2), its exact interval and the two-sided p-value."""
    n = int(k1 + k2)
    if s1 <= 0 or s2 <= 0:
        return dict(k1=int(k1), k2=int(k2), rr=float("nan"), lo=float("nan"), hi=float("nan"), p=1.0, n=n,
                    note="a segment has no observable time")
    if n == 0:
        return dict(k1=0, k2=0, rr=float("nan"), lo=0.0, hi=float("inf"), p=1.0, n=0, note="no events in either segment")
    theta = s1 / (s1 + s2)
    bt = stats.binomtest(int(k1), n, theta)
    ci = bt.proportion_ci(confidence_level=1 - alpha, method="exact")

    def to_rr(q: float) -> float:
        if q >= 1.0:
            return float("inf")
        if q <= 0.0:
            return 0.0
        return float(q / (1 - q) * (s2 / s1))

    rr = float("nan") if k2 == 0 and k1 == 0 else (float("inf") if k2 == 0 else float(k1 / s1) / float(k2 / s2))
    return dict(k1=int(k1), k2=int(k2), rr=rr, lo=to_rr(ci.low), hi=to_rr(ci.high), p=float(bt.pvalue), n=n, note="")


def min_events_for_significance(s1: float, s2: float, alpha: float = 0.05) -> int:
    """Smallest total event count for which the exact test *could* reach p < alpha (all events in one segment)."""
    theta = s1 / (s1 + s2)
    for n in range(1, 200):
        p = min(stats.binomtest(n, n, theta).pvalue, stats.binomtest(0, n, theta).pvalue)
        if p < alpha:
            return n
    return 200


def holm_adjust(p: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control), in the original order."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def circular_shift_test(x: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, valid: Optional[np.ndarray] = None,
                        n_perm: int = 4000, min_shift: int = 10, seed: int = 0) -> dict:
    """Difference of the mean of a time series between two fixed segments, tested against the same statistic after
    circularly shifting the whole series (which keeps its autocorrelation).  ``valid`` marks usable frames and is
    shifted with the series.  Returns the observed difference, a standardised effect (difference over the series'
    robust spread) and a two-sided permutation p-value."""
    x = np.asarray(x, float)
    n = len(x)
    valid = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
    ma, mb = np.asarray(mask_a, bool), np.asarray(mask_b, bool)

    def stat(xx, vv):
        a, b = xx[ma & vv], xx[mb & vv]
        if len(a) < 3 or len(b) < 3:
            return float("nan")
        return float(a.mean() - b.mean())

    obs = stat(x, valid)
    if obs != obs:
        return dict(diff=float("nan"), effect=float("nan"), p=1.0, n_perm=0, note="too few usable frames in a segment")
    if n < 4 * min_shift:
        return dict(diff=obs, effect=float("nan"), p=1.0, n_perm=0, note="series too short for a shift test")
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, n - min_shift, size=n_perm)
    null = np.array([stat(np.roll(x, int(s)), np.roll(valid, int(s))) for s in shifts])
    null = null[np.isfinite(null)]
    p = float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (len(null) + 1)) if len(null) else 1.0
    spread = 1.4826 * float(np.median(np.abs(x[valid] - np.median(x[valid])))) + 1e-9
    return dict(diff=obs, effect=float(obs / spread), p=p, n_perm=int(len(null)), note="")


def forking_paths(n_tests: int, alpha: float = 0.05) -> float:
    """Probability of at least one 'significant' result among ``n_tests`` independent tests when nothing is going on."""
    return float(1.0 - (1.0 - alpha) ** n_tests)


@lru_cache(maxsize=65536)
def _p_exact(k1: int, n: int, theta: float) -> float:
    """Two-sided exact p-value of k1 successes in n trials at success probability theta (cached: simulations repeat the
    same few (k1, n) pairs thousands of times)."""
    if n == 0:
        return 1.0
    return float(stats.binomtest(int(k1), int(n), float(theta)).pvalue)


def simulate_null_comparisons(rate_per_min: float, seconds_a: float, seconds_b: float, n_tests: int, n_sims: int = 2000,
                              alpha: float = 0.05, seed: int = 0) -> dict:
    """A person who behaves identically in two segments: how often does a family of ``n_tests`` independent exact
    tests (one per cue) report at least one p < alpha, uncorrected and with Holm correction?"""
    rng = np.random.default_rng(seed)
    any_raw = any_holm = 0
    for _ in range(n_sims):
        ps = []
        for _t in range(n_tests):
            k1 = rng.poisson(rate_per_min * seconds_a / 60.0)
            k2 = rng.poisson(rate_per_min * seconds_b / 60.0)
            ps.append(_p_exact(int(k1), int(k1 + k2), round(seconds_a / (seconds_a + seconds_b), 9)))
        any_raw += int(min(ps) < alpha)
        any_holm += int(min(holm_adjust(ps)) < alpha)
    return dict(p_any_uncorrected=any_raw / n_sims, p_any_holm=any_holm / n_sims, n_tests=n_tests, n_sims=n_sims,
                analytic_independent_continuous=forking_paths(n_tests, alpha),
                rate_per_min=rate_per_min, seconds_a=seconds_a, seconds_b=seconds_b)


# ======================================================================================================
# 4. Segment study: descriptive within-person cues
# ======================================================================================================
COUNTED_STATUS = ("micro", "expression", "unexplained")


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _event_group(event: dict) -> str:
    """Facial region with the largest share of the event's measured movement (from the per-ROI pattern)."""
    pat = event.get("pattern") or []
    energy = {g: 0.0 for g in G.GROUP_ORDER}
    for r in pat:
        if not r.get("used", True):
            continue
        g = G.GROUP_OF.get(r["roi"])
        if g:
            energy[g] += r["dx"] ** 2 + r["dy"] ** 2
    if not any(energy.values()):
        return "unknown"
    return max(energy, key=energy.get)


def _windows_seconds(windows, t0: float, t1: float) -> float:
    return float(sum(_overlap(float(w[0]), float(w[1]), t0, t1) for w in windows))


def _merge(windows) -> list:
    out = []
    for a, b in sorted((float(w[0]), float(w[1])) for w in windows):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def segment_cues(result: dict, segments: Sequence[dict]) -> list[dict]:
    """Descriptive behavioural measurements for stretches of one analysed recording.

    ``segments`` = [{label, t0, t1}] in seconds.  Rates use the time each region was *observable* as the
    denominator: the eye region is not observable while blinking, the mouth region not while talking."""
    tl = result["timeline"]
    t = np.asarray(tl["t"], float)
    blink_w = _merge(result["nuisance"]["blink_windows"])
    speech_w = _merge(result["nuisance"]["speech_windows"])
    face = np.asarray(tl.get("face_found", np.ones_like(t)), float) > 0.5
    events = result["events"]
    out = []
    for seg in segments:
        t0, t1 = float(seg["t0"]), float(seg["t1"])
        if t1 <= t0:
            raise ValueError(f"segment '{seg.get('label')}' ends before it starts")
        dur = t1 - t0
        in_seg = (t >= t0) & (t <= t1)
        face_frac = float(face[in_seg].mean()) if in_seg.any() else 0.0
        blink_s, speech_s = _windows_seconds(blink_w, t0, t1), _windows_seconds(speech_w, t0, t1)
        observable = {g: dur * face_frac for g in G.GROUP_ORDER}
        observable["eye"] = max(0.0, observable["eye"] - blink_s)
        observable["mouth"] = max(0.0, observable["mouth"] - speech_s)
        counted = [e for e in events if e["status"] in COUNTED_STATUS and t0 <= e["t_apex"] <= t1]
        by_group = {g: 0 for g in G.GROUP_ORDER}
        by_group["unknown"] = 0
        for e in counted:
            by_group[_event_group(e)] += 1
        n_blink = sum(1 for w in blink_w if t0 <= 0.5 * (w[0] + w[1]) <= t1)
        activity = {}
        valid = in_seg & face
        for g in G.GROUP_ORDER:
            a = np.asarray(tl["groups"][g], float)
            ok = valid.copy()
            if g == "eye":
                ok &= np.asarray(tl["blink"], float) < 0.3
            if g == "mouth":
                ok &= np.asarray(tl["speech"], float) < 0.3
            vals = a[ok]
            activity[g] = dict(mean=float(vals.mean()) if len(vals) else float("nan"),
                               p90=float(np.percentile(vals, 90)) if len(vals) else float("nan"), n_frames=int(len(vals)))
        out.append(dict(
            label=str(seg.get("label", "segment")), t0=t0, t1=t1, seconds=dur, face_found_fraction=face_frac,
            observable_seconds=observable, blink_seconds=blink_s, speech_seconds=speech_s,
            speech_fraction=speech_s / dur, n_events=dict(all=len(counted), **by_group),
            n_micro=int(sum(1 for e in counted if e["status"] == "micro")),
            n_blinks=int(n_blink), activity=activity,
            event_ids=[e["id"] for e in counted],
            rate_all=poisson_rate_ci(len(counted), dur * face_frac),
            rate_blink=poisson_rate_ci(n_blink, dur * face_frac)))
    return out


def compare_segments(result: dict, seg_a: dict, seg_b: dict, alpha: float = 0.05, n_perm: int = 4000, seed: int = 0) -> dict:
    """Pre-specified family of comparisons between two segments of one recording.  Counts are compared with exact
    rate tests, movement activity with a circular-shift permutation test; the whole family is Holm-corrected.
    The output says whether behaviour *differs*; it contains no statement about truth or deception."""
    a, b = segment_cues(result, [seg_a, seg_b])
    tl = result["timeline"]
    t = np.asarray(tl["t"], float)
    face = np.asarray(tl.get("face_found", np.ones_like(t)), float) > 0.5
    ma, mb = (t >= a["t0"]) & (t <= a["t1"]), (t >= b["t0"]) & (t <= b["t1"])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.04
    tests = []

    def add_rate(name: str, ka: int, sa: float, kb: int, sb: float):
        r = rate_ratio_test(ka, sa, kb, sb, alpha)
        tests.append(dict(name=name, kind="rate", a=dict(k=ka, seconds=sa), b=dict(k=kb, seconds=sb), rate_ratio=r["rr"],
                          rr_lo=r["lo"], rr_hi=r["hi"], p=r["p"], note=r["note"]))

    add_rate("Movement events (all regions)", a["n_events"]["all"], a["seconds"] * a["face_found_fraction"],
             b["n_events"]["all"], b["seconds"] * b["face_found_fraction"])
    for g in G.GROUP_ORDER:
        add_rate(f"Movement events, {g} region", a["n_events"][g], a["observable_seconds"][g], b["n_events"][g],
                 b["observable_seconds"][g])
    add_rate("Blinks", a["n_blinks"], a["seconds"] * a["face_found_fraction"], b["n_blinks"], b["seconds"] * b["face_found_fraction"])
    for g in G.GROUP_ORDER:
        x = np.asarray(tl["groups"][g], float)
        ok = face.copy()
        if g == "eye":
            ok &= np.asarray(tl["blink"], float) < 0.3
        if g == "mouth":
            ok &= np.asarray(tl["speech"], float) < 0.3
        r = circular_shift_test(x, ma, mb, ok, n_perm=n_perm, min_shift=max(10, int(round(1.0 / max(dt, 1e-3)))), seed=seed)
        tests.append(dict(name=f"Movement activity, {g} region", kind="activity", a=dict(mean=a["activity"][g]["mean"]),
                          b=dict(mean=b["activity"][g]["mean"]), diff=r["diff"], effect=r["effect"], p=r["p"], note=r["note"]))
    adj = holm_adjust([x["p"] for x in tests])
    for x, pa in zip(tests, adj):
        x["p_holm"] = pa
        x["survives"] = bool(pa < alpha)
    total_events = a["n_events"]["all"] + b["n_events"]["all"]
    need = min_events_for_significance(a["seconds"], b["seconds"], alpha)
    return dict(
        a=a, b=b, tests=tests, family_size=len(tests), alpha=alpha,
        min_events_needed=need, total_events=total_events,
        summary=_comparison_summary(tests, total_events, need),
        statement=("Differences are descriptive: they say the measured movement differed between the two stretches. They do "
                   "not say why, and they say nothing about truthfulness."))


def _comparison_summary(tests: list, total_events: int, need: int) -> str:
    surv = [t["name"] for t in tests if t["survives"]]
    if surv:
        return (f"{len(surv)} of {len(tests)} comparisons remain below the threshold after correcting for the number of "
                f"comparisons ({'; '.join(surv)}). Treat them as leads for a pre-registered follow-up, not findings.")
    if total_events < need:
        return (f"No difference can be shown: with {total_events} events in total, even a perfectly one-sided split could "
                f"not reach significance (at least {need} are needed for segments of these lengths).")
    return (f"None of the {len(tests)} comparisons remains significant after correcting for the number of comparisons. "
            "That is not evidence of no difference - the recording may simply be too short.")


# ======================================================================================================
# 5. What may be claimed from this analysis
# ======================================================================================================
def audit_claims(result: dict, comparison: Optional[dict] = None) -> list[dict]:
    """For the data at hand, which kinds of sentence could be defended?  status: supported | conditional | not_supported."""
    q = result.get("quality", {})
    grade = q.get("grade", "?")
    events = result.get("events", [])
    n_micro = sum(1 for e in events if e["status"] == "micro")
    n_expr = sum(1 for e in events if e["status"] in ("micro", "expression", "unexplained"))
    cal = result.get("engine", {}).get("calibrated", False)
    cards = []

    cards.append(dict(
        id="lied", claim="'This person was lying / telling the truth.'", status="not_supported",
        why=["No facial cue with validated individual-level accuracy was found (see the evidence table).",
             "There is no ground truth in this recording, and no base rate for the situation.",
             "Movement counts and AU patterns have many non-deceptive causes (speech, blinks, thinking, camera, culture)."],
        would_need=["A study design with verified ground truth, pre-registered cues and subject-independent evaluation.",
                    "Predictive values at the real base rate - and even then, no basis for decisions about individuals."]))
    cards.append(dict(
        id="felt", claim="'The person felt <emotion>.'", status="not_supported",
        why=["Category labels are hypotheses derived from AU evidence with an expert prior, not a trained recogniser.",
             "Felt emotion and facial expression do not map one-to-one (Burgoon 2018; Barrett et al. 2019, not opened)."],
        would_need=["Self-report or other independent measures of the emotion, and a model evaluated against them."]))
    cond = []
    if grade in ("D", "E"):
        cond.append(f"Capture quality is grade {grade}: small movements may be invisible to this clip.")
    if not cal:
        cond.append("The AU engine used default (uncalibrated) settings; its probabilities are indicative only.")
    cards.append(dict(
        id="count", claim=f"'The recording contains {n_micro} brief facial movement(s) (shorter than 0.5 s).'",
        status="conditional" if grade in ("A", "B", "C") else "not_supported",
        why=["The count is of detections that passed the spotting threshold; the detector's false-alarm rate on movement-free "
             "footage and its sensitivity at this clip's noise level are reported with the analysis.",
             f"{n_expr - n_micro} further movement(s) were longer, unexplained or sustained and are not counted as brief."] + cond,
        would_need=["Annotated real footage to measure false alarms and misses for this camera and lighting."]))
    if comparison is not None:
        surv = [t for t in comparison["tests"] if t["survives"]]
        status = "conditional" if surv else "not_supported"
        why = [comparison["summary"],
               f"The family contained {comparison['family_size']} comparisons; the p-values shown are Holm-corrected."]
        if comparison["a"]["speech_fraction"] > 0.2 or comparison["b"]["speech_fraction"] > 0.2:
            why.append("Speech occupied a large part of at least one segment: mouth-region comparisons are confounded.")
        cards.append(dict(
            id="differs", claim="'Behaviour differed between the two segments.'", status=status, why=why,
            would_need=["Counterbalanced segments, a pre-registered cue and test, and enough events (see the power note)."]))
    return cards


# ======================================================================================================
def cue_lab_demo_null(seed: int = 0) -> dict:
    """A ready-made demonstration for the interface: one person, identical behaviour in both segments, nine cues."""
    return dict(
        sim=simulate_null_comparisons(rate_per_min=6.0, seconds_a=60.0, seconds_b=60.0, n_tests=9, n_sims=1500, seed=seed),
        caption=("The person behaves the same in both minutes. Testing nine cues one by one still turns up a 'significant' "
                 "difference some of the time; correcting for the number of tests removes most of them."))
