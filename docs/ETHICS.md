# Ethics, privacy and limits

ONSET measures small facial movements. Several of the things people want to do with such measurements are ones this
project argues against, so this page says plainly what the software will and will not do, what happens to a recording,
and what a responsible user should do before pointing it at another person.

## 1. What ONSET is not

* **Not a lie detector.** No output says that a person lied, is lying, is deceptive, truthful, nervous, guilty or
  innocent. The Cue Lab, the module that addresses deception, exists to show *why* such a verdict cannot be read off
  facial movement, and to let a researcher test a stated hypothesis properly. A guard in the code
  (`onset/analysis/cue_lab.py: assert_no_verdict`) refuses to emit keys such as `lie`, `liar`, `deceptive`,
  `deception_score` or `truthful`; a test covers it.
* **Not an emotion reader.** The category hypotheses (happiness, disgust, surprise, …) are a loose map from action-unit
  patterns to the labels used by annotated datasets. They are conveniences for comparing with those datasets, not
  statements about what anyone feels. The report says "hypothesis", shows the probability that the pattern is *unclear*,
  and abstains when the pattern is weak.
* **Not an identity, age, gender, ethnicity or health tool.** Nothing in the pipeline infers or stores any of these. The
  face mesh is used for geometry only.
* **Not validated on real people.** Every accuracy-like number shipped with ONSET was measured on synthetic faces
  ("phantoms") whose movements were scripted. They calibrate the instrument. They do not tell you how well it works on
  a person, in your light, with your camera. See `EVALUATION.md`.
* **Not for decisions about people.** Hiring, admissions, policing, border control, insurance, credit, schooling,
  custody, parole, or any other decision that affects a person is outside what this evidence can support.

## 2. The deception module, and why it is built the way it is

The brief asked for a deception component. Building one that outputs "truthful / deceptive" would be building something
the evidence does not support and that causes harm when believed. The Cue Lab is instead a **research-oriented
behavioural-cue analysis module** with these properties:

* It compares **two stretches of one recording** (a baseline and a target) on pre-stated cues, never a person against a
  population and never one stretch against "truth".
* It counts events and reports **rates with exact (Garwood) intervals**, corrected for the time the face was actually
  observable. Rates in two stretches are compared with an **exact conditional test**; movement activity with a
  **circular-shift permutation test**; the whole family of comparisons is **Holm-corrected**, and the number of
  comparisons is displayed.
* It shows, by simulation, **how often a person who behaves identically in both stretches would still "differ"** across
  the family of tests, with and without the correction (the "forking paths" panel).
* It reports the **smallest number of events that could ever reach significance** for the stretch lengths given, so a
  clip with three events cannot be dressed up as a finding.
* It contains a **base-rate calculator** (predictive values at a chosen prevalence) and a binormal ROC view that turns an
  accuracy claim into the separation it would require, with the point that accuracy on a balanced sample says nothing
  about use where liars are rare.
* It carries an **evidence table** with a status for each cue (validated limit, mixed, hypothesis, unsupported,
  confound), a protocol checklist for a proper study (pre-registration, ground truth, baseline, counterbalancing,
  power, independence, ethics approval), and a note on regulation.

### What was actually read

The evidence table cites ten sources. **Only two were opened during the build**: Bond & DePaulo (2006), the
meta-analysis of human lie-detection accuracy (54 % of judgments correct), and Burgoon (2018), an opinion article on
micro-expressions and deception. The figures taken from Porter & ten Brinke (2008) are quoted *as reported by Burgoon*
and are marked as such. The rest (DePaulo et al. 2003, Hartwig & Bond 2011, Vrij et al. 2019, Barrett et al. 2019,
Yan et al. 2013, the National Research Council 2003 polygraph review, the EU AI Act) are listed as "not opened in this
build" with the reason to read them, and no figure from them is quoted. The bottom line the module states is
deliberately modest: *among the sources opened, none established a facial cue that identifies deception for an
individual.* Before you cite anything from this page or from the Cue Lab, read the originals.

## 3. Consent, privacy and what happens to a recording

* **Local only.** The server listens on `127.0.0.1`. Nothing is sent over a network; the interface and its fonts are
  bundled, and the analysis runs on your computer.
* **Consent is required.** Recording or uploading needs an explicit tick. The tick is not a legal consent for the person
  in the video: if the person is not you, **you** must obtain informed, revocable consent for the specific analysis.
* **The recording is deleted when the analysis ends.** If the analysis fails, the whole session goes with it (footage
  and partial crops; this happens in a `finally` block). A webcam recording that is never sent for analysis is deleted
  when you leave the page, and in any case by the server after an hour; anything an interrupted run left behind is
  removed when the server starts. The server-side rules are covered by automated tests; the two browser-side ones
  (leaving the station, closing the tab) were checked by hand in a real browser with a fake camera and are not part of
  the test suite.
* **What stays**: grayscale face crops used to draw the report and the numbers. They are deleted automatically after
  24 hours, immediately with *Delete* on a report, or all at once with *Delete all my analyses now*.
* **Learning-harness caches** hold arrays of numbers only (no pixels, no landmarks).
* **Specimens are synthetic.** No person appears in any of them.
* **Bystanders and minors.** The analysis assumes one face. Do not record a child, or anyone who cannot meaningfully
  consent, without the consent of whoever is responsible for them and the approval your institution requires.

For research with participants, you also need your institution's ethics approval, a data-management plan, and a way for
participants to withdraw their video. The Cue Lab's protocol checklist lists these.

### Law

Face video is personal data in most jurisdictions. Check the law that applies to you and your institution's rules; for
example India's Digital Personal Data Protection Act, 2023, or the EU's General Data Protection Regulation. The Cue Lab
also notes, **without having opened the text**, that Article 5(1)(f) of the EU Artificial Intelligence Act
(Regulation (EU) 2024/1689) prohibits systems that infer emotions of people in workplaces and educational institutions
(with medical and safety exceptions), that the Annex III list of high-risk uses includes emotion-recognition and
polygraph-like systems in some settings, and that research-only systems are treated differently from products. These
summaries are marked unverified: read the current consolidated text and any amendments. Nothing here is legal advice.

## 4. Datasets

Public micro-expression datasets contain identifiable faces and come with licences that forbid redistribution. ONSET
ships none. When you use them: keep them where the licence allows; do not publish frames beyond what it permits; cite
the papers; report which label scheme, protocol and subset you used. See `DATASETS.md`.

## 5. Fairness and who the numbers do not cover

* **No demographic evaluation exists.** The phantoms have no skin-tone variation, facial hair, glasses, make-up, wrinkles
  or scars. Optical-flow measurement of small movements can behave differently across skin tones, lighting, camera
  noise, facial structure and age; none of that was measured here. Do not claim the instrument is fair or unbiased.
* **The datasets are not representative either**, and their emotion labels were assigned by annotators from particular
  settings; a model that agrees with them inherits their conventions.
* **Facial movement is not feeling.** The same movement means different things in different people and situations, and
  the same feeling shows in different movements. The literature the Cue Lab cites argues this at length.

## 6. Misuse and what the design does about it

Software cannot prevent misuse, but design choices can avoid inviting it:

* The words "lie", "deceptive", "truthful" and the like are never output.
* Every claim comes with its uncertainty: 90 % intervals on amplitudes, probabilities that are calibrated on phantoms,
  conformal sets that say which categories cannot be excluded, and an explicit *unclear* option. The instrument abstains.
* The report states, for the very recording analysed, **how small a movement it could have shown** (a grade from A to E).
  Absence of a finding in a poor recording is then visibly not evidence of absence.
* The deletion test and evidence ledger show what each conclusion rests on, so a reader can see when it rests on one
  region or one frame.
* The interface has no batch mode, no gallery of people, no identification and no live analysis of a camera feed: a
  clip is recorded, analysed and reported.

## 7. If you publish with it

Say that results are from synthetic faces unless you have run the learning harness on licensed data. Report subject-
independent protocols, class-balanced metrics with intervals, calibration and the negative controls. Do not describe
anything it produces as detecting lies, emotions or intent. Disclose the limits in section 5. Get the approvals in
section 3 before recording anyone.
