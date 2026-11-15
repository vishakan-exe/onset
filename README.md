# ONSET

**An instrument for facial micro-movements.**

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Runs locally](https://img.shields.io/badge/runs-locally%20only-informational)
![No GPU needed](https://img.shields.io/badge/GPU-not%20required-lightgrey)
![Status: research](https://img.shields.io/badge/status-research%20instrument-orange)

Record a short clip with your webcam, or choose a video file. ONSET analyses the clip as a whole, on your own computer, and writes a report: **what** moved, **where** on the face, **when**, **how far**, **how sure the instrument is**, and the **evidence behind each statement**.

It is built to be questioned. Every claim can be checked against a synthetic face whose movements were scripted, so the right answer is known.

> **ONSET is a research instrument. It is not a lie detector and it does not read emotions.** It measures small skin movements, matches them against action-unit (FACS) patterns, and says so when the pattern is unclear.

> **What the numbers mean.** Every accuracy-like figure in this repository was measured on *synthetic* faces ("phantoms") with scripted movements. That calibrates the instrument; it says nothing about how well it works on people. Public micro-expression datasets require a signed licence from their owners and cannot be redistributed, so none was used. The learning harness (`onset.learn`) evaluates models on such data in one command once you have it. See [docs/EVALUATION.md](docs/EVALUATION.md).

<!-- Add screenshots to docs/img/ and show them here, e.g.
![The Bench](docs/img/bench.png)
-->

---

## Contents

- [Highlights](#highlights)
- [Quick start](#quick-start)
- [The six stations](#the-six-stations)
- [How an analysis works](#how-an-analysis-works)
- [What has been measured](#what-has-been-measured)
- [Learning harness](#learning-harness)
- [Rebuilding the shipped data](#rebuilding-the-shipped-data)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Privacy](#privacy)
- [Known limits](#known-limits)
- [Ethics and intended use](#ethics-and-intended-use)
- [Credits and licences](#credits-and-licences)

---

## Highlights

- **Record, then analyse.** Capture up to 12 seconds with a webcam, or upload a video (2–30 s, up to 256 MB). The whole clip is analysed and a report is produced.
- **Explainable by design.** Regions that moved, movement over time, per-claim evidence, and a *deletion test* that shows what each conclusion rests on.
- **Honest about uncertainty.** Action units carry 90 % intervals; category hypotheses come with an explicit *unclear* option and conformal sets; the report states the smallest movement *that particular recording* could have shown.
- **Works without footage.** A specimen cabinet of 30 synthetic faces with exact scripted truth lets you learn what the instrument can and cannot see.
- **Research-grade evaluation tooling.** Subject-independent protocols, a leakage audit, imbalance-aware metrics, calibrated probabilities and negative controls.
- **Local and private.** Runs on `127.0.0.1` only. No cloud, no downloads at run time, no GPU.
- **A critically examined cue module.** The Cue Lab compares two stretches of one recording and never outputs a verdict about honesty.

---

## Quick start

**Requirements:** Python 3.11 or newer (developed on 3.13), a modern browser (Chrome, Edge or Firefox), and optionally a webcam.

```bash
git clone https://github.com/YOUR-USERNAME/onset.git
cd onset

python -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m onset
```

Then open **http://127.0.0.1:8000** and allow camera access when the browser asks. A browser treats `http://127.0.0.1` as a secure context, so the permission prompt works.

Nothing is downloaded at run time: the face models, fonts and specimen library ship in the repository. The specimen library is about 65 MB. `ffmpeg` is needed only to rebuild the specimen videos, not to run the instrument or analyse a clip.

On first launch a short guided *first look* takes a specimen from nothing to a finding.

---

## The six stations

| Station | What it is for |
|---|---|
| **Capture** | Record up to 12 s with the webcam, or analyse a video file. A pre-flight check looks at the face, light, sharpness and frame rate first and says what to fix. Consent is required before anything is analysed. |
| **Bench** | The report for one recording: a *Time Lens* of movement over time, an anatomical *plate* of the face with the regions that moved, a chronophotographic strip, optional motion magnification, action units with 90 % intervals, category hypotheses (with abstention), an *evidence ledger*, and the recording's own limits. |
| **Cabinet** | 30 synthetic specimens with exact scripted truth: first looks, single movements, look-alikes, things that are *not* expressions (blinks, speech, head motion, gaze), clips where nothing happened, movements too long to be micro, and one movement filmed with five cameras. Watch first, mark what you see, then reveal the truth. A *stress lab* re-films a specimen with another camera, noise, blur, compression or weaker movement and analyses it. |
| **Cue Lab** | A behavioural-cue module for comparing two stretches of one recording. It reports counts with exact intervals, corrects for multiple comparisons, shows how often a person behaving identically in both stretches would still "differ", and includes a base-rate calculator and a note on the legal position (EU AI Act). It never outputs a verdict about honesty. |
| **Evidence** | What has been measured, on what, and what has not: sensitivity curves per camera, the smallest detectable movement, false alarms and where they come from, probability calibration, amplitude checks, category sets with abstention, controls, and learning-harness runs. |
| **Atlas** | The 27 facial regions and 12 action units the instrument uses, the template for each, and which look-alike pairs the geometry can and cannot separate. |

---

## How an analysis works

1. **Find and track the face** with the BlazeFace detector and a 468-point face mesh, run through OpenCV's DNN module.
2. **Stabilise.** Every frame is warped onto an eye-aligned canonical face, so head motion does not masquerade as expression.
3. **Measure movement** with dense inverse-compositional Lucas–Kanade optical flow at four time scales (0.07, 0.13, 0.25 and 0.60 s), averaged over 27 regions in five groups (brow, eye, nose, cheek, mouth).
4. **Spot movements** with a calibrated scan statistic: regional displacement is whitened with a robust noise estimate, head-motion directions are projected out, and the result is matched against twelve action-unit templates plus blink and speech templates, so those are named rather than mistaken. Windows around blinks and speech are re-scanned with the affected regions left out.
5. **Infer action units** with a Bayesian engine over subsets of up to three units, giving an inclusion probability per unit and an amplitude with a 90 % interval.
6. **Hypothesise categories** (three label schemes) from the action-unit posterior. Uncertainty is propagated, an *unclear* option absorbs weak patterns, and split-conformal sets say which categories cannot be excluded.
7. **Explain.** An evidence ledger, a deletion test, magnified playback, a quality grade, and the smallest movement this recording could have shown.

The whole chain lives in `onset/vision/` and `onset/analysis/` and runs without a GPU and without PyTorch.

---

## What has been measured

On a corpus of 240 synthetic clips (six face identities; faces A–C used for calibration, faces D–F held out for testing), the instrument's headline numbers on the **held-out** faces are:

| Measure | Result |
|---|---|
| Recall of scripted movements (strict / loose timing) | 0.49 / 0.54 |
| Spotting precision | 0.76 |
| False alarms | 0.43 per 10 s of video |
| Action-unit F1 | 0.77 |
| 90 % amplitude interval coverage | 0.94 |

Recall depends strongly on conditions: on the whole corpus it falls from 0.85 (still head) to 0.56 (normal) to 0.26 (restless), and from 0.66 under low light to 0.36 on a phone camera. The instrument misses roughly half of the *small* scripted movements, and the report says so.

These are **instrument calibration figures on phantoms, not accuracy on human faces**. One calibration factor (the correction behind the report's "smallest detectable movement") was fitted on all six faces and is checked by refitting on each group; the groups differ by about 1.4×. Full protocol, controls and caveats are in [docs/EVALUATION.md](docs/EVALUATION.md) and [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

---

## Learning harness

`onset.learn` evaluates learned models on licensed micro-expression datasets under protocols that avoid the usual traps:

```bash
python -m onset.learn.run --data casme2=/data/CASME2 --scheme casme2_5 --protocol loso --out runs/casme2_loso
python -m onset.learn.run --data casme2=/data/CASME2 --data samm=/data/SAMM --scheme megc3 --protocol composite --out runs/megc3
python -m onset.learn.run --data casme2=/data/CASME2 --data samm=/data/SAMM --scheme megc3 --protocol lodo --out runs/cross_corpus
python -m onset.learn.report runs/casme2_loso                       # tables for the Evidence station
python -m onset.learn.run --synthetic --out runs/synthetic_check    # checks the machinery; NOT a benchmark
```

What it provides:

- Subject-independent folds (LOSO, composite, leave-one-dataset-out, k-fold) with an explicit **leakage audit**
- Normalisation, hyper-parameters and calibration fitted on training subjects only
- Class-balanced metrics (UF1, UAR) next to accuracy, **subject-level bootstrap** intervals, and paired subject permutation tests with Holm correction
- Temperature scaling and conformal prediction sets
- Negative controls: shuffled labels, an identity probe, a leakage gap
- Latency measurements and a reproducibility manifest
- Models: majority and prior baselines, the expert action-unit map, a regularised linear model, and a small graph-temporal network (PyTorch, optional)

Datasets are not included and require licences from their owners. File layouts the loaders read are described in [docs/DATASETS.md](docs/DATASETS.md).

---

## Rebuilding the shipped data

```bash
python scripts/build_phantom_eval.py --out CORPUS --n 240 --seed 1 --workers 2    # 240 phantom clips with exact truth (about an hour)
python scripts/calibrate_engine.py --corpus CORPUS --out data --cache CACHE --workers 2
python scripts/build_library.py --workers 2 --force                                # the 30 specimens, results, thumbnails, exports
```

Calibration fits the engine's few free parameters, chooses the spotting threshold (best F1 within a stated false-alarm budget) and widens the amplitude intervals until 90 % of true amplitudes fall inside, all on faces A–C, then reports results on faces D–F.

---

## Tests

```bash
python -m pytest tests -m "not slow"     # fast tests (about a minute on two cores)
python -m pytest tests                   # adds slow end-to-end tests that render phantoms and run the whole chain
```

The tests cover the statistics (conformal sets, exact intervals, permutation tests, the Cue Lab's guard against verdicts), the engine and its explanations, the learning harness (including a check that the leakage audit catches a leak), the server's privacy behaviour, and that every figure quoted in the documents is the figure in the shipped data files.

---

## Project layout

```
onset/            the package
  vision/         face tracking, stabilisation, dense flow, spotting
  analysis/       templates, action-unit engine, categories, quality, explanations, magnification, Cue Lab, atlas, pipeline
  synth/          procedural faces ("phantoms"), scripted scenarios, exact ground truth, the specimen library
  learn/          datasets, protocols, features, models, metrics, calibration, controls, run / report
  server/         local Flask app, sessions, job queue
web/              the interface (vanilla JS modules, no build step; fonts are bundled)
data/             calibration.json, specimen library, evidence exports
scripts/          corpus, calibration and library builders
tests/            unit and end-to-end tests
docs/             datasets, evaluation protocol, ethics and limits, model card, design notes
var/              run-time sessions (created on first use; safe to delete)
```

---

## Privacy

- The server listens on `127.0.0.1` only; nothing is sent anywhere.
- Recording or uploading requires an explicit consent tick.
- The recording is **deleted when the analysis ends**. If the analysis fails, the whole session is deleted. A recording that is never sent for analysis is deleted when you leave the page, or after an hour at the latest. Anything an interrupted run left behind is deleted when the server starts.
- After a successful analysis only grayscale face crops (used to draw the report) and the numbers remain. They are deleted automatically after 24 hours, or immediately with *Delete* on a report, or *Delete all my analyses now* in the footer.
- The specimens are synthetic: no real person appears in any of them.

---

## Known limits

- **No real-face accuracy is claimed.** See the note at the top.
- A consumer webcam films at about 30 frames per second. A micro-expression lasts well under half a second, so it spans only a handful of frames; events under about 100 ms cannot be resolved at that rate. Faint movements in poor light are below what any camera chain can show, and the report states how small a movement *this* recording could have shown.
- The face mesh is the 468-point model without attention refinement, so lips and eyelids are tracked less precisely than with the newest models.
- The action-unit templates are written by hand from FACS anatomy, not learned. They separate look-alike pairs only as far as the geometry allows (the Atlas shows which pairs overlap).
- Category hypotheses are a loose map from action units to labels. The instrument abstains rather than guess.
- Categories evaluated on phantoms are scripted from the same prototypes the expert map uses, so those figures check the plumbing, not the psychology.
- One person per clip; no identity, age or demographic inference of any kind.

---

## Ethics and intended use

ONSET is intended for research, teaching and exploring what facial-movement measurement can and cannot support. It must not be used to judge honesty, credibility, intent or emotional state, or to make decisions about people. See [docs/ETHICS.md](docs/ETHICS.md) for the reasoning, the limits of the cue module, and the regulatory context.

---

## Credits and licences

See [NOTICE.md](NOTICE.md). Face detector and mesh: MediaPipe legacy models (Apache-2.0), run through OpenCV. Fonts: Archivo and Source Serif 4 (SIL OFL). The code in this repository was written for this project.

> **Note on licensing:** no licence has been chosen for the project's own code yet. Add a `LICENSE` file (for example MIT or Apache-2.0) before inviting others to reuse it.
