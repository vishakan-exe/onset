# ONSET

**An instrument for facial micro-movements.** Record a short clip with your webcam (or choose a video file) and ONSET
analyses it as a whole, on your own computer, and writes a report: *what* moved, *where* on the face, *when*, *how far*,
*how sure the instrument is*, and the evidence behind each statement. It is built to be questioned: every claim can be
checked against a synthetic face whose movements were scripted, so the right answer is known.

ONSET is a research instrument. It is **not** a lie detector and it does not read emotions. It measures small skin
movements, matches them against action-unit (FACS) patterns, and says so when the pattern is unclear.

> **What the numbers mean.** Every accuracy-like figure shipped with ONSET was measured on *synthetic* faces
> ("phantoms") with scripted movements. That calibrates the instrument; it says nothing about how well it works on
> people. Public micro-expression datasets need a signed licence from their owners and cannot be redistributed, so none
> was available here. The learning harness (`onset.learn`) evaluates models on such data in one command once you have
> it. See [docs/EVALUATION.md](docs/EVALUATION.md).

---

## Run it

```bash
pip install -r requirements.txt        # Python 3.11+ (developed on 3.13)
python -m onset                        # http://127.0.0.1:8000
```

If ONSET reached you as three zip files (`ONSET-part1of3.zip` and so on, split only to stay under an upload limit), unpack all
three into the same place; each has `ONSET/` at its top, and the parts merge. Part 1 alone runs everything, with a smaller cabinet.

Nothing is downloaded at run time: the face models, fonts and the specimen library ship in the folder. The server listens
on `127.0.0.1` only. A browser treats `http://127.0.0.1` as a secure context, so the webcam permission prompt works.

`ffmpeg` is needed only to (re)build the specimen videos; it is not needed to run the instrument or to analyse a clip.

## The six stations

| Station | What it is for |
|---|---|
| **Capture** | Record up to 12 s with the webcam, or analyse a video file (2–30 s, up to 256 MB). A pre-flight check looks at the face, light, sharpness and frame rate first and says what to fix. Consent is required before anything is analysed. |
| **Bench** | The report for one recording. A *Time Lens* of movement over time, an anatomical *plate* of the face with the regions that moved, a chronophotographic strip, optional motion magnification, the action units with 90 % intervals, category hypotheses (with abstention), an *evidence ledger* that shows what each conclusion rests on and what happens when each piece is removed, and the recording's own limits. |
| **Cabinet** | 30 synthetic specimens with exact scripted truth: first looks, one movement at a time, look-alikes, things that are *not* expressions (blinks, speech, head motion, gaze), clips where nothing happened, movements too long to be micro, and one movement filmed with five cameras. Watch first, mark what you see, then reveal the truth. A *stress lab* re-films a specimen with another camera, noise, blur, compression or weaker movement and analyses it. Compare two look-alikes side by side. |
| **Cue Lab** | A critically examined behavioural-cue module for comparing two stretches of one recording. It never outputs a verdict about honesty; it reports counts with exact intervals, corrects for the number of comparisons, shows how often a person who behaves identically in both stretches would still "differ", and includes a base-rate calculator and a note on the legal position (EU AI Act). |
| **Evidence** | What has been measured, on what, and what has not: sensitivity curves per camera, the smallest detectable movement, false alarms and where they come from, calibration of the probabilities, amplitude read-out checks, category sets with abstention, controls, and the learning-harness runs. |
| **Atlas** | The 27 regions and 12 action units the instrument uses, with the template for each and which look-alike pairs the geometry can and cannot separate. |

The first time you open it, a short guided first look takes a specimen from nothing to a finding.

## How an analysis works

1. **Find and track the face** (BlazeFace detector and a 468-point face mesh, run through OpenCV's DNN module).
2. **Stabilise**: every frame is warped onto an eye-aligned canonical face, so head motion does not masquerade as expression.
3. **Measure movement** with dense inverse-compositional Lucas–Kanade flow between frame pairs at four time scales
   (0.07, 0.13, 0.25, 0.60 s), averaged over 27 regions of five groups (brow, eye, nose, cheek, mouth).
4. **Spot** movements with a calibrated scan statistic: the regional displacement is whitened with a robust noise
   estimate, head-motion directions are projected out, and it is matched against twelve action-unit templates, plus
   blink and speech templates so that those are named instead of mistaken. Windows around blinks and speech are
   re-scanned with the affected regions left out.
5. **Infer action units** with a Bayesian engine over subsets of up to three action units (a robust likelihood that
   does not let large movements drown their own evidence), giving an inclusion probability per unit and an amplitude
   with a 90 % interval.
6. **Hypothesise categories** (three label schemes) from the action-unit posterior; the uncertainty is propagated, an
   explicit *unclear* option absorbs weak patterns, and split-conformal sets say which categories cannot be excluded.
7. **Explain**: an evidence ledger and a deletion test (remove a region or an action unit and see what the posterior
   does), magnified playback, quality grade and the smallest movement this very recording could have shown.

The whole chain is in `onset/` (`vision/`, `analysis/`) and runs without a GPU and without PyTorch.

## Learning harness

`onset.learn` evaluates learned models on licensed micro-expression datasets under protocols that avoid the usual traps:

```bash
python -m onset.learn.run --data casme2=/data/CASME2 --scheme casme2_5 --protocol loso --out runs/casme2_loso
python -m onset.learn.run --data casme2=/data/CASME2 --data samm=/data/SAMM --scheme megc3 --protocol composite --out runs/megc3
python -m onset.learn.run --data casme2=/data/CASME2 --data samm=/data/SAMM --scheme megc3 --protocol lodo --out runs/cross_corpus
python -m onset.learn.report runs/casme2_loso          # tables for the Evidence station
python -m onset.learn.run --synthetic --out runs/synthetic_check   # checks the machinery; NOT a benchmark
```

Subject-independent folds with an explicit leakage audit, normalisation / hyper-parameters / calibration fitted on
training subjects only, class-balanced metrics (UF1, UAR) next to accuracy, subject-level bootstrap intervals, paired
subject permutation tests with Holm correction, temperature scaling and conformal sets, negative controls (shuffled
labels, identity probe, leakage gap), latency, and a reproducibility manifest. Models: majority and prior baselines,
the expert action-unit map, a regularised linear model and a small graph-temporal network (PyTorch, optional). Details
and the file layouts the loaders read: [docs/DATASETS.md](docs/DATASETS.md), [docs/EVALUATION.md](docs/EVALUATION.md).

## Rebuild the shipped data

```bash
python scripts/build_phantom_eval.py --out CORPUS --n 240 --seed 1 --workers 2   # 240 phantom clips with exact truth (about an hour)
python scripts/calibrate_engine.py --corpus CORPUS --out data --cache CACHE --workers 2
python scripts/build_library.py --workers 2 --force                    # the 30 specimens, results, thumbnails, exports
python -m onset.learn.run --synthetic --out runs/synthetic_check && python -m onset.learn.report runs/synthetic_check --synthetic
```

The calibration fits the engine's few free parameters (model-error factor, posterior tempering, action-unit count prior),
chooses the spotting threshold (best F1 within a stated false-alarm budget) and widens the amplitude intervals until 90 %
of true amplitudes fall inside, all on faces A–C, and reports its results on faces D–F, which none of those settings saw.
One exception, stated everywhere it matters: the two factors that correct the report's smallest-detectable-movement
statement are fitted on all six faces (they only rescale a printed number) and are checked by refitting on each group alone;
the groups differ by about 1.4× (see [docs/EVALUATION.md](docs/EVALUATION.md)).

## Tests

```bash
python -m pytest tests -m "not slow"     # unit tests (354 when this was written; about a minute on two cores)
python -m pytest tests                   # adds the 13 slow ones: phantoms are rendered and the whole chain runs on them
```

They cover the statistics (conformal sets, exact intervals, permutation tests, the Cue Lab's guard against verdicts),
the engine and explanations, the learning harness (including a check that a graph network really learns on unseen
subjects, and that the leakage audit catches a leak), the server's privacy behaviour (consent required, the recording
deleted after the analysis, also when it fails, unfinished recordings purged, only whitelisted files served), and that every
figure the documents quote is the figure in the shipped data files.
A fresh virtual environment with only `requirements.txt` passes the fast suite.

## Layout

```
onset/            the package
  vision/         face tracking, stabilisation, dense flow, spotting
  analysis/       templates, action-unit engine, categories, quality, explanations, magnification, Cue Lab, atlas, pipeline
  synth/          procedural faces ("phantoms"), scripted scenarios, exact ground truth, the specimen library
  learn/          datasets, protocols, features, models, metrics, calibration, controls, run / report
  server/         local Flask app, sessions, job queue
web/              the interface (vanilla JS modules, no build step); fonts are bundled
data/             calibration.json, specimen library, evidence exports
scripts/          corpus, calibration and library builders
tests/            unit and end-to-end tests
docs/             datasets, evaluation protocol, ethics and limits, model card, design notes
var/              run-time sessions (created on first use; safe to delete)
```

## Privacy

* The server listens on `127.0.0.1` only; nothing is sent anywhere.
* Recording or uploading requires an explicit consent tick.
* The recording itself is deleted when the analysis ends. If the analysis fails, the whole session is deleted: no
  footage and no partial crops. A recording that is never sent for analysis is deleted when you leave the page, or
  after an hour at the latest; anything an interrupted run left behind is deleted when the server starts.
* What stays after a successful analysis are grayscale face crops used to draw the report, and the numbers. They are
  deleted automatically after 24 hours, or immediately with *Delete* on a report, or *Delete all my analyses now* in
  the footer.
* The specimens are synthetic: no person appears in any of them.

## Known limits

* **No real-face accuracy is claimed.** See the note at the top.
* A consumer webcam films at about 30 frames per second. A micro-expression lasts well under half a second, so it spans
  a handful of frames; events under about 100 ms cannot be resolved at that rate. Faint movements in poor light are
  below what any camera chain can show; the report says how small a movement *this* recording could have shown.
* The face mesh is the 468-point model without the attention refinement, so lips and eyelids are tracked less precisely
  than with the newest models.
* The action-unit templates are written by hand from FACS anatomy, not learned. They separate look-alike pairs only
  as far as the geometry allows (the Atlas shows which pairs overlap).
* Category hypotheses are a loose map from action units to labels; the datasets themselves assigned their labels from
  the action units, context and sometimes the participant's report. The instrument abstains rather than guess.
* Categories evaluated on phantoms are scripted from the same prototypes the expert map uses, so those figures
  check the plumbing, not the psychology.
* Single person per clip; no identity, age or demographic inference of any kind.

## Credits and licences

See [NOTICE.md](NOTICE.md). Face detector and mesh: MediaPipe legacy models (Apache-2.0), run through OpenCV. Fonts:
Archivo and Source Serif 4 (SIL OFL). The code in this folder was written for this project.
