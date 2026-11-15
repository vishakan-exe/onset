# Model card: ONSET 1.0

Format after Mitchell et al., "Model Cards for Model Reporting" (2019). ONSET is a measurement chain, not a single model,
so the card describes the chain and says which parts are learned, which are written by hand and which are fitted.

## Summary

ONSET takes a short video of one face and reports small facial movements: when they happened, where on the face, how far
the skin moved (as a percentage of the distance between the eyes), which FACS action units (AUs) best explain the
pattern, and how sure it is. It then offers category *hypotheses* (for comparison with annotated datasets) and says when
the pattern is unclear. It does not read emotions and does not detect lies.

## Components

| Stage | What it is | Learned? | Trained / fitted on |
|---|---|---|---|
| Face detection | BlazeFace short-range (legacy MediaPipe), via OpenCV DNN | yes, by Google | not documented here; see the model's own card |
| Face mesh | 468-point face-landmark model (legacy MediaPipe, no attention refinement), via OpenCV DNN | yes, by Google | as above |
| Stabilisation | eye-aligned warp to a canonical face, landmark smoothing | no | |
| Movement measurement | dense inverse-compositional Lucas–Kanade flow at four time scales (0.07, 0.13, 0.25, 0.60 s), averaged over 27 regions (5 groups) | no | |
| Noise model | robust per-region noise estimate; head-motion directions projected out | no | |
| Spotting | template-matched scan statistic over 12 AU templates plus blink and speech templates; threshold 5.25 | **threshold fitted** | 240 phantom clips, faces A–C |
| AU engine | Bayesian inference over AU subsets (≤ 3 AUs), robust likelihood, size prior | **3 parameters fitted** (`eta` 0.3, tempering 0.5, size prior) | faces A–C |
| AU templates | 12 AUs (1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17), regional displacement patterns written from FACS anatomy | no (hand-written) | |
| Amplitude intervals | engine intervals widened by a factor (×2.44) | **1 factor fitted** | faces A–C |
| Smallest detectable movement | analytic estimate × 2.08 (half found), × ≈7.1 (nine in ten) | **2 factors fitted** | all six faces, because they only rescale a printed number; refitted on A–C alone (1.75, 5.25) and on D–F alone (2.41, 8.25) as a check |
| Category hypotheses | AU posterior → categories through a hand-written map, three label schemes, explicit *unclear* mass | no (hand-written) | |
| Category sets | split-conformal sets, α = 0.1, one threshold per scheme | **thresholds fitted** | faces A–C (circular for categories) |
| Learning harness models (optional) | majority, prior, expert, regularised linear, graph-temporal transformer | trained **by the user**, on data the user holds | nothing ships trained |

The instrument contains no neural network of its own. PyTorch is needed only for the optional graph-temporal model in
the learning harness.

## Intended use

* Exploring and teaching how small facial movements can be measured, and how far such measurements can be trusted.
* Research on micro-movement measurement, with the learning harness for subject-independent evaluation on licensed
  datasets and the Cue Lab for properly designed segment comparisons.
* Demonstration with the synthetic specimen library, which needs no footage, webcam or consent.

## Out-of-scope and prohibited uses

* Detecting deception, "truthfulness", stress, guilt, intent or any mental state. The software does not output these and
  must not be used to infer them.
* Decisions about people: employment, education, policing, border control, insurance, credit, legal proceedings,
  health or any other that affects them.
* Identification or profiling by identity, age, gender, ethnicity or health.
* Covert recording. The tool is built around a person knowing that they are being recorded.
* Footage that does not meet the conditions below, treated as if it did. The report grades the recording (A–E) and says
  how small a movement it could have shown; if the grade is poor, the absence of a finding is not evidence.

## Factors

Performance depends on, at least: frame rate (a micro-expression spans a handful of frames at 30 fps; events under about
100 ms cannot be resolved), lighting and sensor noise, resolution and compression, head motion, speech, blinks and the
size of the movement. On the synthetic corpus the recall of scripted movements fell from 0.85 (still head) to 0.56
(normal) to 0.26 (busy). Factors that were **not** evaluated at all: skin tone, facial hair, glasses, make-up, age,
facial structure, lens distortion, rolling shutter, occlusion by hand or hair.

## Metrics and evaluation data

All shipped metrics come from 240 synthetic clips with exact truth (six face identities: A–C calibration, D–F test).
They are listed in `EVALUATION.md`. Headline, on held-out faces: spotting recall 0.49 (strict timing) / 0.54 (loose), 0.43
false alarms per 10 s, precision 0.76; AU F1 0.77 with the exact AU set recovered in 0.53 of found events; ECE 0.031;
90 % amplitude intervals covering 0.94 of true amplitudes; category sets covering 0.91–0.92 at a nominal 0.90.

**There is no evaluation on real people in this release.** Real-data evaluation is supported by the learning harness and
needs a licensed dataset (see `DATASETS.md`).

## Quantitative caveats

* Phantoms are procedural and clean. Real recordings are harder in ways that were not simulated, so every figure above
  is optimistic as a statement about people.
* Spotting recall near 0.5 on a corpus where blinks, speech and head motion are deliberately frequent means the
  instrument misses about half of the small movements it is given. This is reported, not hidden.
* Median estimated amplitude is 0.82 of the true value on phantoms: a mild systematic under-read.
* AU1/AU2/AU4 combinations are only partly separable with these templates; closely similar pairs can come back as a
  mixture. The Atlas lists the look-alike pairs.
* Category hypotheses are a map from AU patterns to dataset labels. The datasets assigned labels from AUs, context and
  sometimes self-report; the same movement can carry different labels.

## Latency

Everything runs on the CPU. For the specimen clips (about 8 s at 30 fps, 480 × 480) on the two-core machine used for
the build, tracking and stabilising took about 4–5 s, spotting about 16–23 s, and measuring, inferring and explaining a
few more seconds, so roughly half a minute per clip while other work was running. Every report records its own
stage timings. A longer clip or a larger frame takes longer (clips are limited to 30 s). Model inference in the learning harness is milliseconds per clip; fitting the graph-temporal network is the slow part
(see `EVALUATION.md`).

## Ethical considerations

See `ETHICS.md`. In short: local-only processing, explicit consent, deletion of the recording when the analysis ends,
automatic purge of the derived crops after 24 hours, no verdict language, uncertainty and abstention everywhere, and an
open statement that nothing has been validated on real people or checked for demographic fairness.

## Caveats and recommendations

* Treat each output as a measurement with an error bar, not a conclusion. Look at the evidence ledger before quoting a
  label, and at the recording's grade before reading anything into a *lack* of detections.
* Record with a steady head, frontal light, at the highest frame rate the camera allows, uncompressed if possible.
* If you need to know how it performs on people, run the learning harness on licensed data under `loso` and `lodo`
  protocols, with the negative controls, and report class-balanced metrics with intervals.
* Re-fit the spotting threshold and engine parameters (`scripts/calibrate_engine.py`) if you change the camera chain
  substantially, and say that you did.
