# Evaluation

Two different kinds of evidence are produced by this project, and they answer different questions. Do not read one as
the other.

| | **Phantom calibration** | **Learning harness** |
|---|---|---|
| Data | 240 synthetic clips, 372 scripted movements, six invented face identities | licensed datasets of real people that *you* supply |
| Truth | exact: the generator knows what moved, where, when and by how much | human annotation (onset/apex/offset, action units, emotion label) |
| Question | does the instrument measure what it should, and how small a movement can it see? | do learned models recognise annotated categories in people they were not trained on? |
| Shipped results | yes (`data/evidence/phantom_eval.json`, the Evidence station) | machinery checked on synthetic clips only; **no real-data run is included** |
| Says anything about human accuracy | **no** | yes, once run on real data, with the limits below |

## 1. Phantom calibration (`scripts/calibrate_engine.py`)

### Corpus

`scripts/build_phantom_eval.py --n 240 --seed 1` renders 240 clips from procedural faces ("phantoms"). Each clip holds
up to two scripted movements: a single action unit (about 55 % of them), a pair (35 %, including pairs the geometry
finds hard to tell apart such as AU1+AU2 or AU4+AU7) or a triple (10 %); about one movement in eight is deliberately
too long to be a micro-expression (0.7–1.4 s); about one clip in seven is drawn with nothing scripted at all (42 of the 240 came out so). Every clip also
gets blinks, speech, gaze shifts and head motion, at one of three levels (still, normal, busy), and is filmed with one
of five simulated cameras (30 fps webcam, 30 fps phone, 24 fps low-light webcam, 15 fps webcam, 100 fps high-speed).
Six face identities: **A–C calibrate, D–F test**; every setting that decides what the instrument reports was fitted on the calibration faces
only, with one exception stated below (the smallest-detectable-movement factors). Blinks and speech occur at
random times and are allowed to collide with scripted movements, as they do in real recordings.

### What is calibrated, and on which split

* The **engine's few free parameters** (model-error factor `eta`, posterior tempering, the prior on how many action units
  act together) are chosen by grid search on the calibration faces by log-loss of the action-unit posterior.
* The **spotting threshold** is the one with the best spotting F1 on the calibration faces *among thresholds whose
  false-alarm rate stays within a stated budget* (0.5 false alarms per 10 s of video), with a floor of 4.5 on the
  threshold. The rule is stored with the result, together with whether the budget was actually met (it was). If no
  tested threshold had met the budget, the result would say so instead of quietly picking one.
* The **smallest-detectable-movement statement** is a measured quantity: the analytic estimate for a clip is scaled by
  the ratio at which half of the single-action-unit events were actually found (2.08×; nine in ten at about 7.1×). The
  report states both points. **These two factors are the exception to the rule above: they were fitted on all six faces**
  (216 single-action-unit movements), because they only rescale a number the report prints and no held-out score in
  this document depends on them. To show how far they depend on the faces, they are refitted on each group alone and the
  curve fitted on A–C is applied to D–F (next section).
* **Amplitude intervals** are widened by a factor (×2.44) until the 90 % intervals cover 90 % of true amplitudes on
  the calibration faces.
* **Category sets** use split-conformal prediction (`alpha = 0.1`) per label scheme, so a set says which categories
  cannot be excluded.

### Units, so the numbers can be compared

An action unit's *amplitude* is the displacement of its template's dominant region as a percentage of the inter-ocular
distance (IOD). The *size* of a movement is its largest regional displacement; for a single-action-unit event the two
coincide. The generator's own amplitude is not in these units, so the truth used for scoring is computed from the true
canonical optical flow decomposed onto the same templates ("exact events"), after the nuisance directions are projected
out. The generator's raw value is kept only as `generator_pct_iod`.

### Headline numbers (corpus above; calibration run of 2026-11-14; held-out faces D–F unless noted)

| Quantity | Value | Notes |
|---|---|---|
| Spotting threshold | 5.25 | chosen on calibration faces; the false-alarm budget was met |
| Spotting recall, strict / loose timing | 0.49 / 0.54 | overall, all cameras and nuisance levels |
| False alarms | 0.43 per 10 s | movement-free clips alone (all six faces): 0.30 per 10 s |
| Spotting precision | 0.76 | |
| Action-unit F1 (events found) | 0.77 | exact set recovered in 0.53 of events |
| Probability calibration (ECE) | 0.031 | 0.066 with the uncalibrated defaults |
| Smallest detectable movement | 2.08× analytic (half found), ≈ 7.1× (nine in ten) | **fitted on all six faces**, so not held out; see below |
| Amplitude intervals, 90 % nominal | cover 0.94 of true amplitudes | median estimate / truth = 0.82 (a mild under-read) |
| Category set coverage, nominal 0.90 | 0.92 (`megc3`), 0.92 (`casme2_5`), 0.91 (`basic7`) | mean set size 1.2, 1.6, 1.8 |

Recall by condition (all six faces pooled, at the chosen threshold; loose timing): still 0.85, normal 0.56, busy 0.26;
webcam 30 fps 0.54, phone 30 fps 0.36, low-light 24 fps 0.66, 15 fps 0.70, 100 fps 0.50. Detection of the *larger* movements flattens at
about 65–90 % rather than 100 %, because a blink, a burst of speech or a head swing at the same moment hides some of
them; the sensitivity curves in the Evidence station are fitted with that ceiling. The cameras are simulated and the
corpus gave them different mixes of clips, so these figures describe this corpus, not a ranking of real devices.

### How far the smallest-movement factor depends on the faces

| Fitted on | Single-action-unit movements | Found half the time at | Found nine in ten at |
|---|---|---|---|
| All six faces (what the bench uses) | 216 | 2.08× | 7.13× |
| Faces A–C alone | 104 | 1.75× (95 % interval 1.39–2.25) | 5.25× (3.23–9.57) |
| Faces D–F alone | 112 | 2.41× (1.89–3.08) | 8.25× (5.26–15.14) |

Faces D–F needed movements 1.4 times as large as A–C to be found half the time; the intervals overlap, so three faces a
side cannot separate them. Applied to the D–F movements, the curve fitted on A–C predicts that 59 % would be found; 49 %
were. Of the 22 D–F movements at least as large as that curve's "nine in ten" size (5.25×), 20 were found (91 %, 95 %
interval 72–97 %). So the correction carries over to new faces roughly, not precisely: the report's smallest-movement
figure is good to about a sixth of its value either way on phantoms, and that is before the gap between phantoms and
people.

### Controls that were run

* **Shuffled truth**: action-unit F1 with the truth permuted among clips is 0.14 (95th percentile 0.18) against the
  observed 0.77.
* **Prior only** (always name the commonest action units): F1 0.23.
* **Movement-free clips** (all six faces): 7 false alarms in 234 s of video. Across all clips, only 14 % of the 56 false alarms were
  given any action unit at all (0.21 named per false alarm on average): most are reported as an unexplained movement,
  not as an expression.

### What these numbers do **not** establish

1. **Nothing about people.** Phantom skin has no pores, shadows, glasses, beards, make-up, skin-tone variation, or the
   coupling between muscles that real faces show. A sensitivity measured here is an upper bound on what real footage
   allows, not an estimate of it.
2. **Category results are circular.** The categories on phantoms are scripted from the same prototypes that the
   expert map uses, so they test the plumbing (do probabilities add up, do sets cover) and not psychology.
3. **Spotting recall about 0.5 is the honest number on this corpus.** Roughly one scripted movement in two is found; the
   ones missed are mostly small, short, or simultaneous with a blink, speech or a head swing. Some are not recoverable
   by any method from 30 fps video.
4. **Look-alike action units can be indistinguishable.** AU1/AU2/AU4 combinations are only partly identifiable from
   these regional templates (see the Atlas). The engine says so through its inclusion probabilities rather than guessing.
5. **Six faces are few.** The test faces D–F are different identities from the calibration faces, but there are only
   three of them, and they come from the same generator. The smallest-movement factor differs by 1.4× between the two
   groups (above), which is the size of the uncertainty three faces leave.

## 2. Learning harness (`python -m onset.learn.run`)

For use on licensed datasets (see `DATASETS.md`). It is built so that the usual ways of getting an inflated number are
blocked, and so that the result carries its own checks.

### Protocol

* **Subject-independent.** Folds are built from subjects (qualified by dataset), never from clips: leave-one-subject-out
  (`loso`), the same over pooled datasets (`composite`), leave-one-dataset-out (`lodo`, cross-corpus), or subject-disjoint
  k folds (`kfold`, development only).
* **Leakage audit before anything is fitted.** A subject or clip on both sides of a fold, two records sharing a frame
  folder, a clip never tested or tested twice, or an empty side stops the run. Subject labels that look like one person
  and byte-identical first frames across subjects raise warnings.
* **Train-only everything.** Feature normalisation, hyper-parameters (the linear model's regularisation is chosen by
  subject-disjoint inner folds), temperature scaling and conformal thresholds are all fitted on training subjects only; the calibration material is the model's predictions for training subjects it did not learn from.
  Nothing about a test subject reaches fitting, calibration or model selection.
* **Each clip is predicted once per seed**, by the fold in which its subject was held out. Seeds (default three) are
  stacked, so a subject-level resample takes a subject's clips from every seed together.

### Models (in order of ambition; a model has to beat the ones above it)

| Model | What it is |
|---|---|
| `majority` | always the commonest training class |
| `prior` | a class drawn at random with the training frequencies |
| `expert` | ONSET's own action-unit → category map applied to the engine's posterior; **not trained**: it measures how much of the task the hand-written FACS prototypes already solve |
| `linear` (and `linear_pattern`, `linear_au`) | class-balanced multinomial logistic regression on a hand-crafted summary of the clip: apex pattern, regional and group magnitudes, timing, action-unit evidence (the two variants use only one block each, as an ablation) |
| `graphnet` | a small spatio-temporal transformer over the 27 regions and 24 time steps of the clip (2 blocks of region-wise then time-wise attention, width 32, a readout that keeps one weight per region); class-balanced loss, mirror / amplitude / timing / region-dropout augmentation, a one-cycle schedule of at least 180 optimiser steps, a three-member ensemble whose members each hold out a subject-disjoint fifth of the training subjects (calibration material); needs PyTorch |

**A note on what went wrong first, and why the defaults are what they are.** The first version of the network stopped
training when the validation loss stopped improving and restored the best weights. On a validation set of a few clips
the loss is noisy and its minimum is often at the very start of training, so the network that came back had learnt
nothing: on the synthetic reference set it scored at chance (0.31) while the linear model scored 0.79. Early stopping is
now opt-in (`patience`), the schedule always runs to the end, and a floor on the number of optimiser steps keeps small
datasets from being under-trained. A mean-over-all-tokens readout also learned slowly on signals that are sparse in
space and time; the readout now pools each region over time (mean and max) and keeps the regions apart. Hyper-parameters
were set on one split of the synthetic reference set and should be re-checked with a validation protocol of your own on
real data.

**Cost.** The graph-temporal network is the slow part. On the two CPU cores this was developed on, one optimiser step on
a batch of 16 clips takes roughly 0.15–0.2 s. A leave-one-subject-out run on a dataset the size of CASME II (about 250
clips, 26 subjects) with the defaults (3 seeds × 3 ensemble members × 40+ epochs) is a matter of hours, not minutes
(an estimate from the step time, not a measurement). `--seeds 1 --graph-members 1` gives a first look in a fraction of
that, and `--models majority,prior,expert,linear` skips the network entirely. Progress is printed fold by fold.

All models read the **same measured representation** that the report shows (landmark-stabilised dense optical flow
averaged over 27 regions, in percent of the inter-ocular distance). A model trained here is a model of what the
instrument can measure, not of raw pixels.

### Metrics

* **UF1** (unweighted F1) and **UAR** (unweighted average recall), the MEGC convention, next to accuracy, which on
  imbalanced data mostly measures the largest class. The floor for "beats chance" is the UF1 of a uniform guesser on the
  same class counts (not 1/C).
* **Intervals** are subject-level bootstraps (the person is the unit of resampling, since clips of one person are not
  independent); **model comparisons** are paired subject-level permutation tests of UF1 differences with Holm correction
  over the pairs reported.
* **Calibration**: temperature scaling, expected calibration error, log-loss, Brier score, reliability bins; **abstention
  and sets**: split-conformal prediction sets (coverage, mean size, how often a singleton is right), risk–coverage
  curves; for the ensemble, member disagreement when right and when wrong.
* **Latency**: feature extraction per clip (mean, 90th percentile), model inference per clip, fit time per fold.
* **Reproducibility**: `config.json` has the arguments, seeds, package versions, annotation-sheet SHA-256 and feature
  configuration; `predictions.csv` has one row per clip × model × seed.

### Negative controls

A result is only worth reporting if these behave:

* **Shuffled labels**: training labels are permuted inside every fold; the held-out UF1 must fall back to chance
  (a margin of more than 0.08 above chance is reported as a failed control).
* **Identity probe**: can *who the person is* be read from the features? If so, clip-level splits would reward
  recognising faces. This is the reason all folds are subject-disjoint.
* **Leakage gap**: the same model scored with subject-disjoint folds and with (invalid) clip-level folds, to show how much
  a leaky split would have flattered the number.

### The synthetic reference run

`python -m onset.learn.run --synthetic --scheme casme2_5` builds 72 phantom clips from 12 identities, writes them in the
CASME II folder layout with a coding sheet, and sends them through the whole harness. **It checks the machinery** (the
loader, the audit, the folds, the models, the calibration, the controls, the report) **and is not a benchmark**: the
categories are scripted from the same prototypes the expert map uses, the faces are procedural, and 72 clips are a
handful. Its results appear in the Evidence station under "learning runs", labelled as such.

The run recorded with this build (`--scheme casme2_5 --protocol loso --seeds 2 --graph-members 2`: 12 folds, 72 clips, five
classes with 'others' the largest at 42 %) is a useful example of how a result should be read, because most of what it says
is a caution:

* every model is far ahead of `majority` (UF1 0.12) and `prior` (0.17): the expert map 0.66, the linear model 0.76, the
  graph network 0.69, all against `majority` at Holm-adjusted p = 0.006;
* **the models cannot be separated from each other**: linear against graph network is +0.065 with an adjusted p of 0.41, and
  the intervals (subjects are the unit, and there are twelve of them) overlap. The harness says so; a table that ranked
  them would be over-reading twelve phantoms;
* shuffled labels fall to chance (0.18 against 0.19), so the pipeline is not leaking labels;
* subject identity can be read from the features at 40 % against 8 % by chance, which is why the folds are
  subject-disjoint and why a clip-level split would flatter any model;
* ECE is 0.10 to 0.20 for the expert map and the trained models even after temperature scaling on a handful of inner
  subjects; the conformal sets cover about as often as promised (0.90 to 0.96 against a nominal 0.90) at the price of 2.3 to 3.3 labels out of five;
* cost: feature extraction 1.4 s per clip, about 48 s to train the two-member network for one fold on two CPU cores, 2.9 ms per
  clip at inference (timings from a shared two-core machine, not a benchmark).

None of this transfers to people: the movements are scripted from the same prototypes the expert map uses, so `expert`
doing well here only shows that the machinery reads its own prototypes back.

### Reading a real-data result

Before quoting a number from a real run, check, in this order: the audit passed and the counts match the paper's;
shuffled labels fell to chance; the leakage gap is reported and large (it should be); the model beats `expert` and
`majority` with an interval that excludes the difference (the permutation test, adjusted); calibration (ECE) and set
coverage are close to nominal; then, and only then, the headline UF1. Report UF1 and UAR with their subject-bootstrap
intervals, not accuracy alone, and say which label scheme and which protocol. `lodo` numbers are usually far lower than
`loso` numbers, and that is the one that predicts how it would do on a new collection.
