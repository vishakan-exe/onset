# Datasets

ONSET ships **no recordings of real people**. Its specimens and calibration corpus are synthetic faces ("phantoms") drawn
by code. To evaluate learned models on real micro-expressions you need a copy of a public dataset, obtained from its
owners under their licence. This page says what the learning harness (`python -m onset.learn.run`) expects to find in
your copy, how labels are mapped to the schemes ONSET reports, and what the harness does with rows it cannot use.

> Everything about the datasets themselves (participants, clips, frame rate, resolution, label counts) must be taken from
> the papers and release notes of the copy you hold. Releases differ between versions, and some figures quoted in the
> literature differ from each other because papers use different subsets. This page deliberately does not quote counts.

## Obtaining data

Each collection contains identifiable faces. The owners grant access to researchers who sign an agreement, usually for
non-commercial research, and forbid redistribution. Do not copy the data into this folder, into an archive you share, or
into a cloud service. The harness reads from a folder you point it at and writes only numbers (see *What is stored*).

| Dataset | Reference | Where to request it |
|---|---|---|
| CASME II | Yan, Li, Wang, Zhao, Liu, Chen & Fu (2014), PLOS ONE 9(1): e86041 | <http://casme.psych.ac.cn/casme/e2> |
| SAMM | Davison, Lansley, Costen, Tan & Yap (2016), IEEE Trans. Affective Computing 9(1) | <https://personalpages.manchester.ac.uk/staff/adrian.davison/SAMM.html> |
| MMEW | Ben, Ren, Zhang et al. (2021), IEEE TPAMI 44(9) (a survey that introduces the dataset) | <http://www.dpailab.com/database.html> |
| 4DME | Li, Cheng, Li, Behzad, Shen, Zafeiriou, Pantic & Zhao (2022), IEEE Trans. Affective Computing | the Oulu CMVS group page |
| CAS(ME)³ | Li, Dong, Lu, Wang, Yan, Ma, Liu, Huang & Fu (2022), IEEE TPAMI | <http://casme.psych.ac.cn/casme/e4> |

Web addresses move. If one is dead, search for the dataset's name and the paper's title. The SMIC database (Li et al.,
2013) is not given an adapter because its release layout differs by version; the `generic` adapter below can read it
once you write its sheet.

## What the loaders read

Every adapter finds the annotation table (`.xlsx` or `.csv`, searched for under the dataset folder unless you pass
`--annotations NAME=SHEET`), locates the header row, and matches columns **by name** (case and spacing ignored, a few
aliases accepted), not by position. If an essential column is missing it stops and prints the columns it found. The
layouts below are the commonly distributed ones; if your copy differs, use the `generic` adapter.

| Adapter | Frame folders | Annotation sheet (name contains) | Frame rate used |
|---|---|---|---|
| `casme2` | `<root>/sub<NN>/<Filename>/img<N>.jpg` | `CASME2-coding` | 200 fps |
| `samm` | `<root>/<SSS>/<Filename>/<SSS>_<NNNNN>.jpg` | `SAMM_Micro_FACS` | 200 fps |
| `mmew` | `<root>/<emotion>/<Filename>/<N>.jpg` | `MMEW_Micro` | 90 fps |
| `fourd` | `<root>/<sub_ID>/<fold_name>/` (the 2-D video frames) | `4DME_Labelling` | 60 fps |
| `casme3a` | `<root>/<subject>/<Filename>/color/<N>.jpg` | `cas(me)3_part_A` | 30 fps |
| `generic` | any, via `--template` | any sheet with the columns below | `--fps` |

The frame rates are the nominal rates of the releases as the authors describe them; they set only how seconds are
computed (durations, speeds). Pass `--fps` to `generic` when you know the real rate.

Columns, with the aliases tried:

| Meaning | Names accepted (first match wins) | Required |
|---|---|---|
| subject | `Subject` (`sub_ID` for 4DME) | yes |
| clip | `Filename` (`fold_name` for 4DME; `clip` or `filename` for generic) | yes |
| onset, offset | `OnsetFrame`/`Onset Frame`/`Onset`, `OffsetFrame`/`Offset Frame`/`Offset` | yes |
| apex | `ApexFrame`/`Apex Frame`/`Apex` | no: when absent or outside onset..offset the strongest measured frame is used |
| emotion | `Estimated Emotion`, `Emotion` (`Objective class` for CAS(ME)³) | no, but a clip without a label is only usable by schemes that do not need one, i.e. none of the three |
| action units | `Action Units`, `AU`, `AUs` | no: used for the AU read-out check only |

**The `generic` adapter** accepts any sheet with `subject, clip, onset, offset` columns (plus optional `apex, emotion,
aus`) and a folder template that may use `{root}`, `{subject}`, `{clip}` and `{emotion}`:

```bash
python -m onset.learn.run --data generic=/data/mine --annotations generic=/data/mine/labels.csv \
    --template "{root}/{subject}/{clip}" --fps 100 --scheme megc3 --protocol loso --out runs/mine
```

### Frame numbering

Frame numbers in the sheets are matched against the numbers in the image file names. If the sheet's onset and offset
are not file numbers in that folder (some releases number frames from 1 inside each folder), the adapter falls back to
position within the folder and records `numbering: "relative"` for that clip. Which of the two happened is in the
manifest, so it can be checked.

### Rows that cannot be used

Nothing is dropped silently. A row is reported in `problems.csv` (and counted in the run summary) when

* onset or offset is missing, not a number, or offset < onset;
* the frame folder does not exist, holds no images, or does not contain the annotated range.

An unrecognised or empty emotion is kept as `None`: the clip simply does not belong to any scheme that needs a label.
An annotated apex outside onset..offset is ignored (the measured apex is used). Duplicate (subject, clip) pairs are
renamed with the row number so both survive and the audit can see them.

## Label schemes

Datasets disagree about what "others", "tense" or "repression" mean, and the three-class composite scheme merges several
emotions. The maps are choices, written down in `onset/learn/labels.py` so they can be argued with. A clip whose label
has no counterpart in a scheme is **excluded from that scheme** and never relabelled.

| Scheme | Classes | Mapping of dataset labels |
|---|---|---|
| `megc3` | positive, negative, surprise | happiness and positive → positive; surprise → surprise; disgust, fear, sadness, anger, contempt, repression and negative → negative; **others, tense and anything unrecognised are excluded** |
| `casme2_5` | happiness, disgust, surprise, repression, others | those four kept; others, tense, fear, sadness, anger and contempt → others; anything else excluded |
| `basic7` | happiness, surprise, fear, anger, disgust, sadness, contempt | kept as they are; everything else (repression, tense, others) excluded |

Synonyms folded before mapping: happy/joy → happiness; sad → sadness; angry → anger; contemptuous → contempt;
disgusted → disgust; fearful → fear; surprised → surprise; other → others; repress/repressed → repression.

Two cautions that matter when you compare with published numbers:

1. **"Others" is a catch-all.** In `casme2_5` it holds clips of several different emotions plus whatever the coders could
   not place. A model that scores well on it may be learning "not one of the four".
2. **The labels were not produced by looking at the movement alone.** The dataset authors assigned them from the action
   units, the stimulus the participant was watching and often the participant's own report. The same movement can carry
   different labels in different datasets. A classifier measures agreement with those labels, not the person's feeling.

Action-unit strings are parsed loosely (`R12 + R15`, `4+7`, `L14`, `AU9, AU10`, `R14A`): side and intensity letters are
dropped; `(k)` marks (4DME) are dropped; for `12 or 14` only the first alternative is used because the coders did not
decide.

## Protocols

All folds are built from **subjects**, qualified by their dataset (two people both numbered "01" are never merged).

| `--protocol` | Train on | Test on | Use for |
|---|---|---|---|
| `loso` | all other subjects of the dataset | the held-out subject | the standard single-dataset number |
| `composite` | all other subjects of the pooled datasets | the held-out subject | MEGC-style composite evaluation |
| `lodo` | the other datasets | the whole held-out dataset | cross-corpus generalisation; the most honest number and usually the lowest |
| `kfold` | subject-disjoint folds, balanced by class counts | one fold | development only (quicker) |

Before anything is fitted, `protocols.audit` checks the folds and **stops the run** (`LeakageError`) when a subject or
clip is on both sides of a fold, when two records share a frame folder, when a clip is never tested or tested twice, or
when a fold has an empty side. It warns about subject labels that normalise to the same person and about clips of
different subjects that start with a byte-identical first frame (a clip filed twice).

## What is stored

For each clip the harness writes `cache/<hash>.npz`: arrays of numbers only (regional displacement series, apex
pattern, action-unit posterior and summary features). **No pixels and no landmarks are stored.** The hash includes the
feature configuration and the engine calibration, so a changed configuration re-measures instead of reusing stale
numbers. `config.json` records the arguments, package versions, seeds, the SHA-256 of each annotation sheet and the
feature configuration, so a result can be tied to the exact inputs.

## What is not covered

* **Spotting on real long videos** (CAS(ME)², SAMM Long Videos, the MEGC spotting tasks) is not evaluated. The spotting
  threshold is calibrated on synthetic clips only; event-level F1 against real annotations needs its own protocol.
* **Cross-dataset frame-rate differences** are handled by measuring displacement in percent of the inter-ocular
  distance and resampling every clip to a fixed number of steps; this is a convenience, not a guarantee that a model
  trained at 200 fps transfers to 30 fps video. `lodo` is how to find out.
* **Depth and 3-D modalities** (4DME, CAS(ME)³) are not used; only the colour frames.
