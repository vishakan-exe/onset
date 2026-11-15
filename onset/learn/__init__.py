"""Learning and evaluation harness for real micro-expression datasets.

This package is where the instrument meets licensed data.  Nothing here ships data: CASME II, SAMM, MMEW, 4DME and
CAS(ME)^3 need a signed agreement with their owners, so the harness reads the copy *you* are licensed to hold and
writes only numbers (per-clip predictions, metrics, calibration) into a run folder.

What it enforces, because most published micro-expression numbers fail at least one of these:

* **Subject-independent evaluation.**  Folds are built from subjects, never from clips (leave-one-subject-out,
  leave-one-dataset-out, composite).  ``protocols.audit`` refuses to run a fold in which a subject, a clip or a
  duplicated clip folder appears on both sides.
* **No leakage through preprocessing.**  Normalisation statistics, class weights, hyper-parameters, temperature and
  the conformal threshold are fitted on training subjects only, using inner subject-disjoint splits.
* **Class imbalance.**  The headline metrics are UF1 (unweighted F1) and UAR (unweighted average recall) pooled over
  folds, next to accuracy; training is class-balanced; a majority-class model and a prior-only model are always run.
* **Uncertainty.**  Probabilities are temperature-calibrated on held-out subjects, prediction sets are split-conformal,
  every metric carries a *subject-level* bootstrap interval, and model differences get paired subject-level
  permutation tests.
* **Controls.**  A shuffled-label run must fall to chance and a subject-identity probe says whether the features carry
  identity (which would make any clip-level split meaningless).
* **Latency.**  Feature extraction and inference time per clip are measured and reported.
* **Reproducibility.**  Every run writes its configuration, package versions, seeds and the hash of the annotation
  file next to its results.

Run ``python -m onset.learn.run --help``; ``python -m onset.learn.run --synthetic`` runs the whole machinery on
phantom clips laid out like CASME II, which checks the code and is not a benchmark.
"""
from __future__ import annotations

__all__ = ["labels", "datasets", "protocols", "features", "models", "metrics", "calibrate", "controls", "evaluate", "synthetic", "run", "report"]
