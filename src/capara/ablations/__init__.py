"""Ablations reported in the paper.

Each module trains a family of runs that differ in exactly one axis, scores every
run on all four retrieval benchmarks, and writes its JSON, LaTeX table and figure
to ``results/ablations/``.

* :mod:`capara.ablations.pos_neg_scaling` -- more positives vs. more hard negatives.
* :mod:`capara.ablations.data_efficiency` -- how much of CC3M the paragraph
  supervision actually needs.
"""
