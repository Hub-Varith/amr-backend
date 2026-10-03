"""Evaluation of MIC predictions (contract stage 11).

* :mod:`genome2mic.eval.metrics` -- pure metric functions and :func:`~genome2mic.eval.metrics.summarize`,
  which turns a stacked predictions table into one row per species x drug x model x
  evaluation set with **VME first**.
* ``genome2mic.eval.report`` -- ``results/report.md`` and figures (separate module).

Everything here consumes the stage-10 predictions table only; nothing reaches back
to raw data or features. These are evaluations of in-vitro susceptibility
predictions, not of clinical outcomes.
"""

__all__ = ["metrics"]
