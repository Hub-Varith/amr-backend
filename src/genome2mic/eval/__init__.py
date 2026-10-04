"""Evaluation of MIC predictions (contract stage 11).

* :mod:`genome2mic.eval.metrics` -- pure metric functions and :func:`~genome2mic.eval.metrics.summarize`,
  which turns a stacked predictions table into one row per species x drug x model x
  evaluation set with **VME first**.
* ``genome2mic.eval.run`` -- the evaluate stage: ``metrics.parquet`` / ``metrics_by_distance.parquet``.
* ``genome2mic.eval.report`` -- ``results/report.md`` and figures (``eval.figures``), plus the
  automated leakage checklist (``eval.leakage``).
* ``genome2mic.eval.demo`` -- ``make demo-copy``: copies a synthetic run's report into
  ``reports/synthetic_demo/`` with every artefact labelled synthetic.

Everything here consumes the stage-10 predictions table only; nothing reaches back
to raw data or features. These are evaluations of in-vitro susceptibility
predictions, not of clinical outcomes.
"""

__all__ = ["metrics"]
