"""Lineage clustering (stage 6) and frozen train/test splits (stage 7).

Two modules:

* :mod:`genome2mic.splits.lineages` -- sketches every QC-passing genome, clusters
  genomes of one species by Mash distance (single linkage) and writes
  ``data/processed/lineages.parquet``.
* :mod:`genome2mic.splits.make_splits` -- assigns whole lineage clusters to
  ``train`` / ``test``, builds ``GroupKFold`` folds by cluster, marks external
  hold-out sets and leave-one-lineage-out rows, and writes the frozen
  ``data/processed/splits.parquet``.

Both files are contract outputs (``DATA_CONTRACT.md`` stages 6 and 7). Nothing in
this package may ever become a model feature: ``lineage_cluster``, ``st``, ``split``
and ``fold`` exist only for splitting and evaluation.
"""

from __future__ import annotations

__all__ = ["lineages", "make_splits"]
