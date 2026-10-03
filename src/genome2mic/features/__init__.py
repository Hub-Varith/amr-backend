"""Feature construction for genome2mic.

Modules:

* :mod:`genome2mic.features.known_amr` -- stage 5, curated AMR genes and point
  mutations from AMRFinderPlus output (``gene_``, ``point_``, ``n_class_`` columns).
* ``genome2mic.features.unitigs`` -- stage 8, unitig / k-mer presence patterns
  (``u_`` columns), built on training genomes only.
* ``genome2mic.features.select`` -- fold-side feature selection.

Nothing is imported here on purpose: each stage module is imported directly so
that loading one stage never pulls in another stage's optional dependencies.
"""
