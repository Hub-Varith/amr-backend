"""Every file location in the project, derived from one root directory.

The layout follows ``DATA_CONTRACT.md`` section 1 and the repository layout in
``CLAUDE.md``::

    <root>/
      data/raw/            ast_<source>.csv, genomes/<gid>.fasta, references/
      data/interim/<gid>/  amrfinder.tsv, resfinder/, mlst.tsv, mash.tsv
      data/processed/      labels.parquet, qc.parquet, known_amr.parquet, ...
      results/             preds_<SPECIES>_<drug>.parquet, metrics.parquet, figures/
      models/              <SPECIES>/<drug>/ bundles, manifest.json

Stage CLIs take ``--root`` and ``--configs-dir`` and build a :class:`Paths` from
them; no module hard-codes a path. Species keys are always the upper-case 5-letter
keys (``KPNEU``); drug names are already normalized (lowercase, hyphenated) by the
time they reach a path helper, so they are used verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RAW_AST_SOURCES: frozenset[str] = frozenset({"bvbrc", "ncbi"})
"""Accepted ``source`` values for :meth:`Paths.raw_ast` (case-insensitive)."""


@dataclass(frozen=True)
class Paths:
    """Resolved project paths.

    Attributes:
        root: Directory containing ``data/``, ``results/`` and ``models/``.
        configs_dir: Directory holding ``species.yaml``, ``drugs.yaml``,
            ``breakpoints/`` etc. Usually ``<root>/configs`` or ``./configs``.
    """

    root: Path = Path(".")
    configs_dir: Path = Path("configs")

    def __post_init__(self) -> None:
        # Accept plain strings from argparse without breaking the frozen contract.
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "configs_dir", Path(self.configs_dir))

    @classmethod
    def default(cls) -> "Paths":
        """Paths for running from the repository root: ``root='.'``, ``configs_dir='configs'``."""
        return cls(root=Path("."), configs_dir=Path("configs"))

    # ------------------------------------------------------------------ data
    @property
    def data_dir(self) -> Path:
        """``<root>/data``."""
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        """``<root>/data/raw`` -- downloads, never edited."""
        return self.data_dir / "raw"

    def raw_ast(self, source: str) -> Path:
        """``<root>/data/raw/ast_<source>.csv`` for ``source`` in ``{bvbrc, ncbi}``.

        Raises:
            ValueError: for an unknown source (typo guard; the two files are the
                only raw AST inputs the pipeline knows how to read).
        """
        key = source.strip().lower()
        if key not in RAW_AST_SOURCES:
            raise ValueError(
                f"unknown raw AST source {source!r}; expected one of {sorted(RAW_AST_SOURCES)}"
            )
        return self.raw_dir / f"ast_{key}.csv"

    @property
    def genome_metadata(self) -> Path:
        """``<root>/data/raw/genome_metadata.csv`` (genome_id, biosample, species, source, ...)."""
        return self.raw_dir / "genome_metadata.csv"

    @property
    def genomes_dir(self) -> Path:
        """``<root>/data/raw/genomes`` -- one FASTA per genome."""
        return self.raw_dir / "genomes"

    def genome_fasta(self, genome_id: str) -> Path:
        """``<root>/data/raw/genomes/<genome_id>.fasta``."""
        return self.genomes_dir / f"{genome_id}.fasta"

    @property
    def references_dir(self) -> Path:
        """``<root>/data/raw/references`` -- one reference FASTA per species for Mash species ID."""
        return self.raw_dir / "references"

    def reference_fasta(self, species: str) -> Path:
        """``<root>/data/raw/references/<SPECIES>.fasta``."""
        return self.references_dir / f"{_species_key(species)}.fasta"

    @property
    def interim_root(self) -> Path:
        """``<root>/data/interim`` -- parent of every per-genome tool-output directory."""
        return self.data_dir / "interim"

    def interim_dir(self, genome_id: str) -> Path:
        """``<root>/data/interim/<genome_id>`` -- amrfinder.tsv, resfinder/, mlst.tsv, mash.tsv."""
        return self.interim_root / genome_id

    @property
    def processed_dir(self) -> Path:
        """``<root>/data/processed`` -- the contract files."""
        return self.data_dir / "processed"

    @property
    def labels(self) -> Path:
        """Stage 2 contract: ``processed/labels.parquet``."""
        return self.processed_dir / "labels.parquet"

    @property
    def pairs_kept(self) -> Path:
        """Species x drug pairs passing the inclusion rule: ``processed/pairs_kept.csv``."""
        return self.processed_dir / "pairs_kept.csv"

    @property
    def label_counts(self) -> Path:
        """Count table emitted after stage 2: ``processed/label_counts.csv``."""
        return self.processed_dir / "label_counts.csv"

    @property
    def qc(self) -> Path:
        """Stage 3: ``processed/qc.parquet``."""
        return self.processed_dir / "qc.parquet"

    @property
    def known_amr(self) -> Path:
        """Stage 5 contract: ``processed/known_amr.parquet``."""
        return self.processed_dir / "known_amr.parquet"

    @property
    def known_amr_columns(self) -> Path:
        """Column mapping for stage 5: ``processed/known_amr_columns.csv``."""
        return self.processed_dir / "known_amr_columns.csv"

    @property
    def lineages(self) -> Path:
        """Stage 6 contract: ``processed/lineages.parquet``."""
        return self.processed_dir / "lineages.parquet"

    def sketches(self, species: str) -> Path:
        """MinHash sketches of QC-passing genomes: ``processed/sketches_<SPECIES>.npz``."""
        return self.processed_dir / f"sketches_{_species_key(species)}.npz"

    @property
    def splits(self) -> Path:
        """Stage 7 contract (frozen, version-controlled): ``processed/splits.parquet``."""
        return self.processed_dir / "splits.parquet"

    def unitigs(self, species: str) -> Path:
        """Stage 8 CSR matrix: ``processed/unitigs_<SPECIES>.npz``."""
        return self.processed_dir / f"unitigs_{_species_key(species)}.npz"

    def unitig_rows(self, species: str) -> Path:
        """Row index -> genome_id: ``processed/unitigs_<SPECIES>_rows.parquet``."""
        return self.processed_dir / f"unitigs_{_species_key(species)}_rows.parquet"

    def unitig_index(self, species: str) -> Path:
        """Column index -> pattern -> sequences: ``processed/unitigs_<SPECIES>_index.parquet``."""
        return self.processed_dir / f"unitigs_{_species_key(species)}_index.parquet"

    def unitig_kmers(self, species: str) -> Path:
        """Fixed k-mer set used to query new genomes: ``processed/unitigs_<SPECIES>_kmers.npz``."""
        return self.processed_dir / f"unitigs_{_species_key(species)}_kmers.npz"

    def drop_log(self, stage: str) -> Path:
        """Per-stage filter counts: ``processed/drop_log_<stage>.csv``."""
        return self.processed_dir / f"drop_log_{stage}.csv"

    # --------------------------------------------------------------- results
    @property
    def results_dir(self) -> Path:
        """``<root>/results``."""
        return self.root / "results"

    def preds(self, species: str, drug: str) -> Path:
        """Stage 10: ``results/preds_<SPECIES>_<drug>.parquet``."""
        return self.results_dir / f"preds_{_species_key(species)}_{drug}.parquet"

    @property
    def metrics(self) -> Path:
        """Stage 11: ``results/metrics.parquet``."""
        return self.results_dir / "metrics.parquet"

    @property
    def metrics_by_distance(self) -> Path:
        """Accuracy by nearest-training-distance bin: ``results/metrics_by_distance.parquet``."""
        return self.results_dir / "metrics_by_distance.parquet"

    @property
    def figures_dir(self) -> Path:
        """``<root>/results/figures``."""
        return self.results_dir / "figures"

    @property
    def report_md(self) -> Path:
        """``<root>/results/report.md``."""
        return self.results_dir / "report.md"

    # ---------------------------------------------------------------- models
    @property
    def models_dir(self) -> Path:
        """``<root>/models`` -- trained bundles consumed by the prediction pipeline."""
        return self.root / "models"

    @property
    def models_manifest(self) -> Path:
        """``<root>/models/manifest.json``."""
        return self.models_dir / "manifest.json"

    def model_dir(self, species: str, drug: str) -> Path:
        """``<root>/models/<SPECIES>/<drug>`` -- model.ubj, features.json, conformal.json, meta.json."""
        return self.models_dir / _species_key(species) / drug


def _species_key(species: str) -> str:
    """Normalize a species key for use in file names (``kpneu`` -> ``KPNEU``)."""
    key = species.strip().upper()
    if not key:
        raise ValueError("species key must be a non-empty string")
    return key
