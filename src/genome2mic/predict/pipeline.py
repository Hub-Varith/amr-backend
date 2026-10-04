"""Genome-to-report prediction pipeline (``DATA_CONTRACT.md`` stage 12).

One :class:`PredictionPipeline` holds every trained model bundle under ``models/``
and turns one assembled genome into one report dict::

    pipeline = PredictionPipeline(models_dir, configs_dir)
    pipeline.load()                       # FileNotFoundError when manifest.json is missing
    pipeline.available_models()           # {"KPNEU": ["ceftriaxone", "ciprofloxacin", ...]}
    report = pipeline.run(fasta, "BC-0142")   # validates against api.schemas.PredictionReport

Steps of :meth:`PredictionPipeline.run` (``.context/DESIGN.md`` predict section):

1. QC -- assembly stats, MinHash sketch, species = nearest reference sketch
   (``models/reference_sketches.npz``); ``qc_pass`` by the contract rules (contig
   count, genome size against ``species.yaml``, distance to every reference).
   A genome farther than ``qc.max_mash_distance`` from every reference gets
   ``species = None``, ``qc_pass = False``, no predictions and ``in_range = False``.
2. ``nearest_training_distance`` = min Mash distance to ``models/<SPECIES>/train_sketches.npz``;
   ``in_range`` = that distance is within ``qc.max_mash_distance`` and the species has models.
3. Known AMR via :mod:`genome2mic.predict.amr_detect` (AMRFinderPlus CLI, a sidecar
   TSV, or -- for synthetic bundles only -- the bundle's ``markers.fasta``), with the
   training row filter and column rules. Bundles trained on an imported NCBI release
   (``manifest.json`` ``feature_naming: ncbi_release``) use the release rules instead
   (:mod:`genome2mic.predict.release_features`, ``models/<SPECIES>/feature_spec.json``).
4. Unitig pattern vector via ``genome2mic.features.unitigs.query_kmer_set`` against the
   bundle's fixed k-mer set, loaded once by :meth:`PredictionPipeline.load` (only
   when a drug model uses unitig features).
5. Per drug: ``pred_mic`` (rounded **up** to the doubling grid), conformal band
   (asymmetric ``q_up`` / ``q_low`` when the bundle has them), call (``likely_active``
   withheld when the bundle's ``active_gate_open`` is false), ``margin_steps``, reasons.
6. Overrides: natural resistance (model skipped, MIC fields null) and strong
   markers (``drugs.yaml`` ``strong_markers`` prefixes and ``strong_subclasses`` of
   acquired genes; MIC fields kept).
7. ``ranked_active`` from :func:`genome2mic.predict.rank.rank_active`. When
   ``in_range`` is False the calls are kept and the flag tells the UI to show them
   as low confidence (override 3).

Bundle layout (written by ``models/train.py``)::

    models/manifest.json                 {model_version, run_id, created, species: {KPNEU: [drugs]}, synthetic}
    models/reference_sketches.npz        sketch.save_sketches format, ids = species keys
    models/markers.fasta                 optional; MarkerScan fallback, used only when
                                         manifest.json has ``synthetic: true``
    models/<SPECIES>/train_sketches.npz
    models/<SPECIES>/unitig_kmers.npz    optional; absent -> no unitig features
    models/<SPECIES>/unitig_index.parquet  optional; pattern_id -> col_index
    models/<SPECIES>/<drug>/features.json  {model_class, known_columns, unitig_cols, class_by_column,
                                            feature_names, unitig_kmer_set_sha1}
    models/<SPECIES>/<drug>/conformal.json {q, [q_up, q_low, active_gate_open], [cap_low_log2, cap_high_log2]}
    models/<SPECIES>/<drug>/meta.json      free-form
    models/<SPECIES>/<drug>/...            whatever MODEL_CLASSES[model_class].load(dir) reads

Nothing in a feature matrix may come from ``lineage_cluster``, ``st``, ``country``,
``year``, ``source``, ``isolation_source``, ``biosample``, ``split`` or ``fold`` (or the
label / key columns); ``load`` rejects a bundle whose ``known_columns`` contain any
name in :data:`genome2mic.models.base.FORBIDDEN_FEATURES`, and one whose
``known_columns + unitig_cols`` differ (order included) from ``features.json``
``feature_names`` or from the saved model's ``feature_names_``: a same-length
reordering would otherwise feed every feature into the wrong model input.

``unitig_kmers.npz`` is shared by every drug of a species. Each unitig model's
``features.json`` records the set it was trained on (``unitig_kmer_set_sha1``, the
:meth:`genome2mic.features.unitigs.KmerSet.sha1` content hash); ``load`` raises
:class:`BundleError` when that differs from the species' set, because the model's
unitig column indices would then point into a different pattern list. Bundles
written before the key existed load with a warning (the pairing cannot be checked).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from genome2mic import sketch as sk
from genome2mic.api.constants import DISCLAIMER
from genome2mic.config import Config, load_config
from genome2mic.droplog import DropLog
from genome2mic.errors import Genome2MicError
from genome2mic.io import read_fasta
from genome2mic.mic import GRID_MAX_EXPONENT, GRID_MIN_EXPONENT, round_up_to_step
from genome2mic.models.base import FORBIDDEN_FEATURES
from genome2mic.predict import amr_detect, rank, release_features

logger = logging.getLogger(__name__)

__all__ = [
    "FORBIDDEN_FEATURES",
    "AssemblyStats",
    "BundleError",
    "DrugBundle",
    "GenomeQc",
    "PredictionPipeline",
    "SpeciesBundle",
    "assembly_stats",
]

MANIFEST_FILE = "manifest.json"
FEATURE_NAMING_DEFAULT = "genome2mic"
"""``manifest.json`` ``feature_naming`` when absent: ``features/known_amr.py`` column rules."""
REFERENCE_SKETCHES_FILE = "reference_sketches.npz"
MARKERS_FILE = "markers.fasta"
TRAIN_SKETCHES_FILE = "train_sketches.npz"
UNITIG_KMERS_FILE = "unitig_kmers.npz"
UNITIG_INDEX_FILE = "unitig_index.parquet"
FEATURES_FILE = "features.json"
CONFORMAL_FILE = "conformal.json"
META_FILE = "meta.json"

QC_EMPTY_ASSEMBLY = "empty_assembly"
QC_TOO_FRAGMENTED = "too_fragmented"
QC_TOO_DISTANT = "too_distant"
QC_UNKNOWN_REFERENCE = "unknown_reference"
QC_WRONG_GENOME_SIZE = "wrong_genome_size"

UnitigQuery = Callable[[Path, Path], np.ndarray]
"""``(unitig_kmers.npz, fasta) -> int8 pattern-presence vector`` (one entry per pattern column)."""

UNITIG_PATTERN_PREFIX = "u_"
"""Training names unitig pattern column ``c`` ``u_{c:06d}`` (``models/train.py``)."""


class BundleError(Genome2MicError):
    """A model bundle under ``models/`` exists but is malformed or inconsistent.

    Missing files raise :class:`FileNotFoundError` instead. Unreadable sketch, k-mer
    and model files are re-raised as this error with the path. The API registry
    turns any load failure into "not ready"; this error means the bundle cannot be
    trusted.
    """


# --------------------------------------------------------------------------- #
# Bundle records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DrugBundle:
    """One trained species x drug model and the feature layout it expects."""

    species: str
    drug: str
    model_class: str
    model: Any
    known_columns: tuple[str, ...]
    unitig_cols: tuple[int, ...]
    class_by_column: dict[str, tuple[str | None, str | None]]
    q: float
    meta: dict[str, Any]
    unitig_kmer_set_sha1: str | None = None
    """sha1 of the k-mer set the model was trained on (``features.json``); ``None`` when
    the model has no unitig features or the bundle predates the key."""
    caps: tuple[float, float] | None = None
    """Panel-edge caps in log2 mg/L (``conformal.json`` ``cap_low_log2`` / ``cap_high_log2``,
    from all train rows): the raw prediction is clipped to them before rounding up, as in
    training. ``None`` for bundles written before capping existed."""
    q_low: float | None = None
    """Lower half-width in steps of the tuned asymmetric band (``conformal.json`` ``q_low``;
    ``q`` is then the upper half-width ``q_up``). ``None``: symmetric ``+-q`` band."""
    active_gate_open: bool = True
    """``conformal.json`` ``active_gate_open``: False means training could not certify
    call-level VME <= 1.5 % for this pair, so ``likely_active`` becomes ``uncertain``."""

    @property
    def n_features(self) -> int:
        return len(self.known_columns) + len(self.unitig_cols)


@dataclass(frozen=True)
class SpeciesBundle:
    """Everything under ``models/<SPECIES>/``.

    ``kmer_set`` is the frozen unitig k-mer set (``features.unitigs.KmerSet``), read
    once at load when a drug model uses unitig features and the default query is in
    use; ``None`` otherwise (no unitig model, or an injected ``unitig_query``).
    """

    species: str
    directory: Path
    train_ids: tuple[str, ...]
    train_sketches: np.ndarray
    train_k: int
    unitig_kmers: Path | None
    drugs: dict[str, DrugBundle]
    kmer_set: Any | None = None
    feature_spec: Any | None = None
    """``release_features.ReleaseFeatureSpec`` from ``models/<SPECIES>/feature_spec.json`` when
    the bundle was trained on an imported NCBI release (``manifest.json`` ``feature_naming:
    ncbi_release``); ``None`` for bundles built with ``features/known_amr.py`` naming."""

    @property
    def needs_unitigs(self) -> bool:
        return any(bundle.unitig_cols for bundle in self.drugs.values())


# --------------------------------------------------------------------------- #
# QC helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AssemblyStats:
    """Contract stage-3 assembly statistics for one FASTA."""

    n_contigs: int
    total_length: int
    n50: int
    gc_percent: float | None


def assembly_stats(seqs: Sequence[str]) -> AssemblyStats:
    """``n_contigs``, ``total_length``, ``n50`` and GC% of a list of contig sequences.

    GC% counts G and C over A, C, G and T only (ambiguity codes are ignored) and is
    ``None`` for an empty assembly.
    """
    lengths = sorted((len(seq) for seq in seqs), reverse=True)
    total = int(sum(lengths))
    n50 = 0
    running = 0
    for length in lengths:
        running += length
        if running * 2 >= total:
            n50 = int(length)
            break
    gc = at = 0
    for seq in seqs:
        upper = seq.upper()
        gc += upper.count("G") + upper.count("C")
        at += upper.count("A") + upper.count("T")
    gc_percent = (100.0 * gc / (gc + at)) if (gc + at) else None
    return AssemblyStats(n_contigs=len(lengths), total_length=total, n50=n50, gc_percent=gc_percent)


@dataclass(frozen=True)
class GenomeQc:
    """Species identification and QC outcome for one genome."""

    species: str | None
    reference_distance: float | None
    stats: AssemblyStats
    fail_reasons: tuple[str, ...]

    @property
    def qc_pass(self) -> bool:
        return not self.fail_reasons


def _load_sketch_file(path: Path) -> tuple[list[str], np.ndarray, int]:
    """``sk.load_sketches`` with unreadable content re-raised as :class:`BundleError`."""
    if not path.is_file():
        raise FileNotFoundError(f"Sketch file not found: {path}")
    try:
        return sk.load_sketches(path)
    except Exception as error:  # np.load: ValueError, OSError, BadZipFile, EOFError, ...
        raise BundleError(f"{path}: unreadable sketch file ({type(error).__name__}: {error})") from error


def _unitig_feature_names(raw: Sequence[Any]) -> list[str]:
    """``unitig_cols`` entries as training feature names (ints -> ``u_000017``; ids unchanged)."""
    return [item if isinstance(item, str) else f"{UNITIG_PATTERN_PREFIX}{int(item):06d}" for item in raw]


def _feature_order_problem(expected: Sequence[str], found: Sequence[str]) -> str | None:
    """``None`` if the two name lists are identical, else a short description of the first difference."""
    expected, found = list(expected), list(found)
    if expected == found:
        return None
    if len(expected) != len(found):
        expected_set, found_set = set(expected), set(found)
        missing = [name for name in expected if name not in found_set]
        extra = [name for name in found if name not in expected_set]
        return f"{len(found)} names for {len(expected)} columns; missing {missing[:5]}, extra {extra[:5]}"
    index = next(i for i, (a, b) in enumerate(zip(expected, found, strict=True)) if a != b)
    return f"position {index}: the bundle layout has {expected[index]!r} where the names list {found[index]!r}"


def _read_json(path: Path) -> dict[str, Any]:
    """Load a JSON object; missing file -> ``FileNotFoundError``, bad content -> ``BundleError``."""
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle file not found: {path}")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BundleError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(content, dict):
        raise BundleError(f"{path}: expected a JSON object, got {type(content).__name__}")
    return content


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


class PredictionPipeline:
    """Runs one genome through QC, known-AMR detection, unitig query, the models and the call logic.

    Args:
        models_dir: Directory holding ``manifest.json`` and the species bundles.
        configs_dir: Directory holding ``species.yaml``, ``drugs.yaml``, breakpoints, ...
        model_classes: Optional ``{model_class: class with .load(dir)}`` registry.
            Defaults to ``genome2mic.models.MODEL_CLASSES`` (imported lazily).
        unitig_query: Optional ``(kmers_npz, fasta) -> pattern vector`` callable.
            By default the bundle's ``models/<SPECIES>/unitig_kmers.npz`` is read once
            by :meth:`load` and each genome is queried with
            ``genome2mic.features.unitigs.query_kmer_set``.
        amr_detectors: Optional detector chain replacing the default
            (AMRFinderPlus CLI -> sidecar TSV -> ``markers.fasta``).
        amrfinder_tsv: Optional precomputed AMRFinderPlus TSV used for every run
            (the ``--amrfinder-tsv`` CLI option).
    """

    def __init__(
        self,
        models_dir: Path,
        configs_dir: Path,
        *,
        model_classes: Mapping[str, Any] | None = None,
        unitig_query: UnitigQuery | None = None,
        amr_detectors: Sequence[amr_detect.AmrDetector] | None = None,
        amrfinder_tsv: Path | None = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.configs_dir = Path(configs_dir)
        self._model_classes = model_classes
        self._unitig_query = unitig_query
        self._amr_detectors = list(amr_detectors) if amr_detectors is not None else None
        self._amrfinder_tsv = Path(amrfinder_tsv) if amrfinder_tsv is not None else None

        self.config: Config | None = None
        self.manifest: dict[str, Any] = {}
        self.reference_ids: tuple[str, ...] = ()
        self.reference_sketches: np.ndarray | None = None
        self.reference_k: int = sk.K
        self.species_bundles: dict[str, SpeciesBundle] = {}
        self.markers_fasta: Path | None = None
        self.feature_naming: str = FEATURE_NAMING_DEFAULT
        self._loaded = False

    # ------------------------------------------------------------------ load
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load every trained species x drug model into memory.

        Raises:
            FileNotFoundError: ``manifest.json``, the configs dir, the reference
                sketches or a required bundle file is missing.
            BundleError: a bundle file is malformed, unreadable or inconsistent
                (including a feature order that differs from the saved model's).
            ConfigError: the configs are malformed.
        """
        manifest_path = self.models_dir / MANIFEST_FILE
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
        if not self.configs_dir.is_dir():
            raise FileNotFoundError(f"Configs directory not found: {self.configs_dir}")
        manifest = _read_json(manifest_path)
        for key in ("model_version", "run_id", "species"):
            if key not in manifest:
                raise BundleError(f"{manifest_path}: missing key {key!r}")
        if not isinstance(manifest["species"], dict):
            raise BundleError(f"{manifest_path}: 'species' must map species key -> list of drugs")
        for key in ("model_version", "run_id"):
            if not isinstance(manifest[key], str) or not manifest[key].strip():
                raise BundleError(f"{manifest_path}: {key!r} must be a non-empty string")
        synthetic = manifest.get("synthetic", False)
        if not isinstance(synthetic, bool):
            raise BundleError(f"{manifest_path}: 'synthetic' must be true or false, got {synthetic!r}")
        feature_naming = manifest.get("feature_naming", FEATURE_NAMING_DEFAULT)
        if feature_naming not in (FEATURE_NAMING_DEFAULT, release_features.FEATURE_NAMING_NCBI_RELEASE):
            raise BundleError(f"{manifest_path}: unknown feature_naming {feature_naming!r}")
        self.feature_naming = feature_naming

        config = load_config(self.configs_dir)

        reference_path = self.models_dir / REFERENCE_SKETCHES_FILE
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference sketches not found: {reference_path}")
        reference_ids, reference_sketches, reference_k = _load_sketch_file(reference_path)
        unknown_refs = [key for key in reference_ids if key not in config.species]
        if unknown_refs:
            logger.warning("Reference sketches for species not in species.yaml are ignored: %s", unknown_refs)

        bundles: dict[str, SpeciesBundle] = {}
        for species, drugs in manifest["species"].items():
            if species not in config.species:
                raise BundleError(f"{manifest_path}: species {species!r} is not in species.yaml")
            if not isinstance(drugs, list):
                raise BundleError(f"{manifest_path}: species.{species} must be a list of drug names")
            if species not in reference_ids:
                logger.warning("Species %s has models but no reference sketch; it can never be identified", species)
            bundles[species] = self._load_species(species, drugs, config)

        # MarkerScan is an exact full-length substring match: it misses markers split
        # across contigs or carrying one SNP. Only a synthetic bundle may fall back to it.
        markers = self.models_dir / MARKERS_FILE
        markers_fasta: Path | None = None
        if markers.is_file() and synthetic:
            markers_fasta = markers
            logger.log(
                logging.INFO if self._amrfinder_tsv is not None else logging.WARNING,
                "Synthetic bundle: known-AMR detection falls back to the exact-match MarkerScan on %s "
                "when AMRFinderPlus and a precomputed TSV are both unavailable",
                markers,
            )
        elif markers.is_file():
            logger.warning(
                "%s is ignored: the MarkerScan fallback is only for synthetic bundles (manifest.json "
                "synthetic: true); this bundle needs AMRFinderPlus or a precomputed AMRFinderPlus TSV",
                markers,
            )

        self.markers_fasta = markers_fasta
        self.config = config
        self.manifest = manifest
        self.reference_ids = tuple(reference_ids)
        self.reference_sketches = reference_sketches
        self.reference_k = reference_k
        self.species_bundles = bundles
        self._loaded = True
        logger.info(
            "Loaded model bundle %s (run %s): %d species, %d drug model(s), %d reference sketch(es), "
            "synthetic=%s, MarkerScan fallback=%s",
            manifest["model_version"],
            manifest["run_id"],
            len(bundles),
            sum(len(b.drugs) for b in bundles.values()),
            len(reference_ids),
            synthetic,
            self.markers_fasta is not None,
        )

    def _load_species(self, species: str, drugs: Sequence[Any], config: Config) -> SpeciesBundle:
        directory = self.models_dir / species
        train_path = directory / TRAIN_SKETCHES_FILE
        if train_path.is_file():
            train_ids, train_sketches, train_k = _load_sketch_file(train_path)
        else:
            # An imported data release ships no assemblies, so its bundles carry no
            # training sketches: the distance is unknown and every call is reported
            # out of range (low confidence), never silently in range.
            logger.warning(
                "%s: %s not found (bundle trained on an imported release without Mash sketches); "
                "nearest_training_distance is null and calls are flagged low confidence",
                species, train_path,
            )
            train_ids, train_sketches, train_k = [], np.empty((0, 0), dtype=np.uint64), 0
        kmers = directory / UNITIG_KMERS_FILE
        unitig_kmers = kmers if kmers.is_file() else None

        loaded: dict[str, DrugBundle] = {}
        for raw_drug in drugs:
            if not isinstance(raw_drug, str) or not raw_drug.strip():
                raise BundleError(f"manifest: species.{species} contains a non-string drug entry {raw_drug!r}")
            drug = config.normalize_drug(raw_drug)
            if drug is None:
                raise BundleError(f"manifest: species.{species}: unknown drug {raw_drug!r}")
            if drug in loaded:
                raise BundleError(f"manifest: species.{species}: duplicate drug {drug!r}")
            loaded[drug] = self._load_drug(species, drug, directory / raw_drug, unitig_kmers is not None)

        feature_spec = None
        if self.feature_naming == release_features.FEATURE_NAMING_NCBI_RELEASE:
            spec_path = directory / release_features.FEATURE_SPEC_FILE
            if not spec_path.is_file():
                raise BundleError(
                    f"{spec_path} is missing: this bundle was trained on an imported NCBI release "
                    "(feature_naming: ncbi_release); write it with `genome2mic release-feature-spec --root ROOT`"
                )
            try:
                feature_spec = release_features.ReleaseFeatureSpec.load(spec_path)
            except (ValueError, json.JSONDecodeError) as error:
                raise BundleError(f"{spec_path}: {error}") from error
            logger.info(
                "%s: release feature rules (AMRFinderPlus DB %s, %d class symbols)",
                species, feature_spec.amrfinder_db_version, len(feature_spec.class_by_symbol),
            )
        kmer_set = None
        if unitig_kmers is not None and self._unitig_query is None and any(b.unitig_cols for b in loaded.values()):
            kmer_set = self._load_kmer_set(species, unitig_kmers, loaded)
        self._check_kmer_set_pairing(species, unitig_kmers, kmer_set, loaded)
        logger.info(
            "Loaded %s: %d drug model(s), %d training sketches, unitig k-mers=%s",
            species,
            len(loaded),
            len(train_ids),
            unitig_kmers is not None,
        )
        return SpeciesBundle(
            species=species,
            directory=directory,
            train_ids=tuple(train_ids),
            train_sketches=train_sketches,
            train_k=train_k,
            unitig_kmers=unitig_kmers,
            drugs=loaded,
            kmer_set=kmer_set,
            feature_spec=feature_spec,
        )

    @staticmethod
    def _load_kmer_set(species: str, path: Path, drugs: Mapping[str, DrugBundle]) -> Any:
        """Read the frozen k-mer set once and check it against the drug models that use it."""
        try:
            from genome2mic.features import unitigs  # noqa: PLC0415
        except ImportError as error:
            raise BundleError(
                "A model uses unitig features but genome2mic.features.unitigs is not importable"
            ) from error
        try:
            kmer_set = unitigs.load_kmer_set(path)
        except Exception as error:  # np.load / validation: ValueError, OSError, BadZipFile, ...
            raise BundleError(f"{path}: unreadable unitig k-mer set ({type(error).__name__}: {error})") from error
        if kmer_set.species and kmer_set.species != species:
            raise BundleError(f"{path}: k-mer set was built for {kmer_set.species!r}, not {species!r}")
        for drug, bundle in drugs.items():
            if bundle.unitig_cols and max(bundle.unitig_cols) >= kmer_set.n_patterns:
                raise BundleError(
                    f"{species} x {drug}: unitig column {max(bundle.unitig_cols)} is outside the "
                    f"{kmer_set.n_patterns}-pattern k-mer set {path}; bundle and k-mer set do not match"
                )
        logger.info(
            "Loaded %s unitig k-mer set once: %d k-mers in %d patterns (k=%d)",
            species,
            kmer_set.n_kmers,
            kmer_set.n_patterns,
            kmer_set.k,
        )
        return kmer_set

    @staticmethod
    def _check_kmer_set_pairing(
        species: str, path: Path | None, kmer_set: Any | None, drugs: Mapping[str, DrugBundle]
    ) -> None:
        """Every drug's recorded ``unitig_kmer_set_sha1`` must equal the species' set.

        The species' hash is the loaded set's content hash when it was read (default
        query), else the ``sha1`` stored in ``unitig_kmers.npz``. A drug without a
        recorded hash but with unitig columns predates the key: warning only. A
        recorded hash with no species set is not checked here (a drug that actually
        uses unitig columns already failed in :meth:`_load_drug`).
        """
        recorded = {drug: b.unitig_kmer_set_sha1 for drug, b in drugs.items() if b.unitig_kmer_set_sha1}
        unverified = sorted(drug for drug, b in drugs.items() if b.unitig_cols and not b.unitig_kmer_set_sha1)
        if unverified:
            logger.warning(
                "%s: %s record no unitig_kmer_set_sha1 (bundle written before it existed); cannot verify "
                "that their unitig columns index %s", species, unverified, path,
            )
        if not recorded or path is None:
            return
        if kmer_set is not None:
            species_sha1 = str(kmer_set.sha1())
        else:
            try:
                from genome2mic.features import unitigs  # noqa: PLC0415

                species_sha1 = unitigs.read_kmer_set_sha1(path)
            except Exception as error:  # np.load / zip / validation errors
                raise BundleError(f"{path}: unreadable unitig k-mer set ({type(error).__name__}: {error})") from error
        mismatched = {drug: sha for drug, sha in sorted(recorded.items()) if sha != species_sha1}
        if mismatched:
            detail = ", ".join(f"{drug} was trained on {sha[:12]}" for drug, sha in mismatched.items())
            raise BundleError(
                f"{species}: {path} is k-mer set {species_sha1[:12]}, but {detail}; their unitig columns would "
                "index a different pattern list (a partial retrain replaced the shared set). Retrain every drug "
                "of the species together."
            )
        logger.info("%s: %d drug model(s) match unitig k-mer set %s", species, len(recorded), species_sha1[:12])

    def _load_drug(self, species: str, drug: str, drug_dir: Path, has_unitig_kmers: bool) -> DrugBundle:
        features_path = drug_dir / FEATURES_FILE
        features = _read_json(features_path)
        conformal = _read_json(drug_dir / CONFORMAL_FILE)
        meta_path = drug_dir / META_FILE
        meta = _read_json(meta_path) if meta_path.is_file() else {}

        model_class = features.get("model_class")
        if not isinstance(model_class, str) or not model_class.strip():
            raise BundleError(f"{drug_dir / FEATURES_FILE}: 'model_class' must be a non-empty string")

        raw_columns = features.get("known_columns", [])
        if not isinstance(raw_columns, list) or not all(isinstance(c, str) for c in raw_columns):
            raise BundleError(f"{drug_dir / FEATURES_FILE}: 'known_columns' must be a list of strings")
        forbidden = sorted(set(raw_columns) & FORBIDDEN_FEATURES)
        if forbidden:
            raise BundleError(
                f"{drug_dir / FEATURES_FILE}: forbidden feature column(s) {forbidden} "
                "(lineage, geography, time and split columns are never features)"
            )
        if len(set(raw_columns)) != len(raw_columns):
            raise BundleError(f"{drug_dir / FEATURES_FILE}: duplicate known_columns")

        raw_unitigs = features.get("unitig_cols", [])
        unitig_cols = self._resolve_unitig_cols(raw_unitigs, drug_dir)
        expected_names = [*raw_columns, *_unitig_feature_names(raw_unitigs or [])]
        declared_names = features.get("feature_names")
        if declared_names is not None:
            if not isinstance(declared_names, list) or not all(isinstance(n, str) for n in declared_names):
                raise BundleError(f"{features_path}: 'feature_names' must be a list of strings")
            problem = _feature_order_problem(expected_names, declared_names)
            if problem is not None:
                raise BundleError(
                    f"{features_path}: known_columns + unitig_cols do not match feature_names ({problem}); "
                    "the model would read features in the wrong order"
                )
        if unitig_cols and not has_unitig_kmers:
            raise BundleError(
                f"{drug_dir}: model uses {len(unitig_cols)} unitig column(s) but "
                f"{drug_dir.parent / UNITIG_KMERS_FILE} is missing"
            )
        kmer_set_sha1 = features.get("unitig_kmer_set_sha1")
        if kmer_set_sha1 is not None and (not isinstance(kmer_set_sha1, str) or not kmer_set_sha1.strip()):
            raise BundleError(f"{features_path}: 'unitig_kmer_set_sha1' must be a non-empty string or null")

        class_by_column: dict[str, tuple[str | None, str | None]] = {}
        raw_classes = features.get("class_by_column", {}) or {}
        if not isinstance(raw_classes, dict):
            raise BundleError(f"{drug_dir / FEATURES_FILE}: 'class_by_column' must be an object")
        for column, pair in raw_classes.items():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise BundleError(f"{drug_dir / FEATURES_FILE}: class_by_column[{column!r}] must be [CLASS, SUBCLASS]")
            class_by_column[str(column)] = (
                str(pair[0]) if pair[0] is not None else None,
                str(pair[1]) if pair[1] is not None else None,
            )

        q_raw = conformal.get("q")
        if isinstance(q_raw, bool) or not isinstance(q_raw, (int, float)) or not math.isfinite(q_raw) or q_raw < 0:
            raise BundleError(f"{drug_dir / CONFORMAL_FILE}: 'q' must be a finite non-negative number of steps")
        q_low: float | None = None
        gate_open = True
        if "q_up" in conformal or "q_low" in conformal:
            # Tuned asymmetric band (train.TrainConfig.band == "asym_tuned").
            for key in ("q_up", "q_low"):
                v = conformal.get(key)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                    raise BundleError(f"{drug_dir / CONFORMAL_FILE}: {key!r} must be a finite non-negative number of steps")
            q_raw = float(conformal["q_up"])
            q_low = float(conformal["q_low"])
            gate_raw = conformal.get("active_gate_open")
            if not isinstance(gate_raw, bool):
                raise BundleError(f"{drug_dir / CONFORMAL_FILE}: 'active_gate_open' must be true or false")
            gate_open = gate_raw
        caps: tuple[float, float] | None = None
        cap_raw = (conformal.get("cap_low_log2"), conformal.get("cap_high_log2"))
        if cap_raw != (None, None):
            if not all(isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in cap_raw):
                raise BundleError(f"{drug_dir / CONFORMAL_FILE}: cap_low_log2 / cap_high_log2 must both be finite numbers")
            if float(cap_raw[0]) > float(cap_raw[1]):  # type: ignore[arg-type]
                raise BundleError(f"{drug_dir / CONFORMAL_FILE}: cap_low_log2 > cap_high_log2")
            caps = (float(cap_raw[0]), float(cap_raw[1]))  # type: ignore[arg-type]

        model_cls = self._resolve_model_class(model_class)
        try:
            model = model_cls.load(drug_dir)
        except (FileNotFoundError, BundleError):
            raise
        except Exception as error:  # XGBoostError, KeyError, JSONDecodeError, ValueError, ...
            raise BundleError(
                f"{drug_dir}: cannot load the {model_class!r} model ({type(error).__name__}: {error})"
            ) from error
        if not hasattr(model, "predict_log2"):
            raise BundleError(f"{drug_dir}: model class {model_class!r} has no predict_log2 method")
        model_names = getattr(model, "feature_names_", None)
        if model_names is not None:
            problem = _feature_order_problem(expected_names, [str(name) for name in model_names])
            if problem is not None:
                raise BundleError(
                    f"{drug_dir}: features.json known_columns + unitig_cols do not match the saved "
                    f"{model_class!r} model's feature_names_ ({problem}); the model would read features "
                    "in the wrong order"
                )
        elif declared_names is None:
            logger.warning(
                "%s: neither features.json nor the %s model records feature names; the feature order "
                "cannot be verified",
                drug_dir,
                model_class,
            )
        logger.info(
            "Loaded %s x %s: %s with %d known-AMR + %d unitig feature(s), q=%.3g",
            species,
            drug,
            model_class,
            len(raw_columns),
            len(unitig_cols),
            float(q_raw),
        )
        return DrugBundle(
            species=species,
            drug=drug,
            model_class=model_class,
            model=model,
            known_columns=tuple(raw_columns),
            unitig_cols=unitig_cols,
            class_by_column=class_by_column,
            q=float(q_raw),
            q_low=q_low,
            active_gate_open=gate_open,
            meta=meta,
            unitig_kmer_set_sha1=kmer_set_sha1.strip() if isinstance(kmer_set_sha1, str) else None,
            caps=caps,
        )

    def _resolve_unitig_cols(self, raw: Any, drug_dir: Path) -> tuple[int, ...]:
        """``unitig_cols`` as pattern column indices (ints, or ``u_000017``-style ids)."""
        if raw is None:
            return ()
        if not isinstance(raw, list):
            raise BundleError(f"{drug_dir / FEATURES_FILE}: 'unitig_cols' must be a list")
        if not raw:
            return ()
        pattern_to_col: dict[str, int] | None = None
        out: list[int] = []
        for item in raw:
            if isinstance(item, bool):
                raise BundleError(f"{drug_dir / FEATURES_FILE}: unitig_cols contains a boolean")
            if isinstance(item, int):
                out.append(item)
                continue
            if isinstance(item, str):
                if pattern_to_col is None:
                    pattern_to_col = self._pattern_index(drug_dir.parent)
                if item in pattern_to_col:
                    out.append(pattern_to_col[item])
                    continue
                digits = item.rsplit("_", 1)[-1]
                if digits.isdigit():
                    out.append(int(digits))
                    continue
            raise BundleError(f"{drug_dir / FEATURES_FILE}: cannot resolve unitig column {item!r}")
        if any(col < 0 for col in out):
            raise BundleError(f"{drug_dir / FEATURES_FILE}: negative unitig column index")
        return tuple(out)

    @staticmethod
    def _pattern_index(species_dir: Path) -> dict[str, int]:
        """``pattern_id -> col_index`` from the bundle's ``unitig_index.parquet`` (empty if absent)."""
        path = species_dir / UNITIG_INDEX_FILE
        if not path.is_file():
            return {}
        import pandas as pd  # noqa: PLC0415  (only needed for string unitig ids)

        index = pd.read_parquet(path, columns=["col_index", "pattern_id"])
        return {str(pid): int(col) for col, pid in zip(index["col_index"], index["pattern_id"], strict=True)}

    def _resolve_model_class(self, name: str) -> Any:
        registry = self._model_classes
        if registry is None:
            try:
                from genome2mic.models import MODEL_CLASSES as registry  # noqa: PLC0415
            except ImportError as error:
                raise BundleError(
                    f"Cannot load model class {name!r}: genome2mic.models is not importable ({error})"
                ) from error
        if name not in registry:
            raise BundleError(f"Unknown model_class {name!r}; known classes: {sorted(registry)}")
        return registry[name]

    def available_models(self) -> dict[str, list[str]]:
        """Return the trained drugs for each species key (loads the bundle if needed)."""
        if not self._loaded:
            self.load()
        return {species: sorted(bundle.drugs) for species, bundle in self.species_bundles.items()}

    # ------------------------------------------------------------------- run
    def run(self, fasta_path: Path, sample_id: str) -> dict[str, Any]:
        """Return one report as a dict that matches ``DATA_CONTRACT.md`` stage 12.

        The dict validates against ``genome2mic.api.schemas.PredictionReport``.
        """
        if not self._loaded:
            self.load()
        assert self.config is not None and self.reference_sketches is not None
        config = self.config
        fasta = Path(fasta_path)
        if not fasta.is_file():
            raise FileNotFoundError(f"Genome FASTA not found: {fasta}")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")

        droplog = DropLog("predict")
        seqs = [seq for _, seq in read_fasta(fasta)]
        stats = assembly_stats(seqs)
        sketch_cache: dict[tuple[int, int], np.ndarray] = {}

        def genome_sketch(k: int, s: int) -> np.ndarray:
            key = (k, s)
            if key not in sketch_cache:
                sketch_cache[key] = sk.sketch(seqs, k=k, s=s, droplog=droplog)
            return sketch_cache[key]

        qc = self._qc(stats, genome_sketch)
        logger.info(
            "Sample %s: %d contig(s), %d bp, N50 %d, species=%s (reference distance %s), qc_pass=%s%s",
            sample_id,
            stats.n_contigs,
            stats.total_length,
            stats.n50,
            qc.species,
            f"{qc.reference_distance:.4g}" if qc.reference_distance is not None else "n/a",
            qc.qc_pass,
            f" [{'; '.join(qc.fail_reasons)}]" if qc.fail_reasons else "",
        )

        report: dict[str, Any] = {
            "sample_id": sample_id,
            "species": qc.species,
            "qc_pass": qc.qc_pass,
            "nearest_training_distance": None,
            "in_range": False,
            "predictions": [],
            "ranked_active": [],
            "model_version": str(self.manifest["model_version"]),
            "run_id": str(self.manifest["run_id"]),
            "disclaimer": DISCLAIMER,
        }
        if qc.species is None:
            logger.warning("Sample %s: species not identified; no predictions", sample_id)
            return report

        species = qc.species
        bundle = self.species_bundles.get(species)
        if bundle is None:
            logger.warning("Sample %s: species %s has no trained models; natural-resistance rules only", sample_id, species)
            predictions = [
                self._natural_resistance_prediction(species, drug)
                for drug in config.drug_names()
                if config.is_naturally_resistant(species, drug)
            ]
            report["predictions"] = predictions
            report["ranked_active"] = rank.rank_active(predictions, config)
            return report

        nearest: float | None
        if bundle.train_sketches.size == 0:
            nearest, in_range = None, False
            logger.warning("Sample %s: no training sketches in the %s bundle; distance unknown, calls are low confidence", sample_id, species)
        else:
            train_query = genome_sketch(bundle.train_k, bundle.train_sketches.shape[1])
            nearest = float(np.min(sk.distances_to(train_query, bundle.train_sketches, bundle.train_k)))
            in_range = nearest <= config.qc_max_mash_distance
            logger.info(
                "Sample %s: nearest training genome at Mash distance %.4g (%s)",
                sample_id,
                nearest,
                "in range" if in_range else "out of range: calls are low confidence",
            )

        if bundle.feature_spec is not None:
            # Imported NCBI release: every Type == AMR row, converted with the release rules.
            detections = amr_detect.detect(
                fasta,
                species,
                config,
                amrfinder_tsv=self._amrfinder_tsv,
                markers_fasta=self.markers_fasta,
                detectors=self._amr_detectors,
                droplog=droplog,
                parser=release_features.read_amrfinder_rows,
            )
            row = release_features.release_known_amr_row(detections, bundle.feature_spec)
        else:
            detections = amr_detect.detect(
                fasta,
                species,
                config,
                amrfinder_tsv=self._amrfinder_tsv,
                markers_fasta=self.markers_fasta,
                detectors=self._amr_detectors,
                droplog=droplog,
            )
            row = amr_detect.known_amr_row(detections, config)
        pattern_vector = self._unitig_vector(bundle, fasta) if bundle.needs_unitigs else None

        predictions: list[dict[str, Any]] = []
        for drug in config.drug_names():
            if config.is_naturally_resistant(species, drug):
                predictions.append(self._natural_resistance_prediction(species, drug))
            elif drug in bundle.drugs:
                predictions.append(self._predict_drug(bundle.drugs[drug], row, pattern_vector, droplog))

        report["nearest_training_distance"] = nearest
        report["in_range"] = bool(in_range)
        report["predictions"] = predictions
        report["ranked_active"] = rank.rank_active(predictions, config)
        counts = {call: sum(p["call"] == call for p in predictions) for call in (
            rank.CALL_LIKELY_ACTIVE, rank.CALL_UNCERTAIN, rank.CALL_LIKELY_INACTIVE)}
        logger.info("Sample %s: %d drug(s) called: %s", sample_id, len(predictions), counts)
        return report

    # --------------------------------------------------------------- helpers
    def _qc(self, stats: AssemblyStats, genome_sketch: Callable[[int, int], np.ndarray]) -> GenomeQc:
        """Species from the nearest reference sketch plus the contract's QC rules."""
        assert self.config is not None and self.reference_sketches is not None
        config = self.config
        fail: list[str] = []
        if stats.n_contigs == 0 or stats.total_length == 0:
            return GenomeQc(None, None, stats, (QC_EMPTY_ASSEMBLY,))
        if stats.n_contigs > config.qc_max_contigs:
            fail.append(QC_TOO_FRAGMENTED)

        query = genome_sketch(self.reference_k, self.reference_sketches.shape[1])
        if query.size == 0:
            fail.append(QC_TOO_DISTANT)
            return GenomeQc(None, None, stats, tuple(fail))
        distances = sk.distances_to(query, self.reference_sketches, self.reference_k)
        nearest_index = int(np.argmin(distances))
        distance = float(distances[nearest_index])
        species: str | None = self.reference_ids[nearest_index]
        if distance > config.qc_max_mash_distance:
            fail.append(QC_TOO_DISTANT)
            species = None
        elif species not in config.species:
            fail.append(QC_UNKNOWN_REFERENCE)
            species = None
        else:
            expected = config.species[species].expected_genome_size
            tolerance = config.species[species].size_tolerance
            if abs(stats.total_length - expected) > tolerance * expected:
                fail.append(QC_WRONG_GENOME_SIZE)
        return GenomeQc(species, distance, stats, tuple(fail))

    def _unitig_vector(self, bundle: SpeciesBundle, fasta: Path) -> np.ndarray:
        if bundle.unitig_kmers is None:
            raise BundleError(f"{bundle.species}: a drug model needs unitig features but {UNITIG_KMERS_FILE} is missing")
        if self._unitig_query is not None:
            raw = self._unitig_query(bundle.unitig_kmers, fasta)
        else:
            if bundle.kmer_set is None:
                raise BundleError(f"{bundle.species}: the unitig k-mer set was not loaded; call load() first")
            raw = self._query_kmer_set(bundle.kmer_set, fasta)
        vector = np.asarray(raw).ravel()
        logger.info("Unitig query for %s: %d pattern(s), %d present", bundle.species, vector.size, int(np.count_nonzero(vector)))
        return vector

    @staticmethod
    def _query_kmer_set(kmer_set: Any, fasta: Path) -> np.ndarray:
        """``features.unitigs.query_kmer_set`` against the bundle's frozen k-mer set.

        The set was read once from ``models/<SPECIES>/unitig_kmers.npz`` at load (never
        per request); prediction never touches ``data/processed``.
        """
        from genome2mic.features import unitigs  # noqa: PLC0415  (import checked at load)

        return np.asarray(unitigs.query_kmer_set(kmer_set, fasta))

    def _natural_resistance_prediction(self, species: str, drug: str) -> dict[str, Any]:
        """Override 1: intrinsic resistance; the model is skipped and MIC fields are null."""
        assert self.config is not None
        bp = self.config.call_breakpoint(species, drug)
        return {
            "drug": drug,
            "pred_mic": None,
            "band_low": None,
            "band_high": None,
            "s_breakpoint": bp.s_breakpoint if bp else None,
            "r_breakpoint": bp.r_breakpoint if bp else None,
            "call": rank.CALL_LIKELY_INACTIVE,
            "margin_steps": None,
            "reasons": [rank.NATURAL_RESISTANCE_REASON],
            "override": rank.OVERRIDE_NATURAL_RESISTANCE,
        }

    def _predict_drug(
        self,
        bundle: DrugBundle,
        row: amr_detect.KnownAmrRow,
        pattern_vector: np.ndarray | None,
        droplog: DropLog,
    ) -> dict[str, Any]:
        """Model prediction, band, call and the strong-marker override for one drug."""
        assert self.config is not None
        config = self.config
        parts = [amr_detect.feature_vector(row, bundle.known_columns, droplog)]
        if bundle.unitig_cols:
            if pattern_vector is None:
                raise BundleError(f"{bundle.species} x {bundle.drug}: unitig features requested but no pattern vector")
            index = np.asarray(bundle.unitig_cols, dtype=np.int64)
            if index.max() >= pattern_vector.size:
                raise BundleError(
                    f"{bundle.species} x {bundle.drug}: unitig column {int(index.max())} is outside the "
                    f"{pattern_vector.size}-pattern query vector; bundle and k-mer set do not match"
                )
            parts.append(pattern_vector[index].astype(np.float32))
        X = np.concatenate(parts).reshape(1, -1)

        raw = np.asarray(bundle.model.predict_log2(X), dtype=np.float64).ravel()
        if raw.size != 1 or not np.isfinite(raw[0]):
            raise BundleError(f"{bundle.species} x {bundle.drug}: model returned {raw!r} for one genome")
        pred_log2 = float(raw[0])
        if bundle.caps is not None:
            # Panel-edge caps from training (all train rows): same clip as the training preds.
            pred_log2 = min(max(pred_log2, bundle.caps[0]), bundle.caps[1])
        clipped = min(max(pred_log2, float(GRID_MIN_EXPONENT)), float(GRID_MAX_EXPONENT))
        if clipped != pred_log2:
            logger.warning(
                "%s x %s: log2 MIC prediction %.3g clipped to the reference grid edge %.3g",
                bundle.species, bundle.drug, pred_log2, clipped,
            )
        pred_mic = round_up_to_step(2.0**clipped)
        if bundle.q_low is None:
            band_low, band_high = rank.conformal_band(pred_mic, bundle.q)
        else:
            band_low, band_high = rank.conformal_band_asym(pred_mic, bundle.q, bundle.q_low)

        bp = config.call_breakpoint(bundle.species, bundle.drug)
        if bp is None:
            logger.warning("%s x %s: no %s breakpoint; call is uncertain", bundle.species, bundle.drug, config.call_standard)
        call, margin = rank.call_from_band(band_low, band_high, bp)
        gated = False
        if call == rank.CALL_LIKELY_ACTIVE and not bundle.active_gate_open:
            # Same gate as training (conformal.gate_calls): the pair never certified call VME.
            call, margin, gated = rank.CALL_UNCERTAIN, None, True

        override: str | None = None
        hits = rank.strong_marker_hits(
            row.symbols_by_column, config.drugs.get(bundle.drug), row.markers,
            intrinsic_symbols=config.intrinsic_symbols(bundle.species),
        )
        if hits:
            call, margin, override = rank.CALL_LIKELY_INACTIVE, None, rank.OVERRIDE_STRONG_MARKER
            reasons = hits
            logger.info("%s x %s: strong marker override by %s", bundle.species, bundle.drug, hits)
        else:
            reasons = rank.reasons_for(bundle.drug, row.markers, bundle.class_by_column)
            if gated:
                reasons = [*reasons, rank.ACTIVE_GATE_REASON]

        logger.info(
            "%s x %s: pred MIC %g mg/L (band %g-%g), %s%s",
            bundle.species, bundle.drug, pred_mic, band_low, band_high, call,
            f", margin {margin} step(s)" if margin is not None else "",
        )
        return {
            "drug": bundle.drug,
            "pred_mic": pred_mic,
            "band_low": band_low,
            "band_high": band_high,
            "s_breakpoint": bp.s_breakpoint if bp else None,
            "r_breakpoint": bp.r_breakpoint if bp else None,
            "call": call,
            "margin_steps": margin,
            "reasons": reasons,
            "override": override,
        }
