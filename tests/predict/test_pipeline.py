"""End-to-end tests for ``genome2mic.predict.pipeline`` on a tiny FAKE model bundle.

Everything here is synthetic: random DNA, planted marker sequences and a linear
test double standing in for a trained MIC model. No number in this file is a real
MIC. The bundle follows the layout ``models/train.py`` writes (see DESIGN.md) so the
pipeline code under test is exactly what serves real bundles.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from genome2mic import sketch as sk
from genome2mic.api.constants import DISCLAIMER
from genome2mic.api.schemas.prediction_report import PredictionReport
from genome2mic.config import load_config
from genome2mic.droplog import DropLog
from genome2mic.errors import ToolNotAvailable
from genome2mic.features import unitigs as unitig_mod
from genome2mic.io import write_fasta
from genome2mic.models import MODEL_CLASSES, B1Lookup, XgbAft
from genome2mic.predict import amr_detect
from genome2mic.predict.pipeline import (
    FORBIDDEN_FEATURES,
    BundleError,
    PredictionPipeline,
    assembly_stats,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"

GENOME_LEN = 6000
MAX_CONTIGS = 10
SPECIES = "KPNEU"
MODEL_VERSION = "FAKE-TEST-0.1"
RUN_ID = "faketest0001"
FAKE_MODEL_CLASS = "fake_linear"
N_PATTERNS = 10
UNITIG_PRESENT_COL = 3

_BASES = np.frombuffer(b"ACGT", dtype=np.uint8)

# symbol -> (markers.fasta header, sequence length). blaSHV-11 has no class on purpose
# so the class_by_column fallback from features.json is exercised.
MARKERS: dict[str, tuple[str, int]] = {
    "blaKPC-2": ("blaKPC-2 BETA-LACTAM CARBAPENEM AMR", 300),
    "gyrA_S83L": ("gyrA_S83L QUINOLONE QUINOLONE POINT", 150),
    "blaCTX-M-15": ("blaCTX-M-15 BETA-LACTAM CEPHALOSPORIN AMR", 300),
    "aac(6')-Ib-cr": ("aac(6')-Ib-cr AMINOGLYCOSIDE/QUINOLONE AMINOGLYCOSIDE/QUINOLONE AMR", 200),
    "blaSHV-11": ("blaSHV-11", 200),
}

# Fake per-drug models: known columns, their log2 weights, intercept, unitig columns + weights.
# Chosen so the calls below are easy to verify by hand against the EUCAST 2024 breakpoints.
FAKE_MODELS: dict[str, dict[str, object]] = {
    "meropenem": {
        "known": {"gene_blakpc_2": 6.0, "gene_blandm_1": 6.0, "gene_blaoxa_48": 5.0, "point_ompk36_d135dgd": 1.0, "n_class_beta_lactam": 0.0},
        "intercept": -4.0,
        "unitigs": {},
        "classes": {"gene_blakpc_2": ["BETA-LACTAM", "CARBAPENEM"], "gene_blandm_1": ["BETA-LACTAM", "CARBAPENEM"]},
    },
    "ceftriaxone": {
        "known": {"gene_blactx_m": 5.0, "gene_blakpc_2": 5.0, "gene_blashv": 0.0, "n_class_beta_lactam": 0.0},
        "intercept": -3.0,
        "unitigs": {},
        "classes": {"gene_blactx_m": ["BETA-LACTAM", "CEPHALOSPORIN"], "gene_blashv": ["BETA-LACTAM", "BETA-LACTAM"]},
    },
    "ciprofloxacin": {
        "known": {"point_gyra_s83l": 4.0, "point_parc_s80i": 2.0, "gene_qnrb": 2.0, "gene_aac_6_ib_cr": 1.0, "n_class_quinolone": 0.0},
        "intercept": -5.0,
        "unitigs": {UNITIG_PRESENT_COL: 2.0, 7: 0.0},
        "classes": {"point_gyra_s83l": ["QUINOLONE", "QUINOLONE"]},
    },
    "gentamicin": {
        "known": {"gene_aac_6_ib_cr": 2.0, "gene_aac_3": 3.0, "n_class_aminoglycoside": 0.0},
        "intercept": -1.0,
        "unitigs": {},
        "classes": {},
    },
    "piperacillin-tazobactam": {
        "known": {"gene_blakpc_2": 2.0, "gene_blactx_m": 1.0},
        "intercept": 0.0,
        "unitigs": {},
        "classes": {},
    },
    # Natural resistance for KPNEU: the model must be skipped even though it exists.
    "ampicillin": {"known": {"gene_blatem": 3.0}, "intercept": 2.0, "unitigs": {}, "classes": {}},
}


# --------------------------------------------------------------------------- #
# Test doubles and DNA helpers
# --------------------------------------------------------------------------- #


class FakeLinearModel:
    """MicModel test double: ``pred_log2 = intercept + X @ weights``. FAKE, not a trained model."""

    name = FAKE_MODEL_CLASS
    FILE = "fake_linear.json"

    def __init__(self, intercept: float, weights: Sequence[float]) -> None:
        self.intercept = float(intercept)
        self.weights = np.asarray(weights, dtype=np.float64)

    def predict_log2(self, X: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(np.asarray(X, dtype=np.float64))
        assert matrix.shape[1] == self.weights.size, f"expected {self.weights.size} features, got {matrix.shape[1]}"
        return self.intercept + matrix @ self.weights

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / self.FILE).write_text(json.dumps({"intercept": self.intercept, "weights": self.weights.tolist()}))

    @classmethod
    def load(cls, directory: Path) -> FakeLinearModel:
        payload = json.loads((Path(directory) / cls.FILE).read_text())
        return cls(payload["intercept"], payload["weights"])


class RecordingUnitigQuery:
    """Stand-in for ``features.unitigs.query_genome``: records calls, returns a fixed vector."""

    def __init__(self, present: Sequence[int] = (UNITIG_PRESENT_COL,)) -> None:
        self.present = tuple(present)
        self.calls: list[tuple[Path, Path]] = []

    def __call__(self, kmers_path: Path, fasta: Path) -> np.ndarray:
        self.calls.append((Path(kmers_path), Path(fasta)))
        vector = np.zeros(N_PATTERNS, dtype=np.int8)
        vector[list(self.present)] = 1
        return vector


def random_dna(rng: np.random.Generator, length: int) -> str:
    return _BASES[rng.integers(0, 4, size=length)].tobytes().decode("ascii")


def mutate(seq: str, n_snps: int, rng: np.random.Generator) -> str:
    out = bytearray(seq, "ascii")
    for position in rng.choice(len(seq), size=n_snps, replace=False):
        choices = [b for b in b"ACGT" if b != out[position]]
        out[position] = choices[rng.integers(0, 3)]
    return out.decode("ascii")


def insert(seq: str, pieces: dict[int, str]) -> str:
    out = seq
    for position in sorted(pieces, reverse=True):
        out = out[:position] + pieces[position] + out[position:]
    return out


def split_contigs(seq: str, n_contigs: int) -> list[tuple[str, str]]:
    size = -(-len(seq) // n_contigs)
    return [(f"contig_{i + 1}", seq[i * size : (i + 1) * size]) for i in range(n_contigs)]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def dna() -> dict[str, str]:
    """Reference genomes per species plus the planted marker sequences (seeded)."""
    rng = np.random.default_rng(7)
    refs = {key: random_dna(rng, GENOME_LEN) for key in ("KPNEU", "ECOLI", "PAER")}
    markers = {symbol: random_dna(rng, length) for symbol, (_, length) in MARKERS.items()}
    return {**{f"ref_{k}": v for k, v in refs.items()}, **{f"marker_{k}": v for k, v in markers.items()}}


@pytest.fixture
def configs_dir(tmp_path: Path) -> Path:
    """Real configs with genome sizes and contig limit scaled to the tiny synthetic genomes."""
    target = tmp_path / "configs"
    shutil.copytree(CONFIGS_DIR, target)
    species = yaml.safe_load((target / "species.yaml").read_text())
    for entry in species["species"].values():
        entry["expected_genome_size"] = GENOME_LEN
    species["qc"]["max_contigs"] = MAX_CONTIGS
    (target / "species.yaml").write_text(yaml.safe_dump(species))
    # The hand-checked calls and margins below were derived against EUCAST 2024; the
    # project call standard is CLSI 2024 (configs/drugs.yaml), so the copy pins EUCAST
    # to keep these pipeline-mechanics tests independent of that choice.
    drugs_yaml = target / "drugs.yaml"
    text = drugs_yaml.read_text()
    assert "standard: CLSI" in text
    drugs_yaml.write_text(text.replace("standard: CLSI", "standard: EUCAST", 1))
    return target


def write_bundle(
    models_dir: Path, dna: dict[str, str], *, with_markers: bool = True, synthetic: bool | None = True
) -> Path:
    """Write a FAKE model bundle in the ``models/train.py`` layout.

    ``synthetic`` is the manifest flag that allows the MarkerScan fallback (``None``
    leaves the key out, like a bundle written before the flag existed).
    """
    rng = np.random.default_rng(11)
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "run_id": RUN_ID,
        "created": "2026-10-03T00:00:00Z",
        "species": {SPECIES: list(FAKE_MODELS)},
        "note": "FAKE synthetic test bundle",
    }
    if synthetic is not None:
        manifest["synthetic"] = synthetic
    (models_dir / "manifest.json").write_text(json.dumps(manifest))
    ref_keys = ["KPNEU", "ECOLI", "PAER"]
    sk.save_sketches(models_dir / "reference_sketches.npz", ref_keys, [sk.sketch(dna[f"ref_{k}"]) for k in ref_keys])

    species_dir = models_dir / SPECIES
    train = [mutate(dna["ref_KPNEU"], 8, rng) for _ in range(4)]
    sk.save_sketches(species_dir / "train_sketches.npz", [f"train_{i}" for i in range(4)], [sk.sketch(g) for g in train])
    with (species_dir / "unitig_kmers.npz").open("wb") as handle:
        np.savez(handle, kmers=np.zeros(0, dtype=np.uint64), pattern_col=np.zeros(0, dtype=np.int64))

    if with_markers:
        write_fasta([(MARKERS[s][0], dna[f"marker_{s}"]) for s in MARKERS], models_dir / "markers.fasta")

    for drug, spec in FAKE_MODELS.items():
        known: dict[str, float] = spec["known"]  # type: ignore[assignment]
        unitigs: dict[int, float] = spec["unitigs"]  # type: ignore[assignment]
        drug_dir = species_dir / drug
        FakeLinearModel(spec["intercept"], list(known.values()) + list(unitigs.values())).save(drug_dir)  # type: ignore[arg-type]
        (drug_dir / "features.json").write_text(
            json.dumps(
                {
                    "model_class": FAKE_MODEL_CLASS,
                    "known_columns": list(known),
                    "unitig_cols": list(unitigs),
                    "class_by_column": spec["classes"],
                }
            )
        )
        (drug_dir / "conformal.json").write_text(json.dumps({"q": 1.0}))
        (drug_dir / "meta.json").write_text(json.dumps({"synthetic": True, "species": SPECIES, "drug": drug}))
    return models_dir


@pytest.fixture
def models_dir(tmp_path: Path, dna: dict[str, str]) -> Path:
    return write_bundle(tmp_path / "models", dna)


@pytest.fixture
def unitig_query() -> RecordingUnitigQuery:
    return RecordingUnitigQuery()


@pytest.fixture
def pipeline(models_dir: Path, configs_dir: Path, unitig_query: RecordingUnitigQuery) -> PredictionPipeline:
    pipe = PredictionPipeline(
        models_dir=models_dir,
        configs_dir=configs_dir,
        model_classes={FAKE_MODEL_CLASS: FakeLinearModel},
        unitig_query=unitig_query,
    )
    pipe.load()
    return pipe


@pytest.fixture
def genomes(tmp_path: Path, dna: dict[str, str]) -> dict[str, Path]:
    """Query genomes: KPC+gyrA carrier, a clean KPNEU, a far genome, a fragmented one, a PAER."""
    rng = np.random.default_rng(23)
    folder = tmp_path / "genomes"
    base = mutate(dna["ref_KPNEU"], 6, rng)
    # Insert positions are chosen away from the 2-contig split point (~3325) so no marker is cut.
    kpc = insert(base, {1000: dna["marker_blaKPC-2"], 2000: dna["marker_gyrA_S83L"], 4500: dna["marker_blaSHV-11"]})
    clean = insert(mutate(dna["ref_KPNEU"], 6, rng), {4500: dna["marker_blaSHV-11"]})
    out = {
        "kpc": write_fasta(split_contigs(kpc, 2), folder / "kpc.fasta"),
        "clean": write_fasta(split_contigs(clean, 3), folder / "clean.fasta"),
        "far": write_fasta(split_contigs(random_dna(rng, GENOME_LEN), 2), folder / "far.fasta"),
        "fragmented": write_fasta(split_contigs(clean, MAX_CONTIGS + 2), folder / "fragmented.fasta"),
        "paer": write_fasta(split_contigs(mutate(dna["ref_PAER"], 5, rng), 2), folder / "paer.fasta"),
    }
    return out


def by_drug(report: dict) -> dict[str, dict]:
    return {p["drug"]: p for p in report["predictions"]}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_load_raises_file_not_found_without_manifest(tmp_path: Path, configs_dir: Path) -> None:
    pipe = PredictionPipeline(models_dir=tmp_path / "missing_models", configs_dir=configs_dir)
    with pytest.raises(FileNotFoundError):
        pipe.load()


def test_load_raises_file_not_found_without_reference_sketches(models_dir: Path, configs_dir: Path) -> None:
    (models_dir / "reference_sketches.npz").unlink()
    pipe = PredictionPipeline(models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel})
    with pytest.raises(FileNotFoundError):
        pipe.load()


def test_bundle_without_train_sketches_reports_unknown_distance_as_low_confidence(
    models_dir: Path, configs_dir: Path, genomes: dict[str, Path]
) -> None:
    """A bundle trained on an imported release ships no training sketches: it loads, the
    distance is null and the report is never ``in_range`` (calls flagged low confidence)."""
    (models_dir / SPECIES / "train_sketches.npz").unlink()
    pipe = PredictionPipeline(
        models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel}, unitig_query=RecordingUnitigQuery()
    )
    report = pipe.run(genomes["clean"], "FAKE-NO-TRAIN-SKETCHES")
    PredictionReport.model_validate(report)
    assert report["species"] == SPECIES
    assert report["nearest_training_distance"] is None
    assert report["in_range"] is False
    assert report["predictions"]


def test_available_models_lists_manifest_drugs(pipeline: PredictionPipeline) -> None:
    assert pipeline.available_models() == {SPECIES: sorted(FAKE_MODELS)}


def test_load_rejects_unknown_model_class(models_dir: Path, configs_dir: Path) -> None:
    pipe = PredictionPipeline(models_dir, configs_dir, model_classes={"other": FakeLinearModel})
    with pytest.raises(BundleError, match="fake_linear"):
        pipe.load()


@pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_FEATURES))
def test_load_rejects_forbidden_feature_columns(models_dir: Path, configs_dir: Path, forbidden: str) -> None:
    features_path = models_dir / SPECIES / "meropenem" / "features.json"
    features = json.loads(features_path.read_text())
    features["known_columns"].append(forbidden)
    features_path.write_text(json.dumps(features))
    pipe = PredictionPipeline(models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel})
    with pytest.raises(BundleError, match=forbidden):
        pipe.load()


def test_load_rejects_unitig_model_without_kmer_set(models_dir: Path, configs_dir: Path) -> None:
    (models_dir / SPECIES / "unitig_kmers.npz").unlink()
    pipe = PredictionPipeline(models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel})
    with pytest.raises(BundleError, match="unitig"):
        pipe.load()


def test_load_resolves_string_unitig_ids_via_index(models_dir: Path, configs_dir: Path) -> None:
    pd.DataFrame({"col_index": [3, 7], "pattern_id": ["u_000003", "u_000007"]}).to_parquet(
        models_dir / SPECIES / "unitig_index.parquet", index=False
    )
    features_path = models_dir / SPECIES / "ciprofloxacin" / "features.json"
    features = json.loads(features_path.read_text())
    features["unitig_cols"] = ["u_000003", "u_000007"]
    features_path.write_text(json.dumps(features))
    # The FAKE unitig_kmers.npz is not a real k-mer set; the injected query keeps load() from reading it.
    pipe = PredictionPipeline(
        models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel}, unitig_query=RecordingUnitigQuery()
    )
    pipe.load()
    assert pipe.species_bundles[SPECIES].drugs["ciprofloxacin"].unitig_cols == (3, 7)


# --------------------------------------------------------------------------- #
# Feature layout checks at load (features.json vs the saved model)
# --------------------------------------------------------------------------- #


def _edit_features(models_dir: Path, drug: str, **updates: object) -> None:
    path = models_dir / SPECIES / drug / "features.json"
    features = json.loads(path.read_text())
    features.update(updates)
    path.write_text(json.dumps(features))


def _features(models_dir: Path, drug: str) -> dict:
    return json.loads((models_dir / SPECIES / drug / "features.json").read_text())


def test_load_accepts_feature_names_that_match_the_layout(models_dir: Path, configs_dir: Path) -> None:
    for drug in FAKE_MODELS:
        features = _features(models_dir, drug)
        names = features["known_columns"] + [f"u_{int(c):06d}" for c in features["unitig_cols"]]
        _edit_features(models_dir, drug, feature_names=names)
    pipe = PredictionPipeline(
        models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel}, unitig_query=RecordingUnitigQuery()
    )
    pipe.load()
    assert pipe.available_models() == {SPECIES: sorted(FAKE_MODELS)}


def test_load_rejects_reordered_feature_names(models_dir: Path, configs_dir: Path) -> None:
    known = _features(models_dir, "meropenem")["known_columns"]
    _edit_features(models_dir, "meropenem", feature_names=list(reversed(known)))
    pipe = PredictionPipeline(models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel})
    with pytest.raises(BundleError, match="feature_names"):
        pipe.load()


def test_load_rejects_feature_names_with_a_substituted_unitig(models_dir: Path, configs_dir: Path) -> None:
    features = _features(models_dir, "ciprofloxacin")  # unitig_cols [3, 7]
    _edit_features(models_dir, "ciprofloxacin", feature_names=features["known_columns"] + ["u_000003", "u_000008"])
    pipe = PredictionPipeline(models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel})
    with pytest.raises(BundleError, match="u_000008"):
        pipe.load()


def test_load_rejects_known_columns_that_disagree_with_the_saved_model(
    tmp_path: Path, configs_dir: Path, dna: dict[str, str]
) -> None:
    """Same-length reorder: passes a column-count check but would feed KPC into the NDM input."""
    models = write_real_bundle(tmp_path / "real_models", dna)
    _edit_features(models, "meropenem", known_columns=["gene_blandm_1", "gene_blakpc_2"])
    with pytest.raises(BundleError, match="meropenem"):
        PredictionPipeline(models, configs_dir).load()


def test_load_rejects_unitig_columns_that_disagree_with_the_saved_model(
    tmp_path: Path, configs_dir: Path, dna: dict[str, str]
) -> None:
    models = write_real_bundle(tmp_path / "real_models", dna)
    _edit_features(models, "ciprofloxacin", unitig_cols=[1])  # model was fitted on u_000000
    with pytest.raises(BundleError, match="u_000001"):
        PredictionPipeline(models, configs_dir).load()


# --------------------------------------------------------------------------- #
# Frozen k-mer set: loaded once, validated at load
# --------------------------------------------------------------------------- #


def test_default_unitig_query_loads_the_kmer_set_once(
    tmp_path: Path, configs_dir: Path, dna: dict[str, str], genomes: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    models = write_real_bundle(tmp_path / "real_models", dna)
    loads: list[Path] = []
    original = unitig_mod.load_kmer_set

    def counting_load(path: Path) -> unitig_mod.KmerSet:
        loads.append(Path(path))
        return original(path)

    def no_reload(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("query_genome re-reads the k-mer set from disk on every request")

    monkeypatch.setattr(unitig_mod, "load_kmer_set", counting_load)
    monkeypatch.setattr(unitig_mod, "query_genome", no_reload)
    pipe = PredictionPipeline(models, configs_dir)
    pipe.load()
    assert loads == [models / SPECIES / "unitig_kmers.npz"]

    first = pipe.run(genomes["kpc"], "REAL-KPC-1")
    pipe.run(genomes["clean"], "REAL-CLEAN")
    again = pipe.run(genomes["kpc"], "REAL-KPC-2")
    assert len(loads) == 1
    assert by_drug(first)["ciprofloxacin"]["pred_mic"] == by_drug(again)["ciprofloxacin"]["pred_mic"]
    assert by_drug(first)["ciprofloxacin"]["call"] == "likely_inactive"


def test_load_rejects_an_unreadable_kmer_set(models_dir: Path, configs_dir: Path) -> None:
    # The FAKE bundle's unitig_kmers.npz lacks k / n_patterns; only the default query reads it.
    pipe = PredictionPipeline(models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel})
    with pytest.raises(BundleError, match="unitig_kmers.npz"):
        pipe.load()


def test_load_rejects_unitig_columns_outside_the_kmer_set(models_dir: Path, configs_dir: Path, dna: dict[str, str]) -> None:
    kmer_set = unitig_mod.KmerSet.from_sequences(
        [dna["marker_gyrA_S83L"], dna["marker_blaKPC-2"]], pattern_col=np.array([0, 1]), n_patterns=2, species=SPECIES
    )
    unitig_mod.save_kmer_set(models_dir / SPECIES / "unitig_kmers.npz", kmer_set)
    pipe = PredictionPipeline(models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel})
    with pytest.raises(BundleError, match="ciprofloxacin"):  # unitig_cols [3, 7] but only 2 patterns
        pipe.load()


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def test_kpc_genome_report(pipeline: PredictionPipeline, genomes: dict[str, Path]) -> None:
    report = pipeline.run(genomes["kpc"], "FAKE-KPC")
    validated = PredictionReport.model_validate(report)

    assert validated.sample_id == "FAKE-KPC"
    assert report["species"] == SPECIES
    assert report["qc_pass"] is True
    assert report["in_range"] is True
    assert 0.0 <= report["nearest_training_distance"] <= 0.05
    assert report["model_version"] == MODEL_VERSION
    assert report["run_id"] == RUN_ID
    assert report["disclaimer"] == DISCLAIMER
    assert [p["drug"] for p in report["predictions"]] == [
        "ampicillin", "piperacillin-tazobactam", "ceftriaxone", "meropenem", "ciprofloxacin", "gentamicin",
    ]
    preds = by_drug(report)

    # Override 1: natural resistance -> likely_inactive, model skipped, MIC fields null.
    ampicillin = preds["ampicillin"]
    assert ampicillin["call"] == "likely_inactive"
    assert ampicillin["override"] == "natural_resistance"
    assert ampicillin["pred_mic"] is None and ampicillin["band_low"] is None and ampicillin["band_high"] is None
    assert ampicillin["margin_steps"] is None
    assert ampicillin["reasons"] == ["natural resistance"]
    assert ampicillin["s_breakpoint"] == 8.0

    # Override 2: planted blaKPC-2 forces meropenem inactive; the model MIC fields are kept.
    meropenem = preds["meropenem"]
    assert meropenem["call"] == "likely_inactive"
    assert meropenem["override"] == "strong_marker"
    assert meropenem["reasons"] == ["blaKPC-2"]
    assert meropenem["pred_mic"] == 4.0  # -4 + 6 -> 2^2, rounded up stays 4
    assert (meropenem["band_low"], meropenem["band_high"]) == (2.0, 8.0)
    assert (meropenem["s_breakpoint"], meropenem["r_breakpoint"]) == (2.0, 8.0)
    assert meropenem["margin_steps"] is None

    # KPC is also a strong marker for ceftriaxone in drugs.yaml.
    assert preds["ceftriaxone"]["call"] == "likely_inactive"
    assert preds["ceftriaxone"]["override"] == "strong_marker"
    assert preds["ceftriaxone"]["reasons"] == ["blaKPC-2"]

    # Ciprofloxacin: gyrA S83L + unitig pattern -> band (1, 4), band_low > R=0.5 -> inactive by the model.
    cipro = preds["ciprofloxacin"]
    assert cipro["call"] == "likely_inactive"
    assert cipro["override"] is None
    assert cipro["pred_mic"] == 2.0
    assert (cipro["band_low"], cipro["band_high"]) == (1.0, 4.0)
    assert cipro["reasons"] == ["gyrA S83L"]

    # Likely active drugs and their ranking (tier asc, margin desc, name).
    gentamicin = preds["gentamicin"]
    assert gentamicin["call"] == "likely_active"
    assert gentamicin["pred_mic"] == 0.5 and gentamicin["margin_steps"] == 1
    assert gentamicin["reasons"] == []
    tzp = preds["piperacillin-tazobactam"]
    assert tzp["call"] == "likely_active"
    assert tzp["pred_mic"] == 4.0 and tzp["band_high"] == 8.0 and tzp["margin_steps"] == 0
    assert report["ranked_active"] == ["gentamicin", "piperacillin-tazobactam"]
    assert all(preds[d]["call"] == "likely_active" for d in report["ranked_active"])


def test_clean_genome_report_ranks_all_active_drugs(
    pipeline: PredictionPipeline, genomes: dict[str, Path], unitig_query: RecordingUnitigQuery
) -> None:
    report = pipeline.run(genomes["clean"], "FAKE-CLEAN")
    PredictionReport.model_validate(report)
    preds = by_drug(report)

    assert report["qc_pass"] is True and report["in_range"] is True
    assert all(p["override"] is None for d, p in preds.items() if d != "ampicillin")

    assert preds["meropenem"]["call"] == "likely_active"
    assert preds["meropenem"]["pred_mic"] == 0.0625
    assert preds["meropenem"]["margin_steps"] == 4  # log2(2) - log2(0.125)
    assert preds["meropenem"]["reasons"] == []

    # blaSHV-11 has no class in markers.fasta; features.json class_by_column supplies BETA-LACTAM.
    assert preds["ceftriaxone"]["call"] == "likely_active"
    assert preds["ceftriaxone"]["margin_steps"] == 2
    assert preds["ceftriaxone"]["reasons"] == ["blaSHV-11"]
    assert preds["meropenem"]["reasons"] == []  # not in meropenem's class_by_column, no class from the scan

    # The unitig pattern at column 3 adds +2 log2 steps: -5 + 2 -> 0.125, band (0.0625, 0.25), S=0.25.
    assert preds["ciprofloxacin"]["call"] == "likely_active"
    assert preds["ciprofloxacin"]["pred_mic"] == 0.125
    assert preds["ciprofloxacin"]["margin_steps"] == 0
    assert len(unitig_query.calls) == 1
    kmers_path, fasta_path = unitig_query.calls[0]
    assert kmers_path == pipeline.models_dir / SPECIES / "unitig_kmers.npz"
    assert fasta_path == genomes["clean"]

    assert report["ranked_active"] == [
        "ceftriaxone", "gentamicin", "ciprofloxacin", "piperacillin-tazobactam", "meropenem",
    ]


def test_unitig_vector_changes_the_prediction(models_dir: Path, configs_dir: Path, genomes: dict[str, Path]) -> None:
    pipe = PredictionPipeline(
        models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel}, unitig_query=RecordingUnitigQuery(present=())
    )
    report = pipe.run(genomes["clean"], "FAKE-NO-UNITIG")
    cipro = by_drug(report)["ciprofloxacin"]
    assert cipro["pred_mic"] == 0.03125  # intercept only
    assert cipro["margin_steps"] == 2  # log2(0.25) - log2(0.0625)


def test_far_genome_has_no_species_and_no_predictions(pipeline: PredictionPipeline, genomes: dict[str, Path]) -> None:
    report = pipeline.run(genomes["far"], "FAKE-FAR")
    PredictionReport.model_validate(report)
    assert report["species"] is None
    assert report["qc_pass"] is False
    assert report["in_range"] is False
    assert report["nearest_training_distance"] is None
    assert report["predictions"] == []
    assert report["ranked_active"] == []
    assert report["disclaimer"] == DISCLAIMER


def test_fragmented_genome_fails_qc_but_is_still_predicted(pipeline: PredictionPipeline, genomes: dict[str, Path]) -> None:
    report = pipeline.run(genomes["fragmented"], "FAKE-FRAG")
    PredictionReport.model_validate(report)
    assert report["species"] == SPECIES
    assert report["qc_pass"] is False
    assert report["in_range"] is True
    assert len(report["predictions"]) == len(FAKE_MODELS)


def test_out_of_range_training_set_keeps_calls_and_flags(
    models_dir: Path, configs_dir: Path, genomes: dict[str, Path], dna: dict[str, str]
) -> None:
    # Training genomes unrelated to the reference: the species is still identified but far from training.
    rng = np.random.default_rng(99)
    sk.save_sketches(
        models_dir / SPECIES / "train_sketches.npz", ["t0", "t1"], [sk.sketch(random_dna(rng, GENOME_LEN)) for _ in range(2)]
    )
    pipe = PredictionPipeline(
        models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel}, unitig_query=RecordingUnitigQuery()
    )
    report = pipe.run(genomes["clean"], "FAKE-OOR")
    PredictionReport.model_validate(report)
    assert report["species"] == SPECIES
    assert report["in_range"] is False
    assert report["nearest_training_distance"] > 0.05
    assert len(report["predictions"]) == len(FAKE_MODELS)
    assert report["ranked_active"]  # calls are kept; the flag drives the UI


def test_species_without_models_gets_natural_resistance_only(pipeline: PredictionPipeline, genomes: dict[str, Path]) -> None:
    report = pipeline.run(genomes["paer"], "FAKE-PAER")
    PredictionReport.model_validate(report)
    config = load_config(pipeline.configs_dir)
    expected = [d for d in config.drug_names() if config.is_naturally_resistant("PAER", d)]
    assert report["species"] == "PAER"
    assert report["in_range"] is False
    assert report["nearest_training_distance"] is None
    assert [p["drug"] for p in report["predictions"]] == expected
    assert all(p["override"] == "natural_resistance" and p["pred_mic"] is None for p in report["predictions"])
    assert report["ranked_active"] == []


def test_run_autoloads(models_dir: Path, configs_dir: Path, genomes: dict[str, Path]) -> None:
    pipe = PredictionPipeline(
        models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel}, unitig_query=RecordingUnitigQuery()
    )
    assert pipe.is_loaded is False
    report = pipe.run(genomes["clean"], "FAKE-AUTOLOAD")
    assert pipe.is_loaded is True
    assert report["species"] == SPECIES


def test_run_rejects_missing_fasta_and_blank_sample_id(pipeline: PredictionPipeline, genomes: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pipeline.run(tmp_path / "nope.fasta", "x")
    with pytest.raises(ValueError):
        pipeline.run(genomes["clean"], "  ")


# --------------------------------------------------------------------------- #
# Known-AMR detection backends
# --------------------------------------------------------------------------- #

_AMRFINDER_HEADER = (
    "Protein identifier\tContig id\tStart\tStop\tStrand\tElement symbol\tElement name\tScope\tType\tSubtype\t"
    "Class\tSubclass\tMethod\tTarget length\tReference sequence length\t% Coverage of reference\t"
    "% Identity to reference\tAlignment length\tAccession of closest sequence\tName of closest sequence\tHMM id\tHMM description"
)


def _amrfinder_row(symbol: str, type_: str, subtype: str, cls: str, subclass: str) -> str:
    return "\t".join(
        ["NA", "contig_1", "1", "100", "+", symbol, symbol, "core", type_, subtype, cls, subclass, "EXACTX", "100", "100", "100.00", "100.00", "100", "X", "X", "NA", "NA"]
    )


@pytest.mark.skipif(shutil.which("amrfinder") is not None, reason="amrfinder on PATH takes precedence over the sidecar")
def test_precomputed_sidecar_tsv_feeds_the_strong_marker_override(pipeline: PredictionPipeline, genomes: dict[str, Path]) -> None:
    sidecar = amr_detect.sidecar_path(genomes["clean"])
    sidecar.write_text(
        "\n".join(
            [
                _AMRFINDER_HEADER,
                _amrfinder_row("blaNDM-1", "AMR", "AMR", "BETA-LACTAM", "CARBAPENEM"),
                _amrfinder_row("qacE", "STRESS", "BIOCIDE", "QUATERNARY AMMONIUM", "QUATERNARY AMMONIUM"),
            ]
        )
        + "\n"
    )
    report = pipeline.run(genomes["clean"], "FAKE-SIDECAR")
    PredictionReport.model_validate(report)
    meropenem = by_drug(report)["meropenem"]
    assert meropenem["call"] == "likely_inactive"
    assert meropenem["override"] == "strong_marker"
    assert meropenem["reasons"] == ["blaNDM-1"]
    assert meropenem["pred_mic"] == 4.0  # -4 + 6 from gene_blandm_1


@pytest.mark.skipif(shutil.which("amrfinder") is not None, reason="amrfinder on PATH is a valid backend")
def test_run_raises_tool_not_available_without_any_backend(tmp_path: Path, configs_dir: Path, dna: dict[str, str], genomes: dict[str, Path]) -> None:
    models = write_bundle(tmp_path / "models_no_markers", dna, with_markers=False)
    pipe = PredictionPipeline(models, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel}, unitig_query=RecordingUnitigQuery())
    with pytest.raises(ToolNotAvailable):
        pipe.run(genomes["clean"], "FAKE-NO-BACKEND")


@pytest.mark.skipif(shutil.which("amrfinder") is not None, reason="amrfinder on PATH is a valid backend")
@pytest.mark.parametrize("synthetic", [False, None], ids=["synthetic_false", "no_synthetic_key"])
def test_markers_fasta_is_ignored_unless_the_bundle_is_synthetic(
    tmp_path: Path, configs_dir: Path, dna: dict[str, str], genomes: dict[str, Path], synthetic: bool | None
) -> None:
    """A real bundle must never fall back to the exact-substring MarkerScan (misses split/SNP'd markers)."""
    models = write_bundle(tmp_path / "models_real", dna, synthetic=synthetic)
    assert (models / "markers.fasta").is_file()
    pipe = PredictionPipeline(models, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel}, unitig_query=RecordingUnitigQuery())
    pipe.load()
    assert pipe.markers_fasta is None
    with pytest.raises(ToolNotAvailable) as excinfo:
        pipe.run(genomes["kpc"], "FAKE-NOT-SYNTHETIC")
    message = str(excinfo.value)
    assert "AMRFinderPlus" in message and "--amrfinder-tsv" in message and "kpc.fasta.amrfinder.tsv" in message
    assert "synthetic" in message


@pytest.mark.skipif(shutil.which("amrfinder") is not None, reason="amrfinder on PATH takes precedence over the sidecar")
def test_non_synthetic_bundle_reads_a_sidecar_tsv(tmp_path: Path, configs_dir: Path, dna: dict[str, str], genomes: dict[str, Path]) -> None:
    models = write_bundle(tmp_path / "models_real", dna, synthetic=False)
    amr_detect.sidecar_path(genomes["clean"]).write_text(
        _AMRFINDER_HEADER + "\n" + _amrfinder_row("blaNDM-1", "AMR", "AMR", "BETA-LACTAM", "CARBAPENEM") + "\n"
    )
    pipe = PredictionPipeline(models, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel}, unitig_query=RecordingUnitigQuery())
    meropenem = by_drug(pipe.run(genomes["clean"], "FAKE-REAL-SIDECAR"))["meropenem"]
    assert meropenem["override"] == "strong_marker" and meropenem["reasons"] == ["blaNDM-1"]


def test_synthetic_bundle_keeps_the_marker_scan_fallback(pipeline: PredictionPipeline) -> None:
    assert pipeline.markers_fasta == pipeline.models_dir / "markers.fasta"


@pytest.mark.skipif(shutil.which("amrfinder") is not None, reason="amrfinder on PATH takes precedence over the sidecar")
@pytest.mark.parametrize("symbol", ["blaOXA-181", "blaOXA-232", "blaOXA-23", "blaGES-5"])
def test_carbapenemase_outside_the_model_forces_meropenem_inactive(
    pipeline: PredictionPipeline, genomes: dict[str, Path], symbol: str
) -> None:
    """End to end: the model knows nothing about these genes (zero-filled), the override still fires."""
    amr_detect.sidecar_path(genomes["clean"]).write_text(
        _AMRFINDER_HEADER + "\n" + _amrfinder_row(symbol, "AMR", "AMR", "BETA-LACTAM", "CARBAPENEM") + "\n"
    )
    report = pipeline.run(genomes["clean"], f"FAKE-{symbol}")
    PredictionReport.model_validate(report)
    meropenem = by_drug(report)["meropenem"]
    assert meropenem["call"] == "likely_inactive"
    assert meropenem["override"] == "strong_marker"
    assert meropenem["reasons"] == [symbol]
    assert meropenem["pred_mic"] == 0.0625  # the model output is kept, only the call is overridden
    assert "meropenem" not in report["ranked_active"]


@pytest.mark.skipif(shutil.which("amrfinder") is not None, reason="amrfinder on PATH takes precedence over the sidecar")
def test_porin_point_mutation_does_not_override_meropenem(pipeline: PredictionPipeline, genomes: dict[str, Path]) -> None:
    amr_detect.sidecar_path(genomes["clean"]).write_text(
        _AMRFINDER_HEADER + "\n" + _amrfinder_row("ompK36_D135DGD", "AMR", "POINT", "BETA-LACTAM", "CARBAPENEM") + "\n"
    )
    meropenem = by_drug(pipeline.run(genomes["clean"], "FAKE-OMPK36"))["meropenem"]
    assert meropenem["override"] is None
    assert meropenem["pred_mic"] == 0.125  # -4 + 1 (point_ompk36_d135dgd) -> 2^-3


# --------------------------------------------------------------------------- #
# AMRFinderPlus parsing: prediction applies exactly the training row filter
# --------------------------------------------------------------------------- #

# (symbol, Type, Subtype, Class, Subclass). FAKE detections covering every filter branch.
PARITY_ROWS: list[tuple[str, str, str, str, str]] = [
    ("blaKPC-2", "AMR", "AMR", "BETA-LACTAM", "CARBAPENEM"),
    ("blaCTX-M-15", "AMR", "AMR", "BETA-LACTAM", "CEPHALOSPORIN"),
    ("blaCTX-M-27", "AMR", "AMR", "BETA-LACTAM", "CEPHALOSPORIN"),
    ("blaOXA-1", "AMR", "AMR", "BETA-LACTAM", "BETA-LACTAM"),
    ("blaOXA-181", "AMR", "AMR", "BETA-LACTAM", "CARBAPENEM"),
    ("gyrA_S83L", "AMR", "POINT", "QUINOLONE", "QUINOLONE"),
    ("ompK36_D135DGD", "AMR", "POINT", "BETA-LACTAM", "CARBAPENEM"),
    ("aac(6')-Ib-cr", "AMR", "AMR", "AMINOGLYCOSIDE/QUINOLONE", "AMIKACIN/KANAMYCIN/QUINOLONE"),
    ("blaSHV-11", "AMR", "AMR", "NA", "NA"),  # no class: gene_ column, no n_class_ count
    ("blaTEM-1", "AMR", "AMR-SUSCEPTIBLE", "BETA-LACTAM", "BETA-LACTAM"),  # dropped: subtype
    ("blaFAKE-1", "", "AMR", "BETA-LACTAM", "BETA-LACTAM"),  # dropped: blank Type
    ("blaFAKE-2", "amr", "AMR", "BETA-LACTAM", "BETA-LACTAM"),  # dropped: Type is case-sensitive in training
    ("qacE", "STRESS", "BIOCIDE", "QUATERNARY AMMONIUM", "QUATERNARY AMMONIUM"),
    ("iutA", "VIRULENCE", "VIRULENCE", "NA", "NA"),
    ("NA", "AMR", "AMR", "BETA-LACTAM", "BETA-LACTAM"),  # dropped: no symbol
]


def _write_amrfinder(path: Path, rows: Sequence[tuple[str, str, str, str, str]], version: str) -> Path:
    if version == "4":
        header, row_text = _AMRFINDER_HEADER, [_amrfinder_row(*row) for row in rows]
    else:
        header = (
            "Protein identifier\tContig id\tStart\tStop\tStrand\tGene symbol\tSequence name\tScope\tElement type\t"
            "Element subtype\tClass\tSubclass\tMethod\tTarget length\tReference sequence length\t"
            "% Coverage of reference sequence\t% Identity to reference sequence\tAlignment length\t"
            "Accession of closest sequence\tName of closest sequence\tHMM id\tHMM description"
        )
        row_text = [_amrfinder_row(*row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *row_text]) + "\n", encoding="utf-8")
    return path


def test_parse_amrfinder_tsv_applies_the_training_row_filter(tmp_path: Path) -> None:
    droplog = DropLog("test")
    frame = amr_detect.parse_amrfinder_tsv(_write_amrfinder(tmp_path / "a.tsv", PARITY_ROWS, "4"), droplog)
    assert frame["symbol"].tolist() == [r[0] for r in PARITY_ROWS[:9]]
    assert set(frame["type"]) == {"AMR"}
    assert set(frame["subtype"]) == {"AMR", "POINT"}
    assert frame["class"].isna().sum() == 1  # blaSHV-11 "NA" is null, as in training
    counts = {r.reason: r.n_dropped for r in droplog.records}
    assert counts == {
        "amrfinder_type_not_amr": 4,
        "amrfinder_subtype_not_amr_or_point": 1,
        "amrfinder_no_symbol": 1,
    }


@pytest.mark.parametrize("version", ["4", "3"])
def test_amrfinder_tsv_gives_the_same_feature_row_in_training_and_prediction(
    tmp_path: Path, configs_dir: Path, version: str
) -> None:
    """Parity: one AMRFinderPlus TSV -> identical known-AMR features in ``known_amr.build`` and the pipeline."""
    from genome2mic.features import known_amr  # noqa: PLC0415
    from genome2mic.paths import Paths  # noqa: PLC0415

    config = load_config(configs_dir)
    paths = Paths(root=tmp_path / "project", configs_dir=configs_dir)
    paths.qc.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"genome_id": ["g1"], "species": [SPECIES], "qc_pass": [True]}).to_parquet(paths.qc, index=False)
    tsv = _write_amrfinder(paths.interim_dir("g1") / "amrfinder.tsv", PARITY_ROWS, version)

    features, _columns = known_amr.build(paths, config)
    training = {
        str(column): int(value)
        for column, value in features.iloc[0].items()
        if str(column).startswith(known_amr.FEATURE_PREFIXES) and int(value) != 0
    }
    prediction = amr_detect.known_amr_row(amr_detect.parse_amrfinder_tsv(tsv), config).values
    assert prediction == training
    assert training["gene_blaoxa_181"] == 1 and training["n_class_beta_lactam"] == 6
    assert "gene_blatem" not in prediction and "gene_iuta" not in prediction and "gene_blafake" not in prediction


def test_parse_amrfinder_tsv_rejects_a_non_amrfinder_table(tmp_path: Path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text("foo\tbar\n1\t2\n")
    with pytest.raises(ValueError, match="AMRFinderPlus"):
        amr_detect.parse_amrfinder_tsv(path)


def test_parse_amrfinder_tsv_handles_old_header_and_filters_type(tmp_path: Path) -> None:
    path = tmp_path / "old.tsv"
    path.write_text(
        "Gene symbol\tType\tSubtype\tClass\tSubclass\tMethod\t% Coverage of reference\t% Identity to reference\n"
        "blaKPC-2\tAMR\tAMR\tBETA-LACTAM\tCARBAPENEM\tEXACTX\t100.00\t100.00\n"
        "gyrA_S83L\tAMR\tPOINT\tQUINOLONE\tQUINOLONE\tPOINTX\t100.00\t99.70\n"
        "qacE\tSTRESS\tBIOCIDE\tQUATERNARY AMMONIUM\tQUATERNARY AMMONIUM\tBLASTX\t100.00\t100.00\n"
    )
    droplog = DropLog("test")
    frame = amr_detect.parse_amrfinder_tsv(path, droplog)
    assert list(frame.columns) == list(amr_detect.DETECTION_COLUMNS)
    assert frame["symbol"].tolist() == ["blaKPC-2", "gyrA_S83L"]
    assert frame["subtype"].tolist() == ["AMR", "POINT"]
    assert droplog.records[0].reason == "amrfinder_type_not_amr" and droplog.records[0].n_dropped == 1


@pytest.mark.parametrize(
    ("header", "symbol", "subtype", "amr_class", "subclass"),
    [
        (">blaKPC-2 BETA-LACTAM CARBAPENEM AMR", "blaKPC-2", "AMR", "BETA-LACTAM", "CARBAPENEM"),
        ("gyrA_S83L QUINOLONE QUINOLONE POINT", "gyrA_S83L", "POINT", "QUINOLONE", "QUINOLONE"),
        ("ompK36_D135DGD BETA-LACTAM", "ompK36_D135DGD", "POINT", "BETA-LACTAM", "BETA-LACTAM"),
        ("blaSHV-11", "blaSHV-11", "AMR", None, None),
        ("mcr-1 class=COLISTIN subclass=COLISTIN subtype=AMR", "mcr-1", "AMR", "COLISTIN", "COLISTIN"),
    ],
)
def test_parse_marker_header(header: str, symbol: str, subtype: str, amr_class: str | None, subclass: str | None) -> None:
    marker = amr_detect.parse_marker_header(header)
    assert (marker.symbol, marker.subtype, marker.amr_class, marker.subclass) == (symbol, subtype, amr_class, subclass)


def test_marker_scan_finds_forward_and_reverse_complement(tmp_path: Path, dna: dict[str, str], configs_dir: Path) -> None:
    markers_fasta = write_fasta([(MARKERS[s][0], dna[f"marker_{s}"]) for s in MARKERS], tmp_path / "markers.fasta")
    rc = dna["marker_blaCTX-M-15"].translate(str.maketrans("ACGT", "TGCA"))[::-1]
    genome = write_fasta(
        [("c1", random_dna(np.random.default_rng(1), 500) + rc), ("c2", dna["marker_aac(6')-Ib-cr"] + random_dna(np.random.default_rng(2), 500))],
        tmp_path / "g.fasta",
    )
    frame = amr_detect.MarkerScan(markers_fasta).detect(genome, SPECIES, load_config(configs_dir))
    assert sorted(frame["symbol"]) == ["aac(6')-Ib-cr", "blaCTX-M-15"]
    assert set(frame["backend"]) == {"marker_scan"}


@pytest.mark.parametrize(
    ("header", "element_type", "subtype", "amr_class", "subclass"),
    [
        (
            'blaKPC-2 type=AMR subtype=AMR class=BETA-LACTAM subclass=CARBAPENEM scope=core '
            'name="carbapenem-hydrolyzing class A beta-lactamase KPC-2" synthetic=true',
            "AMR", "AMR", "BETA-LACTAM", "CARBAPENEM",
        ),
        (
            'iutA type=VIRULENCE subtype=VIRULENCE class=NA subclass=NA scope=plus name="aerobactin receptor" synthetic=true',
            "VIRULENCE", "VIRULENCE", None, None,
        ),
        ("qacE QUATERNARY_AMMONIUM QUATERNARY_AMMONIUM STRESS", "STRESS", "STRESS", "QUATERNARY_AMMONIUM", "QUATERNARY_AMMONIUM"),
        ("blaTEM-1 type=AMR subtype=AMR-SUSCEPTIBLE class=BETA-LACTAM", "AMR", "AMR-SUSCEPTIBLE", "BETA-LACTAM", "BETA-LACTAM"),
        ("gyrA_S83L type=AMR class=QUINOLONE", "AMR", "POINT", "QUINOLONE", "QUINOLONE"),
    ],
)
def test_parse_marker_header_keeps_type_and_subtype(
    header: str, element_type: str, subtype: str, amr_class: str | None, subclass: str | None
) -> None:
    marker = amr_detect.parse_marker_header(header)
    assert (marker.element_type, marker.subtype, marker.amr_class, marker.subclass) == (element_type, subtype, amr_class, subclass)
    assert marker.is_resistance_feature is (element_type == "AMR" and subtype in ("AMR", "POINT"))


def test_marker_scan_skips_records_training_would_drop(tmp_path: Path, dna: dict[str, str], configs_dir: Path) -> None:
    """A VIRULENCE / STRESS / AMR-SUSCEPTIBLE marker never becomes a gene_ feature (training drops it)."""
    rng = np.random.default_rng(3)
    seqs = {name: random_dna(rng, 120) for name in ("iutA", "qacE", "blaTEM-1")}
    records = [
        (MARKERS["blaKPC-2"][0], dna["marker_blaKPC-2"]),
        ("iutA type=VIRULENCE subtype=VIRULENCE class=NA subclass=NA synthetic=true", seqs["iutA"]),
        ("qacE QUATERNARY_AMMONIUM QUATERNARY_AMMONIUM STRESS", seqs["qacE"]),
        ("blaTEM-1 type=AMR subtype=AMR-SUSCEPTIBLE class=BETA-LACTAM", seqs["blaTEM-1"]),
    ]
    markers_fasta = write_fasta(records, tmp_path / "markers.fasta")
    genome = write_fasta([("c1", random_dna(rng, 300) + "".join(seq for _, seq in records) + random_dna(rng, 300))], tmp_path / "g.fasta")
    droplog = DropLog("test")
    frame = amr_detect.MarkerScan(markers_fasta).detect(genome, SPECIES, load_config(configs_dir), droplog)
    assert frame["symbol"].tolist() == ["blaKPC-2"]
    counts = {r.reason: r.n_dropped for r in droplog.records}
    assert counts["marker_records_not_amr"] == 3
    row = amr_detect.known_amr_row(frame, load_config(configs_dir))
    assert set(row.values) == {"gene_blakpc_2", "n_class_beta_lactam"}


def test_known_amr_row_skips_non_amr_rows_from_custom_detectors(configs_dir: Path) -> None:
    detections = pd.DataFrame(
        {
            "symbol": ["blaKPC-2", "iutA", "blaTEM-1"],
            "type": ["AMR", "VIRULENCE", "AMR"],
            "subtype": ["AMR", "VIRULENCE", "AMR-SUSCEPTIBLE"],
            "class": ["BETA-LACTAM", None, "BETA-LACTAM"],
            "subclass": ["CARBAPENEM", None, "BETA-LACTAM"],
        }
    )
    row = amr_detect.known_amr_row(detections, load_config(configs_dir))
    assert row.values == {"gene_blakpc_2": 1, "n_class_beta_lactam": 1}


# --------------------------------------------------------------------------- #
# Feature naming (local implementation of the DESIGN.md rule)
# --------------------------------------------------------------------------- #

KEEP = ("blaKPC", "blaNDM", "blaOXA-48", "blaOXA-181", "blaOXA-232", "blaVIM", "blaIMP")


@pytest.mark.parametrize(
    ("symbol", "family"),
    [
        ("blaCTX-M-15", "blaCTX-M"),
        ("blaCTX-M-27", "blaCTX-M"),
        ("blaKPC-2", "blaKPC-2"),
        ("blaNDM-1", "blaNDM-1"),
        ("blaOXA-48", "blaOXA-48"),
        ("blaOXA-1", "blaOXA"),
        ("blaTEM-1", "blaTEM"),
        ("blaSHV-11", "blaSHV"),
        ("aac(6')-Ib-cr", "aac(6')-Ib-cr"),
        ("qnrB1", "qnrB1"),
    ],
)
def test_local_family_of_follows_the_contract(symbol: str, family: str) -> None:
    assert amr_detect._local_family_of(symbol, KEEP) == family


@pytest.mark.parametrize(
    ("prefix", "symbol", "column"),
    [
        ("gene_", "blaCTX-M", "gene_blactx_m"),
        ("gene_", "blaKPC-2", "gene_blakpc_2"),
        ("gene_", "aac(6')-Ib-cr", "gene_aac_6_ib_cr"),
        ("point_", "ompK36_D135DGD", "point_ompk36_d135dgd"),
        ("point_", "gyrA_S83L", "point_gyra_s83l"),
        ("n_class_", "BETA-LACTAM", "n_class_beta_lactam"),
        ("n_class_", "AMINOGLYCOSIDE/QUINOLONE", "n_class_aminoglycoside_quinolone"),
    ],
)
def test_local_column_name_follows_the_contract(prefix: str, symbol: str, column: str) -> None:
    assert amr_detect._local_column_name(prefix, symbol) == column


def test_known_amr_row_and_feature_vector(configs_dir: Path) -> None:
    config = load_config(configs_dir)
    detections = pd.DataFrame(
        {
            "symbol": ["blaKPC-2", "blaCTX-M-15", "blaCTX-M-27", "gyrA_S83L", "blaSHV-11"],
            "type": ["AMR"] * 5,
            "subtype": ["AMR", "AMR", "AMR", "POINT", "AMR"],
            "class": ["BETA-LACTAM", "BETA-LACTAM", "BETA-LACTAM", "QUINOLONE", None],
            "subclass": ["CARBAPENEM", "CEPHALOSPORIN", "CEPHALOSPORIN", "QUINOLONE", None],
            "method": ["EXACTX"] * 5,
            "coverage": [100.0] * 5,
            "identity": [100.0] * 5,
            "backend": ["test"] * 5,
        }
    )
    row = amr_detect.known_amr_row(detections, config)
    assert row.values == {
        "gene_blakpc_2": 1,
        "gene_blactx_m": 1,
        "point_gyra_s83l": 1,
        "gene_blashv": 1,
        "n_class_beta_lactam": 3,
        "n_class_quinolone": 1,
    }
    assert row.symbols_by_column["gene_blactx_m"] == ("blaCTX-M-15", "blaCTX-M-27")
    assert {m.column for m in row.markers} == {"gene_blakpc_2", "gene_blactx_m", "point_gyra_s83l", "gene_blashv"}

    droplog = DropLog("test")
    vector = amr_detect.feature_vector(row, ["gene_blakpc_2", "gene_blandm_1", "n_class_beta_lactam"], droplog)
    assert vector.tolist() == [1.0, 0.0, 3.0]
    assert vector.dtype == np.float32
    record = droplog.records[0]
    assert record.reason == "known_amr_column_not_in_training" and record.n_dropped == 4
    assert record.detail == "gene_blactx_m, gene_blashv, n_class_quinolone, point_gyra_s83l"


# --------------------------------------------------------------------------- #
# Assembly stats
# --------------------------------------------------------------------------- #


def test_assembly_stats() -> None:
    stats = assembly_stats(["ACGT" * 10, "GGCC" * 5, "AT" * 5])
    assert stats.n_contigs == 3
    assert stats.total_length == 70
    assert stats.n50 == 40
    assert stats.gc_percent == pytest.approx(100.0 * (20 + 20) / 70)
    empty = assembly_stats([])
    assert (empty.n_contigs, empty.total_length, empty.n50, empty.gc_percent) == (0, 0, 0, None)


# --------------------------------------------------------------------------- #
# Integration: real MODEL_CLASSES (B1Lookup, XgbAft) and a real frozen k-mer set
# --------------------------------------------------------------------------- #

REAL_VERSION = "FAKE-DATA-REAL-CLASSES-0.1"


def write_real_bundle(models_dir: Path, dna: dict[str, str]) -> Path:
    """Bundle whose models are genuine ``genome2mic.models`` classes fitted on 20 synthetic rows.

    The labels are made up; the point is to exercise ``MODEL_CLASSES[model_class].load``,
    ``model.ubj`` round-tripping and ``features.unitigs.query_genome`` on a real k-mer set.
    """
    rng = np.random.default_rng(5)
    models_dir.mkdir(parents=True, exist_ok=True)
    species_dir = models_dir / SPECIES
    (models_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_version": REAL_VERSION,
                "run_id": RUN_ID,
                "created": "2026-10-03",
                "species": {SPECIES: ["meropenem", "ciprofloxacin"]},
                "synthetic": True,
            }
        )
    )
    ref_keys = ["KPNEU", "ECOLI", "PAER"]
    sk.save_sketches(models_dir / "reference_sketches.npz", ref_keys, [sk.sketch(dna[f"ref_{k}"]) for k in ref_keys])
    train = [mutate(dna["ref_KPNEU"], 8, rng) for _ in range(4)]
    sk.save_sketches(species_dir / "train_sketches.npz", [f"train_{i}" for i in range(4)], [sk.sketch(g) for g in train])
    write_fasta([(MARKERS[s][0], dna[f"marker_{s}"]) for s in MARKERS], models_dir / "markers.fasta")

    # Frozen k-mer set with two "unitigs": pattern 0 = the gyrA marker, pattern 1 = the KPC marker.
    kmer_set = unitig_mod.KmerSet.from_sequences(
        [dna["marker_gyrA_S83L"], dna["marker_blaKPC-2"]], pattern_col=np.array([0, 1]), n_patterns=2, species=SPECIES
    )
    unitig_mod.save_kmer_set(species_dir / "unitig_kmers.npz", kmer_set)

    # meropenem: B1Lookup, 8 KPC carriers at (8, 16], 12 others left-censored at <= 0.0625.
    mem_dir = species_dir / "meropenem"
    X = np.array([[1, 0]] * 8 + [[0, 0]] * 12, dtype=np.int8)
    lo = np.array([8.0] * 8 + [0.0] * 12)
    hi = np.array([16.0] * 8 + [0.0625] * 12)
    B1Lookup().fit(X, lo, hi, ["gene_blakpc_2", "gene_blandm_1"]).save(mem_dir)
    (mem_dir / "features.json").write_text(
        json.dumps(
            {
                "model_class": "b1_lookup",
                "known_columns": ["gene_blakpc_2", "gene_blandm_1"],
                "unitig_cols": [],
                "class_by_column": {"gene_blakpc_2": ["BETA-LACTAM", "CARBAPENEM"], "gene_blandm_1": ["BETA-LACTAM", "CARBAPENEM"]},
            }
        )
    )
    (mem_dir / "conformal.json").write_text(json.dumps({"q": 1.0}))

    # ciprofloxacin: XgbAft on point_gyra_s83l + unitig pattern 0; 10 resistant (2, 4], 10 susceptible <= 0.03125.
    cip_dir = species_dir / "ciprofloxacin"
    X = np.array([[1, 1]] * 10 + [[0, 0]] * 10, dtype=np.int8)
    lo = np.array([2.0] * 10 + [0.0] * 10)
    hi = np.array([4.0] * 10 + [0.03125] * 10)
    model = XgbAft(name="aft_known_unitig", scales=(1.0,), max_rounds=100, fallback_rounds=100, min_rounds=5)
    model.fit(X, lo, hi, ["point_gyra_s83l", "u_000000"]).save(cip_dir)
    (cip_dir / "features.json").write_text(
        json.dumps(
            {
                "model_class": "aft_known_unitig",
                "known_columns": ["point_gyra_s83l"],
                "unitig_cols": [0],
                "class_by_column": {"point_gyra_s83l": ["QUINOLONE", "QUINOLONE"]},
            }
        )
    )
    (cip_dir / "conformal.json").write_text(json.dumps({"q": 1.0}))
    return models_dir


def test_real_kmer_set_query_sees_planted_markers(tmp_path: Path, dna: dict[str, str], genomes: dict[str, Path]) -> None:
    models = write_real_bundle(tmp_path / "real_models", dna)
    kmers = models / SPECIES / "unitig_kmers.npz"
    assert unitig_mod.query_genome(kmers, genomes["kpc"]).tolist() == [1, 1]
    assert unitig_mod.query_genome(kmers, genomes["clean"]).tolist() == [0, 0]


def test_real_model_classes_end_to_end(tmp_path: Path, configs_dir: Path, dna: dict[str, str], genomes: dict[str, Path]) -> None:
    models = write_real_bundle(tmp_path / "real_models", dna)
    assert {"b1_lookup", "aft_known_unitig"} <= set(MODEL_CLASSES)
    pipe = PredictionPipeline(models, configs_dir)  # no injected registry, no mocked unitig query
    pipe.load()
    assert pipe.available_models() == {SPECIES: ["ciprofloxacin", "meropenem"]}
    assert isinstance(pipe.species_bundles[SPECIES].drugs["meropenem"].model, B1Lookup)
    assert isinstance(pipe.species_bundles[SPECIES].drugs["ciprofloxacin"].model, XgbAft)

    kpc = pipe.run(genomes["kpc"], "REAL-KPC")
    clean = pipe.run(genomes["clean"], "REAL-CLEAN")
    PredictionReport.model_validate(kpc)
    PredictionReport.model_validate(clean)
    assert kpc["model_version"] == REAL_VERSION
    kpc_p, clean_p = by_drug(kpc), by_drug(clean)

    # B1Lookup profile medians: 16 mg/L with KPC (then the strong-marker override), 0.0625 without.
    assert kpc_p["meropenem"]["pred_mic"] == 16.0
    assert kpc_p["meropenem"]["override"] == "strong_marker" and kpc_p["meropenem"]["reasons"] == ["blaKPC-2"]
    assert clean_p["meropenem"]["pred_mic"] == 0.0625
    assert clean_p["meropenem"]["call"] == "likely_active" and clean_p["meropenem"]["margin_steps"] == 4

    # XgbAft with the real k-mer query: the gyrA pattern is present only in the KPC genome.
    assert kpc_p["ciprofloxacin"]["pred_mic"] > clean_p["ciprofloxacin"]["pred_mic"]
    assert kpc_p["ciprofloxacin"]["call"] == "likely_inactive"
    assert clean_p["ciprofloxacin"]["call"] == "likely_active"
    assert kpc_p["ciprofloxacin"]["reasons"] == ["gyrA S83L"]
    assert clean_p["ciprofloxacin"]["reasons"] == []
    assert kpc["ranked_active"] == []
    assert clean["ranked_active"] == ["ciprofloxacin", "meropenem"]


# --------------------------------------------------------------------------- #
# Shared k-mer set: every unitig model must index the species' set
# --------------------------------------------------------------------------- #


def _set_recorded_sha1(models_dir: Path, drug: str, value: str | None) -> None:
    path = models_dir / SPECIES / drug / "features.json"
    features = json.loads(path.read_text())
    if value is None:
        features.pop("unitig_kmer_set_sha1", None)
    else:
        features["unitig_kmer_set_sha1"] = value
    path.write_text(json.dumps(features))


def test_load_accepts_a_drug_trained_on_the_species_kmer_set(tmp_path: Path, configs_dir: Path, dna: dict[str, str]) -> None:
    models = write_real_bundle(tmp_path / "real_models", dna)
    sha1 = unitig_mod.read_kmer_set_sha1(models / SPECIES / "unitig_kmers.npz")
    _set_recorded_sha1(models, "ciprofloxacin", sha1)
    _set_recorded_sha1(models, "meropenem", None)  # no unitig features: nothing to record
    pipe = PredictionPipeline(models, configs_dir)
    pipe.load()
    assert pipe.species_bundles[SPECIES].drugs["ciprofloxacin"].unitig_kmer_set_sha1 == sha1
    assert pipe.species_bundles[SPECIES].kmer_set.sha1() == sha1


@pytest.mark.parametrize("injected_query", [False, True])
def test_load_rejects_a_drug_trained_on_another_kmer_set(
    tmp_path: Path, configs_dir: Path, dna: dict[str, str], injected_query: bool
) -> None:
    """A partial retrain replaced unitig_kmers.npz: the old model's columns would index the new set."""
    models = write_real_bundle(tmp_path / "real_models", dna)
    _set_recorded_sha1(models, "ciprofloxacin", "f" * 40)
    # With an injected query the set is never loaded; the sha1 stored in the npz is compared instead.
    query = RecordingUnitigQuery() if injected_query else None
    pipe = PredictionPipeline(models, configs_dir, unitig_query=query)
    with pytest.raises(BundleError, match="ciprofloxacin was trained on ffffffffffff"):
        pipe.load()


def test_load_rejects_a_malformed_recorded_kmer_set_sha1(tmp_path: Path, configs_dir: Path, dna: dict[str, str]) -> None:
    models = write_real_bundle(tmp_path / "real_models", dna)
    _set_recorded_sha1(models, "ciprofloxacin", "")
    with pytest.raises(BundleError, match="unitig_kmer_set_sha1"):
        PredictionPipeline(models, configs_dir).load()


def test_load_warns_when_a_unitig_model_predates_the_recorded_sha1(
    tmp_path: Path, configs_dir: Path, dna: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    models = write_real_bundle(tmp_path / "real_models", dna)  # features.json without the key
    with caplog.at_level(logging.WARNING, logger="genome2mic.predict.pipeline"):
        PredictionPipeline(models, configs_dir).load()
    assert any("unitig_kmer_set_sha1" in r.getMessage() and "ciprofloxacin" in r.getMessage() for r in caplog.records)


def test_panel_caps_from_conformal_json_clip_the_raw_prediction(
    models_dir: Path, configs_dir: Path, unitig_query: RecordingUnitigQuery, genomes: dict[str, Path]
) -> None:
    """The pipeline clips the raw log2 prediction to the bundle's panel caps before rounding up.

    KPC genome, meropenem: raw log2 = -4 + 6 = 2 (MIC 4). With caps [-3, 1] the prediction
    becomes 2^1 = 2 and the q = 1 band (1, 4), exactly as training would have written it.
    """
    conformal_path = models_dir / SPECIES / "meropenem" / "conformal.json"
    payload = json.loads(conformal_path.read_text())
    payload.update({"cap_low_log2": -3.0, "cap_high_log2": 1.0})
    conformal_path.write_text(json.dumps(payload))
    pipe = PredictionPipeline(
        models_dir=models_dir, configs_dir=configs_dir,
        model_classes={FAKE_MODEL_CLASS: FakeLinearModel}, unitig_query=unitig_query,
    )
    pipe.load()
    assert pipe.species_bundles[SPECIES].drugs["meropenem"].caps == (-3.0, 1.0)
    assert pipe.species_bundles[SPECIES].drugs["ceftriaxone"].caps is None  # older bundle: no caps
    meropenem = by_drug(pipe.run(genomes["kpc"], "FAKE-KPC"))["meropenem"]
    assert meropenem["pred_mic"] == 2.0
    assert (meropenem["band_low"], meropenem["band_high"]) == (1.0, 4.0)


def test_malformed_panel_caps_are_a_bundle_error(models_dir: Path, configs_dir: Path) -> None:
    conformal_path = models_dir / SPECIES / "meropenem" / "conformal.json"
    payload = json.loads(conformal_path.read_text())
    payload.update({"cap_low_log2": 3.0, "cap_high_log2": 1.0})
    conformal_path.write_text(json.dumps(payload))
    pipe = PredictionPipeline(models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel})
    with pytest.raises(BundleError, match="cap_low_log2"):
        pipe.load()


# --------------------------------------------------------------------------- #
# Tuned asymmetric band + active-call gate (conformal.json q_up / q_low / active_gate_open)
# --------------------------------------------------------------------------- #


def _set_asym_band(models_dir: Path, drug: str, *, q_up: float, q_low: float, gate: bool) -> None:
    path = models_dir / SPECIES / drug / "conformal.json"
    conf = json.loads(path.read_text())
    conf.update({"q": q_up, "q_up": q_up, "q_low": q_low, "active_gate_open": gate, "band_kind": "asymmetric_tuned",
                 "alpha_up": 0.025, "alpha_low": 0.05})
    path.write_text(json.dumps(conf))


def test_asymmetric_band_from_the_bundle_is_applied(tmp_path: Path, configs_dir: Path, dna: dict[str, str], genomes: dict[str, Path]) -> None:
    models = write_real_bundle(tmp_path / "real_models", dna)
    _set_asym_band(models, "meropenem", q_up=0.0, q_low=2.0, gate=True)
    pipe = PredictionPipeline(models, configs_dir)
    pipe.load()
    bundle = pipe.species_bundles[SPECIES].drugs["meropenem"]
    assert (bundle.q, bundle.q_low, bundle.active_gate_open) == (0.0, 2.0, True)
    clean = by_drug(pipe.run(genomes["clean"], "ASYM-CLEAN"))
    # pred 0.0625: band (0.0625 / 2**2, 0.0625 * 2**0), the same asym_band training writes.
    assert (clean["meropenem"]["band_low"], clean["meropenem"]["band_high"]) == (0.015625, 0.0625)
    assert clean["meropenem"]["call"] == "likely_active"


def test_closed_gate_withholds_likely_active(tmp_path: Path, configs_dir: Path, dna: dict[str, str], genomes: dict[str, Path]) -> None:
    from genome2mic.predict import rank  # noqa: PLC0415

    models = write_real_bundle(tmp_path / "real_models", dna)
    _set_asym_band(models, "meropenem", q_up=0.0, q_low=2.0, gate=False)
    pipe = PredictionPipeline(models, configs_dir)
    pipe.load()
    report = pipe.run(genomes["clean"], "GATE-CLEAN")
    PredictionReport.model_validate(report)
    mem = by_drug(report)["meropenem"]
    assert mem["call"] == "uncertain" and mem["margin_steps"] is None
    assert rank.ACTIVE_GATE_REASON in mem["reasons"]
    assert "meropenem" not in report["ranked_active"]
    # The strong-marker override still wins for a KPC carrier.
    kpc = by_drug(pipe.run(genomes["kpc"], "GATE-KPC"))["meropenem"]
    assert kpc["call"] == "likely_inactive" and kpc["override"] == "strong_marker"


@pytest.mark.parametrize("bad", [{"q_low": -1.0}, {"q_low": float("nan")}, {"active_gate_open": "yes"}])
def test_bad_asymmetric_band_is_refused(tmp_path: Path, configs_dir: Path, dna: dict[str, str], bad: dict) -> None:
    models = write_real_bundle(tmp_path / "real_models", dna)
    _set_asym_band(models, "meropenem", q_up=1.0, q_low=1.0, gate=True)
    path = models / SPECIES / "meropenem" / "conformal.json"
    conf = json.loads(path.read_text())
    conf.update(bad)
    path.write_text(json.dumps(conf).replace("NaN", "NaN"))
    pipe = PredictionPipeline(models, configs_dir)
    with pytest.raises(BundleError, match="conformal.json"):
        pipe.load()
