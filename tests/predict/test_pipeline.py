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
    return target


def write_bundle(models_dir: Path, dna: dict[str, str], *, with_markers: bool = True) -> Path:
    """Write a FAKE model bundle in the ``models/train.py`` layout."""
    rng = np.random.default_rng(11)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "run_id": RUN_ID,
                "created": "2026-10-03T00:00:00Z",
                "species": {SPECIES: list(FAKE_MODELS)},
                "note": "FAKE synthetic test bundle",
            }
        )
    )
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


def test_load_raises_file_not_found_without_train_sketches(models_dir: Path, configs_dir: Path) -> None:
    (models_dir / SPECIES / "train_sketches.npz").unlink()
    pipe = PredictionPipeline(models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel})
    with pytest.raises(FileNotFoundError):
        pipe.load()


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
    pipe = PredictionPipeline(models_dir, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel})
    pipe.load()
    assert pipe.species_bundles[SPECIES].drugs["ciprofloxacin"].unitig_cols == (3, 7)


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
        json.dumps({"model_version": REAL_VERSION, "run_id": RUN_ID, "created": "2026-10-03", "species": {SPECIES: ["meropenem", "ciprofloxacin"]}})
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
