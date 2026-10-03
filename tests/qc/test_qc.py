"""Tests for ``genome2mic.qc`` (stage 3: assembly stats, species ID, fail rules).

All genomes are small synthetic sequences generated inline from a fixed seed; configs
are built directly as ``Config`` dataclasses so no file under ``configs/`` is read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genome2mic import qc
from genome2mic.config import Config, SpeciesConfig
from genome2mic.errors import ContractViolation
from genome2mic.io import read_parquet, write_fasta, write_parquet
from genome2mic.paths import Paths

_BASES = np.frombuffer(b"ACGT", dtype=np.uint8)
GENOME_LENGTH = 6000


# --------------------------------------------------------------------------- helpers
def make_config(
    expected_size: int = GENOME_LENGTH,
    *,
    max_contigs: int = 5,
    max_distance: float = 0.05,
    keys: tuple[str, ...] = ("KPNEU", "ECOLI"),
    tolerance: float = 0.2,
) -> Config:
    species = {
        key: SpeciesConfig(
            key=key,
            name=key.title(),
            amrfinder_organism=key,
            expected_genome_size=expected_size,
            size_tolerance=tolerance,
            reference_accession=f"NC_{key}",
        )
        for key in keys
    }
    return Config(
        species=species,
        drugs={},
        call_standard=("EUCAST", "2024"),
        qc_max_contigs=max_contigs,
        qc_max_mash_distance=max_distance,
        breakpoints={},
        natural_resistance=frozenset(),
        keep_variant=("blaKPC",),
        synonym_map={},
    )


def random_dna(rng: np.random.Generator, length: int) -> str:
    return _BASES[rng.integers(0, 4, size=length)].tobytes().decode("ascii")


def mutate(seq: str, rate: float, rng: np.random.Generator) -> str:
    out = bytearray(seq, "ascii")
    for position in rng.choice(len(seq), size=int(len(seq) * rate), replace=False):
        choices = [b for b in b"ACGT" if b != out[position]]
        out[position] = choices[rng.integers(0, 3)]
    return out.decode("ascii")


def split_contigs(seq: str, n: int) -> list[str]:
    bounds = np.linspace(0, len(seq), n + 1).astype(int)
    return [seq[a:b] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


def write_genome(paths: Paths, genome_id: str, contigs: list[str]) -> Path:
    return write_fasta([(f"{genome_id}_contig{i}", s) for i, s in enumerate(contigs, 1)], paths.genome_fasta(genome_id))


def write_mash_tsv(paths: Paths, genome_id: str, distances: dict[str, float], *, header: bool) -> Path:
    path = paths.interim_dir(genome_id) / qc.MASH_TSV_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(qc.MASH_TSV_COLUMNS)] if header else []
    for species, distance in distances.items():
        lines.append(f"data/raw/references/{species}.fasta\t{genome_id}.fasta\t{distance}\t0\t500/1000")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261003)


# ---------------------------------------------------------------------- stats
class TestAssemblyStats:
    @pytest.mark.parametrize(
        ("lengths", "expected"),
        [
            ([10, 20, 30, 40], 30),
            ([2, 2, 2, 3, 3, 4, 8, 8], 8),
            ([7], 7),
            ([5, 5], 5),
            ([1, 1, 1, 1, 100], 100),
            ([], None),
            ([0, 0], None),
        ],
    )
    def test_n50(self, lengths: list[int], expected: int | None) -> None:
        assert qc.n50(lengths) == expected

    def test_counts_lengths_n50_and_gc(self, tmp_path: Path) -> None:
        contigs = ["G" * 4 + "C" * 4 + "A" * 4 + "T" * 4, "GC" * 5, "AT" * 2 + "NN"]
        path = write_fasta([(f"c{i}", s) for i, s in enumerate(contigs)], tmp_path / "g.fasta")
        stats = qc.assembly_stats(path)
        assert stats["n_contigs"] == 3
        assert stats["total_length"] == 32
        assert stats["n50"] == 16
        assert stats["gc_percent"] == pytest.approx(60.0)  # 18 GC over 30 unambiguous bases

    def test_lowercase_and_wrapped_sequences(self, tmp_path: Path) -> None:
        path = tmp_path / "g.fasta"
        path.write_text(">c1\nggcc\naatt\n>c2\nGGGG\n")
        stats = qc.assembly_stats(path)
        assert stats == {"n_contigs": 2, "total_length": 12, "n50": 8, "gc_percent": pytest.approx(100 * 8 / 12)}

    def test_empty_fasta_gives_nulls_not_zero_sentinels(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.fasta"
        path.write_text("")
        stats = qc.assembly_stats(path)
        assert stats["n_contigs"] == 0
        assert stats["total_length"] == 0
        assert stats["n50"] is None
        assert stats["gc_percent"] is None


# ------------------------------------------------------------------- mash tsv
class TestMashTsv:
    @pytest.mark.parametrize(
        ("reference", "expected"),
        [
            ("data/raw/references/KPNEU.fasta", "KPNEU"),
            ("KPNEU.fasta", "KPNEU"),
            ("kpneu.fna.gz", "KPNEU"),
            ("/abs/path/ECOLI.msh", "ECOLI"),
            ("ECOLI", "ECOLI"),
            ("GCF_000240185.1.fasta", None),
            ("SAUR.fasta", None),
        ],
    )
    def test_reference_to_species(self, reference: str, expected: str | None) -> None:
        assert qc.reference_to_species(reference, ("KPNEU", "ECOLI")) == expected

    def test_headerless_table_picks_nearest(self, tmp_path: Path) -> None:
        paths = Paths(root=tmp_path)
        path = write_mash_tsv(paths, "g1", {"ECOLI": 0.21, "KPNEU": 0.004}, header=False)
        assert qc.species_from_mash_tsv(path, ("KPNEU", "ECOLI")) == ("KPNEU", pytest.approx(0.004))

    def test_table_with_header(self, tmp_path: Path) -> None:
        paths = Paths(root=tmp_path)
        path = write_mash_tsv(paths, "g1", {"KPNEU": 0.3, "ECOLI": 0.01}, header=True)
        assert qc.species_from_mash_tsv(path, ("KPNEU", "ECOLI")) == ("ECOLI", pytest.approx(0.01))

    def test_unknown_references_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "mash.tsv"
        path.write_text("other.fasta\tq\t0.001\t0\t1000/1000\nKPNEU.fasta\tq\t0.02\t0\t900/1000\n")
        assert qc.species_from_mash_tsv(path, ("KPNEU",)) == ("KPNEU", pytest.approx(0.02))
        path.write_text("other.fasta\tq\t0.001\t0\t1000/1000\n")
        assert qc.species_from_mash_tsv(path, ("KPNEU",)) == (None, None)

    def test_malformed_rows_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "mash.tsv"
        path.write_text("KPNEU.fasta\tq\tnot_a_number\t0\t1\nECOLI.fasta\tq\n\nKPNEU.fasta\tq\t0.03\t0\t1\n")
        assert qc.read_mash_tsv(path) == [("KPNEU.fasta", 0.03)]
        path.write_text("")
        assert qc.species_from_mash_tsv(path, ("KPNEU",)) == (None, None)


# --------------------------------------------------------------- sketch backend
class TestSketchBackend:
    def test_no_references_returns_none(self, tmp_path: Path) -> None:
        assert qc.build_reference_sketches(Paths(root=tmp_path), make_config()) is None

    def test_identifies_nearest_reference(self, tmp_path: Path, rng: np.random.Generator) -> None:
        paths = Paths(root=tmp_path)
        config = make_config()
        refs = {key: random_dna(rng, GENOME_LENGTH) for key in ("KPNEU", "ECOLI")}
        for key, seq in refs.items():
            write_fasta([(key, seq)], paths.reference_fasta(key))
        references = qc.build_reference_sketches(paths, config)
        assert references is not None
        assert references.species == ("KPNEU", "ECOLI")
        assert references.sketches.shape == (2, references.sketch_size)

        near = write_fasta([("q", mutate(refs["KPNEU"], 0.01, rng))], tmp_path / "near.fasta")
        species, distance = qc.species_from_sketch(near, references)
        assert species == "KPNEU"
        assert 0.0 < distance < 0.05

        far = write_fasta([("q", random_dna(rng, GENOME_LENGTH))], tmp_path / "far.fasta")
        _species, distance = qc.species_from_sketch(far, references)
        assert distance > 0.05

        identical = write_fasta([("q", refs["ECOLI"])], tmp_path / "same.fasta")
        assert qc.species_from_sketch(identical, references) == ("ECOLI", 0.0)

    def test_species_from_mash_backend_order(self, tmp_path: Path, rng: np.random.Generator) -> None:
        paths = Paths(root=tmp_path)
        config = make_config()
        ref = random_dna(rng, GENOME_LENGTH)
        write_fasta([("KPNEU", ref)], paths.reference_fasta("KPNEU"))
        fasta = write_genome(paths, "g1", [ref])
        references = qc.build_reference_sketches(paths, config)
        calls: list[str] = []

        def lazy() -> qc.ReferenceSketches | None:
            calls.append("built")
            return references

        # mash.tsv present and usable -> tsv wins, references never built
        tsv = write_mash_tsv(paths, "g1", {"KPNEU": 0.002}, header=False)
        call = qc.species_from_mash(tsv, fasta, config.species, lazy)
        assert call == qc.SpeciesCall("KPNEU", pytest.approx(0.002), qc.BACKEND_MASH_TSV)
        assert calls == []

        # mash.tsv present but names no known reference -> sketch fallback
        tsv.write_text("unknown.fasta\tg1\t0.001\t0\t1\n")
        call = qc.species_from_mash(tsv, fasta, config.species, lazy)
        assert call.backend == qc.BACKEND_SKETCH
        assert call.mash_species == "KPNEU"
        assert call.mash_distance == 0.0
        assert calls == ["built"]

        # nothing available -> nulls
        assert qc.species_from_mash(None, fasta, config.species, None) == qc.SpeciesCall(None, None, qc.BACKEND_NONE)
        assert qc.species_from_mash(None, None, config.species, lambda: None) == qc.SpeciesCall(None, None, qc.BACKEND_NONE)


# ----------------------------------------------------------------------- rules
class TestRules:
    def good(self) -> dict:
        return dict(n_contigs=3, total_length=GENOME_LENGTH, species="KPNEU", mash_species="KPNEU", mash_distance=0.004)

    def test_pass(self) -> None:
        assert qc.evaluate_rules(**self.good(), config=make_config()) == []
        assert qc.join_reasons([]) is None

    def test_too_fragmented(self) -> None:
        assert qc.evaluate_rules(**{**self.good(), "n_contigs": 6}, config=make_config()) == ["too_fragmented"]
        assert qc.evaluate_rules(**{**self.good(), "n_contigs": 5}, config=make_config()) == []

    @pytest.mark.parametrize("length", [4799, 7201])
    def test_wrong_size_outside_tolerance(self, length: int) -> None:
        assert qc.evaluate_rules(**{**self.good(), "total_length": length}, config=make_config()) == ["wrong_size"]

    @pytest.mark.parametrize("length", [4800, 7200])
    def test_size_bounds_inclusive(self, length: int) -> None:
        assert qc.evaluate_rules(**{**self.good(), "total_length": length}, config=make_config()) == []

    def test_species_mismatch(self) -> None:
        assert qc.evaluate_rules(**{**self.good(), "mash_species": "ECOLI"}, config=make_config()) == ["species_mismatch"]

    def test_too_distant(self) -> None:
        assert qc.evaluate_rules(**{**self.good(), "mash_distance": 0.051}, config=make_config()) == ["too_distant"]
        assert qc.evaluate_rules(**{**self.good(), "mash_distance": 0.05}, config=make_config()) == []

    def test_missing_mash_result_fails_conservatively(self) -> None:
        reasons = qc.evaluate_rules(**{**self.good(), "mash_species": None, "mash_distance": None}, config=make_config())
        assert reasons == ["species_mismatch", "too_distant"]

    def test_all_rules_in_contract_order(self) -> None:
        reasons = qc.evaluate_rules(
            n_contigs=999, total_length=10, species="KPNEU", mash_species="ECOLI", mash_distance=0.4, config=make_config()
        )
        assert reasons == list(qc.FAIL_RULES)
        assert qc.join_reasons(reasons) == "too_fragmented;wrong_size;species_mismatch;too_distant"

    def test_unlabelled_genome_uses_mash_species_for_size(self) -> None:
        config = make_config()
        assert qc.evaluate_rules(**{**self.good(), "species": None}, config=config) == []
        assert qc.evaluate_rules(**{**self.good(), "species": None, "total_length": 100}, config=config) == ["wrong_size"]
        # no label and no mash: size cannot be checked, distance rule still fails
        assert qc.evaluate_rules(
            n_contigs=1, total_length=100, species=None, mash_species=None, mash_distance=None, config=config
        ) == ["too_distant"]

    def test_unknown_species_is_a_contract_violation(self) -> None:
        with pytest.raises(ContractViolation):
            qc.evaluate_rules(**{**self.good(), "species": "SAUR"}, config=make_config())


# ------------------------------------------------------------------------ run
class TestRun:
    def build_dataset(self, tmp_path: Path, rng: np.random.Generator) -> tuple[Paths, Config, dict[str, str]]:
        paths = Paths(root=tmp_path, configs_dir=tmp_path / "configs")
        config = make_config()
        refs = {key: random_dna(rng, GENOME_LENGTH) for key in ("KPNEU", "ECOLI")}
        for key, seq in refs.items():
            write_fasta([(key, seq)], paths.reference_fasta(key))
        kpneu_like = lambda: mutate(refs["KPNEU"], 0.005, rng)  # noqa: E731

        write_genome(paths, "G_ok", split_contigs(kpneu_like(), 3))
        write_mash_tsv(paths, "G_ok", {"KPNEU": 0.004, "ECOLI": 0.3}, header=False)

        write_genome(paths, "G_frag", split_contigs(kpneu_like(), 8))
        write_mash_tsv(paths, "G_frag", {"KPNEU": 0.004, "ECOLI": 0.3}, header=True)

        write_genome(paths, "G_mismatch", split_contigs(kpneu_like(), 2))  # labelled ECOLI, no mash.tsv

        write_genome(paths, "G_small", [kpneu_like()[:3000]])
        write_mash_tsv(paths, "G_small", {"KPNEU": 0.01}, header=False)

        write_genome(paths, "G_far", split_contigs(random_dna(rng, GENOME_LENGTH), 2))  # no mash.tsv

        write_genome(paths, "G_unlabelled", split_contigs(kpneu_like(), 4))  # FASTA only

        label_species = {
            "G_ok": "KPNEU",
            "G_frag": "KPNEU",
            "G_mismatch": "ECOLI",
            "G_nofasta": "KPNEU",
            "G_small": "KPNEU",
            "G_far": "KPNEU",
        }
        labels = pd.DataFrame(
            {
                "genome_id": [g for g in label_species for _ in range(2)],
                "species": [s for s in label_species.values() for _ in range(2)],
                "drug": ["meropenem", "ciprofloxacin"] * len(label_species),
            }
        )
        write_parquet(labels, paths.labels)
        return paths, config, label_species

    def test_end_to_end(self, tmp_path: Path, rng: np.random.Generator) -> None:
        paths, config, label_species = self.build_dataset(tmp_path, rng)
        frame = qc.run(paths, config)

        assert list(frame.columns) == list(qc.QC_COLUMNS)
        assert paths.qc.is_file()
        assert paths.drop_log("qc").is_file()
        by_id = frame.set_index("genome_id")
        assert sorted(by_id.index) == sorted(set(label_species) | {"G_unlabelled"})

        assert by_id.loc["G_ok", "qc_pass"]
        assert pd.isna(by_id.loc["G_ok", "qc_fail_reason"])
        assert by_id.loc["G_ok", "mash_species"] == "KPNEU"
        assert by_id.loc["G_ok", "mash_distance"] == pytest.approx(0.004)
        assert by_id.loc["G_ok", "n_contigs"] == 3
        assert by_id.loc["G_ok", "total_length"] == GENOME_LENGTH

        assert by_id.loc["G_frag", "qc_fail_reason"] == "too_fragmented"
        assert by_id.loc["G_mismatch", "qc_fail_reason"] == "species_mismatch"
        assert by_id.loc["G_mismatch", "mash_species"] == "KPNEU"
        assert by_id.loc["G_mismatch", "species"] == "ECOLI"
        assert by_id.loc["G_small", "qc_fail_reason"] == "wrong_size"
        assert "too_distant" in str(by_id.loc["G_far", "qc_fail_reason"]).split(";")

        nofasta = by_id.loc["G_nofasta"]
        assert not nofasta["qc_pass"]
        assert nofasta["qc_fail_reason"] == "missing_fasta"
        assert nofasta["species"] == "KPNEU"
        for column in ("n_contigs", "total_length", "n50", "gc_percent", "mash_species", "mash_distance"):
            assert pd.isna(nofasta[column])

        unlabelled = by_id.loc["G_unlabelled"]
        assert unlabelled["qc_pass"]
        assert unlabelled["species"] == "KPNEU" == unlabelled["mash_species"]

        assert frame["qc_pass"].dtype == bool
        assert str(frame["n_contigs"].dtype) == "Int64"
        assert str(frame["mash_distance"].dtype) == "Float64"
        assert (frame["qc_pass"] == frame["qc_fail_reason"].isna()).all()

        # parquet round trip stays valid
        reread = read_parquet(paths.qc)
        qc.validate_qc(reread)
        assert len(reread) == len(frame)

        # drop log counts match the table
        log = pd.read_csv(paths.drop_log("qc")).set_index("reason")["n_dropped"]
        reasons = frame.loc[~frame["qc_pass"], "qc_fail_reason"].str.split(";").explode()
        for rule in qc.ALL_REASONS:
            assert log[rule] == int((reasons == rule).sum()), rule
        assert log["missing_fasta"] == 1
        assert log["too_fragmented"] == 1
        assert log["wrong_size"] == 1
        assert log["qc_fail_any"] == 5
        assert set(pd.read_csv(paths.drop_log("qc"))["stage"]) == {"qc"}

    def test_unknown_label_species_raises(self, tmp_path: Path, rng: np.random.Generator) -> None:
        paths, config, _ = self.build_dataset(tmp_path, rng)
        labels = read_parquet(paths.labels)
        labels.loc[labels["genome_id"] == "G_ok", "species"] = "SAUR"
        write_parquet(labels, paths.labels)
        with pytest.raises(ContractViolation, match="SAUR"):
            qc.run(paths, config)

    def test_conflicting_label_species_raises(self, tmp_path: Path, rng: np.random.Generator) -> None:
        paths, config, _ = self.build_dataset(tmp_path, rng)
        labels = read_parquet(paths.labels)
        labels.loc[(labels["genome_id"] == "G_ok") & (labels["drug"] == "meropenem"), "species"] = "ECOLI"
        write_parquet(labels, paths.labels)
        with pytest.raises(ContractViolation, match="more than one species"):
            qc.run(paths, config)

    def test_nothing_to_qc_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            qc.run(Paths(root=tmp_path), make_config())

    def test_genomes_dir_only_without_labels(self, tmp_path: Path, rng: np.random.Generator) -> None:
        paths = Paths(root=tmp_path)
        ref = random_dna(rng, GENOME_LENGTH)
        write_fasta([("KPNEU", ref)], paths.reference_fasta("KPNEU"))
        write_genome(paths, "only", [mutate(ref, 0.002, rng)])
        frame = qc.run(paths, make_config(keys=("KPNEU",)))
        assert list(frame["genome_id"]) == ["only"]
        assert frame["qc_pass"].all()
        assert frame.loc[0, "species"] == "KPNEU"


class TestValidate:
    def frame(self, **overrides: object) -> pd.DataFrame:
        base = {
            "genome_id": ["a", "b"],
            "species": ["KPNEU", "KPNEU"],
            "n_contigs": [3, 900],
            "total_length": [6000, 6000],
            "n50": [2000, 10],
            "gc_percent": [50.0, 50.0],
            "mash_species": ["KPNEU", "KPNEU"],
            "mash_distance": [0.001, 0.001],
            "qc_pass": [True, False],
            "qc_fail_reason": [None, "too_fragmented"],
        }
        base.update(overrides)
        frame = pd.DataFrame(base)
        frame["qc_fail_reason"] = pd.array(base["qc_fail_reason"], dtype="str")
        frame["qc_pass"] = np.asarray(base["qc_pass"], dtype=bool)
        return frame

    def test_valid_frame_passes(self) -> None:
        qc.validate_qc(self.frame())

    def test_reason_on_passing_row(self) -> None:
        with pytest.raises(ContractViolation):
            qc.validate_qc(self.frame(qc_fail_reason=["wrong_size", "too_fragmented"]))

    def test_missing_reason_on_failing_row(self) -> None:
        with pytest.raises(ContractViolation):
            qc.validate_qc(self.frame(qc_fail_reason=[None, None]))

    def test_unknown_reason(self) -> None:
        with pytest.raises(ContractViolation):
            qc.validate_qc(self.frame(qc_fail_reason=[None, "too_fragmented;bogus"]))

    def test_duplicate_genome_id(self) -> None:
        with pytest.raises(ContractViolation):
            qc.validate_qc(self.frame(genome_id=["a", "a"]))

    def test_missing_column(self) -> None:
        with pytest.raises(ContractViolation):
            qc.validate_qc(self.frame().drop(columns=["n50"]))
