from genome2mic.predict.novelty_checker import NoveltyChecker
from genome2mic.predict.species_identifier import SpeciesIdentifier

KEYS = ["ECOLI", "KPNEU", "SAUR", "PAER", "ABAU"]
MASH = (
    "refs/ECOLI.fna.gz\tupload.fasta\t0.203\t0\t12/1000\n"
    "refs/KPNEU.fna.gz\tupload.fasta\t0.0112\t0\t801/1000\n"
    "refs/SAUR.fna.gz\tupload.fasta\t1\t1\t0/1000\n"
)


def test_nearest_reference_names_the_species():
    result = SpeciesIdentifier.parse(MASH, KEYS)
    assert result.species == "KPNEU" and abs(result.mash_distance - 0.0112) < 1e-12


def test_too_far_from_every_reference_is_not_covered():
    result = SpeciesIdentifier.parse("refs/KPNEU.fna.gz\tq\t0.09\t0\t5/1000\n", KEYS)
    assert result.species is None and result.mash_distance == 0.09


def test_empty_mash_output():
    assert SpeciesIdentifier.parse("", KEYS).species is None


class FakeRunner:
    def __init__(self, output):
        self.output = output

    def run(self, arguments, cwd=None):
        return self.output


def test_novelty_without_a_training_sketch_is_unchecked(tmp_path):
    result = NoveltyChecker(tmp_path, FakeRunner("")).check(tmp_path / "x.fasta", "KPNEU")
    assert result.nearest_training_distance is None and result.in_range and not result.checked


def test_novelty_uses_the_nearest_training_genome(tmp_path):
    (tmp_path / "training_KPNEU.msh").write_text("sketch")
    output = "g1\tq\t0.031\t0\t1/1000\ng2\tq\t0.004\t0\t9/1000\n"
    result = NoveltyChecker(tmp_path, FakeRunner(output)).check(tmp_path / "x.fasta", "KPNEU")
    assert result.nearest_training_distance == 0.004 and result.in_range and result.checked
    far = NoveltyChecker(tmp_path, FakeRunner("g1\tq\t0.04\t0\t1/1000\n")).check(tmp_path / "x.fasta", "KPNEU")
    assert not far.in_range
