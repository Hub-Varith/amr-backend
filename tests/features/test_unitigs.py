"""Tests for ``genome2mic.features.unitigs`` (k-mer backend, patterns, build, query).

Fixture genomes are tiny synthetic FASTAs written to ``tmp_path``: a shared 2 kb
core contig in every genome plus accessory blocks carried by subsets. Block A is
in training genomes g00-g05, block B in g02-g08, a private block D only in g11.
Extra genomes (test split, QC failures, a genome without a QC row, a genome
without a FASTA) exist only in the ``with_extra`` root so the frozen k-mer set
can be compared with and without them on disk. No ``labels.parquet`` is ever
written: the build must not need it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from genome2mic.config import load_config
from genome2mic.droplog import DropLog
from genome2mic.errors import ContractViolation, ToolNotAvailable
from genome2mic.features import unitigs as ug
from genome2mic.io import write_fasta, write_parquet
from genome2mic.paths import Paths

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
SPECIES = "KPNEU"
K = ug.K

_RC = str.maketrans("ACGT", "TGCA")
_BASES = np.frombuffer(b"ACGT", dtype=np.uint8)


def revcomp(seq: str) -> str:
    return seq.translate(_RC)[::-1]


def random_dna(rng: np.random.Generator, n: int) -> str:
    return _BASES[rng.integers(0, 4, size=n)].tobytes().decode("ascii")


def mutate(seq: str, positions: list[int]) -> str:
    """Deterministic SNP at each position (A<->C, G<->T)."""
    swap = {"A": "C", "C": "A", "G": "T", "T": "G"}
    out = list(seq)
    for p in positions:
        out[p] = swap[out[p]]
    return "".join(out)


def canonical_strings(seq: str) -> set[str]:
    return set(ug.decode_kmers(np.unique(ug.encode_kmers(seq, K)), K))


# ---------------------------------------------------------------------------
# Fixture world
# ---------------------------------------------------------------------------
class World:
    """Sequences and genome definitions shared by every root."""

    def __init__(self) -> None:
        rng = np.random.default_rng(20261003)
        self.core = random_dna(rng, 2000)
        self.block_a = random_dna(rng, 300)   # 270 k-mers
        self.block_b = random_dna(rng, 200)   # 170 k-mers
        self.block_c = random_dna(rng, 160)   # only in the QC-failing genome
        self.block_d = random_dna(rng, 150)   # private to g11

        self.train: dict[str, list[str]] = {}
        for i in range(12):
            gid = f"g{i:02d}"
            contigs = [self.core]
            if i <= 5:
                contigs.append(self.block_a)
            if 2 <= i <= 8:
                contigs.append(self.block_b)
            if i == 11:
                contigs.append(self.block_d)
            self.train[gid] = contigs

        self.a_snp = mutate(self.block_a, [50, 150, 250])             # 3 SNPs -> >= 65 % k-mers intact
        self.a_mut = mutate(self.block_a, list(range(5, 300, 10)))    # SNP every 10 bp -> no intact k-mer
        self.test: dict[str, list[str]] = {
            "t_a": [self.core, self.block_a],
            "t_none": [self.core],
            "t_snp": [self.core, self.a_snp],
            "t_mut": [self.core, self.a_mut],
        }
        # Extra train-split genomes that must be excluded from the build.
        self.fail_qc = {"g_fail": [self.core, self.block_a, self.block_c]}
        self.no_qc = {"g_noqc": [self.core, self.block_a]}
        self.no_fasta = ["g_nofasta"]
        # A species with test genomes only (nothing to build).
        self.saur = {"s_t0": [random_dna(rng, 800)]}

    @property
    def a_kmers(self) -> set[str]:
        return canonical_strings(self.block_a)

    @property
    def b_kmers(self) -> set[str]:
        return canonical_strings(self.block_b)

    @property
    def d_kmers(self) -> set[str]:
        return canonical_strings(self.block_d)


@pytest.fixture(scope="module")
def world() -> World:
    return World()


def make_root(tmp_path: Path, world: World, with_extra: bool) -> Paths:
    root = tmp_path / ("with_extra" if with_extra else "train_only")
    paths = Paths(root=root, configs_dir=CONFIGS_DIR)
    genomes: list[tuple[str, str, str, bool | None]] = []  # gid, species, split, qc_pass(None = no qc row)

    def add(gid: str, contigs: list[str] | None, species: str, split: str, qc_pass: bool | None) -> None:
        if contigs is not None:
            write_fasta([(f"{gid}_c{j}", s) for j, s in enumerate(contigs)], paths.genome_fasta(gid))
        genomes.append((gid, species, split, qc_pass))

    for gid, contigs in world.train.items():
        add(gid, contigs, SPECIES, "train", True)
    if with_extra:
        for gid, contigs in world.test.items():
            add(gid, contigs, SPECIES, "test", True)
        for gid, contigs in world.fail_qc.items():
            add(gid, contigs, SPECIES, "train", False)
        for gid, contigs in world.no_qc.items():
            add(gid, contigs, SPECIES, "train", None)
        for gid in world.no_fasta:
            add(gid, None, SPECIES, "train", True)
        for gid, contigs in world.saur.items():
            add(gid, contigs, "SAUR", "test", True)

    splits = pd.DataFrame(
        {
            "genome_id": pd.array([g for g, _, _, _ in genomes], dtype="str"),
            "species": pd.array([s for _, s, _, _ in genomes], dtype="str"),
            "split": pd.array([sp_ for _, _, sp_, _ in genomes], dtype="str"),
            "fold": pd.array([i % 5 if sp_ == "train" else None for i, (_, _, sp_, _) in enumerate(genomes)], dtype="Int64"),
            "external_set": pd.array([None] * len(genomes), dtype="str"),
            "lolo_lineage": pd.array([None] * len(genomes), dtype="str"),
        }
    )
    write_parquet(splits, paths.splits)
    qc_rows = [(g, s, q) for g, s, _, q in genomes if q is not None]
    qc = pd.DataFrame(
        {
            "genome_id": pd.array([g for g, _, _ in qc_rows], dtype="str"),
            "n_contigs": np.array([2] * len(qc_rows), dtype="int64"),
            "total_length": np.array([2300] * len(qc_rows), dtype="int64"),
            "n50": np.array([2000] * len(qc_rows), dtype="int64"),
            "gc_percent": np.array([50.0] * len(qc_rows)),
            "mash_species": pd.array([s for _, s, _ in qc_rows], dtype="str"),
            "mash_distance": np.array([0.001] * len(qc_rows)),
            "qc_pass": np.array([q for _, _, q in qc_rows], dtype=bool),
            "qc_fail_reason": pd.array([None if q else "too_fragmented" for _, _, q in qc_rows], dtype="str"),
        }
    )
    write_parquet(qc, paths.qc)
    assert not paths.labels.exists()  # the build must never need labels
    return paths


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIGS_DIR)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory, world: World, config) -> tuple[Paths, ug.UnitigBuild]:
    paths = make_root(tmp_path_factory.mktemp("unitigs"), world, with_extra=True)
    summary = ug.build(paths, config, SPECIES)
    return paths, summary


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def test_encode_decode_roundtrip_is_canonical() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        s = random_dna(rng, K)
        codes = ug.encode_kmers(s, K)
        assert codes.shape == (1,) and codes.dtype == np.uint64
        assert ug.decode_kmer(int(codes[0]), K) == min(s, revcomp(s))


def test_reverse_complement_gives_same_kmers() -> None:
    seq = random_dna(np.random.default_rng(2), 500)
    fwd = np.unique(ug.encode_kmers(seq, K))
    rev = np.unique(ug.encode_kmers(revcomp(seq), K))
    assert np.array_equal(fwd, rev)
    assert fwd.size == 500 - K + 1  # random sequence: all windows distinct


def test_non_acgt_windows_are_skipped_and_lowercase_accepted() -> None:
    seq = random_dna(np.random.default_rng(3), 200)
    n_all = ug.encode_kmers(seq, K).size
    assert n_all == 200 - K + 1
    with_n = seq[:100] + "N" + seq[101:]
    assert ug.encode_kmers(with_n, K).size == n_all - K  # K windows cover position 100
    assert np.array_equal(ug.encode_kmers(seq.lower(), K), ug.encode_kmers(seq, K))
    assert ug.encode_kmers("ACGT", K).size == 0
    with pytest.raises(ValueError):
        ug.encode_kmers(seq, 33)


def test_genome_kmers_chunking_matches_single_pass(tmp_path: Path) -> None:
    seq = random_dna(np.random.default_rng(4), 3000)
    fasta = tmp_path / "g.fasta"
    write_fasta([("c1", seq[:1500]), ("c2", seq[1500:])], fasta)
    a = ug.genome_kmers(fasta, K)
    b = ug.genome_kmers(fasta, K, chunk_windows=97)
    assert np.array_equal(a, b)
    assert np.all(a[1:] > a[:-1])
    expected = np.unique(np.concatenate([ug.encode_kmers(seq[:1500], K), ug.encode_kmers(seq[1500:], K)]))
    assert np.array_equal(a, expected)


def test_kmer_counter_incremental_merge_matches_np_unique() -> None:
    rng = np.random.default_rng(5)
    genomes = [np.unique(rng.integers(0, 400, size=rng.integers(20, 120)).astype(np.uint64)) for _ in range(15)]
    counter = ug._KmerCounter(buffer_elements=64)  # forces many flushes
    for g in genomes:
        counter.add(g)
    kmers, counts = counter.finish()
    exp_k, exp_c = np.unique(np.concatenate(genomes), return_counts=True)
    assert np.array_equal(kmers, exp_k)
    assert np.array_equal(counts, exp_c)
    assert counter.n_genomes == 15


# ---------------------------------------------------------------------------
# Pattern collapse
# ---------------------------------------------------------------------------
def test_collapse_patterns_groups_identical_columns() -> None:
    dense = np.array(
        [
            [1, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    pattern_col, rep = ug.collapse_patterns(sp.csr_matrix(dense))
    assert pattern_col.tolist() == [0, 0, 1, 0, 1, 2]
    assert rep.tolist() == [0, 2, 5]
    assert pattern_col.dtype == np.int32
    pc_empty, rep_empty = ug.collapse_patterns(sp.csr_matrix((4, 0), dtype=np.int8))
    assert pc_empty.size == 0 and rep_empty.size == 0


# ---------------------------------------------------------------------------
# Build on the fixture world
# ---------------------------------------------------------------------------
def test_block_kmers_form_one_pattern_and_core_is_filtered(built, world: World) -> None:
    paths, summary = built
    ks = ug.load_kmer_set(summary.kmers_path)
    decoded = ug.decode_kmers(ks.kmers, ks.k)
    col_of = dict(zip(decoded, ks.pattern_col.tolist()))

    a_cols = {col_of[s] for s in world.a_kmers}
    assert len(a_cols) == 1, "block A k-mers must share one pattern"
    (a_col,) = a_cols
    b_cols = {col_of[s] for s in world.b_kmers}
    assert len(b_cols) == 1 and b_cols != a_cols

    index = summary.index.set_index("col_index")
    assert index.loc[a_col, "n_unitigs"] == len(world.a_kmers) == 270
    assert index.loc[a_col, "train_frequency"] == pytest.approx(6 / 12)
    assert set(index.loc[a_col, "unitig_sequences"]) == world.a_kmers
    assert index.loc[a_col, "pattern_id"] == ug.pattern_id(a_col)

    core = canonical_strings(world.core)
    assert not (core & set(decoded)), "core k-mers (100 % of training genomes) must be filtered"
    assert not (canonical_strings(world.block_c) & set(decoded)), "QC-failing genome must not contribute"
    assert summary.n_kmers_total > summary.n_kmers_kept > 0
    assert summary.n_patterns == 3  # A, B, private D


def test_kmer_set_identical_with_and_without_test_genomes(tmp_path: Path, world: World, config) -> None:
    p_only = make_root(tmp_path, world, with_extra=False)
    p_extra = make_root(tmp_path, world, with_extra=True)
    s_only = ug.build(p_only, config, SPECIES)
    s_extra = ug.build(p_extra, config, SPECIES)
    k_only = ug.load_kmer_set(s_only.kmers_path)
    k_extra = ug.load_kmer_set(s_extra.kmers_path)
    assert np.array_equal(k_only.kmers, k_extra.kmers)
    assert np.array_equal(k_only.pattern_col, k_extra.pattern_col)
    assert k_only.sha1() == k_extra.sha1() == s_only.kmer_set_sha1
    assert s_only.n_queried == 0 and s_extra.n_queried == 4
    # The built part of the matrix is identical too.
    m_only = sp.load_npz(s_only.matrix_path)
    m_extra = sp.load_npz(s_extra.matrix_path)
    assert (m_only != m_extra[: m_only.shape[0]]).nnz == 0


def test_rows_roles_and_leakage_check(built, world: World) -> None:
    paths, summary = built
    rows = pd.read_parquet(summary.rows_path)
    assert list(rows.columns) == ["row_index", "genome_id", "split", "role"]
    assert rows["row_index"].tolist() == list(range(len(rows)))
    built_rows = rows[rows["role"] == ug.ROLE_BUILT]
    assert set(built_rows["genome_id"]) == set(world.train)
    assert (built_rows["split"] == "train").all()
    queried = rows[rows["role"] == ug.ROLE_QUERIED]
    assert set(queried["genome_id"]) == set(world.test)
    assert (queried["split"] == "test").all()
    # Excluded genomes never appear: QC fail, no QC row, missing FASTA, other species.
    assert not ({"g_fail", "g_noqc", "g_nofasta", "s_t0"} & set(rows["genome_id"]))
    assert summary.n_train == 12 and summary.n_queried == 4


def test_queried_genomes_get_right_pattern_bits(built, world: World) -> None:
    paths, summary = built
    matrix, rows, index = ug.load_unitigs(paths, SPECIES)
    ks = ug.load_kmer_set(summary.kmers_path)
    col_of = dict(zip(ug.decode_kmers(ks.kmers, ks.k), ks.pattern_col.tolist()))
    a_col = col_of[next(iter(world.a_kmers))]
    b_col = col_of[next(iter(world.b_kmers))]
    d_col = col_of[next(iter(world.d_kmers))]
    row_of = dict(zip(rows["genome_id"], rows["row_index"]))

    def bits(gid: str) -> tuple[int, int, int]:
        r = matrix[row_of[gid]].toarray().ravel()  # one row only; fine in a test
        return int(r[a_col]), int(r[b_col]), int(r[d_col])

    assert bits("t_a") == (1, 0, 0)
    assert bits("t_none") == (0, 0, 0)
    assert bits("t_snp") == (1, 0, 0), "3 SNPs keep >= 50 % of block-A k-mers -> pattern present"
    assert bits("t_mut") == (0, 0, 0), "a SNP every 10 bp destroys every 31-mer -> pattern absent"
    # Built rows are exact.
    assert bits("g00") == (1, 0, 0)
    assert bits("g03") == (1, 1, 0)
    assert bits("g08") == (0, 1, 0)
    assert bits("g11") == (0, 0, 1)


def test_query_genome_matches_matrix_rows_in_both_call_forms(built, world: World) -> None:
    paths, summary = built
    matrix, rows, _ = ug.load_unitigs(paths, SPECIES)
    row_of = dict(zip(rows["genome_id"], rows["row_index"]))
    for gid in ("t_snp", "g03", "t_none"):
        expected = matrix[row_of[gid]].toarray().ravel().astype(np.int8)
        v1 = ug.query_genome(summary.kmers_path, paths.genome_fasta(gid))
        v2 = ug.query_genome(paths, SPECIES, paths.genome_fasta(gid))
        assert v1.dtype == np.int8 and v1.shape == (summary.n_patterns,)
        assert np.array_equal(v1, expected)
        assert np.array_equal(v2, expected)
    with pytest.raises(TypeError):
        ug.query_genome(paths, SPECIES)
    with pytest.raises(TypeError):
        ug.query_genome(summary.kmers_path, paths.genome_fasta("t_a"), paths.genome_fasta("t_a"))


def test_matrix_is_sparse_int8_and_files_consistent(built) -> None:
    paths, summary = built
    matrix = sp.load_npz(summary.matrix_path)
    assert sp.isspmatrix_csr(matrix) and matrix.dtype == np.int8
    assert matrix.shape == (16, summary.n_patterns)
    assert set(np.unique(matrix.data).tolist()) == {1}
    index = pd.read_parquet(summary.index_path)
    assert list(index.columns) == ["col_index", "pattern_id", "n_unitigs", "unitig_sequences", "train_frequency"]
    assert index["col_index"].tolist() == list(range(summary.n_patterns))
    assert index["pattern_id"].tolist() == [ug.pattern_id(i) for i in range(summary.n_patterns)]
    assert int(index["n_unitigs"].sum()) == summary.n_kmers_kept
    with np.load(summary.kmers_path) as z:
        assert str(z["species"]) == SPECIES and str(z["sha1"]) == summary.kmer_set_sha1
        assert int(z["n_patterns"]) == summary.n_patterns


def test_frequency_filter_bounds_are_inclusive_and_tunable(tmp_path: Path, world: World, config) -> None:
    paths = make_root(tmp_path, world, with_extra=False)
    # Default 1 %: the private block (1/12 = 8.3 %) survives.
    default = ug.build(paths, config, SPECIES)
    decoded = set(ug.decode_kmers(ug.load_kmer_set(default.kmers_path).kmers, K))
    assert world.d_kmers <= decoded
    # min_freq 10 %: it is dropped and the drop log says so.
    log = DropLog("unitigs")
    strict = ug.build(paths, config, SPECIES, backend=ug.KmerBackend(min_freq=0.10), droplog=log)
    decoded_strict = set(ug.decode_kmers(ug.load_kmer_set(strict.kmers_path).kmers, K))
    assert not (world.d_kmers & decoded_strict)
    assert strict.n_patterns == 2
    reasons = {r.reason: r.n_dropped for r in log.records}
    assert reasons["kmer_below_min_freq"] == len(world.d_kmers)
    assert reasons["kmer_above_max_freq"] >= len(canonical_strings(world.core))
    # Exactly-at-threshold is kept: max_freq = 6/12 keeps block A (6/12) and drops B (7/12).
    edge = ug.build(paths, config, SPECIES, backend=ug.KmerBackend(max_freq=0.5))
    decoded_edge = set(ug.decode_kmers(ug.load_kmer_set(edge.kmers_path).kmers, K))
    assert world.a_kmers <= decoded_edge and not (world.b_kmers & decoded_edge)
    with pytest.raises(ValueError):
        ug.KmerBackend(min_freq=0.5, max_freq=0.2)


def test_build_rejects_non_train_rows_marked_built(tmp_path: Path, world: World, config, monkeypatch) -> None:
    paths = make_root(tmp_path, world, with_extra=True)

    class LeakyBackend(ug.KmerBackend):
        name = "leaky"

    original = ug.training_and_query_genomes

    def leak(paths_, species, droplog):
        train, query, genomes = original(paths_, species, droplog)
        train = dict(train)
        train.update(query)  # sneak test genomes into the build set
        return train, {}, genomes

    monkeypatch.setattr(ug, "training_and_query_genomes", leak)
    with pytest.raises(ContractViolation):
        ug.build(paths, config, SPECIES, backend=LeakyBackend())


def test_run_builds_species_with_training_genomes_and_writes_droplog(tmp_path: Path, world: World, config) -> None:
    paths = make_root(tmp_path, world, with_extra=True)
    summary = ug.run(paths, config)
    assert summary["species"].tolist() == ["KPNEU", "SAUR"]
    kp = summary.set_index("species").loc["KPNEU"]
    assert kp["n_train"] == 12 and kp["n_queried"] == 4 and kp["n_patterns"] == 3
    sa = summary.set_index("species").loc["SAUR"]
    assert sa["n_train"] == 0 and pd.isna(sa["kmer_set_sha1"])
    assert not paths.unitigs("SAUR").exists()
    for p in (paths.unitigs("KPNEU"), paths.unitig_rows("KPNEU"), paths.unitig_index("KPNEU"), paths.unitig_kmers("KPNEU")):
        assert p.exists()
    drops = pd.read_csv(paths.drop_log("unitigs"))
    by_reason = drops.groupby("reason")["n_dropped"].sum()
    assert by_reason["qc_fail"] == 1
    assert by_reason["no_qc_row"] == 1
    assert by_reason["fasta_missing"] == 1
    assert by_reason["kmer_above_max_freq"] > 0
    assert "kmer_below_min_freq" in by_reason.index
    assert (drops["stage"] == "unitigs").all()
    # Explicit species list and unknown species.
    again = ug.run(paths, config, species=["kpneu"])
    assert again["species"].tolist() == ["KPNEU"]
    with pytest.raises(ValueError):
        ug.build(paths, config, "NOPE")


def test_build_does_not_read_labels(tmp_path: Path, world: World, config, monkeypatch) -> None:
    paths = make_root(tmp_path, world, with_extra=True)
    assert not paths.labels.exists()
    real_read = ug.read_parquet

    def guarded(path, columns=None):
        assert Path(path).name != "labels.parquet", "unitig build must never open labels.parquet"
        return real_read(path, columns=columns)

    monkeypatch.setattr(ug, "read_parquet", guarded)
    ug.build(paths, config, SPECIES)


def test_empty_species_raises(tmp_path: Path, world: World, config) -> None:
    paths = make_root(tmp_path, world, with_extra=True)
    with pytest.raises(ValueError, match="no QC-passing training genomes"):
        ug.build(paths, config, "SAUR")


# ---------------------------------------------------------------------------
# KmerSet persistence and sequence decomposition
# ---------------------------------------------------------------------------
def test_kmer_set_roundtrip_and_validation(tmp_path: Path) -> None:
    kmers = np.array([3, 9, 27, 81], dtype=np.uint64)
    ks = ug.KmerSet(kmers, np.array([1, 0, 1, 1], dtype=np.int32), n_patterns=2, species="KPNEU")
    path = tmp_path / "set.npz"
    ug.save_kmer_set(path, ks)
    back = ug.load_kmer_set(path)
    assert np.array_equal(back.kmers, kmers) and np.array_equal(back.pattern_col, ks.pattern_col)
    assert back.n_patterns == 2 and back.k == K and back.species == "KPNEU"
    assert back.pattern_sizes().tolist() == [1, 3]
    assert back.pattern_ids() == ["u_000000", "u_000001"]
    assert back.sha1() == ks.sha1()
    with pytest.raises(ValueError):
        ug.KmerSet(np.array([9, 3], dtype=np.uint64), np.array([0, 0], dtype=np.int32), 1)  # unsorted
    with pytest.raises(ValueError):
        ug.KmerSet(kmers, np.array([0, 0, 0, 5], dtype=np.int32), 2)  # column out of range


def test_kmer_set_from_sequences_and_query(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    u1, u2 = random_dna(rng, 60), random_dna(rng, 45)
    ks = ug.KmerSet.from_sequences([u1, u2], np.array([0, 1]), n_patterns=2)
    assert ks.pattern_sizes().tolist() == [30, 15]
    assert ks.backend == "unitig-caller"
    fasta = tmp_path / "q.fasta"
    write_fasta([("c", random_dna(rng, 300) + u2 + random_dna(rng, 100))], fasta)
    assert ug.query_kmer_set(ks, fasta).tolist() == [0, 1]


# ---------------------------------------------------------------------------
# unitig-caller backend: not available + parsers
# ---------------------------------------------------------------------------
def test_unitig_caller_backend_raises_when_not_on_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)
    be = ug.UnitigCallerBackend()
    assert not be.available()
    fasta = tmp_path / "x.fasta"
    write_fasta([("c", "ACGT" * 20)], fasta)
    with pytest.raises(ToolNotAvailable) as exc:
        be.build({"x": fasta}, DropLog("unitigs"))
    assert exc.value.tool == "unitig-caller"
    with pytest.raises(ToolNotAvailable):
        be.query_cli({"x": fasta}, ["ACGT" * 10])
    assert isinstance(ug.make_backend("auto"), ug.KmerBackend)
    assert isinstance(ug.make_backend("unitig-caller"), ug.UnitigCallerBackend)
    with pytest.raises(ValueError):
        ug.make_backend("bifrost")


def test_parse_rtab_and_pyseer(tmp_path: Path) -> None:
    rtab = tmp_path / "u.rtab"
    rtab.write_text("Unitig_sequence\tsA\tsB\tsC\nACGTACGT\t1\t0\t1\nTTTTGGGG\t0\t0\t1\n")
    seqs, samples, presence = ug.parse_rtab(rtab)
    assert seqs == ["ACGTACGT", "TTTTGGGG"] and samples == ["sA", "sB", "sC"]
    assert presence.shape == (3, 2) and presence.dtype == np.int8
    assert presence.toarray().tolist() == [[1, 0], [0, 0], [1, 1]]

    pyseer = tmp_path / "u.pyseer"
    pyseer.write_text("ACGTACGT | sA:1 sC:1\nTTTTGGGG | sC:1\n")
    seqs2, samples2, presence2 = ug.parse_pyseer(pyseer, samples=["sA", "sB", "sC"])
    assert seqs2 == seqs and samples2 == samples
    assert (presence2 != presence).nnz == 0
    with pytest.raises(ValueError):
        ug.parse_pyseer(pyseer, samples=["sA"])


def test_unitig_caller_finish_filters_and_collapses(tmp_path: Path) -> None:
    """The post-processing shared with the CLI path works on a parsed table (no CLI needed)."""
    rng = np.random.default_rng(7)
    seqs = [random_dna(rng, 40) for _ in range(4)]
    # 4 samples x 4 unitigs: u0 and u1 identical pattern, u2 everywhere (dropped), u3 nowhere (dropped)
    dense = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=np.int8)
    be = ug.UnitigCallerBackend()
    log = DropLog("unitigs")
    gids = ["a", "b", "c", "d"]
    result = be._finish(seqs, gids, sp.csr_matrix(dense), {g: g for g in gids}, gids, log)
    assert result.n_kmers_total == 4 and result.n_kmers_kept == 2
    assert result.kmer_set.n_patterns == 1
    assert result.matrix.toarray().ravel().tolist() == [1, 1, 0, 0]
    assert result.train_frequency.tolist() == [0.5]
    assert result.kmer_set.n_kmers == 20  # 2 unitigs x 10 k-mers
    reasons = {r.reason: r.n_dropped for r in log.records}
    assert reasons == {"unitig_below_min_freq": 1, "unitig_above_max_freq": 1}
