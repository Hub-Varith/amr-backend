"""Tests for ``genome2mic.splits.lineages`` on small synthetic genomes.

Fixture design (all sequences are random DNA from a fixed seed):

* Backbone ``A`` (KPNEU): genomes ``A1..A4`` each carry 10 private SNPs
  (Mash d ~ 6e-4 to the backbone, ~1e-3 to each other -- well inside 0.005);
  ``A1_dup`` is an identical copy of ``A1`` (d = 0, the outbreak rule);
  ``A1_snp`` is ``A1`` with one SNP (same lineage).
* Backbone ``B`` (KPNEU): ``B1..B4`` plus ``B5`` which has no label row (species
  comes from ``qc.mash_species``). ``A`` and ``B`` are unrelated (d = 1.0), so
  two clusters are obvious.
* Backbone ``C`` (ECOLI): ``C1, C2`` -- a second species, numbering restarts.
* ``QCFAIL`` fails QC, ``NOFASTA`` passes QC but has no FASTA, ``SAURX`` is a
  species outside the test config. None of them may reach the output.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from genome2mic import sketch as sk
from genome2mic.errors import ContractViolation, ToolNotAvailable
from genome2mic.io import write_fasta, write_parquet
from genome2mic.paths import Paths
from genome2mic.splits import lineages as ln

BACKBONE_LEN = 20_000
SKETCH_SIZE = 500
_BASES = np.frombuffer(b"ACGT", dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def random_dna(rng: np.random.Generator, length: int) -> str:
    return _BASES[rng.integers(0, 4, size=length)].tobytes().decode("ascii")


def mutate(seq: str, n_snps: int, rng: np.random.Generator) -> str:
    out = bytearray(seq, "ascii")
    for pos in rng.choice(len(seq), size=n_snps, replace=False):
        choices = [b for b in b"ACGT" if b != out[pos]]
        out[pos] = choices[rng.integers(0, 3)]
    return out.decode("ascii")


def write_genome(paths: Paths, gid: str, seq: str, n_contigs: int = 3) -> None:
    step = len(seq) // n_contigs
    records = [(f"{gid}_contig{i}", seq[i * step : (i + 1) * step if i < n_contigs - 1 else len(seq)]) for i in range(n_contigs)]
    write_fasta(records, paths.genome_fasta(gid))


@dataclass
class FakeConfig:
    """Only ``.species`` keys are read by the lineages stage."""

    species: dict[str, object] = field(default_factory=lambda: {"KPNEU": object(), "ECOLI": object()})


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(20261003)


@pytest.fixture(scope="module")
def genomes(rng: np.random.Generator) -> dict[str, tuple[str, str]]:
    """``gid -> (species, sequence)`` for every genome that gets a FASTA."""
    a, b, c = (random_dna(rng, BACKBONE_LEN) for _ in range(3))
    g: dict[str, tuple[str, str]] = {}
    for i in range(1, 5):
        g[f"A{i}"] = ("KPNEU", mutate(a, 10, rng))
        g[f"B{i}"] = ("KPNEU", mutate(b, 10, rng))
    g["A1_dup"] = ("KPNEU", g["A1"][1])
    g["A1_snp"] = ("KPNEU", mutate(g["A1"][1], 1, rng))
    g["B5"] = ("KPNEU", mutate(b, 10, rng))  # no label row; species from mash_species
    g["C1"] = ("ECOLI", mutate(c, 10, rng))
    g["C2"] = ("ECOLI", mutate(c, 10, rng))
    g["QCFAIL"] = ("KPNEU", mutate(b, 10, rng))
    g["SAURX"] = ("SAUR", random_dna(rng, BACKBONE_LEN))
    return g


@pytest.fixture
def root(tmp_path: Path, genomes: dict[str, tuple[str, str]]) -> Paths:
    paths = Paths(root=tmp_path, configs_dir=tmp_path / "configs")
    for gid, (_, seq) in genomes.items():
        write_genome(paths, gid, seq)

    qc_ids = list(genomes) + ["NOFASTA"]
    qc = pd.DataFrame(
        {
            "genome_id": pd.array(qc_ids, dtype="str"),
            "n_contigs": np.full(len(qc_ids), 3, dtype=np.int64),
            "total_length": np.full(len(qc_ids), BACKBONE_LEN, dtype=np.int64),
            "n50": np.full(len(qc_ids), BACKBONE_LEN // 3, dtype=np.int64),
            "gc_percent": np.full(len(qc_ids), 50.0),
            "mash_species": pd.array([genomes.get(g, ("KPNEU", ""))[0] for g in qc_ids], dtype="str"),
            "mash_distance": np.full(len(qc_ids), 0.001),
            "qc_pass": np.array([g != "QCFAIL" for g in qc_ids], dtype=bool),
            "qc_fail_reason": pd.array([None if g != "QCFAIL" else "too_fragmented" for g in qc_ids], dtype="str"),
        }
    )
    write_parquet(qc, paths.qc)

    labelled = [g for g in genomes if g != "B5"] + ["NOFASTA"]
    labels = pd.DataFrame(
        {
            "genome_id": pd.array(labelled, dtype="str"),
            "biosample": pd.array([f"SAMN{i:08d}" for i in range(len(labelled))], dtype="str"),
            "species": pd.array([genomes.get(g, ("KPNEU", ""))[0] for g in labelled], dtype="str"),
            "drug": pd.array(["meropenem"] * len(labelled), dtype="str"),
            "mic_lower": np.full(len(labelled), 4.0),
            "mic_upper": np.full(len(labelled), 8.0),
            "censor": pd.array(["interval"] * len(labelled), dtype="str"),
            "sir": pd.array(["R"] * len(labelled), dtype="str"),
        }
    )
    write_parquet(labels, paths.labels)

    # mlst.tsv in the three layouts the parser must accept.
    (paths.interim_dir("A1")).mkdir(parents=True)
    (paths.interim_dir("A1") / "mlst.tsv").write_text("FILE\tSCHEME\tST\tgapA\tinfB\nA1.fasta\tkpneumoniae\t258\t3\t3\n")
    (paths.interim_dir("A2")).mkdir(parents=True)
    (paths.interim_dir("A2") / "mlst.tsv").write_text("A2.fasta\tkpneumoniae\t11\tgapA(3)\tinfB(3)\n")
    (paths.interim_dir("A3")).mkdir(parents=True)
    (paths.interim_dir("A3") / "mlst.tsv").write_text("A3.fasta\tkpneumoniae\t-\tgapA(3)\tinfB(~3)\n")
    return paths


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_single_linkage_hand_matrix_and_inclusive_threshold() -> None:
    # 0-1 linked at 0.001, 1-2 at exactly the threshold (inclusive), 3-4 at 0.002, chain-break 2-3 at 0.5.
    D = np.full((5, 5), 0.5)
    np.fill_diagonal(D, 0.0)
    for i, j, d in ((0, 1, 0.001), (1, 2, 0.005), (3, 4, 0.002)):
        D[i, j] = D[j, i] = d
    numbers = ln.single_linkage_clusters(D, threshold=0.005)
    assert numbers.tolist() == [1, 1, 1, 2, 2]  # largest cluster is number 1
    numbers_strict = ln.single_linkage_clusters(D, threshold=0.0049)
    assert numbers_strict[0] == numbers_strict[1] != numbers_strict[2]
    assert numbers_strict[3] == numbers_strict[4]


def test_single_linkage_equals_connected_components_of_threshold_graph() -> None:
    rng = np.random.default_rng(3)
    n = 40
    D = rng.uniform(0.0, 0.02, size=(n, n))
    D = np.triu(D, 1)
    D = D + D.T
    numbers = ln.single_linkage_clusters(D, threshold=0.005)
    graph = csr_matrix(((D <= 0.005) & ~np.eye(n, dtype=bool)).astype(np.int8))
    n_comp, comp = connected_components(graph, directed=False)
    assert len(set(numbers.tolist())) == n_comp
    pairs = {(numbers[i], comp[i]) for i in range(n)}
    assert len(pairs) == n_comp  # one-to-one relabelling


def test_single_linkage_edge_cases_and_validation() -> None:
    assert ln.single_linkage_clusters(np.zeros((0, 0))).tolist() == []
    assert ln.single_linkage_clusters(np.zeros((1, 1))).tolist() == [1]
    with pytest.raises(ValueError):
        ln.single_linkage_clusters(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        ln.single_linkage_clusters(np.array([[0.0, 0.1], [0.2, 0.0]]))
    with pytest.raises(ValueError):
        ln.single_linkage_clusters(np.zeros((2, 2)), threshold=-1.0)


def test_cluster_label_is_species_prefixed_and_zero_padded() -> None:
    assert ln.cluster_label("KPNEU", 1) == "KPNEU_ML_001"
    assert ln.cluster_label("kpneu", 42) == "KPNEU_ML_042"
    assert ln.cluster_label("ECOLI", 1000) == "ECOLI_ML_1000"
    with pytest.raises(ValueError):
        ln.cluster_label("KPNEU", 0)


def test_check_outbreak_pairs_raises_when_near_identical_pair_is_split() -> None:
    D = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.5, 0.5, 0.0]])
    assert ln.check_outbreak_pairs(D, np.array([1, 1, 2]), ["a", "b", "c"]) == 1
    with pytest.raises(ContractViolation, match="near-identical"):
        ln.check_outbreak_pairs(D, np.array([1, 2, 3]), ["a", "b", "c"])


def test_read_st_layouts(tmp_path: Path) -> None:
    header = tmp_path / "header.tsv"
    header.write_text("FILE\tSCHEME\tST\tgapA\nx.fasta\tkpneumoniae\t258\t3\n")
    headerless = tmp_path / "plain.tsv"
    headerless.write_text("x.fasta\tecoli\t131\tadk(53)\n")
    dash = tmp_path / "dash.tsv"
    dash.write_text("x.fasta\tecoli\t-\tadk(53)\n")
    empty = tmp_path / "empty.tsv"
    empty.write_text("\n")
    short = tmp_path / "short.tsv"
    short.write_text("x.fasta\tecoli\n")
    header_only = tmp_path / "header_only.tsv"
    header_only.write_text("FILE\tSCHEME\tST\n")
    assert ln.read_st(header) == "258"
    assert ln.read_st(headerless) == "131"
    assert ln.read_st(dash) == ln.ST_MISSING
    assert ln.read_st(empty) == ln.ST_MISSING
    assert ln.read_st(short) == ln.ST_MISSING
    assert ln.read_st(header_only) == ln.ST_MISSING
    assert ln.read_st(tmp_path / "missing.tsv") == ln.ST_MISSING


def test_poppunk_backend_is_a_stub(root: Paths) -> None:
    with pytest.raises(ToolNotAvailable, match="poppunk"):
        ln.run(root, FakeConfig(), backend="poppunk")
    with pytest.raises(ValueError):
        ln.run(root, FakeConfig(), backend="kmeans")


# --------------------------------------------------------------------------- #
# Clustering on sketches of fake genomes
# --------------------------------------------------------------------------- #
def test_two_lineages_are_obvious_from_sketches(genomes: dict[str, tuple[str, str]]) -> None:
    ids = ["A1", "A2", "A3", "A4", "A1_dup", "A1_snp", "B1", "B2", "B3", "B4"]
    S = sk.stack_sketches([sk.sketch(genomes[g][1], s=SKETCH_SIZE) for g in ids])
    numbers, close = ln.cluster_species("KPNEU", ids, S, threshold=0.005)
    a_numbers = {int(numbers[i]) for i, g in enumerate(ids) if g.startswith("A")}
    b_numbers = {int(numbers[i]) for i, g in enumerate(ids) if g.startswith("B")}
    assert a_numbers == {1}  # six A genomes -> the largest cluster -> number 1
    assert b_numbers == {2}
    # The returned pairs are exactly the outbreak-radius pairs (d <= 1e-4): A1, A1_dup and the
    # one-SNP A1_snp (a single SNP rarely moves a bottom-500 sketch).
    D = sk.pairwise_distances(S)
    pairs = {(ids[i], ids[j]): d for i, j, d in zip(close.rows, close.cols, close.distances)}
    expected = {(ids[i], ids[j]): D[i, j] for i in range(len(ids)) for j in range(i + 1, len(ids)) if D[i, j] <= 1e-4}
    assert pairs == expected
    assert pairs[("A1", "A1_dup")] == 0.0
    assert D[ids.index("A1"), ids.index("B1")] > 0.5
    assert (D[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])] <= 0.005).all()


# --------------------------------------------------------------------------- #
# Sparse clustering == dense single linkage (#25)
# --------------------------------------------------------------------------- #
def lineage_sketches(rng: np.random.Generator, n: int, s: int = 200, n_families: int = 6) -> np.ndarray:
    """Random bottom-``s`` sketches with family structure and a spread of within-family distances."""
    families = [np.unique(rng.integers(1, 2**63, size=3 * s, dtype=np.uint64)) for _ in range(n_families)]
    rows = []
    for _ in range(n):
        base = families[int(rng.integers(0, n_families))]
        n_swap = int(rng.integers(0, s // 4))  # 0-25 % of hashes replaced -> distances from 0 to ~0.02
        swap = rng.choice(base.size, size=n_swap, replace=False)
        kept = np.delete(base, swap)
        fresh = rng.integers(1, 2**63, size=n_swap, dtype=np.uint64)
        rows.append(np.unique(np.concatenate([kept, fresh]))[:s])
    return sk.stack_sketches(rows)


def edges_from_dense(D: np.ndarray, cut: float) -> list[sk.DistanceEdges]:
    rows, cols = np.nonzero(np.triu(D <= cut, k=1))
    return [sk.DistanceEdges(rows.astype(np.int64), cols.astype(np.int64), D[rows, cols])]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_threshold_clusters_equal_dense_single_linkage_on_random_matrices(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 120))
    D = rng.uniform(0.0, 0.02, size=(n, n)) * (rng.uniform(size=(n, n)) < 0.7) + 0.02 * (rng.uniform(size=(n, n)) >= 0.7)
    D = np.triu(D, 1)
    D = D + D.T
    for threshold in (0.0, 0.001, 0.004, 0.005, 0.01, 0.05):
        expected = ln.single_linkage_clusters(D, threshold)
        got, _ = ln.threshold_clusters(n, edges_from_dense(D, threshold), threshold)
        assert np.array_equal(got, expected), (seed, threshold)
        # Extra edges above the threshold in the stream are ignored; compaction does not change anything.
        got_wide, _ = ln.threshold_clusters(n, edges_from_dense(D, 0.05), threshold, compact_every=1)
        assert np.array_equal(got_wide, expected), (seed, threshold)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_cluster_species_equals_old_dense_method_on_random_sketches(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 90
    S = lineage_sketches(rng, n)
    ids = [f"g{i:03d}" for i in range(n)]
    D = sk.pairwise_distances(S, k=sk.K)  # the old method: dense matrix + scipy single linkage
    off = D[np.triu_indices(n, 1)]
    for threshold in (0.0, float(np.quantile(off, 0.02)), float(np.quantile(off, 0.2)), 0.005, 0.05):
        expected = ln.single_linkage_clusters(D, threshold)
        numbers, close = ln.cluster_species("KPNEU", ids, S, threshold=threshold, outbreak_distance=0.0)
        assert np.array_equal(numbers, expected), threshold
        assert len(set(expected.tolist())) > 1 or threshold >= 0.05
        assert (close.distances <= 0.0).all()


def test_threshold_clusters_numbering_and_validation() -> None:
    blocks = [sk.DistanceEdges(np.array([3, 0]), np.array([4, 1]), np.array([0.001, 0.002]))]
    numbers, close = ln.threshold_clusters(6, blocks, 0.005, outbreak_distance=0.0015)
    # {0,1} and {3,4} tie on size -> the one with the smaller first member is 1; singletons 2 and 5 follow.
    assert numbers.tolist() == [1, 1, 3, 2, 2, 4]
    assert close.rows.tolist() == [3] and close.cols.tolist() == [4]
    assert ln.threshold_clusters(0, [], 0.005)[0].tolist() == []
    assert ln.threshold_clusters(3, [], 0.005)[0].tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        ln.threshold_clusters(2, blocks, 0.005)  # index 4 out of range
    with pytest.raises(ValueError):
        ln.threshold_clusters(6, blocks, -1.0)


def test_check_outbreak_edges_uses_sparse_pairs() -> None:
    edges = sk.DistanceEdges(np.array([0, 1]), np.array([1, 2]), np.array([0.0, 0.5]))
    assert ln.check_outbreak_edges(edges, np.array([1, 1, 2]), ["a", "b", "c"]) == 1
    with pytest.raises(ContractViolation, match="a vs b"):
        ln.check_outbreak_edges(edges, np.array([1, 2, 2]), ["a", "b", "c"])


def test_cluster_species_never_builds_a_dense_matrix(genomes: dict[str, tuple[str, str]], monkeypatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError("dense n x n path used")

    monkeypatch.setattr(sk, "pairwise_distances", boom)
    monkeypatch.setattr(ln, "single_linkage_clusters", boom)
    monkeypatch.setattr(ln, "check_outbreak_pairs", boom)
    ids = ["A1", "A2", "A1_dup", "B1", "B2"]
    S = sk.stack_sketches([sk.sketch(genomes[g][1], s=SKETCH_SIZE) for g in ids])
    numbers, _ = ln.cluster_species("KPNEU", ids, S)
    assert numbers.tolist() == [1, 1, 1, 2, 2]


def test_parallel_sketching_and_distances_match_serial(root: Paths, monkeypatch) -> None:
    from genome2mic import parallel

    gids = ["A1", "A2", "A3", "A4", "A1_dup", "A1_snp", "B1", "B2", "B3", "NOFASTA"]
    serial_log, parallel_log = ln.DropLog("lineages"), ln.DropLog("lineages")
    ids_1, S_1 = ln.sketch_genomes(root, gids, sketch_size=SKETCH_SIZE, droplog=serial_log, threads=1)
    monkeypatch.setattr(parallel, "MIN_PARALLEL_ITEMS", 1)
    monkeypatch.setattr(sk, "PARALLEL_MIN_WORK", 0.0)
    assert parallel.worker_count(2, len(gids)) == 2  # the pool path really runs below
    ids_2, S_2 =ln.sketch_genomes(root, gids, sketch_size=SKETCH_SIZE, droplog=parallel_log, threads=2)
    assert ids_1 == ids_2 and "NOFASTA" not in ids_1
    assert np.array_equal(S_1, S_2)
    assert [(r.reason, r.n_dropped) for r in serial_log.records] == [(r.reason, r.n_dropped) for r in parallel_log.records]

    n_1, close_1 = ln.cluster_species("KPNEU", ids_1, S_1, threads=1)
    edges_2 = list(sk.iter_distance_edges(S_1, max_distance=0.5, threads=2, rows_per_task=2))
    n_2, close_2 = ln.cluster_species("KPNEU", ids_1, S_1, threads=2)
    assert np.array_equal(n_1, n_2)
    assert np.array_equal(close_1.rows, close_2.rows) and np.array_equal(close_1.cols, close_2.cols)
    serial_edges = sk.distance_edges(S_1, max_distance=0.5, threads=1)
    merged = sk.DistanceEdges.concat(edges_2)
    assert np.array_equal(merged.rows, serial_edges.rows) and np.array_equal(merged.cols, serial_edges.cols)
    assert np.array_equal(merged.distances, serial_edges.distances)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def test_run_builds_contract_table(root: Paths, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="genome2mic")
    out = ln.run(root, FakeConfig(), sketch_size=SKETCH_SIZE)

    assert list(out.columns) == list(ln.COLUMNS)
    assert all(str(out[c].dtype) == "str" for c in ln.COLUMNS)
    expected = {"A1", "A2", "A3", "A4", "A1_dup", "A1_snp", "B1", "B2", "B3", "B4", "B5", "C1", "C2"}
    assert set(out["genome_id"]) == expected  # QCFAIL, NOFASTA, SAURX excluded
    assert out["genome_id"].is_unique

    by_id = out.set_index("genome_id")
    a_cluster = {by_id.loc[g, "lineage_cluster"] for g in ("A1", "A2", "A3", "A4", "A1_dup", "A1_snp")}
    b_cluster = {by_id.loc[g, "lineage_cluster"] for g in ("B1", "B2", "B3", "B4", "B5")}
    c_cluster = {by_id.loc[g, "lineage_cluster"] for g in ("C1", "C2")}
    assert a_cluster == {"KPNEU_ML_001"}  # largest KPNEU cluster
    assert b_cluster == {"KPNEU_ML_002"}
    assert c_cluster == {"ECOLI_ML_001"}  # numbering restarts per species
    assert all(re.fullmatch(r"(KPNEU|ECOLI)_ML_\d{3}", c) for c in out["lineage_cluster"])
    assert (out["cluster_method"] == ln.CLUSTER_METHOD_MASH).all()
    assert (by_id.loc[["B1", "B2", "B3", "B4", "B5"], "species"] == "KPNEU").all()
    assert (by_id.loc[["C1", "C2"], "species"] == "ECOLI").all()

    assert by_id.loc["A1", "st"] == "258"
    assert by_id.loc["A2", "st"] == "11"
    assert by_id.loc["A3", "st"] == "NA"
    assert by_id.loc["B1", "st"] == "NA"

    # Persisted outputs
    saved = pd.read_parquet(root.lineages)
    pd.testing.assert_frame_equal(saved, out)
    ids_k, S_k, k = sk.load_sketches(root.sketches("KPNEU"))
    assert ids_k == sorted(g for g in expected if not g.startswith("C"))
    assert S_k.shape == (11, SKETCH_SIZE) and k == sk.K
    ids_e, S_e, _ = sk.load_sketches(root.sketches("ECOLI"))
    assert ids_e == ["C1", "C2"] and S_e.shape == (2, SKETCH_SIZE)

    # Drop log: every filter recorded with its count
    drop = pd.read_csv(root.drop_log("lineages"))
    counts = drop.groupby("reason")["n_dropped"].sum()
    assert counts["qc_fail"] == 1
    assert counts["fasta_missing"] == 1
    assert counts["species_not_in_config"] == 1
    assert counts["labelled_genome_without_qc_record"] == 0
    assert "sketch_too_small" in counts.index

    # Summary and spot-check log lines
    text = caplog.text
    assert "KPNEU: 11 genome(s) -> 2 lineage cluster(s)" in text
    assert "largest cluster KPNEU_ML_001 has 6 genome(s)" in text
    assert "outbreak spot-check passed" in text


def test_outbreak_pairs_share_a_cluster_even_with_a_tiny_threshold(root: Paths) -> None:
    # Far below 0.005 the 10-SNP genomes separate, but the identical pair must stay together
    # and the spot-check must still pass because d = 0 <= 1e-4 is inside any threshold >= 0.
    out = ln.run(root, FakeConfig(), threshold=1e-6, sketch_size=SKETCH_SIZE)
    by_id = out.set_index("genome_id")["lineage_cluster"]
    assert by_id["A1"] == by_id["A1_dup"]
    assert by_id["A1"] != by_id["A2"]
    assert by_id["A1"].startswith("KPNEU_ML_")


def test_run_requires_qc_and_labels(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path, configs_dir=tmp_path / "configs")
    with pytest.raises(FileNotFoundError, match="qc"):
        ln.run(paths, FakeConfig())


def test_run_rejects_genome_with_two_species(root: Paths) -> None:
    labels = pd.read_parquet(root.labels)
    extra = labels.iloc[[0]].assign(species=pd.array(["ECOLI"], dtype="str"), drug=pd.array(["cefepime"], dtype="str"))
    write_parquet(pd.concat([labels, extra], ignore_index=True), root.labels)
    with pytest.raises(ContractViolation, match="more than one species"):
        ln.run(root, FakeConfig(), sketch_size=SKETCH_SIZE)
