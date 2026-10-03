"""Tests for ``genome2mic.sketch`` (pure-numpy Mash-like bottom-s MinHash).

All sequences are random synthetic DNA generated inline from a fixed seed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from genome2mic import sketch as sk

# ---------------------------------------------------------------------------
# Helpers and pure-Python references
# ---------------------------------------------------------------------------
_RC_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")
_BASES = np.frombuffer(b"ACGT", dtype=np.uint8)
_MASK64 = (1 << 64) - 1


def revcomp(seq: str) -> str:
    return seq.translate(_RC_TABLE)[::-1]


def random_dna(rng: np.random.Generator, length: int) -> str:
    return _BASES[rng.integers(0, 4, size=length)].tobytes().decode("ascii")


def mutate(seq: str, positions: np.ndarray, rng: np.random.Generator) -> str:
    """Substitute a different base at each position (SNPs only)."""
    out = bytearray(seq, "ascii")
    for p in positions:
        choices = [b for b in b"ACGT" if b != out[p]]
        out[p] = choices[rng.integers(0, 3)]
    return out.decode("ascii")


def splitmix64_ref(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def kmer_hashes_ref(seq: str, k: int) -> list[int]:
    """Slow reference: canonical 2-bit encoding + splitmix64, skipping non-ACGT."""
    enc = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq = seq.upper()
    out: list[int] = []
    for i in range(len(seq) - k + 1):
        window = seq[i : i + k]
        if any(c not in enc for c in window):
            continue
        fwd = 0
        for c in window:
            fwd = (fwd << 2) | enc[c]
        rev = 0
        for c in revcomp(window):
            rev = (rev << 2) | enc[c]
        out.append(splitmix64_ref(min(fwd, rev)))
    return out


class DropLogDouble:
    """Stand-in for ``genome2mic.droplog.DropLog`` (signature from DESIGN.md)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None]] = []

    def drop(self, reason: str, n: int, detail: str | None = None) -> None:
        self.calls.append((reason, n, detail))


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(20261003)


@pytest.fixture(scope="module")
def genome_50kb(rng: np.random.Generator) -> str:
    return random_dna(rng, 50_000)


# ---------------------------------------------------------------------------
# splitmix64 and k-mer encoding
# ---------------------------------------------------------------------------
def test_splitmix64_known_vector() -> None:
    # Standard SplitMix64 first output for seed 0.
    assert int(sk.splitmix64(0)[0]) == 0xE220A8397B1DCDAF
    assert splitmix64_ref(0) == 0xE220A8397B1DCDAF


def test_splitmix64_matches_reference_and_wraps() -> None:
    values = [0, 1, 6, 2**32, 2**63, 2**64 - 1, 0x9E3779B97F4A7C15]
    got = sk.splitmix64(np.array(values, dtype=np.uint64))
    assert got.dtype == np.uint64
    assert [int(v) for v in got] == [splitmix64_ref(v) for v in values]


def test_kmer_encoding_is_2bit_canonical() -> None:
    # ACG -> 0b000110 = 6 ; its reverse complement CGT -> 0b011011 = 27 ; canonical = 6.
    (h,) = sk.kmer_hashes("ACG", k=3)
    assert int(h) == splitmix64_ref(6)
    (h_rc,) = sk.kmer_hashes("CGT", k=3)
    assert int(h_rc) == splitmix64_ref(6)


@pytest.mark.parametrize("k", [3, 21, 31, 32])
def test_kmer_hashes_matches_reference(rng: np.random.Generator, k: int) -> None:
    seq = random_dna(rng, 300)
    # inject an N, an IUPAC code and lowercase stretches
    seq = seq[:70] + "N" + seq[71:150] + "R" + seq[151:200].lower() + seq[200:]
    got = sk.kmer_hashes(seq, k=k)
    assert got.dtype == np.uint64
    assert got.tolist() == kmer_hashes_ref(seq, k)


def test_kmer_hashes_canonical_under_reverse_complement(rng: np.random.Generator) -> None:
    seq = random_dna(rng, 2_000)
    a = np.sort(sk.kmer_hashes(seq))
    b = np.sort(sk.kmer_hashes(revcomp(seq)))
    assert np.array_equal(a, b)


def test_kmer_hashes_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        sk.kmer_hashes("ACGTACGT", k=0)
    with pytest.raises(ValueError):
        sk.kmer_hashes("ACGTACGT", k=33)


# ---------------------------------------------------------------------------
# Non-ACGT handling
# ---------------------------------------------------------------------------
def test_windows_spanning_n_are_skipped(rng: np.random.Generator) -> None:
    left, right = random_dna(rng, 500), random_dna(rng, 700)
    k = 21
    got = sk.kmer_hashes(left + "N" + right, k=k)
    expected = np.concatenate((sk.kmer_hashes(left, k=k), sk.kmer_hashes(right, k=k)))
    assert got.size == (len(left) - k + 1) + (len(right) - k + 1)
    assert np.array_equal(np.sort(got), np.sort(expected))


def test_lowercase_and_uppercase_hash_identically(rng: np.random.Generator) -> None:
    seq = random_dna(rng, 1_000)
    assert np.array_equal(sk.kmer_hashes(seq), sk.kmer_hashes(seq.lower()))


def test_non_acgt_only_or_short_sequences_give_empty() -> None:
    assert sk.kmer_hashes("N" * 100).size == 0
    assert sk.kmer_hashes("ACGT" * 4).size == 0  # 16 bp < k = 21
    assert sk.kmer_hashes("").size == 0
    assert sk.kmer_hashes("ACGT-ACGT\nACGT", k=5).size == 0  # gap/newline bytes invalidate
    assert sk.sketch(["N" * 100, ""]).size == 0


def test_non_ascii_characters_are_invalid_and_keep_alignment() -> None:
    seq = "ACGTACGT" + "é" + "TTGGCCAA"
    assert sk.kmer_hashes(seq, k=4).size == 2 * (8 - 4 + 1)


def test_sketch_reports_skipped_windows_to_droplog(rng: np.random.Generator) -> None:
    k = 21
    contig = random_dna(rng, 300) + "NN" + random_dna(rng, 300)
    log = DropLogDouble()
    sk.sketch([contig, random_dna(rng, 200)], k=k, s=50, droplog=log)
    assert len(log.calls) == 1
    reason, n, detail = log.calls[0]
    assert reason == "non_acgt_window"
    assert n == k + 1  # windows touching either of the two Ns
    assert detail

    clean_log = DropLogDouble()
    sk.sketch([random_dna(rng, 300)], k=k, s=50, droplog=clean_log)
    assert clean_log.calls == []


# ---------------------------------------------------------------------------
# sketch()
# ---------------------------------------------------------------------------
def test_sketch_is_sorted_unique_and_bottom_s(genome_50kb: str) -> None:
    s = 1000
    got = sk.sketch([genome_50kb], s=s)
    assert got.dtype == np.uint64
    assert got.shape == (s,)
    assert np.all(got[1:] > got[:-1])
    expected = np.unique(sk.kmer_hashes(genome_50kb))[:s]
    assert np.array_equal(got, expected)


def test_sketch_accepts_single_string(genome_50kb: str) -> None:
    assert np.array_equal(sk.sketch(genome_50kb, s=200), sk.sketch([genome_50kb], s=200))


def test_sketch_short_genome_returns_all_distinct(rng: np.random.Generator) -> None:
    seq = random_dna(rng, 120)  # 100 windows < s
    got = sk.sketch([seq], s=1000)
    assert got.size == np.unique(sk.kmer_hashes(seq)).size
    assert got.size <= 100


def test_sketch_merges_contigs_and_is_order_invariant(rng: np.random.Generator) -> None:
    contigs = [random_dna(rng, 5_000), random_dna(rng, 7_000), random_dna(rng, 3_000)]
    s = 500
    expected = np.unique(np.concatenate([sk.kmer_hashes(c) for c in contigs]))[:s]
    assert np.array_equal(sk.sketch(contigs, s=s), expected)
    assert np.array_equal(sk.sketch(reversed(contigs), s=s), expected)


def test_sketch_rejects_bad_s() -> None:
    with pytest.raises(ValueError):
        sk.sketch(["ACGT" * 20], s=0)


# ---------------------------------------------------------------------------
# mash_distance()
# ---------------------------------------------------------------------------
def test_identical_sequences_have_zero_distance(genome_50kb: str) -> None:
    a = sk.sketch([genome_50kb])
    b = sk.sketch([genome_50kb])
    assert sk.mash_distance(a, b) == 0.0


def test_reverse_complement_has_zero_distance(genome_50kb: str) -> None:
    a = sk.sketch([genome_50kb])
    b = sk.sketch([revcomp(genome_50kb)])
    assert np.array_equal(a, b)
    assert sk.mash_distance(a, b) == 0.0


def test_contig_split_is_near_zero(genome_50kb: str) -> None:
    # Splitting into contigs only loses the k-1 windows across each break.
    a = sk.sketch([genome_50kb])
    b = sk.sketch([genome_50kb[:20_000], genome_50kb[20_000:35_000], genome_50kb[35_000:]])
    assert sk.mash_distance(a, b) < 1e-3


def test_snps_give_small_monotone_distance(rng: np.random.Generator, genome_50kb: str) -> None:
    positions = rng.choice(len(genome_50kb), size=200, replace=False)
    base = sk.sketch([genome_50kb])
    distances = []
    for n_snp in (2, 20, 200):
        mutant = mutate(genome_50kb, positions[:n_snp], rng)
        distances.append(sk.mash_distance(base, sk.sketch([mutant])))
    d2, d20, d200 = distances
    assert 0.0 <= d2 <= d20 <= d200 < 0.01
    assert d20 > 0.0


def test_unrelated_sequences_are_far(rng: np.random.Generator, genome_50kb: str) -> None:
    other = random_dna(rng, 50_000)
    d = sk.mash_distance(sk.sketch([genome_50kb]), sk.sketch([other]))
    assert 0.3 < d <= 1.0


def test_mash_distance_formula_and_edges() -> None:
    k = 21
    assert sk.distance_from_jaccard(0.0, k) == 1.0
    assert sk.distance_from_jaccard(1.0, k) == 0.0
    j = 0.5
    assert sk.distance_from_jaccard(j, k) == pytest.approx(-np.log(2 * j / (1 + j)) / k)

    # Hand-built sketches: s = 4, union bottom-4 = {1,2,3,4}, shared within it = {2,3}
    a = np.array([1, 2, 3, 7], dtype=np.uint64)
    b = np.array([2, 3, 4, 9], dtype=np.uint64)
    assert sk.jaccard_counts(a, b) == (2, 4)
    assert sk.mash_distance(a, b, k=k) == pytest.approx(sk.distance_from_jaccard(0.5, k))
    # disjoint -> 1.0 ; identical -> 0.0 ; symmetric
    c = np.array([10, 11, 12, 13], dtype=np.uint64)
    assert sk.mash_distance(a, c, k=k) == 1.0
    assert sk.mash_distance(a, a, k=k) == 0.0
    assert sk.mash_distance(a, b, k=k) == sk.mash_distance(b, a, k=k)


def test_mash_distance_handles_unequal_sketch_lengths(rng: np.random.Generator, genome_50kb: str) -> None:
    full = sk.sketch([genome_50kb], s=1000)
    short = sk.sketch([genome_50kb[:600]], s=1000)  # 580 windows < 1000
    assert short.size < 1000
    d = sk.mash_distance(full, short)
    assert 0.0 < d <= 1.0
    # Explicit s caps the denominator at the union size for tiny sketches.
    common, denom = sk.jaccard_counts(short[:10], short[:10], s=1000)
    assert (common, denom) == (10, 10)


def test_mash_distance_rejects_empty() -> None:
    with pytest.raises(ValueError):
        sk.mash_distance(np.empty(0, dtype=np.uint64), np.array([1], dtype=np.uint64))


# ---------------------------------------------------------------------------
# pairwise_distances() / distances_to()
# ---------------------------------------------------------------------------
def _lineage_sketches(rng: np.random.Generator, n: int, s: int = 300) -> np.ndarray:
    """Sketches of n genomes drawn from two backbones with private SNPs."""
    backbones = [random_dna(rng, 20_000), random_dna(rng, 20_000)]
    rows = []
    for i in range(n):
        bb = backbones[i % 2]
        pos = rng.choice(len(bb), size=int(rng.integers(0, 400)), replace=False)
        rows.append(sk.sketch([mutate(bb, pos, rng)], s=s))
    return sk.stack_sketches(rows)


def test_pairwise_matches_mash_distance(rng: np.random.Generator) -> None:
    S = _lineage_sketches(rng, n=7)
    D = sk.pairwise_distances(S)
    assert D.shape == (7, 7)
    assert np.all(np.diag(D) == 0.0)
    assert np.array_equal(D, D.T)
    for i in range(7):
        for j in range(7):
            assert D[i, j] == pytest.approx(sk.mash_distance(S[i], S[j]), abs=1e-12)
    # within-backbone pairs are close, across-backbone pairs are far
    assert D[0, 2] < 0.05 < D[0, 1]


def test_pairwise_block_size_does_not_change_result(rng: np.random.Generator) -> None:
    S = _lineage_sketches(rng, n=9)
    assert np.array_equal(sk.pairwise_distances(S, block_rows=2), sk.pairwise_distances(S))


def test_distances_to_matches_mash_distance(rng: np.random.Generator) -> None:
    S = _lineage_sketches(rng, n=6)
    query = S[3]
    got = sk.distances_to(query, S)
    expected = np.array([sk.mash_distance(query, row) for row in S])
    assert np.allclose(got, expected, atol=1e-12)
    assert got[3] == 0.0

    # a query shorter than s still works (denominator is the larger size)
    short = query[:100]
    got_short = sk.distances_to(short, S)
    expected_short = np.array([sk.mash_distance(short, row) for row in S])
    assert np.allclose(got_short, expected_short, atol=1e-12)


def test_pairwise_single_sketch_is_zero_matrix(rng: np.random.Generator) -> None:
    S = sk.sketch([random_dna(rng, 3_000)], s=100)[None, :]
    assert np.array_equal(sk.pairwise_distances(S), np.zeros((1, 1)))


def test_pairwise_rejects_malformed_input() -> None:
    with pytest.raises(ValueError):
        sk.pairwise_distances(np.array([3, 1, 2], dtype=np.uint64))  # 1-D
    with pytest.raises(ValueError):
        sk.pairwise_distances(np.array([[3, 1, 2], [1, 2, 3]], dtype=np.uint64))  # unsorted row
    with pytest.raises(ValueError):
        sk.pairwise_distances(np.array([[1, 1, 2]], dtype=np.uint64))  # duplicate


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------
def test_save_load_round_trip(tmp_path, rng: np.random.Generator) -> None:
    S = _lineage_sketches(rng, n=4, s=250)
    ids = ["573.2002", "573.2005", "NCBI_SAMN00000003", "573.2011"]
    path = tmp_path / "sketches_KPNEU.npz"
    sk.save_sketches(path, ids, S, k=21)
    got_ids, got_S, got_k = sk.load_sketches(path)
    assert got_ids == ids
    assert got_k == 21
    assert got_S.dtype == np.uint64
    assert np.array_equal(got_S, S)
    with np.load(path) as z:
        assert set(z.files) == {"ids", "sketches", "k", "s"}
        assert int(z["s"]) == 250


def test_save_accepts_list_of_sketches_and_no_suffix(tmp_path, rng: np.random.Generator) -> None:
    rows = [sk.sketch([random_dna(rng, 4_000)], s=100) for _ in range(3)]
    path = tmp_path / "sketches_without_suffix"
    sk.save_sketches(path, ["g1", "g2", "g3"], rows, k=19)
    assert path.exists()  # numpy must not have appended ".npz"
    ids, S, k = sk.load_sketches(path)
    assert (ids, k, S.shape) == (["g1", "g2", "g3"], 19, (3, 100))
    assert np.array_equal(S, sk.stack_sketches(rows))


def test_save_rejects_bad_ids(tmp_path, rng: np.random.Generator) -> None:
    S = _lineage_sketches(rng, n=3, s=50)
    with pytest.raises(ValueError):
        sk.save_sketches(tmp_path / "a.npz", ["g1", "g2"], S)
    with pytest.raises(ValueError):
        sk.save_sketches(tmp_path / "b.npz", ["g1", "g1", "g2"], S)


def test_stack_rejects_ragged_sketches(rng: np.random.Generator) -> None:
    a = sk.sketch([random_dna(rng, 5_000)], s=100)
    b = sk.sketch([random_dna(rng, 60)], s=100)  # fewer than 100 distinct k-mers
    with pytest.raises(ValueError):
        sk.stack_sketches([a, b])


def test_load_rejects_non_sketch_npz(tmp_path) -> None:
    path = tmp_path / "other.npz"
    with path.open("wb") as fh:
        np.savez(fh, x=np.arange(3))
    with pytest.raises(ValueError):
        sk.load_sketches(path)


# ---------------------------------------------------------------------------
# Performance sanity
# ---------------------------------------------------------------------------
def test_sketching_1mb_is_fast(rng: np.random.Generator) -> None:
    seq = random_dna(rng, 1_000_000)
    t0 = time.perf_counter()
    got = sk.sketch([seq])
    elapsed = time.perf_counter() - t0
    assert got.shape == (sk.SKETCH_SIZE,)
    assert elapsed < 5.0, f"sketching 1 Mb took {elapsed:.2f}s"
