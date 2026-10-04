import gzip

from genome2mic.predict.genome_qc import AssemblyStats, GenomeQc


def write_fasta(path, contigs):
    path.write_text("".join(f">c{i} some description\n{seq[:30]}\n{seq[30:]}\n" for i, seq in enumerate(contigs)))
    return path


def test_stats_from_plain_and_gzipped_fasta(tmp_path):
    contigs = ["ACGT" * 25, "GGCC" * 10, "AT" * 5]          # 100, 40, 10 bp
    plain = write_fasta(tmp_path / "g.fasta", contigs)
    stats = AssemblyStats.from_fasta(plain)
    assert (stats.n_contigs, stats.total_length) == (3, 150)
    assert stats.n50 == 100                                 # half of 150 is reached inside the 100 bp contig
    assert abs(stats.gc_percent - 100 * (50 + 40 + 0) / 150) < 1e-9
    zipped = tmp_path / "g.fasta.gz"
    zipped.write_bytes(gzip.compress(plain.read_bytes()))
    assert AssemblyStats.from_fasta(zipped) == stats


def test_n_bases_are_left_out_of_gc():
    stats = AssemblyStats.from_sequences(["GGNNAA"])
    assert stats.total_length == 6 and abs(stats.gc_percent - 50.0) < 1e-9


def test_qc_rules_list_every_failure():
    stats = AssemblyStats(n_contigs=600, total_length=3_000_000, n50=5_000, gc_percent=57.0)
    result = GenomeQc({"KPNEU": 5_600_000}).evaluate(stats, "KPNEU", 0.08)
    assert not result.qc_pass
    assert result.qc_fail_reason == "too_fragmented;wrong_genome_size;too_distant"


def test_qc_passes_a_normal_assembly_and_flags_unknown_species():
    stats = AssemblyStats(n_contigs=80, total_length=5_500_000, n50=200_000, gc_percent=57.2)
    assert GenomeQc({"KPNEU": 5_600_000}).evaluate(stats, "KPNEU", 0.01).qc_pass
    unknown = GenomeQc({"KPNEU": 5_600_000}).evaluate(stats, None, 0.3)
    assert unknown.qc_fail_reason == "species_not_covered"


def test_empty_file_fails(tmp_path):
    empty = tmp_path / "e.fasta"
    empty.write_text("")
    stats = AssemblyStats.from_fasta(empty)
    assert stats.n_contigs == 0
    assert GenomeQc({"KPNEU": 5_600_000}).evaluate(stats, None, None).qc_fail_reason == "empty_assembly"
