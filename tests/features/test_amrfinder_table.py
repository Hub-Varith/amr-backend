"""AMRFinderPlus output -> the same feature row the training build makes from NCBI's AMR_genotypes."""

from pathlib import Path

import pandas as pd
import pytest

from genome2mic.features.amrfinder_table import AmrFinderTable, KnownAmrRow
from genome2mic.features.ncbi_known_amr import NcbiKnownAmrBuilder

FIXTURE = Path(__file__).parents[1] / "fixtures" / "amrfinder" / "kpneu_example.tsv"
CLASSES = {
    "blaKPC": "BETA-LACTAM", "blaSHV": "BETA-LACTAM", "gyrA_S83I": "QUINOLONE",
    "aadA2": "AMINOGLYCOSIDE", "emrD": "EFFLUX", "ompK35_E42RfsTer47": "BETA-LACTAM",
}


def test_parse_reads_amrfinder_4_columns():
    hits = AmrFinderTable.parse(FIXTURE)
    assert [hit.symbol for hit in hits][:3] == ["blaKPC-2", "blaSHV-12", "gyrA_S83I"]
    assert hits[0].subclass == "CARBAPENEM"


def test_parse_accepts_the_older_gene_symbol_header(tmp_path):
    old = tmp_path / "old.tsv"
    old.write_text(FIXTURE.read_text().replace("Element symbol", "Gene symbol", 1))
    assert AmrFinderTable.parse(old)[0].symbol == "blaKPC-2"


def test_genotype_string_matches_ncbi_tags_and_keeps_amr_only():
    hits = AmrFinderTable.parse(FIXTURE)
    assert AmrFinderTable.genotype_string(hits) == (
        "blaKPC-2,blaSHV-12,gyrA_S83I=POINT,aadA2=PARTIAL_END_OF_CONTIG,tet(A)=MISTRANSLATION,"
        "emrD,ompK35_E42RfsTer47=POINT"
    )   # merA is STRESS: NCBI keeps it in stress_genotypes, not AMR_genotypes


def test_feature_row_goes_through_the_training_builder():
    builder = NcbiKnownAmrBuilder(["blaKPC", "blaOXA"], CLASSES)
    row = KnownAmrRow.from_hits(AmrFinderTable.parse(FIXTURE), "KPNEU", builder)
    assert row["gene_blakpc_2"] == 1
    assert row["gene_blashv"] == 1                    # collapsed exactly like training
    assert row["point_gyra_s83i"] == 1
    assert row["point_ompk35_e42rfster47"] == 1
    assert row["gene_aada2"] == 1                     # partial at contig end still counts
    assert "gene_tet_a" not in row                    # broken gene skipped, as in training
    assert "gene_mera" not in row
    assert row["n_class_beta_lactam"] == 3            # KPC, SHV, OmpK35
    assert row["n_class_quinolone"] == 1


def test_no_amr_hits_gives_an_empty_row():
    builder = NcbiKnownAmrBuilder([], CLASSES)
    assert KnownAmrRow.from_hits([], "KPNEU", builder) == {}


def test_same_row_as_building_from_the_ncbi_string_directly():
    builder = NcbiKnownAmrBuilder(["blaKPC", "blaOXA"], CLASSES)
    hits = AmrFinderTable.parse(FIXTURE)
    genomes = pd.DataFrame([("g1", "KPNEU", "S1")], columns=["genome_id", "species", "biosample"])
    metadata = pd.DataFrame([("S1", AmrFinderTable.genotype_string(hits))], columns=["biosample_acc", "AMR_genotypes"])
    training_features, _ = builder.build(genomes, metadata)
    expected = {k: int(v) for k, v in training_features.drop(columns=["genome_id", "species"]).iloc[0].items() if v}
    assert KnownAmrRow.from_hits(hits, "KPNEU", builder) == expected


def test_missing_symbol_column_is_an_error(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("Type\tSubtype\nAMR\tAMR\n")
    with pytest.raises(ValueError):
        AmrFinderTable.parse(bad)


def test_only_broken_genes_gives_an_empty_row():
    builder = NcbiKnownAmrBuilder([], CLASSES)
    broken = [hit for hit in AmrFinderTable.parse(FIXTURE) if hit.method == "INTERNAL_STOP"]
    assert KnownAmrRow.from_hits(broken, "KPNEU", builder) == {}
