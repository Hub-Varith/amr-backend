"""Known-AMR features from NCBI Pathogen Detection AMR_genotypes strings."""

import pandas as pd

from genome2mic.features.ncbi_known_amr import NcbiKnownAmrBuilder

CLASSES = {
    "blaKPC": "BETA-LACTAM",
    "blaSHV": "BETA-LACTAM",
    "gyrA_S83I": "QUINOLONE",
    "sul1": "SULFONAMIDE",
    "aac(6')-Ib-cr": "AMINOGLYCOSIDE/QUINOLONE",
}


def builder() -> NcbiKnownAmrBuilder:
    return NcbiKnownAmrBuilder(["blaKPC", "blaOXA"], CLASSES)


def test_column_names_follow_the_contract() -> None:
    assert NcbiKnownAmrBuilder.column_name("gene_", "blaCTX-M") == "gene_blactx_m"
    assert NcbiKnownAmrBuilder.column_name("gene_", "aac(6')-Ib") == "gene_aac_6_ib"
    assert NcbiKnownAmrBuilder.column_name("point_", "gyrA_S83I") == "point_gyra_s83i"


def test_families_collapse_except_kept_variants() -> None:
    assert builder().family_for("blaSHV-12") == "blaSHV"
    assert builder().family_for("blaCTX-M-15") == "blaCTX-M"
    assert builder().family_for("blaKPC-2") == "blaKPC-2"
    assert builder().family_for("blaOXA-48") == "blaOXA-48"
    assert builder().family_for("sul1") == "sul1"


def test_parse_skips_broken_genes_and_tags_points() -> None:
    hits = builder().parse_genotypes("blaKPC-2,blaOXA=MISTRANSLATION,aadA2=PARTIAL_END_OF_CONTIG,gyrA_S83I=POINT")
    assert hits == [("gene", "blaKPC-2"), ("gene", "aadA2"), ("point", "gyrA_S83I")]


def test_build_makes_one_wide_row_per_genome() -> None:
    genomes = pd.DataFrame(
        [("573.1", "KPNEU", "SAMN1"), ("573.2", "KPNEU", "SAMN2"), ("573.3", "KPNEU", "SAMN3")],
        columns=["genome_id", "species", "biosample"],
    )
    metadata = pd.DataFrame(
        [("SAMN1", "blaKPC-2,blaSHV-11,gyrA_S83I=POINT,sul1,aac(6')-Ib-cr"), ("SAMN2", "blaSHV-12")],
        columns=["biosample_acc", "AMR_genotypes"],
    )
    features, columns = builder().build(genomes, metadata)
    rows = features.set_index("genome_id")
    assert list(rows.index) == ["573.1", "573.2"]
    assert rows.loc["573.1", "gene_blakpc_2"] == 1
    assert rows.loc["573.2", "gene_blakpc_2"] == 0
    assert rows.loc["573.2", "gene_blashv"] == 1
    assert rows.loc["573.1", "point_gyra_s83i"] == 1
    assert rows.loc["573.1", "n_class_beta_lactam"] == 2
    assert rows.loc["573.1", "n_class_quinolone"] == 2
    assert rows.loc["573.1", "n_class_aminoglycoside"] == 1
    assert features.drop(columns=["genome_id", "species"]).dtypes.eq("int8").all()
    assert not features.isna().any().any()
    assert set(columns.columns) == {"column_name", "source_symbol", "class", "n_genomes_present"}
