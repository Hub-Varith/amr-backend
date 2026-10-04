"""Which genomes to download, from where, and the pilot sample."""

import pandas as pd

from genome2mic.genomes.genome_manifest import GenomeManifestBuilder


def assemblies() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("GCA_000000001.1", "SAMN1", 10),
            ("GCF_000000001.1", "SAMN1", 50),
            ("GCA_000000002.1", "SAMN3", 100),
            ("GCA_000000003.1", "SAMN3", 20),
        ],
        columns=["accession", "biosample", "n_contigs"],
    )


def test_choose_assemblies_prefers_refseq_then_fewer_contigs() -> None:
    chosen = GenomeManifestBuilder.choose_assemblies(assemblies()).set_index("biosample")["accession"]
    assert chosen["SAMN1"] == "GCF_000000001.1"
    assert chosen["SAMN3"] == "GCA_000000003.1"


def test_build_keeps_kept_pairs_of_one_species_and_maps_sources() -> None:
    labels = pd.DataFrame(
        [
            ("573.1", "KPNEU", "meropenem"),
            ("573.1", "KPNEU", "ciprofloxacin"),
            ("NCBI_SAMN1", "KPNEU", "meropenem"),
            ("NCBI_SAMN2", "KPNEU", "meropenem"),
            ("573.9", "KPNEU", "gentamicin"),
            ("562.1", "ECOLI", "meropenem"),
        ],
        columns=["genome_id", "species", "drug"],
    )
    pairs_kept = pd.DataFrame(
        [("KPNEU", "meropenem"), ("KPNEU", "ciprofloxacin"), ("ECOLI", "meropenem")], columns=["species", "drug"]
    )
    manifest, dropped = GenomeManifestBuilder.build(labels, pairs_kept, assemblies(), "KPNEU")

    assert list(manifest.columns) == ["genome_id", "species", "source", "accession"]
    rows = manifest.set_index("genome_id")
    assert list(rows.index) == ["573.1", "NCBI_SAMN1"]
    assert (rows.loc["573.1", "source"], rows.loc["573.1", "accession"]) == ("bvbrc", "573.1")
    assert (rows.loc["NCBI_SAMN1", "source"], rows.loc["NCBI_SAMN1", "accession"]) == ("ncbi", "GCF_000000001.1")
    assert dropped.to_dict("records") == [{"genome_id": "NCBI_SAMN2", "species": "KPNEU", "reason": "no_assembly"}]


def test_pilot_sample_is_fixed_by_seed_and_capped_at_manifest_size() -> None:
    manifest = pd.DataFrame({"genome_id": [f"573.{index}" for index in range(10)]})
    first = GenomeManifestBuilder.pilot_sample(manifest, size=4, seed=0)
    second = GenomeManifestBuilder.pilot_sample(manifest, size=4, seed=0)
    assert len(first) == 4
    assert list(first["genome_id"]) == list(second["genome_id"])
    assert len(GenomeManifestBuilder.pilot_sample(manifest, size=50, seed=0)) == 10
