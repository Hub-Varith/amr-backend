"""Tests for ``genome2mic.predict.release_features`` (NCBI-release known-AMR rules at prediction).

All inputs are hand-written FAKE AMRFinderPlus rows / AMR_genotypes strings; no number here
is a real MIC. Two optional parity tests run only when their data is present locally:
develop's builder (``.context/develop``) and the 200-genome NCBI parity on ``runs/hackathon5``.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genome2mic.predict import amr_detect
from genome2mic.predict import release_features as rf
from genome2mic.predict.pipeline import BundleError, PredictionPipeline
from tests.predict.test_pipeline import (
    _AMRFINDER_HEADER,
    FAKE_MODEL_CLASS,
    SPECIES,
    FakeLinearModel,
    RecordingUnitigQuery,
    by_drug,
    configs_dir,  # noqa: F401  (fixture)
    dna,  # noqa: F401  (fixture)
    genomes,  # noqa: F401  (fixture)
    write_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVELOP_BUILDER = REPO_ROOT / ".context" / "develop" / "src" / "genome2mic" / "features" / "ncbi_known_amr.py"

CLASS_TABLE = {
    "blaKPC-2": "BETA-LACTAM",
    "blaSHV": "BETA-LACTAM",  # family node: blaSHV-28 resolves via the allele-suffix fallback
    "blaCTX-M-15": "BETA-LACTAM",
    "aac(6')-Ib-cr5": "AMINOGLYCOSIDE/QUINOLONE",
    "gyrA_S83L": "QUINOLONE",
}


def _spec() -> rf.ReleaseFeatureSpec:
    return rf.ReleaseFeatureSpec(organism="Klebsiella_pneumoniae", class_by_symbol=dict(CLASS_TABLE))


def _row(symbol: str, type_: str, subtype: str, cls: str, subclass: str, method: str) -> str:
    return "\t".join(
        ["NA", "contig_1", "1", "100", "+", symbol, symbol, "core", type_, subtype, cls, subclass, method,
         "100", "100", "100.00", "100.00", "100", "X", "X", "NA", "NA"]
    )


FAKE_ROWS = [
    ("blaKPC-2", "AMR", "AMR", "BETA-LACTAM", "CARBAPENEM", "ALLELEX"),
    ("blaSHV-28", "AMR", "AMR", "BETA-LACTAM", "CEPHALOSPORIN", "PARTIALX"),
    ("blaSHV-11", "AMR", "AMR", "BETA-LACTAM", "BETA-LACTAM", "INTERNAL_STOP"),
    ("blaCTX-M-15", "AMR", "AMR", "BETA-LACTAM", "CEPHALOSPORIN", "PARTIAL_CONTIG_ENDX"),
    ("aac(6')-Ib-cr5", "AMR", "AMR", "AMINOGLYCOSIDE/QUINOLONE", "AMIKACIN/QUINOLONE", "EXACTX"),
    ("gyrA_S83L", "AMR", "POINT", "QUINOLONE", "QUINOLONE", "POINTX"),
    ("cirA_A36PfsTer30", "AMR", "POINT_DISRUPT", "BETA-LACTAM", "CEFIDEROCOL", "POINT_DISRUPTX"),
    ("qacE", "STRESS", "BIOCIDE", "QUATERNARY AMMONIUM", "QUATERNARY AMMONIUM", "EXACTX"),
]


def _write_tsv(path: Path, rows=FAKE_ROWS) -> Path:
    path.write_text("\n".join([_AMRFINDER_HEADER, *(_row(*r) for r in rows)]) + "\n")
    return path


def test_parse_genotypes_skips_mistranslation_and_marks_points() -> None:
    hits = rf.parse_genotypes("blaKPC-2,blaSHV-11=MISTRANSLATION,gyrA_S83L=POINT,blaSHV-28=PARTIAL, ompK36=HMM")
    assert hits == [("gene", "blaKPC-2"), ("point", "gyrA_S83L"), ("gene", "blaSHV-28"), ("gene", "ompK36")]


def test_release_values_follow_the_release_rules() -> None:
    values = rf.release_values(rf.parse_genotypes("blaKPC-2,blaSHV-28=PARTIAL,blaSHV-12,blaCTX-M-15,aac(6')-Ib-cr5,"
                                                  "gyrA_S83L=POINT,mcr-1,blaOXA-1"), _spec())
    # keep-variant families keep the allele; other bla collapse; non-bla symbols never collapse
    assert values["gene_blakpc_2"] == 1 and values["gene_blaoxa_1"] == 1
    assert values["gene_blashv"] == 1 and values["gene_blactx_m"] == 1
    assert values["gene_mcr_1"] == 1 and values["gene_aac_6_ib_cr5"] == 1
    assert values["point_gyra_s83l"] == 1
    # class counts: every hit, suffix fallback (blaSHV-12/-28 -> blaSHV), A/B counts once in each
    assert values["n_class_beta_lactam"] == 4  # KPC-2, SHV-28, SHV-12, CTX-M-15 (OXA-1 has no class here)
    assert values["n_class_aminoglycoside"] == 1
    assert values["n_class_quinolone"] == 2


@pytest.mark.parametrize(
    ("method", "subtype", "tag"),
    [
        ("EXACTX", "AMR", ""), ("ALLELEX", "AMR", ""), ("BLASTX", "AMR", ""), ("PARTIALX", "AMR", "PARTIAL"),
        ("PARTIAL_CONTIG_ENDX", "AMR", "PARTIAL_END_OF_CONTIG"), ("INTERNAL_STOP", "AMR", "MISTRANSLATION"),
        ("HMM", "AMR", "HMM"), ("POINTX", "POINT", "POINT"), ("POINTN", "POINT", "POINT"),
        ("POINT_DISRUPTX", "POINT_DISRUPT", "POINT"),
    ],
)
def test_tag_for_maps_amrfinder_methods_to_ncbi_tags(method: str, subtype: str, tag: str) -> None:
    assert rf.tag_for(method, subtype) == tag


def test_amrfinder_tsv_gives_the_same_row_as_its_ncbi_genotype_string(tmp_path: Path) -> None:
    detections = rf.read_amrfinder_rows(_write_tsv(tmp_path / "a.tsv"))
    assert "qacE" not in set(detections["symbol"])  # STRESS rows are a separate NCBI column
    assert "cirA_A36PfsTer30" in set(detections["symbol"])  # POINT_DISRUPT kept (NCBI writes =POINT)
    items = rf.amrfinder_items(detections)
    string = rf.genotype_string(items)
    assert "blaSHV-11=MISTRANSLATION" in string and "cirA_A36PfsTer30=POINT" in string and "blaKPC-2," in string
    row = rf.release_known_amr_row(detections, _spec())
    assert row.values == rf.release_values(rf.parse_genotypes(string), _spec())
    assert row.values["point_cira_a36pfster30"] == 1
    # The AMRFinderPlus Class column is not used: cirA has no class in the DB tables.
    assert row.values["n_class_beta_lactam"] == 3  # KPC-2, SHV-28, CTX-M-15; SHV-11 skipped
    assert "gene_blashv" in row.symbols_by_column and row.symbols_by_column["gene_blashv"] == ("blaSHV-28",)
    kpc = next(m for m in row.markers if m.symbol == "blaKPC-2")
    assert kpc.subclass == "CARBAPENEM" and kpc.subtype == amr_detect.SUBTYPE_AMR and kpc.column == "gene_blakpc_2"
    assert all(m.symbol != "blaSHV-11" for m in row.markers)


def test_spec_round_trip_and_class_table_without_organism_file(tmp_path: Path) -> None:
    db = tmp_path / "db"
    db.mkdir()
    (db / "fam.tsv").write_text("#node_id\tparent_node_id\tclass\nblaKPC\tbla\tBETA-LACTAM\nfoo\tbar\t\n")
    (db / "AMRProt-mutation.tsv").write_text("#taxgroup\tstandard_mutation_symbol\tclass\nEscherichia\tgyrA_S83L\tQUINOLONE\n")
    table = rf.load_class_table(db, "Pseudomonas_aeruginosa")  # no AMR_DNA-Pseudomonas_aeruginosa.tsv
    assert table == {"blaKPC": "BETA-LACTAM", "gyrA_S83L": "QUINOLONE"}
    spec = rf.ReleaseFeatureSpec(organism="Pseudomonas_aeruginosa", class_by_symbol=table, amrfinder_db_version="X")
    spec.save(tmp_path / "spec.json")
    again = rf.ReleaseFeatureSpec.load(tmp_path / "spec.json")
    assert again == spec
    assert again.class_of("blaKPC-3") == "BETA-LACTAM"


def test_write_feature_specs_marks_the_manifest(tmp_path: Path) -> None:
    db = tmp_path / "db"
    db.mkdir()
    (db / "fam.tsv").write_text("#node_id\tparent_node_id\tclass\nblaKPC\tbla\tBETA-LACTAM\n")
    (db / "AMRProt-mutation.tsv").write_text("#taxgroup\tstandard_mutation_symbol\tclass\n")
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text(json.dumps({"model_version": "x", "run_id": "y", "species": {"KPNEU": []}}))
    written = rf.write_feature_specs(models, db, {"KPNEU": "Klebsiella_pneumoniae"}, expected_db_version="other")
    assert written == [models / "KPNEU" / rf.FEATURE_SPEC_FILE]
    assert json.loads((models / "manifest.json").read_text())["feature_naming"] == "ncbi_release"
    assert rf.ReleaseFeatureSpec.load(written[0]).notes  # DB version mismatch is recorded


# --------------------------------------------------------------------------- #
# Pipeline: a bundle trained on an imported release uses the release rules
# --------------------------------------------------------------------------- #


def _release_bundle(tmp_path: Path, dna: dict[str, str], *, with_spec: bool = True) -> Path:
    models = write_bundle(tmp_path / "models_release", dna, synthetic=False, with_markers=False)
    manifest = json.loads((models / "manifest.json").read_text())
    manifest["feature_naming"] = "ncbi_release"
    (models / "manifest.json").write_text(json.dumps(manifest))
    if with_spec:
        _spec().save(models / SPECIES / rf.FEATURE_SPEC_FILE)
    return models


def test_release_bundle_without_feature_spec_is_rejected(tmp_path: Path, dna, configs_dir) -> None:  # noqa: F811
    models = _release_bundle(tmp_path, dna, with_spec=False)
    pipe = PredictionPipeline(models, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel}, unitig_query=RecordingUnitigQuery())
    with pytest.raises(BundleError, match="feature_spec.json"):
        pipe.load()


def test_release_bundle_converts_amrfinder_output_with_release_rules(tmp_path: Path, dna, configs_dir, genomes) -> None:  # noqa: F811
    models = _release_bundle(tmp_path, dna)
    tsv = _write_tsv(tmp_path / "explicit.tsv")
    pipe = PredictionPipeline(
        models, configs_dir, model_classes={FAKE_MODEL_CLASS: FakeLinearModel},
        unitig_query=RecordingUnitigQuery(), amrfinder_tsv=tsv,
    )
    pipe.load()
    assert pipe.feature_naming == "ncbi_release"
    assert pipe.species_bundles[SPECIES].feature_spec is not None
    report = pipe.run(genomes["clean"], "FAKE-RELEASE")
    calls = by_drug(report)
    # meropenem fake model: -4 + 6 * gene_blakpc_2 -> 2 log2 -> 4 mg/L, and the KPC strong-marker override
    assert calls["meropenem"]["pred_mic"] == 4.0
    assert calls["meropenem"]["override"] == "strong_marker"
    # ceftriaxone fake model: -3 + 5 * gene_blactx_m (+ 5 * KPC) -> 7 -> 128 mg/L (CTX-M partial at contig end counts)
    assert calls["ceftriaxone"]["pred_mic"] == 128.0


# --------------------------------------------------------------------------- #
# Optional parity checks on local data
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not DEVELOP_BUILDER.is_file(), reason="develop checkout (.context/develop) not present")
def test_matches_develops_builder_on_random_genotype_strings() -> None:
    spec_module = importlib.util.spec_from_file_location("develop_ncbi_known_amr", DEVELOP_BUILDER)
    module = importlib.util.module_from_spec(spec_module)
    assert spec_module.loader is not None
    spec_module.loader.exec_module(module)
    symbols = ["blaKPC-2", "blaKPC-3", "blaSHV-11", "blaSHV-28", "blaCTX-M-15", "blaOXA-48", "blaOXA-1", "blaTEM-1",
               "aac(6')-Ib-cr5", "gyrA_S83L", "parC_S80I", "mcr-1.1", "qnrS1", "blaGES-5", "blaVIM-1", "tet(A)"]
    tags = ["", "=PARTIAL", "=MISTRANSLATION", "=HMM", "=PARTIAL_END_OF_CONTIG"]
    rng = np.random.default_rng(0)
    strings = []
    for _ in range(60):
        picks = rng.choice(len(symbols), size=int(rng.integers(1, 9)), replace=True)
        items = []
        for i in picks:
            tag = "=POINT" if "_" in symbols[i] else tags[int(rng.integers(len(tags)))]
            items.append(symbols[i] + tag)
        strings.append(",".join(items))
    genomes_df = pd.DataFrame({"genome_id": [f"g{i}" for i in range(len(strings))], "species": "KPNEU",
                               "biosample": [f"S{i}" for i in range(len(strings))]})
    metadata = pd.DataFrame({"biosample_acc": genomes_df["biosample"], "AMR_genotypes": strings})
    features, _ = module.NcbiKnownAmrBuilder(list(rf.RELEASE_KEEP_VARIANT_FAMILIES), dict(CLASS_TABLE)).build(genomes_df, metadata)
    columns = [c for c in features.columns if c not in ("genome_id", "species")]
    for gid, string in zip(genomes_df["genome_id"], strings):
        expected = {c: int(v) for c, v in features.loc[features["genome_id"] == gid, columns].iloc[0].items() if int(v)}
        assert rf.release_values(rf.parse_genotypes(string), _spec()) == expected, string


PARITY_ROOT = Path(os.environ.get("G2M_PARITY_ROOT", REPO_ROOT / "runs" / "hackathon5"))
PD_FILES = {
    "KPNEU": Path("/tmp/kp_amr.tsv"),
    "ECOLI": Path("/tmp/pd_Escherichia_coli_Shigella.tsv"),
    "SAUR": Path("/tmp/pd_Staphylococcus_aureus.tsv"),
    "PAER": Path("/tmp/pd_Pseudomonas_aeruginosa.tsv"),
    "ABAU": Path("/tmp/pd_Acinetobacter.tsv"),
}


@pytest.mark.skipif(
    not (PARITY_ROOT / "models" / "KPNEU" / rf.FEATURE_SPEC_FILE).is_file() or not all(p.is_file() for p in PD_FILES.values()),
    reason="needs runs/hackathon5 (release + feature specs) and the NCBI Pathogen Detection metadata in /tmp",
)
def test_reproduces_release_rows_from_ncbi_genotype_strings() -> None:
    """Parity (a): 200 release genomes (40 per species) reproduced exactly from their AMR_genotypes."""
    processed = PARITY_ROOT / "data" / "processed"
    known = pd.read_parquet(processed / "known_amr.parquet")
    biosample = pd.read_parquet(processed / "labels.parquet", columns=["genome_id", "biosample"]).drop_duplicates("genome_id")
    feature_columns = [c for c in known.columns if c not in ("genome_id", "species")]
    rng = np.random.default_rng(0)
    checked = 0
    for species, path in PD_FILES.items():
        spec = rf.ReleaseFeatureSpec.load(PARITY_ROOT / "models" / species / rf.FEATURE_SPEC_FILE)
        metadata = pd.read_csv(path, sep="\t", dtype=str, usecols=["biosample_acc", "AMR_genotypes"], keep_default_na=False)
        genotypes = metadata[metadata["AMR_genotypes"] != ""].drop_duplicates("biosample_acc").set_index("biosample_acc")["AMR_genotypes"]
        rows = known[known["species"] == species].merge(biosample, on="genome_id")
        rows = rows[rows["biosample"].isin(genotypes.index)]
        picked = rows.iloc[np.sort(rng.choice(len(rows), size=min(40, len(rows)), replace=False))]
        for record in picked.to_dict("records"):
            release = {c: int(record[c]) for c in feature_columns if int(record[c]) != 0}
            ours = rf.release_values(rf.parse_genotypes(genotypes[record["biosample"]]), spec)
            assert ours == release, record["genome_id"]
            checked += 1
    assert checked == 200
