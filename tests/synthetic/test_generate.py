"""Tests for the synthetic raw-data generator (tiny sizes, tmp_path only)."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genome2mic import sketch
from genome2mic.config import load_config
from genome2mic.io import read_fasta
from genome2mic.paths import Paths
from genome2mic.synthetic import generate, markers as mk

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
SEED = 11
N_KPNEU, N_ECOLI, GENOME_LENGTH = 40, 20, 8000


def _generate(root: Path, seed: int = SEED) -> tuple[Paths, dict]:
    paths = Paths(root=root, configs_dir=REPO_CONFIGS)
    summary = generate.run(paths, seed=seed, n_kpneu=N_KPNEU, n_ecoli=N_ECOLI, genome_length=GENOME_LENGTH)
    return paths, summary


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> tuple[Paths, dict]:
    return _generate(tmp_path_factory.mktemp("synth_a"))


@pytest.fixture(scope="module")
def synth_kp60(tmp_path_factory: pytest.TempPathFactory) -> tuple[Paths, dict]:
    """KPNEU-only set at the default 60 kb scale, for scale-dependent sketch checks."""
    paths = Paths(root=tmp_path_factory.mktemp("synth_kp60"), configs_dir=REPO_CONFIGS)
    summary = generate.run(paths, seed=5, n_kpneu=30, n_ecoli=0, genome_length=60_000)
    return paths, summary


def _sketch_of(paths: Paths, gid: str) -> np.ndarray:
    return sketch.sketch([s for _, s in read_fasta(paths.genome_fasta(gid))])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_amrfinder(path: Path) -> tuple[str, list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        header = reader.fieldnames or []
    symbol_col = "Element symbol" if "Element symbol" in header else "Gene symbol"
    return symbol_col, rows


def _metadata(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(paths.genome_metadata, dtype=str, keep_default_na=False)


# --------------------------------------------------------------------------- #
# Reproducibility and file layout
# --------------------------------------------------------------------------- #


def test_same_seed_is_bit_for_bit_reproducible(synth: tuple[Paths, dict], tmp_path: Path) -> None:
    paths_a, summary_a = synth
    paths_b, summary_b = _generate(tmp_path / "synth_b")
    for name in ("ast_bvbrc.csv", "ast_ncbi.csv", "genome_metadata.csv", "SYNTHETIC_DATA.md"):
        assert _sha256(paths_a.raw_dir / name) == _sha256(paths_b.raw_dir / name), name
    assert summary_a["genome_ids"] == summary_b["genome_ids"]
    gid = summary_a["genome_ids"][0]
    assert _sha256(paths_a.genome_fasta(gid)) == _sha256(paths_b.genome_fasta(gid))
    assert _sha256(paths_a.interim_dir(gid) / "amrfinder.tsv") == _sha256(paths_b.interim_dir(gid) / "amrfinder.tsv")
    assert _sha256(paths_a.models_dir / "markers.fasta") == _sha256(paths_b.models_dir / "markers.fasta")


def test_different_seed_changes_output(synth: tuple[Paths, dict], tmp_path: Path) -> None:
    paths_a, _ = synth
    paths_c, _ = _generate(tmp_path / "synth_c", seed=SEED + 1)
    assert _sha256(paths_a.raw_ast("bvbrc")) != _sha256(paths_c.raw_ast("bvbrc"))


def test_expected_files_exist(synth: tuple[Paths, dict]) -> None:
    paths, summary = synth
    assert summary["synthetic"] is True
    assert summary["elapsed_s"] < 30
    assert paths.raw_ast("bvbrc").is_file()
    assert paths.raw_ast("ncbi").is_file()
    assert paths.genome_metadata.is_file()
    assert (paths.raw_dir / "SYNTHETIC_DATA.md").is_file()
    assert "simulated" in (paths.raw_dir / "SYNTHETIC_DATA.md").read_text(encoding="utf-8")
    assert paths.drop_log("synth").is_file()
    assert (paths.models_dir / "markers.fasta").is_file()
    for species in generate.SPECIES_ORDER:
        records = read_fasta(paths.reference_fasta(species))
        assert len(records) == 1
        assert len(records[0][1]) == GENOME_LENGTH
        assert "synthetic" in records[0][0]
    meta = _metadata(paths)
    assert len(meta) == N_KPNEU + N_ECOLI == summary["n_genomes"]
    for gid in meta["genome_id"]:
        assert paths.genome_fasta(gid).is_file(), gid
        interim = paths.interim_dir(gid)
        for name in ("amrfinder.tsv", "mash.tsv", "mlst.tsv", "resfinder/pheno_table.txt"):
            assert (interim / name).is_file(), f"{gid}/{name}"


def test_metadata_has_contract_columns_and_no_lineage_leak(synth: tuple[Paths, dict]) -> None:
    paths, summary = synth
    meta = _metadata(paths)
    assert list(meta.columns) == list(generate.METADATA_COLUMNS)
    forbidden = {"lineage_cluster", "st", "split", "fold"}
    assert not forbidden & set(meta.columns)
    assert meta["genome_id"].is_unique
    assert meta["biosample"].is_unique
    assert set(meta["species"]) == {"KPNEU", "ECOLI"}
    assert set(meta["source"]) == {"BVBRC", "NCBI"}
    assert (meta["isolation_source"] == "blood").mean() > 0.5
    assert meta["year"].astype(int).between(2010, 2023).all()
    assert summary["n_per_source"]["BVBRC"] > summary["n_per_source"]["NCBI"]


def test_configs_copy_loads_with_synthetic_sizes(synth: tuple[Paths, dict]) -> None:
    paths, summary = synth
    config = load_config(paths.root / "configs")
    assert config.species["SAUR"].expected_genome_size == GENOME_LENGTH
    kp = config.species["KPNEU"].expected_genome_size
    assert kp == summary["expected_genome_size"]["KPNEU"]
    lengths = [sum(len(s) for _, s in read_fasta(paths.genome_fasta(g)))
               for g in summary["genome_ids"] if g.startswith("573.") or g in summary["genome_ids"][:5]]
    assert min(lengths) <= kp <= max(lengths)
    assert (paths.root / "configs" / "breakpoints" / "eucast_2024.csv").is_file()
    text = (paths.root / "configs" / "species.yaml").read_text(encoding="utf-8")
    assert text.startswith("# SYNTHETIC")


def test_refuses_to_overwrite_source_configs(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "configs").mkdir(parents=True)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        generate.run(Paths(root=root, configs_dir=root / "configs"), seed=1, n_kpneu=4, n_ecoli=4, genome_length=4000)


# --------------------------------------------------------------------------- #
# Interim tool outputs
# --------------------------------------------------------------------------- #


def test_amrfinder_header_variants_and_columns(synth: tuple[Paths, dict]) -> None:
    paths, summary = synth
    new = old = 0
    for gid in summary["genome_ids"]:
        with (paths.interim_dir(gid) / "amrfinder.tsv").open(encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split("\t")
        if header == list(generate.AMRFINDER_COLUMNS_NEW):
            new += 1
        elif header == list(generate.AMRFINDER_COLUMNS_OLD):
            old += 1
            assert gid in summary["old_header_genomes"]
        else:
            pytest.fail(f"unexpected amrfinder header for {gid}: {header}")
    assert new > 0 and old > 0
    assert old == len(summary["old_header_genomes"])
    assert "Element symbol" in generate.AMRFINDER_COLUMNS_NEW and "Gene symbol" in generate.AMRFINDER_COLUMNS_OLD
    assert "Sequence name" in generate.AMRFINDER_COLUMNS_OLD


def test_hidden_blocks_never_reported_but_cryptic_is_planted(synth: tuple[Paths, dict]) -> None:
    paths, summary = synth
    seqs = mk.marker_sequences(SEED)
    cryptic_seq = seqs[mk.CRYPTIC_NAME]
    planted = 0
    for gid in summary["genome_ids"]:
        interim = paths.interim_dir(gid)
        amr_text = (interim / "amrfinder.tsv").read_text(encoding="utf-8")
        pheno_text = (interim / "resfinder" / "pheno_table.txt").read_text(encoding="utf-8")
        assert "cryptic" not in amr_text.lower() and "cryptic" not in pheno_text.lower()
        _, rows = _read_amrfinder(interim / "amrfinder.tsv")
        symbols = {r.get("Element symbol") or r.get("Gene symbol") for r in rows}
        assert mk.OMPK35_NAME not in symbols and mk.OMPK35_LOSS not in symbols
        if any(cryptic_seq in seq for _, seq in read_fasta(paths.genome_fasta(gid))):
            planted += 1
    assert planted >= 1
    markers_fasta = (paths.models_dir / "markers.fasta").read_text(encoding="utf-8")
    assert "cryptic" not in markers_fasta and f">{mk.OMPK35_NAME} " not in markers_fasta


def test_every_listed_marker_is_literally_in_the_fasta_at_the_reported_coordinates(synth: tuple[Paths, dict]) -> None:
    paths, summary = synth
    seqs = mk.marker_sequences(SEED)
    by_name = mk.marker_by_name()
    listed: set[str] = set()
    for gid in summary["genome_ids"]:
        contigs = dict(((h.split()[0], s) for h, s in read_fasta(paths.genome_fasta(gid))))
        symbol_col, rows = _read_amrfinder(paths.interim_dir(gid) / "amrfinder.tsv")
        for row in rows:
            symbol = row[symbol_col]
            assert symbol in by_name and by_name[symbol].reportable, (gid, symbol)
            listed.add(symbol)
            contig = contigs[row["Contig id"]]
            start, stop = int(row["Start"]), int(row["Stop"])
            assert contig[start - 1 : stop] == seqs[symbol], (gid, symbol)
            assert row["Type"] == by_name[symbol].marker_type
            assert row["Subtype"] == by_name[symbol].subtype
            assert row["Class"] == by_name[symbol].amr_class
            assert row["Subclass"] == by_name[symbol].subclass
            assert row["Strand"] == "+"
    assert len(listed) >= 6
    kp_only = [g for g in summary["genome_ids"] if g.startswith("573.")]
    assert kp_only  # blaSHV-11 is chromosomal in every KPNEU genome
    for gid in kp_only[:3]:
        symbol_col, rows = _read_amrfinder(paths.interim_dir(gid) / "amrfinder.tsv")
        assert "blaSHV-11" in {r[symbol_col] for r in rows}


def test_mash_and_mlst_follow_the_tool_formats(synth: tuple[Paths, dict]) -> None:
    paths, summary = synth
    meta = _metadata(paths).set_index("genome_id")
    for gid in summary["genome_ids"]:
        lines = (paths.interim_dir(gid) / "mash.tsv").read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(generate.SPECIES_ORDER)
        fields = [line.split("\t") for line in lines]
        assert [f[0] for f in fields] == list(generate.SPECIES_ORDER)  # no header row
        distances = {f[0]: float(f[2]) for f in fields}
        assert all(len(f) == 5 for f in fields)
        assert all(f[4].endswith("/1000") for f in fields)
        nearest = min(distances, key=distances.get)
        qc_fail = summary["qc_fail"].get(gid)
        if qc_fail == "wrong_species":
            assert nearest == generate.OTHER_SPECIES[meta.loc[gid, "species"]]
        else:
            assert nearest == meta.loc[gid, "species"]
        if qc_fail == "too_distant":
            assert 0.05 < distances[nearest] < 0.3
        else:
            assert distances[nearest] <= 0.05
        assert all(d >= 0.3 for sp, d in distances.items() if sp != nearest)
        mlst = (paths.interim_dir(gid) / "mlst.tsv").read_text(encoding="utf-8").splitlines()
        assert len(mlst) == 1
        parts = mlst[0].split("\t")
        assert parts[1] in ("klebsiella", "ecoli") and len(parts) == 10
        assert parts[2] == "-" if qc_fail == "too_distant" else parts[2].isdigit()


def test_pheno_table_derives_from_acquired_genes_only(synth: tuple[Paths, dict]) -> None:
    paths, summary = synth
    for gid in summary["genome_ids"]:
        symbol_col, rows = _read_amrfinder(paths.interim_dir(gid) / "amrfinder.tsv")
        symbols = {r[symbol_col] for r in rows if r["Subtype"] == "AMR"}
        text = (paths.interim_dir(gid) / "resfinder" / "pheno_table.txt").read_text(encoding="utf-8")
        assert "# Antimicrobial\tClass\tWGS-predicted phenotype\tMatch\tGenetic background" in text
        assert "SYNTHETIC" in text
        table = {line.split("\t")[0]: line.split("\t") for line in text.splitlines() if line and not line.startswith("#")}
        assert set(table) == {name for name, _ in mk.RESFINDER_ANTIMICROBIALS}
        has_carbapenemase = bool(symbols & set(mk.CARBAPENEMASES))
        assert (table["meropenem"][2] == "Resistant") == has_carbapenemase
        assert table["ciprofloxacin"][2] == ("Resistant" if symbols & {"qnrB1", "aac(6')-Ib-cr"} else "No resistance")


# --------------------------------------------------------------------------- #
# Genome model
# --------------------------------------------------------------------------- #


def test_near_identical_pairs_exist_and_are_within_one_snp(synth: tuple[Paths, dict]) -> None:
    paths, summary = synth
    pairs = summary["near_identical_pairs"]
    meta = _metadata(paths).set_index("genome_id")
    per_species = {sp: sum(1 for a, _ in pairs if meta.loc[a, "species"] == sp) for sp in ("KPNEU", "ECOLI")}
    assert all(4 <= n <= 6 for n in per_species.values()), per_species
    for original, partner in pairs:
        assert meta.loc[original, "biosample"] != meta.loc[partner, "biosample"]
        assert meta.loc[original, "species"] == meta.loc[partner, "species"]
        recs_a, recs_b = read_fasta(paths.genome_fasta(original)), read_fasta(paths.genome_fasta(partner))
        seq_a, seq_b = "".join(s for _, s in recs_a), "".join(s for _, s in recs_b)
        assert len(seq_a) == len(seq_b)
        mismatches = int(np.count_nonzero(np.frombuffer(seq_a.encode(), np.uint8) != np.frombuffer(seq_b.encode(), np.uint8)))
        assert mismatches <= 1
        assert [len(s) for _, s in recs_a] == [len(s) for _, s in recs_b]  # shared contig boundaries
        assert sketch.mash_distance(_sketch_of(paths, original), _sketch_of(paths, partner)) < 5e-4
        assert original not in summary["qc_fail"] and partner not in summary["qc_fail"]
        # Same strain, same planted markers.
        _, rows_a = _read_amrfinder(paths.interim_dir(original) / "amrfinder.tsv")
        _, rows_b = _read_amrfinder(paths.interim_dir(partner) / "amrfinder.tsv")
        assert len(rows_a) == len(rows_b)


def test_near_identical_pairs_pass_the_outbreak_threshold_at_default_scale(synth_kp60: tuple[Paths, dict]) -> None:
    paths, summary = synth_kp60
    pairs = summary["near_identical_pairs"]
    assert 4 <= len(pairs) <= 6
    for original, partner in pairs:
        assert sketch.mash_distance(_sketch_of(paths, original), _sketch_of(paths, partner)) <= 1e-4


def test_contig_counts_and_qc_fail_plan(synth: tuple[Paths, dict]) -> None:
    paths, summary = synth
    config = load_config(paths.root / "configs")
    meta = _metadata(paths).set_index("genome_id")
    kinds = set(summary["qc_fail"].values())
    assert kinds == set(generate.QC_FAIL_KINDS)
    assert len(summary["qc_fail"]) == 8  # four kinds planted per species at tiny sizes
    for gid in summary["genome_ids"]:
        records = read_fasta(paths.genome_fasta(gid))
        lengths = [len(s) for _, s in records]
        total = sum(lengths)
        kind = summary["qc_fail"].get(gid)
        if kind == "too_many_contigs":
            assert len(records) > 500
        else:
            assert 3 <= len(records) <= 40
        assert min(lengths) >= min(200, total // (4 * len(records)))
        assert all("synthetic=true" in h for h, _ in records)
        expected = config.species[meta.loc[gid, "species"]].expected_genome_size
        if kind == "wrong_size":
            assert not (0.8 * expected <= total <= 1.2 * expected), (gid, total, expected)


def test_size_rule_separates_only_the_planted_failures_at_default_scale(synth_kp60: tuple[Paths, dict]) -> None:
    paths, summary = synth_kp60
    expected = load_config(paths.root / "configs").species["KPNEU"].expected_genome_size
    outside = set()
    for gid in summary["genome_ids"]:
        total = sum(len(s) for _, s in read_fasta(paths.genome_fasta(gid)))
        if not (0.8 * expected <= total <= 1.2 * expected):
            outside.add(gid)
    assert outside == {g for g, k in summary["qc_fail"].items() if k == "wrong_size"}
    assert summary["n_per_species"] == {"KPNEU": 30, "ECOLI": 0}


def test_lineage_structure_is_visible_to_sketches(synth_kp60: tuple[Paths, dict]) -> None:
    """Same-ST (same lineage) genomes sit below the 0.005 clustering threshold, other lineages above."""
    paths, summary = synth_kp60
    partners = {b for _, b in summary["near_identical_pairs"]}
    normal = [g for g in summary["genome_ids"] if g not in summary["qc_fail"] and g not in partners]
    st_of = {g: (paths.interim_dir(g) / "mlst.tsv").read_text().split("\t")[2] for g in normal}
    sketches = {g: _sketch_of(paths, g) for g in normal}
    same, different = [], []
    for i, a in enumerate(normal):
        for b in normal[i + 1 :]:
            d = sketch.mash_distance(sketches[a], sketches[b])
            (same if st_of[a] == st_of[b] else different).append(d)
    assert same and different
    assert max(same) < min(different)
    assert float(np.median(same)) < 0.005 < min(different)


# --------------------------------------------------------------------------- #
# Raw AST realism
# --------------------------------------------------------------------------- #


def test_raw_ast_files_contain_the_planted_mess(synth: tuple[Paths, dict]) -> None:
    paths, summary = synth
    bv = pd.read_csv(paths.raw_ast("bvbrc"), dtype=str, keep_default_na=False)
    nc = pd.read_csv(paths.raw_ast("ncbi"), dtype=str, keep_default_na=False)
    assert list(bv.columns) == list(generate.BVBRC_COLUMNS)
    assert list(nc.columns) == list(generate.NCBI_COLUMNS)
    assert len(bv) == summary["n_bvbrc_rows"] and len(nc) == summary["n_ncbi_rows"]
    planted = summary["planted"]

    assert (bv["evidence"] == "Computational Prediction").sum() == planted["evidence == Computational Prediction"] > 0
    assert (bv["evidence"] == "Laboratory Method").sum() > 0
    sir_only_blank = ((bv["measurement_value"] == "") & (bv["evidence"] == "Laboratory Method")
                      & (bv["testing_standard"] == ""))
    assert sir_only_blank.sum() > 0
    assert (bv["laboratory_typing_method"] == "Disk diffusion").sum() > 0
    assert set(bv["testing_standard"]) >= {"EUCAST", "CLSI", ""}
    assert "µg/mL" in set(bv["measurement_unit"]) | set(nc["measurement_units"])
    assert {"<=", ">", "=", ""} <= set(bv["measurement_sign"])
    assert {"<", ">="} & (set(bv["measurement_sign"]) | set(nc["measurement_sign"]))
    assert "==" in set(nc["measurement_sign"])
    assert bv["antibiotic"].str.endswith(" ").any() or nc["antibiotic"].str.endswith(" ").any()
    lowered = set(bv["antibiotic"].str.strip().str.lower()) | set(nc["antibiotic"].str.strip().str.lower())
    assert {"mem", "cro", "cip"} & lowered  # abbreviations
    assert {"meropenem", "ceftriaxone", "ciprofloxacin"} <= lowered

    overlap = set(bv["biosample_accession"]) & set(nc["biosample"])
    assert len(overlap) == planted["biosamples present in both sources"] > 0
    assert planted["conflicting cross-source duplicate (genome x drug pairs)"] >= 1
    meta = _metadata(paths)
    ncbi_only = set(meta.loc[meta["source"] == "NCBI", "biosample"])
    assert ncbi_only <= set(nc["biosample"])
    assert not ncbi_only & set(bv["biosample_accession"])
    assert set(bv["genome_id"]) <= set(meta["genome_id"])


def test_drop_log_and_label_counts(synth: tuple[Paths, dict]) -> None:
    paths, summary = synth
    log = pd.read_csv(paths.drop_log("synth"), dtype=str, keep_default_na=False)
    assert list(log.columns) == ["stage", "reason", "n_dropped", "detail"]
    assert (log["stage"] == "synth").all()
    reasons = set(log["reason"])
    assert any("Computational Prediction" in r for r in reasons)
    assert any("QC-fail genome (too_many_contigs)" in r for r in reasons)
    assert any("inclusion rule" in r for r in reasons)
    table = {(r["species"], r["drug"]): r for r in summary["label_counts"]}
    assert set(table) == {(sp, d) for sp in ("KPNEU", "ECOLI") for d in mk.DRUGS}
    assert table[("ECOLI", "gentamicin")]["n"] < table[("ECOLI", "meropenem")]["n"]
    assert table[("ECOLI", "gentamicin")]["passes_inclusion"] is False


# --------------------------------------------------------------------------- #
# markers.py
# --------------------------------------------------------------------------- #


def test_marker_sequences_are_deterministic_and_sized() -> None:
    a = mk.marker_sequences(3)
    b = mk.marker_sequences(3)
    c = mk.marker_sequences(4)
    assert a == b
    assert a != c
    assert set(a) == {m.name for m in mk.MARKERS}
    for name, seq in a.items():
        assert set(seq) <= set("ACGT")
        assert len(seq) % 3 == 0
        if name == mk.CRYPTIC_NAME:
            assert 860 <= len(seq) <= 930
        else:
            assert 598 <= len(seq) <= 1200
    assert not {m.name for m in mk.reportable_markers()} & {mk.CRYPTIC_NAME, mk.OMPK35_NAME}


def test_effect_sum_matches_the_specified_mic_model() -> None:
    assert mk.effect_sum([], "meropenem") == 0
    assert mk.effect_sum(["blaKPC-2"], "meropenem") == 8
    assert mk.effect_sum(["blaNDM-1"], "meropenem") == 9
    assert mk.effect_sum(["blaOXA-48"], "meropenem") == 6
    assert mk.effect_sum(["ompK36_D135DGD"], "meropenem") == 1
    assert mk.effect_sum(["ompK36_D135DGD", "blaKPC-2"], "meropenem") == 8 + 3
    assert mk.effect_sum(["ompK36_D135DGD", "blaCTX-M-15"], "meropenem") == 3
    assert mk.effect_sum(mk.present_set(["blaSHV-11"], "KPNEU"), "meropenem") == 1  # ompK35 lost
    assert mk.effect_sum(mk.present_set(["blaSHV-11", mk.OMPK35_NAME], "KPNEU"), "meropenem") == 0
    assert mk.effect_sum(mk.present_set([], "ECOLI"), "meropenem") == 0
    assert mk.effect_sum(["blaCTX-M-15", "blaSHV-11", "blaTEM-1"], "ceftriaxone") == 9
    assert mk.effect_sum(["blaKPC-2", "blaOXA-48"], "ceftriaxone") == 9
    assert mk.effect_sum(["gyrA_S83L", "gyrA_D87N", "parC_S80I", "qnrB1", "aac(6')-Ib-cr", mk.CRYPTIC_NAME],
                         "ciprofloxacin") == 13
    with pytest.raises(ValueError):
        mk.effect_sum([], "colistin")
    for drug in mk.DRUGS:
        lo, hi = mk.PANELS[drug]
        assert lo < hi
        assert math.isfinite(mk.DRUG_BASE_LOG2[drug])
