"""Tests for ``genome2mic.features.subclass_map`` (gene_ column -> AMRFinderPlus Subclass)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from genome2mic.features import subclass_map as sm

FAM = (
    "#node_id\tparent_node_id\tgene_symbol\thmm_id\thmm_tc1\thmm_tc2\tblastrule_complete_ident\t"
    "blastrule_complete_wp_coverage\tblastrule_complete_br_coverage\tblastrule_partial_ident\t"
    "blastrule_partial_wp_coverage\tblastrule_partial_br_coverage\treportable\ttype\tsubtype\tclass\tsubclass\tfamily_name\n"
    "blaOXA\tbla-D\tblaOXA\t-\t0\t0\t0\t0\t0\t0\t0\t0\t2\tAMR\tAMR\tBETA-LACTAM\tBETA-LACTAM\tclass D\n"
    "blaKPC\tbla-A_carba\tblaKPC\t-\t0\t0\t0\t0\t0\t0\t0\t0\t2\tAMR\tAMR\tBETA-LACTAM\tCARBAPENEM\tKPC\n"
    "blaIMI\tbla-A_carba\tblaIMI\t-\t0\t0\t0\t0\t0\t0\t0\t0\t2\tAMR\tAMR\tBETA-LACTAM\tCARBAPENEM\tIMI\n"
    "armA\trmt\tarmA\t-\t0\t0\t0\t0\t0\t0\t0\t0\t2\tAMR\tAMR\tAMINOGLYCOSIDE\tGENTAMICIN\tArmA\n"
)
PROT = (
    ">WP_1|1|1|blaOXA-23|blaOXA-23_fam|hydrolase|2|CARBAPENEM|BETA-LACTAM|OXA-23\nMA\n"
    ">WP_2|1|1|blaOXA-1|blaOXA-1_fam|hydrolase|2|CEPHALOSPORIN|BETA-LACTAM|OXA-1\nMA\n"
    ">WP_3|1|1|blaKPC-2|blaKPC|hydrolase|2|CARBAPENEM|BETA-LACTAM|KPC-2\nMA\n"
)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    (tmp_path / "fam.tsv").write_text(FAM)
    (tmp_path / "AMRProt.fa").write_text(PROT)
    (tmp_path / "version.txt").write_text("2026-08-07.1\n")
    return tmp_path


def test_allele_table_wins_over_the_family_table(db: Path) -> None:
    t = sm.load_subclass_tables(db)
    assert t.db_version == "2026-08-07.1"
    assert t.subclass_of("blaOXA-23") == "CARBAPENEM"     # AMRProt.fa allele
    assert t.subclass_of("blaOXA-1") == "CEPHALOSPORIN"
    assert t.subclass_of("blaOXA") == "BETA-LACTAM"       # fam.tsv family node
    assert t.subclass_of("blaKPC-99") == "CARBAPENEM"     # fam.tsv without the allele suffix
    assert t.subclass_of("blaIMI") == "CARBAPENEM"
    assert t.subclass_of("nothere") is None


def test_column_subclasses_release_and_own_naming(db: Path) -> None:
    t = sm.load_subclass_tables(db)
    release = pd.DataFrame({
        "column_name": ["gene_blaoxa_23", "gene_blaoxa_1", "gene_blaoxa", "gene_mixed", "point_x", "gene_blaoxa_23"],
        "source_symbol": ["blaOXA-23", "blaOXA-1", "blaOXA", "blaOXA-23,blaOXA-1", "x_A1B", "blaOXA-23"],
        "class": ["BETA-LACTAM"] * 6,
        "species": ["ABAU", "KPNEU", "KPNEU", "KPNEU", "KPNEU", "KPNEU"],
    })
    out = sm.column_subclasses(release, t)
    assert out["gene_blaoxa_23"] == "CARBAPENEM"
    assert out["gene_blaoxa_1"] == "CEPHALOSPORIN"
    assert out["gene_blaoxa"] == "BETA-LACTAM"
    assert out["gene_mixed"] == "CARBAPENEM;CEPHALOSPORIN"
    assert "point_x" not in out
    own = pd.DataFrame({
        "column_name": ["gene_blakpc_2", "gene_unknown"],
        "source_symbol": ["blaKPC-2", "zzz"],
        "class": ["BETA-LACTAM", ""],
        "subclass": ["CARBAPENEM", ""],
        "member_symbols": ["blaKPC-2", "zzz"],
    })
    out = sm.column_subclasses(own, t)
    assert out["gene_blakpc_2"] == "CARBAPENEM" and out["gene_unknown"] == sm.UNKNOWN_SUBCLASS
    assert sm.column_subclasses(own, None)["gene_unknown"] is None


def test_missing_database_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sm.load_subclass_tables(tmp_path)
