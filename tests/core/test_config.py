"""Tests for genome2mic.config and the checked-in configs."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest
import yaml

from genome2mic.config import (
    BREAKPOINT_SITE,
    SPECTRUM_TIERS,
    Breakpoint,
    Config,
    ConfigError,
    load_config,
    normalize_name,
    normalize_standard,
    sir_from_mic,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"

SPECIES_KEYS = {"ECOLI", "KPNEU", "SAUR", "PAER", "ABAU"}
EXPECTED_SIZES = {
    "ECOLI": 5_000_000,
    "KPNEU": 5_500_000,
    "SAUR": 2_800_000,
    "PAER": 6_500_000,
    "ABAU": 4_000_000,
}
EXPECTED_REFERENCES = {
    "ECOLI": "NC_000913.3",
    "KPNEU": "NC_016845.1",
    "SAUR": "NC_007795.1",
    "PAER": "NC_002516.2",
    "ABAU": "CP000521.1",
}
CARBAPENEMASES = {
    "gene_blakpc",
    "gene_blandm",
    "gene_blaoxa_48",
    "gene_blaoxa_181",
    "gene_blaoxa_232",
    "gene_blavim",
    "gene_blaimp",
}
FORBIDDEN_FEATURE_NAMES = {
    "lineage_cluster",
    "st",
    "country",
    "year",
    "source",
    "isolation_source",
    "biosample",
    "split",
    "fold",
}


@pytest.fixture(scope="module")
def config() -> Config:
    return load_config(CONFIGS_DIR)


@pytest.fixture
def configs_copy(tmp_path: Path) -> Path:
    """A writable copy of the real configs for negative tests."""
    target = tmp_path / "configs"
    shutil.copytree(CONFIGS_DIR, target)
    return target


def _bp(s: float, r: float) -> Breakpoint:
    return Breakpoint("KPNEU", "meropenem", s, r, "EUCAST", "2024")


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("meropenem", "meropenem"),
        ("Meropenem", "meropenem"),
        ("  MEROPENEM  ", "meropenem"),
        ("MEM", "meropenem"),
        ("mem", "meropenem"),
        ("meropenem trihydrate", "meropenem"),
        ("Ciprofloxacin", "ciprofloxacin"),
        ("CIP", "ciprofloxacin"),
        ("CRO", "ceftriaxone"),
        ("ceftriaxone sodium", "ceftriaxone"),
        ("Ceftriaxone   Sodium", "ceftriaxone"),
        ("TZP", "piperacillin-tazobactam"),
        ("piperacillin/tazobactam", "piperacillin-tazobactam"),
        ("Piperacillin / Tazobactam", "piperacillin-tazobactam"),
        ("piperacillin tazobactam", "piperacillin-tazobactam"),
        ("piperacillin_tazobactam", "piperacillin-tazobactam"),
        ("Piperacillin-Tazobactam", "piperacillin-tazobactam"),
        ("SXT", "trimethoprim-sulfamethoxazole"),
        ("trimethoprim/sulfamethoxazole", "trimethoprim-sulfamethoxazole"),
        ("Trimethoprim/Sulfamethoxazole", "trimethoprim-sulfamethoxazole"),
        ("sulfamethoxazole/trimethoprim", "trimethoprim-sulfamethoxazole"),
        ("cotrimoxazole", "trimethoprim-sulfamethoxazole"),
        ("Co-trimoxazole", "trimethoprim-sulfamethoxazole"),
        ("TMP-SMX", "trimethoprim-sulfamethoxazole"),
        ("amoxicillin/clavulanic acid", "amoxicillin-clavulanic-acid"),
        ("Amoxicillin-Clavulanic Acid", "amoxicillin-clavulanic-acid"),
        ("AMC", "amoxicillin-clavulanic-acid"),
        ("amoxicillin-clavulanate", "amoxicillin-clavulanic-acid"),
        ("ceftolozane/tazobactam", "ceftolozane-tazobactam"),
        ("Polymyxin B", "polymyxin-b"),
        ("cefuroxime axetil", "cefuroxime"),
        ("rifampin", "rifampicin"),
        ("Rifampin", "rifampicin"),
        ("ceftazidime/avibactam", "ceftazidime-avibactam"),
        ("ampicillin/sulbactam", "ampicillin-sulbactam"),
        ("imipenem/cilastatin", "imipenem"),
        ("gentamycin", "gentamicin"),
        ("polymyxin E", "colistin"),
    ],
)
def test_normalize_drug_synonyms_casing_and_slashes(config: Config, raw: str, expected: str) -> None:
    assert config.normalize_drug(raw) == expected


@pytest.mark.parametrize("raw", ["foobarmycin", "", "   ", "/", "---", None, 12.5, float("nan")])
def test_normalize_drug_unknown_returns_none(config: Config, raw: object) -> None:
    assert config.normalize_drug(raw) is None


def test_normalize_name_is_pure_canonicalization() -> None:
    assert normalize_name(" Amoxicillin / Clavulanic   Acid ") == "amoxicillin-clavulanic-acid"
    assert normalize_name("TMP_SMX") == "tmp-smx"
    assert normalize_name(None) is None
    assert normalize_name("") is None


def test_every_drug_name_is_its_own_normal_form(config: Config) -> None:
    for name in config.drugs:
        assert config.normalize_drug(name) == name
        assert normalize_name(name) == name
        assert " " not in name and "/" not in name and name == name.lower()


def test_synonyms_do_not_collide(config: Config) -> None:
    owners: dict[str, str] = {}
    for name, drug in config.drugs.items():
        for synonym in drug.synonyms:
            key = normalize_name(synonym)
            assert key is not None
            assert owners.setdefault(key, name) == name, f"{synonym!r} is ambiguous"
            assert key not in config.drugs or key == name


# --------------------------------------------------------------------------- #
# Standards and breakpoint lookup
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("EUCAST", "EUCAST"),
        ("eucast", "EUCAST"),
        (" clsi ", "CLSI"),
        ("NCCLS", "CLSI"),
        ("SFM", None),
        ("", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_normalize_standard(raw: object, expected: str | None) -> None:
    assert normalize_standard(raw) == expected


def test_breakpoint_exact_match(config: Config) -> None:
    bp = config.breakpoint("KPNEU", "meropenem", "EUCAST", 2024)
    assert bp == Breakpoint("KPNEU", "meropenem", 2.0, 8.0, "EUCAST", "2024")


def test_breakpoint_lowercase_standard_and_string_year(config: Config) -> None:
    bp = config.breakpoint("KPNEU", "meropenem", "clsi", "2024")  # type: ignore[arg-type]
    assert bp is not None
    assert (bp.standard, bp.version, bp.s_breakpoint, bp.r_breakpoint) == ("CLSI", "2024", 1.0, 2.0)


def test_breakpoint_never_falls_back_to_another_year(config: Config) -> None:
    """DATA_CONTRACT stage 2: the table must match the row's standard *and* standard_year."""
    assert config.breakpoint("KPNEU", "meropenem", "CLSI", 2024) is not None
    assert config.breakpoint("KPNEU", "meropenem", "CLSI", 2016) is None
    assert config.breakpoint("KPNEU", "meropenem", "EUCAST", 2019) is None


def test_breakpoint_with_missing_year_returns_none(config: Config) -> None:
    assert config.breakpoint("ECOLI", "ciprofloxacin", "EUCAST", None) is None
    assert config.breakpoint("ECOLI", "ciprofloxacin", "EUCAST", float("nan")) is None  # type: ignore[arg-type]
    assert config.breakpoint("ECOLI", "ciprofloxacin", "EUCAST", "") is None  # type: ignore[arg-type]


def test_has_breakpoint_table(config: Config) -> None:
    assert config.has_breakpoint_table("EUCAST", 2024)
    assert config.has_breakpoint_table("eucast", "2024")  # type: ignore[arg-type]
    assert config.has_breakpoint_table("NCCLS", 2024)  # alias of CLSI
    assert not config.has_breakpoint_table("EUCAST", 2016)
    assert not config.has_breakpoint_table("EUCAST", None)
    assert not config.has_breakpoint_table(None, 2024)
    assert not config.has_breakpoint_table("SFM", 2024)


def test_breakpoint_null_or_unknown_standard_returns_none(config: Config) -> None:
    assert config.breakpoint("KPNEU", "meropenem", None, 2024) is None
    assert config.breakpoint("KPNEU", "meropenem", "", 2024) is None
    assert config.breakpoint("KPNEU", "meropenem", "SFM", 2024) is None
    assert config.breakpoint("KPNEU", "meropenem", float("nan"), None) is None  # type: ignore[arg-type]


def test_breakpoint_missing_pair_returns_none(config: Config) -> None:
    # EUCAST publishes no systemic gentamicin breakpoint for P. aeruginosa.
    assert config.breakpoint("PAER", "gentamicin", "EUCAST", 2024) is None
    # Unknown species / drug never raise.
    assert config.breakpoint("NOPE", "meropenem", "EUCAST", 2024) is None
    assert config.breakpoint("KPNEU", "unobtainium", "EUCAST", 2024) is None


def test_call_breakpoint_uses_clsi_2024(config: Config) -> None:
    # User decision (US standard): calls, pred_sir and VME use CLSI M100 2024.
    assert config.call_standard == ("CLSI", "2024")
    bp = config.call_breakpoint("KPNEU", "meropenem")
    assert bp == Breakpoint("KPNEU", "meropenem", 1.0, 2.0, "CLSI", "2024")
    # CLSI 2023 aminoglycoside revision: no systemic gentamicin breakpoint for P. aeruginosa.
    assert config.call_breakpoint("PAER", "gentamicin") is None


def test_eucast_susceptible_increased_exposure_stored_as_published(config: Config) -> None:
    bp = config.breakpoint("PAER", "ciprofloxacin", "EUCAST", 2024)
    assert bp is not None
    assert bp.s_breakpoint == 0.001 and bp.r_breakpoint == 0.5
    assert config.sir_from_mic(0.25, bp) == "I"
    assert config.sir_from_mic(1.0, bp) == "R"


def test_clsi_r_breakpoint_is_previous_doubling_step(config: Config) -> None:
    # CLSI meropenem Enterobacterales: S <= 1, I 2, R >= 4  ->  stored r = 2.
    bp = config.breakpoint("KPNEU", "meropenem", "CLSI", 2024)
    assert bp is not None
    assert (bp.s_breakpoint, bp.r_breakpoint) == (1.0, 2.0)
    assert config.sir_from_mic(1, bp) == "S"
    assert config.sir_from_mic(2, bp) == "I"
    assert config.sir_from_mic(4, bp) == "R"


def test_standards_and_versions(config: Config) -> None:
    assert config.standards() == {"CLSI": ("2024",), "EUCAST": ("2024",)}
    assert config.latest_version("EUCAST") == "2024"
    assert config.latest_version("SFM") is None


# --------------------------------------------------------------------------- #
# sir_from_mic
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("mic", "expected"),
    [
        (0.5, "S"),
        (1.0, "S"),  # MIC == s -> S
        (1.5, "I"),
        (2.0, "I"),  # MIC == r (s < r) -> I
        (2.01, "R"),
        (4.0, "R"),  # MIC > r -> R
        (math.inf, "R"),
    ],
)
def test_sir_from_mic_edges(config: Config, mic: float, expected: str) -> None:
    bp = _bp(1.0, 2.0)
    assert sir_from_mic(mic, bp) == expected
    assert config.sir_from_mic(mic, bp) == expected


def test_sir_from_mic_without_intermediate_category() -> None:
    bp = _bp(2.0, 2.0)
    assert sir_from_mic(2.0, bp) == "S"
    assert sir_from_mic(4.0, bp) == "R"
    assert sir_from_mic(0.25, bp) == "S"


@pytest.mark.parametrize("mic", [0.0, -1.0, math.nan, None, "8"])
def test_sir_from_mic_rejects_non_positive(mic: object) -> None:
    if mic == "8":
        assert sir_from_mic(mic, _bp(1.0, 2.0)) == "R"  # numeric strings are tolerated
        return
    with pytest.raises(ValueError):
        sir_from_mic(mic, _bp(1.0, 2.0))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Natural resistance and keep-variant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("species", "drug", "expected"),
    [
        ("KPNEU", "ampicillin", True),
        ("KPNEU", "Ampicillin", True),  # un-normalized drug is accepted
        ("KPNEU", "AMP", True),
        ("KPNEU", "meropenem", False),
        ("ECOLI", "ampicillin", False),
        ("PAER", "ertapenem", True),
        ("PAER", "tigecycline", True),
        ("PAER", "meropenem", False),
        ("ABAU", "aztreonam", True),
        ("SAUR", "aztreonam", True),
        ("SAUR", "colistin", True),
        ("SAUR", "ceftazidime", True),
        ("SAUR", "vancomycin", False),
        ("NOPE", "ampicillin", False),
    ],
)
def test_is_naturally_resistant(config: Config, species: str, drug: str, expected: bool) -> None:
    assert config.is_naturally_resistant(species, drug) is expected


def test_ecoli_has_no_natural_resistance(config: Config) -> None:
    assert not any(species == "ECOLI" for species, _ in config.natural_resistance)


def test_natural_resistance_matches_task_list(config: Config) -> None:
    expected = {
        ("KPNEU", "ampicillin"),
        *{("PAER", d) for d in (
            "ampicillin", "amoxicillin-clavulanic-acid", "ampicillin-sulbactam", "cefazolin",
            "cefotaxime", "ceftriaxone", "ertapenem", "trimethoprim-sulfamethoxazole",
            "tetracycline", "doxycycline", "minocycline", "tigecycline",
        )},
        *{("ABAU", d) for d in (
            "ampicillin", "amoxicillin-clavulanic-acid", "cefazolin", "cefotaxime",
            "ceftriaxone", "aztreonam", "ertapenem",
        )},
        *{("SAUR", d) for d in ("aztreonam", "colistin", "ceftazidime")},
    }
    assert set(config.natural_resistance) == expected


def test_keep_variant_prefixes(config: Config) -> None:
    prefixes = config.keep_variant_prefixes()
    assert isinstance(prefixes, tuple)
    assert {"blaKPC", "blaNDM", "blaOXA-48", "blaOXA-181", "blaOXA-232", "blaVIM", "blaIMP"} <= set(prefixes)
    assert len(prefixes) == len(set(prefixes))


# --------------------------------------------------------------------------- #
# Consistency of the checked-in configs
# --------------------------------------------------------------------------- #


def test_species_config(config: Config) -> None:
    assert set(config.species) == SPECIES_KEYS
    for key, species in config.species.items():
        assert species.key == key
        assert species.expected_genome_size == EXPECTED_SIZES[key]
        assert species.reference_accession == EXPECTED_REFERENCES[key]
        assert species.size_tolerance == pytest.approx(0.20)
        assert species.reference_sketch is None
    assert config.species["KPNEU"].amrfinder_organism == "Klebsiella_pneumoniae"
    assert config.species["ECOLI"].amrfinder_organism == "Escherichia"
    assert config.qc_max_contigs == 500
    assert config.qc_max_mash_distance == pytest.approx(0.05)


def test_every_drug_has_a_valid_tier(config: Config) -> None:
    assert len(config.drugs) == 40
    for drug in config.drugs.values():
        assert drug.spectrum_tier in SPECTRUM_TIERS, drug.name
    assert config.drugs["ampicillin"].spectrum_tier == 1
    assert config.drugs["ceftriaxone"].spectrum_tier == 2
    assert config.drugs["piperacillin-tazobactam"].spectrum_tier == 3
    assert config.drugs["meropenem"].spectrum_tier == 4
    assert config.drugs["colistin"].spectrum_tier == 4


def test_strong_markers(config: Config) -> None:
    for drug in ("ertapenem", "imipenem", "meropenem"):
        assert set(config.drugs[drug].strong_markers) == CARBAPENEMASES
    for drug in ("ceftriaxone", "cefotaxime", "ceftazidime", "cefepime"):
        assert set(config.drugs[drug].strong_markers) == CARBAPENEMASES | {"gene_blactx_m"}
    for drug in ("oxacillin", "cefoxitin"):
        assert set(config.drugs[drug].strong_markers) == {"gene_meca", "gene_mecc"}
    assert set(config.drugs["vancomycin"].strong_markers) == {"gene_vana", "gene_vanb"}
    assert set(config.drugs["colistin"].strong_markers) == {"gene_mcr"}
    assert config.drugs["ampicillin"].strong_markers == ()
    for drug in config.drugs.values():
        for marker in drug.strong_markers:
            assert marker.startswith(("gene_", "point_"))
            assert marker not in FORBIDDEN_FEATURE_NAMES


def test_strong_subclasses_only_on_carbapenems(config: Config) -> None:
    carbapenems = ("doripenem", "ertapenem", "imipenem", "meropenem")
    for drug in carbapenems:
        assert config.drugs[drug].strong_subclasses == ("CARBAPENEM",)
        assert set(config.drugs[drug].strong_markers) == set(config.drugs["meropenem"].strong_markers)
    others = {name: d.strong_subclasses for name, d in config.drugs.items() if name not in carbapenems}
    assert all(subclasses == () for subclasses in others.values()), others


def test_every_kept_carbapenemase_variant_is_a_carbapenem_strong_marker(config: Config) -> None:
    """A keep_variant.csv carbapenemase prefix yields its own gene_ column; the override must cover it.

    Regression guard: blaOXA-181 / blaOXA-232 were kept as variants but missing from the
    carbapenem ``strong_markers``, so the override never fired for them.
    """
    import csv  # noqa: PLC0415

    from genome2mic.features.known_amr import column_name  # noqa: PLC0415

    with (CONFIGS_DIR / "keep_variant.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    carbapenemase_prefixes = [
        row["family_prefix"] for row in rows if "carbapenem" in row["note"].lower() or "metallo" in row["note"].lower()
    ]
    assert carbapenemase_prefixes
    for prefix in carbapenemase_prefixes:
        column = column_name("gene_", prefix)
        for drug in ("ertapenem", "imipenem", "meropenem"):
            markers = config.drugs[drug].strong_markers
            assert any(column == m or column.startswith(m + "_") for m in markers), (prefix, drug)


def test_every_breakpoint_row_is_consistent(config: Config) -> None:
    rows = config.breakpoint_rows()
    assert rows
    for bp in rows:
        assert bp.species in config.species, bp
        assert bp.drug in config.drugs, bp
        assert bp.s_breakpoint <= bp.r_breakpoint, bp
        assert bp.s_breakpoint > 0 and math.isfinite(bp.r_breakpoint), bp
        assert bp.standard in {"EUCAST", "CLSI"}
        assert bp.version == "2024"


def test_breakpoint_tables_cover_starting_pairs(config: Config) -> None:
    for standard in ("EUCAST", "CLSI"):
        for drug in ("ceftriaxone", "meropenem", "ciprofloxacin"):
            assert config.breakpoint("KPNEU", drug, standard, 2024) is not None, (standard, drug)


def test_breakpoint_csv_files_have_contract_columns_and_notes() -> None:
    import csv

    for path in sorted((CONFIGS_DIR / "breakpoints").glob("*.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == [
                "species", "drug", "s_breakpoint", "r_breakpoint", "version", "site", "note",
            ], path.name
            for row in reader:
                assert row["site"] == BREAKPOINT_SITE, path.name
                assert "verify" in row["note"], (path.name, row)


def test_yaml_configs_are_mappings_for_the_api_readiness_check() -> None:
    for name in ("species.yaml", "drugs.yaml"):
        with (CONFIGS_DIR / name).open() as handle:
            assert isinstance(yaml.safe_load(handle), dict), name


# --------------------------------------------------------------------------- #
# load_config validation
# --------------------------------------------------------------------------- #


def test_load_config_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="configs dir not found"):
        load_config(tmp_path / "missing")


def test_load_config_rejects_bad_tier(configs_copy: Path) -> None:
    path = configs_copy / "drugs.yaml"
    content = yaml.safe_load(path.read_text())
    content["drugs"]["meropenem"]["spectrum_tier"] = 7
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(ConfigError, match="spectrum_tier"):
        load_config(configs_copy)


def test_load_config_rejects_synonym_collision(configs_copy: Path) -> None:
    path = configs_copy / "drugs.yaml"
    content = yaml.safe_load(path.read_text())
    content["drugs"]["meropenem"]["synonyms"].append("CIP")
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(ConfigError, match="CIP"):
        load_config(configs_copy)


def test_load_config_rejects_bad_strong_marker(configs_copy: Path) -> None:
    path = configs_copy / "drugs.yaml"
    content = yaml.safe_load(path.read_text())
    content["drugs"]["meropenem"]["strong_markers"].append("lineage_cluster")
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(ConfigError, match="strong marker"):
        load_config(configs_copy)


def test_load_config_rejects_s_greater_than_r(configs_copy: Path) -> None:
    path = configs_copy / "breakpoints" / "eucast_2024.csv"
    with path.open("a", newline="") as handle:
        handle.write("KPNEU,cefazolin,8,2,2024,bloodstream,bad row for the test\n")
    with pytest.raises(ConfigError, match="s_breakpoint 8.0 > r_breakpoint 2.0"):
        load_config(configs_copy)


def test_load_config_rejects_unknown_drug_in_breakpoints(configs_copy: Path) -> None:
    path = configs_copy / "breakpoints" / "clsi_2024.csv"
    with path.open("a", newline="") as handle:
        handle.write("KPNEU,unobtainium,1,2,2024,bloodstream,bad row for the test\n")
    with pytest.raises(ConfigError, match="unknown drug 'unobtainium'"):
        load_config(configs_copy)


def test_load_config_rejects_version_mismatch(configs_copy: Path) -> None:
    path = configs_copy / "breakpoints" / "clsi_2024.csv"
    with path.open("a", newline="") as handle:
        handle.write("KPNEU,cefazolin,1,2,2019,bloodstream,bad row for the test\n")
    with pytest.raises(ConfigError, match="does not match filename"):
        load_config(configs_copy)


def test_load_config_rejects_unknown_species_in_natural_resistance(configs_copy: Path) -> None:
    path = configs_copy / "natural_resistance.csv"
    with path.open("a", newline="") as handle:
        handle.write("SPYO,ampicillin,bad row for the test\n")
    with pytest.raises(ConfigError, match="unknown species 'SPYO'"):
        load_config(configs_copy)


def test_load_config_requires_call_standard_table(configs_copy: Path) -> None:
    path = configs_copy / "drugs.yaml"
    content = yaml.safe_load(path.read_text())
    content["call_standard"]["version"] = "2019"
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(ConfigError, match="has no breakpoint table"):
        load_config(configs_copy)


def test_older_table_is_used_when_year_matches(configs_copy: Path) -> None:
    """A second table for an older year is matched exactly; years without a table match nothing."""
    old = configs_copy / "breakpoints" / "clsi_2019.csv"
    old.write_text(
        "species,drug,s_breakpoint,r_breakpoint,version,site,note\n"
        "KPNEU,meropenem,1,2,2019,bloodstream,test table; verify\n"
        "KPNEU,gentamicin,4,8,2019,bloodstream,test table; verify\n"
    )
    cfg = load_config(configs_copy)
    assert cfg.standards()["CLSI"] == ("2019", "2024")
    assert cfg.breakpoint("KPNEU", "gentamicin", "CLSI", 2019) == Breakpoint(
        "KPNEU", "gentamicin", 4.0, 8.0, "CLSI", "2019"
    )
    assert cfg.breakpoint("KPNEU", "gentamicin", "CLSI", 2024).version == "2024"  # type: ignore[union-attr]
    assert cfg.has_breakpoint_table("CLSI", 2019) and not cfg.has_breakpoint_table("CLSI", 2016)
    # No table for 2016 and no year at all: no breakpoint, never the latest table.
    assert cfg.breakpoint("KPNEU", "gentamicin", "CLSI", 2016) is None
    assert cfg.breakpoint("KPNEU", "gentamicin", "CLSI", None) is None
    # The 2019 test table has no ciprofloxacin row: exact-year match does not fall back.
    assert cfg.breakpoint("KPNEU", "ciprofloxacin", "CLSI", 2019) is None


# --------------------------------------------------------------- strong markers (v0.6)

@pytest.mark.parametrize("drug", ["amikacin", "gentamicin", "tobramycin"])
def test_16s_rrna_methylases_are_strong_markers_for_aminoglycosides(config: Config, drug: str) -> None:
    cfg = config.drugs[drug]
    for column in ("gene_arma", "gene_rmtb1", "gene_rmtb4", "gene_rmtc", "gene_rmtd1", "gene_rmte1", "gene_rmtf1",
                   "gene_rmtg", "gene_rmth", "gene_rmta", "gene_npma"):
        assert cfg.is_strong_column(column), (drug, column)
    for column in ("gene_aac_6_ib", "gene_aph_3_ia", "gene_ant_2_ia", "point_rrs_a1408g"):
        assert not cfg.is_strong_column(column), (drug, column)


def test_mec_van_and_mcr_strong_markers(config: Config) -> None:
    for drug in ("oxacillin", "cefoxitin"):
        assert config.drugs[drug].is_strong_column("gene_meca") and config.drugs[drug].is_strong_column("gene_mecc")
        assert not config.drugs[drug].is_strong_column("gene_meci") and not config.drugs[drug].is_strong_column("gene_mecr1")
    van = config.drugs["vancomycin"]
    assert van.is_strong_column("gene_vana") and van.is_strong_column("gene_vanb")
    assert not van.is_strong_column("gene_vanh_a") and not van.is_strong_column("gene_vanz_a")
    for drug in ("colistin", "polymyxin-b"):
        cfg = config.drugs[drug]
        for column in ("gene_mcr_1", "gene_mcr_1_1", "gene_mcr_3_4", "gene_mcr_5", "gene_mcr_8_1"):
            assert cfg.is_strong_column(column), column
        # mcr-9 / mcr-10 are listed as exceptions (often colistin-susceptible), never forcing the call.
        for column in ("gene_mcr_9", "gene_mcr_9_1", "gene_mcr_10_1"):
            assert not cfg.is_strong_column(column), column


def test_strong_marker_exception_must_narrow_a_prefix(configs_copy: Path) -> None:
    path = configs_copy / "drugs.yaml"
    text = path.read_text()
    text = text.replace("      - gene_mcr_9\n", "      - gene_blakpc_9\n", 1)
    path.write_text(text)
    with pytest.raises(ConfigError, match="does not narrow"):
        load_config(configs_copy)
