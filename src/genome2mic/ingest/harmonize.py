"""Builds labels.parquet from raw BV-BRC and NCBI AST tables. See DATA_CONTRACT.md stage 2."""

import logging
import math

import numpy as np
import pandas as pd

from genome2mic.ingest.breakpoint_table import BreakpointTable
from genome2mic.ingest.constants import (
    DROPPED_COLUMNS,
    ISOLATION_SOURCE_KEYWORDS,
    KNOWN_STANDARDS,
    LAB_EVIDENCE,
    LABEL_COLUMNS,
    METHOD_KEYWORDS,
    MIC_UNITS,
    MISSING_TEXT,
    PAIR_MIN_DISTINCT_MIC,
    PAIR_MIN_NONSUSCEPTIBLE,
    PAIR_MIN_SUSCEPTIBLE,
    PHENOTYPE_TO_SIR,
)
from genome2mic.ingest.mic_interval import MicIntervalConverter

logger = logging.getLogger(__name__)


class LabelHarmonizer:
    """Filters, converts, de-duplicates, and resolves raw AST rows into one interval per genome x drug."""

    def __init__(self, species_config: dict, drug_config: dict, breakpoints: BreakpointTable) -> None:
        self.species_by_name = {spec["name"].lower(): key for key, spec in species_config["species"].items()}
        self.drug_by_alias = {}
        for drug, spec in drug_config["drugs"].items():
            self.drug_by_alias[drug] = drug
            for synonym in spec.get("synonyms") or []:
                self.drug_by_alias[self._alias_key(synonym)] = drug
        self.breakpoints = breakpoints
        self.dropped_frames: list[pd.DataFrame] = []

    def harmonize(
        self,
        bvbrc_ast: pd.DataFrame,
        ncbi_ast: pd.DataFrame,
        bvbrc_meta: pd.DataFrame,
        ncbi_meta: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (labels, dropped). Every dropped row carries a reason."""
        logger.info("Harmonize start: bvbrc_rows=%s ncbi_rows=%s", len(bvbrc_ast), len(ncbi_ast))
        self.dropped_frames = []
        bvbrc_meta = bvbrc_meta.copy()
        bvbrc_meta["genome_id"] = bvbrc_meta["genome_id"].astype(str)
        rows = pd.concat(
            [self._standardize_bvbrc(bvbrc_ast, bvbrc_meta), self._standardize_ncbi(ncbi_ast)], ignore_index=True
        )
        rows = self._apply_filters(rows)
        rows = self._build_intervals(rows)
        rows = self._assign_genome_ids(rows, bvbrc_meta)
        resolved, duplicate_drops = self.resolve_duplicates(rows)
        self.dropped_frames.append(duplicate_drops)
        logger.info("Dropped %s genome x drug pairs: duplicate conflicts", len(duplicate_drops))
        labels = self._attach_metadata(resolved, bvbrc_meta, ncbi_meta)
        dropped = pd.concat(self.dropped_frames, ignore_index=True).reindex(columns=list(DROPPED_COLUMNS))
        logger.info("Harmonize done: labels=%s dropped=%s", len(labels), len(dropped))
        return labels, dropped

    @staticmethod
    def resolve_duplicates(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Collapse repeated (genome_id, drug) rows by the contract's duplicate rule. Returns (resolved, dropped)."""
        key = ["genome_id", "drug"]
        group_size = rows.groupby(key)["genome_id"].transform("size")
        resolved_rows = [rows[group_size == 1]]
        dropped_rows = []
        for (genome_id, drug), group in rows[group_size > 1].groupby(key, sort=False):
            merged, reason = LabelHarmonizer._resolve_group(group)
            if merged is None:
                dropped_rows.append(
                    {
                        "source": group["source"].iloc[0],
                        "record_id": genome_id,
                        "species": group["species"].iloc[0] if "species" in group else None,
                        "drug": drug,
                        "antibiotic": drug,
                        "reason": reason,
                    }
                )
            else:
                resolved_rows.append(pd.DataFrame([merged]))
        resolved = pd.concat(resolved_rows, ignore_index=True)
        resolved["censor"] = np.select(
            [resolved["mic_lower"] == 0, np.isinf(resolved["mic_upper"])], ["left", "right"], default="interval"
        )
        dropped = pd.DataFrame(dropped_rows, columns=list(DROPPED_COLUMNS))
        return resolved, dropped

    @staticmethod
    def count_pairs(labels: pd.DataFrame) -> pd.DataFrame:
        """Species x drug counts and the inclusion rule. Exact means every merged reading was `=`."""
        is_exact = labels["raw_result"].str.split("|").map(lambda parts: all(part.startswith("=") for part in parts))
        frame = labels.assign(
            is_s=labels["sir"] == "S",
            is_i=labels["sir"] == "I",
            is_r=labels["sir"] == "R",
            is_exact=is_exact,
            is_censored=labels["censor"] != "interval",
            exact_upper=labels["mic_upper"].where(is_exact),
        )
        counts = (
            frame.groupby(["species", "drug"])
            .agg(
                n_rows=("genome_id", "size"),
                n_S=("is_s", "sum"),
                n_I=("is_i", "sum"),
                n_R=("is_r", "sum"),
                n_exact=("is_exact", "sum"),
                n_censored=("is_censored", "sum"),
                n_distinct_mic=("exact_upper", "nunique"),
            )
            .reset_index()
        )
        counts["n_nonsusceptible"] = counts["n_I"] + counts["n_R"]
        counts["kept"] = (
            (counts["n_nonsusceptible"] >= PAIR_MIN_NONSUSCEPTIBLE)
            & (counts["n_S"] >= PAIR_MIN_SUSCEPTIBLE)
            & (counts["n_distinct_mic"] >= PAIR_MIN_DISTINCT_MIC)
        )
        return counts

    def _standardize_bvbrc(self, ast: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        genome_ids = ast["genome_id"].astype(str)
        meta_by_id = meta.set_index("genome_id")
        return pd.DataFrame(
            {
                "source": "BVBRC",
                "record_id": genome_ids,
                "bvbrc_genome_id": genome_ids,
                "biosample": genome_ids.map(meta_by_id["biosample_accession"]),
                "organism": genome_ids.map(meta_by_id["species"]),
                "antibiotic": self._text(ast, "antibiotic"),
                "phenotype": self._text(ast, "resistant_phenotype"),
                "sign": self._text(ast, "measurement_sign"),
                "value": self._text(ast, "measurement_value"),
                "unit": self._text(ast, "measurement_unit"),
                "method_text": self._text(ast, "laboratory_typing_method"),
                "standard_text": self._text(ast, "testing_standard"),
                "standard_year": pd.to_numeric(ast.get("testing_standard_year"), errors="coerce"),
                "evidence": self._text(ast, "evidence"),
            }
        )

    def _standardize_ncbi(self, ast: pd.DataFrame) -> pd.DataFrame:
        biosamples = self._text(ast, "biosample")
        # NCBI antibiograms are submitted lab results; there is no prediction evidence class.
        return pd.DataFrame(
            {
                "source": "NCBI",
                "record_id": biosamples,
                "bvbrc_genome_id": "",
                "biosample": biosamples,
                "organism": self._text(ast, "organism"),
                "antibiotic": self._text(ast, "antibiotic"),
                "phenotype": self._text(ast, "resistance_phenotype"),
                "sign": self._text(ast, "measurement_sign"),
                "value": self._text(ast, "measurement"),
                "unit": self._text(ast, "measurement_units"),
                "method_text": self._text(ast, "laboratory_typing_method"),
                "standard_text": self._text(ast, "testing_standard"),
                "standard_year": np.nan,
                "evidence": LAB_EVIDENCE,
            }
        )

    def _apply_filters(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = self._drop(rows, rows["evidence"] != LAB_EVIDENCE, "not_lab_method")

        species_lookup = {name: self._species_key(name) for name in rows["organism"].fillna("").unique()}
        rows = rows.assign(species=rows["organism"].fillna("").map(species_lookup))
        rows = self._drop(rows, rows["species"].isna(), "species_out_of_scope")

        drug_lookup = {name: self.drug_by_alias.get(self._alias_key(name)) for name in rows["antibiotic"].unique()}
        rows = rows.assign(drug=rows["antibiotic"].map(drug_lookup))
        unknown_drugs = rows.loc[rows["drug"].isna(), "antibiotic"].value_counts()
        if not unknown_drugs.empty:
            logger.info("Drugs not in panel (top 15): %s", unknown_drugs.head(15).to_dict())
        rows = self._drop(rows, rows["drug"].isna(), "drug_not_in_panel")

        method_lookup = {text: self._method_for(text) for text in rows["method_text"].unique()}
        rows = rows.assign(method=rows["method_text"].map(method_lookup))
        unknown_methods = rows.loc[rows["method"].isna(), "method_text"].value_counts()
        if not unknown_methods.empty:
            logger.info("Unknown methods: %s", unknown_methods.head(15).to_dict())
        rows = self._drop(rows, rows["method"].isna(), "unknown_method")

        return rows.assign(
            sir=rows["phenotype"].str.strip().str.lower().map(PHENOTYPE_TO_SIR),
            standard=rows["standard_text"].str.strip().str.lower().map(KNOWN_STANDARDS),
        )

    def _build_intervals(self, rows: pd.DataFrame) -> pd.DataFrame:
        is_mic_path = rows["method"].isin(["dilution", "gradient"]) & (rows["value"].str.strip() != "")
        mic_rows = self._build_mic_intervals(rows[is_mic_path])
        sir_rows = self._build_sir_intervals(rows[~is_mic_path])
        return pd.concat([mic_rows, sir_rows], ignore_index=True)

    def _build_mic_intervals(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.assign(
            sign=rows["sign"].map(MicIntervalConverter.normalize_sign),
            parsed=rows["value"].map(MicIntervalConverter.parse_value),
        )
        rows = self._drop(rows, rows["sign"].isna() | rows["parsed"].isna(), "unparseable_value")
        is_mic_unit = rows["unit"].str.strip().str.lower().isin(MIC_UNITS)
        rows = self._drop(rows, ~is_mic_unit, "bad_unit")
        rows = rows.assign(snapped=rows["parsed"].map(MicIntervalConverter.snap_to_doubling_step))
        rows = self._drop(rows, rows["snapped"].isna(), "off_grid_value")
        intervals = [MicIntervalConverter.from_mic(sign, value) for sign, value in zip(rows["sign"], rows["snapped"])]
        lower, upper, censor = zip(*intervals) if intervals else ((), (), ())
        return rows.assign(
            mic_lower=list(lower),
            mic_upper=list(upper),
            censor=list(censor),
            raw_result=rows["sign"] + rows["value"].str.strip(),
        )

    def _build_sir_intervals(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = self._drop(rows, rows["sir"].isna(), "no_value_no_phenotype")
        rows = self._drop(rows, rows["standard"].isna(), "no_standard_for_sir_only")
        intervals = []
        for species, drug, standard, year, sir in zip(
            rows["species"], rows["drug"], rows["standard"], rows["standard_year"], rows["sir"]
        ):
            year_value = None if pd.isna(year) else int(year)
            breakpoints = self.breakpoints.lookup(species, drug, standard, year_value)
            if breakpoints is None:
                intervals.append((np.nan, np.nan, None))
                continue
            interval = MicIntervalConverter.from_sir(sir, breakpoints[0], breakpoints[1])
            intervals.append(interval if interval is not None else (np.nan, np.nan, None))
        lower, upper, censor = zip(*intervals) if intervals else ((), (), ())
        rows = rows.assign(mic_lower=list(lower), mic_upper=list(upper), censor=list(censor), raw_result=rows["sir"])
        return self._drop(rows, rows["censor"].isna(), "no_breakpoint")

    def _assign_genome_ids(self, rows: pd.DataFrame, bvbrc_meta: pd.DataFrame) -> pd.DataFrame:
        meta = bvbrc_meta[["genome_id", "biosample_accession", "contigs"]].copy()
        meta["biosample_accession"] = self._clean_text(meta["biosample_accession"])
        meta = meta.dropna(subset=["biosample_accession"])
        meta["contigs"] = pd.to_numeric(meta["contigs"], errors="coerce")
        # One genome per biosample: the BV-BRC assembly with the fewest contigs, then the lowest ID.
        canonical = (
            meta.sort_values(["contigs", "genome_id"], na_position="last")
            .groupby("biosample_accession")["genome_id"]
            .first()
        )
        biosamples = self._clean_text(rows["biosample"])
        canonical_ids = biosamples.map(canonical)
        own_ids = rows["bvbrc_genome_id"].where(rows["source"] == "BVBRC", "NCBI_" + biosamples.fillna(""))
        genome_ids = canonical_ids.fillna(own_ids)
        is_collapsed = (rows["source"] == "BVBRC") & canonical_ids.notna() & (canonical_ids != rows["bvbrc_genome_id"])
        is_matched = (rows["source"] == "NCBI") & canonical_ids.notna()
        logger.info(
            "Genome IDs: bvbrc_ids_collapsed=%s ncbi_biosamples_matched_to_bvbrc=%s",
            rows.loc[is_collapsed, "bvbrc_genome_id"].nunique(),
            biosamples[is_matched].nunique(),
        )
        rows = rows.assign(genome_id=genome_ids, biosample=biosamples)
        species_per_genome = rows.groupby("genome_id")["species"].transform("nunique")
        return self._drop(rows, species_per_genome > 1, "species_conflict")

    def _attach_metadata(
        self, labels: pd.DataFrame, bvbrc_meta: pd.DataFrame, ncbi_meta: pd.DataFrame
    ) -> pd.DataFrame:
        bvbrc_by_id = bvbrc_meta.set_index("genome_id")
        ncbi_by_biosample = ncbi_meta.drop_duplicates(subset="biosample").set_index("biosample")
        genome_ids = labels["genome_id"]
        biosamples = labels["biosample"].fillna(self._clean_text(genome_ids.map(bvbrc_by_id["biosample_accession"])))

        bvbrc_source = self._clean_text(genome_ids.map(bvbrc_by_id["isolation_source"]))
        ncbi_source = self._clean_text(biosamples.map(ncbi_by_biosample["isolation_source"]))
        isolation_source = bvbrc_source.fillna(ncbi_source).map(self._isolation_category, na_action="ignore")

        bvbrc_country = self._clean_text(genome_ids.map(bvbrc_by_id["isolation_country"]))
        ncbi_country = self._clean_text(biosamples.map(ncbi_by_biosample["geo_loc_name"]).str.split(":").str[0])
        country = bvbrc_country.fillna(ncbi_country)

        bvbrc_year = genome_ids.map(bvbrc_by_id["collection_year"]).map(self._year_from)
        ncbi_year = biosamples.map(ncbi_by_biosample["collection_date"]).map(self._year_from)
        year = bvbrc_year.fillna(ncbi_year)

        labels = labels.assign(
            biosample=biosamples,
            isolation_source=isolation_source,
            country=country,
            year=year.astype("Int64"),
            standard_year=labels["standard_year"].astype("Int64"),
            mic_lower=labels["mic_lower"].astype(float),
            mic_upper=labels["mic_upper"].astype(float),
        )
        labels = labels[list(LABEL_COLUMNS)].sort_values(["species", "drug", "genome_id"]).reset_index(drop=True)
        for column in ("biosample", "sir", "standard", "isolation_source", "country"):
            labels[column] = labels[column].astype(object).where(labels[column].notna(), None)
        return labels

    def _drop(self, rows: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
        dropped_count = int(mask.sum())
        logger.info("Dropped %s rows: %s", dropped_count, reason)
        if dropped_count:
            dropped = rows.loc[mask].reindex(columns=list(DROPPED_COLUMNS[:-1]))
            self.dropped_frames.append(dropped.assign(reason=reason))
        return rows[~mask]

    def _species_key(self, organism: str) -> str | None:
        organism = organism.strip().lower()
        for name, key in self.species_by_name.items():
            if organism == name or organism.startswith(name + " "):
                return key
        return None

    @staticmethod
    def _resolve_group(group: pd.DataFrame) -> tuple[dict | None, str | None]:
        sirs = set(group["sir"].dropna())
        if "S" in sirs and "R" in sirs:
            return None, "conflict_s_vs_r"
        highest_lower = group["mic_lower"].max()
        lowest_upper = group["mic_upper"].min()
        merged = group.sort_values(["mic_lower", "mic_upper"], ascending=False).iloc[0].to_dict()
        if highest_lower < lowest_upper:
            merged["mic_lower"] = highest_lower
            merged["mic_upper"] = lowest_upper
        elif math.log2(highest_lower / lowest_upper) + 1 > 1:
            return None, "conflict_gt_1_step"
        merged["sir"] = next((sir for sir in ("R", "I", "S") if sir in sirs), None)
        merged["raw_result"] = "|".join(dict.fromkeys(group["raw_result"]))
        return merged, None

    @staticmethod
    def _method_for(method_text: str) -> str | None:
        method_text = method_text.strip().lower()
        for keyword, method in METHOD_KEYWORDS:
            if keyword in method_text:
                return method
        return None

    @staticmethod
    def _alias_key(name: str) -> str:
        key = name.strip().lower().replace("/", "-").replace(" ", "-")
        while "--" in key:
            key = key.replace("--", "-")
        return key

    @staticmethod
    def _text(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame:
            return pd.Series("", index=frame.index)
        return frame[column].fillna("").astype(str)

    @staticmethod
    def _clean_text(values: pd.Series) -> pd.Series:
        """Strip text and turn placeholders like `missing` or `NA` into real nulls."""
        text = values.astype(object).where(values.notna(), "").astype(str).str.strip()
        return text.where(~text.str.lower().isin(MISSING_TEXT), None)

    @staticmethod
    def _isolation_category(text: str) -> str:
        text = text.lower()
        for keyword, category in ISOLATION_SOURCE_KEYWORDS:
            if keyword in text:
                return category
        return "other"

    @staticmethod
    def _year_from(value: object) -> float:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return np.nan
        digits = str(value).strip()[:4]
        if not digits.isdigit() or not 1900 <= int(digits) <= 2100:
            return np.nan
        return float(digits)
