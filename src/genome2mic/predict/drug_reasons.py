"""Resistance markers shown next to each drug (the report's `reasons`). Display only."""

from pathlib import Path

import pandas as pd

from genome2mic.features.amrfinder_table import AmrFinderHit, AmrFinderTable


class DrugReasons:
    """configs/drug_reasons.csv: drug -> AMRFinderPlus class (and optional subclass words)."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.rules: dict[str, list[tuple[str, list[str]]]] = {}
        for row in table.fillna("").itertuples(index=False):
            words = [word for word in str(row.subclass_any).split(";") if word]
            self.rules.setdefault(row.drug, []).append((row.amr_class, words))

    @classmethod
    def from_configs(cls, configs_dir: Path) -> "DrugReasons":
        return cls(pd.read_csv(configs_dir / "drug_reasons.csv", comment="#"))

    @staticmethod
    def label(hit: AmrFinderHit) -> str:
        """`gyrA_S83I` -> `gyrA S83I` (as in the contract's report example); genes keep their symbol."""
        return hit.symbol.replace("_", " ", 1) if hit.is_point else hit.symbol

    def for_drug(self, drug: str, hits: list[AmrFinderHit]) -> list[str]:
        rules = self.rules.get(drug, [])
        reasons = []
        for hit in hits:
            if hit.type != "AMR" or AmrFinderTable.ncbi_tag(hit) == "MISTRANSLATION":
                continue
            classes = hit.drug_class.split("/")
            for amr_class, words in rules:
                if amr_class in classes and (not words or any(word in hit.subclass for word in words)):
                    reasons.append(self.label(hit))
                    break
        return sorted(set(reasons))
