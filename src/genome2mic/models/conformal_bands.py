"""Split-conformal bands on the log2 MIC scale, one half-width per species x drug."""

import logging

import numpy as np
import pandas as pd

from genome2mic.data.label_matrix import LabelMatrix

logger = logging.getLogger(__name__)


class ConformalBands:
    """Half-width q (in doubling steps) such that |mu - lab| <= q on about `level` of validation rows.

    Only rows with an exact lab MIC (finite lower and upper) count as residuals. Pairs with too
    few such rows fall back to the drug-wide q, then to the global q, and that fallback is logged.
    """

    def __init__(self, table: pd.DataFrame, drug_fallback: dict[str, float], global_q: float) -> None:
        self.table = table
        self.drug_fallback = drug_fallback
        self.global_q = global_q
        self.lookup = {(row.species, row.drug): row.q for row in table.itertuples()}

    @classmethod
    def fit(
        cls, labels: LabelMatrix, positions: np.ndarray, mu: np.ndarray, level: float, min_rows: int
    ) -> "ConformalBands":
        """mu is (len(positions), n_drugs) validation predictions never used to pick columns or weights."""
        subset = labels.subset(positions)
        exact = subset.mask & np.isfinite(subset.lower_log2) & np.isfinite(subset.upper_log2)
        residual = np.abs(mu - subset.upper_log2)

        all_residuals = residual[exact]
        global_q = cls.quantile(all_residuals, level) if len(all_residuals) else 1.0
        rows = []
        drug_fallback = {}
        for drug_position, drug in enumerate(subset.drugs):
            drug_residuals = residual[:, drug_position][exact[:, drug_position]]
            drug_fallback[drug] = cls.quantile(drug_residuals, level) if len(drug_residuals) >= min_rows else global_q
            for species_key in np.unique(subset.species):
                species_rows = subset.species == species_key
                pair_exact = exact[species_rows, drug_position]
                pair_residuals = residual[species_rows, drug_position][pair_exact]
                n_rows = len(pair_residuals)
                if n_rows >= min_rows:
                    q = cls.quantile(pair_residuals, level)
                    source = "pair"
                else:
                    q = drug_fallback[drug]
                    source = "drug" if len(drug_residuals) >= min_rows else "global"
                    logger.warning(
                        "Conformal fallback", extra={"species": species_key, "drug": drug, "n_exact": n_rows, "source": source}
                    )
                rows.append({"species": species_key, "drug": drug, "q": float(q), "n_exact": n_rows, "source": source})
        table = pd.DataFrame(rows)
        logger.info("Conformal fitted", extra={"level": level, "global_q": float(global_q), "n_pairs": len(table)})
        return cls(table, drug_fallback, float(global_q))

    @staticmethod
    def quantile(residuals: np.ndarray, level: float) -> float:
        """Finite-sample conformal quantile: the ceil((n+1)*level)/n empirical quantile."""
        n_rows = len(residuals)
        rank = min(int(np.ceil((n_rows + 1) * level)), n_rows)
        return float(np.sort(residuals)[rank - 1])

    def half_width(self, species_key: str, drug: str) -> float:
        if (species_key, drug) in self.lookup:
            return self.lookup[(species_key, drug)]
        return self.drug_fallback.get(drug, self.global_q)

    def to_records(self) -> list[dict]:
        return self.table.to_dict(orient="records")

    @classmethod
    def from_records(cls, records: list[dict], drug_fallback: dict[str, float], global_q: float) -> "ConformalBands":
        return cls(pd.DataFrame(records, columns=["species", "drug", "q", "n_exact", "source"]), drug_fallback, global_q)
