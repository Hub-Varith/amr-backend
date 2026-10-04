"""Runs the frozen folds, fits conformal bands out-of-fold, then trains the final model."""

import logging

import numpy as np
import pandas as pd

from genome2mic.data.constants import SPECIES_KEYS, SPLIT_COLUMNS
from genome2mic.data.known_amr_table import KnownAmrTable
from genome2mic.data.label_matrix import LabelMatrix
from genome2mic.data.mic_steps import MicSteps
from genome2mic.data.unitig_store import UnitigStore
from genome2mic.models.conformal_bands import ConformalBands
from genome2mic.models.feature_bundle import FeatureBundle
from genome2mic.models.genome_batcher import GenomeBatcher
from genome2mic.models.model_artifact import MODEL_NAME, ModelArtifact
from genome2mic.models.multitask_trainer import MultitaskTrainer
from genome2mic.models.train_config import TrainConfig

logger = logging.getLogger(__name__)


class CrossValidation:
    """Uses splits.parquet as-is. Test rows are never read here (contract rule 8)."""

    def __init__(
        self,
        labels: LabelMatrix,
        known: KnownAmrTable,
        unitigs: UnitigStore,
        splits: pd.DataFrame,
        config: TrainConfig,
        pairs_kept: pd.DataFrame | None = None,
    ) -> None:
        missing = [column for column in SPLIT_COLUMNS if column not in splits.columns]
        if missing:
            raise ValueError(f"splits is missing columns {missing}")
        self.labels = labels
        self.known = known
        self.unitigs = unitigs
        self.config = config
        self.pairs_kept = pairs_kept
        self.trainer = MultitaskTrainer(config)

        train_rows = splits[(splits["split"] == "train") & splits["genome_id"].isin(labels.genome_ids)]
        n_dropped = int((splits["split"] == "train").sum() - len(train_rows))
        if n_dropped:
            logger.warning("Train genomes without labels dropped", extra={"n_dropped": n_dropped})
        self.train_positions = labels.positions_of(train_rows["genome_id"].to_numpy())
        self.train_folds = train_rows["fold"].to_numpy().astype(int)
        logger.info(
            "CrossValidation ready",
            extra={"n_train": len(self.train_positions), "folds": sorted(set(self.train_folds.tolist()))},
        )

    def run(self) -> tuple[ModelArtifact, pd.DataFrame, pd.DataFrame]:
        """Returns (artifact, out-of-fold predictions, per-fold history)."""
        oof_mu = np.full((len(self.train_positions), len(self.labels.drugs)), np.nan, dtype=np.float32)
        best_epochs = []
        history_rows = []
        for fold in sorted(set(self.train_folds.tolist())):
            val_rows = self.train_folds == fold
            fit_positions = self.train_positions[~val_rows]
            val_positions = self.train_positions[val_rows]
            logger.info("Fold started", extra={"fold": fold, "n_fit": len(fit_positions), "n_val": len(val_positions)})
            bundle = FeatureBundle.fit(self.known, self.unitigs, self.labels, fit_positions, self.config)
            model, best_epoch, history = self.trainer.fit(self.labels, bundle, fit_positions, val_positions)
            batcher = GenomeBatcher(self.labels, bundle, self.config.batch_size, self.config.seed)
            oof_mu[val_rows] = self.trainer.predict_mu(model, batcher, val_positions)
            best_epochs.append(best_epoch)
            for record in history:
                history_rows.append({"fold": fold, **record})

        conformal = ConformalBands.fit(
            self.labels, self.train_positions, oof_mu, self.config.conformal_level, self.config.conformal_min_rows
        )
        final_epochs = max(int(np.median(best_epochs)), 1)
        logger.info("Final fit", extra={"epochs": final_epochs, "fold_best_epochs": best_epochs})
        bundle = FeatureBundle.fit(self.known, self.unitigs, self.labels, self.train_positions, self.config)
        model, _, _ = self.trainer.fit(self.labels, bundle, self.train_positions, None, fixed_epochs=final_epochs)

        artifact = ModelArtifact(
            model=model,
            config=self.config,
            drugs=self.labels.drugs,
            known_columns=bundle.known_column_names,
            unitig_columns={key: value.tolist() for key, value in bundle.unitig_columns.items()},
            unitig_pattern_ids={
                key: self.unitigs.pattern_ids[key][value].tolist() for key, value in bundle.unitig_columns.items()
            },
            drugs_by_species=self.drugs_by_species(),
            conformal=conformal,
            best_epoch=final_epochs,
        )
        predictions = self.prediction_table(oof_mu, conformal, artifact.run_id)
        return artifact, predictions, pd.DataFrame(history_rows)

    def drugs_by_species(self) -> dict[str, list[str]]:
        """Only report allowed pairs that actually have training labels."""
        counts = self.labels.subset(self.train_positions).label_counts()
        counts = counts[counts["n_labels"] > 0]
        if self.pairs_kept is not None:
            counts = counts.merge(self.pairs_kept[["species", "drug"]].drop_duplicates(),
                                  on=["species", "drug"], how="inner", validate="one_to_one")
        return {key: sorted(group["drug"]) for key, group in counts.groupby("species")}

    def prediction_table(self, mu: np.ndarray, conformal: ConformalBands, run_id: str) -> pd.DataFrame:
        """Long table in the shape of DATA_CONTRACT.md stage 10, without the breakpoint-dependent columns."""
        subset = self.labels.subset(self.train_positions)
        rows = []
        for row in range(len(subset.genome_ids)):
            species_key = str(subset.species[row])
            for drug_position, drug in enumerate(subset.drugs):
                if not subset.mask[row, drug_position]:
                    continue
                point = float(mu[row, drug_position])
                q = conformal.half_width(species_key, drug)
                rows.append(
                    {
                        "genome_id": subset.genome_ids[row],
                        "species": species_key,
                        "drug": drug,
                        "split": "oof",
                        "mu_log2": point,
                        "pred_mic": float(MicSteps.round_up_to_step(np.array([point]))[0]),
                        "band_low": float(MicSteps.round_up_to_step(np.array([point - q]))[0]),
                        "band_high": float(MicSteps.round_up_to_step(np.array([point + q]))[0]),
                        "lab_lower": float(2.0 ** subset.lower_log2[row, drug_position]),
                        "lab_upper": float(2.0 ** subset.upper_log2[row, drug_position]),
                        "model": MODEL_NAME,
                        "run_id": run_id,
                    }
                )
        return pd.DataFrame(rows)
