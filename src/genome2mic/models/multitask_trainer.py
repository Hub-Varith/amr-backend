"""Fits MultitaskMicNet on one train/validation split with early stopping."""

import copy
import logging

import numpy as np
import torch

from genome2mic.data.constants import SPECIES_KEYS
from genome2mic.data.label_matrix import LabelMatrix
from genome2mic.models.censored_normal_loss import CensoredNormalLoss
from genome2mic.models.feature_bundle import FeatureBundle
from genome2mic.models.genome_batcher import GenomeBatcher
from genome2mic.models.multitask_mic_net import MultitaskMicNet
from genome2mic.models.train_config import TrainConfig

logger = logging.getLogger(__name__)


class MultitaskTrainer:
    """One fit = one model. Validation rows pick the stopping epoch; nothing else is tuned on them here."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.loss = CensoredNormalLoss(drug_balanced=config.drug_balanced_loss)

    def build_model(self, bundle: FeatureBundle, n_drugs: int) -> MultitaskMicNet:
        n_unitig_by_species = {}
        for species_position, species_key in enumerate(SPECIES_KEYS):
            width = bundle.unitig_width(species_key)
            if width:
                n_unitig_by_species[species_position] = width
        return MultitaskMicNet(
            n_species=len(SPECIES_KEYS),
            n_known=len(bundle.known_columns),
            n_unitig_by_species=n_unitig_by_species,
            n_drugs=n_drugs,
            species_dim=self.config.species_dim,
            known_dim=self.config.known_dim,
            unitig_dim=self.config.unitig_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )

    def fit(
        self,
        labels: LabelMatrix,
        bundle: FeatureBundle,
        train_positions: np.ndarray,
        val_positions: np.ndarray | None,
        fixed_epochs: int | None = None,
    ) -> tuple[MultitaskMicNet, int, list[dict]]:
        """Train. With val_positions, stop early on validation loss; otherwise run fixed_epochs."""
        if val_positions is None and fixed_epochs is None:
            raise ValueError("need either val_positions for early stopping or fixed_epochs")
        torch.manual_seed(self.config.seed)
        model = self.build_model(bundle, len(labels.drugs))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        batcher = GenomeBatcher(labels, bundle, self.config.batch_size, self.config.seed)
        n_epochs = fixed_epochs if fixed_epochs is not None else self.config.max_epochs
        logger.info(
            "Training started",
            extra={"n_train": len(train_positions), "n_val": 0 if val_positions is None else len(val_positions), "n_epochs": n_epochs},
        )

        best_loss = np.inf
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        history = []
        for epoch in range(1, n_epochs + 1):
            model.train()
            train_total = 0.0
            train_count = 0
            for chunk in batcher.batches(train_positions, shuffle=True):
                batch = batcher.tensors(chunk)
                if batch["mask"].sum() == 0:
                    continue
                optimizer.zero_grad()
                mu, log_sigma = model(batch["species"], batch["known"], batch["unitigs"])
                loss = self.loss(mu, log_sigma, batch["lower"], batch["upper"], batch["mask"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_total += float(loss.detach()) * len(chunk)
                train_count += len(chunk)
            record = {"epoch": epoch, "train_loss": train_total / max(train_count, 1)}

            if val_positions is not None:
                record["val_loss"] = self.evaluate_loss(model, batcher, val_positions)
                if not np.isfinite(record["val_loss"]):
                    raise FloatingPointError("Non-finite validation loss")
                if record["val_loss"] < best_loss - 1e-5:
                    best_loss = record["val_loss"]
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                elif epoch - best_epoch >= self.config.patience:
                    history.append(record)
                    logger.info("Early stop", extra={"epoch": epoch, "best_epoch": best_epoch, "best_val_loss": best_loss})
                    break
            else:
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            history.append(record)
            logger.info("Epoch %d/%d train_loss=%.5f val_loss=%s", epoch, n_epochs, record["train_loss"], record.get("val_loss", "final-fit"))

        model.load_state_dict(best_state)
        model.eval()
        logger.info("Training finished", extra={"best_epoch": best_epoch, "best_val_loss": best_loss})
        return model, best_epoch, history

    def evaluate_loss(self, model: MultitaskMicNet, batcher: GenomeBatcher, positions: np.ndarray) -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for chunk in batcher.batches(positions, shuffle=False):
                batch = batcher.tensors(chunk)
                if batch["mask"].sum() == 0:
                    continue
                mu, log_sigma = model(batch["species"], batch["known"], batch["unitigs"])
                loss = self.loss(mu, log_sigma, batch["lower"], batch["upper"], batch["mask"])
                total += float(loss) * len(chunk)
                count += len(chunk)
        return total / max(count, 1)

    @staticmethod
    def predict_mu(model: MultitaskMicNet, batcher: GenomeBatcher, positions: np.ndarray) -> np.ndarray:
        """Point predictions (log2 mg/L) for every drug, rows in the order of positions."""
        model.eval()
        output = np.zeros((len(positions), model.n_drugs), dtype=np.float32)
        order = {position: row for row, position in enumerate(positions)}
        with torch.no_grad():
            for chunk in batcher.batches(positions, shuffle=False):
                batch = batcher.tensors(chunk)
                mu, _ = model(batch["species"], batch["known"], batch["unitigs"])
                rows = [order[position] for position in chunk]
                output[rows] = mu.numpy()
        return output
