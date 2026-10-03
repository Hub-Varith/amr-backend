"""Save and load one trained shared model with everything prediction needs."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from genome2mic.data.constants import SPECIES_KEYS
from genome2mic.models.conformal_bands import ConformalBands
from genome2mic.models.multitask_mic_net import MultitaskMicNet
from genome2mic.models.train_config import TrainConfig

logger = logging.getLogger(__name__)

MODEL_NAME = "multitask_aft"
MODEL_VERSION = "multitask_aft-0.1"


class ModelArtifact:
    """A directory with model.pt plus spec.json (drugs, columns, conformal table, run_id)."""

    def __init__(
        self,
        model: MultitaskMicNet,
        config: TrainConfig,
        drugs: list[str],
        known_columns: list[str],
        unitig_columns: dict[str, list[int]],
        unitig_pattern_ids: dict[str, list[str]],
        drugs_by_species: dict[str, list[str]],
        conformal: ConformalBands,
        best_epoch: int,
    ) -> None:
        self.model = model
        self.config = config
        self.drugs = drugs
        self.known_columns = known_columns
        self.unitig_columns = unitig_columns
        self.unitig_pattern_ids = unitig_pattern_ids
        self.drugs_by_species = drugs_by_species
        self.conformal = conformal
        self.best_epoch = best_epoch
        self.run_id = hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:12]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")
        spec = {
            "model_version": MODEL_VERSION,
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "drugs": self.drugs,
            "species": list(SPECIES_KEYS),
            "known_columns": self.known_columns,
            "unitig_columns": {key: [int(value) for value in values] for key, values in self.unitig_columns.items()},
            "unitig_pattern_ids": self.unitig_pattern_ids,
            "drugs_by_species": self.drugs_by_species,
            "conformal": {
                "table": self.conformal.to_records(),
                "drug_fallback": self.conformal.drug_fallback,
                "global_q": self.conformal.global_q,
            },
            "best_epoch": self.best_epoch,
        }
        (directory / "spec.json").write_text(json.dumps(spec, indent=2))
        logger.info("Artifact saved", extra={"directory": str(directory), "run_id": self.run_id})

    @classmethod
    def load(cls, directory: Path) -> "ModelArtifact":
        spec = json.loads((directory / "spec.json").read_text())
        config = TrainConfig(**spec["config"])
        n_unitig_by_species = {
            SPECIES_KEYS.index(key): len(columns) for key, columns in spec["unitig_columns"].items() if columns
        }
        model = MultitaskMicNet(
            n_species=len(SPECIES_KEYS),
            n_known=len(spec["known_columns"]),
            n_unitig_by_species=n_unitig_by_species,
            n_drugs=len(spec["drugs"]),
            species_dim=config.species_dim,
            known_dim=config.known_dim,
            unitig_dim=config.unitig_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu"))
        model.eval()
        conformal = ConformalBands.from_records(
            spec["conformal"]["table"], spec["conformal"]["drug_fallback"], spec["conformal"]["global_q"]
        )
        artifact = cls(
            model,
            config,
            spec["drugs"],
            spec["known_columns"],
            {key: np.array(value, dtype=np.int64) for key, value in spec["unitig_columns"].items()},
            spec["unitig_pattern_ids"],
            spec["drugs_by_species"],
            conformal,
            spec["best_epoch"],
        )
        if artifact.run_id != spec["run_id"]:
            logger.warning("run_id mismatch on load", extra={"saved": spec["run_id"], "recomputed": artifact.run_id})
        logger.info("Artifact loaded", extra={"directory": str(directory), "run_id": spec["run_id"]})
        return artifact
