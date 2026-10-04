"""Hyperparameters for the shared model. Plain data, no logic."""

from dataclasses import asdict, dataclass, field


@dataclass
class TrainConfig:
    """Everything that changes a training run. Hashed into run_id."""

    drugs: list[str] | None = None
    known_min_count: int = 5
    unitig_min_frac: float = 0.01
    unitig_max_frac: float = 0.99
    unitig_max_columns: int = 20000
    use_unitigs: bool = True
    species_dim: int = 16
    known_dim: int = 256
    unitig_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.2
    drug_balanced_loss: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 15
    conformal_level: float = 0.9
    conformal_min_rows: int = 20
    seed: int = 7
    n_folds: int = 5
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
