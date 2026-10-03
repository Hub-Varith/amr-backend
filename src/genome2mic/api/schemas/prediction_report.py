"""Full report for one isolate."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, model_validator

from genome2mic.api.constants import DISCLAIMER
from genome2mic.api.schemas.call import Call
from genome2mic.api.schemas.drug_prediction import DrugPrediction
from genome2mic.api.schemas.species_key import SpeciesKey


class PredictionReport(BaseModel):
    """Report for one new isolate (DATA_CONTRACT.md stage 12). species is null when not covered."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    species: SpeciesKey | None
    qc_pass: bool
    nearest_training_distance: NonNegativeFloat | None
    in_range: bool
    predictions: list[DrugPrediction]
    ranked_active: list[str]
    model_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    disclaimer: str = Field(default=DISCLAIMER, min_length=1)

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        """Keep the disclaimer and the ranked list honest."""
        if not self.disclaimer.strip():
            raise ValueError("disclaimer must not be blank.")
        active_drugs = set()
        for prediction in self.predictions:
            if prediction.call is Call.LIKELY_ACTIVE:
                active_drugs.add(prediction.drug)
        for drug in self.ranked_active:
            if drug not in active_drugs:
                raise ValueError(f"ranked_active drug '{drug}' is not a likely_active prediction.")
        return self
