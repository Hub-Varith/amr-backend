"""One supported species and its drugs."""

from pydantic import BaseModel, ConfigDict

from genome2mic.api.schemas.species_key import SpeciesKey


class SpeciesInfo(BaseModel):
    """A species in scope and the drugs that have a loaded model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    species: SpeciesKey
    name: str
    drugs: list[str]
