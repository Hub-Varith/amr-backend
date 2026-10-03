"""FAKE registry for API tests that need a ready service."""

from genome2mic.api.services.model_registry import ModelRegistry


class FakeRegistry(ModelRegistry):
    """Reports ready without loading any model."""

    def load(self) -> None:
        self.configs_parsed = True
        self.models_loaded = True
        self.drugs_by_species = {"KPNEU": ["fake-drug"]}
