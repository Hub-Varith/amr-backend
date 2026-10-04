"""Loads configs and models once and reports readiness."""

import logging

import yaml

from genome2mic.api.config import Settings
from genome2mic.api.constants import CONFIG_FILES, SPECIES_NAMES
from genome2mic.api.schemas.species_info import SpeciesInfo
from genome2mic.api.schemas.species_key import SpeciesKey
from genome2mic.config import ConfigError
from genome2mic.errors import Genome2MicError
from genome2mic.predict.pipeline import PredictionPipeline

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Holds the prediction pipeline and knows whether configs and models are ready."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline = PredictionPipeline(models_dir=settings.models_dir, configs_dir=settings.configs_dir)
        self.configs_parsed = False
        self.models_loaded = False
        self.drugs_by_species: dict[str, list[str]] = {}

    @property
    def is_ready(self) -> bool:
        return self.configs_parsed and self.models_loaded

    def load(self) -> None:
        """Parse configs and load models. Failures leave the service up but not ready.

        Any ``Exception`` from bundle loading is caught and logged (a truncated
        ``model.ubj`` raises ``XGBoostError``, a garbage ``.npz`` ``ValueError``, a
        malformed ``params.json`` ``KeyError``, ...): ``/health`` keeps answering and
        ``/ready`` reports 503. ``BaseException`` (``KeyboardInterrupt``, ``SystemExit``)
        is not caught.
        """
        logger.info(
            "Registry loading",
            extra={"configs_dir": str(self.settings.configs_dir), "models_dir": str(self.settings.models_dir)},
        )
        self.models_loaded = False
        self.drugs_by_species = {}
        self.configs_parsed = self._parse_configs()
        try:
            self.pipeline.load()
            self.drugs_by_species = self.pipeline.available_models()
            self.models_loaded = True
        except NotImplementedError as error:
            logger.warning("Models not loaded: pipeline not implemented", extra={"error": str(error)})
        except FileNotFoundError as error:
            logger.warning("Models not loaded: files missing", extra={"error": str(error)})
        except (Genome2MicError, ConfigError) as error:
            # A present-but-malformed bundle or config must not crash startup; stay up but not ready.
            logger.warning("Models not loaded: bundle or config malformed", extra={"error": str(error)})
        except Exception as error:
            # Any other loader failure (corrupt binary, missing key, out of memory, ...): same
            # outcome, logged at ERROR with the traceback so the operator can find the bad file.
            logger.exception(
                "Models not loaded: unexpected error while loading the model bundle",
                extra={"error": f"{type(error).__name__}: {error}"},
            )
        if not self.models_loaded:
            self.drugs_by_species = {}
        logger.info(
            "Registry loaded",
            extra={"configs_parsed": self.configs_parsed, "models_loaded": self.models_loaded},
        )

    def _parse_configs(self) -> bool:
        for config_name in CONFIG_FILES:
            config_path = self.settings.configs_dir / config_name
            try:
                with config_path.open() as handle:
                    content = yaml.safe_load(handle)
            except (OSError, yaml.YAMLError) as error:
                logger.warning("Config not readable", extra={"config_path": str(config_path), "error": str(error)})
                return False
            if not isinstance(content, dict):
                logger.warning("Config is not a mapping", extra={"config_path": str(config_path)})
                return False
        return True

    def species_info(self) -> list[SpeciesInfo]:
        """All species in scope, each with the drugs that have a loaded model."""
        species_list = []
        for species in SpeciesKey:
            drugs = sorted(self.drugs_by_species.get(species.value, []))
            species_list.append(SpeciesInfo(species=species, name=SPECIES_NAMES[species], drugs=drugs))
        return species_list
