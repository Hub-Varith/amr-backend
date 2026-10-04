"""Per-drug MIC models (CLAUDE.md "Models" table) and the conformal band helpers.

All models follow :class:`genome2mic.models.base.MicModel`::

    fit(X, lo, hi, feature_names) -> self ; predict_log2(X) ; save(dir) ; load(dir)

``MODEL_CLASSES`` maps the model ids used in the predictions table to classes.
``aft_known`` and ``aft_known_unitig`` share :class:`XgbAft`; the orchestrator
chooses the feature set and passes ``name=`` accordingly. ``aft_b2_select``
(:class:`AftB2Select`, v0.6) is not fitted directly: training assembles it from the
fitted AFT and B2 models after the in-fold choice of AFT, B2 or their average.
"""

from __future__ import annotations

from genome2mic.models.aft_b2_select import AftB2Select
from genome2mic.models.b1_lookup import B1Lookup
from genome2mic.models.b2_xgb_steps import B2XgbSteps
from genome2mic.models.base import FORBIDDEN_FEATURES, MicModel
from genome2mic.models.conformal import band, conformal_q, residual_steps
from genome2mic.models.xgb_aft import XgbAft

MODEL_CLASSES: dict[str, type] = {
    "b1_lookup": B1Lookup,
    "b2_xgb_steps": B2XgbSteps,
    "aft_known": XgbAft,
    "aft_known_unitig": XgbAft,
    "aft_b2_select": AftB2Select,
}
"""Model id (``model`` column of the predictions table) -> class."""


def make_model(model_id: str, **kwargs: object) -> MicModel:
    """Instantiate a model by id with ``name=model_id`` and any constructor overrides."""
    try:
        cls = MODEL_CLASSES[model_id]
    except KeyError:
        raise KeyError(f"unknown model id {model_id!r}; known: {sorted(MODEL_CLASSES)}") from None
    return cls(name=model_id, **kwargs)  # type: ignore[call-arg]


__all__ = [
    "AftB2Select",
    "B1Lookup",
    "B2XgbSteps",
    "FORBIDDEN_FEATURES",
    "MODEL_CLASSES",
    "MicModel",
    "XgbAft",
    "band",
    "conformal_q",
    "make_model",
    "residual_steps",
]
