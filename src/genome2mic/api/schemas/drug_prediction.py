"""One drug's line in the report."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator

from genome2mic.api.schemas.call import Call
from genome2mic.api.schemas.override import Override


class DrugPrediction(BaseModel):
    """Predicted MIC, 90% band, breakpoints, P(active) and call for one drug (DATA_CONTRACT.md stage 12). MICs in mg/L."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    drug: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    pred_mic: PositiveFloat | None
    band_low: PositiveFloat | None
    band_high: PositiveFloat | None
    s_breakpoint: PositiveFloat | None = None
    r_breakpoint: PositiveFloat | None = None
    p_active: float | None = Field(default=None, ge=0.0, le=1.0)
    call: Call
    margin_steps: int | None
    reasons: list[str] = []
    override: Override | None = None

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        """Reject reports whose numbers contradict each other."""
        mic_values = (self.pred_mic, self.band_low, self.band_high)
        n_missing = sum(value is None for value in mic_values)
        if n_missing not in (0, 3):
            raise ValueError("pred_mic, band_low and band_high must be all set or all null.")
        if n_missing == 0 and not self.band_low <= self.pred_mic <= self.band_high:
            raise ValueError("Expected band_low <= pred_mic <= band_high.")
        if self.s_breakpoint is not None and self.r_breakpoint is not None:
            if self.s_breakpoint > self.r_breakpoint:
                raise ValueError("Expected s_breakpoint <= r_breakpoint.")
        if self.override is not None and self.call is not Call.LIKELY_INACTIVE:
            raise ValueError("A drug with an override must have call likely_inactive.")
        return self
