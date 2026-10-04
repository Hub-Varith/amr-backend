"""Model results + overrides + reasons -> the DATA_CONTRACT.md stage 12 report as a dict."""

from genome2mic.features.amrfinder_table import AmrFinderHit
from genome2mic.predict.call_overrides import NATURAL_RESISTANCE, CallOverrides
from genome2mic.predict.constants import LIKELY_ACTIVE, LIKELY_INACTIVE
from genome2mic.predict.drug_reasons import DrugReasons
from genome2mic.predict.mic_model import DrugResult

DEFAULT_TIER = 9


class ReportBuilder:
    """Applies the overrides after the model and ranks the likely-active drugs (CLAUDE.md call logic)."""

    def __init__(self, overrides: CallOverrides, reasons: DrugReasons, tiers: dict[str, int]) -> None:
        self.overrides = overrides
        self.reasons = reasons
        self.tiers = tiers

    def drug_rows(self, species: str, results: list[DrugResult], hits: list[AmrFinderHit]) -> list[dict]:
        rows = []
        for result in results:
            call = result.call
            row = {
                "drug": result.drug,
                "pred_mic": result.pred_mic,
                "band_low": result.band_low,
                "band_high": result.band_high,
                "s_breakpoint": call.s_breakpoint,
                "r_breakpoint": call.r_breakpoint,
                "p_active": call.p_active,
                "confidence_level": call.confidence_level,
                "call": call.call,
                "margin_steps": call.margin_steps,
                "reasons": self.reasons.for_drug(result.drug, hits),
                "override": None,
            }
            override = self.overrides.find(species, result.drug, hits)
            if override is not None:
                row.update(call=LIKELY_INACTIVE, override=override.kind, margin_steps=None)
                if override.kind == NATURAL_RESISTANCE:
                    # The model is skipped: no MIC and no probability are shown (contract stage 12).
                    row.update(pred_mic=None, band_low=None, band_high=None, p_active=None, confidence_level=None)
                else:
                    row["reasons"] = sorted(set(override.markers))
            rows.append(row)
        return sorted(rows, key=lambda row: row["drug"])

    def ranked_active(self, rows: list[dict]) -> list[str]:
        """Narrowest spectrum tier first, then the larger margin below S, then the higher p_active."""
        active = [row for row in rows if row["call"] == LIKELY_ACTIVE]
        active.sort(key=lambda row: (
            self.tiers.get(row["drug"], DEFAULT_TIER),
            -(row["margin_steps"] or 0),
            -(row["p_active"] if row["p_active"] is not None else 0.0),
            row["drug"],
        ))
        return [row["drug"] for row in active]
