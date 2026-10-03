"""The in-vitro disclaimer must be on every report."""

import pytest
from httpx import AsyncClient
from pydantic import ValidationError

from genome2mic.api.constants import DISCLAIMER
from genome2mic.api.schemas.prediction_report import PredictionReport
from tests.api.conftest import FAKE_FASTA

FAKE_REPORT_FIELDS = {
    "sample_id": "FAKE-TEST-ONLY",
    "species": None,
    "qc_pass": True,
    "nearest_training_distance": None,
    "in_range": False,
    "predictions": [],
    "ranked_active": [],
    "model_version": "fake-0.0",
    "run_id": "fake-run",
}


async def test_every_result_response_has_the_disclaimer(ready_client: AsyncClient) -> None:
    for index in range(3):
        submit = await ready_client.post(
            "/v1/predict",
            files={"file": (f"genome_{index}.fasta", FAKE_FASTA)},
        )
        result = await ready_client.get(f"/v1/jobs/{submit.json()['job_id']}/result")

        assert result.status_code == 200
        assert result.json()["disclaimer"] == DISCLAIMER
        assert "not prescribing advice" in result.json()["disclaimer"]


def test_disclaimer_defaults_when_pipeline_omits_it() -> None:
    report = PredictionReport.model_validate(FAKE_REPORT_FIELDS)

    assert report.disclaimer == DISCLAIMER


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_disclaimer_is_rejected(blank: str) -> None:
    with pytest.raises(ValidationError):
        PredictionReport.model_validate({**FAKE_REPORT_FIELDS, "disclaimer": blank})
