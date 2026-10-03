"""Liveness and readiness."""

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from genome2mic.api.config import Settings
from genome2mic.api.main import create_app


async def test_health_is_ok(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]


async def test_safe_incoming_request_id_is_echoed(client: AsyncClient) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "trace-123"})

    assert response.headers["X-Request-ID"] == "trace-123"


async def test_unsafe_incoming_request_id_is_replaced(client: AsyncClient) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "bad id\nwith newline"})

    assert response.headers["X-Request-ID"] != "bad id\nwith newline"


async def test_ready_is_503_while_pipeline_is_a_stub(client: AsyncClient) -> None:
    response = await client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"ready": False, "configs_parsed": True, "models_loaded": False}


async def test_ready_reports_missing_configs(settings: Settings, tmp_path) -> None:
    app = create_app(settings.model_copy(update={"configs_dir": tmp_path / "missing"}))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/ready")

    assert response.status_code == 503
    assert response.json()["configs_parsed"] is False


async def test_ready_is_200_with_loaded_models(ready_client: AsyncClient) -> None:
    response = await ready_client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"ready": True, "configs_parsed": True, "models_loaded": True}


async def test_species_lists_all_five_with_loaded_drugs(ready_client: AsyncClient) -> None:
    response = await ready_client.get("/v1/species")

    assert response.status_code == 200
    by_key = {item["species"]: item for item in response.json()}
    assert set(by_key) == {"ECOLI", "KPNEU", "SAUR", "PAER", "ABAU"}
    assert by_key["KPNEU"]["drugs"] == ["fake-drug"]
    assert by_key["ECOLI"]["drugs"] == []


async def test_unhandled_error_returns_problem_json(app: FastAPI, client: AsyncClient) -> None:
    async def explode() -> None:
        raise RuntimeError("boom")

    app.add_api_route("/explode", explode)
    response = await client.get("/explode")

    assert response.status_code == 500
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["status"] == 500
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert "boom" not in response.text
