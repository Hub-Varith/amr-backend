.PHONY: install api test-api docker-build docker-run

install:
	pip install -e ".[test]"

# Request logging comes from RequestIdMiddleware, so uvicorn's access log is off.
api:
	uvicorn genome2mic.api.main:create_app --factory --reload --host 127.0.0.1 --port 8000 --no-access-log

test-api:
	pytest tests/api

docker-build:
	docker build -t genome2mic-api .

docker-run:
	docker run --rm -p 8000:8000 \
		-v "$(CURDIR)/configs:/app/configs:ro" \
		-v "$(CURDIR)/models:/app/models:ro" \
		genome2mic-api
