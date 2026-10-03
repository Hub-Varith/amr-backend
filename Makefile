.PHONY: install install-api api test test-api train docker-build docker-run

install:
	pip install -e ".[model,test]"

install-api:
	pip install -e ".[test]"

# Request logging comes from RequestIdMiddleware, so uvicorn's access log is off.
api:
	uvicorn genome2mic.api.main:create_app --factory --reload --host 127.0.0.1 --port 8000 --no-access-log

test:
	pytest

test-api:
	pytest tests/api

docker-build:
	docker build -t genome2mic-api .

docker-run:
	docker run --rm -p 8000:8000 \
		-v "$(CURDIR)/configs:/app/configs:ro" \
		-v "$(CURDIR)/models:/app/models:ro" \
		genome2mic-api

# One shared model for every species x drug. Needs the contract files in data/processed.
train:
	python -m genome2mic.models.train_cli --processed-dir data/processed --out-dir models/multitask
