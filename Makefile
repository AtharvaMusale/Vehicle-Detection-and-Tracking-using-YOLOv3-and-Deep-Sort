.PHONY: install test lint format typecheck check assets demo

install:
	python -m venv .venv && .venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest --cov=vehicle_tracking

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

format:
	.venv/bin/ruff check --fix . && .venv/bin/ruff format .

typecheck:
	.venv/bin/mypy

check: lint typecheck test

assets:
	.venv/bin/vehicle-tracking download-assets

# usage: make demo VIDEO=path/to/traffic.mp4
demo:
	.venv/bin/vehicle-tracking run --video "$(VIDEO)" --output outputs/demo.mp4
