.PHONY: setup clean test test-rf test-property lint format typecheck train eval figures serve ui check

# --- setup -------------------------------------------------------------------

setup:
	uv sync --all-extras
	uv run pre-commit install || true

clean:
	rm -rf .coverage coverage.xml .pytest_cache .mypy_cache .ruff_cache dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +

# --- verification ------------------------------------------------------------

test:
	uv run pytest -q

test-rf:
	uv run pytest tests/unit -q

test-property:
	uv run pytest tests/property -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy src tests

# Full local gate: lint, typecheck, tests.
check: lint typecheck test

# --- pipeline ----------------------------------------------------------------

train:
	uv run python -m flywaydqn.agents.train_dqn --config-path configs/train

eval:
	uv run python -m flywaydqn.agents.eval_main --config-path configs/eval

figures:
	uv run python -m experiments.make_beam_pattern_figure

serve:
	uv run uvicorn flywaydqn.service.app:app --factory --host 127.0.0.1 --port 8000

ui:
	cd ui && npm run dev
