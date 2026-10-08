.PHONY: setup clean test lint typecheck train eval serve ui

setup:
	uv sync --all-extras
	uv run pre-commit install || true

clean:
	rm -rf .coverage dist build *.egg-info
	rm -rf src/flywaydqn/__pycache__ src/flywaydqn -rf __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + || true

test:
	uv run pytest -q

lint:
	uv run ruff check src flywaydqn/tests || true
	uv run ruff format --check src flywaydqn/tests || true

typecheck:
	uv run mypy src tests
	@echo "--- typecheck complete ---"

train:
	uv run python -m flywaydqn.agents.train_dqn --config-path configs/train

eval:
	uv run python -m flywaydqn.agents.eval_main --config-path configs/eval

serve:
	uv run uvicorn flywaydqn.service.app:app --factory --host 127.0.0.1 --port 8000

ui:
	cd ui && npm run dev
