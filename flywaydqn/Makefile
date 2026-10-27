# FlywayDQN developer entry points.
#
# Interpreter resolution. `uv` when it is on PATH (CI installs it, and
# `make setup` creates the environment with it); otherwise the local `.venv311`
# interpreter. Every target goes through one resolved interpreter, so the local
# gate and the CI gate run the same commands and cannot drift apart.
_UV := $(shell command -v uv 2>/dev/null)
ifeq ($(_UV),)
  _PY  ?= ./.venv311/Scripts/python.exe
  _PYM := $(_PY) -m
else
  _PYM := uv run python -m
endif

.PHONY: setup clean test test-rf test-property lint format typecheck check \
        train train-eval eval figures serve ui demo demo-serve demo-ui

# --- setup -------------------------------------------------------------------

# Creates .venv from pyproject.toml. Requires `uv`: there is no requirements
# file in this repository, so this is the only supported bootstrap.
setup:
	uv sync --all-extras
	uv run pre-commit install || true

clean:
	rm -rf .coverage coverage.xml .pytest_cache .mypy_cache .ruff_cache dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +

# --- verification ------------------------------------------------------------

lint:
	$(_PYM) ruff check src tests experiments
	$(_PYM) ruff format --check src tests experiments

format:
	$(_PYM) ruff format src tests experiments
	$(_PYM) ruff check --fix src tests experiments

typecheck:
	$(_PYM) mypy src tests

test:
	$(_PYM) pytest tests -q

test-rf:
	$(_PYM) pytest tests/unit -q

test-property:
	$(_PYM) pytest tests/property -q

# Full local gate: the same three steps the CI job runs.
check: lint typecheck test

# --- pipeline -----------------------------------------------------------------

# Train the Double DQN; checkpoints land in experiments/checkpoints/<run_name>/.
train:
	$(_PYM) flywaydqn.agents.train_dqn --config configs/train/train.yaml

# Refresh the trained-policy comparison table from stored checkpoints without
# retraining. Needs a checkpoint, so run `make train` first.
train-eval:
	$(_PYM) flywaydqn.agents.train_dqn --eval-only

# Regenerate the measured baseline table (experiments/results/baselines.{csv,md}).
# Works on a clean clone: the trained controller is included only when a
# checkpoint is present, and its absence is printed rather than hidden.
eval:
	$(_PYM) experiments.run_baselines

figures:
	$(_PYM) experiments.make_beam_pattern_figure

# --- run ---------------------------------------------------------------------

serve:
	$(_PYM) uvicorn flywaydqn.service.app:app --factory --host 127.0.0.1 --port 8000

ui:
	cd ui && npm run dev

demo-serve:
	$(_PYM) uvicorn flywaydqn.service.app:app --factory --host 127.0.0.1 --port 8000

demo-ui:
	cd ui && npm run dev

demo:
	$(_PYM) uvicorn flywaydqn.service.app:app --factory --host 127.0.0.1 --port 8000 &
	cd ui && npm run dev
