PYTHON ?= python
PIP ?= pip
CONFIG ?= configs/local.yaml

.PHONY: help install install-dev test lint format clean prepare gt osm line candidates tensors train decode evaluate visualize all

help:
	@echo "One-Direction commands"
	@echo "  make install       Install package dependencies"
	@echo "  make install-dev   Install package with dev dependencies"
	@echo "  make test          Run tests"
	@echo "  make lint          Run ruff checks"
	@echo "  make format        Format code with black and isort"
	@echo "  make all           Run full data/model pipeline"
	@echo "  make clean         Remove generated caches and outputs"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev,notebooks]"

test:
	$(PYTHON) -m pytest tests -ra

lint:
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m black --check src scripts tests
	$(PYTHON) -m isort --check-only src scripts tests

format:
	$(PYTHON) -m isort src scripts tests
	$(PYTHON) -m black src scripts tests
	$(PYTHON) -m ruff check src scripts tests --fix

prepare:
	$(PYTHON) scripts/01_prepare_trajectories.py

gt:
	$(PYTHON) scripts/02_prepare_gt_routes.py

osm:
	$(PYTHON) scripts/03_build_osm_graph.py

line:
	$(PYTHON) scripts/04_build_line_graph.py

candidates:
	$(PYTHON) scripts/05_generate_candidates.py

tensors:
	$(PYTHON) scripts/06_build_training_tensors.py

train:
	$(PYTHON) scripts/07_train_gnn_hmm.py --config $(CONFIG)

decode:
	$(PYTHON) scripts/08_decode_gnn_hmm.py --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/09_evaluate.py --config $(CONFIG)

visualize:
	$(PYTHON) scripts/10_visualize_errors.py --config $(CONFIG)

all: prepare gt osm line candidates tensors train decode evaluate visualize

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -rf outputs/checkpoints outputs/emissions outputs/transitions outputs/matches outputs/metrics outputs/figures outputs/tensorboard
