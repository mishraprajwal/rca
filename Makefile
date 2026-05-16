.PHONY: install install-dev train evaluate predict test clean help

PYTHON ?= python3

help:           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:        ## Install runtime dependencies
	$(PYTHON) -m pip install -e .

install-dev:    ## Install all dependencies including dev/test extras
	$(PYTHON) -m pip install -e ".[dev,viz]"

train:          ## Train the model with default config
	$(PYTHON) train.py

train-transformer: ## Train using transformer backend (requires GPU recommended)
	$(PYTHON) -m src train --model-type transformer

evaluate:       ## Run evaluation pipeline and save report
	$(PYTHON) -m src evaluate

pipeline:       ## Train then evaluate in sequence
	$(MAKE) train
	$(MAKE) evaluate

predict:        ## Predict (usage: make predict TEXT="your incident here")
	@if [ -z "$(TEXT)" ]; then echo "Usage: make predict TEXT=\"your incident description\""; exit 1; fi
	$(PYTHON) predict.py "$(TEXT)"

predict-json:   ## Predict with JSON output (usage: make predict-json TEXT="your incident here")
	@if [ -z "$(TEXT)" ]; then echo "Usage: make predict-json TEXT=\"your incident description\""; exit 1; fi
	$(PYTHON) predict.py --format json "$(TEXT)"

test:           ## Run the full test suite
	$(PYTHON) -m pytest tests/ -v

test-cov:       ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

clean:          ## Remove generated model files, caches, and build artifacts
	rm -rf models/*.pkl
	rm -rf data/processed/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache/ dist/ build/ *.egg-info/
