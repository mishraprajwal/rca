# Hierarchical Root-Cause Analysis (RCA) System

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

Automatic triage of IT incident tickets into a 3-level hierarchical taxonomy (Domain → Component → Root cause). This repository implements a production-minded ML pipeline that includes IT-aware preprocessing, EDA, rich feature engineering, a hierarchical classifier (LCPN), experiment logging, and evaluation tooling.

## Redesign Highlights

- Hierarchical LCPN modeling (Local Classifier Per Parent Node) — classifiers trained per level and conditioned on parent labels for stronger hierarchical predictions.
- Rich feature engineering: `src/feature_engineer.py` builds combined word TF‑IDF, character n‑grams and domain meta-features (error codes, IPs, CPU/memory signals).
- EDA tooling: `src/eda.py` creates class-distribution plots, top-term lists, text-length statistics and a label-hierarchy heatmap (saved to `eda_results/`).
- Training & model selection: `src/trainer.py` runs StratifiedKFold CV to compare backends (`lr`, `svm`, `rf`, `cnb`) and supports `auto` selection of the best backend.
- Evaluation: `src/evaluator.py` emits per-level classification reports, confusion matrices (`confusion_level_*.png`), and `error_analysis.txt` describing common confusions.
- Robust CLI: package CLI (`python -m src`) supports `train`, `evaluate`, and `eda`; `predict.py` serves single-text predictions with per-level confidence scores.

## Features

- 3-level hierarchical classification (Domain → Component → Root cause)
- Local Classifier Per Parent Node (LCPN) inference (greedy top-down conditioning)
- Multiple backends: Logistic Regression (`lr`), Linear SVC (`svm`), Random Forest (`rf`), ComplementNB (`cnb`) and `auto` (CV-based selection)
- Multi-modal features: word + char TF-IDF + 14 hand-crafted meta-features
- EDA report and visualizations (`eda_results/`)
- Evaluation artifacts (`evaluation_results/metrics.json`, `report.txt`, `confusion_level_*.png`, `error_analysis.txt`)
- Experiment logging (`experiments.jsonl`) for reproducibility

## Architecture (high-level)

Raw incident text → `TextPreprocessor` → `FeatureEngineer` → `HierarchicalClassifier` (LCPN conditioning per level) → Predicted path + per-level confidences

See the implementation files for details: [src/feature_engineer.py](src/feature_engineer.py), [src/model.py](src/model.py), [src/trainer.py](src/trainer.py), [src/evaluator.py](src/evaluator.py).

## Installation

Prerequisites: Python 3.9+

Install for development:

```bash
git clone https://github.com/your-username/rca.git
cd rca
pip install -e ".[dev]"
```

Install runtime dependencies only:

```bash
pip install -r requirements.txt
```

## Quick Start

Train a model (defaults use `data/raw/incidents.csv` and `data/taxonomy.json`):

```bash
python train.py
# or as the package CLI:
python -m src train --model-name auto --model-path models/
```

Run EDA only:

```bash
python -m src eda --data-file data/raw/incidents.csv --output-dir eda_results/
```

Evaluate a saved model on a test CSV:

```bash
python -m src evaluate --model-path models/ --data-file data/raw/incidents_test.csv
```

Predict a single incident description:

```bash
# Plain text
python predict.py "Database connection timeout in production"

# JSON with per-level confidences
python predict.py --format json --model-path models/ "CPU at 100% causing slowdown"
```

Example prediction output (text):

```
Incident:   Database connection timeout in production
Prediction: Software → Application → Database
Confidence: 0.92 / 0.78 / 0.63
```

## Outputs and Artifacts

- `models/` — saved model artefacts (feature engineer, per-level classifiers, label encoders, metadata)
- `eda_results/` — EDA plots and `eda_report.json`
- `evaluation_results/` — `metrics.json`, `report.txt`, `confusion_level_*.png`, `error_analysis.txt`
- `experiments.jsonl` — per-experiment CV results and timing

Model save format contains `classifier_meta.json` and pickled objects to allow reproducible loading via `HierarchicalClassifier.load()`.

## Configuration

Use `config.yaml` to override defaults. Example keys:

```yaml
model:
  model_name: auto   # 'auto' | 'lr' | 'svm' | 'rf' | 'cnb'
  strategy: hierarchical

data:
  train_file: data/raw/incidents.csv
  test_file: data/raw/incidents_test.csv
  taxonomy_file: data/taxonomy.json

training:
  cv_splits: 3
  model_path: models/
  run_eda: true
```

## Data Format

CSV files must contain the columns: `description`, `level_1`, `level_2`, `level_3`.

## Development & Testing

Run the test suite:

```bash
make test
# or
python -m pytest tests/ -q
```

Run the end-to-end pipeline (train + evaluate):

```bash
make pipeline
```

Clean generated artefacts:

```bash
make clean
```

## Project Structure

```
rca/
├── src/
│   ├── feature_engineer.py   # TF-IDF + char-ngrams + meta-features
│   ├── eda.py                # EDAAnalyzer (plots + report)
│   ├── data_loader.py        # CSV loader / saver
│   ├── preprocessor.py       # IT-aware text cleaning + tokenization
│   ├── taxonomy.py           # TaxonomyManager
│   ├── model.py              # HierarchicalClassifier (LCPN)
│   ├── trainer.py            # ModelTrainer (CV + selection)
│   ├── evaluator.py          # Evaluator (confusion matrices + error analysis)
│   └── __main__.py          # CLI
├── data/
│   ├── taxonomy.json
│   └── raw/
├── models/
├── eda_results/
├── evaluation_results/
├── tests/
├── config.yaml
├── train.py
├── predict.py
└── Makefile
```

## Contributing

Contributions are welcome. High-impact areas:

- Improve Level-3 accuracy (feature engineering, hierarchical training strategies, additional labeled data)
- Add data augmentation and stronger handling for rare leaf classes
- Provide a REST API wrapper for real-time predictions

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

---

If you want, I can commit this change and open a small release PR.
