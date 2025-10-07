# Hierarchical Root-Cause Analysis System

This project implements an automated system for triaging incident tickets into a multi-level taxonomy using machine learning techniques.

## Features

- Hierarchical classification of incident tickets
- Multi-level taxonomy support
- Data preprocessing pipeline
- Machine learning models for root-cause analysis
- Evaluation metrics and reporting

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```

3. Run evaluation:
```bash
python -m src evaluate
```

4. Predict on new incidents:
```bash
python predict.py "Your incident description here"
```

## Data Format

The system expects incident data in CSV format with columns:
- `description`: Text description of the incident
- `level_1`, `level_2`, `level_3`: Taxonomy labels for each level

## Taxonomy

The taxonomy is defined in `data/taxonomy.json` with a hierarchical structure.

## Project Structure

- `src/`: Source code
- `data/`: Data files
- `models/`: Trained models
- `notebooks/`: Jupyter notebooks for exploration
- `tests/`: Unit tests

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.