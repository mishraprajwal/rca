#!/usr/bin/env python3
"""Predict root-cause taxonomy for a new incident description."""

import argparse
import json
import sys
from src.model import HierarchicalClassifier
from src.taxonomy import TaxonomyManager
from src.preprocessor import TextPreprocessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict root-cause taxonomy for an incident.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py "Database connection timeout in production"
  python predict.py --format json "CPU at 100% causing slowdown"
  python predict.py --model-path models/ "Network switch failure"
        """,
    )
    parser.add_argument("description", help="Incident description text")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--model-path",
        default="models",
        help="Path to saved model directory (default: models/)",
    )
    parser.add_argument(
        "--taxonomy",
        default="data/taxonomy.json",
        help="Path to taxonomy JSON file (default: data/taxonomy.json)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load taxonomy
    taxonomy_manager = TaxonomyManager()
    taxonomy_manager.load_taxonomy(args.taxonomy)

    # Load classifier
    classifier = HierarchicalClassifier.load(args.model_path, taxonomy_manager)

    # Preprocess input
    preprocessor = TextPreprocessor()
    clean_text = preprocessor.clean_text(args.description)
    processed_text = " ".join(preprocessor.tokenize_and_lemmatize(clean_text))

    # Predict with confidence
    path, confidences = classifier.predict_with_confidence(processed_text)

    if args.format == "json":
        output = {
            "incident": args.description,
            "prediction": path,
            "path_string": " → ".join(path) if path else "Unknown",
            "confidence": confidences,
        }
        print(json.dumps(output, indent=2))
    else:
        path_str = " → ".join(path) if path else "Unknown"
        conf_str = " / ".join(f"{c:.2f}" for c in confidences) if confidences else "N/A"
        print(f"Incident:   {args.description}")
        print(f"Prediction: {path_str}")
        print(f"Confidence: {conf_str}")


if __name__ == "__main__":
    main()
