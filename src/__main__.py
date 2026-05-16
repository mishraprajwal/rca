#!/usr/bin/env python3
"""Main entry point for the Hierarchical RCA System."""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical Root-Cause Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "action",
        choices=["train", "evaluate", "eda"],
        help="Action to perform",
    )
    parser.add_argument("--data-file", default=None,
                        help="CSV filename in data/raw/ (overrides default)")
    parser.add_argument("--model-name",
                        choices=["auto", "lr", "svm", "rf", "cnb"],
                        default="auto",
                        help="Model backend; 'auto' runs cross-validation (default)")
    parser.add_argument("--model-path", default="models/",
                        help="Directory for saved model artefacts")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for reports/plots")
    parser.add_argument("--no-eda", action="store_true",
                        help="Skip EDA during training")

    args = parser.parse_args()

    if args.action == "train":
        from .trainer import ModelTrainer
        config = {
            "model_name": args.model_name,
            "model_path": args.model_path,
            "run_eda": not args.no_eda,
        }
        if args.data_file:
            config["data_file"] = args.data_file
        if args.output_dir:
            config["eda_output_dir"] = args.output_dir
        ModelTrainer(config).run_training_pipeline()

    elif args.action == "evaluate":
        from .evaluator import Evaluator
        config = {"model_path": args.model_path}
        if args.data_file:
            config["test_data_file"] = args.data_file
        if args.output_dir:
            config["output_dir"] = args.output_dir
        results = Evaluator(config).run_evaluation_pipeline()
        h = results["hierarchical"]
        print(f"\nHierarchical F1: {h['f1']:.4f}  "
              f"(P={h['precision']:.4f}  R={h['recall']:.4f})")
        print(f"Exact-path accuracy: {results['overall_accuracy']:.4f}")

    elif args.action == "eda":
        from .data_loader import DataLoader
        from .preprocessor import TextPreprocessor
        from .eda import EDAAnalyzer
        data_file = args.data_file or "incidents.csv"
        output_dir = args.output_dir or "eda_results/"
        df = DataLoader().load_incident_data(data_file)
        df = TextPreprocessor().preprocess_incident_data(df)
        EDAAnalyzer(df, output_dir=output_dir).run_full_eda()
        print(f"EDA complete — results in {output_dir}")


if __name__ == "__main__":
    main()