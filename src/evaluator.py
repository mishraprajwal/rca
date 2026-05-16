"""
Evaluation pipeline for the Hierarchical RCA system.

Metrics computed
----------------
  Per level  : Accuracy, macro-F1, weighted-F1, sklearn classification_report
  Overall    : Exact-path accuracy, hierarchical precision/recall/F1

Outputs saved to evaluation_results/ (configurable)
  - metrics.json
  - report.txt
  - confusion_level_{1,2,3}.png
  - error_analysis.txt

Usage
-----
    evaluator = Evaluator()
    results = evaluator.run_evaluation_pipeline()
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .data_loader import DataLoader
from .model import HierarchicalClassifier
from .preprocessor import TextPreprocessor
from .taxonomy import TaxonomyManager

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Load a saved HierarchicalClassifier and evaluate on a test CSV.

    Config keys (all optional)
    --------------------------
    test_data_file : CSV filename in data/raw/
    taxonomy_file  : JSON filename in data/
    model_path     : directory of the saved model
    output_dir     : where to write reports and plots
    """

    _DEFAULTS: Dict[str, Any] = {
        "test_data_file": "incidents_test.csv",
        "taxonomy_file": "taxonomy.json",
        "model_path": "models/",
        "output_dir": "evaluation_results/",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self._DEFAULTS, **(config or {})}
        self._data_loader = DataLoader()
        self._preprocessor = TextPreprocessor()
        self._taxonomy_manager = TaxonomyManager()
        self._classifier: Optional[HierarchicalClassifier] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_evaluation_pipeline(self) -> Dict[str, Any]:
        logger.info("━" * 60)
        logger.info("HIERARCHICAL RCA — EVALUATION PIPELINE")
        logger.info("━" * 60)

        df = self._load_test_data()
        self._load_classifier()

        logger.info("Running predictions on %d samples …", len(df))
        results = self.evaluate_predictions(df)

        os.makedirs(self.config["output_dir"], exist_ok=True)
        self._save_metrics(results)
        self._save_text_report(results)
        self._plot_confusion_matrices(results)
        self._save_error_analysis(results)

        logger.info("━" * 60)
        self._print_summary(results)
        logger.info("Results saved to %s", self.config["output_dir"])
        logger.info("━" * 60)

        return results

    def evaluate_predictions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate predictions for all rows in df, compute metrics, return results dict.
        """
        clf = self._classifier
        if clf is None:
            raise RuntimeError("Call _load_classifier() before evaluate_predictions().")

        predictions: List[List[str]] = []
        true_labels: List[List[str]] = []

        for _, row in df.iterrows():
            text = row.get("processed_text", "")
            pred_path, _ = clf.predict_with_confidence(str(text))
            predictions.append(pred_path)

            true_path = []
            for lvl in range(1, 4):
                col = f"level_{lvl}"
                if col in row and pd.notna(row[col]):
                    true_path.append(str(row[col]))
                else:
                    break
            true_labels.append(true_path)

        results: Dict[str, Any] = {}
        results["overall_accuracy"] = self._exact_path_accuracy(true_labels, predictions)
        results["hierarchical"] = self._hierarchical_metrics(true_labels, predictions)
        results["per_level"] = self._per_level_metrics(true_labels, predictions)
        results["n_samples"] = len(df)

        # Store raw predictions for error analysis
        results["_true"] = true_labels
        results["_pred"] = predictions

        return results

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _exact_path_accuracy(true: List[List[str]], pred: List[List[str]]) -> float:
        """Fraction of samples where the full predicted path matches exactly."""
        correct = sum(t == p for t, p in zip(true, pred))
        return correct / len(true) if true else 0.0

    @staticmethod
    def _hierarchical_metrics(
        true: List[List[str]], pred: List[List[str]]
    ) -> Dict[str, float]:
        """
        Hierarchical Precision / Recall / F1.

        Flatten all (true, predicted) label pairs across levels, ignoring
        positions where either true or predicted path is shorter.
        """
        all_true, all_pred = [], []
        for t_path, p_path in zip(true, pred):
            for lvl in range(min(len(t_path), len(p_path))):
                all_true.append(t_path[lvl])
                all_pred.append(p_path[lvl])

        if not all_true:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        p, r, f1, _ = precision_recall_fscore_support(
            all_true, all_pred, average="weighted", zero_division=0
        )
        return {"precision": float(p), "recall": float(r), "f1": float(f1)}

    @staticmethod
    def _per_level_metrics(
        true: List[List[str]], pred: List[List[str]]
    ) -> Dict[str, Dict]:
        """Accuracy + weighted F1 at each taxonomy level."""
        max_levels = max(
            (max(len(t), len(p)) for t, p in zip(true, pred)), default=0
        )
        results = {}
        for lvl in range(1, max_levels + 1):
            y_true, y_pred = [], []
            for t_path, p_path in zip(true, pred):
                if len(t_path) >= lvl and len(p_path) >= lvl:
                    y_true.append(t_path[lvl - 1])
                    y_pred.append(p_path[lvl - 1])

            if not y_true:
                continue

            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
            results[f"level_{lvl}"] = {
                "n_samples": len(y_true),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "classification_report": classification_report(
                    y_true, y_pred, zero_division=0
                ),
                "_y_true": y_true,
                "_y_pred": y_pred,
            }
        return results

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _save_metrics(self, results: Dict) -> None:
        """Save a clean (no private keys) metrics JSON."""
        clean = {
            "overall_accuracy": results["overall_accuracy"],
            "n_samples": results["n_samples"],
            "hierarchical": results["hierarchical"],
            "per_level": {
                lvl: {k: v for k, v in info.items() if not k.startswith("_")}
                for lvl, info in results["per_level"].items()
            },
        }
        path = os.path.join(self.config["output_dir"], "metrics.json")
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        logger.info("Metrics saved → %s", path)

    def _save_text_report(self, results: Dict) -> None:
        lines = [
            "Hierarchical RCA System — Evaluation Report",
            "=" * 52,
            "",
            f"Test samples : {results['n_samples']}",
            f"Exact-path accuracy : {results['overall_accuracy']:.4f}",
            "",
            "Hierarchical Metrics (all levels combined, weighted avg):",
            f"  Precision : {results['hierarchical']['precision']:.4f}",
            f"  Recall    : {results['hierarchical']['recall']:.4f}",
            f"  F1-Score  : {results['hierarchical']['f1']:.4f}",
            "",
            "Per-Level Breakdown:",
        ]
        for lvl, info in sorted(results["per_level"].items()):
            lines += [
                f"\n  {lvl}  (n={info['n_samples']})",
                f"    Accuracy  : {info['accuracy']:.4f}",
                f"    Precision : {info['precision']:.4f}",
                f"    Recall    : {info['recall']:.4f}",
                f"    F1-Score  : {info['f1']:.4f}",
                "",
                "    Per-class report:",
            ]
            for line in info["classification_report"].splitlines():
                lines.append("      " + line)

        report = "\n".join(lines)
        path = os.path.join(self.config["output_dir"], "report.txt")
        with open(path, "w") as f:
            f.write(report)
        logger.info("Text report saved → %s", path)

    def _plot_confusion_matrices(self, results: Dict) -> None:
        """Save a confusion matrix PNG for each taxonomy level."""
        for lvl, info in sorted(results["per_level"].items()):
            y_true = info["_y_true"]
            y_pred = info["_y_pred"]
            labels = sorted(set(y_true) | set(y_pred))

            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig_size = max(6, len(labels) * 0.9)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                linewidths=0.4,
            )
            ax.set_title(
                f"Confusion Matrix — {lvl}  (acc={info['accuracy']:.2f})",
                fontsize=13,
                fontweight="bold",
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout()

            out = os.path.join(self.config["output_dir"], f"confusion_{lvl}.png")
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Confusion matrix saved → %s", out)

    def _save_error_analysis(self, results: Dict, top_n: int = 10) -> None:
        """
        List the most frequent misclassified (true, predicted) pairs per level.
        Useful for identifying systematic errors.
        """
        lines = ["Error Analysis — Most Frequent Misclassifications", "=" * 52, ""]

        for lvl, info in sorted(results["per_level"].items()):
            y_true = info["_y_true"]
            y_pred = info["_y_pred"]

            errors: Dict[Tuple, int] = {}
            for t, p in zip(y_true, y_pred):
                if t != p:
                    key = (t, p)
                    errors[key] = errors.get(key, 0) + 1

            lines.append(f"{lvl}:")
            if not errors:
                lines.append("  No errors!")
            else:
                sorted_errors = sorted(errors.items(), key=lambda x: -x[1])[:top_n]
                for (true_label, pred_label), cnt in sorted_errors:
                    lines.append(
                        f"  ✗  {cnt:2d}×   '{true_label}'  →  '{pred_label}' (predicted)"
                    )
            lines.append("")

        path = os.path.join(self.config["output_dir"], "error_analysis.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        logger.info("Error analysis saved → %s", path)

    def _print_summary(self, results: Dict) -> None:
        logger.info("── Results summary ──────────────────────────────────")
        logger.info(
            "  Exact-path accuracy : %.4f", results["overall_accuracy"]
        )
        logger.info(
            "  Hierarchical F1     : %.4f", results["hierarchical"]["f1"]
        )
        for lvl, info in sorted(results["per_level"].items()):
            logger.info(
                "  %-10s  acc=%.4f  f1=%.4f",
                lvl,
                info["accuracy"],
                info["f1"],
            )
        logger.info("─────────────────────────────────────────────────────")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_test_data(self) -> pd.DataFrame:
        df = self._data_loader.load_incident_data(self.config["test_data_file"])
        taxonomy_path = os.path.join("data", self.config["taxonomy_file"])
        self._taxonomy_manager.load_taxonomy(taxonomy_path)
        return self._preprocessor.preprocess_incident_data(df)

    def _load_classifier(self) -> None:
        self._classifier = HierarchicalClassifier.load(
            self.config["model_path"], self._taxonomy_manager
        )
