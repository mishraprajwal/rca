from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import os
from .data_loader import DataLoader
from .preprocessor import TextPreprocessor
from .taxonomy import TaxonomyManager
from .model import HierarchicalClassifier

class Evaluator:
    """Class for evaluating the hierarchical RCA system."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = {**self._default_config(), **(config or {})}
        self.data_loader = DataLoader()
        self.preprocessor = TextPreprocessor()
        self.taxonomy_manager = TaxonomyManager()
        self.classifier = HierarchicalClassifier(
            self.taxonomy_manager,
            model_type=self.config['model_type']
        )

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for evaluation."""
        return {
            'model_type': 'tfidf_lr',
            'test_data_file': 'incidents_test.csv',
            'taxonomy_file': 'taxonomy.json',
            'model_path': 'models',
            'output_dir': 'evaluation_results'
        }

    def load_test_data(self) -> pd.DataFrame:
        """Load and preprocess test data.

        Returns:
            Preprocessed test DataFrame
        """
        # Load test data
        df = self.data_loader.load_incident_data(self.config['test_data_file'])

        # Preprocess text
        df_processed = self.preprocessor.preprocess_incident_data(df)

        # Load taxonomy
        taxonomy_path = os.path.join('data', self.config['taxonomy_file'])
        self.taxonomy_manager.load_taxonomy(taxonomy_path)

        return df_processed

    def load_models(self) -> None:
        """Load trained models."""
        self.classifier.load_models(self.config['model_path'])

    def evaluate_predictions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model predictions on test data.

        Args:
            df: Test DataFrame with true labels

        Returns:
            Dictionary with evaluation metrics
        """
        results = {}

        # Generate predictions
        predictions = []
        true_labels = []

        for _, row in df.iterrows():
            text = row['processed_text']

            # Get true hierarchical path
            true_path = []
            for level in range(1, len(self.taxonomy_manager.levels) + 1):
                col = f'level_{level}'
                if col in row and pd.notna(row[col]):
                    true_path.append(row[col])
                else:
                    break

            # Get predicted path
            pred_path = self.classifier.predict_hierarchy(text)

            predictions.append(pred_path)
            true_labels.append(true_path)

        # Calculate metrics
        results['accuracy'] = self._calculate_hierarchical_accuracy(true_labels, predictions)
        results['precision'], results['recall'], results['f1'] = self._calculate_hierarchical_f1(true_labels, predictions)

        # Per-level metrics
        results['per_level_metrics'] = self._calculate_per_level_metrics(true_labels, predictions)

        return results

    def _calculate_hierarchical_accuracy(self, true_labels: List[List[str]],
                                       predictions: List[List[str]]) -> float:
        """Calculate hierarchical accuracy (exact path match).

        Args:
            true_labels: List of true hierarchical paths
            predictions: List of predicted hierarchical paths

        Returns:
            Accuracy score
        """
        correct = 0
        total = len(true_labels)

        for true_path, pred_path in zip(true_labels, predictions):
            if true_path == pred_path:
                correct += 1

        return correct / total if total > 0 else 0

    def _calculate_hierarchical_f1(self, true_labels: List[List[str]],
                                 predictions: List[List[str]]) -> Tuple[float, float, float]:
        """Calculate hierarchical precision, recall, and F1.

        Args:
            true_labels: List of true hierarchical paths
            predictions: List of predicted hierarchical paths

        Returns:
            Tuple of (precision, recall, f1)
        """
        # Flatten all predictions and labels
        all_true = []
        all_pred = []

        for true_path, pred_path in zip(true_labels, predictions):
            # Pad shorter paths with None
            max_len = max(len(true_path), len(pred_path))

            true_padded = true_path + [None] * (max_len - len(true_path))
            pred_padded = pred_path + [None] * (max_len - len(pred_path))

            all_true.extend(true_padded)
            all_pred.extend(pred_padded)

        # Calculate metrics (ignoring None values)
        valid_indices = [i for i, (t, p) in enumerate(zip(all_true, all_pred))
                        if t is not None and p is not None]

        if not valid_indices:
            return 0, 0, 0

        true_valid = [all_true[i] for i in valid_indices]
        pred_valid = [all_pred[i] for i in valid_indices]

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_valid, pred_valid, average='weighted', zero_division=0
        )

        return precision, recall, f1

    def _calculate_per_level_metrics(self, true_labels: List[List[str]],
                                   predictions: List[List[str]]) -> Dict[str, Any]:
        """Calculate metrics for each taxonomy level.

        Args:
            true_labels: List of true hierarchical paths
            predictions: List of predicted hierarchical paths

        Returns:
            Dictionary with per-level metrics
        """
        max_level = max(len(path) for path in true_labels + predictions)

        per_level = {}

        for level in range(1, max_level + 1):
            level_true = []
            level_pred = []

            for true_path, pred_path in zip(true_labels, predictions):
                if len(true_path) >= level and len(pred_path) >= level:
                    level_true.append(true_path[level-1])
                    level_pred.append(pred_path[level-1])

            if level_true and level_pred:
                accuracy = accuracy_score(level_true, level_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    level_true, level_pred, average='weighted', zero_division=0
                )

                per_level[f'level_{level}'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

        return per_level

    def generate_report(self, results: Dict[str, Any], output_dir: str = None) -> None:
        """Generate evaluation report.

        Args:
            results: Evaluation results dictionary
            output_dir: Directory to save report
        """
        output_dir = output_dir or self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics to JSON
        import json
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Generate text report
        report = self._generate_text_report(results)
        with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
            f.write(report)

        print(f"Evaluation report saved to {output_dir}")

    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """Generate text-based evaluation report.

        Args:
            results: Evaluation results

        Returns:
            Formatted report string
        """
        report = "Hierarchical RCA System Evaluation Report\n"
        report += "=" * 50 + "\n\n"

        report += f"Overall Hierarchical Accuracy: {results['accuracy']:.4f}\n"
        report += f"Overall Precision: {results['precision']:.4f}\n"
        report += f"Overall Recall: {results['recall']:.4f}\n"
        report += f"Overall F1-Score: {results['f1']:.4f}\n\n"

        report += "Per-Level Metrics:\n"
        for level, metrics in results['per_level_metrics'].items():
            report += f"  {level}:\n"
            report += f"    Accuracy: {metrics['accuracy']:.4f}\n"
            report += f"    Precision: {metrics['precision']:.4f}\n"
            report += f"    Recall: {metrics['recall']:.4f}\n"
            report += f"    F1-Score: {metrics['f1']:.4f}\n\n"

        return report

    def run_evaluation_pipeline(self) -> Dict[str, Any]:
        """Run the complete evaluation pipeline.

        Returns:
            Evaluation results
        """
        print("Starting evaluation pipeline...")

        # Load test data
        df = self.load_test_data()

        # Load models
        self.load_models()

        # Evaluate
        results = self.evaluate_predictions(df)

        # Generate report
        self.generate_report(results)

        print("Evaluation pipeline completed!")
        return results