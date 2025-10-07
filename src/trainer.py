from typing import Dict, List, Any
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from .data_loader import DataLoader
from .preprocessor import TextPreprocessor
from .taxonomy import TaxonomyManager
from .model import HierarchicalClassifier

class Trainer:
    """Class for training the hierarchical RCA system."""

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
        """Default configuration for training."""
        return {
            'model_type': 'tfidf_lr',
            'test_size': 0.2,
            'random_state': 42,
            'data_file': 'incidents.csv',
            'taxonomy_file': 'taxonomy.json',
            'model_path': 'models'
        }

    def prepare_data(self) -> pd.DataFrame:
        """Load and preprocess training data.

        Returns:
            Preprocessed DataFrame
        """
        # Load raw data
        df = self.data_loader.load_incident_data(self.config['data_file'])

        # Preprocess text
        df_processed = self.preprocessor.preprocess_incident_data(df)

        # Load taxonomy
        taxonomy_path = os.path.join('data', self.config['taxonomy_file'])
        self.taxonomy_manager.load_taxonomy(taxonomy_path)

        # Validate that taxonomy columns exist
        taxonomy_columns = [f'level_{i+1}' for i in range(len(self.taxonomy_manager.levels))]
        missing_cols = [col for col in taxonomy_columns if col not in df_processed.columns]

        if missing_cols:
            print(f"Warning: Missing taxonomy columns: {missing_cols}")
            print("Available columns:", list(df_processed.columns))

        return df_processed

    def train_models(self, df: pd.DataFrame) -> None:
        """Train hierarchical classification models.

        Args:
            df: Preprocessed DataFrame with taxonomy labels
        """
        levels = len(self.taxonomy_manager.levels)

        for level in range(1, levels + 1):
            level_col = f'level_{level}'

            if level_col not in df.columns:
                print(f"Skipping level {level}: column {level_col} not found")
                continue

            # Filter out rows with missing labels at this level
            df_level = df.dropna(subset=[level_col])

            if len(df_level) == 0:
                print(f"No data available for level {level}")
                continue

            # Split data
            X = df_level['processed_text'].tolist()
            y = df_level[level_col].tolist()

            # Remove duplicates and empty labels
            valid_data = [(text, label) for text, label in zip(X, y)
                         if text.strip() and label.strip()]
            X, y = zip(*valid_data) if valid_data else ([], [])

            if not X:
                print(f"No valid data for level {level}")
                continue

            print(f"Training level {level} with {len(X)} samples")

            # Train model for this level
            self.classifier.train_level(list(X), list(y), level, self.config['model_path'])

        # Save trained models
        self.classifier.save_models(self.config['model_path'])

    def run_training_pipeline(self) -> None:
        """Run the complete training pipeline."""
        print("Starting training pipeline...")

        # Prepare data
        df = self.prepare_data()

        # Train models
        self.train_models(df)

        print("Training pipeline completed!")

    def evaluate_training_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate the quality of training data.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Dictionary with data quality metrics
        """
        metrics = {}

        # Basic stats
        metrics['total_samples'] = len(df)
        metrics['taxonomy_levels'] = len(self.taxonomy_manager.levels)

        # Missing values per level
        for level in range(1, len(self.taxonomy_manager.levels) + 1):
            col = f'level_{level}'
            if col in df.columns:
                missing = df[col].isnull().sum()
                metrics[f'level_{level}_missing'] = missing
                metrics[f'level_{level}_coverage'] = (len(df) - missing) / len(df)

        # Class distribution
        class_dist = {}
        for level in range(1, len(self.taxonomy_manager.levels) + 1):
            col = f'level_{level}'
            if col in df.columns:
                dist = df[col].value_counts().to_dict()
                class_dist[f'level_{level}'] = dist

        metrics['class_distribution'] = class_dist

        return metrics